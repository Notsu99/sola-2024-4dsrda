import dataclasses
from logging import getLogger

import torch
from src.four_dim_srda.config.base_config import YamlConfig
from tqdm import tqdm

logger = getLogger()


@dataclasses.dataclass
class LETKFBlockProcessingConfig(YamlConfig):
    data_dir: str
    train_valid_test_ratios: list[int]
    train_valid_test_kinds: str
    num_simulation: int
    nx: int
    ny: int
    nz: int
    num_ens: int
    obs_grid_interval_x: int
    obs_grid_interval_y: int
    obs_noise_std: int
    cylinder_radius: float
    cylinder_height: float
    inflation_factor: float
    localization_radius: float
    block_size: int


# This code follows the algorithm in this paper
# Hunt, B. R., Kostelich, E. J. & Szunyogh, I.
# Efficient data assimilation for spatiotemporal chaos: a local ensemble transform Kalman filter.
# Phys. D: Nonlinear Phenom. 230, 112–126 (2007).


def hr_assimilate_with_existing_data(
    *,
    hr_pv: torch.Tensor,
    gt_added_noise: torch.Tensor,
    obs_matrix: torch.Tensor,
    dist_xy: torch.Tensor,
    dist_z: torch.Tensor,
    decay_factors: torch.Tensor,
    block_idx: list[torch.Tensor],
    center_idx: list[torch.Tensor],
    cfg: LETKFBlockProcessingConfig,
    rand_generator: torch.Generator = None,
) -> torch.Tensor:
    #
    assert hr_pv.ndim == 4  # (num_ens, nz, ny, nx)
    assert gt_added_noise.ndim == 3  # (nz, ny, nx)
    assert obs_matrix.ndim == 2  # (num_obs, nz*ny*nx)

    num_ens, hr_nz, hr_ny, hr_nx = hr_pv.shape
    num_obs = obs_matrix.shape[0]

    assert obs_matrix.shape[1] == hr_nz * hr_ny * hr_nx
    assert (
        dist_xy.shape
        == dist_z.shape
        == decay_factors.shape
        == (num_obs, hr_nz * hr_ny * hr_nx)
    )

    logger.info(f"num obs is {num_obs}")

    hr_state = hr_pv.reshape(num_ens, -1)

    obs = _make_observation(
        gt_added_noise=gt_added_noise, obs_matrix=obs_matrix, cfg=cfg
    )

    analysis_all = _assimilate(
        observation=obs,
        model_state=hr_state,
        obs_matrix=obs_matrix,
        dist_xy=dist_xy,
        dist_z=dist_z,
        decay_factors=decay_factors,
        block_idx=block_idx,
        center_idx=center_idx,
        cfg=cfg,
    )

    analysis_all = analysis_all.reshape(num_ens, hr_nz, hr_ny, hr_nx)

    return analysis_all


def calc_distances_from_obs_grid_points(
    *,
    obs_matrix: torch.Tensor,
    flattend_x_coords: torch.Tensor,
    flattend_y_coords: torch.Tensor,
    flattend_z_coords: torch.Tensor,
    cfg: LETKFBlockProcessingConfig,
):
    #
    num_obs = obs_matrix.shape[0]

    obs_flattend_x_coords = obs_matrix.mm(flattend_x_coords.reshape(-1, 1))
    obs_flattend_y_coords = obs_matrix.mm(flattend_y_coords.reshape(-1, 1))
    obs_flattend_z_coords = obs_matrix.mm(flattend_z_coords.reshape(-1, 1))
    assert (
        obs_flattend_x_coords.shape
        == obs_flattend_y_coords.shape
        == obs_flattend_z_coords.shape
        == (num_obs, 1)
    )

    # Calculate distances in xy plane and in z direction
    dist_xy = torch.sqrt(
        (flattend_y_coords - obs_flattend_y_coords) ** 2
        + (flattend_x_coords - obs_flattend_x_coords) ** 2
    )
    dist_z = torch.abs(flattend_z_coords - obs_flattend_z_coords)
    assert dist_xy.shape == dist_z.shape == (num_obs, cfg.nx * cfg.ny * cfg.nz)

    return dist_xy, dist_z


def calc_decay_factors_of_localization(
    *,
    dist_xy: torch.Tensor,
    dist_z: torch.Tensor,
    obs_matrix: torch.Tensor,
    cfg: LETKFBlockProcessingConfig,
) -> torch.Tensor:
    #
    num_obs = obs_matrix.shape[0]

    # Calc distance
    distances = torch.sqrt(dist_xy**2 + dist_z**2)
    decay_factors = torch.exp(-((distances / cfg.localization_radius) ** 2))
    assert decay_factors.shape == (num_obs, cfg.nz * cfg.ny * cfg.nx)

    return decay_factors


def make_splitted_grid_and_its_center_indices_array(
    cfg: LETKFBlockProcessingConfig,
) -> dict[list[torch.Tensor], list[torch.Tensor]]:
    #
    grid = torch.arange(cfg.nz * cfg.ny * cfg.nx).reshape(cfg.nz, cfg.ny, cfg.nx)

    #
    block_idx = []
    center_idx = []

    for iz in range(cfg.nz):
        for iy in range(0, cfg.ny, cfg.block_size):
            for ix in range(0, cfg.nx, cfg.block_size):
                _block_idx = grid[
                    iz, iy : iy + cfg.block_size, ix : ix + cfg.block_size
                ].flatten()  # Flatten blocks into a one-dimensional array in each z-plane
                block_idx.append(_block_idx)

                _center_idx = len(_block_idx) // 2
                center_idx.append(_block_idx[_center_idx])

    return block_idx, center_idx


# The following are private methods.


def _make_observation(
    *,
    gt_added_noise: torch.Tensor,
    obs_matrix: torch.Tensor,
    cfg: LETKFBlockProcessingConfig,
) -> torch.Tensor:
    #
    # (num_obs, nz*ny*nx) @ (nz*ny*nx, 1)
    obs = obs_matrix.mm(gt_added_noise.reshape(-1, 1))
    # obs shape = (num_obs, 1)

    return obs


def _assimilate(
    *,
    observation: torch.Tensor,
    model_state: torch.Tensor,
    obs_matrix: torch.Tensor,
    dist_xy: torch.Tensor,
    dist_z: torch.Tensor,
    decay_factors: torch.Tensor,
    block_idx: list[torch.Tensor],
    center_idx: list[torch.Tensor],
    cfg: LETKFBlockProcessingConfig,
) -> torch.Tensor:
    #
    # model_state dim = (num_ens, nz*ny*nx)
    xgb_mean, Xgb, ygb_mean, Ygb = _calc_forecast_stats(
        state=model_state, obs_matrix=obs_matrix, cfg=cfg
    )  # g means "global", b means "background"

    #
    Rg, Rg_inv = _calc_obs_stats(observation=observation, cfg=cfg)

    #
    analysis_all = torch.zeros(
        cfg.num_ens, cfg.nx * cfg.ny * cfg.nz, dtype=model_state.dtype
    ).to(model_state.device)

    _apply_LETKF(
        analysis_all=analysis_all,
        xgb_mean=xgb_mean,
        Xgb=Xgb,
        observation=observation,
        obs_matrix=obs_matrix,
        ygb_mean=ygb_mean,
        Ygb=Ygb,
        Rg_inv=Rg_inv,
        dist_xy=dist_xy,
        dist_z=dist_z,
        decay_factors=decay_factors,
        block_idx=block_idx,
        center_idx=center_idx,
        cfg=cfg,
    )

    return analysis_all


def _calc_forecast_stats(
    *, state: torch.Tensor, obs_matrix: torch.Tensor, cfg: LETKFBlockProcessingConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    #
    assert state.ndim == 2  # (num_ens, nz*ny*nx)
    assert state.shape[0] == cfg.num_ens

    num_obs = obs_matrix.shape[0]

    xgb_mean = torch.mean(state, dim=0, keepdim=True)
    Xgb = state - xgb_mean

    # (num_obs, nz*ny*nx) @ (nz*ny*nx, 1)
    ygb_mean = obs_matrix.mm(xgb_mean.t())
    assert ygb_mean.shape == (num_obs, 1)

    # (num_obs, nz*ny*nx) @ (nz*ny*nx, num_ens)
    Ygb = obs_matrix.mm(Xgb.t())
    assert Ygb.shape == (num_obs, cfg.num_ens)

    return xgb_mean, Xgb, ygb_mean, Ygb


def _calc_obs_stats(
    *, observation: torch.Tensor, cfg: LETKFBlockProcessingConfig
) -> torch.Tensor:
    #
    num_obs = observation.shape[0]

    # observation covariance
    # each observation is independent from others
    Rg = torch.diag(
        torch.tensor([cfg.obs_noise_std] * num_obs, dtype=observation.dtype)
    ).to(observation.device)
    Rg_inv = torch.inverse(Rg)

    return Rg, Rg_inv


def _get_obs_indices_within_cylinder(
    *,
    dist_xy: torch.Tensor,
    dist_z: torch.Tensor,
    obs_matrix: torch.Tensor,
    cfg: LETKFBlockProcessingConfig,
) -> torch.Tensor:
    #
    # This calculation is conducted on CPU
    # because sizes of dist_xy and dist_z are large,
    # which might cause a memory capacity error.
    num_obs = obs_matrix.shape[0]

    # Find points within the cylinder (True or False)
    is_within_cylinder = (dist_xy <= cfg.cylinder_radius) & (
        dist_z <= cfg.cylinder_height / 2
    )
    assert is_within_cylinder.shape == (num_obs,)

    chosen_indices = torch.nonzero(is_within_cylinder).squeeze()
    if chosen_indices.numel() == 0:
        raise ValueError(
            "Error: No observations were found within the specified cylinder."
        )

    return chosen_indices


def _make_localization_matrix(
    *,
    decay_factors: torch.Tensor,
    chosen_indices: torch.Tensor,
    device: str,
):
    #
    num_obs = len(chosen_indices)

    _decay_factors = decay_factors[chosen_indices].to(device)
    localization_matrix = _decay_factors.unsqueeze(1) * _decay_factors.unsqueeze(0)
    assert localization_matrix.shape == (num_obs, num_obs)

    return localization_matrix


def _apply_LETKF(
    *,
    analysis_all: torch.Tensor,
    xgb_mean: torch.Tensor,
    Xgb: torch.Tensor,
    observation: torch.Tensor,
    obs_matrix: torch.Tensor,
    ygb_mean: torch.Tensor,
    Ygb: torch.Tensor,
    Rg_inv: torch.Tensor,
    dist_xy: torch.Tensor,
    dist_z: torch.Tensor,
    decay_factors: torch.Tensor,
    block_idx: list[torch.Tensor],
    center_idx: list[torch.Tensor],
    cfg: LETKFBlockProcessingConfig,
):
    #
    device = observation.device

    for i, (b_idx, c_idx) in enumerate(zip(block_idx, center_idx)):
        #
        chosen_indices = _get_obs_indices_within_cylinder(
            dist_xy=dist_xy[:, c_idx],
            dist_z=dist_z[:, c_idx],
            obs_matrix=obs_matrix,
            cfg=cfg,
        )

        localization_matrix = _make_localization_matrix(
            chosen_indices=chosen_indices,
            decay_factors=decay_factors[:, c_idx],
            device=device,
        )

        num_obs = len(chosen_indices)
        if i == 0:
            logger.info(f"first block indices:\n {b_idx}")
            logger.info(f"first center index and num chosen obs: {c_idx}, {num_obs}")

        # l means "local"
        Ylb = Ygb[chosen_indices, :]
        assert Ylb.shape == (num_obs, cfg.num_ens)

        Xlb = Xgb[:, b_idx]
        # Xlb.shape == (cfg.num_ens, cfg.block_size)(Almost all)

        Rl_inv = (
            torch.diag(Rg_inv[chosen_indices, chosen_indices]) * localization_matrix
        )
        assert Rl_inv.shape == (num_obs, num_obs)

        C = Ylb.t().mm(Rl_inv)  # (num_ens, num_obs)

        # Eigen value decomposition
        Identity_mat = torch.eye(cfg.num_ens).to(device)

        eigvals, eigvecs = torch.linalg.eig(
            (cfg.num_ens - 1) * Identity_mat / cfg.inflation_factor + C.mm(Ylb)
        )
        # imaginary parts are all 0, so neglectable
        eigvals_real = eigvals.real
        eigvecs_real = eigvecs.real

        # eigvals_real is vector of torch because torch returns like that.
        eigvals_inv = torch.diag(1.0 / eigvals_real)
        sqrt_eigvals_inv = torch.sqrt(eigvals_inv)

        #
        Pa_tilde = eigvecs_real.mm(eigvals_inv).mm(
            torch.linalg.inv(eigvecs_real)
        )  # (num_ens, num_ens)

        #
        num_ens_sqrt = (cfg.num_ens - 1) ** 0.5

        Wa = num_ens_sqrt * eigvecs_real.mm(sqrt_eigvals_inv).mm(
            torch.linalg.inv(eigvecs_real)
        )  # (num_ens, num_ens)

        # Reshape for the case of num_obs == 1.
        wa_mean = Pa_tilde.mm(C).mm(
            observation[chosen_indices, :].reshape(num_obs, 1)
            - ygb_mean[chosen_indices, :].reshape(num_obs, 1)
        )  # (num_ens, 1)

        Wa = Wa + wa_mean  # (num_ens, num_ens)

        analysis_local = xgb_mean[:, b_idx] + Wa.t().mm(Xlb)  # (num_ens, batch_size)
        # analysis_local.shape == (cfg.num_ens, cfg.block_size)(Almost all)

        # analysis_all dim is (num_ens, nz*ny*nx)
        analysis_all[:, b_idx] = analysis_local
