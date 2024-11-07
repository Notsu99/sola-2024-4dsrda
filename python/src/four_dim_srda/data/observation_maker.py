import random
import sys
from logging import getLogger

import torch

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


def generate_is_obses_and_obs_matrices(
    *,
    nx: int,
    ny: int,
    nz: int,
    obs_grid_interval_x: int,
    obs_grid_interval_y: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    #
    is_obses, obs_matrices = [], []
    ratio_mean = []
    #
    for init_x in tqdm(range(obs_grid_interval_x)):
        for init_y in range(obs_grid_interval_y):
            is_obs, obs_mat = _generate_is_obs_and_obs_matrix_from_init_point(
                nx=nx,
                ny=ny,
                nz=nz,
                init_index_x=init_x,
                init_index_y=init_y,
                interval_x=obs_grid_interval_x,
                interval_y=obs_grid_interval_y,
                device=device,
                dtype=dtype,
            )
            is_obses.append(is_obs)
            obs_matrices.append(obs_mat)
            ratio_mean.append(torch.mean(is_obs).item())
    ratio_mean = sum(ratio_mean) / len(ratio_mean)
    logger.info(f"Observation grid ratio = {ratio_mean}")

    return is_obses, obs_matrices


def make_observation(
    hr_pv: torch.Tensor,
    is_obses: list[torch.Tensor],
    obs_noise_std: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    #
    obs = _extract_observation_without_noise_randomly(
        hr_pv=hr_pv,
        is_obses=is_obses,
    )

    if obs_noise_std > 0:
        noise = torch.normal(mean=0, std=obs_noise_std, size=obs.shape, dtype=dtype)
        obs = obs + noise

    return obs


def _extract_observation_without_noise_randomly(
    hr_pv: torch.Tensor,
    is_obses: list[torch.Tensor],
) -> torch.Tensor:
    #
    i = random.randint(0, len(is_obses) - 1)
    is_obs = is_obses[i]

    # skip t dim and compare the other dims (z, y, x)
    assert is_obs.shape == hr_pv.shape[1:]

    # broadcast over t dim.
    is_obs = torch.broadcast_to(is_obs, hr_pv.shape)
    logger.debug(f"index of is_obs = {i}")

    _tmp = torch.full_like(hr_pv, torch.nan)
    obs = torch.where(is_obs > 0, hr_pv, _tmp)

    return obs


def _generate_is_obs_and_obs_matrix_from_init_point(
    *,
    nx: int,
    ny: int,
    nz: int,
    init_index_x: int,
    init_index_y: int,
    interval_x: int,
    interval_y: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    #
    assert 0 <= init_index_x <= interval_x - 1
    assert 0 <= init_index_y <= interval_y - 1

    # is_obs
    is_obs = torch.zeros(nz, ny, nx, dtype=dtype)
    for z in range(nz):
        is_obs[z, init_index_y::interval_y, init_index_x::interval_x] = 1.0

    # obs_matrix
    flattened_is_obs = is_obs.reshape(-1)
    obs_indices = torch.where(flattened_is_obs == 1.0)[0]

    num_obs = len(obs_indices)

    obs_matrix = torch.zeros(num_obs, nx * ny * nz, dtype=dtype)
    for i, j in enumerate(obs_indices):
        obs_matrix[i, j] = 1.0

    p = 100 * torch.sum(obs_matrix).item() / (nx * ny * nz)
    logger.debug(f"observatio prob = {p} [%]")

    return is_obs.to(device), obs_matrix.to(device)