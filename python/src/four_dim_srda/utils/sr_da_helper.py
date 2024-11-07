import random
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from src.four_dim_srda.config.experiment_config import BaseExperimentConfig, CFDConfig
from src.four_dim_srda.data.dataloader import make_dataloaders_and_samplers
from src.four_dim_srda.data.dataset_making_obs import DatasetMakingObs
from src.qg_model.low_pass_filter import LowPassFilter
from src.qg_model.qg_model import QGModel

logger = getLogger()


def get_testdataset(
    *,
    cfg: BaseExperimentConfig,
    root_dir: str,
    world_size: int = None,
    rank: int = None,
):
    #

    dataloaders, _ = make_dataloaders_and_samplers(
        cfg_dataloader=cfg.dataloader_config,
        cfg_data=cfg.dataset_config,
        root_dir=root_dir,
        train_valid_test_kinds=["test"],
    )

    return dataloaders["test"].dataset


def get_uhr_and_hr_pvs(
    *, result_dir_path: str, i_seed_uhr: int, num_times: int, cfg: CFDConfig
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    #
    out_step_start = int(cfg.time_config.start_time / cfg.time_config.output_uhr_dt)
    out_step_end = int(cfg.time_config.end_time / cfg.time_config.output_uhr_dt)

    _path = f"{result_dir_path}/seed{i_seed_uhr:05}_start{out_step_start:03}_end{out_step_end:03}_uhr_pv.npy"
    all_uhr_pvs = torch.from_numpy(np.load(_path))[:num_times]

    assert all_uhr_pvs.shape == (
        num_times,
        cfg.uhr_base_config.nz,
        cfg.uhr_base_config.ny,
        cfg.uhr_base_config.nx,
    )

    tmp = all_uhr_pvs[:, :, 1:, :]

    kernel_size = cfg.uhr_base_config.nx // cfg.hr_base_config.nx
    _pvs = F.avg_pool3d(tmp, kernel_size=kernel_size)

    all_hr_pvs = torch.zeros(
        (
            num_times,
            cfg.hr_base_config.nz,
            cfg.hr_base_config.ny,
            cfg.hr_base_config.nx,
        ),
        dtype=_pvs.dtype,
    )

    all_hr_pvs[:, :, 1:, :] = _pvs

    return all_uhr_pvs, all_hr_pvs


def get_observation_with_noise(
    *, hr_pv: torch.Tensor, test_dataset: DatasetMakingObs, cfg_cfd: CFDConfig
) -> torch.Tensor:
    #
    assert hr_pv.ndim == 5  # ens, time, z, y, x dims
    assert hr_pv.shape[0] == cfg_cfd.hr_base_config.n_batch
    assert hr_pv.shape[2] == cfg_cfd.hr_base_config.nz
    assert hr_pv.shape[3] == cfg_cfd.hr_base_config.ny
    assert hr_pv.shape[4] == cfg_cfd.hr_base_config.nx

    n_ens = cfg_cfd.lr_base_config.n_batch

    is_obses = []
    for _ in range(n_ens):
        _is_obses = []
        for _ in range(hr_pv.shape[1]):
            i = random.randint(0, len(test_dataset.is_obses) - 1)
            is_obs = test_dataset.is_obses[i]
            assert is_obs.shape == (
                cfg_cfd.hr_base_config.nz,
                cfg_cfd.hr_base_config.ny,
                cfg_cfd.hr_base_config.nx,
            )
            _is_obses.append(is_obs)

        is_obses.append(torch.stack(_is_obses, dim=0))

    is_obses = torch.stack(is_obses, dim=0)
    assert is_obses.shape == hr_pv.shape

    hr_obsrv = torch.full_like(hr_pv, np.nan)
    hr_obsrv = torch.where(
        is_obses > 0,
        hr_pv,
        hr_obsrv,
    )

    if test_dataset.cfg.obs_noise_std <= 0:
        logger.info("No observation noise.")
        return hr_obsrv

    noise = np.random.normal(
        loc=0, scale=test_dataset.cfg.obs_noise_std, size=hr_obsrv.shape
    )
    logger.info(f"Observation noise std = {test_dataset.cfg.obs_noise_std}")

    return hr_obsrv + torch.from_numpy(noise)


def get_observation_with_noise_using_fixed_obs_point(
    *, hr_pv: torch.Tensor, test_dataset: DatasetMakingObs, cfg_cfd: CFDConfig
) -> torch.Tensor:
    #
    assert hr_pv.ndim == 5  # ens, time, z, y, x dims
    assert hr_pv.shape[0] == cfg_cfd.hr_base_config.n_batch
    assert hr_pv.shape[2] == cfg_cfd.hr_base_config.nz
    assert hr_pv.shape[3] == cfg_cfd.hr_base_config.ny
    assert hr_pv.shape[4] == cfg_cfd.hr_base_config.nx

    n_ens = cfg_cfd.lr_base_config.n_batch

    is_obses = []
    for _ in range(n_ens):
        i = random.randint(0, len(test_dataset.is_obses) - 1)
        is_obs = test_dataset.is_obses[i]
        assert is_obs.shape == (
            cfg_cfd.hr_base_config.nz,
            cfg_cfd.hr_base_config.ny,
            cfg_cfd.hr_base_config.nx,
        )
        is_obses.append(is_obs)

    is_obses = torch.stack(is_obses, dim=0)

    # add time dim
    is_obses = is_obses[:, None, ...]

    hr_obsrv = torch.full_like(hr_pv, np.nan)
    hr_obsrv = torch.where(
        is_obses > 0,
        hr_pv,
        hr_obsrv,
    )

    if test_dataset.cfg.obs_noise_std <= 0:
        logger.info("No observation noise.")
        return hr_obsrv

    noise = np.random.normal(
        loc=0, scale=test_dataset.cfg.obs_noise_std, size=hr_obsrv.shape
    )
    logger.info(f"Observation noise std = {test_dataset.cfg.obs_noise_std}")

    return hr_obsrv + torch.from_numpy(noise)


def initialize_and_itegrate_lr_cfd_model_for_forecast(
    *,
    last_hr_pv0: torch.Tensor,
    num_integrate_steps: int,
    low_pass_filter: LowPassFilter,
    cfg_cfd: CFDConfig,
) -> torch.Tensor:
    #
    lr_model = QGModel(cfg_cfd.lr_base_config, show_input_cfg_info=False)

    lr_pv0 = low_pass_filter.apply(hr_grid_data=last_hr_pv0)

    lr_model.initialize_pv(grid_pv=lr_pv0)
    lr_forecast = [lr_model.get_grid_pv()]

    for _ in range(num_integrate_steps):
        num_steps_per_loop = int(
            cfg_cfd.time_config.output_lr_dt / cfg_cfd.time_config.lr_dt
        )
        lr_model.integrate_n_steps(
            dt_per_step=cfg_cfd.time_config.lr_dt, n_steps=num_steps_per_loop
        )

        lr_forecast.append(lr_model.get_grid_pv())

    # Stack arrays along time dim
    lr_forecast = torch.stack(lr_forecast, dim=1)
    # lr_forecast dim = batch, time, z, y, x

    return lr_forecast


def make_preprocessed_lr_for_forecast(
    *, lr_forecast: torch.Tensor, test_dataset: DatasetMakingObs, cfg_cfd: CFDConfig
) -> torch.Tensor:
    #
    n_ens = cfg_cfd.lr_base_config.n_batch

    return _preprocess(
        data=lr_forecast,
        pv_max=test_dataset.cfg.pv_max,
        pv_min=test_dataset.cfg.pv_min,
        n_ens=n_ens,
        device=cfg_cfd.lr_base_config.device,
    )


def make_preprocessed_obs_for_forecast(
    *,
    hr_obs: list[torch.Tensor],
    assimilation_period: int,
    forecast_span: int,
    test_dataset: DatasetMakingObs,
    cfg_cfd: CFDConfig,
) -> torch.Tensor:
    #
    n_ens = cfg_cfd.lr_base_config.n_batch

    # obs dim = time, batch, z, y, x
    obs = torch.stack(hr_obs[-(assimilation_period + 1) :], dim=0)
    dummy = torch.full_like(obs[0], fill_value=torch.nan)
    dummy = torch.broadcast_to(dummy, size=(forecast_span,) + dummy.shape)
    # dummy dim = time(forecast_span), batch, z, y, x

    obs = torch.concat([obs, dummy], dim=0)  # stack along time
    # obs dim = time, batch, z, y, x -> batch, time, z, y, x
    obs = obs.permute(1, 0, 2, 3, 4).contiguous()

    obs = _preprocess(
        data=obs,
        pv_max=test_dataset.cfg.pv_max,
        pv_min=test_dataset.cfg.pv_min,
        n_ens=n_ens,
        device=cfg_cfd.lr_base_config.device,
    )

    # obs interval and lr interval are the same
    # Subsample obs at the same interval of lr interval
    obs = obs[:, :: test_dataset.cfg.lr_and_obs_time_interval]

    return torch.nan_to_num(obs, nan=test_dataset.cfg.missing_value)


def make_invprocessed_srda_for_forecast(
    *,
    preds: torch.Tensor,
    assimilation_period: int,
    forecast_span: int,
    test_dataset: DatasetMakingObs,
    cfg_cfd: CFDConfig,
) -> torch.Tensor:
    #
    n_ens = cfg_cfd.lr_base_config.n_batch

    srda = _inv_preprocess(preds, test_dataset.cfg.pv_max, test_dataset.cfg.pv_min)

    # srda dim = batch, time, channel, z, y, x
    # delete channel dim
    srda = srda.squeeze(2)

    # srda dim = batch, time, z, y, x
    srda = _append_zeros_to_y(srda)

    assert srda.shape == (
        n_ens,
        forecast_span + assimilation_period + 1,
        cfg_cfd.hr_base_config.nz,
        cfg_cfd.hr_base_config.ny,
        cfg_cfd.hr_base_config.nx,
    )

    return srda


def make_invprocessed_srda_for_forecast_only_current_time_output(
    *,
    preds: torch.Tensor,
    test_dataset: DatasetMakingObs,
    cfg_cfd: CFDConfig,
) -> torch.Tensor:
    #
    n_ens = cfg_cfd.lr_base_config.n_batch

    srda = _inv_preprocess(preds, test_dataset.cfg.pv_max, test_dataset.cfg.pv_min)

    # srda dim = batch, time, channel, z, y, x
    # delete channel dim
    srda = srda.squeeze(2)

    # srda dim = batch, time, z, y, x
    srda = _append_zeros_to_y(srda)

    assert srda.shape == (
        n_ens,
        1,
        cfg_cfd.hr_base_config.nz,
        cfg_cfd.hr_base_config.ny,
        cfg_cfd.hr_base_config.nx,
    )

    return srda


# The following are private methods.


def _preprocess(
    *,
    data: torch.Tensor,
    pv_max: float,
    pv_min: float,
    n_ens: int,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    #
    # batch, time, z, y, x
    # Drop the last index along y for NN network
    data = data[:, :, :, :-1, :]

    data = data.unsqueeze(2)  # Add channel dim

    # normalization
    data = (data - pv_min) / (pv_max - pv_min)
    data = torch.clamp(data, min=0.0, max=1.0)

    data = data.to(dtype).to(device)

    # batch, time, channel, z, y, x
    assert data.ndim == 6
    assert data.shape[0] == n_ens

    return data


def _inv_preprocess(data: torch.Tensor, pv_max: float, pv_min: float) -> torch.Tensor:
    #
    return data * (pv_max - pv_min) + pv_min


def _append_zeros_to_y(data: torch.Tensor) -> torch.Tensor:
    assert data.ndim == 5  # batch, time, z, y, x

    B, T, Z, Y, X = data.shape
    zs = torch.zeros(B, T, Z, 1, X)
    appended = torch.cat((data, zs), dim=3)

    # Check the last index of y has zero values
    assert torch.max(torch.abs(appended[:, :, :, -1, :])).item() == 0.0

    # Check other values
    assert torch.equal(data, appended[:, :, :, :-1, :])

    return appended
