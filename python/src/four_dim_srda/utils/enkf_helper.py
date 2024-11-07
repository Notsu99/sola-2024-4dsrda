import glob
import os
import random
from logging import getLogger

import numpy as np
import torch
from src.four_dim_srda.da.enkf import EnKFConfig
from src.four_dim_srda.data.dataloader import split_file_paths
from src.four_dim_srda.config.experiment_config import CFDConfig

logger = getLogger()


def load_hr_time_series_data(
    *, root_dir: str, num_times: int, cfg: EnKFConfig
) -> torch.Tensor:
    #
    cfd_data_dir_path = f"{root_dir}/data/four_dim_srda/{cfg.data_dir}"
    logger.info(f"CFD dir path = {cfd_data_dir_path}")

    data_dirs = sorted(
        [p for p in glob.glob(f"{cfd_data_dir_path}/seed*") if os.path.isdir(p)]
    )

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, cfg.train_valid_test_ratios
    )

    if cfg.train_valid_test_kinds == "train":
        target_dirs = train_dirs
    elif cfg.train_valid_test_kinds == "valid":
        target_dirs = valid_dirs
    elif cfg.train_valid_test_kinds == "test":
        target_dirs = test_dirs
    else:
        raise Exception(f"{kind} is not supported.")

    logger.info(
        f"Kind = {cfg.train_valid_test_kinds}, Num of dirs = {len(target_dirs)}"
    )

    all_hr_pvs = []
    for dir_path in sorted(target_dirs):
        for file_path in sorted(glob.glob(f"{dir_path}/*_hr_pv_*.npy")):
            data = np.load(file_path)
            all_hr_pvs.append(data)

            if len(all_hr_pvs) == cfg.num_simulation:
                # Concat along batch dim
                ret = np.stack(all_hr_pvs, axis=0)[:, :num_times]
                return torch.from_numpy(ret).to(torch.float64)

    # Concat along batch dim
    ret = np.stack(all_hr_pvs, axis=0)[:, :num_times]
    return torch.from_numpy(ret).to(torch.float64)


class ObsMatrixSampler:
    def __init__(self, obs_matrices: list[torch.Tensor], device: str):
        self.obs_matrices = obs_matrices
        self.device = device

    def __call__(self):
        i = random.randint(0, len(self.obs_matrices) - 1)
        # obs_matrices shape = [i, num_obs, nx * ny * nz]
        return self.obs_matrices[i].clone().to(self.device)


def get_obs_matrix(
    obs: torch.Tensor, nz: int, ny: int, nx: int, device: str
) -> torch.Tensor:
    assert obs.shape == (nz, ny, nx)
    is_obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), torch.ones_like(obs))

    obs_indices = is_obs.reshape(-1)
    obs_indices = torch.where(obs_indices == 1.0)[0]

    num_obs = len(obs_indices)

    obs_matrix = torch.zeros(num_obs, nz * ny * nx, dtype=torch.float64, device=device)

    for i, j in enumerate(obs_indices):
        obs_matrix[i, j] = 1.0

    p = 100 * torch.sum(obs_matrix).item() / (nz * ny * nx)
    logger.debug(f"observatio prob = {p} [%]")

    return obs_matrix


def get_uhr_pvs(
    *, result_dir_path: str, num_times: int, cfg: CFDConfig
) -> torch.Tensor:
    #
    out_step_start = int(cfg.time_config.start_time / cfg.time_config.output_uhr_dt)
    out_step_end = int(cfg.time_config.end_time / cfg.time_config.output_uhr_dt)

    all_uhr_pvs = []
    for i_seed_uhr in range(
        cfg.seed_config.uhr_seed_start, cfg.seed_config.uhr_seed_end + 1
    ):
        #
        _path = f"{result_dir_path}/seed{i_seed_uhr:05}/seed{i_seed_uhr:05}_start{out_step_start:03}_end{out_step_end:03}_uhr_pv.npy"
        uhr_pvs = torch.from_numpy(np.load(_path))[:num_times]
        assert uhr_pvs.shape == (
            num_times,
            cfg.uhr_base_config.nz,
            cfg.uhr_base_config.ny,
            cfg.uhr_base_config.nx,
        )
        all_uhr_pvs.append(uhr_pvs)

    # Stack along the seed dim
    all_uhr_pvs = torch.stack(all_uhr_pvs, dim=0)

    return all_uhr_pvs


# The following are private methods.
