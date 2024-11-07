import copy
import dataclasses
import os
import re
import sys
from logging import getLogger

import numpy as np
import torch
from src.four_dim_srda.data.base_config import BaseDatasetConfig
from src.four_dim_srda.data.get_path import (
    get_hr_file_paths,
    get_similar_source_lr_path,
)
from src.four_dim_srda.data.observation_maker import (
    generate_is_obses_and_obs_matrices,
    make_observation,
)
from torch.utils.data import Dataset

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


@dataclasses.dataclass
class DatasetMakingObsConfig(BaseDatasetConfig):
    beta_dist_alpha: float
    beta_dist_beta: float
    data_size_per_epoch: int
    is_future_obs_missing: bool
    max_ensemble_number: int
    missing_value: float
    num_searched_lr_states: int
    nx_hr: int
    ny_hr: int
    nz_hr: int
    obs_grid_interval_x: int
    obs_grid_interval_y: int
    obs_noise_std: float
    pv_max: float
    pv_min: float
    use_mixup: bool
    use_observation: bool


class DatasetMakingObs(Dataset):
    def __init__(
        self,
        cfg: DatasetMakingObsConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.cfg = copy.deepcopy(cfg)
        self.dtype = dtype

        self.all_hr_file_paths = get_hr_file_paths(
            self.cfg.data_dirs, self.cfg.max_start_time_index
        )
        self.hr_file_paths = None
        self.set_hr_file_paths_randomly(epoch=0)
        assert self.hr_file_paths is not None, "set_hr_file_paths_randomly doesn't work"

        self.dict_all_lr_data_at_init_time = self._load_all_lr_data_at_init_time()

        self.is_obses, _ = generate_is_obses_and_obs_matrices(
            nx=self.cfg.nx_hr,
            ny=self.cfg.ny_hr,
            nz=self.cfg.nz_hr,
            obs_grid_interval_x=self.cfg.obs_grid_interval_x,
            obs_grid_interval_y=self.cfg.obs_grid_interval_y,
            device="cpu",
            dtype=self.dtype,
        )

    def set_hr_file_paths_randomly(self, epoch: int):
        generator = torch.Generator().manual_seed(epoch)
        ln = len(self.all_hr_file_paths)
        indices = torch.randperm(n=ln, generator=generator)
        indices = indices[: self.cfg.data_size_per_epoch]
        self.hr_file_paths = [self.all_hr_file_paths[i.item()] for i in indices]

    def __len__(self) -> int:
        return len(self.hr_file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._getitem(path_idx=idx)

    def _getitem(
        self, path_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #
        target_lr, source_lr, gt = self._load_np_data(path_idx)

        obs = make_observation(
            hr_pv=gt,
            is_obses=self.is_obses,
            obs_noise_std=self.cfg.obs_noise_std,
            dtype=self.dtype,
        )

        target_lr = self._preprocess(target_lr, use_clipping=True)
        source_lr = self._preprocess(source_lr, use_clipping=True)
        obs = self._preprocess(obs, use_clipping=True)
        gt = self._preprocess(gt, use_clipping=self.cfg.use_ground_truth_clipping)

        if self.cfg.use_observation:
            obs = torch.nan_to_num(obs, nan=self.cfg.missing_value)
        else:
            obs = torch.full_like(obs, fill_value=self.cfg.missing_value)

        if self.cfg.use_mixup:
            source_prob = np.random.beta(
                a=self.cfg.beta_dist_alpha, b=self.cfg.beta_dist_beta, size=1
            )[0]
            logger.debug(f"source prob = {source_prob}")
            lr = source_prob * source_lr + (1 - source_prob) * target_lr
        else:
            lr = target_lr

        # obs interval and lr interval are the same
        # Subsample obs at the same interval of lr interval
        # lr data was already subsampled in lr simulation
        # lr = lr[:: self.cfg.lr_and_obs_time_interval]
        obs = obs[:: self.cfg.lr_and_obs_time_interval]

        if self.cfg.is_future_obs_missing:
            # time_length must be odd number
            time_length = len(obs)
            center_index = time_length // 2
            obs[center_index + 1 :] = self.cfg.missing_value

        return lr, obs, gt

    def _load_np_data(
        self, path_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #
        hr_path = self.hr_file_paths[path_idx]
        hr = np.load(hr_path)
        logger.debug(f"HR path = {os.path.basename(hr_path)}")

        target_lr_path = hr_path.replace("hr_pv", "lr_pv")
        target_lr = np.load(target_lr_path)

        name = os.path.basename(target_lr_path)
        key = re.search(r"start\d+_end\d+", name).group()

        # Pass target_lr at the initial time, which has the index of `0`
        source_lr_path = get_similar_source_lr_path(
            key,
            target_lr[0],
            self.dict_all_lr_data_at_init_time,
            self.cfg.max_ensemble_number,
            self.cfg.num_searched_lr_states,
        )
        source_lr = np.load(source_lr_path)

        hr = torch.from_numpy(hr).to(self.dtype)
        target_lr = torch.from_numpy(target_lr).to(self.dtype)
        source_lr = torch.from_numpy(source_lr).to(self.dtype)

        return target_lr, source_lr, hr

    def _load_all_lr_data_at_init_time(self) -> dict[str, list]:
        dict_all_lr_data_at_init_time = {}

        for hr_path in tqdm(self.all_hr_file_paths):
            lr_path = hr_path.replace("hr_pv", "lr_pv")

            key = re.search(r"start\d+_end\d+", os.path.basename(lr_path)).group()

            if key not in dict_all_lr_data_at_init_time:
                dict_all_lr_data_at_init_time[key] = []
                logger.debug(f"Added key = {key}")

            lr = np.load(lr_path)
            dict_all_lr_data_at_init_time[key].append({"data": lr[0], "path": lr_path})
            # `0` means the initial time

        return dict_all_lr_data_at_init_time

    def _preprocess(
        self, data: torch.Tensor, use_clipping: bool = False
    ) -> torch.Tensor:
        #
        # Add channel dim and drop the last index along y for NN network
        ret = data[:, None, :, :-1, :]
        # ret's dim is (time, channel, z, y, x)

        # normalization
        ret = (ret - self.cfg.pv_min) / (self.cfg.pv_max - self.cfg.pv_min)

        if use_clipping:
            ret = torch.clamp(ret, min=0.0, max=1.0)

        return ret


@dataclasses.dataclass
class DatasetMakingObsMinusOneOneScalingConfig(DatasetMakingObsConfig):
    pass


class DatasetMakingObsMinusOneOneScaling(DatasetMakingObs):
    def __init__(
        self,
        cfg: DatasetMakingObsMinusOneOneScalingConfig,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(cfg, dtype)

    def _preprocess(
        self, data: torch.Tensor, use_clipping: bool = False
    ) -> torch.Tensor:
        #
        # Add channel dim and drop the last index along y for NN network
        ret = data[:, None, :, :-1, :]
        # ret's dim is (time, channel, z, y, x)

        # normalization to [-1, 1]
        ret = 2 * (ret - self.cfg.pv_min) / (self.cfg.pv_max - self.cfg.pv_min) - 1

        if use_clipping:
            ret = torch.clamp(ret, min=-1.0, max=1.0)

        return ret


@dataclasses.dataclass
class DatasetMakingObsUsingOnlyCurrentTimeTargetConfig(DatasetMakingObsConfig):
    pass


class DatasetMakingObsUsingOnlyCurrentTimeTarget(DatasetMakingObs):
    def __init__(
        self,
        cfg: DatasetMakingObsUsingOnlyCurrentTimeTargetConfig,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(cfg, dtype)

    def _getitem(
        self, path_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #
        target_lr, source_lr, gt = self._load_np_data(path_idx)

        obs = make_observation(
            hr_pv=gt,
            is_obses=self.is_obses,
            obs_noise_std=self.cfg.obs_noise_std,
            dtype=self.dtype,
        )

        target_lr = self._preprocess(target_lr, use_clipping=True)
        source_lr = self._preprocess(source_lr, use_clipping=True)
        obs = self._preprocess(obs, use_clipping=True)
        gt = self._preprocess(gt, use_clipping=self.cfg.use_ground_truth_clipping)

        if self.cfg.use_observation:
            obs = torch.nan_to_num(obs, nan=self.cfg.missing_value)
        else:
            obs = torch.full_like(obs, fill_value=self.cfg.missing_value)

        if self.cfg.use_mixup:
            source_prob = np.random.beta(
                a=self.cfg.beta_dist_alpha, b=self.cfg.beta_dist_beta, size=1
            )[0]
            logger.debug(f"source prob = {source_prob}")
            lr = source_prob * source_lr + (1 - source_prob) * target_lr
        else:
            lr = target_lr

        # obs interval and lr interval are the same
        # Subsample obs at the same interval of lr interval
        # lr data was already subsampled in lr simulation
        # lr = lr[:: self.cfg.lr_and_obs_time_interval]
        obs = obs[:: self.cfg.lr_and_obs_time_interval]

        if self.cfg.is_future_obs_missing:
            # time_length must be odd number
            time_length = len(obs)
            center_index = time_length // 2
            obs[center_index + 1 :] = self.cfg.missing_value

        # Extract current time gt
        # time_length must be odd number
        time_length = len(gt)
        current_time_index = time_length // 2
        gt = gt[current_time_index:current_time_index+1, ...]

        return lr, obs, gt
