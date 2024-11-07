import functools
import glob
import os
import re
from logging import getLogger

import numpy as np

logger = getLogger()


def get_hr_file_paths(
    data_dirs: list[str],
    max_start_time_index: int,
):
    lst_hr_file_pahts = [
        glob.glob(f"{dir_path}/*_hr_pv_*.npy") for dir_path in data_dirs
    ]
    hr_file_paths = functools.reduce(lambda l1, l2: l1 + l2, lst_hr_file_pahts, [])

    extracted_paths = []
    for path in hr_file_paths:
        grps = re.match(r"seed(\d+)_start(\d+)_end", os.path.basename(path)).groups()
        start_idx = int(grps[1])
        if start_idx > max_start_time_index:
            continue
        else:
            extracted_paths.append(path)

    return extracted_paths


# def group_paths_by_seeds(all_paths: list[str]) -> dict[str, list[str]]:
#     def _get_seed(p: str) -> str:
#         return os.path.basename(p).split("_")[0]

#     dict_paths = {}

#     # 'seed00000_start00_end08_hr_pv_00.npy' -> `seed00000`
#     for seed, grouped in itertools.groupby(all_paths, key=_get_seed):
#         dict_paths[seed] = sorted(grouped)

#     return dict_paths


def get_similar_source_lr_path(
    key_start_end_time: str,
    target_lr: np.ndarray,
    dict_all_lr_data_at_init_time: dict,
    max_ensemble_number: int,
    num_searched_lr_states: int,
):
    all_lrs = dict_all_lr_data_at_init_time[key_start_end_time]

    min_path, min_norm = None, np.inf

    # Need to modify this process later
    # because the same lr data is included at first
    for i in list(set(np.random.randint(0, len(all_lrs), size=num_searched_lr_states)))[
        :max_ensemble_number
    ]:
        data = all_lrs[i]["data"]
        path = all_lrs[i]["path"]
        assert data.shape == target_lr.shape
        norm = np.mean((data - target_lr) ** 2)

        # To avoid picking the same lr data,
        # the case of norm == 0 is excluded.
        if 0 < norm < min_norm:
            min_norm = norm
            min_path = path
            logger.debug(f"norm = {norm}, path = {min_path}")

    return min_path
