import argparse
import datetime
import os
import pathlib
import re
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
from src.four_dim_srda.config.experiment_config import CFDConfig
from src.four_dim_srda.da.letkf_block_processing import (
    LETKFBlockProcessingConfig,
    calc_decay_factors_of_localization,
    calc_distances_from_obs_grid_points,
    hr_assimilate_with_existing_data,
    make_splitted_grid_and_its_center_indices_array,
)
from src.four_dim_srda.utils.enkf_helper import get_obs_matrix
from src.four_dim_srda.utils.log_maker import output_gpu_memory_summary_log
from src.four_dim_srda.utils.random_seed_helper import set_seeds
from src.qg_model.jet_maker import (
    make_jet_pv_with_linear_profile,
    make_jet_pv_with_tanh_profile,
    make_jet_velocity_with_sech_squared_and_sigmoid_profile,
    make_jet_velocity_with_sech_squared_profile,
)
from src.qg_model.qg_model import QGModel
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=True)

logger = getLogger()
if not logger.hasHandlers():
    logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_cfd_path", type=str, required=True)
parser.add_argument("--config_letkf_path", type=str, required=True)
parser.add_argument("--config_srda_path", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)

ROOT_DIR = pathlib.Path(os.environ["PYTHONPATH"]).parent.resolve()
RESULT_DIR = f"{ROOT_DIR}/python/results/four_dim_srda"
LOG_DIR = f"{ROOT_DIR}/python/logs/four_dim_srda"


if __name__ == "__main__":
    try:
        start_time = time.time()

        cfg_letkf_path = parser.parse_args().config_letkf_path
        experiment_name = cfg_letkf_path.split("/")[-3]
        config_name = os.path.basename(cfg_letkf_path).split(".")[0]

        cfg_cfd_path = parser.parse_args().config_cfd_path
        config_cfg_name = os.path.basename(cfg_cfd_path).split(".")[0]
        pattern = r"(na\d+e[+-]\d+)"
        match = re.search(pattern, config_cfg_name)
        na_value = match.group(1)

        cfg_srda_path = parser.parse_args().config_srda_path
        config_srda_name = os.path.basename(cfg_srda_path).split(".")[0]

        model_name = parser.parse_args().model_name

        result_dir = f"{RESULT_DIR}/{experiment_name}/letkf/perform_letkf_hr_using_uhr/use_narrow_jet/store_only_forecast/{na_value}_{config_name}"
        os.makedirs(result_dir, exist_ok=True)

        log_dir = f"{LOG_DIR}/{experiment_name}/letkf/perform_letkf_hr_using_uhr/use_narrow_jet/store_only_forecast/{na_value}_{config_name}"
        os.makedirs(log_dir, exist_ok=True)

        logger.addHandler(FileHandler(f"{log_dir}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        # config letkf
        cfg_letkf = LETKFBlockProcessingConfig.load(pathlib.Path(cfg_letkf_path))

        logger.info(f"Input config of letkf = {cfg_letkf.to_json_str()}")

        # config cfd
        cfg_cfd = CFDConfig.load(pathlib.Path(cfg_cfd_path))

        if cfg_cfd.hr_base_config.precision == "double":
            complex_dtype = torch.complex128
            real_dtype = torch.float64
        elif cfg_cfd.hr_base_config.precision == "single":
            complex_dtype = torch.complex64
            real_dtype = torch.float32

        # Constant
        NUM_TIMES = int(cfg_cfd.time_config.end_time / cfg_cfd.time_config.output_hr_dt)
        ASSIMILATION_PERIOD = cfg_cfd.da_config.assimilation_interval

        logger.info("\n*********************************************************")
        logger.info("Preparation for simulation starts")
        logger.info("*********************************************************\n")

        model = QGModel(cfg_cfd.hr_base_config, show_input_cfg_info=True)

        #
        x_coords, y_coords, z_coords = model.get_grids()
        flattend_x_coords = x_coords.reshape(-1)
        flattend_y_coords = y_coords.reshape(-1)
        flattend_z_coords = z_coords.reshape(-1)

        block_idx, center_idx = make_splitted_grid_and_its_center_indices_array(
            cfg_letkf
        )

        #
        if os.path.exists(f"{result_dir}/all_elapsed_time.csv"):
            df = pd.read_csv(f"{result_dir}/all_elapsed_time.csv")
            all_elapsed_time = df.to_dict(orient="records")

            #
            i_seed_uhr_max = df["I_SEED_UHR"].max()
            if i_seed_uhr_max == cfg_cfd.seed_config.uhr_seed_end:
                raise ValueError("All simulation is already finished.")

            i_seed_uhr_start = i_seed_uhr_max + 1

            #
            is_all_letkf_fcst = True
            all_letkf_fcst = np.load(f"{result_dir}/all_letkf_fcst.npy")
            all_letkf_fcst = torch.from_numpy(all_letkf_fcst)

            logger.info("Existing result is loaded.")
            logger.info(f"i seed uhr start: {i_seed_uhr_start}\n")
        else:
            all_elapsed_time = []
            i_seed_uhr_start = cfg_cfd.seed_config.uhr_seed_start
            is_all_letkf_fcst = False

            logger.info("This is the first simulation.\n")

        logger.info("\n*********************************************************")
        logger.info("Start simulation")
        logger.info("*********************************************************\n")

        all_letkf_means = []

        for i_seed_uhr in range(i_seed_uhr_start, cfg_cfd.seed_config.uhr_seed_end + 1):
            logger.info("\n*********************************************************")
            logger.info(f"I_SEED_UHR  = {i_seed_uhr }")
            logger.info("*********************************************************\n")

            #
            obs_npz_file_path = f"{RESULT_DIR}/{experiment_name}/srda/{model_name}/use_narrow_jet/store_only_forecast/{config_srda_name}/UHR_seed_{i_seed_uhr:05}.npz"
            all_data = np.load(obs_npz_file_path)

            hr_obs = torch.from_numpy(all_data["hr_obs"])

            #
            start_time_per_sim = time.time()

            set_seeds(seed=(i_seed_uhr + 1) * 457, use_deterministic=True)

            if cfg_cfd.jet_profile == "tanh":
                q0 = make_jet_pv_with_tanh_profile(cfg_cfd.hr_base_config)
            elif cfg_cfd.jet_profile == "linear":
                q0 = make_jet_pv_with_linear_profile(cfg_cfd.hr_base_config)
            elif cfg_cfd.jet_profile == "sech-squared":
                u = make_jet_velocity_with_sech_squared_profile(
                    cfg_cfd.hr_base_config, use_shift_y=False, use_narrow_jet=True
                )[None, ...]
                noise = cfg_cfd.hr_base_config.noise_amplitude * torch.randn(
                    size=(cfg_cfd.hr_base_config.n_batch,) + u.shape[-3:]
                )
                _ = QGModel(cfg_cfd.hr_base_config, show_input_cfg_info=False)
                _.initialize(grid_u=u, grid_pv_noise=noise)
                q0 = _.get_grid_pv()
            elif cfg_cfd.jet_profile == "sech_squared_and_sigmoid":
                u = make_jet_velocity_with_sech_squared_and_sigmoid_profile(
                    cfg_cfd.hr_base_config, use_shift_y=False, use_narrow_jet=True
                )[None, ...]
                noise = cfg_cfd.hr_base_config.noise_amplitude * torch.randn(
                    size=(cfg_cfd.hr_base_config.n_batch,) + u.shape[-3:]
                )
                _ = QGModel(cfg_cfd.hr_base_config, show_input_cfg_info=False)
                _.initialize(grid_u=u, grid_pv_noise=noise)
                q0 = _.get_grid_pv()
            else:
                raise ValueError(f"{cfg_cfd.jet_profile} is not supported.")

            obs_matrix = get_obs_matrix(
                obs=hr_obs[ASSIMILATION_PERIOD],
                nz=cfg_letkf.nz,
                ny=cfg_letkf.ny,
                nx=cfg_letkf.nx,
                device="cpu",
            )
            # Calculate distances and decay factors for localization
            dist_xy, dist_z = calc_distances_from_obs_grid_points(
                obs_matrix=obs_matrix,
                flattend_x_coords=flattend_x_coords,
                flattend_y_coords=flattend_y_coords,
                flattend_z_coords=flattend_z_coords,
                cfg=cfg_letkf,
            )
            decay_factors = calc_decay_factors_of_localization(
                dist_xy=dist_xy,
                dist_z=dist_z,
                obs_matrix=obs_matrix,
                cfg=cfg_letkf,
            )

            model.initialize_pv(q0)

            hr_letkfs, ts = [], []

            for i_cycle in tqdm(range(NUM_TIMES)):
                start_time_per_cycle = time.time()
                logger.info(f"Cycle: {i_cycle + 1} / {NUM_TIMES}")

                # Store data "before" assimilation
                hr_letkfs.append(model.get_grid_pv())

                # Data assimilation
                if i_cycle > 0 and i_cycle % ASSIMILATION_PERIOD == 0:
                    #
                    gt_added_noise = hr_obs[i_cycle].to(real_dtype)

                    # This is to avoid nan when observation operator acts.
                    gt_added_noise = torch.nan_to_num(gt_added_noise, nan=1e10)

                    hr_state = model.get_grid_pv()

                    analysis_all = hr_assimilate_with_existing_data(
                        hr_pv=hr_state.to(cfg_cfd.hr_base_config.device),
                        gt_added_noise=gt_added_noise.to(cfg_cfd.hr_base_config.device),
                        obs_matrix=obs_matrix.to(cfg_cfd.hr_base_config.device),
                        dist_xy=dist_xy,
                        dist_z=dist_z,
                        decay_factors=decay_factors,
                        block_idx=block_idx,
                        center_idx=center_idx,
                        cfg=cfg_letkf,
                    )
                    assert analysis_all.shape == model.get_grid_pv().shape

                    # model.set_pv(grid_pv=analysis_all)
                    model.initialize_pv(grid_pv=analysis_all)

                ts.append(model.t)

                n_steps = int(
                    cfg_cfd.time_config.output_hr_dt / cfg_cfd.time_config.hr_dt
                )

                model.integrate_n_steps(
                    dt_per_step=cfg_cfd.time_config.hr_dt, n_steps=n_steps
                )

                end_time_per_cycle = time.time()
                logger.info(
                    f"Elapsed time per cycle = {end_time_per_cycle - start_time_per_cycle} sec\n"
                )

            # Stack arrays along time dim
            hr_letkfs = torch.stack(hr_letkfs, dim=1)
            # dim = num_ens, time, z, y, x
            assert hr_letkfs.shape[1] == NUM_TIMES

            # Add an ensemble mean
            all_letkf_means.append(torch.mean(hr_letkfs, dim=0))

            end_time_per_sim = time.time()
            logger.info(
                f"Elapsed time per simulation = {end_time_per_sim - start_time_per_sim} sec"
            )

            elapsed_time = {}
            elapsed_time["I_SEED_UHR"] = i_seed_uhr
            elapsed_time["time_second"] = end_time_per_sim - start_time_per_sim
            all_elapsed_time.append(elapsed_time)

            pd.DataFrame(all_elapsed_time).to_csv(
                f"{result_dir}/all_elapsed_time.csv",
                index=False,
            )

        # Stack along the simulation dim
        all_letkf_means = torch.stack(all_letkf_means, dim=0).to(torch.float32)

        if is_all_letkf_fcst:
            # The order is important
            # all_letkf_fcst is stacked first along the seed dimension, followed by all_letkf_means
            all_letkf_means = torch.cat((all_letkf_fcst, all_letkf_means), dim=0)
            np.save(f"{result_dir}/all_letkf_fcst.npy", all_letkf_means)
        else:
            np.save(f"{result_dir}/all_letkf_fcst.npy", all_letkf_means)

        end_time = time.time()
        logger.info(f"Total elapsed time = {(end_time - start_time) / 60.} min")

        #
        output_gpu_memory_summary_log()

        logger.info("\n*********************************************************")
        logger.info(f"End: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())
