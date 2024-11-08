import argparse
import datetime
import os
import pathlib
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
from src.four_dim_srda.config.experiment_config import CFDConfig
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

logger = getLogger()
if not logger.hasHandlers():
    logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)

ROOT_DIR = pathlib.Path(os.environ["PYTHONPATH"]).parent.resolve()

RESULT_DIR = f"{ROOT_DIR}/data/four_dim_srda"
LOG_DIR = f"{ROOT_DIR}/python/logs/four_dim_srda"
SIMULATION_NAME = "cfd_simulation/qg_model"

DF_SEEDS = pd.read_csv(
    f"{ROOT_DIR}/python/configs/four_dim_srda/seed/seeds01.csv"
).set_index("SimulationNumber")


def write_out(
    *,
    hr_pv: torch.Tensor,
    i_seed: int,
    out_step_start: int,
    out_step_end: int,
    result_dir_path: str,
    dtype: torch.dtype,
):
    #
    hr_pv = hr_pv.to(dtype)
    file_path = f"{result_dir_path}/seed{i_seed:05}_start{out_step_start:03}_end{out_step_end:03}_uhr_pv.npy"
    np.save(file_path, hr_pv.numpy())


if __name__ == "__main__":
    try:
        start_time = time.time()

        config_path = parser.parse_args().config_path
        config_name = os.path.basename(config_path).split(".")[0]
        experiment_name = config_path.split("/")[-4]

        cfg = CFDConfig.load(pathlib.Path(config_path))

        result_dir = f"{RESULT_DIR}/{experiment_name}/{SIMULATION_NAME}/uhr_pv_narrow_jet"
        log_dir = f"{LOG_DIR}/{experiment_name}/{SIMULATION_NAME}/uhr_pv_narrow_jet/{config_name}"

        os.makedirs(result_dir, exist_ok=False)
        os.makedirs(log_dir, exist_ok=False)

        # logger.addHandler(FileHandler(f"{log_dir}/log.txt", mode='w'))
        logger.addHandler(FileHandler(f"{log_dir}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        logger.info(f"Input config = {cfg.to_json_str()}")

        logger.info("\n*********************************************************")
        logger.info("Make CFD model")
        logger.info("*********************************************************\n")

        model = QGModel(cfg.uhr_base_config)

        logger.info("\n*********************************************************")
        logger.info("Start simulation")
        logger.info("*********************************************************\n")

        for i_seed in range(
            cfg.seed_config.uhr_seed_start, cfg.seed_config.uhr_seed_end + 1
        ):
            start_time_per_loop = time.time()

            result_dir_path = f"{result_dir}/seed{i_seed:05}"
            os.makedirs(result_dir_path, exist_ok=True)

            seed = DF_SEEDS.loc[i_seed, "Seed0"]
            set_seeds(seed=seed, use_deterministic=True)

            logger.info("\n*********************************************************")
            logger.info(f"i_seed = {i_seed}, seed = {seed}")
            logger.info("*********************************************************\n")

            if cfg.jet_profile == "tanh":
                q0 = make_jet_pv_with_tanh_profile(cfg.uhr_base_config)
            elif cfg.jet_profile == "linear":
                q0 = make_jet_pv_with_linear_profile(cfg.uhr_base_config)
            elif cfg.jet_profile == "sech-squared":
                u = make_jet_velocity_with_sech_squared_profile(
                    cfg.uhr_base_config, use_shift_y=False, use_narrow_jet=True
                )[None, ...]
                noise = cfg.uhr_base_config.noise_amplitude * torch.randn(
                    size=(cfg.uhr_base_config.n_batch,) + u.shape[-3:]
                )
                model.initialize(grid_u=u, grid_pv_noise=noise)
            elif cfg.jet_profile == "sech_squared_and_sigmoid":
                u = make_jet_velocity_with_sech_squared_and_sigmoid_profile(
                    cfg.uhr_base_config, use_shift_y=False, use_narrow_jet=True
                )[None, ...]
                noise = cfg.uhr_base_config.noise_amplitude * torch.randn(
                    size=(cfg.uhr_base_config.n_batch,) + u.shape[-3:]
                )
                model.initialize(grid_u=u, grid_pv_noise=noise)
            else:
                raise ValueError(f"{cfg.jet_profile} is not supported.")

            qs, ts = [model.get_grid_pv()], [model.t]

            output_tsteps = torch.arange(
                cfg.time_config.start_time + cfg.time_config.output_uhr_dt,
                cfg.time_config.end_time + cfg.time_config.output_uhr_dt,
                cfg.time_config.output_uhr_dt,
            )

            for t in tqdm(output_tsteps):
                logger.debug(f"t = {t} start.")

                n_steps = int(cfg.time_config.output_uhr_dt / cfg.time_config.uhr_dt)

                model.integrate_n_steps(
                    dt_per_step=cfg.time_config.uhr_dt,
                    n_steps=n_steps,
                )

                qs.append(model.get_grid_pv())
                ts.append(model.t)

                logger.debug(f"t = {t} end.")

            # Stack arrays along time dim
            qs = torch.stack(qs, dim=1)
            # dim = batch, time, z, y, x before squeeze
            qs = qs.squeeze()
            # dim = time, z, y, x after squeeze

            logger.info(f"uhr_pv shape = {qs.shape}, data num along time = {len(ts)}\n")

            write_out(
                hr_pv=qs,
                i_seed=i_seed,
                out_step_start=int(
                    cfg.time_config.start_time / cfg.time_config.output_uhr_dt
                ),
                out_step_end=int(
                    cfg.time_config.end_time / cfg.time_config.output_uhr_dt
                ),
                result_dir_path=result_dir_path,
                dtype=torch.float32,
            )

            logger.info("data were written out.\n")

            end_time_per_loop = time.time()
            logger.info(
                f"Elapsed time per loop = {end_time_per_loop - start_time_per_loop} sec"
            )

        end_time = time.time()
        logger.info(f"Total elapsed time = {end_time - start_time} sec")

        logger.info("\n*********************************************************")
        logger.info(f"End: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())
