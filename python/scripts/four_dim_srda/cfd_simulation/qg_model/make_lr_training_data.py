import argparse
import datetime
import glob
import os
import pathlib
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import torch
from src.four_dim_srda.config.experiment_config import CFDConfig
from src.qg_model.low_pass_filter import LowPassFilter
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

DATA_DIR = f"{ROOT_DIR}/data/four_dim_srda"
LOG_DIR = f"{ROOT_DIR}/python/logs/four_dim_srda"
SIMULATION_NAME = "cfd_simulation/qg_model"


def load_splitted_n_batch_data(
    *,
    npy_file_paths: list[str],
) -> torch.Tensor:
    #
    # add batch dim by np.expand_dims
    # loaded data dim is changed: (time, nz, ny, nx) -> (batch, time, nz, ny, nx)
    # and appended in arrays

    arrays = [np.expand_dims(np.load(_path), axis=0) for _path in npy_file_paths]

    # combine arrays along batch dim
    combined_array = np.concatenate(arrays, axis=0)

    return torch.tensor(combined_array)


def extract_init_conditions_from_time_series_data(
    hr_data: torch.Tensor,
) -> torch.Tensor:
    #
    # hr_data dim = (batch, time, nz, ny, nx)
    # _data dim -> (batch, nz, ny, nx)
    _data = hr_data[:, 0, :, :, :].squeeze(dim=1)

    return _data


def make_and_initialize_lr_model(hr_pv: torch.Tensor):
    lr_pv = low_pass_filter.apply(hr_grid_data=hr_pv)
    lr_model = QGModel(cfg.lr_base_config, show_input_cfg_info=False)
    lr_model.initialize_pv(grid_pv=lr_pv)

    return lr_model


def write_out(
    *,
    lr_pv: torch.Tensor,
    npy_file_paths: list[str],
    dtype: torch.dtype,
):
    #
    lr_pv = lr_pv.to(dtype)
    for i_batch, _path in enumerate(npy_file_paths):
        _path = _path.replace("hr_pv", "lr_pv")
        np.save(_path, lr_pv[i_batch].numpy())


if __name__ == "__main__":
    try:
        start_time = time.time()

        config_path = parser.parse_args().config_path
        config_name = os.path.basename(config_path).split(".")[0]
        experiment_name = config_path.split("/")[-4]

        cfg = CFDConfig.load(pathlib.Path(config_path))

        data_dir = f"{DATA_DIR}/{experiment_name}/{SIMULATION_NAME}/hr_and_lr_pv"
        log_dir = f"{LOG_DIR}/{experiment_name}/{SIMULATION_NAME}/hr_and_lr_pv/lr_pv/{config_name}"

        os.makedirs(log_dir, exist_ok=False)

        logger.addHandler(FileHandler(f"{log_dir}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        logger.info(f"Input config = {cfg.to_json_str()}")

        logger.info("\n*********************************************************")
        logger.info("Prepare Low-pass filter")
        logger.info("*********************************************************\n")

        if cfg.hr_base_config.precision == "double":
            complex_dtype = torch.complex128
            real_dtype = torch.float64
        elif cfg.hr_base_config.precision == "single":
            complex_dtype = torch.complex64
            real_dtype = torch.float32

        low_pass_filter = LowPassFilter(
            nx_lr=cfg.lr_base_config.nx,
            ny_lr=cfg.lr_base_config.ny,
            nz_lr=cfg.lr_base_config.nz,
            nx_hr=cfg.hr_base_config.nx,
            ny_hr=cfg.hr_base_config.ny,
            nz_hr=cfg.hr_base_config.nz,
            dtype=complex_dtype,
            device=cfg.hr_base_config.device,
        )

        logger.info("\n*********************************************************")
        logger.info("Start making lr data")
        logger.info("*********************************************************\n")

        output_interval = torch.arange(
            cfg.time_config.start_time + cfg.da_config.assimilation_interval,
            cfg.da_config.segment_length,
            cfg.da_config.assimilation_interval,
        )

        for i_seed in tqdm(
            range(cfg.seed_config.seed_start, cfg.seed_config.seed_end + 1)
        ):
            start_time_per_seed = time.time()

            logger.info("\n*********************************************************")
            logger.info(f"i_seed = {i_seed}")
            logger.info("*********************************************************\n")

            npy_file_paths = sorted(
                glob.glob(os.path.join(f"{data_dir}/seed{i_seed:05}", "*_hr_pv_*.npy"))
            )

            #
            # Run lr simulation at each time with n_batch data
            #
            for i in range(
                0,
                len(npy_file_paths) - cfg.hr_base_config.n_batch + 1,
                cfg.hr_base_config.n_batch,
            ):
                start_time_per_n_batch = time.time()
                logger.info(
                    f"start lr simulation with {cfg.hr_base_config.n_batch} data"
                )
                logger.info(
                    f"initial time step = {npy_file_paths[i].split('_start')[1][:3]}"
                )

                hr_data = load_splitted_n_batch_data(
                    npy_file_paths=npy_file_paths[i : i + cfg.hr_base_config.n_batch],
                )
                hr_init_pv = extract_init_conditions_from_time_series_data(
                    hr_data=hr_data
                )

                lr_model = make_and_initialize_lr_model(hr_pv=hr_init_pv.to(real_dtype))

                # run simulation
                qs = [lr_model.get_grid_pv()]
                for _ in output_interval:
                    n_steps = int(cfg.time_config.output_lr_dt / cfg.time_config.lr_dt)

                    lr_model.integrate_n_steps(
                        dt_per_step=cfg.time_config.lr_dt, n_steps=n_steps
                    )

                    qs.append(lr_model.get_grid_pv())

                # Stack arrays along time dim
                qs = torch.stack(qs, dim=1)
                # qs dim = batch, time, z, y, x

                write_out(
                    lr_pv=qs,
                    npy_file_paths=npy_file_paths[i : i + cfg.hr_base_config.n_batch],
                    dtype=torch.float32,
                )

                logger.info("data were written out.")

                end_time_per_n_batch = time.time()
                logger.info(
                    f"Elapsed time per n_batch = {end_time_per_n_batch - start_time_per_n_batch} sec\n"
                )

            end_time_per_seed = time.time()
            logger.info(
                f"Elapsed time per seed = {end_time_per_seed- start_time_per_seed} sec\n"
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
