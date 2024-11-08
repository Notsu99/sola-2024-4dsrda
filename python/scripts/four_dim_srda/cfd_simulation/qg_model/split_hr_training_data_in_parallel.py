import argparse
import datetime
import os
import pathlib
import sys
import time
import traceback
import multiprocessing
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
from src.four_dim_srda.config.experiment_config import CFDConfig

logger = getLogger()
if not logger.hasHandlers():
    logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)

ROOT_DIR = pathlib.Path(os.environ["PYTHONPATH"]).parent.resolve()

DATA_DIR = f"{ROOT_DIR}/data/four_dim_srda"
RESULT_DIR = f"{ROOT_DIR}/data/four_dim_srda"
LOG_DIR = f"{ROOT_DIR}/python/logs/four_dim_srda"
SIMULATION_NAME = "cfd_simulation/qg_model"


def split_and_save_data(
    *,
    i_seed: int,
    out_step_start: int,
    out_step_end: int,
    assimilation_interval: int,
    segment_length: int,
    n_batch: int,
    data_dir_path: str,
    result_dir_path: str,
):
    for ib in range(n_batch):
        hr_data_path = f"{data_dir_path}/seed{i_seed:05}_start{out_step_start:03}_end{out_step_end:03}_hr_pv_{ib:02}.npy"
        hr_data = np.load(hr_data_path)

        max_start_step = out_step_end - segment_length + 1
        for s_step in range(0, out_step_end, assimilation_interval):
            if s_step > max_start_step:
                continue
            e_step = s_step + segment_length
            segment_hr_data = hr_data[s_step:e_step, :, :, :]

            file_name = (
                f"seed{i_seed:05}_start{s_step:03d}_end{e_step-1:03d}_hr_pv_{ib:02}"
            )
            np.save(f"{result_dir_path}/{file_name}.npy", segment_hr_data)
            logger.info(f"segmented hr data shape: {segment_hr_data.shape}")


def process_seed(seed_range, cfg, data_dir, result_dir):
    for i_seed in seed_range:
        _path = f"{data_dir}/seed{i_seed:05}"
        result_dir_path = f"{result_dir}/seed{i_seed:05}"
        os.makedirs(result_dir_path, exist_ok=True)

        split_and_save_data(
            i_seed=i_seed,
            out_step_start=int(
                cfg.time_config.start_time / cfg.time_config.output_hr_dt
            ),
            out_step_end=int(cfg.time_config.end_time / cfg.time_config.output_hr_dt),
            assimilation_interval=cfg.da_config.assimilation_interval,
            segment_length=cfg.da_config.segment_length,
            n_batch=cfg.hr_base_config.n_batch,
            data_dir_path=_path,
            result_dir_path=result_dir_path,
        )


if __name__ == "__main__":
    try:
        start_time = time.time()

        config_path = parser.parse_args().config_path
        config_name = os.path.basename(config_path).split(".")[0]
        experiment_name = config_path.split("/")[-4]

        cfg = CFDConfig.load(pathlib.Path(config_path))

        data_dir = f"{DATA_DIR}/{experiment_name}/{SIMULATION_NAME}/hr_pv"
        result_dir = f"{RESULT_DIR}/{experiment_name}/{SIMULATION_NAME}/hr_and_lr_pv"

        log_dir = f"{LOG_DIR}/{experiment_name}/{SIMULATION_NAME}/hr_and_lr_pv/split_hr_training_data/{config_name}"

        os.makedirs(result_dir, exist_ok=False)
        os.makedirs(log_dir, exist_ok=False)

        logger.addHandler(FileHandler(f"{log_dir}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        logger.info(f"Input config = {cfg.to_json_str()}")

        logger.info("\n*********************************************************")
        logger.info("Start split of data")
        logger.info("*********************************************************\n")

        num_processes = 4 * multiprocessing.cpu_count()

        seed_range = np.array_split(
            range(cfg.seed_config.seed_start, cfg.seed_config.seed_end + 1),
            num_processes,
        )

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                process_seed,
                [(seed_range, cfg, data_dir, result_dir) for seed_range in seed_range],
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
