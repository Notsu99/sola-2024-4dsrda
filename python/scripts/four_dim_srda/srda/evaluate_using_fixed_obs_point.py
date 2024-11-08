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
from src.four_dim_srda.config.config_loader import load_config
from src.four_dim_srda.config.experiment_config import CFDConfig
from src.four_dim_srda.models.model_maker import make_model
from src.four_dim_srda.utils.random_seed_helper import set_seeds
from src.four_dim_srda.utils.sr_da_helper import (
    get_observation_with_noise_using_fixed_obs_point,
    get_testdataset,
    get_uhr_and_hr_pvs,
    initialize_and_itegrate_lr_cfd_model_for_forecast,
    make_invprocessed_srda_for_forecast,
    make_preprocessed_lr_for_forecast,
    make_preprocessed_obs_for_forecast,
)
from src.qg_model.jet_maker import (
    make_jet_pv_with_linear_profile,
    make_jet_pv_with_tanh_profile,
    make_jet_velocity_with_sech_squared_and_sigmoid_profile,
    make_jet_velocity_with_sech_squared_profile,
)
from src.qg_model.low_pass_filter import LowPassFilter
from src.qg_model.qg_model import QGModel
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=True)

logger = getLogger()
if not logger.hasHandlers():
    logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_srda_path", type=str, required=True)
parser.add_argument("--config_cfd_path", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)

ROOT_DIR = pathlib.Path(os.environ["PYTHONPATH"]).parent.resolve()
LOG_DIR = f"{ROOT_DIR}/python/logs/four_dim_srda"
RESULT_DIR = f"{ROOT_DIR}/python/results/four_dim_srda"
DATA_DIR = f"{ROOT_DIR}/data/four_dim_srda"


if __name__ == "__main__":
    try:
        start_time = time.time()

        cfg_srda_path = parser.parse_args().config_srda_path
        cfg_srda_name = os.path.basename(cfg_srda_path).split(".")[0]
        experiment_name = cfg_srda_path.split("/")[-4]

        cfg_cfd_path = parser.parse_args().config_cfd_path
        # cfg_cfd_name = os.path.basename(cfg_cfd_path).split(".")[0]

        model_name = parser.parse_args().model_name

        log_dir = f"{LOG_DIR}/{experiment_name}/evaluation/{model_name}/{cfg_srda_name}"
        result_dir = f"{RESULT_DIR}/{experiment_name}/srda/{model_name}/{cfg_srda_name}"

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        logger.addHandler(FileHandler(f"{log_dir}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        # config srda
        cfg_srda = load_config(model_name=model_name, config_path=cfg_srda_path)

        # config cfd
        cfg_cfd = CFDConfig.load(pathlib.Path(cfg_cfd_path))

        if cfg_cfd.hr_base_config.precision == "double":
            complex_dtype = torch.complex128
            real_dtype = torch.float64
        elif cfg_cfd.hr_base_config.precision == "single":
            complex_dtype = torch.complex64
            real_dtype = torch.float32

        logger.info(f"Input config of Neural network = {cfg_srda.to_json_str()}")

        logger.info(f"Input config of CFD model = {cfg_cfd.to_json_str()}")

        # Constant

        UHR_DATA_DIR = f"{DATA_DIR}/{experiment_name}/cfd_simulation/qg_model/uhr_pv"

        weight_path = f"{DATA_DIR}/{experiment_name}/training/{model_name}/{cfg_srda_name}/best_weights.pth"

        ASSIMILATION_PERIOD = (
            cfg_cfd.da_config.segment_length - cfg_cfd.da_config.forecast_span - 1
        )
        FORECAST_SPAN = cfg_cfd.da_config.forecast_span

        # 対象とする時間は0 <= t <= (NUM_TIMES - 1) * cfg_cfd.time_config.output_hr_dt
        NUM_TIMES = (
            cfg_srda.dataset_config.max_start_time_index
            + ASSIMILATION_PERIOD
            + FORECAST_SPAN
        )

        OUTPUT_SCALE_FACTOR = int(
            cfg_cfd.time_config.output_lr_dt // cfg_cfd.time_config.output_hr_dt
        )

        logger.info("\n###############################")
        logger.info("Prepare Low-pass filter")
        logger.info("###############################\n")

        low_pass_filter = LowPassFilter(
            nx_lr=cfg_cfd.lr_base_config.nx,
            ny_lr=cfg_cfd.lr_base_config.ny,
            nz_lr=cfg_cfd.lr_base_config.nz,
            nx_hr=cfg_cfd.hr_base_config.nx,
            ny_hr=cfg_cfd.hr_base_config.ny,
            nz_hr=cfg_cfd.hr_base_config.nz,
            dtype=complex_dtype,
            device=cfg_cfd.hr_base_config.device,
        )

        logger.info("\n###############################")
        logger.info("Prepare SRDA Model")
        logger.info("###############################\n")

        current_device = torch.cuda.current_device()
        srda_model = make_model(cfg=cfg_srda)
        srda_model.load_state_dict(
            torch.load(weight_path, map_location=f"cuda:{current_device}")
        )
        srda_model.to(f"cuda:{current_device}")
        _ = srda_model.eval()

        logger.info("\n###############################")
        logger.info("Prepare Test Dataset")
        logger.info("###############################\n")

        test_dataset = get_testdataset(cfg=cfg_srda, root_dir=ROOT_DIR)

        assert (
            test_dataset.cfg.lr_and_obs_time_interval
            == cfg_cfd.da_config.assimilation_interval
        )
        assert test_dataset.cfg.is_future_obs_missing == True

        logger.info("\n###############################")
        logger.info("Start simulation")
        logger.info("###############################\n")

        all_elapsed_time = []

        for i_seed_uhr in range(
            cfg_cfd.seed_config.uhr_seed_start, cfg_cfd.seed_config.uhr_seed_end + 1
        ):
            logger.info("\n###############################")
            logger.info(f"i_seed_uhr = {i_seed_uhr}")
            logger.info("###############################\n")

            # It's important to set the different seed number from the simulation of making UHR data
            # This is related to random noise added to init_data
            set_seeds(555 * i_seed_uhr, use_deterministic=True)

            # Initialize data
            if cfg_cfd.jet_profile == "tanh":
                init_hr_pv = make_jet_pv_with_tanh_profile(cfg_cfd.hr_base_config)
            elif cfg_cfd.jet_profile == "linear":
                init_hr_pv = make_jet_pv_with_linear_profile(cfg_cfd.hr_base_config)
            elif cfg_cfd.jet_profile == "sech-squared":
                u = make_jet_velocity_with_sech_squared_profile(cfg_cfd.hr_base_config)[
                    None, ...
                ]
                noise = cfg_cfd.hr_base_config.noise_amplitude * torch.randn(
                    size=(cfg_cfd.hr_base_config.n_batch,) + u.shape[-3:]
                )
                _ = QGModel(cfg_cfd.hr_base_config, show_input_cfg_info=False)
                _.initialize(grid_u=u, grid_pv_noise=noise)
                init_hr_pv = _.get_grid_pv()
            elif cfg_cfd.jet_profile == "sech_squared_and_sigmoid":
                u = make_jet_velocity_with_sech_squared_and_sigmoid_profile(cfg_cfd.hr_base_config)[
                    None, ...
                ]
                noise = cfg_cfd.hr_base_config.noise_amplitude * torch.randn(
                    size=(cfg_cfd.hr_base_config.n_batch,) + u.shape[-3:]
                )
                _ = QGModel(cfg_cfd.hr_base_config, show_input_cfg_info=False)
                _.initialize(grid_u=u, grid_pv_noise=noise)
                init_hr_pv = _.get_grid_pv()
            else:
                raise ValueError(f"{cfg_cfd.jet_profile} is not supported.")

            assert init_hr_pv.shape == (
                1,
                cfg_cfd.hr_base_config.nz,
                cfg_cfd.hr_base_config.ny,
                cfg_cfd.hr_base_config.nx,
            )

            uhr_data_dir_path = f"{UHR_DATA_DIR}/seed{i_seed_uhr:05}"

            uhr_pvs, hr_pvs = get_uhr_and_hr_pvs(
                result_dir_path=uhr_data_dir_path,
                i_seed_uhr=i_seed_uhr,
                num_times=NUM_TIMES,
                cfg=cfg_cfd,
            )

            hr_obsrvs = get_observation_with_noise_using_fixed_obs_point(
                hr_pv=hr_pvs[None, ...],  # add ens channel (dummy channel)
                test_dataset=test_dataset,
                cfg_cfd=cfg_cfd,
            ).squeeze()

            assert (
                hr_pvs.shape
                == hr_obsrvs.shape
                == (
                    NUM_TIMES,
                    cfg_cfd.hr_base_config.nz,
                    cfg_cfd.hr_base_config.ny,
                    cfg_cfd.hr_base_config.nx,
                )
            )

            logger.info("uhr pvs, hr_pvs, hr_obrvs were made")

            start_time_per_seed = time.time()

            last_hr_pv0 = init_hr_pv
            hr_obs, srda_forecast, all_lr_forecast = [], [], []

            for i_cycle in tqdm(range(NUM_TIMES)):
                if i_cycle % ASSIMILATION_PERIOD == 0:
                    o = hr_obsrvs[i_cycle]
                    hr_obs.append(o[None, ...])  # add channel dim
                else:
                    o = hr_obsrvs[i_cycle]
                    hr_obs.append(torch.full_like(o[None, ...], torch.nan))

                if i_cycle > 0 and i_cycle % ASSIMILATION_PERIOD == 0:
                    num_integrate_steps = (
                        ASSIMILATION_PERIOD + FORECAST_SPAN
                    ) // OUTPUT_SCALE_FACTOR

                    lr_forecast = initialize_and_itegrate_lr_cfd_model_for_forecast(
                        last_hr_pv0=last_hr_pv0,
                        num_integrate_steps=num_integrate_steps,
                        low_pass_filter=low_pass_filter,
                        cfg_cfd=cfg_cfd,
                    )
                    # lr_forecast shape = (batch, time, z, y, x)
                    assert lr_forecast.shape[1] == num_integrate_steps + 1

                    x = make_preprocessed_lr_for_forecast(
                        lr_forecast=lr_forecast,
                        test_dataset=test_dataset,
                        cfg_cfd=cfg_cfd,
                    )
                    o = make_preprocessed_obs_for_forecast(
                        hr_obs=hr_obs,
                        assimilation_period=ASSIMILATION_PERIOD,
                        forecast_span=FORECAST_SPAN,
                        test_dataset=test_dataset,
                        cfg_cfd=cfg_cfd,
                    )

                    # Check num of time dims
                    # batch, time, channel, z, y, x
                    _sum = ASSIMILATION_PERIOD + FORECAST_SPAN
                    _nt = int(
                        _sum / cfg_srda.dataset_config.lr_and_obs_time_interval + 1
                    )
                    assert x.shape[1] == o.shape[1] == _nt

                    with torch.no_grad():
                        srda = srda_model(x, o).detach().cpu().clone()

                    # srda dim = batch, time, channel, z, y, x
                    srda = make_invprocessed_srda_for_forecast(
                        preds=srda,
                        assimilation_period=ASSIMILATION_PERIOD,
                        forecast_span=FORECAST_SPAN,
                        test_dataset=test_dataset,
                        cfg_cfd=cfg_cfd,
                    )
                    # srda dim = batch, time, z, y, x

                    last_hr_pv0 = srda[:, ASSIMILATION_PERIOD, :, :, :].clone()

                    assert last_hr_pv0.shape == (
                        1,
                        cfg_cfd.hr_base_config.nz,
                        cfg_cfd.hr_base_config.ny,
                        cfg_cfd.hr_base_config.nx,
                    )

                    # The indices of srda_forecast between 0 to ASSIMILATION_PERIOD - 1 are past
                    # So NaN values are substituted for it.
                    if len(srda_forecast) == 0:
                        dummy = torch.full(
                            size=(
                                srda.shape[0],
                                ASSIMILATION_PERIOD,
                            )
                            + srda.shape[2:],
                            fill_value=torch.nan,
                            dtype=srda.dtype,
                        )
                        srda_forecast += dummy[0]

                        dummy_lr = torch.full(
                            size=(
                                lr_forecast.shape[0],
                                ASSIMILATION_PERIOD // OUTPUT_SCALE_FACTOR,
                            )
                            + lr_forecast.shape[2:],
                            fill_value=torch.nan,
                            dtype=srda.dtype,
                        )
                        all_lr_forecast += dummy_lr[0]

                    i_start = ASSIMILATION_PERIOD
                    i_end = ASSIMILATION_PERIOD + FORECAST_SPAN
                    srda_forecast += srda[0][i_start:i_end]

                    all_lr_forecast += lr_forecast[0][
                        int(i_start / OUTPUT_SCALE_FACTOR) : int(
                            i_end / OUTPUT_SCALE_FACTOR
                        )
                    ]

                    logger.debug(f"Assimilation at i = {i_cycle}")

            end_time_per_seed = time.time()
            logger.info(
                f"Elapsed time per seed = {end_time_per_seed - start_time_per_seed} sec\n"
            )

            # Stack along time dim
            srda_forecast = torch.stack(srda_forecast, dim=0)
            hr_obs = torch.stack(hr_obs, dim=1).squeeze()
            all_lr_forecast = torch.stack(all_lr_forecast, dim=0)

            assert (
                hr_obs.shape
                == srda_forecast.shape
                == hr_pvs.shape
                == (
                    NUM_TIMES,
                    cfg_cfd.hr_base_config.nz,
                    cfg_cfd.hr_base_config.ny,
                    cfg_cfd.hr_base_config.nx,
                )
            )

            output_npz_file_path = f"{result_dir}/UHR_seed_{i_seed_uhr:05}.npz"
            np.savez(
                output_npz_file_path,
                hr_obs=hr_obs,
                srda_forecast=srda_forecast,
                all_lr_forecast=all_lr_forecast,
            )

            elapsed_time = {}
            elapsed_time["I_SEED_UHR"] = i_seed_uhr
            elapsed_time["time_second"] = end_time_per_seed - start_time_per_seed

            all_elapsed_time.append(elapsed_time)

            pd.DataFrame(all_elapsed_time).to_csv(
                f"{result_dir}/all_elapsed_time.csv",
                index=False,
            )

            logger.info(
                f"result data shape: hr_obs = {hr_obs.shape}, srda_forecast = {srda_forecast.shape}, all_lr_forecast = {all_lr_forecast.shape}\n"
            )

            logger.info("result were written out.\n")

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
