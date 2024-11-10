#!/bin/sh

# Specify your root directory of this repository in the Singularity container.
ROOT_DIR=$(pwd)

IMAGE_PATH=${ROOT_DIR}/pytorch_es.sif

EXPERIMENT_NAME=experiment7

SCRIPT_PATH=${ROOT_DIR}/python/scripts/four_dim_srda/cfd_simulation/qg_model/make_uhr_data_using_shifted_narrow_jet.py
CONFIG_PATH=${ROOT_DIR}/python/configs/four_dim_srda/${EXPERIMENT_NAME}/cfd_simulation/qg_model/gpu_make_data_config.yml

singularity exec \
  --nv \
  --env PYTHONPATH=${ROOT_DIR}/python \
  --bind ${ROOT_DIR}:${ROOT_DIR} \
  ${IMAGE_PATH} python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH}
