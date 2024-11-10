#!/bin/sh

# Specify your root directory of this repository in the Singularity container.
ROOT_DIR=$(pwd)

IMAGE_PATH=${ROOT_DIR}/pytorch_es.sif

EXPERIMENT_NAME=experiment7
MODEL_NAME=UNetVitVer02

BEA_NUM=2
BED_NUM=2
DESPE_NUM=360
NSLS_NUM=100
OGX_NUM=08
OGY_NUM=08

N3DRB_NUM=3
NVB_NUM=4
BIAS=0
VITS=0

BS_NUM=12
LR_NUM=1e-04

SCRIPT_PATH=${ROOT_DIR}/python/scripts/four_dim_srda/srda/evaluate_using_fixed_obs_point_and_narrow_jet.py

CONFIG_DIR=${ROOT_DIR}/python/configs/four_dim_srda/${EXPERIMENT_NAME}

CONFIG_SRDA_PATH=${CONFIG_DIR}/perform_4D_SRDA/${MODEL_NAME}/\
bea${BEA_NUM}_bed${BED_NUM}_dspe${DESPE_NUM}_nsls${NSLS_NUM}_ogx${OGX_NUM}_ogy${OGY_NUM}_\
n3drb${N3DRB_NUM}_nvb${NVB_NUM}_bias${BIAS}_vits${VITS}_bs${BS_NUM}_lr${LR_NUM}.yml

CONFIG_CFD_PATH=${CONFIG_DIR}/cfd_simulation/qg_model/gpu_evaluation_config.yml

singularity exec \
  --nv \
  --env PYTHONPATH=${ROOT_DIR}/python \
  --bind ${ROOT_DIR}:${ROOT_DIR} \
  ${IMAGE_PATH} python3 ${SCRIPT_PATH} \
  --config_srda_path ${CONFIG_SRDA_PATH} \
  --config_cfd_path ${CONFIG_CFD_PATH} \
  --model_name ${MODEL_NAME}