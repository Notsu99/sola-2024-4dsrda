#!/bin/sh

# Specify your root directory of this repository in the Singularity container.
ROOT_DIR=$(pwd)

IMAGE_PATH=${ROOT_DIR}/pytorch_es.sif

SCRIPT_PATH=${ROOT_DIR}/python/scripts/four_dim_srda/letkf/perform_letkf_hr_block_processing_using_uhr_and_fixed_obs_point_for_tuning.py

EXPERIMENT_NAME=experiment7
MODEL_NAME=UNetMaxVitVer01

OGX_NUM=08
OGY_NUM=08

ENS_NUM=100
NA_NUM=3e-03

CH_NUM=125e-03
CR_NUM=628e-01
INF_NUM=12e-01
LR_NUM=4e-01
BS_NUM_LETKF=6

BEA_NUM=2
BED_NUM=2
DESPE_NUM=360
NSLS_NUM=100

N3DRB_NUM=3
NMB_NUM=6
BIAS=0

BS_SRDA_NUM=12
LR_SRDA_NUM=1e-04

CONFIG_DIR=${ROOT_DIR}/python/configs/four_dim_srda/${EXPERIMENT_NAME}

CONFIG_CFD_PATH=${CONFIG_DIR}/perform_letkf_hr_using_uhr/tuning/ogx${OGX_NUM}_ogy${OGY_NUM}/cfd_cfg_ne${ENS_NUM}_na${NA_NUM}.yml

CONFIG_LETKF_PATH=${CONFIG_DIR}/perform_letkf_hr_using_uhr/tuning/ogx${OGX_NUM}_ogy${OGY_NUM}/letkf_cfg_ogx${OGX_NUM}_ogy${OGY_NUM}_\
ne${ENS_NUM}_ch${CH_NUM}_cr${CR_NUM}_if${INF_NUM}_lr${LR_NUM}_bs${BS_NUM_LETKF}.yml

CONFIG_SRDA_PATH=${CONFIG_DIR}/perform_4D_SRDA/${MODEL_NAME}/\
bea${BEA_NUM}_bed${BED_NUM}_dspe${DESPE_NUM}_nsls${NSLS_NUM}_ogx${OGX_NUM}_ogy${OGY_NUM}_\
n3drb${N3DRB_NUM}_nmb${NMB_NUM}_bias${BIAS}_bs${BS_SRDA_NUM}_lr${LR_SRDA_NUM}.yml

singularity exec \
  --nv \
  --env PYTHONPATH=${ROOT_DIR}/python \
  --bind ${ROOT_DIR}:${ROOT_DIR} \
  ${IMAGE_PATH} python3 ${SCRIPT_PATH} \
  --config_cfd_path ${CONFIG_CFD_PATH} \
  --config_letkf_path ${CONFIG_LETKF_PATH} \
  --config_srda_path ${CONFIG_SRDA_PATH} \
  --model_name ${MODEL_NAME}
