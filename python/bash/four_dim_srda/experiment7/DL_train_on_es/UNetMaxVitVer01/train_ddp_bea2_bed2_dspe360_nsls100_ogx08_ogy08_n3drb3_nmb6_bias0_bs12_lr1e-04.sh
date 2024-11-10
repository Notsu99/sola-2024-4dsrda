#!/bin/sh

# Specify your root directory of this repository in the Singularity container.
ROOT_DIR=$(pwd)

IMAGE_PATH=${ROOT_DIR}/pytorch_es.sif

EXPERIMENT_NAME=experiment7
MODEL_NAME=UNetMaxVitVer01

BEA_NUM=2
BED_NUM=2
DESPE_NUM=360
NSLS_NUM=100
OGX_NUM=08
OGY_NUM=08

N3DRB_NUM=3
NMB_NUM=6
BIAS=0

BS_NUM=12
LR_NUM=1e-04

SCRIPT_PATH=${ROOT_DIR}/python/scripts/four_dim_srda/srda/train_ddp_ml_model.py

CONFIG_PATH=${ROOT_DIR}/python/configs/four_dim_srda/${EXPERIMENT_NAME}/perform_4D_SRDA/${MODEL_NAME}/\
bea${BEA_NUM}_bed${BED_NUM}_dspe${DESPE_NUM}_nsls${NSLS_NUM}_ogx${OGX_NUM}_ogy${OGY_NUM}_\
n3drb${N3DRB_NUM}_nmb${NMB_NUM}_bias${BIAS}_bs${BS_NUM}_lr${LR_NUM}.yml

# Specify the number of GPUs to be used for distributed training.
# Adjust this value according to the number of GPUs available in your environment.
# In this study, we used 3 GPUs for training.
singularity exec \
  --nv \
  --env PYTHONPATH=${ROOT_DIR}/python \
  --bind ${ROOT_DIR}:${ROOT_DIR} \
  ${IMAGE_PATH} python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH} --world_size 3 --model_name ${MODEL_NAME}