# sola-2024-4dsrda <!-- omit in toc -->

This repository contains the source code used in *Four-Dimensional Super-Resolution Data Assimilation (4D-SRDA) for Prediction of Three-Dimensional Quasi-Geostrophic Flows*

- [Setup](#setup)
  - [Singularity Container](#singularity-container)
- [How to Perform Experiments](#how-to-perform-experiments)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Testing](#testing)
- [Evaluation](#evaluation)

## Setup

- The Singularity container was used in this research.
  - The same environment is available through Docker.
- At least, 1 GPU board is required.

### Singularity Container

1. Install [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/quick_start.html).
2. Build Singularity container:
    - `$ singularity build -f pytorch_es.sif ./singularity/pytorch_es/pytorch_es.def`
3. Start singularity container:

```sh
$ singularity exec --nv --env PYTHONPATH="$(pwd)/python" \
    pytorch_es.sif jupyter lab \
    --no-browser --ip=0.0.0.0 --allow-root --LabApp.token='' --port=8888
```

## How to Perform Experiments

### Data Preparation

- CFD Simulations
  - Training data (Run the following scripts in order)
    1. [Make HR target data](./python/bash/four_dim_srda/experiment7/cfd_simulation/qg_model/es/es_make_hr_training_data.sh)
    2. [Split HR target data](./python/bash/four_dim_srda/experiment7/cfd_simulation/qg_model/es/es_split_hr_training_data_in_parallel.sh)
    3. [Make LR input data](./python/bash/four_dim_srda/experiment7/cfd_simulation/qg_model/es/es_make_lr_training_data.sh)
  - [Make LR forecast data](./python/bash/four_dim_srda/experiment7/cfd_simulation/qg_model/es/es_make_lr_forecast_data_using_narrow_jet.sh)
  - [Make UHR ground truth data](./python/bash/four_dim_srda/experiment7/cfd_simulation/qg_model/es/es_make_uhr_data_using_narrow_jet.sh)

### Training

- YO23 : A reference model
  - [Train YO23 with multiple GPUs](./python/bash/four_dim_srda/experiment7/DL_train_on_es/ConvTransNetVer01/train_ddp_bea2_bed2_dspe360_nsls100_ogx08_ogy08_bias1_bs12_lr1e-04.sh)
- U-Net MaxViT : A new model presented in this paper
  - [Train U-Net MaxViT with multiple GPUs](./python/bash/four_dim_srda/experiment7/DL_train_on_es/UNetMaxVitVer01/train_ddp_bea2_bed2_dspe360_nsls100_ogx08_ogy08_n3drb3_nmb6_bias0_bs12_lr1e-04.sh)


### Testing

- Simulations for obtaining analysis state
  - YO23
    - [Test YO23](./python/bash/four_dim_srda/experiment7/DL_evaluate_on_es/ConvTransNetVer01/bea2_bed2_dspe360_nsls100_ogx08_ogy08_bias1_bs12_lr1e-04_using_narrow_jet.sh)
  - U-Net MaxViT
    - [Test U-Net MaxViT](./python/bash/four_dim_srda/experiment7/DL_evaluate_on_es/UNetMaxVitVer01/bea2_bed2_dspe360_nsls100_ogx08_ogy08_n3drb3_nmb6_bias0_bs12_lr1e-04_using_narrow_jet.sh)
  - HR-LETKF : LETKF using an HR fluid model
    - [Test HR-LETKF (Run after completing U-Net MaxViT testing)](./python/bash/four_dim_srda/experiment7/letkf_simulation/perform_letkf_hr_using_uhr/es/use_narrow_jet/na3e-03_letkf_cfg_ogx08_ogy08_ne100_ch16e-04_cr6e+00_if12e-01_lr57e-01_bs6.sh)

- Simulations for obtaining forecast state
  - YO23
    - [Test YO23](./python/bash/four_dim_srda/experiment7/DL_evaluate_on_es/ConvTransNetVer01/bea2_bed2_dspe360_nsls100_ogx08_ogy08_bias1_bs12_lr1e-04_using_narrow_jet_storing_only_forecast.sh)
  - U-Net MaxViT
    - [Test U-Net MaxViT](./python/bash/four_dim_srda/experiment7/DL_evaluate_on_es/UNetMaxVitVer01/bea2_bed2_dspe360_nsls100_ogx08_ogy08_n3drb3_nmb6_bias0_bs12_lr1e-04_using_narrow_jet_storing_only_forecast.sh)
  - HR-LETKF : LETKF using an HR fluid model
    - [Test HR-LETKF (Run after completing U-Net MaxViT testing)](./python/bash/four_dim_srda/experiment7/letkf_simulation/perform_letkf_hr_using_uhr/es/use_narrow_jet/na3e-03_letkf_cfg_ogx08_ogy08_ne100_ch16e-04_cr6e%2B00_if12e-01_lr57e-01_bs6_storing_only_forecast.sh)

## Evaluation

- Calculation of MAE ratio and MSSIM loss
  - [For analysis states](./python/notebooks/four_dim_srda/experiment7/calc_statistics_and_save_as_pickle_using_narrow_jet.ipynb)
  - [For forecast states](./python/notebooks/four_dim_srda/experiment7/calc_statistics_and_save_as_pickle_using_narrow_jet_only_forecast.ipynb)
- Visualization of Results
  - [MAE ratio and MSSIM loss for analysis states](./python/notebooks/four_dim_srda/experiment7/plot_maer_and_mssim_loss_using_narrow_jet.ipynb)
  - [MAE ratio and MSSIM loss for forecast states](./python/notebooks/four_dim_srda/experiment7/plot_maer_and_mssim_loss_storing_only_forecast_using_narrow_jet.ipynb)
  - [Potential vorticity snapshots](./python/notebooks/four_dim_srda/experiment7/plot_result_snapshots_using_narrow_jet.ipynb)