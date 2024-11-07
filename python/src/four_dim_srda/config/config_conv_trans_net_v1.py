import dataclasses

from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.data.dataloader import DataloaderConfig
from src.four_dim_srda.data.dataset_making_obs import DatasetMakingObsConfig
from src.four_dim_srda.models.neural_nets.conv_trans_net_v1 import (
    ConvTransNetVer01Config,
)


@dataclasses.dataclass()
class ConvTransNetVer01ExpConfig(BaseExperimentConfig):
    model_config: ConvTransNetVer01Config
    dataset_config: DatasetMakingObsConfig
    dataloader_config: DataloaderConfig
