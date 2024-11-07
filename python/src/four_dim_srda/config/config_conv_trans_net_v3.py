import dataclasses

from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.data.dataloader import DataloaderConfig
from src.four_dim_srda.data.dataset_making_obs import DatasetMakingObsConfig
from src.four_dim_srda.models.neural_nets.conv_trans_net_v3 import (
    ConvTransNetVer03Config,
)


@dataclasses.dataclass()
class ConvTransNetVer03ExpConfig(BaseExperimentConfig):
    model_config: ConvTransNetVer03Config
    dataset_config: DatasetMakingObsConfig
    dataloader_config: DataloaderConfig
