import dataclasses

from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.data.dataloader import DataloaderConfig
from src.four_dim_srda.data.dataset_making_obs import DatasetMakingObsConfig
from src.four_dim_srda.models.neural_nets.u_net_vit_v3 import (
    UNetVitVer03Config,
)


@dataclasses.dataclass()
class UNetVitVer03ExpConfig(BaseExperimentConfig):
    model_config: UNetVitVer03Config
    dataset_config: DatasetMakingObsConfig
    dataloader_config: DataloaderConfig