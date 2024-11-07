import dataclasses

from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.data.dataloader import DataloaderConfig
from src.four_dim_srda.data.dataset_making_obs import DatasetMakingObsUsingOnlyCurrentTimeTargetConfig
from src.four_dim_srda.models.neural_nets.u_net_vit_v1 import (
    UNetVitVer01Config,
)


@dataclasses.dataclass()
class UNetVitVer01ExpConfig(BaseExperimentConfig):
    model_config: UNetVitVer01Config
    dataset_config: DatasetMakingObsUsingOnlyCurrentTimeTargetConfig
    dataloader_config: DataloaderConfig