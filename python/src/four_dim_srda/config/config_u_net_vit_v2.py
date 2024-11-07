import dataclasses

from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.data.dataloader import DataloaderConfig
from src.four_dim_srda.data.dataset_making_obs import DatasetMakingObsConfig
from src.four_dim_srda.models.neural_nets.u_net_vit_v2 import (
    UNetVitVer02Config,
)


@dataclasses.dataclass()
class UNetVitVer02ExpConfig(BaseExperimentConfig):
    model_config: UNetVitVer02Config
    dataset_config: DatasetMakingObsConfig
    dataloader_config: DataloaderConfig