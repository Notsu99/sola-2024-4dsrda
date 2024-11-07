import dataclasses

from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.data.dataloader import DataloaderConfig
from src.four_dim_srda.data.dataset_making_obs import DatasetMakingObsConfig
from src.four_dim_srda.models.neural_nets.u_net_v2 import (
    UNetVer02Config,
)


@dataclasses.dataclass()
class UNetVer02ExpConfig(BaseExperimentConfig):
    model_config: UNetVer02Config
    dataset_config: DatasetMakingObsConfig
    dataloader_config: DataloaderConfig