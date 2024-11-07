import dataclasses

from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.data.dataloader import DataloaderConfig
from src.four_dim_srda.data.dataset_making_obs import DatasetMakingObsConfig
from src.four_dim_srda.models.neural_nets.u_net_vit_v4 import (
    UNetVitVer04Config,
)


@dataclasses.dataclass()
class UNetVitVer04ExpConfig(BaseExperimentConfig):
    model_config: UNetVitVer04Config
    dataset_config: DatasetMakingObsConfig
    dataloader_config: DataloaderConfig