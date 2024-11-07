import dataclasses

from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.data.dataloader import DataloaderConfig
from src.four_dim_srda.data.dataset_making_obs import DatasetMakingObsConfig
from src.four_dim_srda.models.neural_nets.u_net_maxvit_v1 import UNetMaxVitVer01Config


@dataclasses.dataclass()
class UNetMaxVitVer01ExpConfig(BaseExperimentConfig):
    model_config: UNetMaxVitVer01Config
    dataset_config: DatasetMakingObsConfig
    dataloader_config: DataloaderConfig
