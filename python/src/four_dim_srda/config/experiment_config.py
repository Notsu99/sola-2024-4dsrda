import dataclasses

from src.four_dim_srda.config.base_config import YamlConfig
from src.four_dim_srda.data.base_config import BaseDatasetConfig
from src.four_dim_srda.data.dataloader import DataloaderConfig
from src.four_dim_srda.models.neural_nets.base_config import BaseModelConfig
from src.qg_model.utils.config import JetConfig

# Neural Network


@dataclasses.dataclass
class TrainConfig(YamlConfig):
    early_stopping_patience: int
    num_epochs: int
    lr: float
    seed: int
    use_zero: bool


@dataclasses.dataclass
class LossConfig(YamlConfig):
    name: str


@dataclasses.dataclass
class BaseExperimentConfig(YamlConfig):
    model_config: BaseModelConfig
    dataset_config: BaseDatasetConfig
    dataloader_config: DataloaderConfig
    train_config: TrainConfig
    loss_config: LossConfig


# CFD


@dataclasses.dataclass
class CFDTimeConfig(YamlConfig):
    start_time: int
    end_time: int
    lr_dt: float
    hr_dt: float
    uhr_dt: float
    output_lr_dt: float
    output_hr_dt: float
    output_uhr_dt: float


@dataclasses.dataclass
class SeedConfig(YamlConfig):
    seed_start: int
    seed_end: int
    uhr_seed_start: int
    uhr_seed_end: int


@dataclasses.dataclass
class DAConfig(YamlConfig):
    assimilation_dt: float
    assimilation_interval: int
    forecast_span: int
    segment_length: int


@dataclasses.dataclass
class CFDConfig(YamlConfig):
    jet_profile: str
    hr_base_config: JetConfig
    lr_base_config: JetConfig
    uhr_base_config: JetConfig
    time_config: CFDTimeConfig
    seed_config: SeedConfig
    da_config: DAConfig
