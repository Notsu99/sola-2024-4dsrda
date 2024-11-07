import dataclasses

from src.four_dim_srda.config.base_config import YamlConfig


@dataclasses.dataclass()
class BaseModelConfig(YamlConfig):
    model_name: str
