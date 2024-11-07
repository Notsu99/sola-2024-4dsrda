import dataclasses

from src.four_dim_srda.config.base_config import YamlConfig


@dataclasses.dataclass()
class BaseDatasetConfig(YamlConfig):
    data_dirs: list[str]
    use_ground_truth_clipping: bool
    lr_and_obs_time_interval: int
    max_start_time_index: int

