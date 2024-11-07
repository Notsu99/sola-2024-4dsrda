import dataclasses
from logging import getLogger

from src.four_dim_srda.config.base_config import YamlConfig
from torch import nn

logger = getLogger()


@dataclasses.dataclass
class LossConfig(YamlConfig):
    name: str


def make_loss(cfg: LossConfig) -> nn.Module:

    if cfg.name == "L1":
        logger.info("L1 loss is created.")
        return nn.L1Loss(reduction="mean")

    elif cfg.name == "MSE":
        logger.info("MSE loss is created.")
        return nn.MSELoss(reduction="mean")

    else:
        raise NotImplementedError(
            f'{cfg.name} is not supported.'
        )
