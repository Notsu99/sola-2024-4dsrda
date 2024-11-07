from logging import getLogger

from src.four_dim_srda.config.experiment_config import BaseExperimentConfig
from src.four_dim_srda.models.neural_nets.conv_trans_net_v1 import ConvTransNetVer01
from src.four_dim_srda.models.neural_nets.conv_trans_net_v2 import ConvTransNetVer02
from src.four_dim_srda.models.neural_nets.conv_trans_net_v3 import ConvTransNetVer03
from src.four_dim_srda.models.neural_nets.u_net_maxvit_v1 import UNetMaxVitVer01
from src.four_dim_srda.models.neural_nets.u_net_v1 import UNetVer01
from src.four_dim_srda.models.neural_nets.u_net_v2 import UNetVer02
from src.four_dim_srda.models.neural_nets.u_net_vit_v1 import UNetVitVer01
from src.four_dim_srda.models.neural_nets.u_net_vit_v2 import UNetVitVer02
from src.four_dim_srda.models.neural_nets.u_net_vit_v3 import UNetVitVer03
from src.four_dim_srda.models.neural_nets.u_net_vit_v4 import UNetVitVer04
from torch import nn

logger = getLogger()


def make_model(cfg: BaseExperimentConfig) -> nn.Module:
    if cfg.model_config.model_name == "ConvTransNetVer01":
        logger.info("ConvTransNetVer01 is created.")
        return ConvTransNetVer01(cfg.model_config)
    elif cfg.model_config.model_name == "ConvTransNetVer02":
        logger.info("ConvTransNetVer02 is created.")
        return ConvTransNetVer02(cfg.model_config)
    elif cfg.model_config.model_name == "ConvTransNetVer03":
        logger.info("ConvTransNetVer03 is created.")
        return ConvTransNetVer03(cfg.model_config)
    elif cfg.model_config.model_name == "UNetVer01":
        logger.info("UNetVer01 is created.")
        return UNetVer01(cfg.model_config)
    elif cfg.model_config.model_name == "UNetVer02":
        logger.info("UNetVer02 is created.")
        return UNetVer02(cfg.model_config)
    elif cfg.model_config.model_name == "UNetVitVer01":
        logger.info("UNetVitVer01 is created.")
        return UNetVitVer01(cfg.model_config)
    elif cfg.model_config.model_name == "UNetVitVer02":
        logger.info("UNetVitVer02 is created.")
        return UNetVitVer02(cfg.model_config)
    elif cfg.model_config.model_name == "UNetVitVer03":
        logger.info("UNetVitVer03 is created.")
        return UNetVitVer03(cfg.model_config)
    elif cfg.model_config.model_name == "UNetVitVer04":
        logger.info("UNetVitVer04 is created.")
        return UNetVitVer04(cfg.model_config)
    elif cfg.model_config.model_name == "UNetMaxVitVer01":
        logger.info("UNetMaxVitVer01 is created.")
        return UNetMaxVitVer01(cfg.model_config)
    else:
        raise NotImplementedError(f"{cfg.model_config.model_name} is not supported.")
