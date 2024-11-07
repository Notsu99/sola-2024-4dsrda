import pathlib

from src.four_dim_srda.config.config_conv_trans_net_v1 import ConvTransNetVer01ExpConfig
from src.four_dim_srda.config.config_conv_trans_net_v2 import ConvTransNetVer02ExpConfig
from src.four_dim_srda.config.config_conv_trans_net_v3 import ConvTransNetVer03ExpConfig
from src.four_dim_srda.config.config_u_net_maxvit_v1 import UNetMaxVitVer01ExpConfig
from src.four_dim_srda.config.config_u_net_v1 import UNetVer01ExpConfig
from src.four_dim_srda.config.config_u_net_v2 import UNetVer02ExpConfig
from src.four_dim_srda.config.config_u_net_vit_v1 import UNetVitVer01ExpConfig
from src.four_dim_srda.config.config_u_net_vit_v2 import UNetVitVer02ExpConfig
from src.four_dim_srda.config.config_u_net_vit_v3 import UNetVitVer03ExpConfig
from src.four_dim_srda.config.config_u_net_vit_v4 import UNetVitVer04ExpConfig
from src.four_dim_srda.config.experiment_config import BaseExperimentConfig


def load_config(model_name: str, config_path: str) -> BaseExperimentConfig:
    if model_name == "ConvTransNetVer01":
        return ConvTransNetVer01ExpConfig.load(pathlib.Path(config_path))
    elif model_name == "ConvTransNetVer02":
        return ConvTransNetVer02ExpConfig.load(pathlib.Path(config_path))
    elif model_name == "ConvTransNetVer03":
        return ConvTransNetVer03ExpConfig.load(pathlib.Path(config_path))
    elif model_name == "UNetVer01":
        return UNetVer01ExpConfig.load(pathlib.Path(config_path))
    elif model_name == "UNetVer02":
        return UNetVer02ExpConfig.load(pathlib.Path(config_path))
    elif model_name == "UNetVitVer01":
        return UNetVitVer01ExpConfig.load(pathlib.Path(config_path))
    elif model_name == "UNetVitVer02":
        return UNetVitVer02ExpConfig.load(pathlib.Path(config_path))
    elif model_name == "UNetVitVer03":
        return UNetVitVer03ExpConfig.load(pathlib.Path(config_path))
    elif model_name == "UNetVitVer04":
        return UNetVitVer04ExpConfig.load(pathlib.Path(config_path))
    elif model_name == "UNetMaxVitVer01":
        return UNetMaxVitVer01ExpConfig.load(pathlib.Path(config_path))
    else:
        raise Exception()
