import dataclasses
import inspect
import json
import pathlib

import yaml
from src.qg_model.utils.config import EslerJetConfig, JetConfig


@dataclasses.dataclass
class YamlConfig:
    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(dataclasses.asdict(self), indent=indent)

    def save(self, config_path: pathlib.Path):
        """Export config as YAML file"""
        assert (
            config_path.parent.exists()
        ), f"directory {config_path.parent} does not exist"

        def convert_dict(data):
            for key, val in data.items():
                if isinstance(val, pathlib.Path):
                    data[key] = str(val)
                if isinstance(val, dict):
                    data[key] = convert_dict(val)
            return data

        with open(config_path, "w") as f:
            yaml.dump(convert_dict(dataclasses.asdict(self)), f)

    @classmethod
    def load(cls, config_path: pathlib.Path):
        """Load config from YAML file"""
        assert config_path.exists(), f"YAML config {config_path} does not exist"

        def convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if child_class == pathlib.Path:
                    data[key] = pathlib.Path(val)
                if inspect.isclass(child_class) and issubclass(child_class, YamlConfig):
                    data[key] = child_class(**convert_from_dict(child_class, val))
                # Classes inheriting from JetConfig
                elif inspect.isclass(child_class) and issubclass(child_class, JetConfig):
                    # Support for EslerJetConfig as well
                    if 'width_z' in val:
                        data[key] = EslerJetConfig(**convert_from_dict(EslerJetConfig, val))
                    else:
                        data[key] = JetConfig(**convert_from_dict(JetConfig, val))
            return data

        with open(config_path) as f:
            config_data = yaml.full_load(f)
            # Recursively convert config item to YamlConfig
            config_data = convert_from_dict(cls, config_data)
            return cls(**config_data)
