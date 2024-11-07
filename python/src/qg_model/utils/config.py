import dataclasses
import inspect
import json
import os

import torch
import yaml


@dataclasses.dataclass()
class Config:
    nx: int  # number of grid points along x axis
    ny: int  # number of grid points along y axis
    nz: int  # number of grid points along z axis
    f_0: float  # Coriolis parameter
    beta: float  # derivative of f_0 with respect to y
    Lx: float  # domain width along x axis
    Ly: float  # domain width along y axis
    Lz: float  # domain width along z axis
    reduced_gravity: float  # magnitude of reduced gravity accelaration
    diffusion_coefficient: float  # hyper diffusion coefficient
    diffusion_exponent: int  # hyper diffusion exponent
    device: str  # cpu, cuda, cuda:0, cuda:1 etc.
    precision: str  # single or double

    def __post_init__(self):
        dic = dataclasses.asdict(self)

        for name, expected_type in self.__annotations__.items():
            assert isinstance(dic[name], expected_type)

            if isinstance(dic[name], float):
                if name in [
                    "f_0",
                    "Lx",
                    "Ly",
                    "Lz",
                    "reduced_gravity",
                    "diffusion_coefficient",
                ]:
                    assert dic[name] > 0.0
                if name in ["beta"]:
                    assert dic[name] >= 0.0

        assert self.nx >= 2 and self.nx % 2 == 0
        assert self.ny >= 3 and self.ny % 2 == 1
        assert self.nz >= 2
        assert self.diffusion_exponent > 0

        assert self.precision in ["single", "double"]
        assert self.device == "cpu" or self.device.startswith("cuda")

        assert self.Lx == (2 * torch.pi), "Not implemented yet."
        assert self.Ly == (1 * torch.pi), "Not implemented yet."

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(dataclasses.asdict(self), indent=indent)

    def save(self, config_path: str):
        # Ref: https://qiita.com/kzmssk/items/483f25f47e0ed10aa948
        #

        def convert_dict(data):
            for key, val in data.items():
                if isinstance(val, dict):
                    data[key] = convert_dict(val)
            return data

        with open(config_path, "w") as f:
            yaml.safe_dump(convert_dict(dataclasses.asdict(self)), f)

    @classmethod
    def load(cls, config_path: str):
        # Ref: https://qiita.com/kzmssk/items/483f25f47e0ed10aa948
        #
        assert os.path.exists(config_path), f"YAML config {config_path} does not exist"

        def convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if inspect.isclass(child_class) and issubclass(child_class, Config):
                    data[key] = child_class(**convert_from_dict(child_class, val))
            return data

        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            # recursively convert config item to Config
            config_data = convert_from_dict(cls, config_data)
            return cls(**config_data)


@dataclasses.dataclass()
class JetConfig(Config):
    n_batch: int  # batch size
    jet_width: float
    jet_max_velocity: float
    noise_amplitude: float  # noise amplitude added to jet

    def __post_init__(self):
        assert self.n_batch > 0
        assert self.jet_width > 0.0
        assert self.jet_max_velocity > 0.0
        assert self.noise_amplitude >= 0.0


@dataclasses.dataclass()
class EslerJetConfig(JetConfig):
    width_z: float

    def __post_init__(self):
        assert self.width_z > 0.0