import copy
import dataclasses
import glob
import os
from logging import getLogger

from sklearn.model_selection import train_test_split
from src.four_dim_srda.config.base_config import YamlConfig
from src.four_dim_srda.data.base_config import BaseDatasetConfig
from src.four_dim_srda.data.dataset_making_obs import (
    DatasetMakingObs,
    DatasetMakingObsMinusOneOneScaling,
    DatasetMakingObsUsingOnlyCurrentTimeTarget,
)
from src.four_dim_srda.utils.random_seed_helper import get_torch_generator, seed_worker
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = getLogger()


@dataclasses.dataclass
class DataloaderConfig(YamlConfig):
    dataset_name: str
    batch_size: int
    data_dir_name: str
    num_workers: int
    train_valid_test_ratios: list[int]
    seed: int


def make_dataloaders_and_samplers(
    *,
    cfg_dataloader: DataloaderConfig,
    cfg_data: BaseDatasetConfig,
    root_dir: str,
    train_valid_test_kinds: list[str] = ["train", "valid", "test"],
    world_size: int = None,
    rank: int = None,
):
    cfd_data_dir_path = f"{root_dir}/data/four_dim_srda/{cfg_dataloader.data_dir_name}"
    logger.info(f"CFD data dir path = {cfd_data_dir_path}")

    data_dirs = sorted(
        [p for p in glob.glob(f"{cfd_data_dir_path}/seed*") if os.path.isdir(p)]
    )

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, cfg_dataloader.train_valid_test_ratios
    )
    _train = copy.deepcopy(cfg_data)
    _train.data_dirs = train_dirs
    _train.use_ground_truth_clipping = True

    _valid = copy.deepcopy(cfg_data)
    _valid.data_dirs = valid_dirs
    _valid.use_ground_truth_clipping = True

    _test = copy.deepcopy(cfg_data)
    _test.data_dirs = test_dirs
    _test.use_ground_truth_clipping = False

    dict_cfg_datas = {"train": _train, "valid": _valid, "test": _test}

    if cfg_dataloader.dataset_name == "DatasetMakingObs":
        dataset_initilizer = DatasetMakingObs
        logger.info("Dataset is DatasetMakingObs")
    elif cfg_dataloader.dataset_name == "DatasetMakingObsMinusOneOneScaling":
        dataset_initilizer = DatasetMakingObsMinusOneOneScaling
        logger.info("Dataset is DatasetMakingObsMinusOneOneScaling")
    elif cfg_dataloader.dataset_name == "DatasetMakingObsUsingOnlyCurrentTimeTarget":
        dataset_initilizer = DatasetMakingObsUsingOnlyCurrentTimeTarget
        logger.info("Dataset is DatasetMakingObsUsingOnlyCurrentTimeTarget")
    else:
        raise NotImplementedError(f"{cfg_dataloader.dataset_name} is not supported.")

    return _make_dataloaders_and_samplers(
        dataset_initilizer=dataset_initilizer,
        dict_cfg=dict_cfg_datas,
        train_valid_test_kinds=train_valid_test_kinds,
        batch_size=cfg_dataloader.batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=cfg_dataloader.num_workers,
        seed=cfg_dataloader.seed,
    )


def _make_dataloaders_and_samplers(
    *,
    dataset_initilizer,
    dict_cfg: dict[str, YamlConfig],
    train_valid_test_kinds: list[str] = ["train", "valid", "test"],
    batch_size: int,
    rank: int = None,
    world_size: int = None,
    num_workers: int = 2,
    seed: int = 42,
):
    logger.info(
        f"Batch size = {batch_size}, world_size = {world_size}, rank = {rank}\n"
    )

    dict_dataloaders, dict_samplers = {}, {}
    for kind in train_valid_test_kinds:
        dataset = dataset_initilizer(cfg=dict_cfg[kind])

        if world_size is None or rank is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
            )

            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num ={len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(seed),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )
    return dict_dataloaders, dict_samplers


def split_file_paths(
    paths: list[str], train_valid_test_ratios: list[float]
) -> tuple[list[str], list[str], list[str]]:
    assert len(train_valid_test_ratios) == 3  # train, valid, test, three ratios

    test_size = train_valid_test_ratios[-1]
    _paths, test_paths = train_test_split(paths, test_size=test_size, shuffle=False)

    valid_size = train_valid_test_ratios[1] / (
        train_valid_test_ratios[0] + train_valid_test_ratios[1]
    )
    train_paths, valid_paths = train_test_split(
        _paths, test_size=valid_size, shuffle=False
    )

    assert set(train_paths).isdisjoint(set(valid_paths))
    assert set(train_paths).isdisjoint(set(test_paths))
    assert set(valid_paths).isdisjoint(set(test_paths))

    return train_paths, valid_paths, test_paths
