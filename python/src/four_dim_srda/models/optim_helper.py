import random
import typing
from logging import getLogger

import numpy as np
import torch
import torch.distributed as dist
from src.four_dim_srda.utils.average_meter import AverageMeter
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler

logger = getLogger()


def optimize(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    mode: typing.Literal["train", "valid", "test"],
) -> float:
    loss_meter = AverageMeter()

    if mode == "train":
        model.train()
    elif mode in ["valid", "test"]:
        model.eval()
    else:
        raise NotImplementedError(f"{mode} is not supported.")

    random.seed(epoch)
    np.random.seed(epoch)
    dataloader.dataset.set_hr_file_paths_randomly(epoch=epoch)
    print(epoch, dataloader.dataset.hr_file_paths[0])

    for lr_pv, obs, gt in dataloader:
        lr_pv, obs, gt = lr_pv.to(device), obs.to(device), gt.to(device)

        if mode == "train":
            preds = model(lr_pv, obs)
            loss = loss_fn(predicts=preds, targets=gt, masks=obs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(lr_pv, obs)
                loss = loss_fn(predicts=preds, targets=gt, masks=obs)

        loss_meter.update(loss.item(), n=lr_pv.shape[0])

    logger.info(f"{mode} error: avg loss = {loss_meter.avg:.8f}")

    return loss_meter.avg


def optimize_ddp(
    dataloader: DataLoader,
    sampler: Sampler,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    epoch: int,
    rank: int,
    world_size: int,
    mode: typing.Literal["train", "valid", "test"],
) -> float:
    mean_loss, cnt = 0.0, 0

    if mode == "train":
        model.train()
    elif mode in ["valid", "test"]:
        model.eval()
    else:
        raise NotImplementedError(f"{mode} is not supported.")

    sampler.set_epoch(epoch)
    random.seed(epoch)
    np.random.seed(epoch)

    for lr_pv, obs, gt in dataloader:
        lr_pv, obs, gt = lr_pv.to(rank), obs.to(rank), gt.to(rank)

        if mode == "train":
            preds = model(lr_pv, obs)
            loss = loss_fn(preds, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(lr_pv, obs)
                loss = loss_fn(preds, gt)

        mean_loss += loss * lr_pv.shape[0]
        cnt += lr_pv.shape[0]

    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size