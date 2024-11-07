from logging import getLogger

import torch
from src.four_dim_srda.utils.ssim import MSSIM, MSSIMLoss

logger = getLogger()


def calc_maer(all_gt: torch.Tensor, all_fcst: torch.Tensor) -> torch.Tensor:
    #
    # batch, time, z, y, x dims
    assert all_gt.ndim == 5
    assert all_gt.shape == all_fcst.shape

    mae = torch.mean(
        torch.abs(all_fcst - all_gt), dim=(-3, -2, -1)
    )  # mean over z, y and x
    nrms = torch.mean(torch.abs(all_gt), dim=(-3, -2, -1))

    maer = torch.mean(mae / nrms, dim=0)  # mean over batch dim

    return maer


def calc_mssim_loss(
    all_gt: torch.Tensor, all_fcst: torch.Tensor, mssim_loss: MSSIMLoss = MSSIMLoss()
) -> torch.Tensor:
    #
    # batch, time, z, y, x dims
    assert all_gt.ndim == 5
    assert all_gt.shape == all_fcst.shape

    return mssim_loss(img1=all_gt, img2=all_fcst)


def calc_mssim(
    all_gt: torch.Tensor, all_fcst: torch.Tensor, mssim: MSSIM = MSSIM()
) -> torch.Tensor:
    #
    # batch, time, z, y, x dims
    assert all_gt.ndim == 5
    assert all_gt.shape == all_fcst.shape

    return mssim(img1=all_gt, img2=all_fcst)
