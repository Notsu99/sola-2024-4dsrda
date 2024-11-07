# Reference: GitHub, jinh0park/pytorch-ssim-3D
# https://github.com/jinh0park/pytorch-ssim-3D/blob/ada88564a754cd857730d649c511384dd41f9b4e/pytorch_ssim/__init__.py
# https://pytorch.org/ignite/_modules/ignite/metrics/ssim.html#SSIM

from logging import getLogger
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable

logger = getLogger()


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2.0 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def _uniform(window_size: int) -> torch.Tensor:
    uniform = torch.ones(window_size)
    return uniform / uniform.sum()


def _create_window_3D(
    window_3d_size: tuple[int, int, int],
    channel: int,
    sigma_3d_size: tuple[float, float, float],
    use_gaussian: bool,
) -> torch.Tensor:
    #

    if use_gaussian:
        _win_z = _gaussian(window_3d_size[0], sigma_3d_size[0])
        _win_y = _gaussian(window_3d_size[1], sigma_3d_size[1])
        _win_x = _gaussian(window_3d_size[2], sigma_3d_size[2])
    else:
        _win_z = _uniform(window_3d_size[0])
        _win_y = _uniform(window_3d_size[1])
        _win_x = _uniform(window_3d_size[2])

    win = _win_y[:, None].mm(_win_x[None, :])

    win = _win_z[:, None].mm(win.reshape(1, -1)).reshape(window_3d_size).float()

    # add batch and channel dims, then broad cast to align the channel dim
    win = torch.broadcast_to(win, (channel, 1) + window_3d_size)

    return Variable(win.contiguous())


def _ssim_3D(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_3d_size: tuple[int, int, int],
    channel: int,
    value_magnitude: float,
) -> torch.Tensor:
    #
    assert img1.shape == img2.shape

    padding = (window_3d_size[0] // 2, window_3d_size[1] // 2, window_3d_size[2] // 2)

    mu1 = F.conv3d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = (value_magnitude * 0.01) ** 2
    C2 = (value_magnitude * 0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map


class MSSIMLoss(torch.nn.Module):
    def __init__(
        self,
        window_3d_size: tuple[int, int, int] = (5, 11, 11),  # z, y, x
        sigma_3d: tuple[float, float, float] = (0.7, 1.5, 1.5),  # z, y, x
        value_magnitude: float = 1.0,
        use_gaussian: bool = True,
    ):
        """Calculate Mean Structual Similarity Index Measure Loss. This loss takes a value equal to zero or larger than zero. A smaller value indicates that input two data have more similar patterns.

        Args:
            window_3d_size (tuple[int, int, int], optional): 3D window size. The order is in z, y, and x. Defaults to (5, 11, 11).
            sigma_3d (tuple[float, float, float], optional): 3D sigma for Gaussian window. The order is in z, y, and x. Defaults to (0.7, 1.5, 1.5).
            value_magnitude (float, optional): Data value range. If data takes values between min and max, you should set max - min. Defaults to 1.0.
            use_gaussian (bool, optional): use a Gaussian window or a uniform window. If it is True (False), a Gaussian (a uniform) window is used. Defaults to True.
        """
        super().__init__()

        self.window_3d_size = window_3d_size
        self.sigma_3d = sigma_3d
        self.value_magnitude = value_magnitude
        self.use_gaussian = use_gaussian

        # this value is automatically adjusted, depending on input.
        # so 1 is substituted, as a default value
        self.channel = 1

        self._set_window()

    def _set_window(self, reference_image: torch.Tensor = None):
        self.window = _create_window_3D(
            window_3d_size=self.window_3d_size,
            channel=self.channel,
            sigma_3d_size=self.sigma_3d,
            use_gaussian=self.use_gaussian,
        )

        if reference_image is None:
            return

        if reference_image.is_cuda:
            self.window = self.window.cuda(reference_image.get_device())
        self.window = self.window.type_as(reference_image)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # dims: batch, time, z, y, x dims
        # Here, time dim is regarded as channel

        (_, channel, _, _, _) = img1.size()

        if channel != self.channel or self.window.data.type() != img1.data.type():
            self.channel = channel
            self._set_window(reference_image=img1)

        ssim = _ssim_3D(
            img1=img1,
            img2=img2,
            window=self.window,
            window_3d_size=self.window_3d_size,
            channel=self.channel,
            value_magnitude=self.value_magnitude,
        )

        ssim = torch.mean(ssim, dim=(0, 2, 3, 4))
        # mean over batch, z, y, x.
        # only the time dim remains.

        return 1.0 - ssim


class MSSIM(MSSIMLoss):
    #
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # dims: batch, time, z, y, x dims
        # Here, time dim is regarded as channel

        (_, channel, _, _, _) = img1.size()

        if channel != self.channel or self.window.data.type() != img1.data.type():
            self.channel = channel
            self._set_window(reference_image=img1)

        ssim = _ssim_3D(
            img1=img1,
            img2=img2,
            window=self.window,
            window_3d_size=self.window_3d_size,
            channel=self.channel,
            value_magnitude=self.value_magnitude,
        )

        ssim = torch.mean(ssim, dim=(0, 3, 4))
        # mean over batch, y, x.
        # the time and z dim remain.

        return ssim


# The following methods are for debugging.


def _create_window_3D_for_debug(window_size, channel, sigma, use_gaussian=True):
    if use_gaussian:
        _win = _gaussian(window_size, sigma)
    else:
        _win = _uniform(window_size)

    _1D_window = _win.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = (
        _1D_window.mm(_2D_window.reshape(1, -1))
        .reshape(window_size, window_size, window_size)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    window = Variable(
        _3D_window.expand(
            channel, 1, window_size, window_size, window_size
        ).contiguous()
    )
    return window


def _ssim_3D_debug(
    img1,
    img2,
    mask,
    window,
    window_size,
    channel,
    size_average=True,
    max_val=1.0,
    eps=1e-7,
):
    assert img1.shape == img2.shape == mask.shape

    _img1 = img1 * mask
    _img2 = img2 * mask

    mu1 = F.conv3d(_img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(_img2, window, padding=window_size // 2, groups=channel)

    weights = 1.0
    mu1 = mu1 / weights
    mu2 = mu2 / weights

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv3d(_img1 * _img1, window, padding=window_size // 2, groups=channel)
        / weights
        - mu1_sq
    )
    sigma2_sq = (
        F.conv3d(_img2 * _img2, window, padding=window_size // 2, groups=channel)
        / weights
        - mu2_sq
    )
    sigma12 = (
        F.conv3d(_img1 * _img2, window, padding=window_size // 2, groups=channel)
        / weights
        - mu1_mu2
    )

    C1 = (max_val * 0.01) ** 2
    C2 = (max_val * 0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


class _MSSIMLossDebug(torch.nn.Module):
    def __init__(
        self,
        window_size=11,
        sigma=1.5,
        max_val=1.0,
        use_gaussian=True,
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 4
        self.max_val = max_val
        self.use_gaussian = use_gaussian
        self.window = _create_window_3D_for_debug(
            self.window_size, self.channel, self.sigma, self.use_gaussian
        )
        logger.info(f"Use Gaussian = {self.use_gaussian}")

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = _create_window_3D_for_debug(
                self.window_size, channel, self.sigma, self.use_gaussian
            )

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        mask = torch.ones_like(img1)

        ssim = _ssim_3D_debug(
            img1=img1,
            img2=img2,
            mask=mask,
            window=window,
            window_size=self.window_size,
            channel=self.channel,
            size_average=False,
            max_val=self.max_val,
            eps=0.0,
        )

        ssim = torch.mean(ssim, dim=(0, 2, 3, 4))

        return 1.0 - ssim
