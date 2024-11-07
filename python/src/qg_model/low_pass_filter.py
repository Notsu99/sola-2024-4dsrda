from logging import DEBUG, getLogger

import numpy as np
import torch
import torch.nn.functional as F

from .utils.fft import FftPeriodicChannel

logger = getLogger()


class LowPassFilter:
    def __init__(
        self,
        *,
        nx_lr: int,
        ny_lr: int,
        nz_lr: int,
        nx_hr: int,
        ny_hr: int,
        nz_hr: int,
        dtype: torch.dtype = torch.complex128,
        device: str,
    ):
        assert nx_lr < nx_hr
        assert ny_lr < ny_hr
        assert nz_lr < nz_hr

        self.dtype = dtype
        self.device = device
        self.norm = "ortho"  # the other case is not tested yet.
        assert self.norm == "ortho", "Only ortho is supported."

        # lr
        self.nx_lr = nx_lr
        self.half_nx_lr = nx_lr // 2 + 1

        self.ny_lr = ny_lr
        self.reflected_ny_lr = (ny_lr - 1) * 2

        self.nz_lr = nz_lr

        self.fft_lr = FftPeriodicChannel(
            nx=self.nx_lr,
            ny=self.ny_lr,
            device=self.device,
            dtype=self.dtype,
        )

        # hr
        self.nx_hr = nx_hr
        self.half_nx_hr = nx_hr // 2 + 1

        self.ny_hr = ny_hr
        self.reflected_ny_hr = (ny_hr - 1) * 2

        self.nz_hr = nz_hr

        self.fft_hr = FftPeriodicChannel(
            nx=self.nx_hr,
            ny=self.ny_hr,
            device=self.device,
            dtype=self.dtype,
        )

        #
        self.rescale_factor = np.sqrt(
            (self.nx_lr * self.reflected_ny_lr) / (self.nx_hr * self.reflected_ny_hr)
        )

    def apply(self, hr_grid_data: torch.Tensor) -> torch.Tensor:
        lr_spec_data_in_xy = self._apply_filter_in_xy(hr_grid_data)
        lr_spec_data = self._apply_filter_in_z(lr_spec_data_in_xy)

        return self.fft_lr.apply_ifft2(lr_spec_data)

    # The following are private methods.
    def _apply_filter_in_xy(self, hr_grid_data: torch.Tensor) -> torch.Tensor:
        assert hr_grid_data.ndim >= 3  # the last three dims = z, y, x
        assert hr_grid_data.shape[-3] == self.nz_hr
        assert hr_grid_data.shape[-2] == self.ny_hr
        assert hr_grid_data.shape[-1] == self.nx_hr

        hr_spec_data = self.fft_hr.apply_fft2(hr_grid_data.to(self.device), is_odd=True)

        return self._truncate(hr_spec_data)

    def _truncate(self, hr_spec_data: torch.Tensor) -> torch.Tensor:
        # keep nz_hr(= hr_spec_data.shape[-3]) and downsample it in _apply_filter_in_z
        size = hr_spec_data.shape[:-2] + (self.reflected_ny_lr, self.half_nx_lr)
        lr_spec_data_in_xy = torch.zeros(
            size=size,
            dtype=self.dtype,
            device=self.device,
        )
        for j in range(self.reflected_ny_hr):
            ky = FftPeriodicChannel.get_wavenumber(j, self.reflected_ny_hr)
            for i in range(self.half_nx_hr):
                kx = FftPeriodicChannel.get_wavenumber(i, self.nx_hr)
                if abs(kx) >= self.nx_lr / 2 or abs(ky) >= self.reflected_ny_lr / 2:
                    continue
                _i = self._get_array_index(kx, self.nx_lr)
                _j = self._get_array_index(ky, self.reflected_ny_lr)
                lr_spec_data_in_xy[..., _j, _i] = hr_spec_data[..., j, i]

                logger.debug(f"kx = {kx}, i = {i}, _i = {_i}")
                logger.debug(f"ky = {ky}, j = {j}, _j = {_j}")
                logger.debug("-------------------------------")

        return lr_spec_data_in_xy

    def _apply_filter_in_z(self, spec_data):
        # Downsample spec_data along z
        B, Z, Y, X = spec_data.shape
        kernel_size = int(1 // self.rescale_factor)

        real_part = spec_data.real
        imaginary_part = spec_data.imag

        real_part = real_part.permute(0, 2, 3, 1).view(B, -1, Z)
        imaginary_part = imaginary_part.permute(0, 2, 3, 1).view(B, -1, Z)

        interpolated_real = F.avg_pool1d(real_part, kernel_size=kernel_size).view(B, Y, X, Z // kernel_size)
        interpolated_imaginary = F.avg_pool1d(imaginary_part, kernel_size=kernel_size).view(B, Y, X, Z // kernel_size)

        interpolated_real = interpolated_real.permute(0, 3, 1, 2)
        interpolated_imaginary = interpolated_imaginary.permute(0, 3, 1, 2)

        lr_spec_data = interpolated_real + 1j * interpolated_imaginary

        return lr_spec_data * self.rescale_factor

    def _get_array_index(self, wavenumber: int, total_num: int) -> int:
        if wavenumber >= 0:
            return wavenumber
        return wavenumber + total_num
