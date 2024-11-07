import typing
from logging import getLogger

import torch

logger = getLogger()


class FftPeriodicChannel:
    def __init__(
        self,
        nx: int,
        ny: int,
        dtype: torch.dtype = torch.complex128,
        device: str = "cpu",
    ):
        assert isinstance(nx, int) and nx > 0 and nx % 2 == 0
        assert isinstance(ny, int) and ny > 0 and ny % 2 == 1

        self.nx = nx
        self.ny = ny
        self.device = device

        # Since `rfft2` only stores the half for the last dim,
        # the last dim of arrays = nx // 2.
        # `+ 1` is necessary due to the const component (wavenumber being zero)
        half_nx = nx // 2 + 1
        self.half_nx = half_nx

        # The second last dim (i.e., y) is doubled because of reflection.
        reflected_ny = (ny - 1) * 2
        self.reflected_ny = reflected_ny

        self.grid_data_shape = (ny, nx)
        self.spec_data_shape = (reflected_ny, half_nx)

        self.norm = "ortho"  # the other case is not tested yet.
        self.dtype = dtype

        # `j` means the imaginary unit
        self.jkx = torch.zeros((reflected_ny, half_nx), dtype=dtype, device=device)
        self.jky = torch.zeros((reflected_ny, half_nx), dtype=dtype, device=device)
        self.k2 = torch.zeros((reflected_ny, half_nx), dtype=dtype, device=device)

        # Filtering to remove aliasing errors (2/3-rule is applied).
        # See Canuto (1988) for details.
        # https://doi.org/10.1007/978-3-642-84108-8_3
        self.filter_weight = torch.ones(
            (reflected_ny, half_nx), dtype=dtype, device=device
        )

        for j in range(reflected_ny):
            ky = FftPeriodicChannel.get_wavenumber(j, reflected_ny)

            for i in range(half_nx):
                kx = FftPeriodicChannel.get_wavenumber(i, nx)

                self.jkx[j, i] = 1j * kx
                self.jky[j, i] = 1j * ky

                k2 = kx**2 + ky**2
                self.k2[j, i] = k2

                # 2/3-rule
                if 3 * abs(kx) > nx or 3 * abs(ky) > reflected_ny:
                    self.filter_weight[j, i] = 0.0

    @staticmethod
    def reflect_along_y(z: torch.Tensor, is_odd: bool) -> torch.Tensor:
        #
        reflected = z.flip(dims=(-2,))[..., 1:-1, :]  # drop boundary values
        if is_odd:
            reflected *= -1
        return torch.cat([z, reflected], dim=-2)

    @staticmethod
    def get_wavenumber(idx: int, total_num: int) -> int:
        if idx <= total_num // 2:
            return idx
        return idx - total_num

    def apply_fft2(self, grid_data: torch.Tensor, is_odd: bool) -> torch.Tensor:
        #
        assert grid_data.shape[-2:] == self.grid_data_shape

        reflected = FftPeriodicChannel.reflect_along_y(grid_data, is_odd)
        spec = torch.fft.rfft2(reflected, dim=(-2, -1), norm=self.norm)

        # Filtering is applied to remove aliasing errors.
        return spec * self.filter_weight

    def apply_ifft2(self, spec_data: torch.Tensor) -> torch.Tensor:
        #
        assert spec_data.shape[-2:] == self.spec_data_shape

        grid_data = torch.fft.irfft2(spec_data, dim=(-2, -1), norm=self.norm)

        # returning the half because of reflection
        return grid_data[..., : self.ny, :]

    def sum_squared_data_over_space_by_parseval(
        self, spec_data: torch.Tensor
    ) -> torch.Tensor:
        #
        # Ref: Parseval identity
        # https://en.wikipedia.org/wiki/Parseval%27s_identity
        #
        assert spec_data.shape[-2:] == self.spec_data_shape

        sq_spec = torch.abs(spec_data) ** 2

        # since only the half of the last dim data is stored,
        # `* 2.0` is necessary to get the correct sum.
        # Here, we assume `nx` is even, as in `__init__` method.
        s = (
            torch.sum(sq_spec[..., 0], dim=-1)
            + torch.sum(sq_spec[..., 1:-1], dim=(-2, -1)) * 2.0
            + torch.sum(sq_spec[..., -1], dim=-1)
        )

        # since y domain is reflected, 1/2 is necessary.
        return s / 2.0
