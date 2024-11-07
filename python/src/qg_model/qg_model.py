import copy
import warnings
from collections import namedtuple
from logging import getLogger

import torch

from .utils.config import Config
from .utils.fft import FftPeriodicChannel
from .utils.grids import GridXYZ, make_grids_for_periodic_channel
from .utils.pv_inversion import (
    get_3d_inversed_laplacian,
    get_3d_laplacian,
    get_vertical_laplacian,
)
from .utils.time_integration import integrate_one_step_rk2

logger = getLogger()


HorizontalVelocity = namedtuple("HorizontalVelocity", ["u", "v"])
Energy = namedtuple("Energy", ["total", "kinetic_energy", "available_potential_energy"])


class QGModel:
    def __init__(self, conf: Config, show_input_cfg_info: bool = True):
        #
        if show_input_cfg_info:
            logger.info("Input config to QGModel is in the following:")
            logger.info(f"{conf.to_json_str()}")

        if conf.precision == "double":
            self.complex_dtype = torch.complex128
            self.real_dtype = torch.float64
        elif conf.precision == "single":
            self.complex_dtype = torch.complex64
            self.real_dtype = torch.float32
        else:
            raise ValueError(f"{conf.precision} is not supported")

        self.nx, self.ny, self.nz = conf.nx, conf.ny, conf.nz
        self.grid_data_shape = (conf.nz, conf.ny, conf.nx)
        self.device = conf.device
        self.beta = conf.beta
        self.conf = copy.deepcopy(conf)

        self.fft = FftPeriodicChannel(
            nx=self.nx, ny=self.ny, device=self.device, dtype=self.complex_dtype
        )

        self.laplacian = get_3d_laplacian(
            f_0=conf.f_0,
            Lz=conf.Lz,
            reduced_gravity=conf.reduced_gravity,
            nz=conf.nz,
            squared_magnitude_wavenumbers=self.fft.k2,
            dtype=self.complex_dtype,
        ).to(self.device)

        self.laplacian_z = get_vertical_laplacian(
            f_0=conf.f_0,
            Lz=conf.Lz,
            reduced_gravity=conf.reduced_gravity,
            nz=conf.nz,
            dtype=self.real_dtype,
            device=self.device,
        )

        self.inv_laplacian = get_3d_inversed_laplacian(
            f_0=conf.f_0,
            Lz=conf.Lz,
            reduced_gravity=conf.reduced_gravity,
            nz=conf.nz,
            squared_magnitude_wavenumbers=self.fft.k2,
            dtype=self.complex_dtype,
        ).to(self.device)

        self.vertical_thicknesses = (
            torch.ones(size=(self.nz,), dtype=self.real_dtype, device=self.device)
            * self.conf.Lz
            / self.nz
        )

        # diffusion is expressed as -\kappa (\-Delta)^c pv in physical space
        # -\Delta is k^2 + l^2 in spectral space.
        self.diffusion_operator = -conf.diffusion_coefficient * (
            self.fft.k2**conf.diffusion_exponent
        )

        # Both are initialized in the method `initialize_pv` or `initialize`
        self.t = None
        self.spec_pv = None
        self.spec_forcing = None
        self.horizontally_averaged_u = None
        self.background_pv_y = None

        # Both are set in `_do_diffusion`
        self.dt = None
        self.diffusion_matrix = None

    def reset(self):
        self.t = None
        self.spec_pv = None
        self.spec_forcing = None
        self.horizontally_averaged_u = None
        self.background_pv_y = None
        self.dt = None
        self.diffusion_matrix = None

    def initialize_pv(self, grid_pv: torch.Tensor, grid_forcing: torch.Tensor = None):
        #
        warnings.warn(
            'This method (initialize_pv) does not support horizontally averaged compnents of PV. These components are set to zero inside the method. To include such components, please use the method named "initialize."'
        )
        #
        assert grid_pv.shape[-3:] == self.grid_data_shape
        self.reset()

        q = QGModel._remove_wavenumber_zero_components(grid_pv)
        self.spec_pv = self.fft.apply_fft2(q.to(self.device), is_odd=True)

        self.spec_forcing = torch.tensor(
            [0.0], dtype=self.complex_dtype, device=self.device
        )
        self.t = 0.0

        # Currently, this method of initialize_pv does not support horizontally averaged U.
        # So, this tensor is set to 0.
        self.horizontally_averaged_u = torch.full(
            size=[self.conf.nz, 1, 1],
            fill_value=0.0,
            dtype=self.real_dtype,
            device=self.device,
        )
        # for broadcasting, y and x dims are set to 1.

        self._calc_background_pv_derivative_with_respect_to_y(is_only_beta=True)

        if grid_forcing is not None:
            assert grid_pv.shape == grid_forcing.shape
            f = QGModel._remove_wavenumber_zero_components(grid_forcing)
            self.spec_forcing = self.fft.apply_fft2(f.to(self.device), is_odd=True)

    def initialize(
        self, grid_u: torch.Tensor, grid_pv_noise: torch.Tensor = None
    ) -> None:
        #
        if grid_pv_noise is not None:
            assert grid_pv_noise.shape[-3:] == self.grid_data_shape

        assert grid_u.shape[-3:] == self.grid_data_shape
        self.reset()

        self.spec_pv = self._calc_spec_pv_from_zonal_velocity(grid_u)

        if grid_pv_noise is not None:
            q = QGModel._remove_wavenumber_zero_components(grid_pv_noise)
            self.spec_pv = self.spec_pv + self.fft.apply_fft2(
                q.to(self.device), is_odd=True
            )

        self.spec_forcing = torch.tensor(
            [0.0], dtype=self.complex_dtype, device=self.device
        )
        self.t = 0.0

        self._calc_background_pv_derivative_with_respect_to_y(is_only_beta=False)

    def get_grid_pv(self) -> torch.Tensor:
        return self.fft.apply_ifft2(self.spec_pv).detach().cpu()

    def get_total_grid_pv(self) -> torch.Tensor:
        pv = self.get_grid_pv()
        grids = self.get_grids()
        background_pv = self.background_pv_y.detach().cpu() * grids.y
        return pv + background_pv

    def get_grid_psi(self) -> torch.Tensor:
        spec_psi = self._pv_inversion(self.spec_pv)
        return self.fft.apply_ifft2(spec_psi).detach().cpu()

    def get_total_grid_psi(self) -> torch.Tensor:
        psi = self.get_grid_psi()
        grids = self.get_grids()
        background_psi = -1.0 * self.horizontally_averaged_u.detach().cpu() * grids.y
        return psi + background_psi

    def get_grid_velocity(self) -> HorizontalVelocity:
        spec_psi = self._pv_inversion(self.spec_pv)
        grid_u = self.fft.apply_ifft2(-self.fft.jky * spec_psi)
        grid_v = self.fft.apply_ifft2(self.fft.jkx * spec_psi)
        return HorizontalVelocity(grid_u.detach().cpu(), grid_v.detach().cpu())

    def get_total_grid_velocity(self) -> HorizontalVelocity:
        velocity = self.get_grid_velocity()
        u = velocity.u + self.horizontally_averaged_u.detach().cpu()
        v = velocity.v
        return HorizontalVelocity(u, v)

    def get_enstropy(self) -> torch.Tensor:
        # sum over y and x
        sum_squred_q = self.fft.sum_squared_data_over_space_by_parseval(self.spec_pv)

        # sum over z
        enstrophy = torch.sum(0.5 * sum_squred_q * self.vertical_thicknesses, dim=-1)

        return enstrophy

    def get_total_enstropy(self) -> torch.Tensor:
        q2 = (self.get_total_grid_pv()) ** 2
        q2 = q2.to(self.device)

        # integral over y and x dims
        # Boundary values on y axis (y = Ly) is excluded, i.e., q2[..., :-1, :]
        sum_squred_q = (
            torch.mean(q2[..., :-1, :], dim=(-2, -1)) * self.conf.Ly * self.conf.Lx
        )

        # integral over z dim
        enstrophy = torch.sum(0.5 * sum_squred_q * self.vertical_thicknesses, dim=-1)

        return enstrophy

    def get_energy(self) -> Energy:
        spec_psi = self._pv_inversion(self.spec_pv)
        spec_psi_x = self.fft.jkx * spec_psi
        spec_psi_y = self.fft.jky * spec_psi

        x = self.fft.sum_squared_data_over_space_by_parseval(spec_psi_x)
        y = self.fft.sum_squared_data_over_space_by_parseval(spec_psi_y)
        kinetic_ene = torch.sum(0.5 * (x + y) * self.vertical_thicknesses, dim=-1)
        # `-1` specifies vertical dim

        eps2 = (self.conf.f_0**2) / (
            self.conf.reduced_gravity * self.vertical_thicknesses
        )

        # take difference over vertical layers
        spec_diff_psi = spec_psi[..., 1:, :, :] - spec_psi[..., :-1, :, :]
        z = self.fft.sum_squared_data_over_space_by_parseval(spec_diff_psi)
        potential_ene = torch.sum(
            0.5 * z * (eps2[:-1] * self.vertical_thicknesses[:-1]), dim=-1
        )

        return Energy(
            total=kinetic_ene + potential_ene,
            kinetic_energy=kinetic_ene,
            available_potential_energy=potential_ene,
        )

    def get_total_energy(self) -> Energy:
        u, v = self.get_total_grid_velocity()
        uv2 = (u**2 + v**2).to(self.device)

        # integral over y and x, then integral over z.
        # Boundary values on y axis (y = Ly) is excluded, i.e., uv2[..., :-1, :]
        ke = torch.mean(uv2[..., :-1, :], dim=(-2, -1)) * self.conf.Ly * self.conf.Lx
        kinetic_ene = torch.sum(0.5 * ke * self.vertical_thicknesses, dim=-1)

        psi = self.get_total_grid_psi().to(self.device)
        diff_psi2 = (psi[..., 1:, :, :] - psi[..., :-1, :, :]) ** 2  # dif along z
        pe = (
            torch.mean(diff_psi2[..., :-1, :], dim=(-2, -1))
            * self.conf.Ly
            * self.conf.Lx
        )  # Boundary values on y axis (y = Ly) is excluded, diff_psi2[..., :-1, :]

        eps2 = (self.conf.f_0**2) / (
            self.conf.reduced_gravity * self.vertical_thicknesses
        )

        potential_ene = torch.sum(
            0.5 * pe * (eps2[:-1] * self.vertical_thicknesses[:-1]), dim=-1
        )

        return Energy(
            total=kinetic_ene + potential_ene,
            kinetic_energy=kinetic_ene,
            available_potential_energy=potential_ene,
        )

    def get_grids(self) -> GridXYZ:
        return make_grids_for_periodic_channel(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            Lx=self.conf.Lx,
            Ly=self.conf.Ly,
            Lz=self.conf.Lz,
            dtype=self.real_dtype,
        )

    def integrate_n_steps(self, dt_per_step: float, n_steps: int):
        assert isinstance(dt_per_step, float) and dt_per_step > 0
        assert isinstance(n_steps, int) and n_steps > 0
        for _ in range(n_steps):
            self._step_forward(dt=dt_per_step)

    # The following are private methods.

    @staticmethod
    def _remove_wavenumber_zero_components(data):
        mean = torch.mean(data, dim=(-2, -1), keepdim=True)
        return data - mean

    def _step_forward(self, dt: float):
        pv = self._do_advection(dt, spec_pv=self.spec_pv)
        self.spec_pv = self._do_diffusion(dt, spec_pv=pv)
        assert torch.all(~torch.isnan(self.spec_pv))
        self.t += dt

    def _do_diffusion(self, dt: float, spec_pv: torch.Tensor) -> torch.Tensor:
        #
        if self.dt is None or self.dt != dt:
            self.dt = dt
            self.diffusion_matrix = torch.exp(self.diffusion_operator * dt)

        return self.diffusion_matrix * spec_pv

    def _do_advection(self, dt: float, spec_pv: torch.Tensor) -> torch.Tensor:
        #
        return integrate_one_step_rk2(
            dt=dt, x=spec_pv, dxdt=self._calc_advection_tendency
        )

    def _calc_advection_tendency(self, spec_pv: torch.Tensor) -> torch.Tensor:
        # Obtain stream function.
        spec_psi = self._pv_inversion(spec_pv)

        # adv = (u + U) * \partial_x pv + v \partial_y pv + v Q_y
        # U: horizontally averaged zonal velocity
        # Q_y: y-derivative of background PV field, including "beta" term
        adv = self._jacobian(spec_psi=spec_psi, spec_pv=spec_pv)

        # minus is necessary because it is on the right hand side.
        # \partial_t pv = - adv + forcing

        return -self.fft.apply_fft2(adv, is_odd=True) + self.spec_forcing

    def _pv_inversion(self, spec_pv: torch.Tensor) -> torch.Tensor:
        #
        return torch.einsum("lkji,...lji->...kji", self.inv_laplacian, spec_pv)

    def _jacobian(self, spec_psi: torch.Tensor, spec_pv: torch.Tensor) -> torch.Tensor:
        #
        # jacobian = \partial_x psi \partial_y pv - \partial_y psi \partial_x pv
        # beta effect is also added, which is the advection of the planetary PV.
        # effect of background flow is also added, which is attributed to horizontally averaged velocity.
        #
        # here, u = -\partial_y \psi and v = +\partial_x \psi

        u = self.fft.apply_ifft2(self.fft.jky * spec_psi) * -1.0
        v = self.fft.apply_ifft2(self.fft.jkx * spec_psi)

        pv_x = self.fft.apply_ifft2(self.fft.jkx * spec_pv)
        pv_y = self.fft.apply_ifft2(self.fft.jky * spec_pv)

        _u = u + self.horizontally_averaged_u

        return _u * pv_x + v * pv_y + self.background_pv_y * v

    def _calc_spec_pv_from_zonal_velocity(self, grid_u: torch.Tensor) -> torch.Tensor:
        #
        assert grid_u.shape[-3:] == self.grid_data_shape

        # average over y and x (-2 and -1) dims
        ave_u = torch.mean(grid_u, dim=(-2, -1), keepdim=True)
        u = grid_u - ave_u

        self.horizontally_averaged_u = ave_u.to(self.real_dtype).to(self.device)

        spec_u = self.fft.apply_fft2(u.to(self.device), is_odd=False)

        spec_psi = -spec_u / self.fft.jky
        spec_psi = torch.where(
            torch.isnan(spec_psi), torch.full_like(spec_psi, 0.0), spec_psi
        )
        # fill NaNs caused by zero division from `spec_psi = -spec_u / self.fft.jky`

        return torch.einsum("lkji,...lji->...kji", self.laplacian, spec_psi)

    def _calc_background_pv_derivative_with_respect_to_y(self, is_only_beta: bool):
        if is_only_beta:
            self.background_pv_y = torch.full(
                size=[self.conf.nz, 1, 1],
                fill_value=self.beta,
                dtype=self.real_dtype,
                device=self.device,
            )
            return

        u_zz = torch.einsum(
            "lk,...lji->...kji", self.laplacian_z, self.horizontally_averaged_u
        )

        self.background_pv_y = self.beta - u_zz
