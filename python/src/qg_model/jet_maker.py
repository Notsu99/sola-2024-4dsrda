from logging import getLogger

import torch

from .utils.config import EslerJetConfig, JetConfig
from .utils.grids import make_grids_for_periodic_channel

logger = getLogger()


# Return the potential vorticity corresponding to the jet profile


def make_jet_pv_with_linear_profile(conf: JetConfig) -> torch.Tensor:
    #
    grids = make_grids_for_periodic_channel(
        nx=conf.nx,
        ny=conf.ny,
        nz=conf.nz,
        Lx=conf.Lx,
        Ly=conf.Ly,
        Lz=conf.Lz,
        dtype=torch.float64,
    )

    ys, zs = grids.y, grids.z

    coeff = conf.jet_max_velocity * ((zs - conf.Lz / 2.0) / (conf.Lz - conf.Lz / 2.0))
    normalizd_ys = (ys - conf.Ly / 2.0) / conf.jet_width

    q = (
        2
        * coeff
        * torch.sinh(normalizd_ys)
        / (torch.cosh(normalizd_ys) ** 3)
        / conf.jet_width
    )
    q = q - torch.mean(q, dim=(-2, -1), keepdim=True)
    q = q.to(conf.device)

    # noise depends on batch
    noise = conf.noise_amplitude * torch.randn(
        size=(conf.n_batch,) + (q.shape), dtype=q.dtype, device=q.device
    )

    q = noise + torch.broadcast_to(q, size=noise.shape)

    return q


def make_jet_pv_with_tanh_profile(conf: JetConfig) -> torch.Tensor:
    #
    grids = make_grids_for_periodic_channel(
        nx=conf.nx,
        ny=conf.ny,
        nz=conf.nz,
        Lx=conf.Lx,
        Ly=conf.Ly,
        Lz=conf.Lz,
        dtype=torch.float64,
    )

    ys, zs = grids.y, grids.z

    width_z = conf.Lz / 8
    normalized_zs = (zs - conf.Lz / 2.0) / width_z
    normalizd_ys = (ys - conf.Ly / 2.0) / conf.jet_width

    q = (
        2.0
        * conf.jet_max_velocity
        * torch.tanh(normalized_zs)
        * torch.sinh(normalizd_ys)
        / (torch.cosh(normalizd_ys) ** 3)
    )
    q = q - torch.mean(q, dim=(-2, -1), keepdim=True)
    q = q.to(conf.device)

    # noise depends on batch
    noise = conf.noise_amplitude * torch.randn(
        size=(conf.n_batch,) + (q.shape), dtype=q.dtype, device=q.device
    )

    q = noise + torch.broadcast_to(q, size=noise.shape)

    return q


# Return velocity


def make_jet_velocity_with_sech_squared_profile(
    conf: JetConfig,
    use_shift_y: bool = False,
    use_narrow_jet: bool = False,
) -> torch.Tensor:
    #
    grids = make_grids_for_periodic_channel(
        nx=conf.nx,
        ny=conf.ny,
        nz=conf.nz,
        Lx=conf.Lx,
        Ly=conf.Ly,
        Lz=conf.Lz,
        dtype=torch.float64,
    )

    ys, zs = grids.y, grids.z

    if use_shift_y:
        logger.info("Shifted jet is used")
        shifted_ys = ys - conf.Ly * 0.1
        centered_ys = torch.abs(shifted_ys - conf.Ly / 2)
    else:
        centered_ys = torch.abs(ys - conf.Ly / 2)

    scaled_zs = zs / conf.Lz

    if use_narrow_jet:
        logger.info("Narrow jet is used")
        jet_width = conf.jet_width * 0.9
    else:
        jet_width = conf.jet_width

    u = (
        conf.jet_max_velocity
        * scaled_zs
        * torch.cosh(centered_ys / jet_width) ** -2
    )

    u = u - torch.mean(u, dim=(-2, -1), keepdim=True)
    u = u.to(conf.device)

    return u


# This jet is based on Esler(2008)
# ESLER JG. The turbulent equilibration of an unstable baroclinic jet. Journal of Fluid Mechanics.
# 2008;599:241-268. https://doi.org/10.1017/S0022112008000153

def make_jet_velocity_with_sech_squared_and_sigmoid_profile(
    conf: EslerJetConfig,
    use_shift_y: bool = False,
    use_narrow_jet: bool = False,
) -> torch.Tensor:
    #
    grids = make_grids_for_periodic_channel(
        nx=conf.nx,
        ny=conf.ny,
        nz=conf.nz,
        Lx=conf.Lx,
        Ly=conf.Ly,
        Lz=conf.Lz,
        dtype=torch.float64,
    )

    ys, zs = grids.y, grids.z

    if use_shift_y:
        logger.info("Shifted jet is used")
        shifted_ys = ys - conf.Ly * 0.1
        centered_ys = torch.abs(shifted_ys - conf.Ly / 2)
    else:
        centered_ys = torch.abs(ys - conf.Ly / 2)

    scaled_zs = (zs - conf.Lz / 2.0) / conf.width_z

    if use_narrow_jet:
        logger.info("Narrow jet is used")
        jet_width = conf.jet_width * 0.9
    else:
        jet_width = conf.jet_width

    u = (
        conf.jet_max_velocity
        * torch.sigmoid(scaled_zs)
        * torch.cosh(centered_ys / jet_width) ** -2
    )
    u = u - torch.mean(u, dim=(-2, -1), keepdim=True)
    u = u.to(conf.device)

    return u
