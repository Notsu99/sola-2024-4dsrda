import typing
import warnings
from logging import getLogger

import numpy as np
import torch

logger = getLogger()


def get_vertical_laplacian(
    *,
    f_0: float,
    Lz: float,
    reduced_gravity: float,
    nz: int,
    dtype: torch.dtype,
    device: str = "cpu",
) -> torch.Tensor:
    #
    for x in [f_0, Lz, reduced_gravity]:
        assert isinstance(x, float) and x > 0
    assert isinstance(nz, int) and nz >= 2

    # See Vallis (2017), Eqs. 5.87 and 5.88
    # https://doi.org/10.1017/9781107588417

    each_layer_depth = Lz / nz
    fac = f_0**2 / (reduced_gravity * each_layer_depth)

    laplacian = torch.zeros((nz, nz), dtype=dtype, device=device)

    logger.debug(f"\nFactor for vertical second-order derivatives = {fac}.")
    logger.debug("The input params for get_vertical_laplacian are as follows:")
    logger.debug(f"nz = {nz},")
    logger.debug(f"f_0 = {f_0},")
    logger.debug(f"Lz = {Lz},")
    logger.debug(f"each layer depth = {each_layer_depth}")
    logger.debug(f"reduced_gravity = {reduced_gravity}")
    logger.debug(f"Effective dz = {1/np.sqrt(fac)}")
    logger.debug(f"Actual dz = {Lz/nz}\n")

    for i in range(nz):
        if 0 < i < nz - 1:
            laplacian[i - 1, i] = 1
            laplacian[i, i] = -2
            laplacian[i + 1, i] = 1
        elif i == 0:
            laplacian[i + 1, i] = 1
            laplacian[i, i] = -1
        elif i == nz - 1:
            laplacian[i - 1, i] = 1
            laplacian[i, i] = -1

    return laplacian * fac


def get_3d_laplacian(
    *,
    f_0: float,
    Lz: float,
    reduced_gravity: float,
    nz: int,
    squared_magnitude_wavenumbers: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    #
    for x in [f_0, Lz, reduced_gravity]:
        assert isinstance(x, float) and x > 0
    assert isinstance(nz, int) and nz >= 2

    assert squared_magnitude_wavenumbers.ndim == 2  # y and x dims

    shape = (nz, nz) + squared_magnitude_wavenumbers.shape
    device = squared_magnitude_wavenumbers.device
    logger.debug(f"laplacian_xyz shape = {shape}")

    laplacian_z = get_vertical_laplacian(
        f_0=f_0,
        Lz=Lz,
        reduced_gravity=reduced_gravity,
        nz=nz,
        dtype=torch.float64,
        device=device,
    )

    # add y and x dims
    laplacian_z = torch.broadcast_to(laplacian_z[..., None, None], shape)

    # for horizontal wavenumber vectors (add y and x dims)
    eyes = torch.eye(nz, device=device)[:, :, None, None]
    eyes = torch.broadcast_to(eyes, shape)

    # `fft.k2` is complex.128, so a warning is thrown when casting it to float64
    # This warning says "casting to float64 ignores imaginary parts."
    # but actually `fft.k2` imaginary part is zero, so casting is just changing types.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k2 = eyes * squared_magnitude_wavenumbers.to(torch.float64)

    lap_zyx = -k2 + laplacian_z

    # For wavenumber-0 components, laplacian is assumed to be zero.
    # This treatment is consistent with `get_3d_inversed_laplacian`,
    # in which lap_zyx[:, :, 0, 0] = torch.nan
    lap_zyx[:, :, 0, 0] = 0.0

    return lap_zyx.to(dtype)


def get_3d_inversed_laplacian(
    *,
    f_0: float,
    Lz: float,
    reduced_gravity: float,
    nz: int,
    squared_magnitude_wavenumbers: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    #
    lap_zyx = get_3d_laplacian(
        f_0=f_0,
        Lz=Lz,
        reduced_gravity=reduced_gravity,
        nz=nz,
        squared_magnitude_wavenumbers=squared_magnitude_wavenumbers,
        dtype=torch.float64,
    )

    # to avoid nan in matrix inverse operation
    # this is the case of zero wavenumber vector.
    lap_zyx[:, :, 0, 0] = torch.nan

    lap_zyx = lap_zyx.permute(3, 2, 1, 0)

    inv_lap = torch.linalg.inv(lap_zyx)
    inv_lap = inv_lap.permute(3, 2, 1, 0).to(dtype)

    return torch.where(torch.isnan(inv_lap), torch.full_like(inv_lap, 0.0), inv_lap)
