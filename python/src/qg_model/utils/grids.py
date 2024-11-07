from collections import namedtuple
from logging import getLogger

import torch

logger = getLogger()

GridXYZ = namedtuple("GridXYZ", ["x", "y", "z"])


def make_grids_for_periodic_channel(
    *, nx: int, ny: int, nz: int, Lx: float, Ly: float, Lz: float, dtype: torch.dtype
) -> GridXYZ:
    #
    for x in [nx, ny, nz]:
        assert isinstance(x, int) and x >= 2
    for x in [Lx, Ly, Lz]:
        assert isinstance(x, float) and x > 0

    # x range is [0, Lx), which does not include x = Lx
    # y range is [0, Ly], which includes y = Ly

    xs = torch.linspace(0, Lx, nx + 1, dtype=dtype)[:-1]
    ys = torch.linspace(0, Ly, ny, dtype=dtype)

    # Each z point locates in the middle of each layer
    # Layer width is `dz`
    dz = Lz / nz
    zs = torch.arange(nz, dtype=dtype) * dz + dz / 2

    zs, ys, xs = torch.meshgrid(zs, ys, xs, indexing="ij")

    return GridXYZ(xs, ys, zs)
