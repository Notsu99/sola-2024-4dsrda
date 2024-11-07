import torch
import torch.nn.functional as F


def interpolate_2d(
    data: torch.Tensor, nx: int, ny: int, mode: str = "bicubic"
) -> torch.Tensor:
    #
    assert data.ndim == 4

    interpolated_data = F.interpolate(
        data,
        size=(ny, nx),
        mode=mode,
        align_corners=None if mode == "nearest" else False,
    )
    assert interpolated_data.shape[2:] == (ny, nx)

    return interpolated_data


def interpolate_along_z(
    data: torch.Tensor, nz: int, mode: str = "linear"
) -> torch.Tensor:
    #
    assert data.ndim == 4

    T, Z, Y, X = data.shape
    tmp = data.permute(0, 2, 3, 1).view(T, -1, Z)

    interpolated_data = (
        F.interpolate(
            tmp, size=(nz), mode=mode, align_corners=True if mode == "linear" else None
        )
        .view(T, Y, X, nz)
        .permute(0, 3, 1, 2)
    )

    assert interpolated_data.shape == (T, nz, Y, X)

    return interpolated_data
