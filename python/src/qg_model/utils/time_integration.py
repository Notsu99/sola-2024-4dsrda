from logging import getLogger
from typing import Callable

import torch

logger = getLogger()


def integrate_one_step_rk2(
    *, dt: float, x: torch.Tensor, dxdt: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    #
    k1 = dxdt(x)
    k2 = dxdt(x + k1 * dt / 2)

    return x + k2 * dt
