from logging import getLogger

import torch
import torch.nn as nn
from src.four_dim_srda.models.neural_nets.block.maxvit_block.mb_conv import MBConv
from src.four_dim_srda.models.neural_nets.block.maxvit_block.multi_axis_self_attention import (
    MulitAxisSelfAttention,
)

logger = getLogger()


class MaxViTBlock(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        expansion_rate: int,
        shrink_rate: float,
        bias: bool,
        n_head: int,
        grid_window_size_h: int,
        grid_window_size_w: int,
        use_downsample: bool,
        dropout: float,
    ):
        super().__init__()

        self.mbconv = MBConv(
            in_channels=emb_dim,
            expansion_rate=expansion_rate,
            shrink_rate=shrink_rate,
            bias=bias,
            use_downsample=use_downsample,
        )
        self.multi_axis_attn = MulitAxisSelfAttention(
            n_head=n_head,
            emb_dim=emb_dim,
            grid_window_size_h=grid_window_size_h,
            grid_window_size_w=grid_window_size_w,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        identity = x
        y = self.mbconv(x)
        y = self.multi_axis_attn(y)

        return y + identity
