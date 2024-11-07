from logging import getLogger
from typing import Callable

import torch
import torch.nn as nn

logger = getLogger()


def window_partition(
    x: torch.Tensor,
    window_size_h: int,
    window_size_w: int,
) -> torch.Tensor:
    #
    B, C, H, W = x.shape

    # Unfold input and permute
    windows = x.view(
        B, C, H // window_size_h, window_size_h, W // window_size_w, window_size_w
    ).permute(0, 2, 4, 3, 5, 1)

    # Reshape to (B * windows, window_size_h, window_size_w, channel)
    windows = windows.contiguous().view(-1, window_size_h, window_size_w, C)

    return windows


def window_reverse(
    windows: torch.Tensor,
    original_size: tuple[int, int],
    window_size_h: int,
    window_size_w: int,
) -> torch.Tensor:
    #
    H, W = original_size

    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size_h / window_size_w))

    # Fold
    output = windows.view(
        B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1
    )
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)

    return output


def grid_partition(
    x: torch.Tensor,
    grid_size_h: int,
    grid_size_w: int,
) -> torch.Tensor:
    #
    B, C, H, W = x.shape

    # Unfold input
    grid = x.view(B, C, grid_size_h, H // grid_size_h, grid_size_w, W // grid_size_w)

    # Permute and reshape to (B * (H // grid_size_h) * (W // grid_size_w), grid_size_h, grid_size_w, C)
    grid = (
        grid.permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(-1, grid_size_h, grid_size_w, C)
    )

    return grid


def grid_reverse(
    grid: torch.Tensor,
    original_size: tuple[int, int],
    grid_size_h: int,
    grid_size_w: int,
) -> torch.Tensor:
    #
    (H, W), C = original_size, grid.shape[-1]

    # Compute original batch size
    B = int(grid.shape[0] / (H * W / grid_size_h / grid_size_w))

    # Fold
    output = grid.view(
        B, H // grid_size_h, W // grid_size_w, grid_size_h, grid_size_w, C
    )
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)

    return output


def get_relative_position_index(
    grid_win_h: int,
    grid_win_w: int,
) -> torch.Tensor:
    #
    coords = torch.stack(
        torch.meshgrid(
            [torch.arange(grid_win_h), torch.arange(grid_win_w)], indexing="ij"
        )
    )
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += grid_win_h - 1
    relative_coords[:, :, 1] += grid_win_w - 1
    relative_coords[:, :, 0] *= 2 * grid_win_w - 1

    return relative_coords.sum(-1)


class RelativeSelfAttention(nn.Module):
    #
    def __init__(
        self,
        *,
        n_head: int,
        emb_dim: int,
        grid_window_size_h: int,
        grid_window_size_w: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Parameters
        self.n_head = n_head

        self.scale = n_head**-0.5
        self.attn_area = grid_window_size_h * grid_window_size_w

        #
        self.qkv_mapping = nn.Linear(emb_dim, 3 * emb_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.proj_drop = nn.Dropout(dropout)

        #
        self.relative_position_bias_table = nn.Parameter(
            torch.randn(
                (2 * grid_window_size_h - 1) * (2 * grid_window_size_w - 1), n_head
            )
        )
        self.register_buffer(
            "relative_position_index",
            get_relative_position_index(grid_window_size_h, grid_window_size_w),
        )

    def _get_relative_positional_bias(self) -> torch.Tensor:
        #
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        B, N, C = x.shape

        # Perform query key value mapping
        qkv = (
            self.qkv_mapping(x).reshape(B, N, 3, self.n_head, -1).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaling and compute attention
        q = q * self.scale
        attn = self.softmax(
            q @ k.transpose(-2, -1) + self._get_relative_positional_bias()
        )

        #
        output = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class MaxViTTransformerBlock(nn.Module):
    #
    def __init__(
        self,
        *,
        n_head: int,
        partition_function: Callable,
        reverse_function: Callable,
        emb_dim: int,
        grid_window_size_h: int,
        grid_window_size_w: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        #
        self.partition_function = partition_function
        self.reverse_function = reverse_function

        self.grid_window_size_h = grid_window_size_h
        self.grid_window_size_w = grid_window_size_w

        #
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attention = RelativeSelfAttention(
            n_head=n_head,
            emb_dim=emb_dim,
            grid_window_size_h=grid_window_size_h,
            grid_window_size_w=grid_window_size_w,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        B, C, H, W = x.shape

        # Perform partition
        input_partitioned = self.partition_function(
            x,
            self.grid_window_size_h,
            self.grid_window_size_w,
        )
        input_partitioned = input_partitioned.view(
            -1, self.grid_window_size_h * self.grid_window_size_w, C
        )

        #
        output = input_partitioned + self.attention(self.norm1(input_partitioned))

        # Reverse partition
        output = self.reverse_function(
            output,
            (H, W),
            self.grid_window_size_h,
            self.grid_window_size_w,
        )

        return output


class MulitAxisSelfAttention(nn.Module):
    #
    def __init__(
        self,
        *,
        n_head: int,
        emb_dim: int,
        grid_window_size_h: int,
        grid_window_size_w: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        #

        self.grid_window_size_h = grid_window_size_h
        self.grid_window_size_w = grid_window_size_w

        #
        self.block_transformer = MaxViTTransformerBlock(
            n_head=n_head,
            partition_function=window_partition,
            reverse_function=window_reverse,
            emb_dim=emb_dim,
            grid_window_size_h=grid_window_size_h,
            grid_window_size_w=grid_window_size_w,
            dropout=dropout,
        )
        self.grid_transformer = MaxViTTransformerBlock(
            n_head=n_head,
            partition_function=grid_partition,
            reverse_function=grid_reverse,
            emb_dim=emb_dim,
            grid_window_size_h=grid_window_size_h,
            grid_window_size_w=grid_window_size_w,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        output = self.block_transformer(x)
        output = self.grid_transformer(output)

        return output
