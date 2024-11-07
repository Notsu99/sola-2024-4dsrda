from logging import getLogger

import torch
import torch.nn as nn
from src.four_dim_srda.models.neural_nets.block.voxel_shuffle import VoxelShuffle

logger = getLogger()


class VoxelShuffleBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        bias: bool,
        upscale_factor: int,
    ):
        super().__init__()

        # Need to adjust　channel numbers for VoxelShuffle
        self.conv = nn.Conv3d(
            in_channels,
            in_channels * upscale_factor**3,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        self.act = nn.LeakyReLU()

        self.upsample = VoxelShuffle(factor=upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.act(y)
        return self.upsample(y)


class UpBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        bias: bool,
        num_layers_in_block: int,
        upscale_factor: int,
    ):
        super().__init__()

        assert num_layers_in_block >= 2

        self.up = VoxelShuffleBlock(
            in_channels=in_channels,
            bias=bias,
            upscale_factor=upscale_factor,
        )

        convs = []
        for i in range(num_layers_in_block - 1):
            convs.append(
                nn.Conv3d(
                    (in_channels if i == 0 else out_channels),
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                )
            )
            convs.append(nn.LeakyReLU())

        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        return self.convs(y)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool):
        #
        super().__init__()
        #
        if in_channels != out_channels:
            self.use_identity_conv = True
        else:
            self.use_identity_conv = False

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        if self.use_identity_conv:
            self.identity_conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        identity = self.identity_conv(x) if self.use_identity_conv else x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + identity)


class UpResBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        bias: bool,
        num_layers_in_block: int,
        upscale_factor: int,
    ):
        super().__init__()

        assert num_layers_in_block >= 2

        self.up = VoxelShuffleBlock(
            in_channels=in_channels,
            bias=bias,
            upscale_factor=upscale_factor,
        )

        resblocks = []
        for i in range(num_layers_in_block - 1):
            resblocks.append(
                ResBlock(
                    (in_channels if i == 0 else out_channels),
                    out_channels,
                    bias,
                )
            )

        self.resblocks = nn.Sequential(*resblocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        return self.resblocks(y)
