from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger()


class DownBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        bias: bool,
        stride: int,
        num_layers_in_block: int,
    ):
        super().__init__()

        assert num_layers_in_block >= 2

        self.down = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
            ),
            nn.ReLU(),
        )

        convs = []
        for _ in range(num_layers_in_block - 1):
            convs.append(
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                )
            )
            convs.append(nn.ReLU())

        self.convs = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        return self.convs(y)


class ResBlock(nn.Module):
    def __init__(self, channels: int, bias: bool):
        #
        super().__init__()

        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + identity)


class DownResBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        bias: bool,
        stride: int,
        num_layers_in_block: int,
    ):
        super().__init__()

        assert num_layers_in_block >= 2

        self.down = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
            ),
            nn.ReLU(),
        )

        resblocks = []
        for _ in range(num_layers_in_block - 1):
            resblocks.append(ResBlock(out_channels, bias))

        self.resblocks = nn.Sequential(*resblocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        return self.resblocks(y)
