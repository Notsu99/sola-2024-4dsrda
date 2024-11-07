from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger()


class DepthwiseConv(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise_conv(x)


class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        shrink_rate: float,
        bias: bool = True,
    ):
        super().__init__()

        self.shrinked_channels = int(in_channels * shrink_rate)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, self.shrinked_channels, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.shrinked_channels, in_channels, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        # x dim is (batch, channel, height, width)
        B, C, _, _ = x.shape

        # Apply average pooling and flatten
        y = self.avg_pool(x).view(B, C)

        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(B, C, 1, 1)

        return x * y.expand_as(x)


class MBConv(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        expansion_rate: int,
        shrink_rate: float,
        bias: bool = True,
        use_downsample: bool = False,
    ):
        super().__init__()

        self.expanded_channels = in_channels * expansion_rate
        self.use_downsample = use_downsample

        # Downsampling shortcut branch
        if use_downsample:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.conv_shortcut = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gelu = nn.GELU()

        self.bn1 = nn.BatchNorm2d(in_channels)

        # 1x1 Pointwise Convolution for Expansion
        self.expand_conv = nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1)

        # Depthwise Convolution
        self.bn2 = nn.BatchNorm2d(self.expanded_channels)

        stride = 2 if use_downsample else 1
        self.dwconv = DepthwiseConv(
            in_channels=self.expanded_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )

        # Squeeze-Excitation
        self.se = SqueezeExcitation(
            in_channels=self.expanded_channels, shrink_rate=shrink_rate, bias=bias
        )

        # 1x1 Pointwise Convolution for Projection
        self.project_conv = nn.Conv2d(
            self.expanded_channels, in_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        x = self.bn1(x)
        x = self.gelu(x)
        identity = x

        if self.use_downsample:
            identity = self.pool(identity)
            identity = self.conv_shortcut(identity)

        # Expansion
        x = self.expand_conv(x)

        # Depthwise Convolution
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.dwconv(x)

        # Squeeze-Excitation
        x = self.se(x)

        # Projection
        x = self.project_conv(x)

        # Residual Connection
        return x + identity
