from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger()


class ResNet2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        activation_type: str,
        norm_type: str,
        *,
        normalized_space_shape: list[int] = None,
    ):
        super().__init__()

        self.normalized_space_shape = normalized_space_shape

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.norm1 = self._get_norm_layer(out_channels, norm_type)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.norm2 = self._get_norm_layer(out_channels, norm_type)

        if in_channels != out_channels:
            self.identity_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
            )
        else:
            self.identity_conv = nn.Identity()

        self.activation = self._get_activation_layer(activation_type)

    def _get_norm_layer(self, num_features: int, norm_type: str) -> nn.Module:
        if norm_type == "batch":
            return nn.BatchNorm2d(num_features)
        elif norm_type == "layer":
            if self.normalized_space_shape is None:
                raise ValueError("normalized_space_shape must be provided for LayerNorm")
            return nn.LayerNorm([num_features, *self.normalized_space_shape])
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")

    def _get_activation_layer(self, activation_type: str) -> nn.Module:
        if activation_type == "relu":
            return nn.ReLU()
        elif activation_type == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_type == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity_conv(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        return self.activation(out + identity)


class ResNet2dBlocks(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        activation_type: str,
        norm_type: str,
        num_resnet_block: int,
        *,
        normalized_space_shape: list[int] = None,
    ):
        super().__init__()

        assert num_resnet_block >= 1

        resnet_blocks = []
        for i in range(num_resnet_block):
            resnet_blocks.append(
                ResNet2dBlock(
                    (in_channels if i == 0 else out_channels),
                    out_channels,
                    bias,
                    activation_type,
                    norm_type,
                    normalized_space_shape=normalized_space_shape,
                )
            )

        self.resnet_blocks = nn.Sequential(*resnet_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet_blocks(x)


class ResNet3dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        activation_type: str,
        norm_type: str,
        *,
        normalized_space_shape: list[int] = None,
    ):
        #
        super().__init__()

        self.normalized_space_shape = normalized_space_shape

        #
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.norm1 = self._get_norm_layer(out_channels, norm_type)

        #
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.norm2 = self._get_norm_layer(out_channels, norm_type)

        #
        if in_channels != out_channels:
            self.identity_conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
            )
        else:
            self.identity_conv = nn.Identity()

        self.activation = self._get_activation_layer(activation_type)

    def _get_norm_layer(self, num_features: int, norm_type: str) -> nn.Module:
        if norm_type == "batch":
            return nn.BatchNorm3d(num_features)
        elif norm_type == "layer":
            if self.normalized_space_shape is None:
                raise ValueError("normalized_space_shape must be provided for LayerNorm")
            return nn.LayerNorm([num_features, *self.normalized_space_shape])
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")

    def _get_activation_layer(self, activation_type: str) -> nn.Module:
        if activation_type == "relu":
            return nn.ReLU()
        elif activation_type == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_type == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        identity = self.identity_conv(x)

        #
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        #
        out = self.conv2(out)
        out = self.norm2(out)

        return self.activation(out + identity)


class ResNet3dBlocks(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        activation_type: str,
        norm_type: str,
        num_resnet_block: int,
        *,
        normalized_space_shape: list[int] = None,
    ):
        #
        super().__init__()

        assert num_resnet_block >= 1

        #
        resnet_blocks = []
        for i in range(num_resnet_block):
            resnet_blocks.append(
                ResNet3dBlock(
                    (in_channels if i == 0 else out_channels),
                    out_channels,
                    bias,
                    activation_type,
                    norm_type,
                    normalized_space_shape=normalized_space_shape,
                )
            )

        self.resnet_blocks = nn.Sequential(*resnet_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        return self.resnet_blocks(x)
