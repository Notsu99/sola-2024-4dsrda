import copy
import dataclasses
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.four_dim_srda.models.neural_nets.base_config import BaseModelConfig
from src.four_dim_srda.models.neural_nets.block.resnet import (
    ResNet2dBlocks,
    ResNet3dBlocks,
)
from src.four_dim_srda.models.neural_nets.block.up_block import VoxelShuffleBlock

logger = getLogger()


@dataclasses.dataclass
class UNetVer02Config(BaseModelConfig):
    bias: bool
    input_channels: int
    feat_channels_0: int
    feat_channels_1: int
    feat_channels_2: int
    feat_channels_3: int
    encoder_output_channels: int
    input_sequence_length: int
    hr_sequence_length: int
    num_2d_resnet_block: int
    num_3d_resnet_block: int
    encoder_activation_type: str
    decoder_activation_type: str
    other_activation_type: str
    encoder_norm_type: str
    decoder_norm_type: str
    other_norm_type: str
    hr_x_size: int
    hr_y_size: int
    hr_z_size: int
    lr_x_size: int
    lr_y_size: int
    lr_z_size: int


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        feat_channels_0: int,
        feat_channels_1: int,
        feat_channels_2: int,
        feat_channels_3: int,
        out_channels: int,
        bias: bool,
        activation_type: str,
        norm_type: str,
        num_resnet_block: int,
        init_space_shape: list[int] = None,
    ):
        #
        super().__init__()

        self.scale_factor = 2

        if norm_type == "layer":
            self.normalized_space_shape1 = [
                size // self.scale_factor for size in init_space_shape
            ]
            self.normalized_space_shape2 = [
                size // self.scale_factor for size in self.normalized_space_shape1
            ]
            self.normalized_space_shape3 = [
                size // self.scale_factor for size in self.normalized_space_shape2
            ]
            self.normalized_space_shape4 = [
                size // self.scale_factor for size in self.normalized_space_shape3
            ]

        #
        self.enc1 = ResNet3dBlocks(
            feat_channels_0,
            feat_channels_1,
            bias,
            activation_type,
            norm_type,
            num_resnet_block,
            normalized_space_shape=self.normalized_space_shape1
            if norm_type == "layer"
            else None,
        )
        self.enc2 = ResNet3dBlocks(
            feat_channels_1,
            feat_channels_2,
            bias,
            activation_type,
            norm_type,
            num_resnet_block,
            normalized_space_shape=self.normalized_space_shape2
            if norm_type == "layer"
            else None,
        )
        self.enc3 = ResNet3dBlocks(
            feat_channels_2,
            feat_channels_3,
            bias,
            activation_type,
            norm_type,
            num_resnet_block,
            normalized_space_shape=self.normalized_space_shape3
            if norm_type == "layer"
            else None,
        )
        self.enc4 = ResNet3dBlocks(
            feat_channels_3,
            out_channels,
            bias,
            activation_type,
            norm_type,
            num_resnet_block,
            normalized_space_shape=self.normalized_space_shape4
            if norm_type == "layer"
            else None,
        )

        #
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        enc1 = self.enc1(self.pool(x))
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.enc4(self.pool(enc3))
        return identity, enc1, enc2, enc3, bottleneck


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        feat_channels_0: int,
        feat_channels_1: int,
        feat_channels_2: int,
        feat_channels_3: int,
        out_channels: int,
        bias: bool,
        activation_type: str,
        norm_type: str,
        num_resnet_block: int,
        init_space_shape: list[int] = None,
    ):
        #
        super().__init__()

        self.scale_factor = 2

        if norm_type == "layer":
            self.normalized_space_shape1 = [
                size * self.scale_factor for size in init_space_shape
            ]
            self.normalized_space_shape2 = [
                size * self.scale_factor for size in self.normalized_space_shape1
            ]
            self.normalized_space_shape3 = [
                size * self.scale_factor for size in self.normalized_space_shape2
            ]
            self.normalized_space_shape4 = [
                size * self.scale_factor for size in self.normalized_space_shape3
            ]

        self.up1 = VoxelShuffleBlock(
            in_channels=feat_channels_0, bias=bias, upscale_factor=self.scale_factor
        )
        self.dec1 = ResNet3dBlocks(
            feat_channels_0 + feat_channels_1,
            feat_channels_1,
            bias,
            activation_type,
            norm_type,
            num_resnet_block,
            normalized_space_shape=self.normalized_space_shape1
            if norm_type == "layer"
            else None,
        )

        self.up2 = VoxelShuffleBlock(
            in_channels=feat_channels_1, bias=bias, upscale_factor=self.scale_factor
        )
        self.dec2 = ResNet3dBlocks(
            feat_channels_1 + feat_channels_2,
            feat_channels_2,
            bias,
            activation_type,
            norm_type,
            num_resnet_block,
            normalized_space_shape=self.normalized_space_shape2
            if norm_type == "layer"
            else None,
        )

        self.up3 = VoxelShuffleBlock(
            in_channels=feat_channels_2, bias=bias, upscale_factor=self.scale_factor
        )
        self.dec3 = ResNet3dBlocks(
            feat_channels_2 + feat_channels_3,
            feat_channels_3,
            bias,
            activation_type,
            norm_type,
            num_resnet_block,
            normalized_space_shape=self.normalized_space_shape3
            if norm_type == "layer"
            else None,
        )

        self.up4 = VoxelShuffleBlock(
            in_channels=feat_channels_3, bias=bias, upscale_factor=self.scale_factor
        )
        self.dec4 = ResNet3dBlocks(
            feat_channels_3 + out_channels,
            out_channels,
            bias,
            activation_type,
            norm_type,
            num_resnet_block,
            normalized_space_shape=self.normalized_space_shape4
            if norm_type == "layer"
            else None,
        )

    def forward(
        self,
        identity: torch.Tensor,
        enc1: torch.Tensor,
        enc2: torch.Tensor,
        enc3: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        #
        x = self.up1(bottleneck)
        x = torch.cat((x, enc3), dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x = torch.cat((x, identity), dim=1)
        x = self.dec4(x)

        return x


class UNetVer02(nn.Module):
    def __init__(
        self,
        cfg: UNetVer02Config,
    ):
        #
        super().__init__()

        self.cfg = copy.deepcopy(cfg)

        logger.info("UNetVer01 is initialized")

        self.scale_factor = 2

        self.init_space_shape = [
            self.cfg.hr_z_size,
            self.cfg.hr_y_size,
            self.cfg.hr_x_size,
        ]

        self.latent_2d_space_shape = [
            size // self.scale_factor**4 for size in self.init_space_shape[1:]
        ]

        self.latent_3d_space_shape = [
            size // self.scale_factor**4 for size in self.init_space_shape
        ]

        # lr and obs time series data are concatenated along channel dim
        self.feat_extractor = ResNet3dBlocks(
            2 * self.cfg.input_channels * self.cfg.input_sequence_length,
            self.cfg.feat_channels_0,
            self.cfg.bias,
            self.cfg.other_activation_type,
            self.cfg.other_norm_type,
            num_resnet_block=1,
            normalized_space_shape=self.init_space_shape
            if self.cfg.other_norm_type == "layer"
            else None,
        )

        # Encoder
        self.encoder = Encoder(
            feat_channels_0=self.cfg.feat_channels_0,
            feat_channels_1=self.cfg.feat_channels_1,
            feat_channels_2=self.cfg.feat_channels_2,
            feat_channels_3=self.cfg.feat_channels_3,
            out_channels=self.cfg.encoder_output_channels,
            bias=self.cfg.bias,
            activation_type=self.cfg.encoder_activation_type,
            norm_type=self.cfg.encoder_norm_type,
            num_resnet_block=self.cfg.num_3d_resnet_block,
            init_space_shape=self.init_space_shape
            if self.cfg.encoder_norm_type == "layer"
            else None,
        )

        self.latent_feature_extractor = ResNet2dBlocks(
            self.cfg.encoder_output_channels,
            self.cfg.encoder_output_channels,
            self.cfg.bias,
            self.cfg.other_activation_type,
            self.cfg.other_norm_type,
            self.cfg.num_2d_resnet_block,
            normalized_space_shape=self.latent_2d_space_shape
            if self.cfg.other_norm_type == "layer"
            else None,
        )

        # Decoder
        self.decoder = Decoder(
            feat_channels_0=self.cfg.encoder_output_channels,
            feat_channels_1=self.cfg.feat_channels_3,
            feat_channels_2=self.cfg.feat_channels_2,
            feat_channels_3=self.cfg.feat_channels_1,
            out_channels=self.cfg.feat_channels_0,
            bias=self.cfg.bias,
            activation_type=self.cfg.decoder_activation_type,
            norm_type=self.cfg.decoder_norm_type,
            num_resnet_block=self.cfg.num_3d_resnet_block,
            init_space_shape=self.latent_3d_space_shape
            if self.cfg.decoder_norm_type == "layer"
            else None,
        )

        self.reconstructor = ResNet3dBlocks(
            self.cfg.feat_channels_0,
            self.cfg.hr_sequence_length,
            self.cfg.bias,
            self.cfg.other_activation_type,
            self.cfg.other_norm_type,
            num_resnet_block=1,
            normalized_space_shape=self.init_space_shape
            if self.cfg.other_norm_type == "layer"
            else None,
        )

    def forward(self, lr_pv: torch.Tensor, hr_obs: torch.Tensor) -> torch.Tensor:
        # Skip batch dim and assert the other dims
        assert lr_pv.shape[1:] == (
            self.cfg.input_sequence_length,
            self.cfg.input_channels,
            self.cfg.lr_z_size,
            self.cfg.lr_y_size,
            self.cfg.lr_x_size,
        )
        assert hr_obs.shape[1:] == (
            self.cfg.input_sequence_length,
            self.cfg.input_channels,
            self.cfg.hr_z_size,
            self.cfg.hr_y_size,
            self.cfg.hr_x_size,
        )

        B, T, C, D, H, W = lr_pv.shape

        # Interpolate to hr grid space, while timesteps remain.
        size = (
            self.cfg.hr_z_size,
            self.cfg.hr_y_size,
            self.cfg.hr_x_size,
        )
        x = F.interpolate(
            lr_pv.view(B, -1, D, H, W),
            size=size,
            mode="nearest-exact",
        ).view((B, C, T) + size)

        # Reshape x and obs by flattening time and channel dim
        x = x.view((B, -1) + size)
        obs = hr_obs.view((B, -1) + size)

        # Concat x and obs along channel dim
        combined_input = torch.cat((x, obs), dim=1)

        # Extract x and obs feature
        # Only the num of channels is changed, the other shapes remain
        feat = self.feat_extractor(combined_input)

        # Encoder block
        identity, enc1, enc2, enc3, bottleneck = self.encoder(feat)

        # Squeeze the depth dim (assuming it is 1) to convert to 2D
        bottleneck = torch.squeeze(bottleneck, dim=2)
        bottleneck = self.latent_feature_extractor(bottleneck)

        # Add the depth dim back to convert to 3D
        bottleneck = torch.unsqueeze(bottleneck, dim=2)

        # Decoder block
        y = self.decoder(identity, enc1, enc2, enc3, bottleneck)

        y = self.reconstructor(y)

        y = y.view(
            B,
            self.cfg.hr_sequence_length,
            1,
            self.cfg.hr_z_size,
            self.cfg.hr_y_size,
            self.cfg.hr_x_size,
        )

        return y
