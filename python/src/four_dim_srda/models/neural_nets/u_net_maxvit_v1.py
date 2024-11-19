import copy
import dataclasses
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.four_dim_srda.models.neural_nets.base_config import BaseModelConfig
from src.four_dim_srda.models.neural_nets.block.maxvit_block.maxvit_block import (
    MaxViTBlock,
)
from src.four_dim_srda.models.neural_nets.block.resnet import ResNet3dBlocks
from src.four_dim_srda.models.neural_nets.block.up_block import VoxelShuffleBlock

logger = getLogger()


@dataclasses.dataclass
class UNetMaxVitVer01Config(BaseModelConfig):
    bias: bool
    #
    input_channels: int
    feat_channels_0: int
    feat_channels_1: int
    encoder_output_channels: int
    #
    input_sequence_length: int
    hr_sequence_length: int
    #
    num_3d_resnet_block: int
    #
    encoder_activation_type: str
    decoder_activation_type: str
    other_activation_type: str
    #
    encoder_norm_type: str
    decoder_norm_type: str
    other_norm_type: str
    #
    expansion_rate: int
    shrink_rate: float
    use_downsample: bool
    grid_window_size_h: int
    grid_window_size_w: int
    n_head: int
    dropout: float
    num_maxvit_block: int
    #
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
            out_channels,
            bias,
            activation_type,
            norm_type,
            num_resnet_block,
            normalized_space_shape=self.normalized_space_shape2
            if norm_type == "layer"
            else None,
        )

        #
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #
        identity = x
        enc1 = self.enc1(self.pool(x))
        bottleneck = self.enc2(self.pool(enc1))

        return identity, enc1, bottleneck


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        feat_channels_0: int,
        feat_channels_1: int,
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
            feat_channels_1 + out_channels,
            out_channels,
            bias,
            activation_type,
            norm_type,
            num_resnet_block,
            normalized_space_shape=self.normalized_space_shape2
            if norm_type == "layer"
            else None,
        )

    def forward(
        self,
        identity: torch.Tensor,
        enc1: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        #
        x = self.up1(bottleneck)
        x = torch.cat((x, enc1), dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat((x, identity), dim=1)
        x = self.dec2(x)

        return x


class MaxViTBlocks(nn.Module):
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
        num_block: int,
    ):
        super().__init__()

        assert num_block >= 1

        blocks = []
        for i in range(num_block):
            blocks.append(
                MaxViTBlock(
                    emb_dim=emb_dim,
                    expansion_rate=expansion_rate,
                    shrink_rate=shrink_rate,
                    bias=bias,
                    n_head=n_head,
                    grid_window_size_h=grid_window_size_h,
                    grid_window_size_w=grid_window_size_w,
                    use_downsample=use_downsample,
                    dropout=dropout,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class UNetMaxVitVer01(nn.Module):
    def __init__(
        self,
        cfg: UNetMaxVitVer01Config,
    ):
        #
        super().__init__()

        self.cfg = copy.deepcopy(cfg)

        logger.info("U-Net MaxViT Ver01 is initialized")

        self.scale_factor = 2

        self.init_space_shape = [
            self.cfg.hr_z_size,
            self.cfg.hr_y_size,
            self.cfg.hr_x_size,
        ]

        self.latent_2d_space_shape = [
            size // self.scale_factor**2 for size in self.init_space_shape[1:]
        ]

        self.latent_3d_space_shape = [
            size // self.scale_factor**2 for size in self.init_space_shape
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
            out_channels=self.cfg.encoder_output_channels,
            bias=self.cfg.bias,
            activation_type=self.cfg.encoder_activation_type,
            norm_type=self.cfg.encoder_norm_type,
            num_resnet_block=self.cfg.num_3d_resnet_block,
            init_space_shape=self.init_space_shape
            if self.cfg.encoder_norm_type == "layer"
            else None,
        )

        # MaxViT
        self.latent_feature_extractor = MaxViTBlocks(
            # emd_sim is decided by input channels and nz in latent space
            emb_dim=self.cfg.encoder_output_channels * self.latent_3d_space_shape[0],
            expansion_rate=self.cfg.expansion_rate,
            shrink_rate=self.cfg.shrink_rate,
            bias=self.cfg.bias,
            use_downsample=self.cfg.use_downsample,
            n_head=self.cfg.n_head,
            grid_window_size_h=self.cfg.grid_window_size_h,
            grid_window_size_w=self.cfg.grid_window_size_w,
            dropout=self.cfg.dropout,
            num_block=self.cfg.num_maxvit_block,
        )

        # Decoder
        self.decoder = Decoder(
            feat_channels_0=self.cfg.encoder_output_channels,
            feat_channels_1=self.cfg.feat_channels_1,
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
        identity, enc1, bottleneck = self.encoder(feat)

        # Reshape to 2D for MaxViT
        b, c, d, h, w = bottleneck.shape
        bottleneck = bottleneck.view(b, -1, h, w)

        # MaxViT
        bottleneck = self.latent_feature_extractor(bottleneck)

        # Reshape for decoder
        bottleneck = bottleneck.view(b, c, d, h, w)

        # Decoder block
        y = self.decoder(identity, enc1, bottleneck)

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
