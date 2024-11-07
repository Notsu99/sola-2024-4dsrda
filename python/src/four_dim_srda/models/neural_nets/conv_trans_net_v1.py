import copy
import dataclasses
import math
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.four_dim_srda.models.neural_nets.base_config import BaseModelConfig
from src.four_dim_srda.models.neural_nets.block.down_block import DownBlock
from src.four_dim_srda.models.neural_nets.block.up_block import UpBlock

logger = getLogger()


@dataclasses.dataclass
class ConvTransNetVer01Config(BaseModelConfig):
    bias: bool
    feat_channels_0: int
    feat_channels_1: int
    feat_channels_2: int
    feat_channels_3: int
    latent_channels: int
    hr_sequence_length: int
    hr_x_size: int
    hr_y_size: int
    hr_z_size: int
    input_channels: int
    output_channels: int
    input_sequence_length: int
    lr_x_size: int
    lr_y_size: int
    lr_z_size: int
    num_layers_x_in_block: int
    num_layers_o_in_block: int
    num_multi_attention_heads: int
    num_transformer_blocks: int
    use_global_skip_connection: bool


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        x_feat_channels_0: int,
        x_feat_channels_1: int,
        x_feat_channels_2: int,
        x_feat_channels_3: int,
        o_feat_channels_0: int,
        o_feat_channels_1: int,
        o_feat_channels_2: int,
        o_feat_channels_3: int,
        out_channels: int,
        bias_x_encoder: bool,
        bias_o_encoder: bool,
        num_layers_x_in_block: int,
        num_layers_o_in_block: int,
    ):
        super().__init__()

        self.x_encoder = nn.Sequential(
            DownBlock(
                in_channels=(x_feat_channels_0 + o_feat_channels_0),
                out_channels=x_feat_channels_1,
                bias=bias_x_encoder,
                stride=2,
                num_layers_in_block=num_layers_x_in_block,
            ),
            DownBlock(
                in_channels=x_feat_channels_1,
                out_channels=x_feat_channels_2,
                bias=bias_x_encoder,
                stride=2,
                num_layers_in_block=num_layers_x_in_block,
            ),
            DownBlock(
                in_channels=x_feat_channels_2,
                out_channels=x_feat_channels_3,
                bias=bias_x_encoder,
                stride=2,
                num_layers_in_block=num_layers_x_in_block,
            ),
            DownBlock(
                in_channels=x_feat_channels_3,
                out_channels=out_channels,
                bias=bias_x_encoder,
                stride=2,
                num_layers_in_block=num_layers_x_in_block,
            ),
        )

        self.o_encoder = nn.Sequential(
            DownBlock(
                in_channels=o_feat_channels_0,
                out_channels=o_feat_channels_1,
                bias=bias_o_encoder,
                stride=2,
                num_layers_in_block=num_layers_o_in_block,
            ),
            DownBlock(
                in_channels=o_feat_channels_1,
                out_channels=o_feat_channels_2,
                bias=bias_o_encoder,
                stride=2,
                num_layers_in_block=num_layers_o_in_block,
            ),
            DownBlock(
                in_channels=o_feat_channels_2,
                out_channels=o_feat_channels_3,
                bias=bias_o_encoder,
                stride=2,
                num_layers_in_block=num_layers_o_in_block,
            ),
            DownBlock(
                in_channels=o_feat_channels_3,
                out_channels=out_channels,
                bias=bias_o_encoder,
                stride=2,
                num_layers_in_block=num_layers_o_in_block,
            ),
        )

    def forward(
        self, x: torch.Tensor, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.cat([x, obs], dim=1)  # concat along channel dims
        y = self.x_encoder(y)

        z = self.o_encoder(obs)

        return y, z


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
        num_layers_in_block: int,
        upscale_factor: int = 2,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            UpBlock(
                in_channels=feat_channels_0,
                out_channels=feat_channels_1,
                bias=bias,
                num_layers_in_block=num_layers_in_block,
                upscale_factor=upscale_factor,
            ),
            UpBlock(
                in_channels=feat_channels_1,
                out_channels=feat_channels_2,
                bias=bias,
                num_layers_in_block=num_layers_in_block,
                upscale_factor=upscale_factor,
            ),
            UpBlock(
                in_channels=feat_channels_2,
                out_channels=feat_channels_3,
                bias=bias,
                num_layers_in_block=num_layers_in_block,
                upscale_factor=upscale_factor,
            ),
            UpBlock(
                in_channels=feat_channels_3,
                out_channels=out_channels,
                bias=bias,
                num_layers_in_block=num_layers_in_block,
                upscale_factor=upscale_factor,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class TransformerTimeSeriesMappingBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        sequence_length: int,
        bias: bool,
    ):
        super().__init__()

        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=False,
        )

        self.pe = self._sinusoidal_positional_encoding(sequence_length, d_model)

        self.linear = nn.Linear(2 * d_model, d_model, bias=bias)

    def _sinusoidal_positional_encoding(self, sequence_length: int, dim: int):
        pe = torch.zeros(sequence_length, dim)
        pos = torch.arange(0, sequence_length).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(100.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        # add batch dim
        pe = pe.unsqueeze(0)

        return pe

    def forward(self, xs: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        y = torch.cat([xs, obs], dim=-1)  # cocat along channel
        y = self.linear(y)
        self.pe = self.pe.to(y.device)

        y = y + self.pe

        return self.transformer(y)


class ConvTransNetVer01(nn.Module):
    def __init__(
        self,
        config_ins: ConvTransNetVer01Config,
    ):
        super().__init__()
        logger.info("Conv Trans Net v01")

        self.cfg = copy.deepcopy(config_ins)

        # 16 == 2**4 (encoder has 4 blocks to down sample)
        self.latent_x_size = self.cfg.hr_x_size // 16
        self.latent_y_size = self.cfg.hr_y_size // 16
        self.latent_z_size = self.cfg.hr_z_size // 16
        self.latent_dim = (
            self.latent_x_size
            * self.latent_y_size
            * self.latent_z_size
            * self.cfg.latent_channels
        )

        logger.info(
            f"LR size z = {self.cfg.lr_z_size}, y = {self.cfg.lr_y_size}, x = {self.cfg.lr_x_size}"
        )
        logger.info(
            f"HR size z = {self.cfg.hr_z_size}, y = {self.cfg.hr_y_size}, x = {self.cfg.hr_x_size}"
        )
        logger.info(
            f"Latent size z = {self.latent_z_size}, y = {self.latent_y_size}, x = {self.latent_x_size}"
        )
        logger.info(
            f"latent: dim= {self.latent_dim}, channels= {self.cfg.latent_channels}"
        )
        logger.info(f"bias = {self.cfg.bias}")
        logger.info(
            f"use global skip connection = {self.cfg.use_global_skip_connection}"
        )

        # Feature extractor
        self.x_feat_extractor = nn.Conv3d(
            self.cfg.input_channels,
            self.cfg.feat_channels_0,
            kernel_size=3,
            padding=1,
            bias=self.cfg.bias,
        )
        self.o_feat_extractor = nn.Conv3d(
            self.cfg.input_channels,
            self.cfg.feat_channels_0,
            kernel_size=3,
            padding=1,
            bias=self.cfg.bias,
        )

        # Encoder
        self.encoder = Encoder(
            x_feat_channels_0=self.cfg.feat_channels_0,
            x_feat_channels_1=self.cfg.feat_channels_1,
            x_feat_channels_2=self.cfg.feat_channels_2,
            x_feat_channels_3=self.cfg.feat_channels_3,
            o_feat_channels_0=self.cfg.feat_channels_0,
            o_feat_channels_1=self.cfg.feat_channels_1,
            o_feat_channels_2=self.cfg.feat_channels_2,
            o_feat_channels_3=self.cfg.feat_channels_3,
            out_channels=self.cfg.latent_channels,
            bias_x_encoder=self.cfg.bias,
            bias_o_encoder=self.cfg.bias,
            num_layers_x_in_block=self.cfg.num_layers_x_in_block,
            num_layers_o_in_block=self.cfg.num_layers_o_in_block,
        )

        # Transformer
        self.transformers = []
        for _ in range(self.cfg.num_transformer_blocks):
            self.transformers.append(
                TransformerTimeSeriesMappingBlock(
                    d_model=self.latent_dim,
                    nhead=self.cfg.num_multi_attention_heads,
                    dim_feedforward=self.latent_dim,
                    sequence_length=self.cfg.hr_sequence_length,
                    bias=self.cfg.bias,
                )
            )
        self.transformers = nn.ModuleList(self.transformers)

        # Decoder
        self.decoder = Decoder(
            feat_channels_0=self.cfg.latent_channels,
            feat_channels_1=self.cfg.feat_channels_3,
            feat_channels_2=self.cfg.feat_channels_2,
            feat_channels_3=self.cfg.feat_channels_1,
            out_channels=self.cfg.feat_channels_0,
            bias=self.cfg.bias,
            num_layers_in_block=self.cfg.num_layers_x_in_block,
        )

        # Reconstructor
        if self.cfg.use_global_skip_connection:
            self.reconstructor = nn.Sequential(
                nn.Conv3d(
                    3 * self.cfg.feat_channels_0,
                    self.cfg.feat_channels_0,
                    kernel_size=3,
                    padding=1,
                    bias=self.cfg.bias,
                ),
                nn.LeakyReLU(),
                nn.Conv3d(
                    self.cfg.feat_channels_0,
                    self.cfg.output_channels,
                    kernel_size=3,
                    padding=1,
                    bias=self.cfg.bias,
                ),
            )
        else:
            self.reconstructor = nn.Sequential(
                nn.Conv3d(
                    self.cfg.feat_channels_0,
                    self.cfg.feat_channels_0,
                    kernel_size=3,
                    padding=1,
                    bias=self.cfg.bias,
                ),
                nn.LeakyReLU(),
                nn.Conv3d(
                    self.cfg.feat_channels_0,
                    self.cfg.output_channels,
                    kernel_size=3,
                    padding=1,
                    bias=self.cfg.bias,
                ),
            )

    def _interpolate_features_along_time(self, feat: torch.Tensor, batch_size: int):
        feat = feat.view(
            batch_size,
            self.cfg.input_sequence_length,
            -1,
            self.cfg.hr_z_size,
            self.cfg.hr_y_size,
            self.cfg.hr_x_size,
        )

        # Interpolate along time
        feat = feat.permute(0, 2, 3, 4, 5, 1)
        _, C, D, H, W, T = feat.shape
        feat = F.interpolate(
            feat.view(batch_size, -1, T),
            size=self.cfg.hr_sequence_length,
            mode="linear",
            align_corners=True,
        )
        feat = feat.view(batch_size, C, D, H, W, self.cfg.hr_sequence_length)
        feat = feat.permute(0, 5, 1, 2, 3, 4)

        return feat.reshape(
            batch_size * self.cfg.hr_sequence_length,
            C,
            self.cfg.hr_z_size,
            self.cfg.hr_y_size,
            self.cfg.hr_x_size,
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
            mode="nearest",
        ).view((B, C, T) + size)

        # Reshape x and obs to apply the same encoder at each time step
        x = x.view((-1, self.cfg.input_channels) + size)
        obs = hr_obs.view((-1, self.cfg.input_channels) + size)

        # Extract x and obs feature
        # Only the num of channels is changed, the other shapes remain
        feat_x = self.x_feat_extractor(x)
        feat_o = self.o_feat_extractor(obs)

        # Encoder block
        latent_x, latent_o = self.encoder(feat_x, feat_o)

        # latent_x.shape[1:] is (
        #     self.cfg.latent_channels,
        #     self.latent_z_size,
        #     self.latent_y_size,
        #     self.latent_x_size,
        # )

        # latent_o.shape[1:] is == (
        #     self.cfg.latent_channels,
        #     self.latent_z_size,
        #     self.latent_y_size,
        #     self.latent_x_size,
        # )

        # Transformer block
        # Change dim for using transformer
        latent_x = latent_x.view(-1, self.cfg.input_sequence_length, self.latent_dim)
        latent_o = latent_o.view(-1, self.cfg.input_sequence_length, self.latent_dim)

        # Interpolate along time
        latent_x = latent_x.permute(0, 2, 1)
        latent_o = latent_o.permute(0, 2, 1)
        latent_x = F.interpolate(
            latent_x,
            size=self.cfg.hr_sequence_length,
            mode="linear",
            align_corners=True,
        )
        latent_o = F.interpolate(
            latent_o,
            size=self.cfg.hr_sequence_length,
            mode="linear",
            align_corners=True,
        )
        latent_x = latent_x.permute(0, 2, 1)
        latent_o = latent_o.permute(0, 2, 1)

        y = latent_x
        for transformer in self.transformers:
            y = transformer(y, latent_o)
        y = y + latent_x

        y = y.view(
            -1,
            self.cfg.latent_channels,
            self.latent_z_size,
            self.latent_y_size,
            self.latent_x_size,
        )

        # Decoder block
        y = self.decoder(y)

        if self.cfg.use_global_skip_connection:
            feat_x = self._interpolate_features_along_time(feat_x, B)
            feat_o = self._interpolate_features_along_time(feat_o, B)
            y = torch.cat([y, feat_x, feat_o], dim=1)  # along channel dim

        y = self.reconstructor(y)

        hr_pv = y.view(
            -1,
            self.cfg.hr_sequence_length,
            self.cfg.output_channels,
            self.cfg.hr_z_size,
            self.cfg.hr_y_size,
            self.cfg.hr_x_size,
        )

        return hr_pv
