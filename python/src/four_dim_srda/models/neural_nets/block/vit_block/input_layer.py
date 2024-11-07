from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger()


class VitInputLayer(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        emb_dim: int,
        input_size_x: int,
        input_size_y: int,
        patch_size_x: int,
        patch_size_y: int,
    ):
        super().__init__()

        self.n_patches = (input_size_x // patch_size_x) * (input_size_y // patch_size_y)

        # Patch splitting and Embedding
        self.patch_emb_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=(patch_size_y, patch_size_x),
            stride=(patch_size_y, patch_size_x),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Positional embedding
        # Class token is concatenated along　num　patch dim, so add 1
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches + 1, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        # x dim is (batch, channel, y, x)

        B, C, Y, X = x.shape

        # dim becomes (batch, emb_dim, y / patch_size_y, x / patch_size_x)
        y = self.patch_emb_layer(x)

        # dim becomes (batch, emb_dim, num_patch)
        y = y.flatten(2)

        y = y.transpose(1, 2)

        # Repeat cls token for each batch
        cls_tokens = self.cls_token.expand(B, -1, -1)

        y = torch.cat([cls_tokens, y], dim=1)

        y = y + self.pos_emb

        return y
