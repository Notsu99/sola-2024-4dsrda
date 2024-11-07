from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = getLogger()


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        n_head: int,
        dropout: float,
    ):
        super().__init__()

        self.n_head = n_head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_head
        self.sqrt_dh = self.head_dim**0.5

        #
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        self.w = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        # x dim is (B, N, D)
        batch_size, num_patch, _ = x.shape

        # embedding
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # split into multiple n_heads
        # (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.n_head, self.head_dim)
        k = k.view(batch_size, num_patch, self.n_head, self.head_dim)
        v = v.view(batch_size, num_patch, self.n_head, self.head_dim)

        # (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) / self.sqrt_dh

        attn = F.softmax(dots, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.einsum("bhij,bhjd->bhid", attn, v)

        # (B, h, N, D//h) -> (B, N, h, D//h)
        y = y.transpose(1, 2)

        # (B, N, h, D//h) -> (B, N, D)
        y = y.reshape(batch_size, num_patch, self.emb_dim)

        y = self.w(y)
        return y


class VitEncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        n_head: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(emb_dim)

        self.mheads_attn = MultiHeadSelfAttentionBlock(
            emb_dim=emb_dim,
            n_head=n_head,
            dropout=dropout,
        )

        self.layer_norm2 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        y = self.mheads_attn(self.layer_norm1(x)) + x

        y = self.mlp(self.layer_norm2(y)) + y
        return y


class VitEncoderBlocks(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        n_head: int,
        hidden_dim: int,
        dropout: float,
        num_block: int,
    ):
        super().__init__()

        assert num_block >= 1

        blocks = []
        for i in range(num_block):
            blocks.append(
                VitEncoderBlock(
                    emb_dim=emb_dim,
                    n_head=n_head,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
