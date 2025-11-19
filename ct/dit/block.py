from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ct.dit.utils import apply_rotary_pos_emb

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    FLASH_ATTN_AVAILABLE = True
except:
    FLASH_ATTN_AVAILABLE = False


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            emb, 6, dim=-1
        )

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class FeedForward(nn.Module):
    def __init__(
        self, dim, dim_out=None, mult=4, dropout=0.0, approximate="none", **kwargs
    ):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        up_proj = nn.Linear(dim, dim * mult)
        dropout = nn.Dropout(dropout)
        down_proj = nn.Linear(dim * mult, dim_out)

        self.ff = nn.Sequential(up_proj, activation, dropout, down_proj)

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8, head_dim: int = 64, dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)

        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = inner_dim

    def forward(
        self,
        x,
        mask=None,
        rope=None,
        attn_impl: Literal["flash_attention_2", "sdpa"] = "sdpa",
    ):
        batch_size, seq_len, _ = x.shape
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.num_heads

        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        query = self.q_norm(query)
        key = self.q_norm(key)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            )
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if attn_impl == "flash_attention_2":
            x = self.flash_attn_forward(query, key, value, mask)
        elif attn_impl == "sdpa":
            x = self.sdpa_forward(query, key, value, mask)

        x = x.to(query.dtype)
        x = self.to_out(x)
        x = self.dropout(x)
        return x

    def flash_attn_forward(self, query, key, value, mask):
        query, key, value = (
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        if mask is not None:
            query, indices, q_cu_seqlens, q_max_seqlen_in_batch, _ = unpad_input(
                query, mask
            )
            key, _, k_cu_seqlens, k_max_seqlen_in_batch, _ = unpad_input(key, mask)
            value, _, _, _, _ = unpad_input(value, mask)
            x = flash_attn_varlen_func(
                query,
                key,
                value,
                q_cu_seqlens,
                k_cu_seqlens,
                q_max_seqlen_in_batch,
                k_max_seqlen_in_batch,
            )
            x = pad_input(x, indices, query.shape[0], q_max_seqlen_in_batch)
            x = x.reshape(query.shape[0], -1, self.inner_dim)
        else:
            x = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
            x = x.reshape(query.shape[0], -1, self.inner_dim)

        return x

    def sdpa_forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(
                batch_size, self.num_heads, query.shape[-2], key.shape[-2]
            )
        else:
            attn_mask = None
        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)

        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        head_dim,
        ff_mult=4,
        dropout=0.1,
    ):
        super().__init__()
        self.norm = AdaLayerNorm(dim)
        self.attn = Attention(
            dim=dim, head_dim=head_dim, num_heads=num_heads, dropout=dropout
        )
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = FeedForward(dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ffn_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ffn(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


class SemanticConnector(nn.Module):
    def __init__(self, vae_dim, dim):
        super().__init__()
        self.fc1 = nn.Linear(vae_dim, dim)
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, features):
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    ffn = FeedForward(dim=8)
    tensor = torch.randn(2, 11, 8)
    output = ffn(tensor)
    print(output.shape)

    attn = Attention(dim=16, num_heads=4, head_dim=4)
    tensor = torch.randn(2, 11, 16)
    output = attn(tensor)
    print(output.shape)

    block = DiTBlock(dim=16, num_heads=4, head_dim=4)
    tensor = torch.randn(2, 11, 16)
    t = torch.randn(2, 16)
    output = block(tensor, t=t)
    print(output.shape)
