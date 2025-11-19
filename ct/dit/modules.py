import math

import torch
from torch import Tensor, nn

from ct.dit.block import AdaLayerNorm_Final, DiTBlock
from ct.dit.text import TextEmbedding
from ct.dit.utils import RotaryEmbedding


def sinusoids(x, dim=256, scale=1000):
    device = x.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
    emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: Tensor, mask: bool = None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out


class TimeStepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )


class InputEmbedding(nn.Module):
    def __init__(self, vae_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(vae_dim + vae_dim + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, drop_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)

        x = torch.cat((x, cond, text_embed), dim=-1)
        x = self.proj(x)
        x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        num_layers=8,
        num_heads=8,
        head_dim=64,
        dropout=0.1,
        ff_mult=4,
        vae_dim=100,
        text_dim=512,
        conv_layers=4,
        checkpoint_activations=True,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.text_embed = TextEmbedding(
            vocab_size,
            text_dim,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(vae_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(head_dim)

        self.dim = dim
        self.num_layers = num_layers

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, vae_dim)

        self.initialize_weights()
        self.checkpoint_activations = checkpoint_activations

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.to_q.weight)
            nn.init.xavier_uniform_(block.attn.to_k.weight)
            nn.init.xavier_uniform_(block.attn.to_v.weight)
            nn.init.zeros_(block.attn.to_q.bias)
            nn.init.zeros_(block.attn.to_k.bias)
            nn.init.zeros_(block.attn.to_v.bias)

            nn.init.constant_(block.attn.to_out.weight, 0.0)
            nn.init.constant_(block.attn.to_out.bias, 0.0)

            for layer in block.ffn.ff:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

            if hasattr(block.norm, "linear"):
                nn.init.xavier_uniform_(block.norm.linear.weight)
                nn.init.zeros_(block.norm.linear.bias)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
        self,
        x,
        cond,
        text,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: bool = None,
    ):
        seq_len = x.shape[1]
        if cache:
            if drop_text:
                text_embed = self.text_embed(
                    text, seq_len, drop_text=True, audio_mask=audio_mask
                )
                self.text_uncond = text_embed
            else:
                text_embed = self.text_embed(
                    text, seq_len, drop_text=False, audio_mask=audio_mask
                )
                self.text_cond = text_embed
        else:
            text_embed = self.text_embed(
                text, seq_len, drop_text=drop_text, audio_mask=audio_mask
            )

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def embed_time(self, time: Tensor, batch: int):
        if time.ndim == 0:
            time = time.repeat(batch)
        time = sinusoids(time)
        time = self.time_embed(time)
        return time

    def forward(
        self,
        noised_input: Tensor,
        masked_input: Tensor,
        text: Tensor,
        time: Tensor,
        mask=None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ):
        batch, seq_len = noised_input.shape[:2]
        time = self.embed_time(time, batch)
        if cfg_infer:
            x_cond = self.get_input_embed(
                noised_input,
                masked_input,
                text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask,
            )
            x_uncond = self.get_input_embed(
                noised_input,
                masked_input,
                text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask,
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((time, time), dim=0)
            mask = torch.concat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                noised_input,
                masked_input,
                text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask,
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False
                )
            else:
                x = block(x, t, mask, rope)

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
