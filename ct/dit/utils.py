from typing import Optional

import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn

from ct.tokenizer.audio.model import (
    AcousticTokenizerConfig,
    AcousticTokenizerModel,
    SemanticTokenizerConfig,
    SemanticTokenizerModel,
)


def lens_to_mask(t: Tensor, length: Optional[int] = None):
    if not length:
        length = t.amax()
    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len, start, end):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len, frac_lengths: Tensor):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths
    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).log().clamp(min=0)
    end = start + lengths
    return mask_from_start_end_indices(seq_len, start, end)


def rotate_half(x: Tensor):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@torch.autocast("cuda", enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")

    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.concat((t, t_unrotated), dim=-1)

    return out.type(orig_dtype)


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


def manual_euler(func, x0: Tensor, t: Tensor):
    x = x0.clone()  # Initial, currently would be just noise
    xs = [x0]

    for i in range(len(t) - 1):
        t_i = t[i]
        t_next = t[i + 1]
        dt = t_next - t_i

        dx = func(t_i, x)
        x = x + dt * dx
        xs.append(x)
    return xs


def load_vae_models(repo_id="odunola/vibevoice_vae_weights"):
    acoustic_config = AcousticTokenizerConfig()
    semantic_config = SemanticTokenizerConfig()

    acoustic_model = AcousticTokenizerModel(acoustic_config)
    semantic_model = SemanticTokenizerModel(semantic_config)

    # acoustic_path = hf_hub_download(repo_id=repo_id, filename="acoustic.safetensors")
    # semantic_path = hf_hub_download(repo_id=repo_id, filename="semantic.safetensors")

    # weights = load_file(acoustic_path)
    # acoustic_model.load_state_dict(weights)

    # weights = load_file(semantic_path)
    # semantic_model.load_state_dict(weights)
    return acoustic_model, semantic_model


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos=False,
        scale_base=512,
        interpolation_factor=1.0,
        base=10000,
        base_rescale_factor=1.0,
    ):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device=device)
        return self.forward(t)

    @torch.autocast("cuda", enabled=False)
    def forward(self, t, offset=0):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = rearrange(t, "n -> 1 n")

        freqs = (
            torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq)
            / self.interpolation_factor
        )
        freqs = torch.stack((freqs, freqs), dim=-1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")

        if not self.scale:
            return freqs, 1.0

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "... n -> ... n 1")
        scale = torch.stack((scale, scale), dim=-1)
        scale = rearrange(scale, "... d r -> ... (d r)")

        return freqs, scale


if __name__ == "__main__":
    seq_len = torch.arange(0, 10).expand(3, 1)
    start = 2
    end = 7
    output = mask_from_start_end_indices(seq_len, start, end)
    print(output.shape)
