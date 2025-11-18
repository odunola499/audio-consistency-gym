from dataclasses import dataclass
from typing import Tuple


@dataclass
class DITModelConfig:
    num_layers: int = 8
    num_heads: int = 8
    head_dim: int = 64
    embed_dim: int = 128
    dropout: int = 0.1
    ff_mult: int = 4
    text_dim: int = 512
    conv_layers: int = 4
    vocab_size: int = 100
    vae_dim: int = 64
    checkpoint_activations: bool = True
    sigma: int = 0.0
    audio_drop_prob: float = 0.3
    cond_drop_prob: float = 0.2
    frac_lengths_mask:Tuple[float] = (0.1, 0.7)
    vae_hf_url: str = "odunola/vibevoice_vae_weights"
