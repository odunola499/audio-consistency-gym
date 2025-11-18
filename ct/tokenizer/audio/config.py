from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class AcousticTokenizerConfig:
    channels: int = 1
    corpus_normalize: float = 0.0
    causal: bool = True
    vae_dim: int = 64
    fix_std: float = 0.5
    std_dist_type: str = "gaussian"

    mixer_layer: str = "depthwise_conv"
    conv_norm: str = "none"
    pad_mode: str = "constant"
    disable_last_norm: bool = True
    layernorm: str = "RMSNorm"
    layernorm_eps: float = 1e-5
    layernorm_elementwise_affine: bool = True
    conv_bias: bool = True
    layer_scale_init_value: float = 1e-6
    weight_init_value: float = 1e-2
    # encoder specific
    encoder_n_filters: int = 32
    encoder_ratios: Optional[List[int]] = (8, 5, 5, 4, 2, 2)
    encoder_depths: str = "3-3-3-3-3-3-8"
    # decoder specific
    decoder_n_filters: int = 32
    decoder_ratios: Optional[List[int]] = (8, 5, 5, 4, 2, 2)  # if None same as encoder
    decoder_depths: Optional[str] = None


@dataclass
class SemanticTokenizerConfig:
    channels: int = 1
    corpus_normalize: float = 0.0
    causal: bool = True
    vae_dim: int = 128
    fix_std: float = 0.5
    std_dist_type: str = "gaussian"

    mixer_layer: str = "depthwise_conv"
    conv_norm: str = "none"
    pad_mode: str = "constant"
    disable_last_norm: bool = True
    layernorm: str = "RMSNorm"
    layernorm_eps: float = 1e-5
    layernorm_elementwise_affine: bool = True
    conv_bias: bool = True
    layer_scale_init_value: float = 1e-6
    weight_init_value: float = 1e-2
    # encoder specific
    encoder_n_filters: int = 32
    encoder_ratios: Optional[List[int]] = (8, 5, 5, 4, 2, 2)
    encoder_depths: str = "3-3-3-3-3-3-8"
