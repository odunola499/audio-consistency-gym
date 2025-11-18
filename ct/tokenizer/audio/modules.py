from functools import partial
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .conv import (
    Convlayer,
    ConvLayerNorm,
    SConv1d,
    SConvTranspose1d,
    StreamingConvlayer,
)
from .norm import ConvLayerNorm, ConvRMSNorm


class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(self.embed_dim, ffn_dim, bias=bias)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ffn_dim, self.embed_dim, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Block1D(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        drop_path=0.0,
        mixer_layer="conv",
        layer_scale_init_value=1e-6,
        **kwargs,
    ):
        super().__init__()

        if kwargs.get("layernorm", "LN") == "LN":
            self.norm = ConvLayerNorm(dim, eps=kwargs.get("eps", 1e-6))
            self.ffn_norm = ConvLayerNorm(dim, eps=kwargs.get("eps", 1e-6))
        elif kwargs.get("layernorm", "RMSNorm") == "RMSNorm":
            self.norm = ConvRMSNorm(dim, eps=kwargs.get("eps", 1e-6))
            self.ffn_norm = ConvRMSNorm(dim, eps=kwargs.get("eps", 1e-6))

        if mixer_layer == "conv":
            self.mixer = Convlayer(
                dim,
                dim,
                groups=kwargs.get("groups", 1),
                kernel_size=kernel_size,
                pad_mode=kwargs.get("pad_mode", "reflect"),
                norm=kwargs.get("norm", "none"),
                causal=kwargs.get("causal", True),
                bias=kwargs.get("bias", True),
            )
        elif mixer_layer == "depthwise_conv":
            self.mixer = Convlayer(
                dim,
                dim,
                groups=dim,
                kernel_size=kernel_size,
                pad_mode=kwargs.get("pad_mode", "reflect"),
                norm=kwargs.get("norm", "none"),
                causal=kwargs.get("causal", True),
                bias=kwargs.get("bias", True),
            )
        else:
            raise ValueError(f"Unsupported mixer layer: {mixer_layer}")

        self.ffn = FFN(
            dim,
            kwargs.get("ffn_expansion", 4) * dim,
            bias=kwargs.get("bias", False),
        )
        self.drop_path = (
            nn.Identity() if drop_path <= 0.0 else nn.modules.DropPath(drop_path)
        )

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.ffn_gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma = None
            self.ffn_gamma = None

    def forward(self, x, cache=None, use_cache=True):
        # mixer
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        # ffn
        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        return x


class TokenizerEncoder(nn.Module):
    """
    Encoder component for the VibeVoice tokenizer that converts audio to latent representations.

    Args:
        config: Configuration object with model parameters
    """

    def __init__(self, config):
        super().__init__()

        # Extract parameters from config
        self.channels = config.channels
        self.dimension = config.dimension
        self.n_filters = config.n_filters
        self.ratios = list(reversed(config.ratios))

        self.depths = config.depths
        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal

        # Additional config parameters with defaults
        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        layernorm_elementwise_affine = getattr(
            config, "layernorm_elementwise_affine", True
        )
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)

        # determine the norm type based on layernorm
        if layernorm == "LN":
            norm_type = ConvLayerNorm
        elif layernorm == "RMSNorm":
            norm_type = partial(
                ConvRMSNorm, elementwise_affine=layernorm_elementwise_affine
            )
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")

        # stem and intermediate downsampling conv layers
        stem = nn.Sequential(
            SConv1d(
                self.channels,
                self.n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=self.causal,
                pad_mode=pad_mode,
                bias=bias,
            ),
        )

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2**i)
            out_ch = self.n_filters * (2 ** (i + 1))
            downsample_layer = nn.Sequential(
                SConv1d(
                    in_ch,
                    out_ch,
                    kernel_size=self.ratios[i] * 2,
                    stride=self.ratios[i],
                    causal=self.causal,
                    pad_mode=pad_mode,
                    norm=norm,
                    bias=bias,
                )
            )
            self.downsample_layers.append(downsample_layer)

        # configure the transformer blocks
        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.stages = nn.ModuleList()
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0

        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2**i)
            stage = nn.Sequential(
                *[
                    layer_type(dim=in_ch, drop_path=dp_rates[cur + j])
                    for j in range(self.depths[i])
                ]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(
            in_ch,
            self.dimension,
            kernel_size=last_kernel_size,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
        )

    def forward_features(
        self, x, cache=None, sample_indices=None, use_cache=False, debug=False
    ):
        for i in range(len(self.depths)):
            # Apply downsampling
            for layer in self.downsample_layers[i]:
                if isinstance(layer, SConv1d):
                    x = layer(
                        x,
                        cache=cache,
                        sample_indices=sample_indices,
                        use_cache=use_cache,
                        debug=debug,
                    )
                else:
                    x = layer(x)

            # Apply stage (Block1D contains Convlayer which contains SConv1d)
            for block in self.stages[i]:
                if (
                    hasattr(block, "mixer")
                    and hasattr(block.mixer, "conv")
                    and isinstance(block.mixer.conv, SConv1d)
                ):
                    # Block1D forward with cache support
                    residual = x
                    x = block.norm(x)
                    x = block.mixer.conv(
                        x,
                        cache=cache,
                        sample_indices=sample_indices,
                        use_cache=use_cache,
                        debug=debug,
                    )
                    if block.gamma is not None:
                        x = x * block.gamma.unsqueeze(-1)
                    x = residual + x

                    # FFN part
                    residual = x
                    x = block.ffn_norm(x)
                    x = x.permute(0, 2, 1)
                    x = block.ffn(x)
                    x = x.permute(0, 2, 1)
                    if block.ffn_gamma is not None:
                        x = x * block.ffn_gamma.unsqueeze(-1)
                    x = residual + x
                else:
                    x = block(x)

        return self.norm(x)

    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        x = self.forward_features(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        x = self.head(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        return x


class TokenizerDecoder(nn.Module):
    """
    Decoder component for the VibeVoice tokenizer that converts latent representations back to audio.

    Args:
        config: Configuration object with model parameters
    """

    def __init__(self, config):
        super().__init__()

        # Extract parameters from config
        self.dimension = config.dimension
        self.channels = config.channels
        self.n_filters = config.n_filters
        self.ratios = config.ratios

        # IMPORTANT CHANGE: Don't reverse depths again since they're already reversed in VibeVoiceAcousticTokenizerModel
        self.depths = config.depths  # Changed from list(reversed(config.depths))

        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal

        # Additional config parameters with defaults
        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        trim_right_ratio = getattr(config, "trim_right_ratio", 1.0)
        layernorm_elementwise_affine = getattr(
            config, "layernorm_elementwise_affine", True
        )
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)

        # determine the norm type based on layernorm
        if layernorm == "LN":
            norm_type = ConvLayerNorm
        elif layernorm == "RMSNorm":
            norm_type = partial(
                ConvRMSNorm, elementwise_affine=layernorm_elementwise_affine
            )
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")

        # stem and upsampling layers
        stem = nn.Sequential(
            SConv1d(
                self.dimension,
                self.n_filters * 2 ** (len(self.depths) - 1),
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=self.causal,
                pad_mode=pad_mode,
                bias=bias,
            ),
        )

        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            out_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i - 1))
            upsample_layer = nn.Sequential(
                SConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=self.ratios[i] * 2,
                    stride=self.ratios[i],
                    norm=norm,
                    norm_kwargs=norm_params,
                    bias=bias,
                    causal=self.causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            )
            self.upsample_layers.append(upsample_layer)

        # configure transformer blocks
        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.stages = nn.ModuleList()
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0

        # Create stages in the same order as the original model
        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            stage = nn.Sequential(
                *[
                    layer_type(dim=in_ch, drop_path=dp_rates[cur + j])
                    for j in range(self.depths[i])
                ]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(
            in_ch,
            self.channels,
            kernel_size=last_kernel_size,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
        )

    def forward_features(
        self, x, cache=None, sample_indices=None, use_cache=False, debug=False
    ):
        for i in range(len(self.depths)):
            # Apply upsampling
            for layer in self.upsample_layers[i]:
                if isinstance(layer, (SConv1d, SConvTranspose1d)):
                    x = layer(
                        x,
                        cache=cache,
                        sample_indices=sample_indices,
                        use_cache=use_cache,
                        debug=debug,
                    )
                else:
                    x = layer(x)

            # Apply stage (Block1D contains Convlayer which contains SConv1d)
            for block in self.stages[i]:
                if (
                    hasattr(block, "mixer")
                    and hasattr(block.mixer, "conv")
                    and isinstance(block.mixer.conv, SConv1d)
                ):
                    # Block1D forward with cache support
                    residual = x
                    x = block.norm(x)
                    x = block.mixer.conv(
                        x,
                        cache=cache,
                        sample_indices=sample_indices,
                        use_cache=use_cache,
                        debug=debug,
                    )
                    if block.gamma is not None:
                        x = x * block.gamma.unsqueeze(-1)
                    x = residual + x

                    # FFN part
                    residual = x
                    x = block.ffn_norm(x)
                    x = x.permute(0, 2, 1)
                    x = block.ffn(x)
                    x = x.permute(0, 2, 1)
                    if block.ffn_gamma is not None:
                        x = x * block.ffn_gamma.unsqueeze(-1)
                    x = residual + x
                else:
                    x = block(x)

        return self.norm(x)

    def forward(self, x, cache=None, sample_indices=None, use_cache=False, debug=False):
        x = self.forward_features(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        x = self.head(
            x,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        return x
