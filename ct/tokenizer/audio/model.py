import copy
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import AcousticTokenizerConfig, SemanticTokenizerConfig
from .modules import TokenizerDecoder, TokenizerEncoder


@dataclass
class EncoderOutput:
    mean: torch.Tensor
    std: Optional[Union[float, Tensor]] = None

    def sample(self, dist_type="fix"):
        if dist_type == "fix":
            x = self.mean + self.std * torch.randn_like(self.mean)
            return x, self.std
        elif dist_type == "gaussian":
            batch_size = self.mean.size(0)
            value = self.std / 0.8
            std = (
                torch.randn(batch_size, device=self.mean.device, dtype=self.mean.dtype)
                * value
            )

            while std.dim() < self.mean.dim():
                std = std.unsqueeze(-1)

            x = self.mean + std * torch.randn_like(self.mean)
            return x, std
        else:
            return self.mean, self.std

    def kl(self):
        target = torch.zeros_like(self.mean)
        return F.mse_loss(self.mean, target, reduction="none")

    def mode(self):
        return self.mean


class AcousticTokenizerModel(nn.Module):
    def __init__(self, config: AcousticTokenizerConfig):
        super().__init__()
        self.config = config

        self.register_buffer("fix_std", torch.tensor(config.fix_std), persistent=False)
        self.std_dist_type = getattr(config, "std_dist_type", "fix")

        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split("-")]
        else:
            encoder_depths = config.encoder_depths

        if config.decoder_depths is not None and isinstance(config.decoder_depths, str):
            decoder_depths = [int(d) for d in config.decoder_depths.split("-")]
        else:
            decoder_depths = list(reversed(encoder_depths))

        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = list(config.encoder_ratios)
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = (
            config.layernorm_elementwise_affine
        )
        encoder_config.mixer_layer = config.mixer_layer
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm

        # Create decoder config
        decoder_config = copy.deepcopy(config)
        decoder_config.dimension = config.vae_dim
        decoder_config.n_filters = config.decoder_n_filters
        decoder_config.ratios = list(config.decoder_ratios)
        decoder_config.depths = decoder_depths
        decoder_config.norm = config.conv_norm
        decoder_config.pad_mode = config.pad_mode
        decoder_config.bias = config.conv_bias
        decoder_config.layernorm_eps = config.layernorm_eps
        decoder_config.layernorm_elementwise_affine = (
            config.layernorm_elementwise_affine
        )
        decoder_config.mixer_layer = config.mixer_layer
        decoder_config.layer_scale_init_value = config.layer_scale_init_value
        decoder_config.disable_last_norm = config.disable_last_norm

        # Initialize encoder and decoder
        self.encoder = TokenizerEncoder(encoder_config)
        self.decoder = TokenizerDecoder(decoder_config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for the model"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def encode(
        self, audio, cache=None, sample_indices=None, use_cache=False, debug=False
    ):
        latents = self.encoder(
            audio,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=False,
            debug=debug,
        )
        return EncoderOutput(mean=latents.permute(0, 2, 1), std=self.fix_std)

    @torch.no_grad()
    def sampling(self, encoder_output, dist_type=None):
        dist_type = dist_type or self.std_dist_type

        if dist_type == "fix":
            return encoder_output.sample(dist_type="fix")
        elif dist_type == "gaussian":
            return encoder_output.sample(dist_type="gaussian")
        else:
            raise ValueError(
                f"Unsupported dist_type: {dist_type}, expected 'fix' or 'gaussian'"
            )

    @torch.no_grad()
    def decode(
        self, latents, cache=None, sample_indices=None, use_cache=False, debug=False
    ):
        """Convert latent representations back to audio"""
        if latents.shape[1] == self.config.vae_dim:
            pass
        else:
            latents = latents.permute(0, 2, 1)

        audio = self.decoder(
            latents,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        return audio

    def forward(
        self, audio, cache=None, sample_indices=None, use_cache=False, debug=False
    ):
        """Full forward pass: encode audio to latents, then decode back to audio"""
        encoder_output = self.encode(
            audio,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        sampled_latents, _ = self.sampling(encoder_output)
        reconstructed = self.decode(
            sampled_latents,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        return reconstructed, sampled_latents


class SemanticTokenizerModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split("-")]
        else:
            encoder_depths = config.encoder_depths

        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = list(config.encoder_ratios)
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = (
            config.layernorm_elementwise_affine
        )
        encoder_config.mixer_layer = config.mixer_layer
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm

        self.encoder = TokenizerEncoder(encoder_config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def encode(
        self, audio, cache=None, sample_indices=None, use_cache=False, debug=False
    ):
        latents = self.encoder(
            audio,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        return EncoderOutput(mean=latents.permute(0, 2, 1))

    @torch.no_grad()
    def sampling(self, encoder_output, dist_type=None):
        return encoder_output.sample(dist_type="none")

    def forward(
        self, audio, cache=None, sample_indices=None, use_cache=False, debug=False
    ):
        encoder_output = self.encode(
            audio,
            cache=cache,
            sample_indices=sample_indices,
            use_cache=use_cache,
            debug=debug,
        )
        sampled_latents, _ = self.sampling(encoder_output, dist_type="none")
        return None, sampled_latents
