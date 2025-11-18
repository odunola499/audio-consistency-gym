import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .cache import StreamingCache
from .norm import ConvLayerNorm

CONV_NORMALIZATIONS = frozenset(
    [
        "none",
        "weight_norm",
        "spectral_norm",
        "time_layer_norm",
        "layer_norm",
        "time_group_norm",
    ]
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return nn.utils.weight_norm(module)
    elif norm == "spectral_norm":
        return nn.utils.spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def pad1d(
    x: torch.Tensor, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0
):
    """Pad 1D input with handling for small inputs in reflect mode"""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert (
        padding_left >= 0 and padding_right >= 0
    ), f"{padding_left, padding_right} is invalid padding"
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """Calculate extra padding needed for convolution to have the same output length"""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv"""

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv"""

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs=None,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}

        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.context_size = (kernel_size - 1) * dilation - (stride - 1)
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)

        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconv1d_{id(self)}"
        return self._layer_id

    def forward(
        self,
        x: torch.Tensor,
        cache: StreamingCache,
        sample_indices=None,
        use_cache: bool = False,
        debug: bool = False,
    ):

        B, C, T = x.shape
        if not use_cache or cache is None:
            return self._forward_non_streaming(x, debug=debug)
        assert self.causal, "Streaming mode is only supported for causal convolutions"
        assert (
            sample_indices is not None
        ), "sample_indices must be provided for streaming mode"
        assert len(sample_indices) == B, "sample_indices must match batch size"

        return self._forward_streaming(x, cache, sample_indices, debug)

    def _forward_streaming(
        self,
        x: torch.Tensor,
        cache: StreamingCache,
        sample_indices: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        """Streaming forward pass with cache operations kept separate from compiled code"""
        B, C, T = x.shape

        # Cache operations (not compiled)
        cached_states = cache.get(self.layer_id, sample_indices)

        if cached_states is None:
            # First chunk - initialize with zeros for context
            if self.context_size > 0:
                cached_states = torch.zeros(
                    B, C, self.context_size, device=x.device, dtype=x.dtype
                )
                if debug:
                    print(
                        f"[DEBUG] Initialized cache with shape: {cached_states.shape}, context_size={self.context_size}"
                    )
            else:
                cached_states = torch.zeros(B, C, 0, device=x.device, dtype=x.dtype)
                if debug:
                    print(f"[DEBUG] No context needed (kernel_size=stride)")

        # Concatenate cached states with input
        if cached_states.shape[2] > 0:
            input_with_context = torch.cat([cached_states, x], dim=2)
        else:
            input_with_context = x

        if debug:
            print(
                f"[DEBUG] Input shape: {x.shape}, Cache shape: {cached_states.shape}, Combined: {input_with_context.shape}"
            )

        # Apply convolution directly - no extra padding in streaming mode
        # The conv layer will handle its own padding internally
        output = self.conv(input_with_context)

        if debug:
            print(f"[DEBUG] Output shape: {output.shape}")

        # Update cache for next chunk
        if self.context_size > 0:
            # Calculate how many samples to keep
            total_input_length = input_with_context.shape[2]

            # Keep the last context_size samples
            if total_input_length >= self.context_size:
                new_cache_start = total_input_length - self.context_size
                new_cache = input_with_context[:, :, new_cache_start:]
            else:
                # If we have less than context_size samples, keep everything
                new_cache = input_with_context

            if debug:
                print(f"[DEBUG] New cache shape: {new_cache.shape}")

            cache.set(self.layer_id, sample_indices, new_cache)

        return output

    def _forward_non_streaming(self, x: torch.Tensor, debug: bool = False):
        B, C, T = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        dilation = self.dilation
        padding_total = self.padding_total

        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )

        if debug:
            print(
                f"[DEBUG NON-STREAMING] Input shape: {x.shape}, padding_total={padding_total}, extra_padding={extra_padding}"
            )

        if self.causal:
            # Left padding for causal
            if self.pad_mode == "constant":
                x = pad1d(
                    x, (padding_total, extra_padding), mode=self.pad_mode, value=0
                )
            else:
                x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Symmetric padding for non-causal
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )

        if debug:
            print(f"[DEBUG NON-STREAMING] After padding: {x.shape}")

        output = self.conv(x)

        if debug:
            print(f"[DEBUG NON-STREAMING] Output shape: {output.shape}")

        return output


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in handling of asymmetric or causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: Dict[str, Any] = {},
        bias: bool = True,
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

        # Store configuration
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # For transposed convolution, padding calculation is different
        self.padding_total = kernel_size - stride

        # For streaming, we need to keep track of input history
        # Transposed conv needs to see multiple input samples to produce correct output
        self.context_size = kernel_size - 1

        # Create a unique layer ID for cache management
        self._layer_id = None

    @property
    def layer_id(self):
        if self._layer_id is None:
            self._layer_id = f"sconvtr1d_{id(self)}"
        return self._layer_id

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[StreamingCache] = None,
        sample_indices: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional streaming support via cache.
        """
        B, C, T = x.shape

        # Non-streaming mode
        if not use_cache or cache is None:
            return self._forward_non_streaming(x, debug=debug)

        # Streaming mode
        assert (
            sample_indices is not None
        ), "sample_indices must be provided for streaming mode"
        assert len(sample_indices) == B, "sample_indices must match batch size"

        return self._forward_streaming(x, cache, sample_indices, debug)

    def _forward_streaming(
        self,
        x: torch.Tensor,
        cache: StreamingCache,
        sample_indices: torch.Tensor,
        debug: bool = False,
    ) -> torch.Tensor:
        """Streaming forward pass with cache operations kept separate from compiled code"""
        B, C, T = x.shape

        # Cache operations (not compiled)
        cached_input = cache.get(self.layer_id, sample_indices)

        if cached_input is None:
            # First chunk - no history yet
            cached_input = torch.zeros(B, C, 0, device=x.device, dtype=x.dtype)
            if debug:
                print(f"[DEBUG] Initialized empty cache for transposed conv")

        # Concatenate cached input with new input
        full_input = torch.cat([cached_input, x], dim=2)

        if debug:
            print(
                f"[DEBUG] Input shape: {x.shape}, Cache shape: {cached_input.shape}, Combined: {full_input.shape}"
            )

        # First chunk or debug mode - use uncompiled version
        full_output = self.convtr(full_input)

        if debug:
            print(f"[DEBUG] Full transposed conv output shape: {full_output.shape}")

        # Calculate padding to remove
        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
            padding_left = self.padding_total - padding_right
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right

        # Remove padding
        if padding_left + padding_right > 0:
            full_output = unpad1d(full_output, (padding_left, padding_right))

        if debug:
            print(f"[DEBUG] After unpadding: {full_output.shape}")

        # Determine which part of the output corresponds to the new input
        if cached_input.shape[2] == 0:
            # First chunk - return all output
            output = full_output
        else:
            # Subsequent chunks - return only the new output
            expected_new_output = T * self.stride

            # Take the last expected_new_output samples
            if full_output.shape[2] >= expected_new_output:
                output = full_output[:, :, -expected_new_output:]
            else:
                output = full_output

        if debug:
            print(f"[DEBUG] Final streaming output shape: {output.shape}")

        # Update cache
        if full_input.shape[2] > self.context_size:
            new_cache = full_input[:, :, -self.context_size :]
        else:
            new_cache = full_input

        if debug:
            print(f"[DEBUG] New cache shape: {new_cache.shape}")

        cache.set(self.layer_id, sample_indices, new_cache)

        return output

    def _forward_non_streaming(
        self, x: torch.Tensor, debug: bool = False
    ) -> torch.Tensor:
        """Standard forward pass without streaming"""
        if debug:
            print(f"[DEBUG NON-STREAMING] Input shape: {x.shape}")

        # Apply transposed convolution
        y = self.convtr(x)

        if debug:
            print(f"[DEBUG NON-STREAMING] After transposed conv: {y.shape}")

        # Calculate and remove padding
        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
            padding_left = self.padding_total - padding_right
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right

        if padding_left + padding_right > 0:
            y = unpad1d(y, (padding_left, padding_right))

        if debug:
            print(f"[DEBUG NON-STREAMING] Final output shape: {y.shape}")

        return y


class Convlayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        pad_mode="zeros",
        norm="weight_norm",
        causal=True,
    ):
        super().__init__()
        self.conv = SConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            pad_mode=pad_mode,
            norm=norm,
            causal=causal,
        )

    def forward(self, x):
        return self.conv(x)


class StreamingConvlayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        pad_mode="zeros",
        norm="weight_norm",
        causal=True,
    ):
        super().__init__()
        self.conv = SConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            pad_mode=pad_mode,
            norm=norm,
            causal=causal,
        )

    def forward(
        self, x, cache: StreamingCache, sample_indices=None, use_cache: bool = True
    ):
        return self.conv(x, cache, sample_indices, use_cache=use_cache)
