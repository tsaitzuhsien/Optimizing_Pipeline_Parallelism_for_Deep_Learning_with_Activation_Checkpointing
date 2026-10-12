"""Standalone ConvNeXt image-classification models.

The public builder functions mirror the no-argument API used by the local
ResNet models while keeping the shared architecture available through
``ConvNeXt`` for focused tests and future variants.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F


__all__ = [
    "ConvNeXt",
    "ConvNeXtBlock",
    "LayerNorm2d",
    "StochasticDepth",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
]


def _validate_nonnegative_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


@torch.fx.wrap
def stochastic_depth(input: Tensor, probability: float, training: bool) -> Tensor:
    """Drop complete residual branches independently for each sample."""
    if probability == 0.0 or not training:
        return input

    survival_rate = 1.0 - probability
    noise_shape = [input.shape[0]] + [1] * (input.ndim - 1)
    noise = torch.empty(noise_shape, dtype=input.dtype, device=input.device)
    noise.bernoulli_(survival_rate).div_(survival_rate)
    return input * noise


class StochasticDepth(nn.Module):
    """Per-sample stochastic depth with expectation-preserving scaling."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("stochastic-depth probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.probability, self.training)

    def extra_repr(self) -> str:
        return f"probability={self.probability}"


class LayerNorm2d(nn.LayerNorm):
    """Layer normalization over the channel dimension of NCHW tensors."""

    def forward(self, input: Tensor) -> Tensor:
        input = input.permute(0, 2, 3, 1)
        input = F.layer_norm(
            input,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return input.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """A ConvNeXt residual block."""

    def __init__(
        self,
        dim: int,
        drop_path_rate: float,
        layer_scale_init_value: float,
        expansion_ratio: int = 4,
    ) -> None:
        super().__init__()
        if (
            not isinstance(expansion_ratio, int)
            or isinstance(expansion_ratio, bool)
            or expansion_ratio <= 0
        ):
            raise ValueError("expansion_ratio must be a positive integer")
        layer_scale_init_value = _validate_nonnegative_real(
            "layer_scale_init_value",
            layer_scale_init_value,
        )
        self.expansion_ratio = expansion_ratio
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion_ratio * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0.0
            else None
        )
        self.drop_path = StochasticDepth(drop_path_rate)

    def forward(self, input: Tensor) -> Tensor:
        residual = input
        output = self.dwconv(input)
        output = output.permute(0, 2, 3, 1)
        output = self.norm(output)
        output = self.pwconv1(output)
        output = self.act(output)
        output = self.pwconv2(output)
        if self.gamma is not None:
            output = self.gamma * output
        output = output.permute(0, 3, 1, 2)
        return residual + self.drop_path(output)


class ConvNeXt(nn.Module):
    """Four-stage ConvNeXt classifier."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        expansion_pattern: Sequence[int] = (4,),
    ) -> None:
        super().__init__()
        depths = self._validate_stages("depths", depths)
        dims = self._validate_stages("dims", dims)
        try:
            expansion_pattern = tuple(expansion_pattern)
        except TypeError as error:
            raise TypeError(
                "expansion_pattern must be a sequence of integers"
            ) from error
        if not expansion_pattern:
            raise ValueError("expansion_pattern must not be empty")
        if any(
            not isinstance(ratio, int) or isinstance(ratio, bool)
            for ratio in expansion_pattern
        ):
            raise TypeError("expansion_pattern values must be integers")
        if any(ratio <= 0 for ratio in expansion_pattern):
            raise ValueError("expansion_pattern values must be positive")
        if not isinstance(in_channels, int) or isinstance(in_channels, bool) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        layer_scale_init_value = _validate_nonnegative_real(
            "layer_scale_init_value",
            layer_scale_init_value,
        )
        head_init_scale = _validate_nonnegative_real(
            "head_init_scale",
            head_init_scale,
        )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = float(drop_path_rate)
        self.expansion_pattern = expansion_pattern

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0], eps=1e-6),
            )
        )
        for stage_index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[stage_index], eps=1e-6),
                    nn.Conv2d(
                        dims[stage_index],
                        dims[stage_index + 1],
                        kernel_size=2,
                        stride=2,
                    ),
                )
            )

        drop_path_rates = torch.linspace(
            0.0,
            drop_path_rate,
            sum(depths),
        ).tolist()
        self.stages = nn.ModuleList()
        block_index = 0
        for stage_index, depth in enumerate(depths):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        dims[stage_index],
                        drop_path_rates[block_index],
                        layer_scale_init_value,
                        expansion_ratio=expansion_pattern[
                            block_index % len(expansion_pattern)
                        ],
                    )
                )
                block_index += 1
            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    @staticmethod
    def _validate_stages(name: str, values: Sequence[int]) -> tuple[int, int, int, int]:
        try:
            values = tuple(values)
        except TypeError as error:
            raise TypeError(f"{name} must be a sequence of four integers") from error
        if len(values) != 4:
            raise ValueError(f"{name} must contain exactly four values")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in values
        ):
            raise ValueError(f"{name} values must be positive integers")
        return values

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, input: Tensor) -> Tensor:
        output = input
        for stage_index in range(4):
            output = self.downsample_layers[stage_index](output)
            output = self.stages[stage_index](output)
        output = output.mean(dim=(-2, -1))
        return self.norm(output)

    def forward(self, input: Tensor) -> Tensor:
        return self.head(self.forward_features(input))


def convnext_tiny(
    expansion_pattern: Sequence[int] = (4,),
) -> ConvNeXt:
    """Build a canonical ConvNeXt-Tiny classifier."""
    return ConvNeXt(
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.1,
        expansion_pattern=expansion_pattern,
    )

def convnext_mini(
    expansion_pattern: Sequence[int] = (4,),
) -> ConvNeXt:
    """Build a canonical ConvNeXt-Small classifier."""
    return ConvNeXt(
        depths=(3, 3, 18, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.4,
        expansion_pattern=expansion_pattern,
    )


def convnext_small(
    expansion_pattern: Sequence[int] = (4,),
) -> ConvNeXt:
    """Build a canonical ConvNeXt-Small classifier."""
    return ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.4,
        expansion_pattern=expansion_pattern,
    )

def convnext_base(
    expansion_pattern: Sequence[int] = (4,),
) -> ConvNeXt:
    """Build a canonical ConvNeXt-Base classifier."""
    return ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        drop_path_rate=0.4,
        expansion_pattern=expansion_pattern,
    )

def convnext_medium(
    expansion_pattern: Sequence[int] = (4,),
) -> ConvNeXt:
    """Build a canonical ConvNeXt-Small classifier."""
    return ConvNeXt(
        depths=(3, 3, 36, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.4,
        expansion_pattern=expansion_pattern,
    )


def convnext_mega(
    expansion_pattern: Sequence[int] = (4,),
) -> ConvNeXt:
    """Build a canonical ConvNeXt-mega classifier."""
    return ConvNeXt(
        depths=(3, 3, 45, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.4,
        expansion_pattern=expansion_pattern,
    )

def convnext_ultra(
    expansion_pattern: Sequence[int] = (4,),
) -> ConvNeXt:
    """Build a canonical ConvNeXt-ultra classifier."""
    return ConvNeXt(
        depths=(3, 3, 54, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.4,
        expansion_pattern=expansion_pattern,
    )

def convnext_large(
    expansion_pattern: Sequence[int] = (4,),
) -> ConvNeXt:
    """Build a canonical ConvNeXt-Large classifier."""
    return ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        drop_path_rate=0.4,
        expansion_pattern=expansion_pattern,
    )

def convnext_xlarge(
    expansion_pattern: Sequence[int] = (4,),
) -> ConvNeXt:
    """Build a canonical ConvNeXt-XLarge classifier."""
    return ConvNeXt(
        depths=(3, 3, 27, 3),
        dims=(256, 512, 1024, 2048),
        drop_path_rate=0.4,
        expansion_pattern=expansion_pattern,
    )
