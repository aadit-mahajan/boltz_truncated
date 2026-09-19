"""Fusing a stack of LayerNorm -> Linear pairs that all read the same tensor.

``DiffusionConditioning`` builds the diffusion module's attention biases by
running 24 (token transformer) + 3 (atom encoder) + 3 (atom decoder) independent
``LayerNorm -> Linear(bias=False)`` pairs over one pair representation and
concatenating the results. Each pair reads the whole ``[B, N, N, C]`` tensor, so
the stack costs 30 passes over the largest tensor in the model to produce a few
hundred channels.

The affine part of a LayerNorm is a per-channel scale, and the Linear that
follows is a matrix product, so the two can be folded into each other::

    Linear_i(LayerNorm_i(x)) = (x_hat * w_i + b_i) @ W_i.T
                             = x_hat @ (W_i * w_i).T + (W_i @ b_i)

where ``x_hat`` is the affine-free normalization, which is the *same tensor* for
every layer in the stack. One normalization and one GEMM against the stacked
weights therefore reproduce the concatenated output.

Numerics: the products ``W_i * w_i`` are rounded once at build time instead of
being applied to the activations, and the GEMM sums in its own order, so this is
a ``fast``-class lever, not a bitwise one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class FusionUnsupported(TypeError):
    """The module stack is not a uniform LayerNorm -> Linear(bias=False) stack."""


class FusedNormLinearStack:
    """Folded weights for one such stack, rebuilt per (device, dtype)."""

    __slots__ = ("normalized_shape", "eps", "weight", "bias", "out_per_layer")

    def __init__(self, layers: nn.ModuleList) -> None:
        norms, linears = [], []
        for layer in layers:
            if not isinstance(layer, nn.Sequential) or len(layer) != 2:
                raise FusionUnsupported("expected Sequential(LayerNorm, Linear)")
            norm, linear = layer[0], layer[1]
            if not isinstance(norm, nn.LayerNorm) or not isinstance(linear, nn.Linear):
                raise FusionUnsupported("expected Sequential(LayerNorm, Linear)")
            if norm.weight is None or norm.bias is None:
                raise FusionUnsupported("LayerNorm without affine parameters")
            if linear.bias is not None:
                raise FusionUnsupported("Linear with a bias is not folded here")
            norms.append(norm)
            linears.append(linear)
        if not norms:
            raise FusionUnsupported("empty stack")

        self.normalized_shape = tuple(norms[0].normalized_shape)
        self.eps = norms[0].eps
        for norm in norms[1:]:
            if tuple(norm.normalized_shape) != self.normalized_shape or norm.eps != self.eps:
                raise FusionUnsupported("stack layers normalize differently")
        outs = {linear.out_features for linear in linears}
        if len(outs) != 1:
            raise FusionUnsupported("stack layers project to different widths")
        self.out_per_layer = outs.pop()

        with torch.no_grad():
            # cat along the output axis reproduces torch.cat(..., dim=-1).
            self.weight = torch.cat(
                [linear.weight * norm.weight for norm, linear in zip(norms, linears)]
            )
            self.bias = torch.cat(
                [linear.weight @ norm.bias for norm, linear in zip(norms, linears)]
            )

    def key(self) -> tuple[torch.device, torch.dtype]:
        return self.weight.device, self.weight.dtype

    def __call__(self, x: Tensor) -> Tensor:
        normalized = F.layer_norm(x, self.normalized_shape, eps=self.eps)
        return F.linear(normalized, self.weight.to(normalized.dtype), self.bias.to(normalized.dtype))


def fused_stack(owner: nn.Module, name: str, layers: nn.ModuleList):
    """Return the cached fusion of ``layers``, or None if it cannot be built.

    The folded weights are derived from checkpoint parameters, so they are built
    on first use rather than at construction, and they are *not* registered as
    buffers: doing so would add keys the strict checkpoint load rejects. The
    cache is keyed on device and dtype so a ``.to()`` between calls rebuilds it.
    """
    attribute = f"_fused_{name}"
    cached = getattr(owner, attribute, None)
    if cached is False:  # A previous attempt found the stack unsupported.
        return None
    source = layers[0][1].weight
    if cached is not None and cached.key() == (source.device, source.dtype):
        return cached
    try:
        fused = FusedNormLinearStack(layers)
    except FusionUnsupported:
        object.__setattr__(owner, attribute, False)
        return None
    object.__setattr__(owner, attribute, fused)
    return fused
