"""Normalization layers for Qwen3.5."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

class RMSNorm(nnx.Module):
    """Qwen3.5 RMSNorm: output = (1 + weight) * norm(x).

    Weight is initialized to zeros so the initial behaviour is identity normalization.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
        param_dtype: Any = jnp.float32,
        sharding: tuple[str | None, ...] = ("hidden",),
    ):
        self.weight = nnx.Param(jnp.zeros(dim, dtype=param_dtype), sharding=sharding)
        self.eps = eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
        normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
        return ((1.0 + self.weight[...].astype(jnp.float32)) * normed).astype(dtype)


class RMSNormGated(nnx.Module):
    """Gated RMSNorm: weight * norm(x) * silu(gate)."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
        param_dtype: Any = jnp.float32,
        sharding: tuple[str | None, ...] = ("hidden",),
    ):
        self.weight = nnx.Param(jnp.ones(dim, dtype=param_dtype), sharding=sharding)
        self.eps = eps

    @jax.named_scope("rms_norm_gated")
    def __call__(self, x: jax.Array, gate: jax.Array) -> jax.Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
        normed = jnp.astype(x_f32 * jax.lax.rsqrt(variance + self.eps), dtype)
        weight = jnp.astype(self.weight[...], dtype)
        out = weight * normed
        out = out * nnx.silu(gate.astype(jnp.float32))
        return out.astype(dtype)


class LayerNorm(nnx.Module):
    """Standard LayerNorm used in the vision encoder."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
        param_dtype: Any = jnp.float32,
        sharding: tuple[str | None, ...] = ("hidden",),
    ):
        self.weight = nnx.Param(jnp.ones(dim, dtype=param_dtype), sharding=sharding)
        self.bias = nnx.Param(jnp.zeros(dim, dtype=param_dtype), sharding=sharding)
        self.eps = eps

    @jax.named_scope("layer_norm")
    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.mean((x_f32 - mean) ** 2, axis=-1, keepdims=True)
        normed = (x_f32 - mean) * jax.lax.rsqrt(var + self.eps)
        return (self.weight[...] * normed + self.bias[...]).astype(dtype)
