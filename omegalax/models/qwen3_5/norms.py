"""Normalization layers for Qwen3.5."""

import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    """Qwen3.5 RMSNorm: output = (1 + weight) * norm(x).

    Weight is initialized to zeros so the initial behaviour is identity normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.zeros(dim))
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

    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.ones(dim))
        self.eps = eps

    @jax.named_scope("rms_norm_gated")
    def __call__(self, x: jax.Array, gate: jax.Array) -> jax.Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
        normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
        out = self.weight[...] * normed.astype(dtype)
        out = out * nnx.silu(gate.astype(jnp.float32))
        return out.astype(dtype)


class LayerNorm(nnx.Module):
    """Standard LayerNorm used in the vision encoder."""

    def __init__(self, dim: int, eps: float = 1e-6, *, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.ones(dim))
        self.bias = nnx.Param(jnp.zeros(dim))
        self.eps = eps

    @jax.named_scope("layer_norm")
    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.mean((x_f32 - mean) ** 2, axis=-1, keepdims=True)
        normed = (x_f32 - mean) * jax.lax.rsqrt(var + self.eps)
        return (self.weight[...] * normed + self.bias[...]).astype(dtype)
