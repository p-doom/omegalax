import jax
import jax.numpy as jnp
from flax import nnx

from .config import ShardingSpec


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, norm_eps: float, shd: ShardingSpec, *, rngs: nnx.Rngs):
        self.scale = nnx.Param(nnx.initializers.ones_init()(rngs.params(), (dim,)))
        self.shd = shd
        self.norm_eps = norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        variance = jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.norm_eps)
        return jnp.astype(self.scale[...] * normed, dtype)
