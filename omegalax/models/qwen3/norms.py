import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    def __init__(
        self,
        dim: int,
        norm_eps: float,
        *,
        rngs: nnx.Rngs,
        sharding: tuple[str | None, ...] = ("hidden",),
    ):
        self.scale = nnx.Param(
            nnx.initializers.ones_init()(rngs.params(), (dim,)),
            sharding=sharding,
        )
        self.norm_eps = norm_eps

    @jax.named_scope("rms_norm")
    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x_f32 = jnp.astype(x, jnp.float32)
        variance = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
        normed = jnp.astype(x_f32 * jax.lax.rsqrt(variance + self.norm_eps), dtype)
        return jnp.astype(self.scale[...], dtype) * normed
