import jax
import jax.numpy as jnp
from flax import nnx
from tokamax import layer_norm


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
        return layer_norm(
            x,
            scale=self.scale[...],
            offset=None,
            epsilon=self.norm_eps,
            subtract_mean=False,
        )
