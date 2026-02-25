import jax
import jax.numpy as jnp


def generate_pos_embeddings(positions_BT: jax.Array, head_dim: int, rope_theta: int = 1_000_000):
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction
    inv_freq_K = 1.0 / timescale
    sinusoid_BTK = jnp.einsum("BT,K->BTK", positions_BT, inv_freq_K)
    return jnp.sin(sinusoid_BTK), jnp.cos(sinusoid_BTK)


def apply_rope(x_BTHK: jax.Array, sin_BTK: jax.Array, cos_BTK: jax.Array) -> jax.Array:
    assert x_BTHK.ndim == 4 and sin_BTK.ndim == 3 and cos_BTK.ndim == 3
    x1, x2 = x_BTHK[..., : x_BTHK.shape[-1] // 2], x_BTHK[..., x_BTHK.shape[-1] // 2 :]
    sin_BTK = sin_BTK[:, :, None, :]
    cos_BTK = cos_BTK[:, :, None, :]
    return jnp.concatenate([x1 * cos_BTK - x2 * sin_BTK, x2 * cos_BTK + x1 * sin_BTK], axis=-1).astype(x_BTHK.dtype)
