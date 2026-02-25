"""Rotary position embeddings for Qwen3.5 vision and text models."""

import jax
import jax.numpy as jnp


# Vision RoPE (2D grid positions)
def generate_vision_rope(seqlen: int, dim: int, theta: float = 10000.0):
    """Generate rotary frequencies for vision positions.

    Returns:
        freqs_NK: (seqlen, dim // 2)
    """
    fraction = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    inv_freq = 1.0 / (theta ** fraction)
    seq = jnp.arange(seqlen, dtype=jnp.float32)
    return jnp.outer(seq, inv_freq)


def apply_vision_rope(
    q_NHK: jax.Array, k_NHK: jax.Array, cos_NK: jax.Array, sin_NK: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Apply RoPE to vision query / key tensors.

    Args:
        q_NHK, k_NHK: (seq_len, num_heads, head_dim)
        cos_NK, sin_NK: (seq_len, head_dim)
    """
    q_f32, k_f32 = q_NHK.astype(jnp.float32), k_NHK.astype(jnp.float32)
    cos_NK = cos_NK[:, None, :].astype(jnp.float32)
    sin_NK = sin_NK[:, None, :].astype(jnp.float32)
    q_embed_NHK = q_f32 * cos_NK + _rotate_half(q_f32) * sin_NK
    k_embed_NHK = k_f32 * cos_NK + _rotate_half(k_f32) * sin_NK
    return q_embed_NHK.astype(q_NHK.dtype), k_embed_NHK.astype(k_NHK.dtype)


# Text MRoPE (multi-dimensional RoPE with interleaving)
def generate_text_rope(
    position_ids_ZBT: jax.Array,
    head_dim: int,
    partial_rotary_factor: float,
    rope_theta: float,
    mrope_section: tuple[int, ...],
):
    """Generate MRoPE cos/sin for text model.

    Args:
        position_ids_ZBT: (3, B, T)
        head_dim: full head dimension.
        partial_rotary_factor: fraction of head_dim that gets RoPE.
        rope_theta: base for inverse frequency.
        mrope_section: interleaving section sizes, sums to rotary_dim // 2.

    Returns:
        cos_BTK, sin_BTK: each (B, T, rotary_dim)
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    fraction = jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim
    inv_freq = 1.0 / (rope_theta ** fraction)

    inv_freq_exp = inv_freq[None, None, :, None]
    pos_exp = position_ids_ZBT[:, :, None, :].astype(jnp.float32)
    freqs_ZBTK = jnp.einsum("dbhp,dbhp->dbph", inv_freq_exp * jnp.ones_like(pos_exp), pos_exp)

    freqs_BTK = _apply_interleaved_mrope(freqs_ZBTK, mrope_section)
    emb_BTK = jnp.concatenate([freqs_BTK, freqs_BTK], axis=-1)
    return jnp.cos(emb_BTK), jnp.sin(emb_BTK)


def _apply_interleaved_mrope(
    freqs_ZBTK: jax.Array, mrope_section: tuple[int, ...]
) -> jax.Array:
    """Interleave T, H, W frequencies.

    Args:
        freqs_ZBTK: (3, B, T, half_dim)
        mrope_section: sizes for each dimension
    Returns:
        (B, T, half_dim) — interleaved.
    """
    freqs_BTK = freqs_ZBTK[0]
    for dim_idx, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim_idx] * 3
        indices = jnp.arange(offset, length, 3)
        freqs_BTK = freqs_BTK.at[..., indices].set(freqs_ZBTK[dim_idx][..., indices])
    return freqs_BTK


def apply_text_rope(
    q_BHTK: jax.Array, k_BHTK: jax.Array, cos_BTK: jax.Array, sin_BTK: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Apply partial RoPE to query and key.

    Args:
        q_BHTK, k_BHTK: (B, num_heads, T, head_dim)
        cos_BTK, sin_BTK: (B, T, rotary_dim)
    Returns:
        q, k with RoPE applied to the first rotary_dim dimensions.
    """
    rotary_dim = cos_BTK.shape[-1]
    cos_BTK = cos_BTK[:, None, :, :]
    sin_BTK = sin_BTK[:, None, :, :]

    q_rot, q_pass = q_BHTK[..., :rotary_dim], q_BHTK[..., rotary_dim:]
    k_rot, k_pass = k_BHTK[..., :rotary_dim], k_BHTK[..., rotary_dim:]

    q_embed = q_rot * cos_BTK + _rotate_half(q_rot) * sin_BTK
    k_embed = k_rot * cos_BTK + _rotate_half(k_rot) * sin_BTK

    return (
        jnp.concatenate([q_embed, q_pass], axis=-1),
        jnp.concatenate([k_embed, k_pass], axis=-1),
    )


# Shared helpers
def _rotate_half(x: jax.Array) -> jax.Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)
