"""Rotary position embeddings for Qwen3.5 vision and text models."""

import jax
import jax.numpy as jnp


# Vision RoPE (2D grid positions)
def generate_vision_rope(seqlen: int, dim: int, theta: float = 10000.0):
    """Generate rotary frequencies for vision positions.

    Returns:
        freqs: (seqlen, dim // 2)
    """
    fraction = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    inv_freq = 1.0 / (theta ** fraction)
    seq = jnp.arange(seqlen, dtype=jnp.float32)
    return jnp.outer(seq, inv_freq)


def apply_vision_rope(
    q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Apply RoPE to vision query / key tensors.

    Args:
        q, k: (seq_len, num_heads, head_dim)
        cos, sin: (seq_len, head_dim)  (already doubled via cat)
    """
    q_f32, k_f32 = q.astype(jnp.float32), k.astype(jnp.float32)
    cos = cos[:, None, :].astype(jnp.float32)
    sin = sin[:, None, :].astype(jnp.float32)
    q_embed = q_f32 * cos + _rotate_half(q_f32) * sin
    k_embed = k_f32 * cos + _rotate_half(k_f32) * sin
    return q_embed.astype(q.dtype), k_embed.astype(k.dtype)


# Text MRoPE (multi-dimensional RoPE with interleaving)
def generate_text_rope(
    position_ids: jax.Array,
    head_dim: int,
    partial_rotary_factor: float,
    rope_theta: float,
    mrope_section: tuple[int, ...],
):
    """Generate MRoPE cos/sin for text model.

    Args:
        position_ids: (3, B, T) — temporal, height, width positions.
        head_dim: full head dimension.
        partial_rotary_factor: fraction of head_dim that gets RoPE.
        rope_theta: base for inverse frequency.
        mrope_section: interleaving section sizes, sums to rotary_dim // 2.

    Returns:
        cos, sin: each (B, T, rotary_dim)
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    half_dim = rotary_dim // 2
    fraction = jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim
    inv_freq = 1.0 / (rope_theta ** fraction)  # (half_dim,)

    # (3, B, 1, T) x (1, 1, half_dim, 1) → (3, B, half_dim, T) → (3, B, T, half_dim)
    inv_freq_exp = inv_freq[None, None, :, None]
    pos_exp = position_ids[:, :, None, :].astype(jnp.float32)
    freqs = jnp.einsum("dbhp,dbhp->dbph", inv_freq_exp * jnp.ones_like(pos_exp), pos_exp)
    # freqs: (3, B, T, half_dim)

    freqs = _apply_interleaved_mrope(freqs, mrope_section)  # (B, T, half_dim)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (B, T, rotary_dim)
    return jnp.cos(emb), jnp.sin(emb)


def _apply_interleaved_mrope(
    freqs: jax.Array, mrope_section: tuple[int, ...]
) -> jax.Array:
    """Interleave T, H, W frequencies.

    Args:
        freqs: (3, B, T, half_dim) — [temporal, height, width]
        mrope_section: sizes for each dimension
    Returns:
        (B, T, half_dim) — interleaved.
    """
    freqs_t = freqs[0]  # (B, T, half_dim) — start with temporal
    for dim_idx, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim_idx] * 3
        indices = jnp.arange(offset, length, 3)
        freqs_t = freqs_t.at[..., indices].set(freqs[dim_idx][..., indices])
    return freqs_t


def apply_text_rope(
    q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Apply partial RoPE to query and key.

    Args:
        q, k: (B, num_heads, T, head_dim)
        cos, sin: (B, T, rotary_dim)
    Returns:
        q, k with RoPE applied to the first rotary_dim dimensions.
    """
    rotary_dim = cos.shape[-1]
    cos = cos[:, None, :, :]  # (B, 1, T, rotary_dim)
    sin = sin[:, None, :, :]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = q_rot * cos + _rotate_half(q_rot) * sin
    k_embed = k_rot * cos + _rotate_half(k_rot) * sin

    return (
        jnp.concatenate([q_embed, q_pass], axis=-1),
        jnp.concatenate([k_embed, k_pass], axis=-1),
    )


# Shared helpers
def _rotate_half(x: jax.Array) -> jax.Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)
