"""Gated Delta Net for Qwen3.5.

This implements the chunked gated delta rule, a linear-attention variant
that combines a depthwise causal Conv1D with a recurrent delta-rule update.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .config import Qwen3_5TextConfig
from .norms import RMSNormGated


# Helpers
def _l2norm(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
    inv_norm = jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _causal_depthwise_conv1d(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Depthwise causal conv1d.

    Args:
        x: (B, C, T)
        weight: (C, K) — per-channel kernel
    Returns:
        (B, C, T)
    """
    K = weight.shape[1]
    T = x.shape[2]
    x_padded = jnp.pad(x, ((0, 0), (0, 0), (K - 1, 0)))
    result = jnp.zeros_like(x)
    for k in range(K):
        result = result + weight[None, :, k : k + 1] * x_padded[:, :, k : k + T]
    return result


# Chunked Gated Delta Rule (prefill path)
def chunk_gated_delta_rule(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    chunk_size: int = 64,
) -> jax.Array:
    """Chunked gated delta rule.

    All inputs are in (B, T, H, D) layout.
    """
    # L2 normalize Q, K
    query = _l2norm(query, axis=-1)
    key = _l2norm(key, axis=-1)

    # Transpose to (B, H, T, D)
    query, key, value = [x.transpose(0, 2, 1, 3).astype(jnp.float32) for x in (query, key, value)]
    beta = beta.transpose(0, 2, 1).astype(jnp.float32)
    g = g.transpose(0, 2, 1).astype(jnp.float32)

    B, H, T, Dk = key.shape
    Dv = value.shape[-1]

    # Pad to chunk_size multiple
    pad_size = (chunk_size - T % chunk_size) % chunk_size
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_size)))
    total_T = T + pad_size

    scale = Dk ** -0.5
    query = query * scale

    v_beta = value * beta[..., None]
    k_beta = key * beta[..., None]

    NC = total_T // chunk_size
    query = query.reshape(B, H, NC, chunk_size, Dk)
    key = key.reshape(B, H, NC, chunk_size, Dk)
    value = value.reshape(B, H, NC, chunk_size, Dv)
    k_beta = k_beta.reshape(B, H, NC, chunk_size, Dk)
    v_beta = v_beta.reshape(B, H, NC, chunk_size, Dv)
    g = g.reshape(B, H, NC, chunk_size)

    # Cumulative gate within chunks
    g = jnp.cumsum(g, axis=-1)

    # Decay mask: exp(g[..., i] - g[..., j]) for i >= j
    g_row = g[..., :, None]  # (B, H, NC, CS, 1)
    g_col = g[..., None, :]  # (B, H, NC, 1, CS)
    diff = g_row - g_col
    tril_mask = jnp.tril(jnp.ones((chunk_size, chunk_size)))
    decay_mask = jnp.exp(diff * tril_mask) * tril_mask

    # Within-chunk correction matrix
    upper_mask = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    attn = -(jnp.einsum("bhcid,bhcjd->bhcij", k_beta, key) * decay_mask)
    attn = jnp.where(upper_mask, 0.0, attn)

    # Sequential correction: (I + L)^{-1} approximation
    def correction_step(i, attn):
        row = attn[..., i, :]
        contribution = jnp.einsum("...j,...jk->...k", row, attn)
        new_row = row + contribution
        return attn.at[..., i, :].set(new_row)

    attn = jax.lax.fori_loop(1, chunk_size, correction_step, attn)
    attn = attn + jnp.eye(chunk_size)

    # Modified V and cumulative decay K
    value_corrected = jnp.einsum("bhcij,bhcjd->bhcid", attn, v_beta)
    k_cumdecay = jnp.einsum("bhcij,bhcjd->bhcid", attn, k_beta * jnp.exp(g)[..., None])

    # Cross-chunk recurrence
    last_state = jnp.zeros((B, H, Dk, Dv), dtype=jnp.float32)
    core_out = jnp.zeros_like(value_corrected)
    upper_mask_1 = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1)

    def chunk_step(carry, chunk_idx):
        last_st = carry
        q_i = query[:, :, chunk_idx]
        k_i = key[:, :, chunk_idx]
        v_i = value_corrected[:, :, chunk_idx]
        g_i = g[:, :, chunk_idx]
        kcd_i = k_cumdecay[:, :, chunk_idx]
        dm_i = decay_mask[:, :, chunk_idx]

        intra_attn = (jnp.einsum("bhid,bhjd->bhij", q_i, k_i) * dm_i)
        intra_attn = jnp.where(upper_mask_1, 0.0, intra_attn)

        v_prime = jnp.einsum("bhid,bhdv->bhiv", kcd_i, last_st)
        v_new = v_i - v_prime

        attn_inter = jnp.einsum("bhi,bhiv->bhiv", jnp.exp(g_i), jnp.einsum("bhid,bhdv->bhiv", q_i, last_st))
        chunk_out = attn_inter + jnp.einsum("bhij,bhjv->bhiv", intra_attn, v_new)

        # Update recurrent state
        g_last = g_i[:, :, -1, None, None]  # (B, H, 1, 1)
        g_decay = jnp.exp(g_i[:, :, -1:] - g_i)  # (B, H, CS)
        k_decayed = k_i * g_decay[..., None]  # (B, H, CS, Dk)
        new_state = last_st * jnp.exp(g_last) + jnp.einsum("bhid,bhiv->bhdv", k_decayed, v_new)

        return new_state, chunk_out

    last_state, core_out_chunks = jax.lax.scan(
        chunk_step, last_state, jnp.arange(NC)
    )
    # core_out_chunks: (NC, B, H, CS, Dv) → (B, H, NC, CS, Dv)
    core_out = core_out_chunks.transpose(1, 2, 0, 3, 4)

    # Reshape and trim
    core_out = core_out.reshape(B, H, -1, Dv)[:, :, :T, :]
    return core_out.transpose(0, 2, 1, 3)  # (B, T, H, Dv)


# Gated Delta Net Module

class GatedDeltaNet(nnx.Module):
    """Gated Delta Net linear attention block."""

    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        D = cfg.hidden_size
        self.num_v_heads = cfg.linear_num_value_heads
        self.num_k_heads = cfg.linear_num_key_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = cfg.linear_conv_kernel_dim
        self.gqa_factor = self.num_v_heads // self.num_k_heads

        conv_dim = self.key_dim * 2 + self.value_dim

        # Projections
        self.in_proj_qkv = nnx.Linear(D, conv_dim, use_bias=False, rngs=rngs)
        self.in_proj_z = nnx.Linear(D, self.value_dim, use_bias=False, rngs=rngs)
        self.in_proj_b = nnx.Linear(D, self.num_v_heads, use_bias=False, rngs=rngs)
        self.in_proj_a = nnx.Linear(D, self.num_v_heads, use_bias=False, rngs=rngs)

        # Depthwise causal conv1d — stored as (conv_dim, kernel_size)
        self.conv_weight = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (conv_dim, self.conv_kernel_size))
        )

        # Gating parameters
        self.dt_bias = nnx.Param(jnp.ones(self.num_v_heads))
        self.A_log = nnx.Param(jnp.log(jax.random.uniform(rngs.params(), (self.num_v_heads,)) * 16))

        # Output norm + projection
        self.norm = RMSNormGated(self.head_v_dim, cfg.rms_norm_eps, rngs=rngs)
        self.out_proj = nnx.Linear(self.value_dim, D, use_bias=False, rngs=rngs)

    @jax.named_scope("gated_delta_net")
    def __call__(self, x: jax.Array, attention_mask: jax.Array | None = None) -> jax.Array:
        """Forward pass (prefill, no cache).

        Args:
            x: (B, T, D)
            attention_mask: (B, T) boolean — optional padding mask
        """
        if attention_mask is not None and attention_mask.shape[1] > 1:
            x = x * attention_mask[:, :, None]

        B, T, _ = x.shape

        # Project
        mixed_qkv = self.in_proj_qkv(x).transpose(0, 2, 1)  # (B, conv_dim, T)
        z = self.in_proj_z(x).reshape(B, T, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)  # (B, T, num_v_heads)
        a = self.in_proj_a(x)  # (B, T, num_v_heads)

        # Causal depthwise conv1d + SiLU
        mixed_qkv = nnx.silu(_causal_depthwise_conv1d(mixed_qkv, self.conv_weight[...]))
        mixed_qkv = mixed_qkv.transpose(0, 2, 1)  # (B, T, conv_dim)

        # Split into Q, K, V
        query, key, value = jnp.split(
            mixed_qkv, [self.key_dim, self.key_dim * 2], axis=-1
        )
        query = query.reshape(B, T, self.num_k_heads, self.head_k_dim)
        key = key.reshape(B, T, self.num_k_heads, self.head_k_dim)
        value = value.reshape(B, T, self.num_v_heads, self.head_v_dim)

        # Compute beta and gate
        beta = jax.nn.sigmoid(b)  # (B, T, num_v_heads)
        A = -jnp.exp(self.A_log[...].astype(jnp.float32))
        g = A * jax.nn.softplus(a.astype(jnp.float32) + self.dt_bias[...])  # (B, T, num_v_heads)

        # GQA expansion for delta net
        if self.gqa_factor > 1:
            query = jnp.repeat(query, self.gqa_factor, axis=2)
            key = jnp.repeat(key, self.gqa_factor, axis=2)

        # Chunked delta rule
        core_out = chunk_gated_delta_rule(query, key, value, g, beta)

        # Gated norm + output projection
        core_out_flat = core_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        normed = self.norm(core_out_flat, z_flat)
        normed = normed.reshape(B, T, -1)
        return self.out_proj(normed)
