"""Gated Delta Net for Qwen3.5.

This implements the chunked gated delta rule, a linear-attention variant
that combines a depthwise causal Conv1D with a recurrent delta-rule update.

Module-local dimension key (supplements the global key in models/__init__.py):

B — batch size
H — number of value heads (num_v_heads)
T — sequence length
A — key head dimension (linear_key_head_dim)
U — value head dimension (linear_value_head_dim)
J — number of chunks (total_T // chunk_size)
L — chunk position (row / target)
M — chunk position (column / source)
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec, reshard

from .config import Qwen3_5TextConfig
from .norms import RMSNormGated

P = PartitionSpec


def _l2norm(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
    inv_norm = jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _causal_depthwise_conv1d(x_BCT: jax.Array, weight_CK: jax.Array) -> jax.Array:
    """Depthwise causal conv1d.

    Args:
        x_BCT: (B, C, T)
        weight_CK: (C, K), per-channel kernel
    Returns:
        (B, C, T)
    """
    K = weight_CK.shape[1]
    T = x_BCT.shape[2]
    x_padded = jnp.pad(x_BCT, ((0, 0), (0, 0), (K - 1, 0)))
    result = jnp.zeros_like(x_BCT)
    for k in range(K):
        result = result + weight_CK[None, :, k : k + 1] * x_padded[:, :, k : k + T]
    return result


def chunk_gated_delta_rule(
    q_BTHA: jax.Array,
    k_BTHA: jax.Array,
    v_BTHU: jax.Array,
    g_BTH: jax.Array,
    beta_BTH: jax.Array,
    scan_state_shd: PartitionSpec,
    chunk_size: int = 64,
) -> jax.Array:
    """Chunked gated delta rule.

    All inputs are in (B, T, H, dim) layout.
    """
    q_BTHA = _l2norm(q_BTHA, axis=-1)
    k_BTHA = _l2norm(k_BTHA, axis=-1)

    q_BHTA, k_BHTA, v_BHTU = [x.transpose(0, 2, 1, 3).astype(jnp.float32) for x in (q_BTHA, k_BTHA, v_BTHU)]
    beta_BHT = beta_BTH.transpose(0, 2, 1).astype(jnp.float32)
    g_BHT = g_BTH.transpose(0, 2, 1).astype(jnp.float32)

    B, H, T, A = k_BHTA.shape
    U = v_BHTU.shape[-1]

    pad_size = (chunk_size - T % chunk_size) % chunk_size
    if pad_size > 0:
        q_BHTA = jnp.pad(q_BHTA, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        k_BHTA = jnp.pad(k_BHTA, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        v_BHTU = jnp.pad(v_BHTU, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta_BHT = jnp.pad(beta_BHT, ((0, 0), (0, 0), (0, pad_size)))
        g_BHT = jnp.pad(g_BHT, ((0, 0), (0, 0), (0, pad_size)))
    total_T = T + pad_size

    scale = A ** -0.5
    q_BHTA = q_BHTA * scale

    vb_BHTU = v_BHTU * beta_BHT[..., None]
    kb_BHTA = k_BHTA * beta_BHT[..., None]

    J = total_T // chunk_size
    q_BHJLA = q_BHTA.reshape(B, H, J, chunk_size, A)
    k_BHJLA = k_BHTA.reshape(B, H, J, chunk_size, A)
    v_BHJLU = v_BHTU.reshape(B, H, J, chunk_size, U)
    kb_BHJLA = kb_BHTA.reshape(B, H, J, chunk_size, A)
    vb_BHJLU = vb_BHTU.reshape(B, H, J, chunk_size, U)
    g_BHJL = g_BHT.reshape(B, H, J, chunk_size)

    g_BHJL = jnp.cumsum(g_BHJL, axis=-1)

    g_row = g_BHJL[..., :, None]
    g_col = g_BHJL[..., None, :]
    diff = g_row - g_col
    tril_mask = jnp.tril(jnp.ones((chunk_size, chunk_size)))
    decay_mask_LM = jnp.exp(diff * tril_mask) * tril_mask

    upper_mask_LM = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_))
    attn_BHJLM = -(jnp.einsum("BHJLA,BHJMA->BHJLM", kb_BHJLA, k_BHJLA) * decay_mask_LM)
    attn_BHJLM = jnp.where(upper_mask_LM, 0.0, attn_BHJLM)

    def correction_step(i, attn):
        row = attn[..., i, :]
        contribution = jnp.einsum("...j,...jk->...k", row, attn)
        new_row = row + contribution
        return attn.at[..., i, :].set(new_row)

    attn_BHJLM = jax.lax.fori_loop(1, chunk_size, correction_step, attn_BHJLM)
    attn_BHJLM = attn_BHJLM + jnp.eye(chunk_size)

    v_corrected_BHJLU = jnp.einsum("BHJLM,BHJMU->BHJLU", attn_BHJLM, vb_BHJLU)
    k_cumdecay_BHJLA = jnp.einsum("BHJLM,BHJMA->BHJLA", attn_BHJLM, kb_BHJLA * jnp.exp(g_BHJL)[..., None])

    state_BHAU = jnp.zeros((B, H, A, U), dtype=jnp.float32)
    state_BHAU = reshard(state_BHAU, scan_state_shd)
    upper_mask_1_LM = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=jnp.bool_), k=1)

    def chunk_step(carry, chunk_idx):
        st_BHAU = carry
        q_j_BHLA = q_BHJLA[:, :, chunk_idx]
        k_j_BHMA = k_BHJLA[:, :, chunk_idx]
        v_j_BHLU = v_corrected_BHJLU[:, :, chunk_idx]
        g_j_BHL = g_BHJL[:, :, chunk_idx]
        kcd_j_BHLA = k_cumdecay_BHJLA[:, :, chunk_idx]
        dm_j_LM = decay_mask_LM[:, :, chunk_idx]

        intra_BHLM = (jnp.einsum("BHLA,BHMA->BHLM", q_j_BHLA, k_j_BHMA) * dm_j_LM)
        intra_BHLM = jnp.where(upper_mask_1_LM, 0.0, intra_BHLM)

        v_prime_BHLU = jnp.einsum("BHLA,BHAU->BHLU", kcd_j_BHLA, st_BHAU)
        v_new_BHLU = v_j_BHLU - v_prime_BHLU

        inter_BHLU = jnp.einsum("BHL,BHLU->BHLU", jnp.exp(g_j_BHL), jnp.einsum("BHLA,BHAU->BHLU", q_j_BHLA, st_BHAU))
        chunk_out_BHLU = inter_BHLU + jnp.einsum("BHLM,BHMU->BHLU", intra_BHLM, v_new_BHLU)

        g_last = g_j_BHL[:, :, -1, None, None]
        g_decay_BHL = jnp.exp(g_j_BHL[:, :, -1:] - g_j_BHL)
        k_decayed_BHMA = k_j_BHMA * g_decay_BHL[..., None]
        new_st_BHAU = st_BHAU * jnp.exp(g_last) + jnp.einsum("BHMA,BHMU->BHAU", k_decayed_BHMA, v_new_BHLU)
        new_st_BHAU = reshard(new_st_BHAU, scan_state_shd)

        return new_st_BHAU, chunk_out_BHLU

    state_BHAU, core_out_chunks = jax.lax.scan(
        chunk_step, state_BHAU, jnp.arange(J)
    )
    # core_out_chunks: (J, B, H, L, U) -> (B, H, J, L, U)
    core_out_BHJLU = core_out_chunks.transpose(1, 2, 0, 3, 4)

    core_out_BHTU = core_out_BHJLU.reshape(B, H, -1, U)[:, :, :T, :]
    return core_out_BHTU.transpose(0, 2, 1, 3)  # (B, T, H, U)


class GatedDeltaNet(nnx.Module):
    """Gated Delta Net linear attention block."""

    def __init__(self, cfg: Qwen3_5TextConfig, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
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

        self.in_proj_qkv = nnx.Linear(D, conv_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype)
        self.in_proj_z = nnx.Linear(D, self.value_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype)
        self.in_proj_b = nnx.Linear(D, self.num_v_heads, use_bias=False, rngs=rngs, dtype=cfg.dtype)
        self.in_proj_a = nnx.Linear(D, self.num_v_heads, use_bias=False, rngs=rngs, dtype=cfg.dtype)

        self.conv_weight = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (conv_dim, self.conv_kernel_size))
        )

        self.dt_bias = nnx.Param(jnp.ones(self.num_v_heads))
        self.A_log = nnx.Param(jnp.log(jax.random.uniform(rngs.params(), (self.num_v_heads,)) * 16))

        batch_axis = cfg.shd_cfg.act_btd[0]
        head_axis = cfg.shd_cfg.act_btnh[2]
        if batch_axis is None:
            flat_axis = head_axis
        elif head_axis is None or head_axis == batch_axis:
            flat_axis = batch_axis
        else:
            flat_axis = (batch_axis, head_axis)
        self.hidden_shd = cfg.shd_cfg.act_btd
        self.scan_state_shd = P(batch_axis, head_axis, None, None)
        self.flat_norm_shd = P(flat_axis, None)
        self.norm = RMSNormGated(self.head_v_dim, cfg.rms_norm_eps, rngs=rngs)
        self.out_proj = nnx.Linear(self.value_dim, D, use_bias=False, rngs=rngs, dtype=cfg.dtype)

    @jax.named_scope("gated_delta_net")
    def __call__(self, hidden_BTD: jax.Array, attention_mask_BT: jax.Array | None = None) -> jax.Array:
        hidden_BTD = reshard(hidden_BTD, self.hidden_shd)

        if attention_mask_BT is not None and attention_mask_BT.shape[1] > 1:
            hidden_BTD = hidden_BTD * attention_mask_BT[:, :, None]

        B, T, _ = hidden_BTD.shape

        batch_axis = self.shd_cfg.act_btd[0]
        head_axis = self.shd_cfg.act_btnh[2]
        mixed_qkv_BCT = self.in_proj_qkv(hidden_BTD, out_sharding=self.shd_cfg.act_btf).transpose(0, 2, 1)
        z_BTHU = self.in_proj_z(hidden_BTD, out_sharding=self.shd_cfg.act_btf).reshape(B, T, self.num_v_heads, self.head_v_dim)
        b_BTH = self.in_proj_b(hidden_BTD, out_sharding=P(batch_axis, None, head_axis))
        a_BTH = self.in_proj_a(hidden_BTD, out_sharding=P(batch_axis, None, head_axis))

        mixed_qkv_BCT = nnx.silu(_causal_depthwise_conv1d(mixed_qkv_BCT, self.conv_weight[...]))
        mixed_qkv_BTC = mixed_qkv_BCT.transpose(0, 2, 1)

        q_BTHA, k_BTHA, v_BTHU = jnp.split(
            mixed_qkv_BTC, [self.key_dim, self.key_dim * 2], axis=-1
        )
        q_BTHA = q_BTHA.reshape(B, T, self.num_k_heads, self.head_k_dim)
        k_BTHA = k_BTHA.reshape(B, T, self.num_k_heads, self.head_k_dim)
        v_BTHU = v_BTHU.reshape(B, T, self.num_v_heads, self.head_v_dim)

        beta_BTH = jax.nn.sigmoid(b_BTH)
        A_H = -jnp.exp(self.A_log[...].astype(jnp.float32))
        g_BTH = A_H * jax.nn.softplus(a_BTH.astype(jnp.float32) + self.dt_bias[...])

        if self.gqa_factor > 1:
            q_BTHA = jnp.broadcast_to(
                q_BTHA[:, :, :, None, :],
                (B, T, self.num_k_heads, self.gqa_factor, self.head_k_dim),
            ).reshape(B, T, self.num_v_heads, self.head_k_dim)
            k_BTHA = jnp.broadcast_to(
                k_BTHA[:, :, :, None, :],
                (B, T, self.num_k_heads, self.gqa_factor, self.head_k_dim),
            ).reshape(B, T, self.num_v_heads, self.head_k_dim)

        core_out_BTHU = chunk_gated_delta_rule(
            q_BTHA,
            k_BTHA,
            v_BTHU,
            g_BTH,
            beta_BTH,
            scan_state_shd=self.scan_state_shd,
        )

        flat_rows = B * T * self.num_v_heads
        core_flat = jax.lax.reshape(core_out_BTHU, (flat_rows, self.head_v_dim), out_sharding=self.flat_norm_shd)
        z_flat = jax.lax.reshape(z_BTHU, (flat_rows, self.head_v_dim), out_sharding=self.flat_norm_shd)
        normed = self.norm(core_flat, z_flat)
        normed_BTD = jax.lax.reshape(normed, (B, T, self.value_dim), out_sharding=self.shd_cfg.act_btd)
        out_BTD = self.out_proj(normed_BTD, out_sharding=self.shd_cfg.act_btd)
        return reshard(out_BTD, self.shd_cfg.act_btd)
