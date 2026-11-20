"""Gated Delta Net for Qwen3.5.

This implements the chunked gated delta rule, a linear-attention variant
that combines a depthwise causal Conv1D with a recurrent delta-rule update.

Module-local dimension key (supplements the global key in models/__init__.py):

B — batch size
H — number of value heads (num_v_heads)
T — sequence length
C — combined qkv projection dimension
    (conv_dim = 2 * (num_k_heads * A) + (num_v_heads * U))
D — model / hidden dimension (hidden_size)
A — key head dimension (linear_key_head_dim)
U — value head dimension (linear_value_head_dim)
P — flattened q/k projection dimension (key_dim = num_k_heads * A)
O — flattened value projection dimension (value_dim = num_v_heads * U)
K — convolution kernel dimension (linear_conv_kernel_dim)
J — number of chunks (total_T // chunk_size)
L — chunk position (row / target)
M — chunk position (column / source)
"""

import os as _os

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec

from .kernels import chunk_gated_delta_rule
from .config import Qwen3_5TextConfig
from .norms import RMSNormGated

P = PartitionSpec
wp = nnx.with_partitioning


_CONV_BTC = _os.environ.get("MFU_CONV_BTC", "0") == "1"

# Only active for MFU_REMAT=names, which pairs with save_only_these_names in
# model.py. The coarse policies save every dot output and roughly double HBM, so
# they only fit at a batch small enough that the lost batch costs more than the
# saved recompute. This names one intermediate instead: the delta-rule output,
# which sits behind the most expensive non-GEMM work in the forward
# (_state_pass_kernel 95.5 ms/step plus batch_trsm 73 ms/step) yet is only
# [B,T,H,U] bf16, about 201 MB per GDN layer.
_CKPT_NAMES = _os.environ.get("MFU_REMAT", "none") == "names"

# Delta-rule chunk size. Production uses the kernel default of 64. Smaller chunks
# cut the per-token FLOPs of the recurrence (they scale as C^2 + 1280*C for
# A=U=128, so 64 -> 32 is about -24 %) but double the number of chunks the kernel
# walks sequentially per (batch, head). Which side wins is not derivable, only
# measurable. Same dtypes throughout -- this regroups the recurrence exactly as
# the cascade does, so it is mathematically equivalent but not bit-identical.
_CHUNK_SIZE = int(_os.environ.get("MFU_CHUNK", "64"))


def _causal_depthwise_conv1d_btc(
    x_BTC: jax.Array,
    weight_CK: jax.Array,
    initial_state_BCS: jax.Array | None = None,
) -> jax.Array:
    """Depthwise causal conv1d in (B, T, C) layout.

    Same arithmetic as ``_causal_depthwise_conv1d``, same k-loop order, same
    dtypes -- only the memory layout differs, so results are bit-identical.

    The BCT version exists because the K taps shift along T, which is the last
    (contiguous) axis there. But a shift along T is just as expressible in BTC as
    a slice on axis 1, and the caller both produces and consumes BTC. Staying in
    BCT costs two transposes of the full mixed-QKV tensor per GDN layer, and XLA
    folds both of them plus the conv and the silu into one kernel
    (``input_transpose_fusion.198``) that moves ~1.2 GiB in 10.0 ms -- roughly 19x
    slower than the same traffic at the measured 2300 GB/s, because every element
    is read and written strided.

    Only the shape production needs: ``pass_conv_state=false``, so no final state
    is returned and the mask path is not reachable.
    """
    B, T, C = x_BTC.shape
    K = weight_CK.shape[1]
    state_len = K - 1
    if initial_state_BCS is None:
        # Derive the zeros from x rather than building them free-standing, so the
        # batch axis keeps x's sharding. A plain jnp.zeros is unsharded, and the
        # concatenate below then fails under FSDP with "All operands should have
        # the same sharding" -- invisible on a single device. The BCT version
        # does the same thing for the same reason.
        init_BSC = jnp.zeros_like(jnp.broadcast_to(x_BTC[:, :1, :], (B, state_len, C)))
    else:
        init_BSC = initial_state_BCS.astype(x_BTC.dtype).transpose(0, 2, 1)
    x_padded = jnp.concatenate([init_BSC, x_BTC], axis=1)
    result = jnp.zeros_like(x_BTC)
    for k in range(K):
        result = result + weight_CK[None, None, :, k] * x_padded[:, k : k + T, :]
    return result


def _causal_depthwise_conv1d(
    x_BCT: jax.Array,
    weight_CK: jax.Array,
    initial_state_BCS: jax.Array | None = None,
    attention_mask_BT: jax.Array | None = None,
    *,
    return_final_state: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Depthwise causal conv1d.

    Args:
        x_BCT: (B, C, T)
        weight_CK: (C, K), per-channel kernel
    Returns:
        (B, C, T)
    """
    B, C, T = x_BCT.shape
    K = weight_CK.shape[1]
    state_len = K - 1
    if initial_state_BCS is None:
        if state_len == 0:
            initial_state_BCS = jnp.zeros((B, C, 0), dtype=x_BCT.dtype)
        else:
            initial_state_BCS = jnp.zeros_like(jnp.broadcast_to(x_BCT[:, :, :1], (B, C, state_len)))
    elif initial_state_BCS.shape != (B, C, state_len):
        raise ValueError(
            f"Expected conv initial state shape {(B, C, state_len)}, got {initial_state_BCS.shape}"
        )
    else:
        initial_state_BCS = initial_state_BCS.astype(x_BCT.dtype)
    x_padded = jnp.concatenate([initial_state_BCS, x_BCT], axis=2)
    result = jnp.zeros_like(x_BCT)
    for k in range(K):
        result = result + weight_CK[None, :, k : k + 1] * x_padded[:, :, k : k + T]
    if return_final_state:
        if state_len == 0:
            return result, jnp.zeros((B, C, 0), dtype=x_BCT.dtype)
        if attention_mask_BT is None:
            return result, x_padded[:, :, -state_len:]
        valid_len_B = jnp.sum(attention_mask_BT.astype(jnp.int32), axis=1)
        gather_start_B = jnp.maximum(valid_len_B, 0)
        history = []
        for offset in range(state_len):
            idx_B = gather_start_B + offset
            history.append(jnp.take_along_axis(x_padded, idx_B[:, None, None], axis=2))
        return result, jnp.concatenate(history, axis=2)
    return result


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
        init = nnx.initializers.lecun_normal()

        in_proj_init = wp(init, ("embed", "mlp"))
        self.in_proj_qkv = nnx.Linear(
            D,
            conv_dim,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            kernel_init=in_proj_init,
        )
        self.in_proj_z = nnx.Linear(
            D,
            self.value_dim,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            kernel_init=in_proj_init,
        )
        self.in_proj_b = nnx.Linear(
            D,
            self.num_v_heads,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            kernel_init=in_proj_init,
        )
        self.in_proj_a = nnx.Linear(
            D,
            self.num_v_heads,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            kernel_init=in_proj_init,
        )

        self.conv_weight = nnx.Param(
            init(rngs.params(), (conv_dim, self.conv_kernel_size)),
            sharding=(None, None),
        )

        self.dt_bias = nnx.Param(jnp.ones(self.num_v_heads), sharding=(None,))
        self.A_log = nnx.Param(
            jnp.log(jax.random.uniform(rngs.params(), (self.num_v_heads,)) * 16),
            sharding=(None,),
        )

        heads_shd = cfg.shd_cfg.act_btnh
        batch_axis = heads_shd[0]
        head_axis = heads_shd[2]
        if batch_axis is None:
            flat_axis = head_axis
        elif head_axis is None or head_axis == batch_axis:
            flat_axis = batch_axis
        else:
            # Flatten into a single tuple — PartitionSpec forbids nested tuples.
            ba = batch_axis if isinstance(batch_axis, tuple) else (batch_axis,)
            ha = head_axis if isinstance(head_axis, tuple) else (head_axis,)
            flat_axis = (*ba, *ha)
        self.hidden_shd = cfg.shd_cfg.act_btd
        self.scan_state_shd = P(batch_axis, head_axis, None, None)
        self.flat_norm_shd = P(flat_axis, None)
        self.norm = RMSNormGated(self.head_v_dim, cfg.rms_norm_eps, rngs=rngs, sharding=(None,))
        self.out_proj = nnx.Linear(
            self.value_dim,
            D,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            param_dtype=cfg.param_dtype,
            kernel_init=wp(init, ("mlp", "embed")),
        )

    @jax.named_scope("gated_delta_net")
    def __call__(
        self,
        hidden_BTD: jax.Array,
        attention_mask_BT: jax.Array | None = None,
        initial_state_BHAU: jax.Array | None = None,
        initial_conv_state_BCS: jax.Array | None = None,
        *,
        return_final_state: bool = False,
        return_conv_state: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
        if attention_mask_BT is not None and attention_mask_BT.shape[1] > 1:
            hidden_BTD = hidden_BTD * attention_mask_BT[:, :, None]

        B, T, _ = hidden_BTD.shape
        if initial_state_BHAU is None:
            initial_state_BHAU = jnp.zeros(
                (B, self.num_v_heads, self.head_k_dim, self.head_v_dim),
                dtype=jnp.float32,
            )
        mesh = jax.sharding.get_abstract_mesh()
        initial_state_BHAU = jax.sharding.reshard(
            initial_state_BHAU, jax.sharding.NamedSharding(mesh, self.scan_state_shd)
        )

        heads_shd = self.shd_cfg.act_btnh
        batch_axis = heads_shd[0]
        head_axis = heads_shd[2]
        beta_g_shd = P(batch_axis, None, head_axis)
        mixed_qkv_proj_BTC = self.in_proj_qkv(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        use_conv_btc = _CONV_BTC and not return_conv_state
        mixed_qkv_BCT = (
            mixed_qkv_proj_BTC if use_conv_btc else mixed_qkv_proj_BTC.transpose(0, 2, 1)
        )
        z_BTHU = jax.lax.reshape(
            self.in_proj_z(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
            (B, T, self.num_v_heads, self.head_v_dim),
            out_sharding=heads_shd,
        )
        b_BTH = self.in_proj_b(hidden_BTD, out_sharding=beta_g_shd)
        a_BTH = self.in_proj_a(hidden_BTD, out_sharding=beta_g_shd)

        if use_conv_btc:
            final_conv_state_BCS = None
            mixed_qkv_BTC = nnx.silu(
                _causal_depthwise_conv1d_btc(
                    mixed_qkv_BCT,
                    self.conv_weight[...].astype(mixed_qkv_BCT.dtype),
                    initial_state_BCS=initial_conv_state_BCS,
                )
            )
        else:
            conv_result = _causal_depthwise_conv1d(
                mixed_qkv_BCT,
                self.conv_weight[...].astype(mixed_qkv_BCT.dtype),
                initial_state_BCS=initial_conv_state_BCS,
                attention_mask_BT=attention_mask_BT,
                return_final_state=return_conv_state,
            )
            if return_conv_state:
                mixed_qkv_BCT, final_conv_state_BCS = conv_result
            else:
                final_conv_state_BCS = None
                mixed_qkv_BCT = conv_result
            mixed_qkv_BCT = nnx.silu(mixed_qkv_BCT)
            mixed_qkv_BTC = mixed_qkv_BCT.transpose(0, 2, 1)
        q_BTP, k_BTP, v_BTO = jnp.split(mixed_qkv_BTC, [self.key_dim, self.key_dim * 2], axis=-1)
        q_BTHA = jax.lax.reshape(
            q_BTP,
            (B, T, self.num_k_heads, self.head_k_dim),
            out_sharding=heads_shd,
        )
        k_BTHA = jax.lax.reshape(
            k_BTP,
            (B, T, self.num_k_heads, self.head_k_dim),
            out_sharding=heads_shd,
        )
        v_BTHU = jax.lax.reshape(
            v_BTO,
            (B, T, self.num_v_heads, self.head_v_dim),
            out_sharding=heads_shd,
        )

        from jax.experimental.shard_map import shard_map

        beta_BTH = jax.nn.sigmoid(b_BTH)
        A_H = -jnp.exp(self.A_log[...].astype(jnp.float32))
        g_BTH = A_H * jax.nn.softplus(a_BTH.astype(jnp.float32) + self.dt_bias[...])
        if attention_mask_BT is not None:
            mask_BTH = attention_mask_BT[:, :, None].astype(q_BTHA.dtype)
            q_BTHA = q_BTHA * mask_BTH[..., None]
            k_BTHA = k_BTHA * mask_BTH[..., None]
            v_BTHU = v_BTHU * mask_BTH[..., None]
            z_BTHU = z_BTHU * mask_BTH[..., None]
            beta_BTH = beta_BTH * mask_BTH
            g_BTH = g_BTH * mask_BTH.astype(g_BTH.dtype)

        norm_w = self.norm.weight[...]
        norm_eps = self.norm.eps
        head_k_dim = self.head_k_dim
        gqa_factor = self.gqa_factor

        def _full_deltanet(q_BTHA, k_BTHA, v_BTHU, z_BTHU, g_BTH, beta_BTH, init_BHAU, nw):
            """GQA → DeltaNet → norm. All inputs have shard-local head axes."""
            B, T = q_BTHA.shape[:2]
            local_k_heads = q_BTHA.shape[2]
            local_v_heads = v_BTHU.shape[2]
            assert k_BTHA.shape[2] == local_k_heads
            assert z_BTHU.shape[2] == local_v_heads
            assert local_v_heads == local_k_heads * gqa_factor
            if gqa_factor > 1:
                q_BTHA = jnp.broadcast_to(
                    q_BTHA[:, :, :, None, :],
                    (B, T, local_k_heads, gqa_factor, head_k_dim),
                ).reshape(B, T, local_v_heads, head_k_dim)
                k_BTHA = jnp.broadcast_to(
                    k_BTHA[:, :, :, None, :],
                    (B, T, local_k_heads, gqa_factor, head_k_dim),
                ).reshape(B, T, local_v_heads, head_k_dim)
            out, final_state = chunk_gated_delta_rule(
                q_BTHA,
                k_BTHA,
                v_BTHU,
                g_BTH,
                beta_BTH,
                chunk_size=_CHUNK_SIZE,
                initial_state_BHAU=init_BHAU,
                return_final_state=True,
            )
            if _CKPT_NAMES:
                out = jax.ad_checkpoint.checkpoint_name(out, "gdn_core")
            # Inline RMSNormGated
            BL = out.shape[0] * out.shape[1]
            H, U = out.shape[2], out.shape[3]
            core_flat = out.reshape(BL * H, U)
            z_flat = z_BTHU.reshape(BL * H, U)
            dtype = core_flat.dtype
            x_f32 = core_flat.astype(jnp.float32)
            variance = jnp.mean(x_f32**2, axis=-1, keepdims=True)
            normed = (x_f32 * jax.lax.rsqrt(variance + norm_eps)).astype(dtype)
            normed = nw.astype(dtype) * normed
            gated = normed * jax.nn.silu(z_flat.astype(jnp.float32))
            return gated.astype(dtype).reshape(out.shape[0], out.shape[1], H * U), final_state

        normed_BTD, final_state_BHAU = shard_map(
            _full_deltanet,
            mesh,
            in_specs=(
                heads_shd,
                heads_shd,
                heads_shd,
                heads_shd,
                beta_g_shd,
                beta_g_shd,
                self.scan_state_shd,
                P(None),
            ),
            out_specs=(self.shd_cfg.act_btf, self.scan_state_shd),
            check_rep=False,
        )(q_BTHA, k_BTHA, v_BTHU, z_BTHU, g_BTH, beta_BTH, initial_state_BHAU, norm_w)

        out_BTD = self.out_proj(normed_BTD, out_sharding=self.shd_cfg.act_btd)
        if return_final_state and return_conv_state:
            return out_BTD, final_state_BHAU, final_conv_state_BCS
        if return_final_state:
            return out_BTD, final_state_BHAU
        if return_conv_state:
            return out_BTD, final_conv_state_BCS
        return out_BTD
