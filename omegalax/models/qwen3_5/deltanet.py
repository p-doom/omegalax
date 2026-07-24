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

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec

from .kernels import chunk_gated_delta_rule
from .config import Qwen3_5TextConfig
from .norms import RMSNormGated

P = PartitionSpec
wp = nnx.with_partitioning


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
            D, conv_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype, kernel_init=in_proj_init
        )
        self.in_proj_z = nnx.Linear(
            D, self.value_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype, kernel_init=in_proj_init
        )
        self.in_proj_b = nnx.Linear(
            D, self.num_v_heads, use_bias=False, rngs=rngs, dtype=cfg.dtype, kernel_init=in_proj_init
        )
        self.in_proj_a = nnx.Linear(
            D, self.num_v_heads, use_bias=False, rngs=rngs, dtype=cfg.dtype, kernel_init=in_proj_init
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
        self.norm = RMSNormGated(
            self.head_v_dim, cfg.rms_norm_eps, rngs=rngs, sharding=(None,)
        )
        self.out_proj = nnx.Linear(
            self.value_dim,
            D,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=wp(init, ("mlp", "embed")),
        )

    @jax.named_scope("gated_delta_net")
    def __call__(self, hidden_BTD: jax.Array, attention_mask_BT: jax.Array | None = None) -> jax.Array:
        if attention_mask_BT is not None and attention_mask_BT.shape[1] > 1:
            hidden_BTD = hidden_BTD * attention_mask_BT[:, :, None]

        B, T, _ = hidden_BTD.shape

        heads_shd = self.shd_cfg.act_btnh
        batch_axis = heads_shd[0]
        head_axis = heads_shd[2]
        beta_g_shd = P(batch_axis, None, head_axis)
        mixed_qkv_BCT = self.in_proj_qkv(hidden_BTD, out_sharding=self.shd_cfg.act_btf).transpose(0, 2, 1)
        z_BTHU = jax.lax.reshape(
            self.in_proj_z(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
            (B, T, self.num_v_heads, self.head_v_dim),
            out_sharding=heads_shd,
        )
        b_BTH = self.in_proj_b(hidden_BTD, out_sharding=beta_g_shd)
        a_BTH = self.in_proj_a(hidden_BTD, out_sharding=beta_g_shd)

        mixed_qkv_BCT = nnx.silu(_causal_depthwise_conv1d(mixed_qkv_BCT, self.conv_weight[...].astype(mixed_qkv_BCT.dtype)))
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

        mesh = jax.sharding.get_abstract_mesh()
        beta_BTH = jax.nn.sigmoid(b_BTH)
        A_H = -jnp.exp(self.A_log[...].astype(jnp.float32))
        g_BTH = A_H * jax.nn.softplus(a_BTH.astype(jnp.float32) + self.dt_bias[...])

        norm_w = self.norm.weight[...]
        norm_eps = self.norm.eps
        head_k_dim = self.head_k_dim
        gqa_factor = self.gqa_factor

        def _full_deltanet(q_BTHA, k_BTHA, v_BTHU, z_BTHU, g_BTH, beta_BTH, nw):
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
            out = chunk_gated_delta_rule(q_BTHA, k_BTHA, v_BTHU, g_BTH, beta_BTH)
            # Inline RMSNormGated
            BL = out.shape[0] * out.shape[1]
            H, U = out.shape[2], out.shape[3]
            core_flat = out.reshape(BL * H, U)
            z_flat = z_BTHU.reshape(BL * H, U)
            dtype = core_flat.dtype
            x_f32 = core_flat.astype(jnp.float32)
            variance = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
            normed = (x_f32 * jax.lax.rsqrt(variance + norm_eps)).astype(dtype)
            normed = nw.astype(dtype) * normed
            gated = normed * jax.nn.silu(z_flat.astype(jnp.float32))
            return gated.astype(dtype).reshape(out.shape[0], out.shape[1], H * U)

        # jax.shard_map, not the deprecated jax.experimental one (whose Sharding custom call has no CUDA handler under fsdp>=2).
        normed_BTD = jax.shard_map(
            _full_deltanet,
            mesh=mesh,
            in_specs=(heads_shd, heads_shd, heads_shd, heads_shd, beta_g_shd, beta_g_shd, P(None)),
            out_specs=self.shd_cfg.act_btf,
            check_vma=False,
        )(q_BTHA, k_BTHA, v_BTHU, z_BTHU, g_BTH, beta_BTH, norm_w)

        out_BTD = self.out_proj(normed_BTD, out_sharding=self.shd_cfg.act_btd)
        return out_BTD
