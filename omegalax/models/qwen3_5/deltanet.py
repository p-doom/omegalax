"""Gated Delta Net for Qwen3.5: chunked gated delta rule (linear attention) =
depthwise causal Conv1D + recurrent delta-rule update.

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
from .kernels.cp import _segment_state_transition, _exclusive_prefix_S_in
from .config import Qwen3_5TextConfig
from .norms import RMSNormGated

P = PartitionSpec
wp = nnx.with_partitioning


def _causal_depthwise_conv1d(x_BCT: jax.Array, weight_CK: jax.Array) -> jax.Array:
    """Depthwise causal conv1d: x (B, C, T), per-channel weight (C, K) -> (B, C, T)."""
    K = weight_CK.shape[1]
    T = x_BCT.shape[2]
    x_padded = jnp.pad(x_BCT, ((0, 0), (0, 0), (K - 1, 0)))
    result = jnp.zeros_like(x_BCT)
    for k in range(K):
        result = result + weight_CK[None, :, k : k + 1] * x_padded[:, :, k : k + T]
    return result


def _causal_depthwise_conv1d_cp(
    x_BCT: jax.Array, weight_CK: jax.Array, cp_axis: str, cp_size: int
) -> jax.Array:
    """Depthwise causal conv1d over a cp-sharded segment (inside a shard_map over
    ``cp``). The causal conv needs the previous ``K-1`` tokens from the LEFT
    neighbor, so we ``ppermute`` a ``K-1``-token halo r-1 -> r (rank 0's halo is
    zeros, matching the global left zero-pad). Bit-identical to the full conv.
    """
    K = weight_CK.shape[1]
    Tloc = x_BCT.shape[2]
    r = jax.lax.axis_index(cp_axis)
    tail_BCH = x_BCT[:, :, Tloc - (K - 1) :]  # last K-1 cols of this rank
    # send each rank's tail to its RIGHT neighbor (i -> i+1), i.e. receive from left.
    halo_BCH = jax.lax.ppermute(
        tail_BCH, cp_axis, [(i, (i + 1) % cp_size) for i in range(cp_size)]
    )
    halo_BCH = jnp.where(r == 0, jnp.zeros_like(halo_BCH), halo_BCH)
    x_padded = jnp.concatenate([halo_BCH, x_BCT], axis=2)  # (B, C, K-1 + Tloc)
    result = jnp.zeros_like(x_BCT)
    for k in range(K):
        result = result + weight_CK[None, :, k : k + 1] * x_padded[:, :, k : k + Tloc]
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
        # q/k/v projected separately (not one fused in_proj_qkv) so each emits its
        # output already head-sharded on tp; a fused projection would tp-shard the
        # concatenated conv_dim contiguously and force reshards on the per-head
        # split. The fused HF checkpoint weight is sliced into these at load time.
        self.in_proj_q = nnx.Linear(
            D, self.key_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype, kernel_init=in_proj_init
        )
        self.in_proj_k = nnx.Linear(
            D, self.key_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype, kernel_init=in_proj_init
        )
        self.in_proj_v = nnx.Linear(
            D, self.value_dim, use_bias=False, rngs=rngs, dtype=cfg.dtype, kernel_init=in_proj_init
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

        # Depthwise conv weights, per q/k/v. The channel axis is tp-sharded to line
        # up with the head-sharded activations (channels are head-major, so a
        # contiguous tp split of the channel axis is exactly a head tp split).
        self.conv_weight_q = nnx.Param(
            init(rngs.params(), (self.key_dim, self.conv_kernel_size)),
            sharding=("mlp", None),
        )
        self.conv_weight_k = nnx.Param(
            init(rngs.params(), (self.key_dim, self.conv_kernel_size)),
            sharding=("mlp", None),
        )
        self.conv_weight_v = nnx.Param(
            init(rngs.params(), (self.value_dim, self.conv_kernel_size)),
            sharding=("mlp", None),
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
        cp_axis = heads_shd[1]  # "cp" under a CP config, else None (Stage 1 axis)
        beta_g_shd = P(batch_axis, None, head_axis)

        mesh = jax.sharding.get_abstract_mesh()
        cp_size = mesh.shape[cp_axis] if cp_axis is not None else 1
        use_cp = cp_size > 1

        norm_eps = self.norm.eps
        head_k_dim = self.head_k_dim
        gqa_factor = self.gqa_factor
        num_k_heads = self.num_k_heads
        num_v_heads = self.num_v_heads
        head_v_dim = self.head_v_dim

        def _apply_gqa(q_BTHA, k_BTHA, v_BTHU):
            B, T = q_BTHA.shape[:2]
            local_k_heads = q_BTHA.shape[2]
            local_v_heads = v_BTHU.shape[2]
            assert k_BTHA.shape[2] == local_k_heads
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
            return q_BTHA, k_BTHA

        def _norm_gate(out_BTHU, z_BTHU, nw):
            """Inline RMSNormGated (positionwise → CP-safe)."""
            BL = out_BTHU.shape[0] * out_BTHU.shape[1]
            H, U = out_BTHU.shape[2], out_BTHU.shape[3]
            core_flat = out_BTHU.reshape(BL * H, U)
            z_flat = z_BTHU.reshape(BL * H, U)
            dtype = core_flat.dtype
            x_f32 = core_flat.astype(jnp.float32)
            variance = jnp.mean(x_f32**2, axis=-1, keepdims=True)
            normed = (x_f32 * jax.lax.rsqrt(variance + norm_eps)).astype(dtype)
            normed = nw.astype(dtype) * normed
            gated = normed * jax.nn.silu(z_flat.astype(jnp.float32))
            return gated.astype(dtype).reshape(out_BTHU.shape[0], out_BTHU.shape[1], H * U)

        from jax.experimental.shard_map import shard_map

        A_H = -jnp.exp(self.A_log[...].astype(jnp.float32))
        norm_w = self.norm.weight[...]

        if not use_cp:
            # ---- Non-CP path (unchanged): conv outside the head shard_map. ----
            def _proj_conv_heads(proj, conv_w, n_heads, head_dim):
                x_BTF = proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
                x_BTHc = jax.lax.reshape(
                    x_BTF, (B, T, n_heads, head_dim), out_sharding=heads_shd
                )
                x_BCT = x_BTHc.reshape(B, T, n_heads * head_dim).transpose(0, 2, 1)
                x_BCT = _causal_depthwise_conv1d(x_BCT, conv_w[...].astype(x_BCT.dtype))
                x_BCT = nnx.silu(x_BCT)
                return jax.lax.reshape(
                    x_BCT.transpose(0, 2, 1),
                    (B, T, n_heads, head_dim),
                    out_sharding=heads_shd,
                )

            q_BTHA = _proj_conv_heads(
                self.in_proj_q, self.conv_weight_q, self.num_k_heads, self.head_k_dim
            )
            k_BTHA = _proj_conv_heads(
                self.in_proj_k, self.conv_weight_k, self.num_k_heads, self.head_k_dim
            )
            v_BTHU = _proj_conv_heads(
                self.in_proj_v, self.conv_weight_v, self.num_v_heads, self.head_v_dim
            )
            z_BTHU = jax.lax.reshape(
                self.in_proj_z(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
                (B, T, self.num_v_heads, self.head_v_dim),
                out_sharding=heads_shd,
            )
            b_BTH = self.in_proj_b(hidden_BTD, out_sharding=beta_g_shd)
            a_BTH = self.in_proj_a(hidden_BTD, out_sharding=beta_g_shd)
            beta_BTH = jax.nn.sigmoid(b_BTH)
            g_BTH = A_H * jax.nn.softplus(a_BTH.astype(jnp.float32) + self.dt_bias[...])

            def _full_deltanet(q_BTHA, k_BTHA, v_BTHU, z_BTHU, g_BTH, beta_BTH, nw):
                q_BTHA, k_BTHA = _apply_gqa(q_BTHA, k_BTHA, v_BTHU)
                out = chunk_gated_delta_rule(q_BTHA, k_BTHA, v_BTHU, g_BTH, beta_BTH)
                return _norm_gate(out, z_BTHU, nw)

            normed_BTD = shard_map(
                _full_deltanet, mesh,
                in_specs=(heads_shd, heads_shd, heads_shd, heads_shd, beta_g_shd, beta_g_shd, P(None)),
                out_specs=self.shd_cfg.act_btf,
                check_rep=False,
            )(q_BTHA, k_BTHA, v_BTHU, z_BTHU, g_BTH, beta_BTH, norm_w)

            out_BTD = self.out_proj(normed_BTD, out_sharding=self.shd_cfg.act_btd)
            return out_BTD

        # ---- Context-parallel path (Stage 2): conv-halo + boundary-state ring
        # inside the head shard_map. Project WITHOUT conv (the conv moves inside the
        # shard_map so its causal halo can be ppermute'd from the left cp neighbor);
        # projections stay cp-seq-sharded on the T axis and tp head-sharded. ----
        q_pre_BTHA = jax.lax.reshape(
            self.in_proj_q(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
            (B, T, self.num_k_heads, self.head_k_dim), out_sharding=heads_shd,
        )
        k_pre_BTHA = jax.lax.reshape(
            self.in_proj_k(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
            (B, T, self.num_k_heads, self.head_k_dim), out_sharding=heads_shd,
        )
        v_pre_BTHU = jax.lax.reshape(
            self.in_proj_v(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
            (B, T, self.num_v_heads, self.head_v_dim), out_sharding=heads_shd,
        )
        z_BTHU = jax.lax.reshape(
            self.in_proj_z(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
            (B, T, self.num_v_heads, self.head_v_dim), out_sharding=heads_shd,
        )
        # Under CP the beta/g per-token arrays MUST be cp-seq-sharded on the T
        # axis (so each rank holds its segment); the shard_map in_spec matches.
        beta_g_cp_shd = P(batch_axis, cp_axis, head_axis)
        b_BTH = self.in_proj_b(hidden_BTD, out_sharding=beta_g_cp_shd)
        a_BTH = self.in_proj_a(hidden_BTD, out_sharding=beta_g_cp_shd)
        beta_BTH = jax.nn.sigmoid(b_BTH)
        g_BTH = A_H * jax.nn.softplus(a_BTH.astype(jnp.float32) + self.dt_bias[...])

        conv_wq = self.conv_weight_q[...]
        conv_wk = self.conv_weight_k[...]
        conv_wv = self.conv_weight_v[...]
        cp_name = cp_axis
        chunk_size = 64

        def _cp_deltanet(
            q_pre, k_pre, v_pre, z_BTHU, g_BTH, beta_BTH, nw, wq, wk, wv
        ):
            """Local segment: conv(halo) → GQA → boundary-state ring → kernel → norm.

            Runs inside the shard_map, so q/k/v are this rank's LOCAL (cp) segment
            with LOCAL (tp) heads. The conv gets its K-1 causal halo from the left
            cp neighbor; the recurrent state is chained across cp via the affine
            (A_r,B_r) associative-scan ring (see kernels/cp.py).
            """
            b, tl, nkh, ak = q_pre.shape
            nvh = v_pre.shape[2]

            def _conv_silu(x_BTHc, w_CK, n_heads, hd):
                x_BCT = x_BTHc.reshape(b, tl, n_heads * hd).transpose(0, 2, 1)
                x_BCT = _causal_depthwise_conv1d_cp(
                    x_BCT, w_CK.astype(x_BCT.dtype), cp_name, cp_size
                )
                x_BCT = nnx.silu(x_BCT)
                return x_BCT.transpose(0, 2, 1).reshape(b, tl, n_heads, hd)

            q_BTHA = _conv_silu(q_pre, wq, nkh, ak)
            k_BTHA = _conv_silu(k_pre, wk, nkh, ak)
            v_BTHU = _conv_silu(v_pre, wv, nvh, v_pre.shape[3])

            q_BTHA, k_BTHA = _apply_gqa(q_BTHA, k_BTHA, v_BTHU)

            # Boundary-state ring: aggregate (A_r,B_r) of THIS segment, resolve the
            # exclusive prefix over cp to get this rank's incoming state S_in, then
            # run the local kernel seeded with S_in.
            A_r, B_r = _segment_state_transition(
                q_BTHA, k_BTHA, v_BTHU, g_BTH, beta_BTH, chunk_size
            )
            S_in = _exclusive_prefix_S_in(A_r, B_r, cp_name)
            out = chunk_gated_delta_rule(
                q_BTHA, k_BTHA, v_BTHU, g_BTH, beta_BTH,
                chunk_size=chunk_size, state_init_BHAU=S_in,
            )
            return _norm_gate(out, z_BTHU, nw)

        normed_BTD = shard_map(
            _cp_deltanet, mesh,
            in_specs=(
                heads_shd, heads_shd, heads_shd, heads_shd, beta_g_cp_shd, beta_g_cp_shd,
                P(None), P(head_axis, None), P(head_axis, None), P(head_axis, None),
            ),
            out_specs=self.shd_cfg.act_btf,
            check_rep=False,
        )(
            q_pre_BTHA, k_pre_BTHA, v_pre_BTHU, z_BTHU, g_BTH, beta_BTH,
            norm_w, conv_wq, conv_wk, conv_wv,
        )

        out_BTD = self.out_proj(normed_BTD, out_sharding=self.shd_cfg.act_btd)
        return out_BTD
