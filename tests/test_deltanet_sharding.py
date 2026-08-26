"""Numerical equivalence of the DeltaNet sharding rework (CPU, faked devices).

The Gated-DeltaNet input projections were changed from a single fused
``in_proj_qkv`` (whose tp-sharded ``conv_dim`` had to be split into per-head
q/k/v via collective-permute reshards) to three head-native projections
``in_proj_q/k/v`` (+ per-segment conv weights), so q/k/v arrive at the
``shard_map`` already head-sharded on ``tp`` with no reshard round-trip.

This test asserts the reworked module is numerically identical (forward AND
backward, to ~roundoff) to a reference that reproduces the original *fused*
math, on a Qwen3.5 smoke config, under TP=1 and TP=2. The reference below is a
faithful copy of the pre-rework fused forward (git HEAD d98d0c0); we copy the
same weights into both and compare.

The ``shard_map`` cannot be evaluated eagerly, so everything runs under
``jax.jit`` / ``jax.grad``.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from flax import nnx
from jax.sharding import PartitionSpec as P

from omegalax.distributed.mesh import make_mesh, mesh_rules
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.qwen3_5.deltanet import GatedDeltaNet, _causal_depthwise_conv1d
from omegalax.models.qwen3_5.kernels import chunk_gated_delta_rule
from omegalax.models.shard_config import ShardConfig, shard_config_for_mesh


def _smoke_cfg() -> Qwen3_5TextConfig:
    # Head counts divisible by tp=2; A == U keeps the smoke config simple.
    return Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        layer_types=("linear_attention",),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_value_head_dim=16,
        shd_cfg=ShardConfig.default(),
        dtype=jnp.float32,
    )


class _BaselineGatedDeltaNet(nnx.Module):
    """Reference: the pre-rework *fused* forward (single in_proj_qkv + conv)."""

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
        wp = nnx.with_partitioning
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
            init(rngs.params(), (conv_dim, self.conv_kernel_size)), sharding=(None, None)
        )
        self.dt_bias = nnx.Param(jnp.ones(self.num_v_heads), sharding=(None,))
        self.A_log = nnx.Param(
            jnp.log(jax.random.uniform(rngs.params(), (self.num_v_heads,)) * 16), sharding=(None,)
        )
        from omegalax.models.qwen3_5.norms import RMSNormGated

        self.norm = RMSNormGated(self.head_v_dim, cfg.rms_norm_eps, rngs=rngs, sharding=(None,))
        self.out_proj = nnx.Linear(
            self.value_dim, D, use_bias=False, rngs=rngs, dtype=cfg.dtype,
            kernel_init=wp(init, ("mlp", "embed")),
        )

    @jax.named_scope("gated_delta_net")
    def __call__(self, hidden_BTD, attention_mask_BT=None):
        if attention_mask_BT is not None and attention_mask_BT.shape[1] > 1:
            hidden_BTD = hidden_BTD * attention_mask_BT[:, :, None]
        B, T, _ = hidden_BTD.shape
        heads_shd = self.shd_cfg.act_btnh
        batch_axis, head_axis = heads_shd[0], heads_shd[2]
        beta_g_shd = P(batch_axis, None, head_axis)
        mixed_qkv_BCT = self.in_proj_qkv(
            hidden_BTD, out_sharding=self.shd_cfg.act_btf
        ).transpose(0, 2, 1)
        z_BTHU = jax.lax.reshape(
            self.in_proj_z(hidden_BTD, out_sharding=self.shd_cfg.act_btf),
            (B, T, self.num_v_heads, self.head_v_dim), out_sharding=heads_shd,
        )
        b_BTH = self.in_proj_b(hidden_BTD, out_sharding=beta_g_shd)
        a_BTH = self.in_proj_a(hidden_BTD, out_sharding=beta_g_shd)
        mixed_qkv_BCT = nnx.silu(
            _causal_depthwise_conv1d(mixed_qkv_BCT, self.conv_weight[...].astype(mixed_qkv_BCT.dtype))
        )
        mixed_qkv_BTC = mixed_qkv_BCT.transpose(0, 2, 1)
        q_BTP, k_BTP, v_BTO = jnp.split(
            mixed_qkv_BTC, [self.key_dim, self.key_dim * 2], axis=-1
        )
        q_BTHA = jax.lax.reshape(q_BTP, (B, T, self.num_k_heads, self.head_k_dim), out_sharding=heads_shd)
        k_BTHA = jax.lax.reshape(k_BTP, (B, T, self.num_k_heads, self.head_k_dim), out_sharding=heads_shd)
        v_BTHU = jax.lax.reshape(v_BTO, (B, T, self.num_v_heads, self.head_v_dim), out_sharding=heads_shd)

        from jax.experimental.shard_map import shard_map

        mesh = jax.sharding.get_abstract_mesh()
        beta_BTH = jax.nn.sigmoid(b_BTH)
        A_H = -jnp.exp(self.A_log[...].astype(jnp.float32))
        g_BTH = A_H * jax.nn.softplus(a_BTH.astype(jnp.float32) + self.dt_bias[...])
        norm_w = self.norm.weight[...]
        norm_eps = self.norm.eps
        head_k_dim = self.head_k_dim
        gqa_factor = self.gqa_factor

        def _full_deltanet(q_BTHA, k_BTHA, v_BTHU, z_BTHU, g_BTH, beta_BTH, nw):
            B, T = q_BTHA.shape[:2]
            local_k_heads = q_BTHA.shape[2]
            local_v_heads = v_BTHU.shape[2]
            if gqa_factor > 1:
                q_BTHA = jnp.broadcast_to(
                    q_BTHA[:, :, :, None, :], (B, T, local_k_heads, gqa_factor, head_k_dim)
                ).reshape(B, T, local_v_heads, head_k_dim)
                k_BTHA = jnp.broadcast_to(
                    k_BTHA[:, :, :, None, :], (B, T, local_k_heads, gqa_factor, head_k_dim)
                ).reshape(B, T, local_v_heads, head_k_dim)
            out = chunk_gated_delta_rule(q_BTHA, k_BTHA, v_BTHU, g_BTH, beta_BTH)
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

        normed_BTD = shard_map(
            _full_deltanet, mesh,
            in_specs=(heads_shd, heads_shd, heads_shd, heads_shd, beta_g_shd, beta_g_shd, P(None)),
            out_specs=self.shd_cfg.act_btf, check_rep=False,
        )(q_BTHA, k_BTHA, v_BTHU, z_BTHU, g_BTH, beta_BTH, norm_w)
        return self.out_proj(normed_BTD, out_sharding=self.shd_cfg.act_btd)


def _copy_baseline_to_reworked(baseline: _BaselineGatedDeltaNet, rework: GatedDeltaNet) -> None:
    """Slice the fused baseline weights into the reworked per-q/k/v params
    exactly the way the safetensors loader does."""
    kd, vd = baseline.key_dim, baseline.value_dim
    W = np.asarray(jax.device_get(baseline.in_proj_qkv.kernel[...]))  # (D, conv_dim)
    rework.in_proj_q.kernel[...] = jnp.asarray(W[:, :kd])
    rework.in_proj_k.kernel[...] = jnp.asarray(W[:, kd : 2 * kd])
    rework.in_proj_v.kernel[...] = jnp.asarray(W[:, 2 * kd :])
    cw = np.asarray(jax.device_get(baseline.conv_weight[...]))  # (conv_dim, K)
    rework.conv_weight_q[...] = jnp.asarray(cw[:kd])
    rework.conv_weight_k[...] = jnp.asarray(cw[kd : 2 * kd])
    rework.conv_weight_v[...] = jnp.asarray(cw[2 * kd :])
    for name in ("in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        getattr(rework, name).kernel[...] = getattr(baseline, name).kernel[...]
    rework.dt_bias[...] = baseline.dt_bias[...]
    rework.A_log[...] = baseline.A_log[...]
    rework.norm.weight[...] = baseline.norm.weight[...]


def _run(mesh_shape, cfg_base):
    """Build baseline + reworked on the mesh, run jitted forward and jitted
    grad-w.r.t.-parameters, and return host-numpy results.

    Backward is taken w.r.t. the parameters (the training-relevant gradient).
    Grad w.r.t. the *input* activations is not compared: ``nnx.Linear`` with an
    ``out_sharding`` whose output feature is tp-sharded has an ambiguous
    input-VJP under TP>1 (a pre-existing, codebase-wide JAX limitation that
    predates and is orthogonal to this sharding rework); it raises for both the
    baseline and the reworked module identically.
    """
    tp, fsdp, dp = mesh_shape
    mesh = make_mesh(tp_size=tp, fsdp_size=fsdp, dp_size=dp)
    B, T, D = 4, 32, cfg_base.hidden_size
    rng = np.random.RandomState(0)
    hidden_np = rng.randn(B, T, D).astype(np.float32) * 0.5
    with mesh_rules(mesh):
        cfg = dataclasses.replace(cfg_base, shd_cfg=shard_config_for_mesh(cfg_base.shd_cfg, mesh))
        baseline = _BaselineGatedDeltaNet(cfg, rngs=nnx.Rngs(0))
        rework = GatedDeltaNet(cfg, rngs=nnx.Rngs(1))
        _copy_baseline_to_reworked(baseline, rework)
        hidden = jnp.asarray(hidden_np)

        gdb, sb = nnx.split(baseline)
        gdr, sr = nnx.split(rework)

        fb = jax.jit(lambda s, h: nnx.merge(gdb, s)(h))(sb, hidden)
        fr = jax.jit(lambda s, h: nnx.merge(gdr, s)(h))(sr, hidden)
        grad_b = jax.jit(jax.grad(lambda s, h: nnx.merge(gdb, s)(h).sum()))(sb, hidden)
        grad_r = jax.jit(jax.grad(lambda s, h: nnx.merge(gdr, s)(h).sum()))(sr, hidden)

    pb = nnx.to_pure_dict(grad_b)
    pr = nnx.to_pure_dict(grad_r)
    return np.asarray(fb), np.asarray(fr), pb, pr


def _get(pure, *path):
    node = pure
    for p in path:
        node = node[p]
    return np.asarray(jax.device_get(node["value"] if isinstance(node, dict) and "value" in node else node))


class DeltaNetShardingEquivalenceTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("tp1", (1, 1, 4)),
        ("tp2", (2, 2, 1)),
    )
    def test_fused_vs_split_equivalence(self, mesh_shape):
        cfg = _smoke_cfg()
        kd, vd = cfg.linear_key_head_dim * cfg.linear_num_key_heads, (
            cfg.linear_value_head_dim * cfg.linear_num_value_heads
        )
        fb, fr, gb, gr = _run(mesh_shape, cfg)

        # ---- Forward equivalence ----
        self.assertEqual(fb.shape, fr.shape)
        self.assertTrue(np.isfinite(fr).all())
        fwd_diff = float(np.abs(fb - fr).max())
        fwd_scale = max(float(np.abs(fb).max()), 1e-12)

        # ---- Backward (param-grad) equivalence ----
        # Shared params must match directly.
        shared = {
            "in_proj_z": ("in_proj_z", "kernel"),
            "in_proj_b": ("in_proj_b", "kernel"),
            "in_proj_a": ("in_proj_a", "kernel"),
            "out_proj": ("out_proj", "kernel"),
            "dt_bias": ("dt_bias",),
            "A_log": ("A_log",),
            "norm": ("norm", "weight"),
        }
        grad_pairs = {}
        for label, path in shared.items():
            grad_pairs[label] = (_get(gb, *path), _get(gr, *path))

        # Fused qkv grad must equal the concatenation of the split q/k/v grads.
        grad_pairs["in_proj_qkv"] = (
            _get(gb, "in_proj_qkv", "kernel"),  # (D, conv_dim)
            np.concatenate([_get(gr, f"in_proj_{p}", "kernel") for p in ("q", "k", "v")], axis=1),
        )
        # Fused conv grad must equal the concatenation of the split conv grads.
        grad_pairs["conv_weight"] = (
            _get(gb, "conv_weight"),  # (conv_dim, K)
            np.concatenate([_get(gr, f"conv_weight_{p}") for p in ("q", "k", "v")], axis=0),
        )

        def _reldiff(a, b):
            return float(np.abs(a - b).max() / max(float(np.abs(a).max()), 1e-12))

        bwd_reldiffs = {k: _reldiff(*v) for k, v in grad_pairs.items()}
        max_bwd_rel = max(bwd_reldiffs.values())
        print(
            f"[deltanet-sharding {mesh_shape}] fwd max_abs_diff={fwd_diff:.3e} "
            f"(scale={fwd_scale:.3e}); bwd param-grad max_rel_diff={max_bwd_rel:.3e} "
            f"(per-param: { {k: float(f'{v:.2e}') for k, v in bwd_reldiffs.items()} })",
            flush=True,
        )

        # Same math, only sharding/layout differs → roundoff-level agreement
        # (fp32; different reduction orders in the conv/scan cost a few ULP).
        np.testing.assert_allclose(fr, fb, atol=1e-4, rtol=1e-4)
        for label, (gb_p, gr_p) in grad_pairs.items():
            np.testing.assert_allclose(
                gr_p, gb_p, atol=1e-4, rtol=1e-4, err_msg=f"param-grad mismatch: {label}"
            )


if __name__ == "__main__":
    absltest.main()
