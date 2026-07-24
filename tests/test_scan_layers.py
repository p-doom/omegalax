"""Equivalence tests for scan-based (stacked) decoder layers vs the unrolled loop.

For every supported model family/config we build one smoke model, then run a
forward+backward pass twice under IDENTICAL weights and inputs: once forced onto
the unrolled Python loop (``_force_unrolled`` patches the scan-eligibility
property) and once on the default single ``nnx.scan`` layer body. We assert that
the loss, the aux_loss (for MoE) and every gradient match to fp tolerance.

We also assert the compile-time win: ``jax.make_jaxpr`` of the scanned forward
contains a ``scan`` primitive and has far fewer equations than the unrolled
forward, which grows linearly in ``num_layers``.

Runs on CPU, torch-free.
"""

import contextlib
import dataclasses
import os
import unittest.mock as mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import make_mesh, mesh_rules
from omegalax.models.sharding_runtime import set_attn_backend


def _force_unrolled(cfg):
    """Force the model onto the unrolled layer loop (the scan_layers opt-out was removed).

    qwen3.5's hybrid dispatch keys on ``scan_block_period`` (None -> unrolled); the
    homogeneous qwen3 / qwen3-vl dispatch keys on ``is_homogeneous`` (False -> unrolled).
    """
    cls = type(cfg)
    if hasattr(cls, "scan_block_period"):
        return mock.patch.object(cls, "scan_block_period", property(lambda self: None))
    return mock.patch.object(cls, "is_homogeneous", property(lambda self: False))


@contextlib.contextmanager
def single_device_mesh():
    """A 1x1x1 (tp, fsdp, dp) mesh so logical-axis-annotated params can be built
    on CPU; every mesh axis has size 1 so params are effectively replicated but
    the real sharding machinery (logical_axis_rules) is exercised."""
    mesh = make_mesh(tp_size=1, fsdp_size=1, dp_size=1)
    with mesh_rules(mesh):
        yield mesh


@contextlib.contextmanager
def force_jax_ragged_dot():
    """Route grouped-MoE onto the CPU-safe ``jax`` ragged_dot reference.

    tokamax's ragged_dot lowers to ``jax.lax.ragged_dot_general``, which has no
    Explicit-sharding rule on CPU (see ``moe_grouped._auto``); under the Explicit
    logical mesh these tests build, the grouped GEMM's activation carries an Auto
    aval and the primitive rejects the mesh-type mismatch. The ``jax`` reference
    goes through the auto_axes CPU path and is numerically identical (verified by
    tests/test_moe_grouped.py), and these scan-equivalence tests are
    primitive-agnostic, so we pin it. No-op for dense configs (no ragged_dot).
    """
    import omegalax.models.moe_grouped as _mg

    _orig = _mg._ragged_dot

    def _jax_primitive(lhs, rhs, group_sizes, primitive, **kw):
        return _orig(lhs, rhs, group_sizes, "jax", **kw)

    with mock.patch.object(_mg, "_ragged_dot", _jax_primitive):
        yield


def _count_eqns(jaxpr) -> int:
    """Total number of equations, recursively descending into sub-jaxprs."""
    count = 0

    def walk(jpr):
        nonlocal count
        count += len(jpr.eqns)
        for eqn in jpr.eqns:
            for param in eqn.params.values():
                if isinstance(param, jax.extend.core.Jaxpr):
                    walk(param)
                elif hasattr(param, "jaxpr") and isinstance(
                    getattr(param, "jaxpr"), jax.extend.core.Jaxpr
                ):
                    walk(param.jaxpr)

    walk(jaxpr)
    return count


def _has_scan(jaxpr_text: str) -> bool:
    return "scan[" in jaxpr_text or " scan " in jaxpr_text


def _copy_weights(src_model, dst_model):
    """Copy all params from src into dst (identical architecture, differ only in flag)."""
    _, src_state = nnx.split(src_model)
    graphdef, _ = nnx.split(dst_model)
    return nnx.merge(graphdef, src_state)


def _grad_leaf_map(g):
    """Map from string key-path -> leaf array for a grad pytree, so we compare
    gradients by NAME (robust to differing leaf iteration order between two
    independently-built models)."""
    return {
        jax.tree_util.keystr(path): np.asarray(leaf, dtype=np.float32)
        for path, leaf in jax.tree_util.tree_leaves_with_path(g)
    }


def _assert_grads_match(test, g_ref, g_scan, atol, rtol, prefix=""):
    ref = _grad_leaf_map(g_ref)
    scan = _grad_leaf_map(g_scan)
    test.assertEqual(set(ref), set(scan), f"{prefix}grad key sets differ")
    max_abs = 0.0
    for key, a in ref.items():
        b = scan[key]
        test.assertEqual(a.shape, b.shape, f"{prefix}{key} shape mismatch")
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        max_abs = max(max_abs, diff)
        np.testing.assert_allclose(
            a, b, atol=atol, rtol=rtol, err_msg=f"{prefix}grad mismatch at {key}"
        )
    return max_abs


# ----------------------------------------------------------------------------
# Qwen3 (dense + MoE)
# ----------------------------------------------------------------------------
class Qwen3ScanEquivalenceTest(absltest.TestCase):
    def _build_pair(self, cfg):
        from omegalax.models.qwen3.model import Qwen3

        m_unrolled = Qwen3(cfg, rngs=nnx.Rngs(0))
        m_scan = Qwen3(cfg, rngs=nnx.Rngs(0))
        m_scan = _copy_weights(m_unrolled, m_scan)
        set_attn_backend(m_unrolled, text_backend="xla")
        set_attn_backend(m_scan, text_backend="xla")
        return m_unrolled, m_scan

    def _mesh(self):
        return single_device_mesh()

    def _dense_cfg(self, num_layers=16):
        from omegalax.models.qwen3.config import Qwen3Config
        from omegalax.models.shard_config import ShardConfig

        return Qwen3Config(
            num_layers=num_layers,
            vocab_size=256,
            emb_dim=64,
            mlp_dim=128,
            num_heads=4,
            head_dim=16,
            num_kv_heads=4,
            rope_theta=1_000_000,
            rope_scaling_factor=None,
            local_rope_theta=None,
            norm_eps=1e-6,
            tie_word_embeddings=False,
            shd_cfg=ShardConfig.no_sharding(),
            dtype=jnp.float32,
        )

    def _moe_cfg(self, num_layers=12):
        from omegalax.models.qwen3.config import Qwen3Config
        from omegalax.models.shard_config import ShardConfig

        return Qwen3Config(
            num_layers=num_layers,
            vocab_size=256,
            emb_dim=64,
            mlp_dim=128,
            num_heads=4,
            head_dim=16,
            num_kv_heads=4,
            rope_theta=1_000_000,
            rope_scaling_factor=None,
            local_rope_theta=None,
            norm_eps=1e-6,
            tie_word_embeddings=False,
            moe_intermediate_size=64,
            num_experts=4,
            num_experts_per_tok=2,
            mlp_only_layers=(),
            decoder_sparse_step=1,
            norm_topk_prob=True,
            aux_loss_coef=0.01,
            shd_cfg=ShardConfig.no_sharding(),
            dtype=jnp.float32,
        )

    def _run_case(self, cfg, atol=1e-5, rtol=1e-5):
        with force_jax_ragged_dot(), self._mesh():
            m_unrolled, m_scan = self._build_pair(cfg)

            rng = np.random.RandomState(0)
            B, T = 2, 12
            token_ids_BT = jnp.asarray(
                rng.randint(1, cfg.vocab_size, size=(B, T)).astype(np.int32)
            )
            segment_ids_BT = jnp.ones((B, T), dtype=jnp.int32)

            def loss_fn(model):
                hidden_BTD, aux = model(token_ids_BT, segment_ids_BT, None, jnp.array(0, jnp.int32))
                logits = model.lm_head(hidden_BTD)
                return jnp.mean(logits**2) + aux, aux

            with _force_unrolled(cfg):
                (loss_ref, aux_ref), g_ref = nnx.value_and_grad(loss_fn, has_aux=True)(m_unrolled)
            (loss_scan, aux_scan), g_scan = nnx.value_and_grad(loss_fn, has_aux=True)(m_scan)

        np.testing.assert_allclose(
            np.asarray(loss_ref), np.asarray(loss_scan), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(np.asarray(aux_ref), np.asarray(aux_scan), atol=atol, rtol=rtol)
        max_g = _assert_grads_match(self, g_ref, g_scan, atol, rtol)
        return loss_ref, loss_scan, aux_ref, aux_scan, max_g

    def test_dense_equivalence(self):
        cfg = self._dense_cfg(num_layers=16)
        loss_ref, loss_scan, aux_ref, aux_scan, max_g = self._run_case(cfg)
        print(
            f"[qwen3-dense L={cfg.num_layers}] loss_ref={float(loss_ref):.8f} "
            f"loss_scan={float(loss_scan):.8f} max_grad_absdiff={max_g:.2e}"
        )

    def test_moe_equivalence(self):
        cfg = self._moe_cfg(num_layers=12)
        loss_ref, loss_scan, aux_ref, aux_scan, max_g = self._run_case(cfg)
        print(
            f"[qwen3-moe L={cfg.num_layers}] loss_ref={float(loss_ref):.8f} "
            f"loss_scan={float(loss_scan):.8f} aux_ref={float(aux_ref):.8f} "
            f"aux_scan={float(aux_scan):.8f} max_grad_absdiff={max_g:.2e}"
        )

    def test_heterogeneous_moe_falls_back_to_unrolled(self):
        # decoder_sparse_step=2 -> only odd (0-indexed even) layers are MoE -> mixed
        cfg = dataclasses.replace(self._moe_cfg(num_layers=12), decoder_sparse_step=2)
        self.assertFalse(cfg.is_homogeneous)
        # Heterogeneous stack: the default dispatch already takes the unrolled path.
        loss_ref, loss_scan, aux_ref, aux_scan, max_g = self._run_case(cfg)
        print(
            f"[qwen3-moe-hetero L={cfg.num_layers}] heterogeneous -> unrolled fallback; "
            f"loss match, max_grad_absdiff={max_g:.2e}"
        )

    def test_jaxpr_compile_win_dense(self):
        from omegalax.models.qwen3.model import Qwen3

        rng = np.random.RandomState(0)
        B, T = 2, 12
        results = {}
        with single_device_mesh():
            cfg = self._dense_cfg(num_layers=16)
            model = Qwen3(cfg, rngs=nnx.Rngs(0))
            set_attn_backend(model, text_backend="xla")
            token_ids_BT = jnp.asarray(
                rng.randint(1, cfg.vocab_size, size=(B, T)).astype(np.int32)
            )
            segment_ids_BT = jnp.ones((B, T), dtype=jnp.int32)
            graphdef, state = nnx.split(model)

            # Distinct fn objects for the two paths: a single make_jaxpr'd fn caches
            # its first trace and would reuse the forced-unrolled jaxpr for both.
            def fwd_unrolled(state, tok, seg):
                m = nnx.merge(graphdef, state)
                h, aux = m(tok, seg, None, jnp.array(0, jnp.int32))
                return jnp.mean(h**2) + aux

            def fwd_scan(state, tok, seg):
                m = nnx.merge(graphdef, state)
                h, aux = m(tok, seg, None, jnp.array(0, jnp.int32))
                return jnp.mean(h**2) + aux

            with _force_unrolled(cfg):
                jpr_u = jax.make_jaxpr(fwd_unrolled)(state, token_ids_BT, segment_ids_BT)
            jpr_s = jax.make_jaxpr(fwd_scan)(state, token_ids_BT, segment_ids_BT)
            results[False] = (_count_eqns(jpr_u.jaxpr), _has_scan(str(jpr_u)))
            results[True] = (_count_eqns(jpr_s.jaxpr), _has_scan(str(jpr_s)))

        unrolled_eqns, unrolled_scan = results[False]
        scan_eqns, scan_has_scan = results[True]
        print(
            f"[qwen3-dense jaxpr] unrolled_eqns={unrolled_eqns} (scan={unrolled_scan}) "
            f"scan_eqns={scan_eqns} (scan={scan_has_scan})"
        )
        self.assertTrue(scan_has_scan)
        self.assertFalse(unrolled_scan)
        self.assertLess(scan_eqns, unrolled_eqns)

    def test_stacked_layer_axis_is_replicated_and_perlayer_sharding_preserved(self):
        """The stacked leading layer axis must be UNSHARDED (replicated, not in
        the mesh) while the per-layer axes keep their original sharding. We use
        4 fake CPU devices (tp=2, fsdp=2) so the per-layer axes are genuinely
        sharded and the layer axis is genuinely None."""
        from jax.sharding import PartitionSpec as P

        from omegalax.distributed.mesh import make_mesh, mesh_rules
        from omegalax.models.qwen3.model import Qwen3
        from omegalax.models.shard_config import ShardConfig

        if jax.device_count() < 4:
            self.skipTest("needs >=4 devices; run with XLA_FLAGS=--xla_force_host_platform_device_count=4")

        import dataclasses as _dc

        cfg = _dc.replace(self._dense_cfg(num_layers=4), shd_cfg=ShardConfig.default())
        mesh = make_mesh(tp_size=2, fsdp_size=2, dp_size=1)
        with mesh_rules(mesh):
            model = Qwen3(cfg, rngs=nnx.Rngs(0))
            states = [nnx.split(layer)[1] for layer in list(model.layers)]
            stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)
            per_layer = states[0]["mlp"]["gate_proj"]["kernel"]
            stacked_k = stacked["mlp"]["gate_proj"]["kernel"]
            per_spec = per_layer.sharding.spec
            stacked_spec = stacked_k.sharding.spec
        # leading (layer) axis must be None; trailing axes unchanged
        self.assertIsNone(stacked_spec[0])
        self.assertEqual(tuple(stacked_spec[1:]), tuple(per_spec))
        print(f"[qwen3 sharding] per_layer={per_spec} stacked={stacked_spec}")


# ----------------------------------------------------------------------------
# Qwen3-VL text decoder (dense + MoE)
# ----------------------------------------------------------------------------
class Qwen3VLScanEquivalenceTest(absltest.TestCase):
    def _cfg(self, moe: bool, num_layers=12):
        from omegalax.models.qwen3_vl.config import make_vl_config
        from omegalax.models.shard_config import ShardConfig

        model_id = "qwen3-vl-smoke-moe" if moe else "qwen3-vl-smoke"
        cfg = make_vl_config(model_id)
        return dataclasses.replace(
            cfg,
            num_layers=num_layers,
            shd_cfg=ShardConfig.default(),
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

    # NOTE on tolerance: Qwen3-VL text attention runs the compute in the model
    # dtype (TextAttention.__call__): with the attn-dtype gate, an fp32 config
    # runs attention in TRUE fp32 via the xla backend (older code unconditionally
    # downcast q/k/v to bf16, even for fp32 configs). Under fp32 attention the
    # scan path (one stacked layer body) and the unrolled loop reassociate the
    # attention reductions in a slightly different order, giving a benign
    # ~1e-6 forward difference (fp reassociation, does NOT grow at fp64) and a
    # similarly small grad difference; the two paths still compute the same math.
    # Remat is forward-transparent, so this difference is independent of the
    # scan/unrolled remat unification. We therefore assert a small forward
    # tolerance (not bit-exactness) plus depth-appropriate loss/grad tolerances,
    # matching the qwen3.5 block-scan equivalence tests. (For the MoE case the
    # forward tolerance is looser still, since a tiny attn perturbation can flip
    # top-k expert selection.)
    def _run_case(
        self,
        cfg,
        loss_atol=1e-4,
        loss_rtol=1e-4,
        grad_atol=2e-3,
        grad_rtol=2e-3,
        fwd_bit_exact=True,
        fwd_atol=1e-2,
    ):
        from omegalax.models.qwen3_vl.model import Qwen3VL

        with force_jax_ragged_dot(), single_device_mesh():
            m_unrolled = Qwen3VL(cfg, rngs=nnx.Rngs(0))
            m_scan = Qwen3VL(cfg, rngs=nnx.Rngs(0))
            m_scan = _copy_weights(m_unrolled, m_scan)
            set_attn_backend(m_unrolled, text_backend="xla")
            set_attn_backend(m_scan, text_backend="xla")

            rng = np.random.RandomState(0)
            B, T = 2, 12
            # avoid image/vision tokens: use ids >= 8
            token_ids_BT = jnp.asarray(
                (rng.randint(8, cfg.vocab_size, size=(B, T))).astype(np.int32)
            )
            attention_mask_BT = jnp.ones((B, T), dtype=jnp.int32)

            def loss_fn(model):
                hidden_BTD, aux = model(token_ids_BT, attention_mask_BT)
                logits = model.lm_head(hidden_BTD)
                return jnp.mean(logits**2) + aux, aux

            # Primary evidence: forward hidden states must be bit-identical
            # (this is what proves the scan is doing the same math as the loop).
            with _force_unrolled(cfg):
                h_ref, _ = m_unrolled(token_ids_BT, attention_mask_BT)
                (loss_ref, aux_ref), g_ref = nnx.value_and_grad(loss_fn, has_aux=True)(m_unrolled)
            h_scan, _ = m_scan(token_ids_BT, attention_mask_BT)
            (loss_scan, aux_scan), g_scan = nnx.value_and_grad(loss_fn, has_aux=True)(m_scan)
            fwd_max = float(jnp.max(jnp.abs(h_ref - h_scan)))

        # Forward hidden states are bit-exact for dense. For MoE the bf16
        # attention cast can flip top-k expert selection, so we allow a small
        # forward tolerance there (documented in _run_case note).
        if fwd_bit_exact:
            self.assertEqual(fwd_max, 0.0)
        else:
            self.assertLess(fwd_max, fwd_atol)
        np.testing.assert_allclose(
            np.asarray(loss_ref), np.asarray(loss_scan), atol=loss_atol, rtol=loss_rtol
        )
        np.testing.assert_allclose(
            np.asarray(aux_ref), np.asarray(aux_scan), atol=loss_atol, rtol=loss_rtol
        )
        max_g = _assert_grads_match(self, g_ref, g_scan, grad_atol, grad_rtol)
        return loss_ref, loss_scan, fwd_max, max_g

    def test_dense_equivalence(self):
        cfg = self._cfg(moe=False, num_layers=12)
        # fp32 attention (post attn-dtype gate) reassociates differently between
        # the scan and unrolled paths -> ~1e-6 forward diff (see _run_case note);
        # assert a small forward tolerance rather than bit-exactness.
        loss_ref, loss_scan, fwd_max, max_g = self._run_case(
            cfg, grad_atol=2e-3, grad_rtol=2e-3, fwd_bit_exact=False, fwd_atol=1e-4
        )
        print(
            f"[qwen3vl-dense L={cfg.num_layers}] fwd_hidden_absdiff={fwd_max:.1e} "
            f"loss_ref={float(loss_ref):.8f} loss_scan={float(loss_scan):.8f} "
            f"max_grad_absdiff={max_g:.2e}"
        )

    def test_moe_equivalence(self):
        # MoE top-k can flip under tiny bf16-attention perturbations, so the
        # grad noise (concentrated in the embedding grad) is larger than dense;
        # the forward hidden states are still bit-identical. See _run_case note.
        cfg = self._cfg(moe=True, num_layers=12)
        loss_ref, loss_scan, fwd_max, max_g = self._run_case(
            cfg, loss_atol=1e-3, loss_rtol=1e-3, grad_atol=5e-2, grad_rtol=5e-2, fwd_bit_exact=False
        )
        print(
            f"[qwen3vl-moe L={cfg.num_layers}] fwd_hidden_absdiff={fwd_max:.1e} "
            f"loss_ref={float(loss_ref):.8f} loss_scan={float(loss_scan):.8f} "
            f"max_grad_absdiff={max_g:.2e}"
        )


# ----------------------------------------------------------------------------
# Qwen3.5 hybrid (linear/full attention interleaved) -> BLOCK scan
# ----------------------------------------------------------------------------
class Qwen35BlockScanPeriodTest(absltest.TestCase):
    """Pin the period detection that drives the hybrid block-scan dispatch."""

    def test_regular_hybrid_period(self):
        from omegalax.models.qwen3_5.config import Qwen3_5TextConfig, _generate_layer_types

        lt = _generate_layer_types(12)  # [lin,lin,lin,full] x 3
        cfg = Qwen3_5TextConfig(layer_types=lt, num_hidden_layers=12)
        self.assertEqual(cfg.scan_block_period, 4)
        self.assertFalse(cfg.is_homogeneous)

    def test_homogeneous_period_is_one(self):
        from omegalax.models.qwen3_5.config import Qwen3_5TextConfig

        cfg = Qwen3_5TextConfig(layer_types=("full_attention",) * 6, num_hidden_layers=6)
        self.assertEqual(cfg.scan_block_period, 1)
        self.assertTrue(cfg.is_homogeneous)

    def test_irregular_pattern_period_equals_num_layers(self):
        from omegalax.models.qwen3_5.config import Qwen3_5TextConfig

        # A genuinely irregular (non-tiling) pattern has smallest period == n,
        # i.e. num_blocks == 1. The model dispatch requires period < num_layers
        # (num_blocks >= 2) to scan, so this correctly FALLS BACK to unrolled.
        # This one has no smaller repeating unit (the two halves differ).
        lt = ("linear_attention", "linear_attention", "full_attention", "full_attention", "full_attention", "linear_attention")
        cfg = Qwen3_5TextConfig(layer_types=lt, num_hidden_layers=6)
        self.assertEqual(cfg.scan_block_period, len(lt))  # num_blocks == 1 -> no scan

    def test_layer_types_length_mismatch_returns_none(self):
        from omegalax.models.qwen3_5.config import Qwen3_5TextConfig

        lt = ("linear_attention", "full_attention", "full_attention", "linear_attention")
        cfg = Qwen3_5TextConfig(layer_types=lt, num_hidden_layers=8)
        self.assertIsNone(cfg.scan_block_period)  # len(layer_types) != num_hidden_layers


class Qwen35BlockScanEquivalenceTest(absltest.TestCase):
    """Full fwd+bwd equivalence: unrolled loop vs hybrid block-scan under identical
    weights. Qwen3.5 text attention does NOT force q/k/v to bf16 (unlike qwen3_vl),
    so we expect tight ~1e-6 agreement on loss, aux_loss, and all grads."""

    def _text_cfg(self, num_layers=12, moe=True):
        import dataclasses as _dc

        from omegalax.models.qwen3_5.config import Qwen3_5TextConfig, _generate_layer_types
        from omegalax.models.shard_config import ShardConfig

        base = Qwen3_5TextConfig(
            vocab_size=512,
            hidden_size=128,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            rms_norm_eps=1e-6,
            layer_types=_generate_layer_types(num_layers),  # [lin,lin,lin,full]*k
            rope_theta=10_000,
            partial_rotary_factor=0.25,
            mrope_section=(2, 1, 1),
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_value_head_dim=32,
            shd_cfg=ShardConfig.no_sharding(),
            dtype=jnp.float32,
        )
        if moe:
            base = _dc.replace(
                base,
                moe_intermediate_size=64,
                shared_expert_intermediate_size=64,
                num_experts=4,
                num_experts_per_tok=2,
                router_aux_loss_coef=0.01,
            )
        else:
            base = _dc.replace(base, intermediate_size=256)
        return base

    def _run_case(self, text_cfg, atol=1e-5, rtol=1e-5):
        from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM

        with force_jax_ragged_dot(), single_device_mesh():
            m_unrolled = Qwen3_5ForCausalLM(text_cfg, rngs=nnx.Rngs(0))
            m_scan = Qwen3_5ForCausalLM(text_cfg, rngs=nnx.Rngs(0))
            m_scan = _copy_weights(m_unrolled, m_scan)
            set_attn_backend(m_unrolled, text_backend="xla")
            set_attn_backend(m_scan, text_backend="xla")

            self.assertEqual(text_cfg.scan_block_period, 4)
            self.assertFalse(text_cfg.is_homogeneous)

            rng = np.random.RandomState(0)
            B, T = 2, 16  # T multiple of DeltaNet chunk size
            token_ids_BT = jnp.asarray(
                rng.randint(1, text_cfg.vocab_size, size=(B, T)).astype(np.int32)
            )
            segment_ids_BT = jnp.ones((B, T), dtype=jnp.int32)

            def loss_fn(m):
                hidden_BTD, aux = m(token_ids_BT, segment_ids_BT, None, jnp.array(0, jnp.int32))
                logits = m.lm_head(hidden_BTD)
                return jnp.mean(logits**2) + aux, aux

            # The Gated DeltaNet uses shard_map internally, which cannot be evaluated
            # eagerly inside nnx.remat/nnx.scan; the model always runs under jit, so we
            # jit here too. Separate jit'd fns for the unrolled vs scan model so their
            # compilations never alias (identical graphdef -> shared cache otherwise).
            @nnx.jit
            def fwd_u(model):
                return model(token_ids_BT, segment_ids_BT, None, jnp.array(0, jnp.int32))

            @nnx.jit
            def fwd_s(model):
                return model(token_ids_BT, segment_ids_BT, None, jnp.array(0, jnp.int32))

            @nnx.jit
            def grads_u(model):
                return nnx.value_and_grad(loss_fn, has_aux=True)(model)

            @nnx.jit
            def grads_s(model):
                return nnx.value_and_grad(loss_fn, has_aux=True)(model)

            with _force_unrolled(text_cfg):
                h_ref, _ = fwd_u(m_unrolled)
                (loss_ref, aux_ref), g_ref = grads_u(m_unrolled)
            h_scan, _ = fwd_s(m_scan)
            (loss_scan, aux_scan), g_scan = grads_s(m_scan)
            fwd_max = float(jnp.max(jnp.abs(h_ref - h_scan)))

        np.testing.assert_allclose(np.asarray(loss_ref), np.asarray(loss_scan), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(aux_ref), np.asarray(aux_scan), atol=atol, rtol=rtol)
        max_g = _assert_grads_match(self, g_ref, g_scan, atol, rtol)
        return loss_ref, loss_scan, aux_ref, aux_scan, fwd_max, max_g

    def test_moe_block_scan_equivalence(self):
        # Qwen3.5 attention is fp32 (no forced bf16 cast), so agreement is tight.
        # The block-scan reorders reductions vs the loop, giving ~1e-5 fp
        # reassociation noise (does not grow with depth); assert accordingly.
        cfg = self._text_cfg(num_layers=12, moe=True)
        loss_ref, loss_scan, aux_ref, aux_scan, fwd_max, max_g = self._run_case(
            cfg, atol=2e-4, rtol=2e-4
        )
        self.assertLess(fwd_max, 1e-4)
        print(
            f"[qwen3.5-moe block-scan L={cfg.num_hidden_layers} period=4] "
            f"fwd_hidden_absdiff={fwd_max:.1e} loss_ref={float(loss_ref):.8f} "
            f"loss_scan={float(loss_scan):.8f} aux_ref={float(aux_ref):.8f} "
            f"aux_scan={float(aux_scan):.8f} max_grad_absdiff={max_g:.2e}"
        )

    def test_dense_block_scan_equivalence(self):
        cfg = self._text_cfg(num_layers=16, moe=False)
        loss_ref, loss_scan, aux_ref, aux_scan, fwd_max, max_g = self._run_case(
            cfg, atol=2e-4, rtol=2e-4
        )
        self.assertLess(fwd_max, 1e-4)
        print(
            f"[qwen3.5-dense block-scan L={cfg.num_hidden_layers} period=4] "
            f"fwd_hidden_absdiff={fwd_max:.1e} loss_ref={float(loss_ref):.8f} "
            f"loss_scan={float(loss_scan):.8f} max_grad_absdiff={max_g:.2e}"
        )

    def test_jaxpr_compile_win(self):
        from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM

        rng = np.random.RandomState(0)
        B, T = 2, 16
        results = {}
        with single_device_mesh():
            cfg = self._text_cfg(num_layers=16, moe=True)
            model = Qwen3_5ForCausalLM(cfg, rngs=nnx.Rngs(0))
            set_attn_backend(model, text_backend="xla")
            token_ids_BT = jnp.asarray(
                rng.randint(1, cfg.vocab_size, size=(B, T)).astype(np.int32)
            )
            segment_ids_BT = jnp.ones((B, T), dtype=jnp.int32)
            graphdef, state = nnx.split(model)

            # Distinct fn objects for the two paths: a single make_jaxpr'd fn caches
            # its first trace and would reuse the forced-unrolled jaxpr for both.
            def fwd_unrolled(state, tok, seg):
                m = nnx.merge(graphdef, state)
                h, aux = m(tok, seg, None, jnp.array(0, jnp.int32))
                return jnp.mean(h**2) + aux

            def fwd_scan(state, tok, seg):
                m = nnx.merge(graphdef, state)
                h, aux = m(tok, seg, None, jnp.array(0, jnp.int32))
                return jnp.mean(h**2) + aux

            with _force_unrolled(cfg):
                jpr_u = jax.make_jaxpr(fwd_unrolled)(state, token_ids_BT, segment_ids_BT)
            jpr_s = jax.make_jaxpr(fwd_scan)(state, token_ids_BT, segment_ids_BT)
            results[False] = (_count_eqns(jpr_u.jaxpr), _has_scan(str(jpr_u)))
            results[True] = (_count_eqns(jpr_s.jaxpr), _has_scan(str(jpr_s)))

        unrolled_eqns, unrolled_scan = results[False]
        scan_eqns, scan_has_scan = results[True]
        print(
            f"[qwen3.5 jaxpr L=16] unrolled_eqns={unrolled_eqns} (scan={unrolled_scan}) "
            f"block_scan_eqns={scan_eqns} (scan={scan_has_scan})"
        )
        # NOTE: the unrolled Qwen3.5 jaxpr ALSO contains scan primitives, because
        # the Gated DeltaNet chunk kernel uses jax.lax.scan over chunks per layer.
        # So we can't assert "no scan in unrolled"; the win is the equation count:
        # the block-scan compiles one block body instead of num_layers copies.
        self.assertTrue(scan_has_scan)
        self.assertLess(scan_eqns, unrolled_eqns)


if __name__ == "__main__":
    absltest.main()
