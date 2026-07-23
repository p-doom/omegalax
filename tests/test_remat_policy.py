"""Equivalence tests for configurable activation-checkpointing (remat) policies.

Remat policies are numerically transparent: whether an intermediate is *saved*
or *recomputed* in the backward pass, the math is identical. These tests build
small smoke models under the default full-remat policy (``"full"``) and under
a selective policy (``"dots_saveable"``), share identical weights across the
two, and assert that both the forward loss and every gradient match within
floating-point tolerance.

Runs on CPU (``JAX_PLATFORMS=cpu``). The Qwen3.5 DeltaNet kernel auto-selects
its XLA fallback on CPU; the full-attention path's tokamax kernel is GPU-only,
so we swap its ``_attn_backend`` to ``"xla"`` for the duration of the test
(a test-only shim -- it does not touch the model source).
"""

import contextlib
import dataclasses
import os
import unittest.mock as mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Force the CPU-friendly DeltaNet backend regardless of visible devices.
os.environ.setdefault("OMEGALAX_DELTANET_KERNEL", "xla")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from flax import nnx

from omegalax.distributed.mesh import ensure_mesh
from omegalax.models.remat_policy import (
    DEFAULT_REMAT_POLICY,
    available_remat_policies,
    resolve_remat_policy,
)
from omegalax.text import api as text_api

@contextlib.contextmanager
def force_jax_ragged_dot():
    """Route grouped-MoE onto the CPU-safe ``jax`` ragged_dot reference.

    tokamax's ragged_dot lowers to ``jax.lax.ragged_dot_general``, which has no
    Explicit-sharding rule on CPU (see ``moe_grouped._auto``); under the Explicit
    mesh these tests build, the grouped GEMM's activation carries an Auto aval and
    the primitive rejects the mesh-type mismatch. The ``jax`` reference is
    numerically identical (verified by tests/test_moe_grouped.py) and these
    full-vs-selective remat cases are primitive-agnostic. No-op for dense.
    """
    import omegalax.models.moe_grouped as _mg

    _orig = _mg._ragged_dot

    def _jax_primitive(lhs, rhs, group_sizes, primitive, **kw):
        return _orig(lhs, rhs, group_sizes, "jax", **kw)

    with mock.patch.object(_mg, "_ragged_dot", _jax_primitive):
        yield


FULL_POLICY = "full"  # == DEFAULT_REMAT_POLICY
# A selective policy, contrasted against full remat for the equivalence tests.
# Decoupled from DEFAULT_REMAT_POLICY on purpose: the default is now "full", but
# these tests must still exercise a *saved-activation* policy for the comparison.
SELECTIVE_POLICY = "dots_saveable"
_SEED = 0


def _patch_attention_backend_to_xla(model: nnx.Module) -> None:
    """Swap every attention module's kernel backend to XLA (CPU-safe)."""
    for _, mod in nnx.iter_modules(model):
        if hasattr(mod, "_attn_backend"):
            object.__setattr__(mod, "_attn_backend", "xla")


class RematPolicyResolverTest(absltest.TestCase):
    def test_default_is_full(self):
        # Default flipped to full remat: safe at 8B/16k (activations aren't
        # fsdp-sharded, so selective policies spike saved-matmul HBM past 80GB).
        self.assertEqual(DEFAULT_REMAT_POLICY, "full")
        self.assertIsNone(resolve_remat_policy(DEFAULT_REMAT_POLICY))
        # The selective policy used for the equivalence tests still resolves.
        self.assertIsNotNone(resolve_remat_policy(SELECTIVE_POLICY))

    def test_full_resolves_to_none(self):
        # "full" recomputes everything; policy=None == full remat.
        self.assertIsNone(resolve_remat_policy("full"))

    def test_known_names_resolve(self):
        for name in available_remat_policies():
            resolve_remat_policy(name)  # must not raise

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            resolve_remat_policy("does_not_exist")


class RematPolicyEquivalenceTest(parameterized.TestCase):
    """Loss + gradients must be identical under full vs. selective remat."""

    def _build_pair(self, text_cfg):
        """Return (mesh, full_model, selective_model) sharing identical weights.

        ``init_model`` is deterministic in (config, rng); the remat policy only
        changes a static call attribute, not parameter initialization, so
        building both with the same seed yields byte-identical weights.
        """
        cfg_full = dataclasses.replace(text_cfg, remat_policy=FULL_POLICY)
        cfg_sel = dataclasses.replace(text_cfg, remat_policy=SELECTIVE_POLICY)
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        full, _ = text_api.init_model(cfg_full, jax.random.PRNGKey(_SEED),
                                      tp_size=1, fsdp_size=1, dp_size=1)
        sel, _ = text_api.init_model(cfg_sel, jax.random.PRNGKey(_SEED),
                                     tp_size=1, fsdp_size=1, dp_size=1)
        _patch_attention_backend_to_xla(full)
        _patch_attention_backend_to_xla(sel)
        return mesh, full, sel

    @staticmethod
    @nnx.jit
    def _loss_and_grads(model, token_ids_BT, segment_ids_BT):
        # jit is required: the tokamax attention and DeltaNet kernels use
        # ``shard_map`` internally, which cannot be evaluated eagerly. Training
        # runs under jit in production, so this mirrors real usage.
        def loss_fn(m):
            hidden_BTD, aux = m(
                token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32)
            )
            logits_BTV = m.lm_head(hidden_BTD)
            ce = -jax.nn.log_softmax(logits_BTV.astype(jnp.float32), axis=-1).mean()
            return ce + aux.astype(jnp.float32)

        return nnx.value_and_grad(loss_fn)(model)

    def _assert_equivalent(self, mesh, full, sel, token_ids_BT, segment_ids_BT, grad_tol):
        with force_jax_ragged_dot(), jax.set_mesh(mesh):
            loss_full, grads_full = self._loss_and_grads(full, token_ids_BT, segment_ids_BT)
            loss_sel, grads_sel = self._loss_and_grads(sel, token_ids_BT, segment_ids_BT)

        loss_full = float(loss_full)
        loss_sel = float(loss_sel)
        self.assertTrue(np.isfinite(loss_full))
        # Remat is numerically transparent; the loss must match tightly.
        np.testing.assert_allclose(loss_sel, loss_full, rtol=1e-5, atol=1e-5)

        gf = jax.tree_util.tree_leaves(nnx.state(grads_full))
        gs = jax.tree_util.tree_leaves(nnx.state(grads_sel))
        self.assertEqual(len(gf), len(gs))
        self.assertGreater(len(gf), 0)
        # Per-element gradients: recompute-vs-save differ only by low-order
        # rounding in low precision (grads are computed in the model dtype).
        # In fp32 this is bit-exact; in bf16 it is within a small ULP band.
        max_abs_diff = 0.0
        for a, b in zip(gf, gs):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            self.assertEqual(a.shape, b.shape)
            np.testing.assert_allclose(b, a, rtol=grad_tol, atol=grad_tol)
            if a.size:
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(a - b))))
        return loss_full, loss_sel, max_abs_diff

    @staticmethod
    def _grad_tol(cfg) -> float:
        # bf16 machine epsilon ~ 2^-8; allow a few ULP for recompute rounding.
        if cfg.dtype != jnp.bfloat16:
            return 1e-6
        # The hybrid stack (DeltaNet linear-attn + full-attn + MoE) has deeper
        # recompute chains than the dense / MoE-only models, so full-vs-selective
        # remat drift accumulates a wider (but still tiny) bf16 ULP band: ~0.012
        # abs on <0.1% of grad elements. Dense/MoE bf16 stay at 5e-3.
        hybrid = "linear_attention" in getattr(cfg, "layer_types", ())
        return 2e-2 if hybrid else 5e-3

    @parameterized.named_parameters(
        ("qwen3_dense_bf16", "qwen3-smoke", jnp.bfloat16),
        ("qwen3_dense_fp32", "qwen3-smoke", jnp.float32),
        ("qwen3_moe_bf16", "qwen3-smoke-moe", jnp.bfloat16),
        ("qwen3_moe_fp32", "qwen3-smoke-moe", jnp.float32),
    )
    def test_qwen3_equivalence(self, model_id, dtype):
        text_cfg = text_api.resolve_config(model_id)
        self.assertEqual(text_cfg.remat_policy, FULL_POLICY)  # default is now full
        text_cfg = dataclasses.replace(text_cfg, dtype=dtype)
        mesh, full, sel = self._build_pair(text_cfg)
        token_ids_BT, segment_ids_BT = _make_batch(text_cfg.vocab_size)
        loss_full, loss_sel, gdiff = self._assert_equivalent(
            mesh, full, sel, token_ids_BT, segment_ids_BT, self._grad_tol(text_cfg)
        )
        print(
            f"[{model_id}/{jnp.dtype(dtype).name}] loss_full={loss_full:.8f} "
            f"loss_sel={loss_sel:.8f} max|grad_full-grad_sel|={gdiff:.3e}"
        )

    @parameterized.named_parameters(
        ("bf16", jnp.bfloat16),
        ("fp32", jnp.float32),
    )
    def test_qwen3_5_hybrid_equivalence(self, dtype):
        """Hybrid: DeltaNet linear-attn + full-attn + MoE (matches production shape)."""
        text_cfg = text_api.resolve_config("qwen3.5-smoke")
        self.assertEqual(text_cfg.remat_policy, FULL_POLICY)  # default is now full
        # Sanity: config exercises both linear- and full-attention layers, plus MoE.
        self.assertIn("linear_attention", text_cfg.layer_types)
        self.assertIn("full_attention", text_cfg.layer_types)
        self.assertTrue(text_cfg.is_moe)
        text_cfg = dataclasses.replace(text_cfg, dtype=dtype)

        mesh, full, sel = self._build_pair(text_cfg)
        token_ids_BT, segment_ids_BT = _make_batch(text_cfg.vocab_size)
        loss_full, loss_sel, gdiff = self._assert_equivalent(
            mesh, full, sel, token_ids_BT, segment_ids_BT, self._grad_tol(text_cfg)
        )
        print(
            f"[qwen3.5-smoke/{jnp.dtype(dtype).name}] loss_full={loss_full:.8f} "
            f"loss_sel={loss_sel:.8f} max|grad_full-grad_sel|={gdiff:.3e}"
        )


class RematRetraceTest(absltest.TestCase):
    """Two fresh same-config instances must compile once, not once-per-instance.

    Regression guard: storing the remat transform per-instance made each fresh
    module carry a distinct closure in its graphdef, so ``nnx.jit`` saw an
    unequal key and retraced for every new instance (3 traces for 3 models).
    Building ``nnx.remat`` inline keeps graphdefs equal across instances, so the
    jit cache hits and only one trace happens.
    """

    def test_fresh_instances_trace_once(self):
        model_id = "qwen3-smoke"
        text_cfg = dataclasses.replace(
            text_api.resolve_config(model_id), dtype=jnp.float32
        )
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)

        trace_count = {"n": 0}

        @nnx.jit
        def run(model, token_ids_BT, segment_ids_BT):
            trace_count["n"] += 1  # increments only while tracing
            hidden_BTD, _ = model(
                token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32)
            )
            return hidden_BTD.sum()

        token_ids_BT, segment_ids_BT = _make_batch(text_cfg.vocab_size)
        n_instances = 3
        with jax.set_mesh(mesh):
            for seed in range(n_instances):
                # Fresh instance each iteration (distinct RNG seed).
                model, _ = text_api.init_model(
                    text_cfg, jax.random.PRNGKey(seed), tp_size=1, fsdp_size=1, dp_size=1
                )
                _patch_attention_backend_to_xla(model)
                out = run(model, token_ids_BT, segment_ids_BT)
                out.block_until_ready()

        print(
            f"[retrace] {n_instances} fresh instances -> {trace_count['n']} trace(s) "
            f"(inline nnx.remat expected: 1; old per-instance closure: {n_instances})"
        )
        self.assertEqual(
            trace_count["n"],
            1,
            "nnx.jit retraced for a fresh same-config instance; graphdef is not "
            "stable across instances (per-instance remat-closure regression).",
        )


def _make_batch(vocab_size: int, batch_size: int = 2, seq_len: int = 16, pad_id: int = 0):
    rng = np.random.RandomState(0)
    token_ids_BT = rng.randint(1, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    token_ids_BT[:, 0] = pad_id  # exercise the padding / segment-id path
    segment_ids_BT = (token_ids_BT != pad_id).astype(np.int32)
    return jnp.asarray(token_ids_BT), jnp.asarray(segment_ids_BT)


if __name__ == "__main__":
    absltest.main()
