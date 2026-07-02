"""Context Parallelism (CP) Stage 1 -- all-gather-KV full-attention -- tests.

Two guarantees are encoded here (see the CP design in omegalax/attention.py):

1. **cp_size == 1 is a strict no-op.** The CP ShardConfig collapses to the
   non-CP ``default`` layout (the size-1 ``cp`` axis is filtered out), the CP
   loss ``shift=False``/``cp_axis`` path is not taken, and the attention CP gate
   (``mesh.shape["cp"] > 1``) stays off. This is covered structurally here and by
   the unchanged existing suites (test_topology_mesh, test_scan_layers,
   test_remat_policy).

2. **cp_size > 1 equals cp_size == 1** for a Qwen3 DENSE (all full-attention,
   NO linear-attention layers -- DeltaNet CP is Stage 2) model, forward AND
   backward, within fp tolerance. Run on faked multi-CPU (cp=2 and cp=4). This is
   the acceptance test.

The equivalence is checked by building the model with a fixed init seed on both
the cp=1 and cp>1 meshes: ``init_model_sharded`` is deterministic in the seed and
CP shards ACTIVATIONS (not parameters), so both meshes get numerically identical
weights. To keep the mesh product == device_count while isolating CP, the spare
devices go to pure data-parallel (dp = devices // cp), which never changes
per-example numerics.
"""

import os

# Faked 4 host devices, before jax is imported, so cp=2 and cp=4 both fit.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest, parameterized  # noqa: E402
from flax import nnx  # noqa: E402

from omegalax.distributed.mesh import make_mesh, mesh_rules  # noqa: E402
from omegalax.models.qwen3.config import make_config as make_qwen3_config  # noqa: E402
from omegalax.models.qwen3.model import Qwen3  # noqa: E402
from omegalax.models.shard_config import (  # noqa: E402
    ShardConfig,
    axis_rules_for_mesh,
    shard_config_for_mesh,
)
from omegalax.models.sharding_runtime import (  # noqa: E402
    init_model_sharded,
    set_attn_backend,
    shard_batch_dict,
)
import optax  # noqa: E402

from omegalax.trainers import text as text_trainer  # noqa: E402
from omegalax.trainers.loss import chunked_cross_entropy_loss, shift_for_next_token  # noqa: E402
from omegalax.trainers.optim import MixedPrecisionOptimizer  # noqa: E402


def _dense_cfg():
    """Qwen3 dense smoke config: fp32, all full-attention layers (no DeltaNet)."""
    cfg = make_qwen3_config("qwen3-smoke")
    # fp32 for a tight equivalence tolerance; CP is an fp-reduction-order change.
    return dataclasses.replace(cfg, dtype=jnp.float32)


def _make_model(cfg, mesh):
    """Build a Qwen3 on ``mesh`` with the fixed init seed.

    ``init_model_sharded`` is deterministic in the seed, and CP shards
    activations (not parameters), so the SAME seed yields numerically identical
    weights on the cp=1 and cp>1 meshes -- no cross-mesh state reload needed
    (which would be a test-harness sharding artifact, not the code under test).
    """
    cfg_m = dataclasses.replace(cfg, shd_cfg=shard_config_for_mesh(cfg.shd_cfg, mesh))
    axis_rules = axis_rules_for_mesh(mesh)
    model = init_model_sharded(Qwen3, cfg_m, jax.random.key(0), mesh, axis_rules)
    # CPU: tokamax mosaic_gpu is GPU-only, so both the non-CP tokamax path and the
    # CP all-gather-KV local attention use the xla backend here.
    set_attn_backend(model, "xla")
    return model, cfg_m


def _forward_and_grad(model, cfg, batch):
    """Jitted forward hidden + jitted param-grad of the CP-aware SFT loss.

    Mirrors the trainer: the shift-before-shard ``targets_BT`` is produced by
    ``shard_batch_dict`` under CP; here we read it back (and pass shift=False /
    cp_axis). Non-CP reads token_ids as targets and shifts internally.
    """
    cp_axis = cfg.shd_cfg.act_btd[1]
    token_ids_BT = batch["token_ids_BT"]
    loss_mask_BT = batch["loss_mask_BT"]
    targets_BT = batch["targets_BT"] if cp_axis is not None else token_ids_BT

    segment_ids_BT = (token_ids_BT != 0).astype(jnp.int32)
    gd, state = nnx.split(model)

    def _fwd(state):
        m = nnx.merge(gd, state)
        hidden_BTD, _ = m(token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32))
        return hidden_BTD

    def _loss(state):
        m = nnx.merge(gd, state)
        hidden_BTD, _ = m(token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32))
        return chunked_cross_entropy_loss(
            hidden_BTD,
            m.lm_head.kernel[...],
            targets_BT,
            loss_mask_BT,
            num_tiles=1,
            logits_out_sharding=cfg.shd_cfg.logits_btv,
            shift=cp_axis is None,
            cp_axis=cp_axis,
        )

    hidden = jax.jit(_fwd)(state)
    loss = jax.jit(_loss)(state)
    grads = jax.jit(jax.grad(_loss))(state)
    return np.asarray(jax.device_get(hidden)), float(loss), nnx.to_pure_dict(grads)


def _flat_grads(pure):
    """Flatten a pure grad dict to a sorted list of (path, host-array)."""
    out = {}

    def _rec(node, prefix):
        if isinstance(node, dict):
            if "value" in node and not isinstance(node["value"], dict):
                out[prefix] = np.asarray(jax.device_get(node["value"]))
            else:
                for k, v in node.items():
                    _rec(v, f"{prefix}/{k}")
        else:
            out[prefix] = np.asarray(jax.device_get(node))

    _rec(pure, "")
    return out


class CpNoOpTest(absltest.TestCase):
    """cp_size == 1 collapses the CP config/loss to the non-CP behavior."""

    def test_shard_config_cp1_equals_default(self):
        # Faked device_count is 4; put cp=1 with dp=4 so the product matches.
        mesh = make_mesh(tp_size=1, fsdp_size=1, dp_size=4, cp_size=1)
        cp = shard_config_for_mesh(ShardConfig.context_parallel(), mesh)
        default = shard_config_for_mesh(ShardConfig.default(), mesh)
        self.assertEqual(cp.act_btd, default.act_btd)
        self.assertEqual(cp.act_btf, default.act_btf)
        self.assertEqual(cp.act_btnh, default.act_btnh)
        # No "cp" left in any spec -> the T axis is replicated exactly as before.
        self.assertIsNone(cp.act_btd[1])

    def test_default_logits_btv_unchanged(self):
        # The logits_btv seq entry is None for the non-CP configs (byte-compatible
        # with the pre-CP P(batch, None, vocab)).
        self.assertIsNone(ShardConfig.default().logits_btv[1])
        self.assertIsNone(ShardConfig.no_sharding().logits_btv[1])

    def test_shift_before_shard_matches_internal_shift(self):
        # The shift-before-shard aligned loss equals the internal +1 shift loss.
        B, T, D, V = 2, 16, 8, 32
        hidden = jax.random.normal(jax.random.key(0), (B, T, D))
        kernel = jax.random.normal(jax.random.key(1), (D, V))
        tokens = jax.random.randint(jax.random.key(2), (B, T), 1, V)
        mask = (jax.random.uniform(jax.random.key(3), (B, T)) > 0.3).astype(jnp.float32)
        l_shift = chunked_cross_entropy_loss(hidden, kernel, tokens, mask, num_tiles=1)
        _, targets, mask2 = shift_for_next_token(tokens, mask)
        l_align = chunked_cross_entropy_loss(
            hidden, kernel, targets, mask2, num_tiles=1, shift=False
        )
        np.testing.assert_allclose(float(l_shift), float(l_align), rtol=0, atol=1e-5)


class CpEquivalenceTest(parameterized.TestCase):
    """cp>1 forward+backward == cp=1 for a Qwen3 dense (full-attention) model."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cfg = _dense_cfg()
        # Faked device_count is 4. To keep the mesh product == 4 across cp in
        # {1,2,4} while isolating CP, we carve the remaining devices into pure
        # data-parallel (dp = 4 // cp): dp replicates params and splits INDEPENDENT
        # batch elements, so per-example numerics are cp-agnostic. B=4 divides
        # every dp; T=16 divides every cp.
        cls.B, cls.T = 4, 16
        rng = np.random.RandomState(0)
        cls.tokens = rng.randint(1, cls.cfg.vocab_size, size=(cls.B, cls.T)).astype(np.int32)
        cls.mask = np.ones((cls.B, cls.T), dtype=np.float32)

        # Baseline (cp=1, dp=4): fixed-seed init, run forward + grad.
        cls.mesh1 = make_mesh(tp_size=1, fsdp_size=1, dp_size=4, cp_size=1)
        with mesh_rules(cls.mesh1):
            model1, cfg1 = _make_model(cls.cfg, cls.mesh1)
            batch1 = shard_batch_dict(
                {"token_ids_BT": cls.tokens, "loss_mask_BT": cls.mask}, cfg1.shd_cfg, cls.mesh1
            )
            cls.hidden1, cls.loss1, cls.grads1 = _forward_and_grad(model1, cfg1, batch1)

    @parameterized.named_parameters(("cp2", 2), ("cp4", 4))
    def test_cp_matches_non_cp(self, cp):
        mesh = make_mesh(tp_size=1, fsdp_size=1, dp_size=4 // cp, cp_size=cp)
        with mesh_rules(mesh):
            cfg_cp = dataclasses.replace(self.cfg, shd_cfg=ShardConfig.context_parallel())
            model, cfg_m = _make_model(cfg_cp, mesh)
            # Confirm CP is actually active: the T axis carries "cp".
            self.assertEqual(cfg_m.shd_cfg.act_btd[1], "cp")
            batch = shard_batch_dict(
                {"token_ids_BT": self.tokens, "loss_mask_BT": self.mask},
                cfg_m.shd_cfg,
                mesh,
            )
            hidden, loss, grads = _forward_and_grad(model, cfg_m, batch)

        # ---- Forward hidden equivalence ----
        self.assertEqual(hidden.shape, self.hidden1.shape)
        self.assertTrue(np.isfinite(hidden).all())
        fwd_diff = float(np.abs(hidden - self.hidden1).max())
        fwd_scale = max(float(np.abs(self.hidden1).max()), 1e-6)
        self.assertLess(fwd_diff / fwd_scale, 2e-5, f"forward rel diff {fwd_diff / fwd_scale}")

        # ---- Loss equivalence ----
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(loss, self.loss1, rtol=2e-5, atol=1e-5)

        # ---- Backward (param-grad) equivalence ----
        g_cp = _flat_grads(grads)
        g_ref = _flat_grads(self.grads1)
        self.assertEqual(set(g_cp), set(g_ref))
        worst = 0.0
        for k in g_ref:
            a, b = g_cp[k], g_ref[k]
            self.assertTrue(np.isfinite(a).all(), f"non-finite grad at {k}")
            scale = max(float(np.abs(b).max()), 1e-6)
            worst = max(worst, float(np.abs(a - b).max()) / scale)
        self.assertLess(worst, 2e-4, f"worst param-grad rel diff {worst}")


class CpTrainStepTest(absltest.TestCase):
    """A full CP-aware SFT train step (fwd+bwd+optimizer update) runs finite."""

    def test_cp_train_step_finite(self):
        cfg = _dense_cfg()
        mesh = make_mesh(tp_size=1, fsdp_size=1, dp_size=2, cp_size=2)
        B, T = 4, 16
        rng = np.random.RandomState(0)
        tokens = rng.randint(1, cfg.vocab_size, size=(B, T)).astype(np.int32)
        mask = np.ones((B, T), dtype=np.float32)
        with mesh_rules(mesh):
            cfg_m = dataclasses.replace(cfg, shd_cfg=shard_config_for_mesh(
                ShardConfig.context_parallel(), mesh
            ))
            # CP is actually active.
            self.assertEqual(cfg_m.shd_cfg.act_btd[1], "cp")
            model, _ = _make_model(dataclasses.replace(cfg, shd_cfg=ShardConfig.context_parallel()),
                                   mesh)
            opt = MixedPrecisionOptimizer(
                model, optax.adamw(1e-3), wrt=nnx.Param
            )
            step = text_trainer.make_sft_train_step(cfg_m, pad_id=0)
            batch = shard_batch_dict(
                {"token_ids_BT": tokens, "loss_mask_BT": mask}, cfg_m.shd_cfg, mesh
            )
            loss, metrics = step(opt, batch)
        self.assertTrue(np.isfinite(float(loss)), f"CP train-step loss not finite: {loss}")
        self.assertTrue(np.isfinite(float(metrics["grad_norm"])))
        self.assertGreater(float(metrics["supervised_tokens"]), 0.0)


if __name__ == "__main__":
    absltest.main()
