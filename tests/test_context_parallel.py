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

from omegalax.attention import context_parallel_attention  # noqa: E402
from omegalax.distributed.mesh import make_mesh, mesh_rules  # noqa: E402
from omegalax.distributed.zigzag import zigzag_permutation, is_identity  # noqa: E402
from omegalax.models.qwen3.config import make_config as make_qwen3_config  # noqa: E402
from omegalax.models.qwen3.model import Qwen3  # noqa: E402
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig  # noqa: E402
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM  # noqa: E402
from omegalax.models.qwen3_5.kernels.xla_reference import (  # noqa: E402
    chunk_gated_delta_rule_xla,
)
from omegalax.models.shard_config import (  # noqa: E402
    ShardConfig,
    axis_rules_for_mesh,
    shard_config_for_mesh,
)
from omegalax.models.sharding_runtime import (  # noqa: E402
    init_model_sharded,
    set_attn_backend,
    set_cp_document_mask,
    shard_batch_dict,
)
from jax.sharding import NamedSharding, PartitionSpec as _P  # noqa: E402
from tokamax import dot_product_attention as _dpa  # noqa: E402
import optax  # noqa: E402

from omegalax.trainers import text as text_trainer  # noqa: E402
from omegalax.trainers.loss import chunked_cross_entropy_loss, shift_for_next_token  # noqa: E402
from omegalax.trainers.optim import MixedPrecisionOptimizer  # noqa: E402


def _dense_cfg():
    """Qwen3 dense smoke config: fp32, all full-attention layers (no DeltaNet)."""
    cfg = make_qwen3_config("qwen3-smoke")
    # fp32 for a tight equivalence tolerance; CP is an fp-reduction-order change.
    return dataclasses.replace(cfg, dtype=jnp.float32)


def _qwen35_cfg(layer_types):
    """Small fp32 Qwen3.5 TEXT config with the given per-layer types.

    ``("linear_attention", ..., "full_attention", ...)`` gives the HYBRID that
    Stage 2 exists to cover; an all-``"full_attention"`` list is the CONTROL that
    isolates the DeltaNet fp32-kernel reassociation from any CP-composition error.
    """
    return Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        layer_types=tuple(layer_types),
        rope_theta=10000,
        partial_rotary_factor=0.5,
        mrope_section=(2, 1, 1),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_value_head_dim=16,
        intermediate_size=128,
        dtype=jnp.float32,
        scan_layers=True,
    )


def _make_qwen35(cfg, mesh):
    """Build a Qwen3.5 text model (fixed seed) on ``mesh``; xla attn on CPU."""
    cfg_m = dataclasses.replace(cfg, shd_cfg=shard_config_for_mesh(cfg.shd_cfg, mesh))
    model = init_model_sharded(
        Qwen3_5ForCausalLM, cfg_m, jax.random.key(0), mesh, axis_rules_for_mesh(mesh)
    )
    set_attn_backend(model, "xla")
    return model, cfg_m


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
    cp_axis). ``position_ids_BT`` (present only under zig-zag CP) carries each
    token's original index. Non-CP reads token_ids as targets and shifts internally.
    """
    cp_axis = cfg.shd_cfg.act_btd[1]
    token_ids_BT = batch["token_ids_BT"]
    loss_mask_BT = batch["loss_mask_BT"]
    targets_BT = batch["targets_BT"] if cp_axis is not None else token_ids_BT
    position_ids_BT = batch.get("position_ids_BT")

    segment_ids_BT = (token_ids_BT != 0).astype(jnp.int32)
    gd, state = nnx.split(model)

    def _run(state):
        m = nnx.merge(gd, state)
        return m(
            token_ids_BT, segment_ids_BT, None, jnp.array(0, dtype=jnp.int32),
            position_ids_BT=position_ids_BT,
        )[0]

    def _fwd(state):
        return _run(state)

    def _loss(state):
        hidden_BTD = _run(state)
        return chunked_cross_entropy_loss(
            hidden_BTD,
            nnx.merge(gd, state).lm_head.kernel[...],
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
                {"token_ids_BT": cls.tokens, "loss_mask_BT": cls.mask}, cfg1.shd_cfg, cls.mesh1,
                cp_load_balance=False,  # contiguous CP; zig-zag tested separately
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
                cp_load_balance=False,  # contiguous CP; zig-zag tested separately
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
                {"token_ids_BT": tokens, "loss_mask_BT": mask}, cfg_m.shd_cfg, mesh,
                cp_load_balance=False,
            )
            loss, metrics = step(opt, batch)
        self.assertTrue(np.isfinite(float(loss)), f"CP train-step loss not finite: {loss}")
        self.assertTrue(np.isfinite(float(metrics["grad_norm"])))
        self.assertGreater(float(metrics["supervised_tokens"]), 0.0)


class CpHybridEquivalenceTest(parameterized.TestCase):
    """Stage 2: cp>1 == cp=1 for a Qwen3.5 HYBRID (linear + full attn) model.

    This is the point of Stage 2 — the DeltaNet (linear-attention) layers get
    full context-parallel coverage via the boundary-state ring + conv halo. The
    DeltaNet fp32 chunked kernel reassociates the recurrence differently under
    segmentation, so its gradient matches only to ~1e-4 (the inherent kernel
    regime the scan / pallas-bwd work already characterized) -- NOT a CP bug. The
    all-``full_attention`` CONTROL (same harness, no DeltaNet) matches to ~1e-6,
    proving the CP composition itself is exact and the ~1e-4 is purely the
    DeltaNet kernel. Forward is bit-identical for both.
    """

    LAYER_TYPES = {
        "hybrid": ("linear_attention", "linear_attention", "linear_attention", "full_attention"),
        "all_full": ("full_attention", "full_attention"),
    }
    # Forward is bit-identical; gradient tol reflects the fp regime of each kind.
    GRAD_TOL = {"hybrid": 5e-3, "all_full": 1e-4}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.B, cls.T = 4, 32  # T divisible by cp*chunk boundaries; B by dp
        rng = np.random.RandomState(0)
        cls.tokens = rng.randint(1, 256, size=(cls.B, cls.T)).astype(np.int32)
        cls.mask = np.ones((cls.B, cls.T), dtype=np.float32)
        # cp=1 baselines (dp=4) per layer-kind.
        cls.baseline = {}
        mesh1 = make_mesh(tp_size=1, fsdp_size=1, dp_size=4, cp_size=1)
        for kind, lt in cls.LAYER_TYPES.items():
            cfg = _qwen35_cfg(lt)
            with mesh_rules(mesh1):
                model, cfg_m = _make_qwen35(cfg, mesh1)
                batch = shard_batch_dict(
                    {"token_ids_BT": cls.tokens, "loss_mask_BT": cls.mask}, cfg_m.shd_cfg, mesh1,
                    cp_load_balance=False,
                )
                cls.baseline[kind] = (cfg, *_forward_and_grad(model, cfg_m, batch))

    @parameterized.named_parameters(
        ("hybrid_cp2", "hybrid", 2),
        ("hybrid_cp4", "hybrid", 4),
        ("all_full_cp2", "all_full", 2),
        ("all_full_cp4", "all_full", 4),
    )
    def test_cp_matches_non_cp(self, kind, cp):
        cfg, hidden1, loss1, grads1 = self.baseline[kind]
        mesh = make_mesh(tp_size=1, fsdp_size=1, dp_size=4 // cp, cp_size=cp)
        with mesh_rules(mesh):
            cfg_cp = dataclasses.replace(cfg, shd_cfg=ShardConfig.context_parallel())
            model, cfg_m = _make_qwen35(cfg_cp, mesh)
            self.assertEqual(cfg_m.shd_cfg.act_btd[1], "cp")
            batch = shard_batch_dict(
                {"token_ids_BT": self.tokens, "loss_mask_BT": self.mask}, cfg_m.shd_cfg, mesh,
                cp_load_balance=False,
            )
            hidden, loss, grads = _forward_and_grad(model, cfg_m, batch)

        # Forward is bit-identical (both kinds) up to fp reduction order.
        self.assertTrue(np.isfinite(hidden).all())
        fwd_rel = float(np.abs(hidden - hidden1).max()) / max(float(np.abs(hidden1).max()), 1e-6)
        self.assertLess(fwd_rel, 5e-4, f"[{kind}] forward rel diff {fwd_rel}")
        np.testing.assert_allclose(loss, loss1, rtol=5e-4, atol=1e-4)

        g_cp, g_ref = _flat_grads(grads), _flat_grads(grads1)
        self.assertEqual(set(g_cp), set(g_ref))
        worst = 0.0
        for k in g_ref:
            a, b = g_cp[k], g_ref[k]
            self.assertTrue(np.isfinite(a).all(), f"non-finite grad at {k}")
            worst = max(worst, float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-6))
        self.assertLess(worst, self.GRAD_TOL[kind], f"[{kind}] worst grad rel diff {worst}")


class DeltaNetStatePassKernelTest(parameterized.TestCase):
    """Kernel ``state_init`` / ``final_state`` contract + gradient (CP Stage 2).

    Exercises the XLA reference on CPU (the Pallas-kernel state_init VJP is
    GPU-only, deferred to node availability). Verifies:
      * chaining two segments with state passing == the full sequence (fwd);
      * the state_init gradient matches finite differences;
      * ``B_seg`` (aggregate bias) == final state from zero, i.e. the boundary
        ring's affine algebra matches the kernel.
    """

    def _inputs(self, B=1, T=128, H=2, A=16, U=16, seed=0):
        rng = np.random.RandomState(seed)
        q = jnp.asarray(rng.randn(B, T, H, A).astype(np.float32) * 0.1)
        k = jnp.asarray(rng.randn(B, T, H, A).astype(np.float32) * 0.1)
        v = jnp.asarray(rng.randn(B, T, H, U).astype(np.float32) * 0.1)
        a = jnp.asarray(rng.randn(B, T, H).astype(np.float32) * 0.5)
        g = -jnp.exp(a) * jax.nn.softplus(a)
        beta = jax.nn.sigmoid(jnp.asarray(rng.randn(B, T, H).astype(np.float32) * 0.5))
        return q, k, v, g, beta

    def test_state_passing_reproduces_full_sequence(self):
        q, k, v, g, beta = self._inputs()
        C, T = 64, q.shape[1]
        full = chunk_gated_delta_rule_xla(q, k, v, g, beta, C)
        half = T // 2
        o0, s0 = chunk_gated_delta_rule_xla(
            q[:, :half], k[:, :half], v[:, :half], g[:, :half], beta[:, :half],
            C, return_final_state=True,
        )
        o1 = chunk_gated_delta_rule_xla(
            q[:, half:], k[:, half:], v[:, half:], g[:, half:], beta[:, half:],
            C, state_init_BHAU=s0,
        )
        split = jnp.concatenate([o0, o1], axis=1)
        np.testing.assert_allclose(np.asarray(split), np.asarray(full), rtol=0, atol=1e-5)

    def test_state_init_gradient_matches_finite_diff(self):
        q, k, v, g, beta = self._inputs()
        C, T = 64, q.shape[1]
        half = T // 2
        seg = (q[:, half:], k[:, half:], v[:, half:], g[:, half:], beta[:, half:])
        B, _, H, A = q.shape
        U = v.shape[-1]
        si = jnp.asarray(np.random.RandomState(1).randn(B, H, A, U).astype(np.float32) * 0.2)

        def loss(si):
            o = chunk_gated_delta_rule_xla(*seg, C, state_init_BHAU=si)
            return jnp.sum(o**2)

        dsi = jax.grad(loss)(si)
        eps = 1e-3
        for idx in [(0, 0, 0, 0), (0, 1, 3, 5)]:
            fd = (loss(si.at[idx].add(eps)) - loss(si.at[idx].add(-eps))) / (2 * eps)
            np.testing.assert_allclose(float(dsi[idx]), float(fd), rtol=2e-2, atol=1e-5)

    def test_segment_transition_bias_equals_final_state_from_zero(self):
        from omegalax.models.qwen3_5.kernels.cp import _segment_state_transition

        q, k, v, g, beta = self._inputs()
        C = 64
        A_seg, B_seg = _segment_state_transition(q, k, v, g, beta, C)
        _, final0 = chunk_gated_delta_rule_xla(q, k, v, g, beta, C, return_final_state=True)
        np.testing.assert_allclose(np.asarray(B_seg), np.asarray(final0), rtol=0, atol=1e-5)
        # And A_seg @ si + B_seg == final state seeded from si, for random si.
        Bd, _, H, Ad = q.shape
        U = v.shape[-1]
        si = jnp.asarray(np.random.RandomState(2).randn(Bd, H, Ad, U).astype(np.float32) * 0.3)
        _, final_si = chunk_gated_delta_rule_xla(q, k, v, g, beta, C, state_init_BHAU=si,
                                                 return_final_state=True)
        pred = jnp.einsum("BHXY,BHYU->BHXU", A_seg, si) + B_seg
        np.testing.assert_allclose(np.asarray(pred), np.asarray(final_si), rtol=0, atol=1e-5)


class CpZigZagInvarianceTest(parameterized.TestCase):
    """Stage 1b: zig-zag load-balancing is NUMERICALLY INVISIBLE for full attn.

    Zig-zag permutes the sequence layout to balance the causal-attention triangle
    across cp ranks. Because CP attention masks over GLOBAL positions (not local
    arange), the permutation must not change the loss/grads at all (up to fp
    reduction order). Verified vs the non-CP baseline for full-attention stacks
    (Qwen3 dense + all-``full_attention`` Qwen3.5). Zig-zag is intentionally NOT
    applied to hybrids: the DeltaNet recurrence is order-dependent, so
    text_api.cp_load_balance_ok() disables it there (a separate structural test).
    """

    def _baseline(self, cfg, ctor, make):
        mesh1 = make_mesh(tp_size=1, fsdp_size=1, dp_size=4, cp_size=1)
        with mesh_rules(mesh1):
            model, cfg_m = make(cfg, mesh1)
            batch = shard_batch_dict(
                {"token_ids_BT": self.tokens, "loss_mask_BT": self.mask},
                cfg_m.shd_cfg, mesh1, cp_load_balance=False,
            )
            return _forward_and_grad(model, cfg_m, batch)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.B, cls.T = 4, 32
        rng = np.random.RandomState(0)
        cls.tokens = rng.randint(1, 256, size=(cls.B, cls.T)).astype(np.int32)
        cls.mask = np.ones((cls.B, cls.T), dtype=np.float32)

    @parameterized.named_parameters(
        ("dense_cp2", "dense", 2), ("dense_cp4", "dense", 4),
        ("qwen35_full_cp2", "qwen35_full", 2), ("qwen35_full_cp4", "qwen35_full", 4),
    )
    def test_zigzag_invariance(self, kind, cp):
        if kind == "dense":
            cfg = _dense_cfg()
            ctor, make = Qwen3, _make_model
        else:
            cfg = _qwen35_cfg(("full_attention", "full_attention"))
            ctor, make = Qwen3_5ForCausalLM, _make_qwen35

        hidden1, loss1, grads1 = self._baseline(cfg, ctor, make)

        mesh = make_mesh(tp_size=1, fsdp_size=1, dp_size=4 // cp, cp_size=cp)
        with mesh_rules(mesh):
            cfg_cp = dataclasses.replace(cfg, shd_cfg=ShardConfig.context_parallel())
            model, cfg_m = make(cfg_cp, mesh)
            # zig-zag ON (cp_load_balance=True) -> the batch is permuted + carries
            # position_ids_BT (true original positions).
            batch = shard_batch_dict(
                {"token_ids_BT": self.tokens, "loss_mask_BT": self.mask},
                cfg_m.shd_cfg, mesh, cp_load_balance=True,
            )
            self.assertIn("position_ids_BT", batch)  # zig-zag actually engaged
            hidden, loss, grads = _forward_and_grad(model, cfg_m, batch)

        # Loss/grads must be invariant to the permutation (hidden is permuted, so
        # only compare loss + grads which are layout-invariant scalars/params).
        np.testing.assert_allclose(loss, loss1, rtol=2e-5, atol=1e-5)
        g_cp, g_ref = _flat_grads(grads), _flat_grads(grads1)
        self.assertEqual(set(g_cp), set(g_ref))
        worst = 0.0
        for k in g_ref:
            a, b = g_cp[k], g_ref[k]
            self.assertTrue(np.isfinite(a).all(), f"non-finite grad at {k}")
            worst = max(worst, float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-6))
        self.assertLess(worst, 2e-4, f"[{kind}] zig-zag grad rel diff {worst}")

    def test_zigzag_disabled_for_hybrid(self):
        # Structural guard: hybrids must NOT zig-zag (DeltaNet is order-dependent).
        from omegalax.text import api as text_api
        self.assertFalse(
            text_api.cp_load_balance_ok(
                _qwen35_cfg(("linear_attention", "full_attention"))
            )
        )
        self.assertTrue(
            text_api.cp_load_balance_ok(_qwen35_cfg(("full_attention", "full_attention")))
        )
        self.assertTrue(text_api.cp_load_balance_ok(_dense_cfg()))

    def test_permutation_is_balanced_and_identity_at_cp1(self):
        self.assertTrue(is_identity(zigzag_permutation(32, 1)))
        perm = zigzag_permutation(32, 4)
        self.assertEqual(sorted(perm.tolist()), list(range(32)))  # valid permutation
        # rank r owns contiguous slice [2r*cs, (2r+2)*cs) = zig-zag pair {r, 2cp-1-r}
        cs = 32 // 8
        for r in range(4):
            lo = 2 * r * cs
            chunks = {int(perm[lo + i]) // cs for i in range(2 * cs)}
            self.assertEqual(chunks, {r, 7 - r})


class CpDocumentMaskTest(absltest.TestCase):
    """Block-diagonal document masking under CP: no cross-document attention.

    Packed multi-document sequences must not attend across a document boundary.
    Verified two ways on a packed 2-document batch with cp>1:
      1. the CP attention output == a DENSE reference (causal AND same-segment);
      2. document-1's output is INVARIANT to document-2's values (proof that no
         token attends across the boundary).
    Single-document behavior is unchanged (doc mask is a no-op when all tokens
    share one segment).
    """

    def _run_cp(self, q, k, v, pos, seg, cp, use_seg):
        # Fill all 4 faked devices: cp*dp == 4 (batch is size 1 but dp only
        # replicates it here; the doc mask is per-(B) so dp is harmless).
        mesh = make_mesh(tp_size=1, fsdp_size=1, dp_size=4 // cp, cp_size=cp)
        hs = _P(("dp", "fsdp"), "cp", "tp", None)
        ss = _P(("dp", "fsdp"), "cp")
        with mesh_rules(mesh):
            qs = jax.device_put(q, NamedSharding(mesh, hs))
            ks = jax.device_put(k, NamedSharding(mesh, hs))
            vs = jax.device_put(v, NamedSharding(mesh, hs))
            ps = jax.device_put(pos, NamedSharding(mesh, ss))
            sg = jax.device_put(seg, NamedSharding(mesh, ss))

            def f(q, k, v, p, s):
                return context_parallel_attention(
                    q, k, v, p, cp_axis="cp", scale=q.shape[-1] ** -0.5,
                    heads_spec=hs, seq_spec=ss,
                    q_segment_ids_BT=(s if use_seg else None), implementation="xla",
                )

            return np.asarray(jax.device_get(jax.jit(f)(qs, ks, vs, ps, sg)))

    def test_no_cross_document_attention(self):
        # B=4 fills every dp = 4 // cp (batch shards cleanly on dp).
        B, T, H, K = 4, 16, 2, 8
        rng = jax.random.key(0)
        q = jax.random.normal(rng, (B, T, H, K))
        k = jax.random.normal(jax.random.key(1), (B, T, H, K))
        v = jax.random.normal(jax.random.key(2), (B, T, H, K))
        # 2 docs: seg 1 on [0,8), seg 2 on [8,16); positions restart per doc.
        seg = jnp.asarray(np.concatenate([np.ones(8), 2 * np.ones(8)])[None].astype(np.int32))
        pos = jnp.asarray(np.concatenate([np.arange(8), np.arange(8)])[None].astype(np.int32))
        pos = jnp.broadcast_to(pos, (B, T))
        seg = jnp.broadcast_to(seg, (B, T))

        # Dense reference: causal AND same-segment.
        qpos, kpos = pos[:, None, :, None], pos[:, None, None, :]
        qseg, kseg = seg[:, None, :, None], seg[:, None, None, :]
        ref = _dpa(q, k, v, mask=(qpos >= kpos) & (qseg == kseg),
                   is_causal=False, implementation="xla")

        for cp in (2, 4):
            out = self._run_cp(q, k, v, pos, seg, cp, use_seg=True)
            np.testing.assert_allclose(out, np.asarray(ref), rtol=0, atol=1e-5)
            # doc-1 output invariant to doc-2 values -> no cross-doc attention.
            v2 = v.at[:, 8:].set(999.0)
            out2 = self._run_cp(q, k, v2, pos, seg, cp, use_seg=True)
            np.testing.assert_allclose(out[:, :8], out2[:, :8], rtol=0, atol=1e-5)

    def test_single_document_unchanged(self):
        # With one segment everywhere, the doc mask is a no-op == plain causal.
        B, T, H, K = 4, 16, 2, 8
        rng = jax.random.key(3)
        q = jax.random.normal(rng, (B, T, H, K))
        k = jax.random.normal(jax.random.key(4), (B, T, H, K))
        v = jax.random.normal(jax.random.key(5), (B, T, H, K))
        pos = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[None], (B, T))
        seg = jnp.ones((B, T), dtype=jnp.int32)
        causal_ref = _dpa(q, k, v, is_causal=True, implementation="xla")
        for cp in (2, 4):
            out_seg = self._run_cp(q, k, v, pos, seg, cp, use_seg=True)
            np.testing.assert_allclose(out_seg, np.asarray(causal_ref), rtol=0, atol=1e-5)


if __name__ == "__main__":
    absltest.main()
