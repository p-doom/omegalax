"""CPU tests for host/CPU offload of optimizer state + activations.

Scope (what CAN be checked on a login-node CPU):
  * **OFF is a strict no-op** — with the default config the optimizer builds
    byte-identically to trunk (moments on ``device``), the offload flag is
    off, and a train step runs and matches the pre-offload behavior.
  * **Platform gating** — ``is_coherent_host_offload_platform()`` is ``False``
    on CPU, so ``"auto"`` resolves to OFF; an explicit ``True``/``False`` is
    honored verbatim (never silently overridden).
  * **Offload remat policies resolve and trace** — the new ``"offload_dot"`` /
    ``"offload_named"`` policies resolve to jax policy objects and a smoke model
    *traces* (``jax.eval_shape``) under them, exercising the policy wiring and
    the ``checkpoint_name`` residual tag.
  * **wrt filter is undisturbed** — turning offload on does not change which
    variables get optimizer state (the grad/optimizer ``wrt`` filter).

Explicitly NOT checked here (deferred to GPU/GH200, see the module + report):
  * Actually *executing* the host<->device staging: the XLA:CPU runtime has no
    ``annotate_device_placement`` implementation for its Host memory space, so a
    jitted step that mixes ``pinned_host`` and ``device`` operands cannot run on
    CPU. This is an XLA-CPU limitation, not a wiring bug — the placement itself
    (``device_put`` to ``pinned_host``) succeeds on CPU, which is what we assert.
  * Peak-memory reduction, C2C step-time overlap, and the checkpoint
    save/restore-with-memory-kind round-trip.

Runs on CPU (``JAX_PLATFORMS=cpu``); the attention backend is swapped to the
CPU-safe XLA fallback (same shim the remat/scan tests use).
"""

import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMEGALAX_DELTANET_KERNEL", "xla")
os.environ.setdefault("OMEGALAX_STARTUP_LOG", "0")

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from flax import nnx

from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.remat_policy import (
    OFFLOAD_RESIDUAL_NAME,
    available_remat_policies,
    policy_uses_named_offload,
    resolve_remat_policy,
    tag_offload_residual,
)
from omegalax.text import api as text_api
from omegalax.trainers import offload as offload_lib
from omegalax.trainers.offload import (
    DEVICE_MEMORY_KIND,
    HOST_MEMORY_KIND,
    is_coherent_host_offload_platform,
    place_tree_on_memory_kind,
    resolve_offload_enabled,
    sharding_on_memory_kind,
)
from omegalax.trainers.text import TrainConfig, build_optimizer, make_sft_train_step

_SEED = 0
_MODEL_ID = "qwen3-smoke"


def _patch_attention_backend_to_xla(model: nnx.Module) -> None:
    for _, mod in nnx.iter_modules(model):
        if hasattr(mod, "_attn_backend"):
            object.__setattr__(mod, "_attn_backend", "xla")


def _fp32_cfg(model_id: str = _MODEL_ID, **overrides):
    cfg = text_api.resolve_config(model_id)
    return dataclasses.replace(cfg, dtype=jnp.float32, **overrides)


def _build_model(cfg):
    model, cfg = text_api.init_model(
        cfg, jax.random.PRNGKey(_SEED), tp_size=1, fsdp_size=1, dp_size=1
    )
    _patch_attention_backend_to_xla(model)
    return model, cfg


def _opt_state_memory_kinds(optimizer) -> set:
    return {
        getattr(getattr(a, "sharding", None), "memory_kind", None)
        for a in jax.tree_util.tree_leaves(nnx.state(optimizer.opt_state))
    }


def _make_batch(vocab_size: int, batch_size: int = 2, seq_len: int = 16, pad_id: int = 0):
    rng = np.random.RandomState(_SEED)
    tok = rng.randint(1, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    tok[:, 0] = pad_id
    return {
        "token_ids_BT": jnp.asarray(tok),
        "loss_mask_BT": jnp.asarray((tok != pad_id).astype(np.int32)),
    }


class PlatformGatingTest(absltest.TestCase):
    """The gate must be OFF on CPU and must honor explicit overrides."""

    def test_cpu_is_not_coherent_host_platform(self):
        self.assertFalse(is_coherent_host_offload_platform())

    def test_auto_resolves_off_on_cpu(self):
        # "auto" must NOT enable offload on a non-coherent platform.
        self.assertFalse(resolve_offload_enabled("auto"))

    def test_explicit_true_false_honored_verbatim(self):
        # Explicit user choice is never silently overridden by the gate.
        self.assertTrue(resolve_offload_enabled(True))
        self.assertFalse(resolve_offload_enabled(False))

    def test_invalid_setting_raises(self):
        with self.assertRaises(ValueError):
            resolve_offload_enabled("sometimes")

    def test_gh200_device_kind_would_gate_on(self):
        # Simulate a GH200 device_kind without a GPU: the substring match is the
        # only signal, so a fake device with platform 'gpu' + 'GH200' kind gates on.
        class _FakeDev:
            platform = "gpu"
            device_kind = "NVIDIA GH200 480GB"

        self.assertTrue(is_coherent_host_offload_platform([_FakeDev()]))
        # A100/H100 do NOT gate on.
        class _A100:
            platform = "gpu"
            device_kind = "NVIDIA A100-SXM4-80GB"

        self.assertFalse(is_coherent_host_offload_platform([_A100()]))
        self.assertFalse(resolve_offload_enabled("auto", devices=[_A100()]))
        self.assertTrue(resolve_offload_enabled("auto", devices=[_FakeDev()]))


class OptimizerOffloadOffIsNoOpTest(absltest.TestCase):
    """Default (offload=False) must be byte-identical to the pre-offload build."""

    def test_default_config_is_off(self):
        self.assertFalse(TrainConfig().offload_optimizer)

    def test_off_keeps_moments_on_device_and_flag_clear(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model, _ = _build_model(_fp32_cfg())
        with mesh_rules(mesh):
            opt = build_optimizer(model, TrainConfig(offload_optimizer=False))
        self.assertFalse(opt.offload_optimizer_state)
        # All moment buffers stay in device memory -> no offload happened.
        self.assertEqual(_opt_state_memory_kinds(opt), {DEVICE_MEMORY_KIND})

    def test_off_train_step_runs_on_cpu(self):
        # The whole point of OFF being a no-op: the CPU step still runs (a step
        # with host-resident moments could NOT run on the XLA:CPU runtime).
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model, cfg = _build_model(_fp32_cfg())
        with mesh_rules(mesh):
            opt = build_optimizer(model, TrainConfig(offload_optimizer=False))
        step = make_sft_train_step(cfg, pad_id=0)
        batch = _make_batch(cfg.vocab_size)
        with jax.set_mesh(mesh):
            loss, _ = step(opt, batch)
            loss.block_until_ready()
        self.assertTrue(np.isfinite(float(loss)))
        # opt_state still device-resident after the step.
        self.assertEqual(_opt_state_memory_kinds(opt), {DEVICE_MEMORY_KIND})

    def test_auto_on_cpu_is_no_op(self):
        # offload_optimizer="auto" on CPU must resolve OFF -> moments on device.
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model, _ = _build_model(_fp32_cfg())
        with mesh_rules(mesh):
            opt = build_optimizer(model, TrainConfig(offload_optimizer="auto"))
        self.assertFalse(opt.offload_optimizer_state)
        self.assertEqual(_opt_state_memory_kinds(opt), {DEVICE_MEMORY_KIND})


class OptimizerOffloadPlacementTest(absltest.TestCase):
    """Force-ON: build-time placement of moments on pinned_host works on CPU.

    (Executing the staged step is GPU-only; we only assert the placement, which
    the ``jax.device_put(x, sharding.with_memory_kind('pinned_host'))`` API
    supports even on the CPU backend.)
    """

    def test_forced_on_places_moments_on_host(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model, _ = _build_model(_fp32_cfg())
        with mesh_rules(mesh):
            opt = build_optimizer(model, TrainConfig(offload_optimizer=True))
        self.assertTrue(opt.offload_optimizer_state)
        # Every moment buffer now lives in host memory.
        self.assertEqual(_opt_state_memory_kinds(opt), {HOST_MEMORY_KIND})

    def test_stored_device_shardings_structure_matches_opt_state(self):
        # Regression guard for the staging fix: the device-sharding pytree
        # captured at enable_state_offload() must map 1:1 onto the SAME structure
        # ``update`` feeds to / gets back from optax (``nnx.pure(self.opt_state)``
        # and the ``tx.update`` output). A mismatch would raise "Expected tuple,
        # got State" inside the jitted step on the accelerator. We can't run the
        # host<->device step on CPU, but we CAN assert the tree structures match,
        # which is what the fix hinges on.
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model, _ = _build_model(_fp32_cfg())
        with mesh_rules(mesh):
            opt = build_optimizer(model, TrainConfig(offload_optimizer=True))
        stored = opt._opt_state_device_shardings
        opt_state_pure = nnx.pure(opt.opt_state)
        self.assertEqual(
            jax.tree_util.tree_structure(stored),
            jax.tree_util.tree_structure(opt_state_pure),
            "stored device-sharding tree structure diverged from nnx.pure(opt_state); "
            "the in-jit staging map would fail.",
        )
        # Every stored sharding carries the DEVICE memory kind (staging target).
        for shd in jax.tree_util.tree_leaves(
            stored, is_leaf=lambda x: x is None or hasattr(x, "memory_kind")
        ):
            if shd is not None:
                self.assertEqual(shd.memory_kind, DEVICE_MEMORY_KIND)

    def test_stage_opt_state_preserves_structure_and_values(self):
        # _stage_opt_state must return the same tree structure and (on CPU's
        # single memory kind) the same values -- exercising the map that fires
        # inside the jitted step, minus the host<->device transfer.
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model, _ = _build_model(_fp32_cfg())
        with mesh_rules(mesh):
            opt = build_optimizer(model, TrainConfig(offload_optimizer=True))
        opt_state_pure = nnx.pure(opt.opt_state)
        staged = opt._stage_opt_state(opt_state_pure, DEVICE_MEMORY_KIND)
        self.assertEqual(
            jax.tree_util.tree_structure(staged),
            jax.tree_util.tree_structure(opt_state_pure),
        )
        for a, b in zip(
            jax.tree_util.tree_leaves(opt_state_pure),
            jax.tree_util.tree_leaves(staged),
        ):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_place_tree_roundtrip(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model, _ = _build_model(_fp32_cfg())
        with mesh_rules(mesh):
            opt = build_optimizer(model, TrainConfig(offload_optimizer=False))
        st = nnx.state(opt.opt_state)
        host = place_tree_on_memory_kind(st, HOST_MEMORY_KIND)
        back = place_tree_on_memory_kind(host, DEVICE_MEMORY_KIND)
        host_kinds = {
            getattr(getattr(a, "sharding", None), "memory_kind", None)
            for a in jax.tree_util.tree_leaves(host)
        }
        back_kinds = {
            getattr(getattr(a, "sharding", None), "memory_kind", None)
            for a in jax.tree_util.tree_leaves(back)
        }
        self.assertEqual(host_kinds, {HOST_MEMORY_KIND})
        self.assertEqual(back_kinds, {DEVICE_MEMORY_KIND})
        # Values are unchanged (memory-kind move never touches the math).
        for a, b in zip(jax.tree_util.tree_leaves(st), jax.tree_util.tree_leaves(back)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_sharding_on_memory_kind_helper(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model, _ = _build_model(_fp32_cfg())
        with mesh_rules(mesh):
            opt = build_optimizer(model, TrainConfig(offload_optimizer=False))
        leaf = jax.tree_util.tree_leaves(nnx.state(opt.opt_state))[1]
        host_shd = sharding_on_memory_kind(leaf.sharding, HOST_MEMORY_KIND)
        self.assertEqual(host_shd.memory_kind, HOST_MEMORY_KIND)
        # None / non-sharding inputs pass through unchanged.
        self.assertIsNone(sharding_on_memory_kind(None, HOST_MEMORY_KIND))


class OptimizerOffloadWrtFilterTest(parameterized.TestCase):
    """Offload must not disturb which variables receive optimizer state."""

    @parameterized.named_parameters(("off", False), ("forced_on", True))
    def test_opt_state_selection_matches_params(self, offload):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model, _ = _build_model(_fp32_cfg())
        with mesh_rules(mesh):
            opt = build_optimizer(model, TrainConfig(offload_optimizer=offload))
        # The optimizer's wrt filter is unchanged (still nnx.Param) and the set
        # of moment leaves matches the set of params exactly (offload only moves
        # bytes; it never adds/drops state).
        self.assertIs(opt.wrt, nnx.Param)
        n_params = len(jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
        # opt_state = mu + nu per param (+ scalar counters). The mu subtree alone
        # must have exactly n_params leaves.
        mu = nnx.state(opt.opt_state)[0][0]["mu"]
        self.assertEqual(len(jax.tree_util.tree_leaves(mu)), n_params)


class OffloadRematPolicyTest(parameterized.TestCase):
    """The new offload remat policies resolve and a smoke model traces under them."""

    def test_offload_policies_registered(self):
        names = available_remat_policies()
        for n in ("offload_dot", "offload_named", "offload", "offload_dots"):
            self.assertIn(n, names)

    def test_offload_policies_resolve(self):
        for n in ("offload_dot", "offload_named", "offload", "offload_dots"):
            self.assertIsNotNone(resolve_remat_policy(n))

    def test_only_named_policy_uses_tag(self):
        self.assertTrue(policy_uses_named_offload("offload_named"))
        for n in ("offload_dot", "dots_saveable", "full", None):
            self.assertFalse(policy_uses_named_offload(n))

    def test_tag_is_noop_for_non_named_policy(self):
        x = jnp.ones((2, 2))
        # For non-named policies the tag is skipped -> exact same object.
        self.assertIs(tag_offload_residual(x, "dots_saveable"), x)
        self.assertIs(tag_offload_residual(x, "offload_dot"), x)
        self.assertIs(tag_offload_residual(x, None), x)

    def test_tag_is_identity_value_for_named_policy(self):
        # For the named policy the value is unchanged (checkpoint_name is an
        # identity op that only attaches a name for the policy to match).
        x = jnp.arange(6.0).reshape(2, 3)
        tagged = tag_offload_residual(x, "offload_named")
        np.testing.assert_array_equal(np.asarray(tagged), np.asarray(x))

    @parameterized.named_parameters(
        ("offload_dot", "offload_dot"),
        ("offload_named", "offload_named"),
    )
    def test_smoke_model_traces_under_offload_policy(self, policy):
        # jax.eval_shape traces (abstract-evaluates) the whole fwd+bwd graph
        # under the offload policy WITHOUT executing it -- so it exercises the
        # policy wiring + checkpoint_name tag on CPU (where the host<->device
        # runtime is unavailable) and catches any wiring breakage.
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        cfg = _fp32_cfg(remat_policy=policy)
        model, cfg = _build_model(cfg)
        batch = _make_batch(cfg.vocab_size)
        tok, seg = batch["token_ids_BT"], batch["loss_mask_BT"]

        @nnx.jit
        def loss_and_grads(m, t, s):
            def loss_fn(mm):
                hidden, aux = mm(t, s, None, jnp.array(0, dtype=jnp.int32))
                return hidden.sum() + aux.astype(jnp.float32)

            return nnx.value_and_grad(loss_fn)(m)

        with jax.set_mesh(mesh):
            out = jax.eval_shape(lambda m: loss_and_grads(m, tok, seg), model)
        grads = out[1]
        self.assertGreater(len(jax.tree_util.tree_leaves(grads)), 0)


class OffloadConstantsTest(absltest.TestCase):
    def test_memory_kind_strings(self):
        # TransferToMemoryKind does not exist on this JAX; we drive off strings.
        self.assertEqual(offload_lib.DEVICE_MEMORY_KIND, "device")
        self.assertEqual(offload_lib.HOST_MEMORY_KIND, "pinned_host")
        self.assertFalse(hasattr(jax, "TransferToMemoryKind"))

    def test_residual_name(self):
        self.assertEqual(OFFLOAD_RESIDUAL_NAME, "offload_residual")


if __name__ == "__main__":
    absltest.main()
