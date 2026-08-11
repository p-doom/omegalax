"""Unit tests for the dedicated vision-tower learning rate.

These deliberately avoid the Grain data pipeline so the module imports and runs
standalone: they exercise the optimizer-construction path (label routing +
optax.multi_transform) directly, driving it exactly as
``MixedPrecisionOptimizer.update`` does (pure state + params, fp32 grads).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp

from omegalax.trainers import vlm as vlm_trainer
from omegalax.trainers.vlm import TrainConfig, build_optimizer


class _Sub(nnx.Module):
    def __init__(self, rngs, dtype=jnp.float32):
        self.lin = nnx.Linear(4, 4, rngs=rngs, param_dtype=dtype)


class _TinyVLM(nnx.Module):
    """Minimal stand-in with a ``vision`` submodule, so the state-tree path key
    ``"vision"`` (the same key the real Qwen3VL exposes) is present."""

    def __init__(self, rngs, dtype=jnp.float32):
        self.vision = _Sub(rngs, dtype)
        self.decoder = _Sub(rngs, dtype)


def _one_leaf(model, want_vision):
    """Return one param array from the vision (or non-vision) subtree."""
    out = {}
    jax.tree_util.tree_map_with_path(
        lambda path, v: (
            out.setdefault("v", v) if vlm_trainer._is_vision_path(path) == want_vision else None
        ),
        nnx.state(model, nnx.Param),
    )
    return out["v"]


class VisionLabelRoutingTest(absltest.TestCase):
    def test_label_fn_matches_is_vision_path_and_splits_both_groups(self):
        model = _TinyVLM(nnx.Rngs(0))
        params = nnx.state(model, nnx.Param)
        labels = vlm_trainer._vision_label_fn(params)

        seen = {"vision": 0, "text": 0}

        def _check(path, label):
            expected = "vision" if vlm_trainer._is_vision_path(path) else "text"
            self.assertEqual(label, expected)
            seen[label] += 1

        jax.tree_util.tree_map_with_path(_check, labels)
        # Both groups populated => multi_transform genuinely partitions the tree.
        self.assertGreater(seen["vision"], 0)
        self.assertGreater(seen["text"], 0)


class BuildOptimizerVisionLRTest(absltest.TestCase):
    def _run_one_step(self, model, cfg, vision_lr_schedule_fn):
        before_v = jnp.asarray(_one_leaf(model, True))
        before_t = jnp.asarray(_one_leaf(model, False))
        opt = build_optimizer(
            model,
            cfg.learning_rate,
            cfg,
            wrt=nnx.Param,
            vision_lr_schedule_fn=vision_lr_schedule_fn,
        )
        # Unit gradients everywhere: a first AdamW step moves each param by ~lr.
        grads = jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param))
        opt.update(nnx.State(grads))
        vis = float(jnp.abs(_one_leaf(model, True) - before_v).mean())
        txt = float(jnp.abs(_one_leaf(model, False) - before_t).mean())
        return vis, txt

    def test_routes_vision_through_dedicated_lr(self):
        text_lr, vision_lr = 1.0, 0.1
        cfg = TrainConfig(learning_rate=text_lr, weight_decay=0.0, vision_learning_rate=vision_lr)
        model = _TinyVLM(nnx.Rngs(0))
        vis, txt = self._run_one_step(model, cfg, vision_lr_schedule_fn=vision_lr)
        self.assertAlmostEqual(vis, vision_lr, delta=5e-3)
        self.assertAlmostEqual(txt, text_lr, delta=5e-3)

    def test_default_uses_single_lr_for_all_params(self):
        text_lr = 1.0
        cfg = TrainConfig(learning_rate=text_lr, weight_decay=0.0)
        model = _TinyVLM(nnx.Rngs(0))
        vis, txt = self._run_one_step(model, cfg, vision_lr_schedule_fn=None)
        self.assertAlmostEqual(vis, text_lr, delta=5e-3)
        self.assertAlmostEqual(txt, text_lr, delta=5e-3)

    def test_bf16_params_route_correctly(self):
        # Mirror production dtype: bf16 params, fp32 accumulation in the opt.
        text_lr, vision_lr = 1.0, 0.1
        cfg = TrainConfig(learning_rate=text_lr, weight_decay=0.0, vision_learning_rate=vision_lr)
        model = _TinyVLM(nnx.Rngs(0), dtype=jnp.bfloat16)
        vis, txt = self._run_one_step(model, cfg, vision_lr_schedule_fn=vision_lr)
        self.assertAlmostEqual(vis, vision_lr, delta=1e-2)
        self.assertAlmostEqual(txt, text_lr, delta=1e-2)


if __name__ == "__main__":
    absltest.main()
