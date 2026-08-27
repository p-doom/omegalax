"""Unit tests for the ``--train_vision_merger`` trainable-parameter filter.

Run on CPU; no model loading. Validate:
* ``_trainable_non_vision`` (flag off) still excludes the whole vision subtree
* ``_trainable_non_vision_except_merger`` selects exactly ``vision.merger.*``
  on top of the text params, and nothing else from the vision tower
* the merger param set matches the real Qwen3.5 ``VisionModel`` attribute
  names (norm.weight/bias, fc1.kernel/bias, fc2.kernel/bias)
* deepstack mergers (Qwen3-VL) are NOT carved out
* after one optimizer step the merger weights moved while the rest of the
  vision tower is bit-exact unchanged
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax

from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.qwen3_5.config import Qwen3_5VisionConfig
from omegalax.models.qwen3_5.vision import VisionModel
from omegalax.models.shard_config import ShardConfig
from omegalax.trainers.vlm import (
    TrainConfig,
    _trainable_non_vision,
    _trainable_non_vision_except_merger,
)

MERGER_PARAM_PATHS = {
    "vision.merger.fc1.bias",
    "vision.merger.fc1.kernel",
    "vision.merger.fc2.bias",
    "vision.merger.fc2.kernel",
    "vision.merger.norm.bias",
    "vision.merger.norm.weight",
}


def _tiny_vision_config() -> Qwen3_5VisionConfig:
    return Qwen3_5VisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        out_hidden_size=8,
        num_position_embeddings=16,
        dtype=jnp.float32,
    )


class _MiniTextLayer(nnx.Module):
    def __init__(self, d: int, *, rngs: nnx.Rngs):
        self.q_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.o_proj(self.q_proj(x))


class _MiniText(nnx.Module):
    def __init__(self, d: int, n_layers: int, *, rngs: nnx.Rngs):
        self.embedder = nnx.Embed(32, d, rngs=rngs)
        self.layers = nnx.List([_MiniTextLayer(d, rngs=rngs) for _ in range(n_layers)])

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _RealVisionVLM(nnx.Module):
    """Real Qwen3.5 ``VisionModel`` under ``.vision`` + a stand-in text tower."""

    def __init__(self, d: int = 8, n_layers: int = 2, *, rngs: nnx.Rngs):
        self.vision = VisionModel(
            _tiny_vision_config(), shd_cfg=ShardConfig.no_sharding(), rngs=rngs
        )
        self.text = _MiniText(d, n_layers, rngs=rngs)
        self.lm_head = nnx.Linear(d, 32, use_bias=False, rngs=rngs)


class _MiniMerger(nnx.Module):
    def __init__(self, d: int, *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(d, rngs=rngs)
        self.fc1 = nnx.Linear(d, d, rngs=rngs)
        self.fc2 = nnx.Linear(d, d, rngs=rngs)

    def __call__(self, x):
        return self.fc2(nnx.gelu(self.fc1(self.norm(x))))


class _MiniVisionBlock(nnx.Module):
    def __init__(self, d: int, *, rngs: nnx.Rngs):
        self.q_proj = nnx.Linear(d, d, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.q_proj(x)


class _MiniVisionTower(nnx.Module):
    def __init__(self, d: int, *, rngs: nnx.Rngs):
        self.patch_embed = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.blocks = nnx.List([_MiniVisionBlock(d, rngs=rngs) for _ in range(2)])
        self.deepstack_mergers = nnx.List([_MiniMerger(d, rngs=rngs)])
        self.merger = _MiniMerger(d, rngs=rngs)

    def __call__(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.merger(x)


class _MiniVLM(nnx.Module):
    def __init__(self, d: int = 8, *, rngs: nnx.Rngs):
        self.vision = _MiniVisionTower(d, rngs=rngs)
        self.text = _MiniText(d, 2, rngs=rngs)
        self.lm_head = nnx.Linear(d, 32, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.lm_head(self.text(self.vision(x)))


def _selected_paths(model, filt) -> set[str]:
    state = nnx.state(model, filt)
    return {".".join(str(p) for p in path) for path, _ in nnx.to_flat_state(state)}


def _param_count(model, filt) -> int:
    state = nnx.state(model, filt)
    return sum(int(x.size) for x in jax.tree.leaves(nnx.to_pure_dict(state)))


class VisionMergerFilterTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)

    def _real_model(self):
        with mesh_rules(self.mesh):
            return _RealVisionVLM(rngs=nnx.Rngs(0))

    def test_train_config_default_is_off(self):
        self.assertFalse(TrainConfig().train_vision_merger)

    def test_default_filter_excludes_entire_vision_subtree(self):
        model = self._real_model()
        selected = _selected_paths(model, _trainable_non_vision)
        self.assertTrue(selected)
        self.assertFalse({p for p in selected if p.startswith("vision.")})

    def test_default_filter_unchanged_by_new_code(self):
        model = self._real_model()
        all_params = _selected_paths(model, nnx.Param)
        expected = {p for p in all_params if not p.startswith("vision.")}
        self.assertEqual(_selected_paths(model, _trainable_non_vision), expected)

    def test_merger_filter_selects_exactly_merger_plus_text(self):
        model = self._real_model()
        all_params = _selected_paths(model, nnx.Param)
        text_params = {p for p in all_params if not p.startswith("vision.")}
        selected = _selected_paths(model, _trainable_non_vision_except_merger)
        self.assertEqual(selected, text_params | MERGER_PARAM_PATHS)

    def test_merger_paths_match_real_vision_model(self):
        model = self._real_model()
        all_params = _selected_paths(model, nnx.Param)
        self.assertEqual(
            {p for p in all_params if p.startswith("vision.merger.")}, MERGER_PARAM_PATHS
        )

    def test_merger_filter_excludes_rest_of_tower(self):
        model = self._real_model()
        selected = _selected_paths(model, _trainable_non_vision_except_merger)
        excluded = _selected_paths(model, nnx.Param) - selected
        self.assertTrue(excluded)
        for path in excluded:
            self.assertTrue(path.startswith("vision."), path)
            self.assertFalse(path.startswith("vision.merger."), path)
        self.assertIn("vision.patch_embed.proj.kernel", excluded)
        self.assertIn("vision.pos_embed.embedding", excluded)
        self.assertIn("vision.blocks.0.attn.qkv.kernel", excluded)

    def test_merger_filter_param_count_delta_is_merger_only(self):
        model = self._real_model()
        base = _param_count(model, _trainable_non_vision)
        with_merger = _param_count(model, _trainable_non_vision_except_merger)
        merger_numel = _param_count(model.vision.merger, nnx.Param)
        self.assertEqual(with_merger - base, merger_numel)

    def test_deepstack_mergers_are_not_carved_out(self):
        model = _MiniVLM(rngs=nnx.Rngs(0))
        selected = _selected_paths(model, _trainable_non_vision_except_merger)
        self.assertFalse({p for p in selected if p.startswith("vision.deepstack_mergers")})
        self.assertTrue({p for p in selected if p.startswith("vision.merger.")})

    def test_non_param_variables_rejected(self):
        rest = nnx.Variable(jnp.zeros((2,)))
        self.assertFalse(_trainable_non_vision_except_merger(("text", "x"), rest))
        self.assertFalse(_trainable_non_vision(("text", "x"), rest))

    def test_optimizer_step_moves_merger_and_freezes_the_rest(self):
        model = _MiniVLM(rngs=nnx.Rngs(0))
        before = jax.tree.map(lambda v: np.array(v), nnx.to_pure_dict(nnx.state(model, nnx.Param)))
        tx = optax.sgd(0.1)
        optimizer = nnx.Optimizer(model, tx, wrt=_trainable_non_vision_except_merger)
        x = jax.random.normal(jax.random.key(0), (4, 8), dtype=jnp.float32)

        def loss_fn(m):
            return jnp.mean(m(x) ** 2)

        grads = nnx.grad(loss_fn, argnums=nnx.DiffState(0, _trainable_non_vision_except_merger))(
            model
        )
        optimizer.update(model, grads)
        after = jax.tree.map(lambda v: np.array(v), nnx.to_pure_dict(nnx.state(model, nnx.Param)))

        self.assertFalse(
            np.array_equal(
                before["vision"]["merger"]["fc1"]["kernel"],
                after["vision"]["merger"]["fc1"]["kernel"],
            )
        )
        self.assertFalse(
            np.array_equal(
                before["vision"]["merger"]["fc2"]["kernel"],
                after["vision"]["merger"]["fc2"]["kernel"],
            )
        )
        np.testing.assert_array_equal(
            before["vision"]["patch_embed"]["kernel"], after["vision"]["patch_embed"]["kernel"]
        )
        np.testing.assert_array_equal(
            before["vision"]["blocks"][0]["q_proj"]["kernel"],
            after["vision"]["blocks"][0]["q_proj"]["kernel"],
        )
        np.testing.assert_array_equal(
            before["vision"]["deepstack_mergers"][0]["fc1"]["kernel"],
            after["vision"]["deepstack_mergers"][0]["fc1"]["kernel"],
        )
        self.assertFalse(
            np.array_equal(
                before["text"]["layers"][0]["q_proj"]["kernel"],
                after["text"]["layers"][0]["q_proj"]["kernel"],
            )
        )


if __name__ == "__main__":
    absltest.main()
