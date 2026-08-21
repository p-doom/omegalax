"""Round-trip export/import smoke tests for all supported families."""

import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.export import model_config_to_hf_dict
from omegalax.models.params_utils import flatten_pure_state
from omegalax.models.qwen3.config import make_config as make_qwen3_config
from omegalax.models.qwen3.model import Qwen3
from omegalax.models.qwen3.loader import create_qwen3_from_safetensors
from omegalax.models.qwen3.params import export_qwen3_to_safetensors
from omegalax.models.qwen3_vl import Qwen3VL, make_vl_config
from omegalax.models.qwen3_vl.params import (
    create_qwen3_vl_from_safetensors,
    export_qwen3_vl_to_safetensors,
)
from omegalax.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration, make_config
from omegalax.models.qwen3_5.params import (
    create_qwen3_5_from_safetensors,
    export_qwen3_5_to_safetensors,
)


def _flatten_model(model):
    _, state = nnx.split(model)
    pure = nnx.to_pure_dict(state)
    return flatten_pure_state(pure)


def _assert_params_equal(testcase: absltest.TestCase, model_a, model_b):
    flat_a = _flatten_model(model_a)
    flat_b = _flatten_model(model_b)
    testcase.assertSetEqual(set(flat_a.keys()), set(flat_b.keys()))
    for key in flat_a:
        a = np.asarray(jax.device_get(flat_a[key]))
        b = np.asarray(jax.device_get(flat_b[key]))
        testcase.assertEqual(a.shape, b.shape, f"Shape mismatch at {key}")
        np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=key)


class ExportRoundTripTest(absltest.TestCase):
    def test_qwen3_dense_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg = make_qwen3_config("qwen3-smoke")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_to_safetensors(model, cfg, tmpdir)
                loaded = create_qwen3_from_safetensors(
                    tmpdir, "qwen3-smoke", tp_size=1, fsdp_size=1, dp_size=1
                )
        _assert_params_equal(self, model, loaded)

    def test_qwen3_moe_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg = make_qwen3_config("qwen3-smoke-moe")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_to_safetensors(model, cfg, tmpdir)
                loaded = create_qwen3_from_safetensors(
                    tmpdir, "qwen3-smoke-moe", tp_size=1, fsdp_size=1, dp_size=1
                )
        _assert_params_equal(self, model, loaded)

    def test_qwen3_vl_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg = make_vl_config("qwen3-vl-smoke")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3VL(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_vl_to_safetensors(model, cfg, tmpdir)
                loaded, _ = create_qwen3_vl_from_safetensors(
                    tmpdir,
                    "qwen3-vl-smoke",
                    tp_size=1,
                    fsdp_size=1,
                    dp_size=1,
                )
        _assert_params_equal(self, model, loaded)

    def test_qwen3_vl_moe_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg = make_vl_config("qwen3-vl-smoke-moe")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3VL(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_vl_to_safetensors(model, cfg, tmpdir)
                loaded, _ = create_qwen3_vl_from_safetensors(
                    tmpdir,
                    "qwen3-vl-smoke-moe",
                    tp_size=1,
                    fsdp_size=1,
                    dp_size=1,
                )
        _assert_params_equal(self, model, loaded)

    def test_qwen3_5_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg: Qwen3_5Config = make_config("qwen3.5-smoke")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3_5ForConditionalGeneration(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_5_to_safetensors(model, cfg, tmpdir)
                loaded, _ = create_qwen3_5_from_safetensors(
                    tmpdir,
                    "qwen3.5-smoke",
                    tp_size=1,
                    fsdp_size=1,
                    dp_size=1,
                )
        _assert_params_equal(self, model, loaded)

    def test_qwen3_5_dense_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg: Qwen3_5Config = make_config("qwen3.5-smoke-dense")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3_5ForConditionalGeneration(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_5_to_safetensors(model, cfg, tmpdir)
                loaded, _ = create_qwen3_5_from_safetensors(
                    tmpdir,
                    "qwen3.5-smoke-dense",
                    tp_size=1,
                    fsdp_size=1,
                    dp_size=1,
                )
        _assert_params_equal(self, model, loaded)


# The top-level keys a serving stack dereferences, per model_type, taken from the
# published Qwen configs these exports stand in for. `architectures` alone is not
# enough to assert: the export that killed an eval was missing it *and*
# `vision_end_token_id`, and a check for one would have passed the other.
_SERVABLE_KEYS = {
    "qwen3": {
        "architectures",
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "tie_word_embeddings",
    },
    "qwen3_moe": {
        "architectures",
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "tie_word_embeddings",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    },
    "qwen3_vl": {
        "architectures",
        "model_type",
        "text_config",
        "vision_config",
        "tie_word_embeddings",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    },
    "qwen3_vl_moe": {
        "architectures",
        "model_type",
        "text_config",
        "vision_config",
        "tie_word_embeddings",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    },
    "qwen3_5": {
        "architectures",
        "model_type",
        "text_config",
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    },
    "qwen3_5_moe": {
        "architectures",
        "model_type",
        "text_config",
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    },
}


class ExportedConfigIsServableTest(absltest.TestCase):
    """Every family's exported config.json must name a resolvable architecture and
    carry the keys a serving stack reads."""

    def _check(self, cfg):
        import transformers

        hf_cfg = model_config_to_hf_dict(cfg)
        model_type = hf_cfg["model_type"]
        self.assertIn(model_type, _SERVABLE_KEYS, f"no servable key set declared for {model_type}")

        missing = _SERVABLE_KEYS[model_type] - set(hf_cfg)
        self.assertEqual(
            set(), missing, f"{model_type} export omits servable keys: {sorted(missing)}"
        )

        arch = hf_cfg["architectures"]
        self.assertIsInstance(arch, list)
        self.assertLen(arch, 1)
        # sglang does `hf_config.architectures[0] in ...`; transformers instantiates
        # the class by this name. A plausible-looking string that names nothing is
        # exactly as unservable as None.
        self.assertTrue(
            hasattr(transformers, arch[0]),
            f"{model_type} declares architecture {arch[0]!r}, which transformers "
            f"{transformers.__version__} does not define",
        )

    def test_qwen3_dense_config_servable(self):
        self._check(make_qwen3_config("qwen3-smoke"))

    def test_qwen3_moe_config_servable(self):
        self._check(make_qwen3_config("qwen3-smoke-moe"))

    def test_qwen3_vl_config_servable(self):
        self._check(make_vl_config("qwen3-vl-smoke"))

    def test_qwen3_vl_moe_config_servable(self):
        self._check(make_vl_config("qwen3-vl-smoke-moe"))

    def test_qwen3_5_config_servable(self):
        self._check(make_config("qwen3.5-smoke"))

    def test_qwen3_5_dense_config_servable(self):
        self._check(make_config("qwen3.5-smoke-dense"))


if __name__ == "__main__":
    absltest.main()
