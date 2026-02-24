"""Round-trip export/import smoke tests for all supported families."""

import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from absl.testing import absltest
from flax import nnx

from omegalax.models.params_utils import flatten_pure_state
from omegalax.models.qwen3.dense.config import make_dense_config
from omegalax.models.qwen3.dense.model import Qwen3Dense
from omegalax.models.qwen3.dense.params_dense import (
    create_qwen3_dense_from_safetensors,
    export_qwen3_dense_to_safetensors,
)
from omegalax.models.qwen3.moe.config import make_moe_config
from omegalax.models.qwen3.moe.model import Qwen3Moe
from omegalax.models.qwen3.moe.params_moe import (
    create_qwen3_moe_from_safetensors,
    export_qwen3_moe_to_safetensors,
)
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
        cfg = make_dense_config("qwen3-smoke")
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = Qwen3Dense(cfg, rngs=rngs)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_qwen3_dense_to_safetensors(model, cfg, tmpdir)
            loaded = create_qwen3_dense_from_safetensors(tmpdir, "qwen3-smoke")
        _assert_params_equal(self, model, loaded)

    def test_qwen3_moe_round_trip(self):
        cfg = make_moe_config("qwen3-smoke-moe")
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = Qwen3Moe(cfg, rngs=rngs)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_qwen3_moe_to_safetensors(model, cfg, tmpdir)
            loaded = create_qwen3_moe_from_safetensors(tmpdir, "qwen3-smoke-moe")
        _assert_params_equal(self, model, loaded)

    def test_qwen3_vl_round_trip(self):
        cfg = make_vl_config("qwen3-vl-smoke")
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = Qwen3VL(cfg, rngs=rngs)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_qwen3_vl_to_safetensors(model, cfg, tmpdir)
            loaded, _ = create_qwen3_vl_from_safetensors(tmpdir, "qwen3-vl-smoke")
        _assert_params_equal(self, model, loaded)

    def test_qwen3_vl_moe_round_trip(self):
        cfg = make_vl_config("qwen3-vl-smoke-moe")
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = Qwen3VL(cfg, rngs=rngs)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_qwen3_vl_to_safetensors(model, cfg, tmpdir)
            loaded, _ = create_qwen3_vl_from_safetensors(tmpdir, "qwen3-vl-smoke-moe")
        _assert_params_equal(self, model, loaded)

    def test_qwen3_5_round_trip(self):
        cfg: Qwen3_5Config = make_config("qwen3.5-smoke")
        rngs = nnx.Rngs(params=jax.random.key(0))
        model = Qwen3_5ForConditionalGeneration(cfg, rngs=rngs)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_qwen3_5_to_safetensors(model, cfg, tmpdir)
            loaded, _ = create_qwen3_5_from_safetensors(tmpdir, "qwen3.5-smoke")
        _assert_params_equal(self, model, loaded)


if __name__ == "__main__":
    absltest.main()
