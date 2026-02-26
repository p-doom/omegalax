"""Round-trip export/import test for Qwen3 dense models."""

import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from absl.testing import absltest
from flax import nnx

from omegalax.models.qwen3.params import create_qwen3_from_safetensors
from omegalax.models.qwen3.dense.params_dense import export_qwen3_dense_to_safetensors
from omegalax.text import api as text_api


def _get_value(model: nnx.Module, dotted: str):
    state = nnx.to_pure_dict(nnx.state(model))
    tokens = dotted.split(".")
    node = state
    for t in tokens:
        key = int(t) if t.isdigit() else t
        node = node[key]
    return node


class ExportDenseTest(absltest.TestCase):
    def test_round_trip_export_import(self):
        rng = jax.random.key(0)
        model, cfg = text_api.init_model("qwen3-smoke", rng, tp_size=1, fsdp_size=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_qwen3_dense_to_safetensors(model, cfg, tmpdir)
            reloaded = create_qwen3_from_safetensors(tmpdir, "qwen3-smoke", tp_size=1, fsdp_size=1)

            embed_orig = np.asarray(jax.device_get(_get_value(model, "embedder.embedding")))
            embed_new = np.asarray(jax.device_get(_get_value(reloaded, "embedder.embedding")))
            np.testing.assert_allclose(embed_new, embed_orig, atol=0, rtol=0)

            q_proj_orig = np.asarray(jax.device_get(_get_value(model, "layers.0.attn.q_proj.kernel")))
            q_proj_new = np.asarray(jax.device_get(_get_value(reloaded, "layers.0.attn.q_proj.kernel")))
            np.testing.assert_allclose(q_proj_new, q_proj_orig, atol=0, rtol=0)

            norm_orig = np.asarray(jax.device_get(_get_value(model, "layers.0.input_layernorm.scale")))
            norm_new = np.asarray(jax.device_get(_get_value(reloaded, "layers.0.input_layernorm.scale")))
            np.testing.assert_allclose(norm_new, norm_orig, atol=0, rtol=0)


if __name__ == "__main__":
    absltest.main()
