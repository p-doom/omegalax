"""Structural and parameter-count tests for the Qwen3.5 memory-scale config ladder."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.tree_util
from absl.testing import absltest, parameterized
from flax import nnx

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
from omegalax.text import api as text_api


def _expected_layer_types(num_hidden_layers: int) -> tuple[str, ...]:
    return tuple(
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(num_hidden_layers)
    )


class Qwen3_5MemoryConfigLadderTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("50m", "qwen3.5-memory-50m", 9, 512, 47_996_400),
        ("200m", "qwen3.5-memory-200m", 17, 768, 200_558_776),
    )
    def test_config_resolves_to_expected_model(
        self, model_id, num_hidden_layers, hidden_size, expected_param_count
    ):
        cfg = text_api.resolve_config(model_id)

        self.assertEqual(cfg.num_hidden_layers, num_hidden_layers)
        self.assertEqual(cfg.hidden_size, hidden_size)
        self.assertTrue(cfg.tie_word_embeddings)
        self.assertEqual(cfg.layer_types, _expected_layer_types(num_hidden_layers))

        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            abs_model = nnx.eval_shape(lambda: Qwen3_5ForCausalLM(cfg, rngs=nnx.Rngs(0)))

        self.assertIsNone(abs_model.lm_head)

        _, abs_state = nnx.split(abs_model)
        pure_state = nnx.to_pure_dict(abs_state)
        param_count = sum(math.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(pure_state))
        self.assertEqual(param_count, expected_param_count)


if __name__ == "__main__":
    absltest.main()
