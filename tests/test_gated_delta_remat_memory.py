"""Compile-time memory of Gated DeltaNet layers under the decoder's remat (GPU, no weights)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import dataclasses

import jax
import jax.numpy as jnp
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import mesh_rules
from omegalax.models.qwen3_5 import model as qwen3_5_model
from omegalax.models.qwen3_5.config import make_config
from omegalax.text import api as text_api

B, T = 4, 4096
# The plain-JAX state pass keeps ~3.9 KB/token per layer alive; cross-layer hoisting of the
# backward's zero-initialized gradient buffers once kept 53.6.
MAX_KB_PER_TOKEN_PER_LAYER = 6.0


def _temp_bytes(num_layers: int) -> int:
    """Temp bytes of jit(grad(loss)) for a stack of Qwen3.5-2B-shaped DeltaNet layers."""
    mesh = jax.make_mesh((1, 1, 1), ("tp", "fsdp", "dp"), devices=jax.devices()[:1])
    cfg = dataclasses.replace(
        make_config("qwen3.5-smoke-dense").text_config,
        vocab_size=248320,
        tie_word_embeddings=True,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=num_layers,
        layer_types=("linear_attention",) * num_layers,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    cfg = text_api.align_config_to_mesh(cfg, mesh)
    with mesh_rules(mesh):
        abstract = nnx.eval_shape(lambda: qwen3_5_model.Qwen3_5ForCausalLM(cfg, rngs=nnx.Rngs(0)))
    graphdef, params = nnx.split(abstract)
    tokens = jnp.ones((B, T), jnp.int32)

    def loss(params):
        hidden, _ = nnx.merge(graphdef, params)(tokens, jnp.ones_like(tokens), None, None)
        return jnp.sum(hidden.astype(jnp.float32) ** 2)

    with jax.set_mesh(mesh):
        compiled = jax.jit(jax.grad(loss)).lower(params).compile()
    return compiled.memory_analysis().temp_size_in_bytes


class RematMemoryTest(absltest.TestCase):
    def test_deltanet_layer_memory_per_token(self):
        per_layer = (_temp_bytes(6) - _temp_bytes(2)) / 4 / (B * T) / 1024
        print(f"{per_layer:.1f} KB/token per Gated DeltaNet layer")
        self.assertLess(per_layer, MAX_KB_PER_TOKEN_PER_LAYER)


if __name__ == "__main__":
    absltest.main()
