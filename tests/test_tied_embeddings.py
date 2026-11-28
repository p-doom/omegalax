"""Tied word embeddings: one parameter serves the input lookup and the output head, in training,
export to HF and checkpoints."""

import dataclasses
import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
from absl.testing import absltest
from flax import nnx
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration as HFQwen3_5,
)

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3.config import make_config as make_qwen3_config
from omegalax.models.qwen3.model import Qwen3
from omegalax.models.qwen3.params import export_qwen3_to_safetensors
from omegalax.models.qwen3_5 import Qwen3_5ForConditionalGeneration, make_config
from omegalax.models.qwen3_5.params import export_qwen3_5_to_safetensors
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.trainers.loss import chunked_cross_entropy_loss
from tests.logits_assert import assert_logits_close

T = 24


def _qwen3(tied: bool):
    cfg = dataclasses.replace(make_qwen3_config("qwen3-smoke"), tie_word_embeddings=tied)
    model = Qwen3(cfg, rngs=nnx.Rngs(params=jax.random.key(0)))
    set_attn_backend(model, text_backend="xla")
    return model, cfg


def _qwen3_5(tied: bool):
    cfg = make_config("qwen3.5-smoke-dense")
    cfg = dataclasses.replace(
        cfg, text_config=dataclasses.replace(cfg.text_config, tie_word_embeddings=tied)
    )
    model = Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(params=jax.random.key(0)))
    set_attn_backend(model, text_backend="xla")
    return model, cfg


def _tokens(vocab):
    return jnp.asarray(np.random.default_rng(0).integers(1, vocab, (2, T)), jnp.int32)


def _loss(model, tokens):
    """The trainers' loss: chunked cross-entropy against ``model.output_weight()``."""
    hidden_BTD, _ = model(tokens, jnp.ones_like(tokens), None, jnp.array(0, jnp.int32))
    return chunked_cross_entropy_loss(
        hidden_BTD, model.output_weight(), tokens, jnp.ones_like(tokens), num_tiles=2
    )


_grad = nnx.jit(nnx.grad(_loss))


@nnx.jit
def _logits_jit(model, tokens):
    hidden_BTD, _ = model(tokens, jnp.ones_like(tokens), None, jnp.array(0, jnp.int32))
    return hidden_BTD @ model.output_weight()


def _logits(model, tokens):
    return np.asarray(_logits_jit(model, tokens), np.float32)


def _embedding(model):
    if isinstance(model, Qwen3_5ForConditionalGeneration):
        return model.text.embedder
    return model.embedder


class TiedEmbeddingTest(absltest.TestCase):
    """Qwen3.5's text forward does not run on CPU on this branch (tokamax's XLA attention
    repeats KV heads without an out_sharding), so its checks here need no forward; it shares
    ``_output_weight`` with its causal LM and the engine's tests cover it end to end."""

    def test_one_parameter(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            for build in (_qwen3, _qwen3_5):
                model, _ = build(tied=True)
                self.assertIsNone(model.lm_head)
                weight = model.output_weight()
                np.testing.assert_array_equal(weight, _embedding(model).embedding[...].T)
                self.assertEqual(weight.dtype, jnp.float32)

    def test_grad_is_sum_of_lookup_and_head_paths(self):
        """The tied parameter's gradient equals an untied twin's embedding gradient plus its
        head gradient, when the twin's head starts as the embedding transposed."""
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            tied, _ = _qwen3(tied=True)
            twin, _ = _qwen3(tied=False)
            table = _embedding(tied).embedding[...]
            nnx.update(twin, nnx.state(tied, nnx.Param))
            twin.lm_head.kernel.set_value(table.T)
            tokens = _tokens(table.shape[0])
            g_tied, g_twin = _grad(tied, tokens), _grad(twin, tokens)
            want = g_twin.embedder.embedding[...] + g_twin.lm_head.kernel[...].T
            np.testing.assert_allclose(g_tied.embedder.embedding[...], want, rtol=1e-5, atol=1e-7)
            self.assertGreater(np.abs(g_twin.lm_head.kernel[...]).max(), 0)

    def test_export_reloads_in_hf_after_a_step(self):
        """After an SGD step the export holds one vocab matrix, and HF ``from_pretrained``
        computes the trained model's logits."""
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model, cfg = _qwen3(tied=True)
            tokens = _tokens(cfg.vocab_size)
            params = nnx.state(model, nnx.Param)
            nnx.update(model, jax.tree.map(lambda p, g: p - g, params, _grad(model, tokens)))
            ours = _logits(model, tokens)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_to_safetensors(model, cfg, tmpdir)
                self.assertNotIn("lm_head.weight", _tensor_names(tmpdir))
                hf = AutoModelForCausalLM.from_pretrained(tmpdir, dtype=torch.float32).eval()
                with torch.no_grad():
                    theirs = hf(input_ids=torch.tensor(np.asarray(tokens))).logits.numpy()
        print(f"max |logit diff| {np.abs(ours - theirs).max():.4f}")
        assert_logits_close(self, ours, theirs)

    def test_qwen3_5_export_is_tied_in_hf(self):
        """A changed embedding exports once, and HF's head is that matrix."""
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model, cfg = _qwen3_5(tied=True)
            table = _embedding(model).embedding
            table.set_value(table[...] + jax.random.normal(jax.random.key(1), table.shape))
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_5_to_safetensors(model, cfg, tmpdir)
                self.assertNotIn("lm_head.weight", _tensor_names(tmpdir))
                hf = HFQwen3_5.from_pretrained(tmpdir, dtype=torch.float32)
        head = hf.get_output_embeddings().weight.detach().numpy()
        np.testing.assert_array_equal(head, np.asarray(model.output_weight()).T)

    def test_save_restore(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            for build in (_qwen3, _qwen3_5):
                model, _ = build(tied=True)
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = os.path.join(tmpdir, "state")
                    with ocp.StandardCheckpointer() as ckpt:
                        ckpt.save(path, nnx.state(model))
                    restored, _ = build(tied=True)
                    table = _embedding(restored).embedding
                    table.set_value(jnp.zeros_like(table[...]))
                    with ocp.StandardCheckpointer() as ckpt:
                        state = ckpt.restore(path, nnx.state(restored))
                self.assertNotIn("lm_head", nnx.to_pure_dict(state))
                nnx.update(restored, state)
                np.testing.assert_array_equal(restored.output_weight(), model.output_weight())


def _tensor_names(export_dir):
    with safe_open(os.path.join(export_dir, "model.safetensors"), "np") as f:
        return list(f.keys())


if __name__ == "__main__":
    absltest.main()
