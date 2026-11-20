"""Qwen3.5 GDN/conv state hand-over, padding, sharding, generation and checkpoint loading."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=1"
).strip()

import dataclasses
import functools
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from absl import flags
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from omegalax import export as export_lib
from omegalax.distributed.mesh import ensure_mesh, mesh_rules, mesh_rules_for
from omegalax.models.params_utils import save_hf_config
from omegalax.models.qwen3_5 import Qwen3_5ForConditionalGeneration, make_config
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.qwen3_5.deltanet import GatedDeltaNet
from omegalax.models.qwen3_5.kernels.xla_reference import chunk_gated_delta_rule_xla
from omegalax.models.qwen3_5.params import (
    create_qwen3_5_from_safetensors,
    export_qwen3_5_to_safetensors,
)
from omegalax.models.shard_config import ShardConfig
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.text.chat import ChatMode, ChatRuntime, StatePassingConfig, generate_token_ids
from omegalax.text.checkpoint import resolve_checkpoint, restore_model_params
from omegalax.trainers import pretrain as pt
from omegalax.trainers.lora import inject_lora
from tests.test_inference_checkpoint import _save_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


def _give_gdn_memory(layers) -> None:
    # Random init forgets the GDN state within one or two tokens; A_log = -4 keeps it ~70 tokens.
    for attn in layers:
        attn.A_log[...] = jnp.full_like(attn.A_log[...], -4.0)


@functools.lru_cache(maxsize=None)
def _smoke_f32_model():
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["test_qwen3_5_state_handover"])
    cfg = dataclasses.replace(text_api.resolve_config("qwen3.5-smoke-dense"), dtype=jnp.float32)
    model, cfg = text_api.init_model(cfg, jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1)
    set_attn_backend(model, "xla")
    _give_gdn_memory(
        layer.linear_attn for layer in model.text.layers if layer.layer_type != "full_attention"
    )
    return model, cfg


def _layer_cfg(dtype) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=48,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=("linear_attention",),
        rope_theta=10000,
        partial_rotary_factor=0.25,
        mrope_section=(2, 1, 1),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_value_head_dim=12,
        intermediate_size=64,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=dtype,
    )


def _memory_layer(cfg) -> GatedDeltaNet:
    layer = GatedDeltaNet(cfg, rngs=nnx.Rngs(0))
    _give_gdn_memory([layer])
    return layer


def _logits(model, hidden):
    return np.asarray(jnp.einsum("BTD,DV->BTV", hidden, model.output_weight()))


class ChatMatchesTrainerTest(absltest.TestCase):
    B, C, T = 1, 3, 8

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model, cls.cfg = _smoke_f32_model()
        rng = np.random.default_rng(0)
        tokens = jnp.asarray(
            rng.integers(1, cls.cfg.vocab_size, size=(cls.B, cls.C, cls.T)), jnp.int32
        )
        cls.tokens = tokens
        cls.batch = {
            "token_ids_BCT": tokens,
            "attention_mask_BCT": jnp.ones_like(tokens),
            "reset_state_BC": jnp.asarray([[True, False, False]]),
        }

    def _trainer(self, g, c, **kw):
        hidden, _ = pt._statepassing_hidden_states(
            self.model,
            self.batch,
            self.cfg,
            0,
            pt.PretrainMode.STATEPASSING_BPTT,
            pass_gdn_state=g,
            pass_conv_state=c,
            **kw,
        )
        return _logits(self.model, hidden)

    def _chat(self, g, c, limit=None):
        runtime = ChatRuntime(
            self.model,
            self.cfg,
            mode=ChatMode.TRAINING_RESET,
            state_config=StatePassingConfig(g, limit, c, False),
            segment_length=self.T,
            event_sink=lambda message: None,
        )
        return np.asarray(runtime.consume(self.tokens.reshape(1, -1)))

    def _per_segment(self, a, b):
        return np.abs(a - b).reshape(self.C, self.T, -1).max(axis=(1, 2))

    def _check(self, trainer, chat, wrong):
        np.testing.assert_allclose(chat, trainer, atol=1e-4, rtol=1e-4)
        self.assertTrue(
            np.all(self._per_segment(wrong, trainer)[1:] > 1e-3), self._per_segment(wrong, trainer)
        )

    # Pins: chat TRAINING_RESET with GDN state equals the trainer's scan path; dropping the state changes it.
    def test_scan_gdn_state(self):
        self._check(self._trainer(True, False), self._chat(True, False), self._chat(False, False))

    # Pins: chat with GDN and conv state equals the trainer's scan path; restarting conv changes it.
    def test_scan_gdn_and_conv_state(self):
        self._check(self._trainer(True, True), self._chat(True, True), self._chat(True, False))

    # Pins: without state passing chat and trainer both restart every segment; passing the state changes it.
    def test_scan_no_state(self):
        self._check(self._trainer(False, False), self._chat(False, False), self._chat(True, False))

    # Pins: the unrolled trainer path (bptt_chunks=1, gdn_layer_limit=1) equals chat with the same layer limit.
    def test_unrolled_gdn_layer_limit(self):
        trainer = self._trainer(True, False, bptt_chunks=1, gdn_layer_limit=1)
        self._check(trainer, self._chat(True, False, limit=1), self._chat(True, False))


class LayerHandoverTest(absltest.TestCase):
    CUTS = (1, 3, 70, 149)

    def _run(self, layer, x, gdn=None, conv=None):
        return layer(x, None, gdn, conv, return_final_state=True, return_conv_state=True)

    # Pins: one pass equals two segments with GDN + conv hand-over (outputs, states, dtypes) for non-chunk cuts.
    def test_values_and_dtypes(self):
        rng = np.random.default_rng(0)
        with mesh_rules_for(1, 1, 1):
            for dtype, out_tol in ((jnp.float32, 1e-4), (jnp.bfloat16, 3e-2)):
                cfg = _layer_cfg(dtype)
                layer = _memory_layer(cfg)
                x = jnp.asarray(rng.standard_normal((2, 150, cfg.hidden_size)), dtype=dtype)
                full_out, full_gdn, full_conv = self._run(layer, x)
                for cut in self.CUTS:
                    msg = f"dtype={jnp.dtype(dtype).name} cut={cut}"
                    o1, g1, c1 = self._run(layer, x[:, :cut])
                    o2, g2, c2 = self._run(layer, x[:, cut:], g1, c1)
                    self.assertLess(_rel(jnp.concatenate([o1, o2], axis=1), full_out), out_tol, msg)
                    self.assertLess(_rel(g2, full_gdn), 1e-4, msg)
                    if dtype == jnp.float32:
                        np.testing.assert_allclose(c2, full_conv, atol=1e-6, rtol=1e-6, err_msg=msg)
                    else:
                        np.testing.assert_array_equal(
                            np.asarray(c2, np.float32),
                            np.asarray(full_conv, np.float32),
                            err_msg=msg,
                        )
                    self.assertEqual(g2.dtype, jnp.float32, msg)
                    self.assertEqual(c2.dtype, jnp.dtype(dtype), msg)
                    o2_no_gdn, g2_no_gdn, _ = self._run(layer, x[:, cut:], None, c1)
                    o2_no_conv, _, _ = self._run(layer, x[:, cut:], g1, None)
                    self.assertGreater(_rel(o2_no_gdn, full_out[:, cut:]), 1e-2, msg)
                    if cut >= 70:
                        self.assertGreater(_rel(g2_no_gdn, full_gdn), 1e-2, msg)
                    self.assertGreater(
                        _rel(o2_no_conv[:, :3], full_out[:, cut : cut + 3]), 1e-2, msg
                    )

    # Pins: BPTT across the hand-over: gradients of one pass equal those of two segments (f32).
    def test_gradients(self):
        rng = np.random.default_rng(1)
        with mesh_rules_for(1, 1, 1):
            cfg = _layer_cfg(jnp.float32)
            layer = _memory_layer(cfg)
            x = jnp.asarray(rng.standard_normal((2, 150, cfg.hidden_size)), dtype=jnp.float32)
            graphdef, params, rest = nnx.split(layer, nnx.Param, ...)

            def loss(params, cut, detach):
                m = nnx.merge(graphdef, params, rest)
                if cut is None:
                    return jnp.sum(m(x) ** 2)
                o1, g1, c1 = m(
                    x[:, :cut], None, None, None, return_final_state=True, return_conv_state=True
                )
                if detach:
                    g1, c1 = jax.lax.stop_gradient(g1), jax.lax.stop_gradient(c1)
                o2 = m(x[:, cut:], None, g1, c1)
                return jnp.sum(o1**2) + jnp.sum(o2**2)

            grad = jax.jit(jax.grad(loss), static_argnums=(1, 2))
            full = jax.tree.leaves(grad(params, None, False))
            for cut in self.CUTS:
                split = jax.tree.leaves(grad(params, cut, False))
                for a, b in zip(split, full, strict=True):
                    self.assertLess(_rel(a, b), 1e-3, f"cut={cut}")
            detached = jax.tree.leaves(grad(params, 70, True))
            self.assertGreater(max(_rel(a, b) for a, b in zip(detached, full, strict=True)), 1e-2)


class PaddingNeutralityTest(absltest.TestCase):
    # Pins: right padding (25 valid, 1 valid) leaves outputs and states equal to the run on the valid prefix.
    def test_layer_right_padding(self):
        rng = np.random.default_rng(2)
        with mesh_rules_for(1, 1, 1):
            cfg = _layer_cfg(jnp.float32)
            layer = _memory_layer(cfg)
            B, T = 2, 40
            x = jnp.asarray(rng.standard_normal((B, T, cfg.hidden_size)), jnp.float32)
            _, s, c = layer(x, None, return_final_state=True, return_conv_state=True)
            s0 = jnp.asarray(rng.standard_normal(s.shape), jnp.float32)
            c0 = jnp.asarray(rng.standard_normal(c.shape), jnp.float32)
            for valid in (25, 1):
                mask = jnp.asarray(
                    np.tile(np.r_[np.ones(valid), np.zeros(T - valid)], (B, 1)), jnp.float32
                )
                om, gm, cm = layer(x, mask, s0, c0, return_final_state=True, return_conv_state=True)
                oc, gc, cc = layer(
                    x[:, :valid], None, s0, c0, return_final_state=True, return_conv_state=True
                )
                self.assertLess(_rel(om[:, :valid], oc), 1e-5, valid)
                self.assertLess(_rel(gm, gc), 1e-5, valid)
                self.assertLess(_rel(cm, cc), 1e-5, valid)
                np.testing.assert_array_equal(np.asarray(om[:, valid:]), 0.0)
                self.assertTrue(bool(jnp.isfinite(om).all()))
                _, g_fresh, _ = layer(
                    x[:, :valid], None, return_final_state=True, return_conv_state=True
                )
                self.assertGreater(_rel(g_fresh, gc), 1e-2, valid)

    # Pins: an all-padding segment and a padded T == 1 call leave GDN and conv state bit-identical.
    def test_layer_all_padding_and_single_padded_token(self):
        rng = np.random.default_rng(3)
        with mesh_rules_for(1, 1, 1):
            cfg = _layer_cfg(jnp.float32)
            layer = _memory_layer(cfg)
            B = 2
            x = jnp.asarray(rng.standard_normal((B, 40, cfg.hidden_size)), jnp.float32)
            _, s, c = layer(x, None, return_final_state=True, return_conv_state=True)
            s0 = jnp.asarray(rng.standard_normal(s.shape), jnp.float32)
            c0 = jnp.asarray(rng.standard_normal(c.shape), jnp.float32)
            for T in (40, 1):
                mask = jnp.zeros((B, T), jnp.float32)
                o, g, cs = layer(
                    x[:, :T], mask, s0, c0, return_final_state=True, return_conv_state=True
                )
                np.testing.assert_array_equal(np.asarray(g), np.asarray(s0))
                np.testing.assert_array_equal(np.asarray(cs), np.asarray(c0))
                np.testing.assert_array_equal(np.asarray(o), 0.0)

    # Pins: model level: a right-padded row and an all-padding row in a batch with carried state.
    def test_model_right_padded_and_all_padding_rows(self):
        model, cfg = _smoke_f32_model()
        rng = np.random.default_rng(4)
        B, T, valid = 3, 40, 25
        seg1 = jnp.asarray(rng.integers(1, cfg.vocab_size, size=(B, T)), jnp.int32)
        mask2 = np.ones((B, T), np.int32)
        mask2[1, valid:] = 0
        mask2[2, :] = 0
        seg2 = jnp.asarray(rng.integers(1, cfg.vocab_size, size=(B, T)) * mask2, jnp.int32)
        mask2 = jnp.asarray(mask2)

        def fwd(tokens, mask, gdn=None, conv=None):
            return text_api.forward_with_gdn_state(
                model,
                tokens,
                0,
                cfg,
                attention_mask_BT=mask,
                initial_gdn_states=gdn,
                initial_conv_states=conv,
                return_conv_states=True,
            )

        _, _, g1, c1 = fwd(seg1, jnp.ones_like(seg1))
        h2, _, g2, c2 = fwd(seg2, mask2, g1, c1)
        self.assertTrue(bool(jnp.isfinite(h2).all()))

        def row(states, i):
            return tuple(s[i : i + 1] for s in states)

        hr, _, gr, cr = fwd(
            seg2[1:2, :valid], jnp.ones((1, valid), jnp.int32), row(g1, 1), row(c1, 1)
        )
        np.testing.assert_allclose(_logits(model, h2[1:2, :valid]), _logits(model, hr), atol=1e-4)
        for a, b in zip(row(g2, 1) + row(c2, 1), gr + cr, strict=True):
            self.assertLess(_rel(a, b), 1e-4)
        for a, b in zip(row(g2, 2) + row(c2, 2), row(g1, 2) + row(c1, 2), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        hn, _, _, _ = fwd(seg2[1:2, :valid], jnp.ones((1, valid), jnp.int32))
        self.assertGreater(np.abs(_logits(model, hn) - _logits(model, hr)).max(), 1e-2)


def _naive_delta_rule_f64(q, k, v, g, beta, state=None):
    q, k, v, g, beta = [np.asarray(x, dtype=np.float64) for x in (q, k, v, g, beta)]
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    B, T, H, A = k.shape
    q = q * A**-0.5
    S = np.zeros((B, H, A, v.shape[-1])) if state is None else np.asarray(state, np.float64).copy()
    out = np.zeros((B, T, H, v.shape[-1]))
    for t in range(T):
        S = S * np.exp(g[:, t])[..., None, None]
        delta = (v[:, t] - np.einsum("bhau,bha->bhu", S, k[:, t])) * beta[:, t][..., None]
        S = S + np.einsum("bha,bhu->bhau", k[:, t], delta)
        out[:, t] = np.einsum("bhau,bha->bhu", S, q[:, t])
    return out, S


class ChunkedKernelReferenceTest(absltest.TestCase):
    # Pins: chunked XLA delta rule equals a float64 token recurrence (output and final state, with/without init).
    def test_matches_float64_recurrence(self):
        rng = np.random.default_rng(5)
        B, H, A, U = 2, 3, 16, 8
        for T in (1, 63, 64, 65, 130):
            q, k = (rng.standard_normal((B, T, H, A)).astype(np.float32) for _ in range(2))
            v = rng.standard_normal((B, T, H, U)).astype(np.float32)
            g = -np.exp(rng.uniform(-6, 0.5, size=(B, T, H))).astype(np.float32)
            beta = rng.uniform(0.05, 0.95, size=(B, T, H)).astype(np.float32)
            s0 = rng.standard_normal((B, H, A, U)).astype(np.float32)
            outs = {}
            for name, init in (("zero-init", None), ("given-init", s0)):
                out, st = chunk_gated_delta_rule_xla(
                    *map(jnp.asarray, (q, k, v, g, beta)),
                    initial_state_BHAU=None if init is None else jnp.asarray(init),
                    return_final_state=True,
                )
                ref_out, ref_state = _naive_delta_rule_f64(q, k, v, g, beta, init)
                self.assertLess(_rel(out, ref_out), 1e-5, f"T={T} {name}")
                self.assertLess(_rel(st, ref_state), 1e-5, f"T={T} {name}")
                self.assertEqual(st.dtype, jnp.float32)
                outs[name] = out
            self.assertGreater(_rel(outs["zero-init"], outs["given-init"]), 1e-2, f"T={T}")


_SHARDED_WORKER = r"""
import sys
tp, fsdp, dp = (int(x) for x in sys.argv[1:4])
out = sys.argv[4]
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api

assert jax.device_count() == tp * fsdp * dp, jax.device_count()
cfg = dataclasses.replace(text_api.resolve_config("qwen3.5-smoke-dense"), dtype=jnp.float32)
model, cfg = text_api.init_model(cfg, jax.random.key(0), tp_size=tp, fsdp_size=fsdp, dp_size=dp)
mesh = ensure_mesh()
set_attn_backend(model, "xla")
for layer in model.text.layers:
    if layer.layer_type != "full_attention":
        old = layer.linear_attn.A_log[...]
        layer.linear_attn.A_log[...] = jax.device_put(jnp.full(old.shape, -4.0, old.dtype), old.sharding)
checksum = float(sum(np.abs(np.asarray(v, np.float64)).sum() for v in jax.tree.leaves(nnx.state(model, nnx.Param))))

rng = np.random.default_rng(0)
B, T = 4, 70
tokens = rng.integers(1, cfg.vocab_size, size=(B, 2 * T)).astype(np.int32)
mask = np.ones((B, 2 * T), np.int32)
mask[3, 2 * T - 20:] = 0
tokens = tokens * mask


@nnx.jit
def fwd(model, toks, m, gdn, conv):
    return text_api.forward_with_gdn_state(
        model, toks, 0, cfg, attention_mask_BT=m, initial_gdn_states=gdn, initial_conv_states=conv,
        return_conv_states=True,
    )


with mesh_rules(mesh):
    shard = lambda t, m: text_api.shard_batch_dict({"token_ids_BT": t, "attention_mask_BT": m}, cfg, mesh)
    b1, b2, bf = shard(tokens[:, :T], mask[:, :T]), shard(tokens[:, T:], mask[:, T:]), shard(tokens, mask)
    h1, _, g1, c1 = fwd(model, b1["token_ids_BT"], b1["attention_mask_BT"], None, None)
    h2, _, g2, c2 = fwd(model, b2["token_ids_BT"], b2["attention_mask_BT"], g1, c1)
    h2n, _, _, _ = fwd(model, b2["token_ids_BT"], b2["attention_mask_BT"], None, None)
    _, _, gf, cf = fwd(model, bf["token_ids_BT"], bf["attention_mask_BT"], None, None)
stack = lambda xs: np.stack([np.asarray(x) for x in xs])
np.savez(out, checksum=checksum, h1=np.asarray(h1), h2=np.asarray(h2), h2n=np.asarray(h2n),
         g2=stack(g2), c2=stack(c2), gf=stack(gf), cf=stack(cf))
"""


class ShardedHandoverTest(absltest.TestCase):
    MESHES = ((1, 1, 1), (1, 4, 1), (2, 2, 1), (2, 1, 2))

    # Pins: FSDP / TP sharded two-segment forward with GDN + conv hand-over equals the single-device run.
    def test_sharded_matches_single_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            procs = {}
            for mesh in self.MESHES:
                env = dict(os.environ)
                env.update(
                    JAX_PLATFORMS="cpu",
                    OMEGALAX_DELTANET_KERNEL="xla",
                    PYTHONPATH=str(REPO_ROOT),
                    XLA_FLAGS=f"--xla_force_host_platform_device_count={int(np.prod(mesh))}",
                )
                path = Path(tmp) / ("m_%d%d%d.npz" % mesh)
                procs[mesh] = (
                    path,
                    subprocess.Popen(
                        [sys.executable, "-c", _SHARDED_WORKER, *map(str, mesh), str(path)],
                        env=env,
                        cwd=str(REPO_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    ),
                )
            results = {}
            for mesh, (path, proc) in procs.items():
                output, _ = proc.communicate(timeout=900)
                self.assertEqual(proc.returncode, 0, f"mesh {mesh}:\n{output[-3000:]}")
                results[mesh] = dict(np.load(path))
        ref = results[(1, 1, 1)]
        self.assertGreater(np.abs(ref["h2n"] - ref["h2"]).max(), 1e-2)
        for mesh, res in results.items():
            self.assertEqual(res["checksum"], ref["checksum"], mesh)
            for key in ("h1", "h2", "g2", "c2"):
                np.testing.assert_allclose(
                    res[key], ref[key], atol=1e-4, rtol=0, err_msg=f"{mesh} {key}"
                )
            np.testing.assert_allclose(
                res["g2"], res["gf"], atol=1e-5, rtol=0, err_msg=f"{mesh} gdn"
            )
            np.testing.assert_allclose(
                res["c2"], res["cf"], atol=1e-4, rtol=0, err_msg=f"{mesh} conv"
            )


class GenerationAcrossBoundaryTest(absltest.TestCase):
    # Pins: greedy generation across two segment boundaries equals teacher forcing of the same tokens.
    def test_greedy_equals_teacher_forcing(self):
        model, cfg = _smoke_f32_model()
        state = StatePassingConfig(True, None, False, False)

        def make(state_config):
            return ChatRuntime(
                model,
                cfg,
                mode=ChatMode.TRAINING_RESET,
                state_config=state_config,
                segment_length=4,
                event_sink=lambda message: None,
            )

        prompt = jnp.asarray([[11, 47, 5]], jnp.int32)
        runtime = make(state)
        prompt_logits = runtime.consume(prompt)
        result = generate_token_ids(
            runtime,
            initial_logits_V=prompt_logits[0, -1],
            eos_id=-1,
            max_new_tokens=7,
            key=jax.random.key(0),
            temperature=0.0,
        )
        self.assertLen(result.token_ids, 7)
        self.assertEqual(runtime.completed_segments, 2)
        for layer_cache in runtime.cache:
            if layer_cache is not None:
                self.assertEqual(int(layer_cache.cur_ind[...]), 2)
        full = jnp.concatenate([prompt, jnp.asarray([result.token_ids], jnp.int32)], axis=1)
        teacher = make(state)
        tf_logits = np.asarray(teacher.consume(full))
        self.assertEqual(list(tf_logits[0, 2:9].argmax(-1)), list(result.token_ids))
        for a, b in zip(runtime.gdn_states, teacher.gdn_states, strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)
        no_state = np.asarray(make(StatePassingConfig(False, None, False, False)).consume(full))
        self.assertGreater(np.abs(no_state[0, 4:] - tf_logits[0, 4:]).max(), 1e-2)


class OutputHeadRoundTripTest(absltest.TestCase):
    def _export_and_load(self, tied):
        with mesh_rules_for(1, 1, 1):
            cfg = make_config("qwen3.5-smoke-dense")
            cfg = dataclasses.replace(
                cfg, text_config=dataclasses.replace(cfg.text_config, tie_word_embeddings=tied)
            )
            model = Qwen3_5ForConditionalGeneration(cfg, rngs=nnx.Rngs(params=jax.random.key(0)))
            if not tied:
                old = model.lm_head.kernel[...]
                model.lm_head.kernel[...] = jax.random.normal(
                    jax.random.key(1), old.shape, old.dtype
                )
            with tempfile.TemporaryDirectory() as tmp:
                export_qwen3_5_to_safetensors(model, cfg, tmp)
                loaded, loaded_cfg = create_qwen3_5_from_safetensors(
                    tmp, tp_size=1, fsdp_size=1, dp_size=1, text_only=True
                )
            with tempfile.TemporaryDirectory() as tmp:
                save_hf_config(export_lib.model_config_to_hf_dict(cfg.text_config), tmp)
                resolved = text_api.resolve_config(tmp)
        self.assertEqual(loaded_cfg.text_config.tie_word_embeddings, tied)
        self.assertEqual(resolved.tie_word_embeddings, tied)
        return model, loaded

    # Pins: an untied head survives export + load as the trained lm_head, not embedding.T.
    def test_untied_head(self):
        model, loaded = self._export_and_load(tied=False)
        out_w = np.asarray(loaded.output_weight(), np.float32)
        src_head = np.asarray(
            model.lm_head.kernel[...].astype(loaded.output_weight().dtype), np.float32
        )
        np.testing.assert_array_equal(out_w, src_head)
        emb_T = np.asarray(model.text.embedder.embedding[...], np.float32).T
        self.assertGreater(np.abs(out_w - emb_T).max(), 1e-2)

    # Pins: a tied model loads without lm_head and projects with the source embedding.T.
    def test_tied_head(self):
        model, loaded = self._export_and_load(tied=True)
        self.assertIsNone(loaded.lm_head)
        out_w = loaded.output_weight()
        emb_T = np.asarray(model.text.embedder.embedding[...].astype(out_w.dtype), np.float32).T
        np.testing.assert_array_equal(np.asarray(out_w, np.float32), emb_T)


class _CheckpointFixture(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.base = text_api.resolve_config("qwen3.5-smoke-dense")
        cls.tied = dataclasses.replace(cls.base, tie_word_embeddings=True)

    def _init(self, cfg, seed):
        return text_api.init_model(cfg, jax.random.key(seed), tp_size=1, fsdp_size=1, dp_size=1)


class ResolveCheckpointTest(_CheckpointFixture):
    # Pins: root resolves to the newest complete step; unfinished and tmp directories are never chosen.
    def test_unfinished_directories_and_explicit_steps(self):
        model, cfg = self._init(self.base, 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _save_checkpoint(root, 3, model, cfg)
            _save_checkpoint(root, 7, model, cfg)
            shutil.copytree(root / "000007", root / "000009.orbax-checkpoint-tmp-1")
            shutil.copytree(root / "000007", root / "000011")
            (root / "000011" / "_CHECKPOINT_METADATA").unlink()
            self.assertEqual(resolve_checkpoint(root).step, 7)
            self.assertEqual(resolve_checkpoint(root / "000003").step, 3)
            self.assertEqual(
                resolve_checkpoint(root / "000007").step_path, root.resolve() / "000007"
            )
            with self.assertRaisesRegex(ValueError, "complete checkpoint"):
                resolve_checkpoint(root / "000011")


class RestoreTreeMismatchTest(_CheckpointFixture):
    def _restore(self, source, source_cfg, target):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _save_checkpoint(root, 5, source, source_cfg)
            checkpoint = resolve_checkpoint(root)
            with self.assertRaises(Exception):
                restore_model_params(target, checkpoint)

    # Pins: a tied checkpoint (no lm_head) restored into an untied model must raise at restore time.
    # Defect: restore returns and leaves lm_head.kernel as a ShapeDtypeStruct placeholder that fails only at first use.
    @unittest.expectedFailure
    def test_tied_checkpoint_into_untied_model_raises(self):
        source, source_cfg = self._init(self.tied, 0)
        target, _ = self._init(self.base, 1)
        self._restore(source, source_cfg, target)

    # Pins: an untied checkpoint restored into a tied model must raise at restore time.
    # Defect: restore succeeds silently and the trained lm_head in the checkpoint is ignored.
    @unittest.expectedFailure
    def test_untied_checkpoint_into_tied_model_raises(self):
        source, source_cfg = self._init(self.base, 0)
        target, _ = self._init(self.tied, 1)
        self._restore(source, source_cfg, target)

    # Pins: a LoRA checkpoint without lora_metadata.json restored into a plain model must raise at restore time.
    # Defect: restore returns and leaves the wrapped base kernels as placeholders that fail only at first use.
    @unittest.expectedFailure
    def test_lora_checkpoint_without_metadata_raises(self):
        source, source_cfg = self._init(self.base, 0)
        with mesh_rules(ensure_mesh()):
            inject_lora(source, r=4, alpha=8.0, rngs=nnx.Rngs(0))
        target, _ = self._init(self.base, 1)
        self._restore(source, source_cfg, target)


if __name__ == "__main__":
    absltest.main()
