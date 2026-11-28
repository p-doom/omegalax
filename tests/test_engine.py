"""Rollout engine against the trainer forward and HF transformers (GPU, real weights)."""

import collections
import dataclasses
import logging
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from omegalax.distributed.mesh import mesh_rules
from omegalax.engine import Engine, EngineConfig, Request
from omegalax.text import api as text_api
from omegalax.models.qwen3.loader import create_qwen3_from_safetensors
from omegalax.models.qwen3_5.loader import create_qwen3_5_from_safetensors
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
from omegalax.models.sharding_runtime import _finalize_q_shardings
from tests.real_weights import requires_real_weights

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome, with tests.",
    "Explain how a B-tree insertion splits nodes.",
    "What is the derivative of x^3 * sin(x)? Show the steps.",
    "List three differences between TCP and UDP.",
]
# Fixed before optimizing the engine; see the noise floor of trainer attention backends.
LOGP_MEAN, LOGP_P99, LOGP_P999 = 0.03, 0.30, 0.70
NEAR_TIE_MARGIN = 1.0


class _CompileLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.compiles = []

    def emit(self, record):
        if "Compiling" in record.getMessage():
            self.compiles.append(record.getMessage()[:300])


def _keys(rng, n):
    return rng.integers(0, 2**32, size=(n, 2), dtype=np.uint32)


def _qwen3_5_causal_lm(path, tp):
    vlm, cfg = create_qwen3_5_from_safetensors(path, tp_size=tp, fsdp_size=1, dp_size=1)
    mesh = jax.tree.leaves(nnx.state(vlm))[0].sharding.mesh
    text_cfg = text_api.align_config_to_mesh(cfg.text_config, mesh)
    with mesh_rules(mesh):
        abstract = nnx.eval_shape(lambda: Qwen3_5ForCausalLM(text_cfg, rngs=nnx.Rngs(0)))
    state = nnx.state(abstract)
    params = nnx.to_pure_dict(nnx.state(vlm))
    del params["vision"]
    nnx.replace_by_pure_dict(state, params)
    model = nnx.merge(nnx.graphdef(abstract), state)
    _finalize_q_shardings(model, mesh)
    return model


def _smoke_model():
    """Random-init Qwen3 smoke model; the extend kernel needs head_dim % 64 == 0."""
    cfg = dataclasses.replace(text_api.resolve_config("qwen3-smoke"), head_dim=64)
    return text_api.init_model(cfg, jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1)[0]


def _engine(model, cfg, mesh=None):
    """Builds the engine on a fresh thread, under ``mesh`` or none: the loaders leave their mesh
    set on the calling thread, which would hide any dependence of the engine on it."""

    def build():
        if mesh is not None:
            with jax.set_mesh(mesh):
                return Engine(model, cfg)
        assert jax.sharding.get_abstract_mesh().empty
        return Engine(model, cfg)

    with ThreadPoolExecutor(1) as pool:
        return pool.submit(build).result()


class _EngineTest:
    """Runs with tensor parallelism over all visible devices; each class covers one TP size."""

    MODEL_ID: str
    TP = 1

    @classmethod
    def setUpClass(cls):
        if jax.device_count() != cls.TP:
            raise unittest.SkipTest(f"needs {cls.TP} visible devices")
        cls.path = snapshot_download(cls.MODEL_ID)
        cls.tok = AutoTokenizer.from_pretrained(cls.path)
        cls.model = cls.load(cls.path, cls.TP)
        eos = cls.tok.convert_tokens_to_ids("<|im_end|>")
        pad = cls.tok.convert_tokens_to_ids("<|endoftext|>")
        cls.cfg = EngineConfig(
            num_pages=48,
            max_context=4096,
            decode_steps=8,
            decode_buckets=(1, 2, 4),
            extend_buckets=(256, 512, 1024),
            eos_token_ids=(eos, pad),
            pad_id=pad,
            max_group=4,
            max_sessions=16,
        )
        cls.engine = _engine(cls.model, cls.cfg)
        cls.engine.start()

    @classmethod
    def tearDownClass(cls):
        cls.engine.stop()

    def _chat(self, text):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=True,
        )["input_ids"]

    def _rollouts(self, rng, turns):
        """A group of 4 sessions sharing a prompt, then tool-style continuations."""
        prompt = self._chat(PROMPTS[0])
        futures = self.engine.generate(
            Request(prompt, _keys(rng, 4), [97, 130, 64, 160], [1.0] * 4, stop_at_eos=False)
        )
        seqs, scored = {}, {}
        for f in futures:
            c = f.result()
            seqs[c.session] = prompt + c.tokens
            scored[c.session] = [(len(prompt) + i, lp) for i, lp in enumerate(c.logprobs)]
        for turn in range(turns):
            tool = self.tok(
                f"<|im_end|>\n<|im_start|>user\n<tool_response>\nturn {turn}: "
                + "ok " * int(rng.integers(20, 300))
                + "</tool_response><|im_end|>\n<|im_start|>assistant\n"
            )["input_ids"]
            futures = []
            for sid in seqs:
                seqs[sid] += tool
                futures += self.engine.generate(
                    Request(tool, _keys(rng, 1), [int(rng.integers(1, 200))], [1.0], session=sid)
                )
            for f in futures:
                c = f.result()
                base = len(seqs[c.session])
                seqs[c.session] += c.tokens
                scored[c.session] += [(base + i, lp) for i, lp in enumerate(c.logprobs)]
        return seqs, scored

    def _logp_errors(self, seqs, scored):
        ref = self.engine.score(list(seqs.values())).result()
        errs = []
        for (sid, _), ref_T in zip(seqs.items(), ref):
            errs += [abs(lp - ref_T[pos - 1]) for pos, lp in scored[sid]]
        return np.asarray(errs)

    def _assert_logp_close(self, errs):
        mean, p99, p999 = errs.mean(), np.quantile(errs, 0.99), np.quantile(errs, 0.999)
        print(
            f"logprob |engine - trainer| over {errs.size} tokens: "
            f"mean {mean:.4f} p99 {p99:.4f} p99.9 {p999:.4f} max {errs.max():.4f}"
        )
        self.assertLess(mean, LOGP_MEAN)
        self.assertLess(p99, LOGP_P99)
        self.assertLess(p999, LOGP_P999)

    def test_logprobs_match_trainer_forward(self):
        self.engine.stats.clear()
        seqs, scored = self._rollouts(np.random.default_rng(0), turns=3)
        print(dict(self.engine.stats))
        self.assertGreater(self.engine.stats["evict_idle"] + self.engine.stats["evict_waiting"], 0)
        self._assert_logp_close(self._logp_errors(seqs, scored))
        for sid in seqs:
            self.engine.release(sid)

    def test_forked_groups_share_extends(self):
        """Forked groups submitted while the engine is busy share extends; together they carry
        more forks than one extend may and more decoding sessions than the largest bucket."""
        rng = np.random.default_rng(3)
        n = self.cfg.max_group - 1
        prompts = [self._chat(text) for text in PROMPTS[:3]]
        self.engine.stats.clear()
        busy = self.engine.generate(
            Request(self._chat(PROMPTS[3]), _keys(rng, 1), [200], [1.0], stop_at_eos=False)
        )
        groups = [
            self.engine.generate(Request(p, _keys(rng, n), [96] * n, [1.0] * n, stop_at_eos=False))
            for p in prompts
        ]
        seqs, scored = {}, {}
        for prompt, futures in zip([self._chat(PROMPTS[3]), *prompts], [busy, *groups]):
            for f in futures:
                c = f.result()
                seqs[c.session] = prompt + c.tokens
                scored[c.session] = [(len(prompt) + i, lp) for i, lp in enumerate(c.logprobs)]
        print(dict(self.engine.stats))
        self.assertLess(self.engine.stats["extend_calls"], 1 + len(groups))
        self._assert_logp_close(self._logp_errors(seqs, scored))
        for sid in seqs:
            self.engine.release(sid)

    def test_update_weights_no_recompile(self):
        old_params = nnx.to_pure_dict(nnx.state(self.model))

        def perturb(x):
            noise = jax.random.normal(jax.random.key(0), x.shape, jnp.float32) * 0.05 * jnp.std(x)
            # Replicated, so that under TP the engine has to reshard what it receives.
            return jax.device_put((x + noise).astype(x.dtype), NamedSharding(x.sharding.mesh, P()))

        new_params = jax.tree.map(perturb, old_params)
        jax.block_until_ready(new_params)
        # A trainer forward on a new thread, as in RL: each thread's first tokamax op adds a
        # context to JAX's jit cache key.
        mesh = jax.tree.leaves(old_params)[0].sharding.mesh
        tokens = jnp.zeros((1, 128), jnp.int32)

        def trainer_forward():
            with jax.set_mesh(mesh):
                nnx.eval_shape(lambda: self.model(tokens, jnp.ones_like(tokens), None, None))

        with ThreadPoolExecutor(1) as pool:
            pool.submit(trainer_forward).result()
        rng = np.random.default_rng(2)
        counter = _CompileLog()
        logging.getLogger("jax").addHandler(counter)
        jax.config.update("jax_log_compiles", True)
        try:
            self.engine.update_weights(new_params, version=7).result()
            # Prompts of 100, 300 and 700 tokens extend in buckets 256, 512 and 1024; n forks
            # decode in bucket n.
            for n, length in zip(self.cfg.decode_buckets, (100, 300, 700)):
                prompt = rng.integers(10, 50000, length).tolist()
                futures = self.engine.generate(
                    Request(prompt, _keys(rng, n), [9] * n, [1.0] * n, stop_at_eos=False)
                )
                for f in futures:
                    self.engine.release(f.result().session)
            seqs, scored = self._rollouts(np.random.default_rng(1), turns=1)
        finally:
            jax.config.update("jax_log_compiles", False)
            logging.getLogger("jax").removeHandler(counter)
        self.assertEqual(counter.compiles, [])
        new_errs = self._logp_errors(seqs, scored)
        self._assert_logp_close(new_errs)
        self.engine.update_weights(old_params, version=0).result()
        old_errs = self._logp_errors(seqs, scored)
        print(f"same tokens under the old weights: mean |dlogp| {old_errs.mean():.3f}")
        self.assertGreater(old_errs.mean(), 5 * new_errs.mean())
        for sid in seqs:
            self.engine.release(sid)

    def test_greedy_matches_hf_until_near_tie(self):
        hf = self.HF_CLASS.from_pretrained(self.path, dtype=torch.float32).cuda().eval()
        rng = np.random.default_rng(2)
        for text in PROMPTS:
            prompt = self._chat(text)
            with torch.no_grad():
                out = hf.generate(
                    torch.tensor([prompt]).cuda(),
                    max_new_tokens=96,
                    do_sample=False,
                    output_logits=True,
                    return_dict_in_generate=True,
                )
            hf_tokens = out.sequences[0, len(prompt) :].tolist()
            top2 = torch.stack(out.logits)[:, 0].topk(2, dim=-1).values
            margins = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            near_tie = next((t for t, m in enumerate(margins) if m < NEAR_TIE_MARGIN), len(margins))
            c = self.engine.generate(Request(prompt, _keys(rng, 1), [96], [1e-4]))[0].result()
            agree = next(
                (t for t, (a, b) in enumerate(zip(c.tokens, hf_tokens)) if a != b),
                min(len(c.tokens), len(hf_tokens)),
            )
            print(f"first near-tie {near_tie}, tokens agree for {agree} of {len(hf_tokens)}")
            self.assertEqual(c.tokens[:near_tie], hf_tokens[:near_tie])
            self.engine.release(c.session)
        del hf
        torch.cuda.empty_cache()


class AmbientMeshTest(absltest.TestCase):
    """The engine depends on no mesh context: none set, or another one (the trainer's)."""

    def test_construction(self):
        if jax.device_count() != 1:
            self.skipTest("single-device test")
        foreign = jax.make_mesh((1,), ("x",))
        for name in ("qwen3-smoke", "qwen3.5-smoke-dense"):
            model, _ = text_api.init_model(
                name, jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1
            )
            cfg = EngineConfig(
                num_pages=8,
                max_context=1024,
                decode_steps=8,
                decode_buckets=(1,),
                extend_buckets=(256,),
                eos_token_ids=(0,),
                pad_id=1,
            )
            for mesh in (None, foreign):
                with self.subTest(model=name, mesh=mesh):
                    _engine(model, cfg, mesh)


class EvictionTest(absltest.TestCase):
    """Random-init model, page pool far below the working set: forks, tail evictions of idle
    and waiting sessions and re-extends must leave every logprob consistent with the trainer."""

    def test_logprobs_under_eviction(self):
        if jax.device_count() != 1:
            self.skipTest("single-device test")
        model = _smoke_model()
        cfg = EngineConfig(
            num_pages=40,
            max_context=4096,
            decode_steps=8,
            decode_buckets=(4, 8, 16),
            extend_buckets=(256, 512, 1024),
            eos_token_ids=(5,),
            pad_id=model.lm_head.kernel.shape[1],
            max_group=4,
            max_sessions=32,
        )
        engine = _engine(model, cfg)
        engine.start()

        def rollout(seed):
            rng = np.random.default_rng(seed)
            prompt = rng.integers(10, 1000, size=int(rng.integers(30, 200))).tolist()
            max_new = [int(x) for x in rng.integers(1, 80, 4)]
            futures = engine.generate(
                Request(prompt, _keys(rng, 4), max_new, [1.0] * 4, stop_at_eos=False)
            )
            out = []
            for f in futures:
                c = f.result()
                seq, scored = (
                    prompt + c.tokens,
                    [(len(prompt) + i, lp) for i, lp in enumerate(c.logprobs)],
                )
                for _ in range(6):
                    tool = rng.integers(10, 1000, size=int(rng.integers(5, 120))).tolist()
                    seq += tool
                    req = Request(
                        tool,
                        _keys(rng, 1),
                        [int(rng.integers(1, 80))],
                        [1.0],
                        session=c.session,
                        stop_at_eos=False,
                    )
                    c = engine.generate(req)[0].result()
                    scored += [(len(seq) + i, lp) for i, lp in enumerate(c.logprobs)]
                    seq += c.tokens
                out.append((seq, scored))
            return out

        with ThreadPoolExecutor(4) as pool:
            results = [r for rs in pool.map(rollout, range(4)) for r in rs]
        stats = dict(engine.stats)
        ref = engine.score([seq for seq, _ in results]).result()
        engine.stop()
        errs = np.asarray(
            [abs(lp - r[pos - 1]) for (_, scored), r in zip(results, ref) for pos, lp in scored]
        )
        print(stats, f"mean {errs.mean():.4f} max {errs.max():.4f} over {errs.size} tokens")
        self.assertGreater(stats["evict_idle"], 0)
        self.assertLess(errs.mean(), LOGP_MEAN)
        self.assertLess(np.quantile(errs, 0.999), LOGP_P999)


class KVPressureTest(absltest.TestCase):
    """Random-init model, working set ~6x the page pool, near-zero tool waits: admitted turns
    keep their pages, so decodes run side by side instead of evicting each other's tails."""

    def test_decode_concurrency_under_kv_pressure(self):
        if jax.device_count() != 1:
            self.skipTest("single-device test")
        model = _smoke_model()
        cfg = EngineConfig(
            num_pages=128,
            max_context=8192,
            decode_steps=8,
            decode_buckets=(4, 8, 16),
            extend_buckets=(256, 512, 1024),
            eos_token_ids=(5,),
            pad_id=model.lm_head.kernel.shape[1],
            max_group=4,
            max_sessions=32,
        )
        engine = _engine(model, cfg)
        engine.start()
        turn_contexts = []

        def rollout(seed):
            rng = np.random.default_rng(seed)
            new, session, length = rng.integers(10, 1000, size=200).tolist(), None, 0
            for _ in range(9):
                length += len(new)
                turn_contexts.append(length)
                max_new = [int(rng.integers(100, 300))]
                req = Request(new, _keys(rng, 1), max_new, [1.0], session, stop_at_eos=False)
                c = engine.generate(req)[0].result()
                session, length = c.session, length + len(c.tokens)
                new = rng.integers(10, 1000, size=int(rng.integers(100, 200))).tolist()

        with ThreadPoolExecutor(16) as pool:
            list(pool.map(rollout, range(16)))
        stats = dict(engine.stats)
        engine.stop()
        concurrency = stats["decode_tokens"] / (stats["decode_calls"] * cfg.decode_steps)
        print(stats, f"decode concurrency {concurrency:.2f}")
        # Decodes evicting the tails of sessions mid-extend held this at 1.3.
        self.assertGreater(concurrency, 2.5)
        # An admitted turn is extended at most once: its context, padded per extend item.
        self.assertGreater(stats["evict_idle"] + stats["evict_waiting"], 0)
        self.assertLessEqual(stats["extend_tokens"], sum(turn_contexts) + 128 * len(turn_contexts))


class OverlapTest(absltest.TestCase):
    """Decode call n + 1 is queued before call n's results are read. A turn that samples EOS
    runs at most one more call, masked; aborts and continuations right after a completion reuse
    pages and slots while that call is in flight, and every logprob stays consistent with the
    trainer."""

    def test_eos_turns_and_aborts(self):
        if jax.device_count() != 1:
            self.skipTest("single-device test")
        model = _smoke_model()
        vocab = model.lm_head.kernel.shape[1]
        cfg = EngineConfig(
            num_pages=64,
            max_context=4096,
            decode_steps=8,
            decode_buckets=(4, 8, 16),
            extend_buckets=(256, 512, 1024),
            eos_token_ids=tuple(range(10, vocab, 16)),  # turns end by EOS after ~16 tokens
            pad_id=vocab,
            max_group=4,
            max_sessions=32,
        )
        engine = _engine(model, cfg)
        engine.start()

        def rollout(seed):
            rng = np.random.default_rng(seed)
            seq = rng.integers(10, 1000, size=int(rng.integers(30, 200))).tolist()
            scored, reasons, session, new = [], [], None, seq
            for turn in range(6):
                req = Request(new, _keys(rng, 1), [200], [1.0], session)
                c = engine.generate(req)[0].result()
                reasons.append(c.finish_reason)
                scored += [(len(seq) + i, lp) for i, lp in enumerate(c.logprobs)]
                seq, session = seq + c.tokens, c.session
                if turn % 2:
                    # A long continuation, aborted while its decode calls are in flight; the
                    # session's pages and slot go back to the pool.
                    tool = rng.integers(10, 1000, size=20).tolist()
                    long = Request(tool, _keys(rng, 1), [1000], [1.0], session, stop_at_eos=False)
                    future = engine.generate(long)[0]
                    time.sleep(0.05)
                    engine.abort([session])
                    reasons.append(future.result().finish_reason)
                    break
                new = rng.integers(10, 1000, size=int(rng.integers(5, 120))).tolist()
                seq += new
            return seq, scored, reasons

        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(rollout, range(8)))
        stats = dict(engine.stats)
        ref = engine.score([seq for seq, _, _ in results]).result()
        engine.stop()
        reasons = collections.Counter(r for _, _, rs in results for r in rs)
        errs = np.asarray(
            [abs(lp - r[pos - 1]) for (_, scored, _), r in zip(results, ref) for pos, lp in scored]
        )
        print(stats, dict(reasons), f"mean {errs.mean():.4f} over {errs.size} tokens")
        self.assertGreater(reasons["stop"], 0)
        self.assertGreater(stats["decode_masked_rows"], 0)
        # An EOS turn runs at most one masked call; an aborted turn at most two (both in flight).
        self.assertLessEqual(stats["decode_masked_rows"], reasons["stop"] + 2 * reasons["abort"])
        self.assertLess(errs.mean(), LOGP_MEAN)
        self.assertLess(np.quantile(errs, 0.999), LOGP_P999)


class SettlementTest(absltest.TestCase):
    """Every future of a request settles exactly once: the forks' futures of a rejected group,
    turns in flight at stop, and commands still queued when the engine thread fails."""

    def setUp(self):
        if jax.device_count() != 1:
            self.skipTest("single-device test")
        self.model = _smoke_model()
        self.cfg = EngineConfig(
            num_pages=64,
            max_context=2048,
            decode_steps=8,
            decode_buckets=(4,),
            extend_buckets=(256,),
            eos_token_ids=(5,),
            pad_id=self.model.lm_head.kernel.shape[1],
            max_group=4,
            max_sessions=4,
        )
        self.engine = _engine(self.model, self.cfg)
        self.engine.start()
        self.rng = np.random.default_rng(0)
        self.prompt = self.rng.integers(10, 1000, 100).tolist()

    def _group(self, max_new):
        keys = _keys(self.rng, 4)
        return self.engine.generate(Request(self.prompt, keys, [max_new] * 4, [1.0] * 4))

    def _failed(self, futures, kind):
        for f in futures:
            self.assertIsInstance(f.exception(timeout=60), kind)

    def test_rejected_group_fails_every_future(self):
        first = self._group(16)
        sessions = [f.result(timeout=60).session for f in first]
        self._failed(self._group(self.cfg.max_context), ValueError)
        # The first group still holds every session slot.
        self._failed(self._group(16), ValueError)
        for sid in sessions:
            self.engine.release(sid)
        self.engine.stop()

    def test_stop_and_failure_settle_pending_futures(self):
        in_flight = self._group(1000)
        self.engine.stop()
        self._failed(in_flight, RuntimeError)
        with self.assertRaises(RuntimeError):
            self._group(16)
        self.engine = _engine(self.model, self.cfg)
        self.engine.start()
        params = nnx.to_pure_dict(nnx.state(self.model))
        params["lm_head"]["kernel"] = jnp.zeros((1, 1))
        # Compiling the first score of this shape keeps the engine thread busy while the rest
        # queue behind the update that kills it.
        busy = self.engine.score([self.prompt * 8] * 16)
        bad = self.engine.update_weights(params, version=1)
        queued = [self.engine.score([self.prompt]), *self._group(16)]
        self.assertEqual(len(busy.result(timeout=120)), 16)
        self._failed([bad], ValueError)
        self._failed(queued, ValueError)


@requires_real_weights
class Qwen3EngineTest(_EngineTest, absltest.TestCase):
    MODEL_ID = "Qwen/Qwen3-1.7B"
    HF_CLASS = AutoModelForCausalLM

    @staticmethod
    def load(path, tp):
        return create_qwen3_from_safetensors(path, tp_size=tp, fsdp_size=1, dp_size=1)


class Qwen3TP2EngineTest(Qwen3EngineTest):
    TP = 2


class Qwen3TP4EngineTest(Qwen3EngineTest):
    TP = 4


class Qwen3TP8EngineTest(Qwen3EngineTest):
    TP = 8


@requires_real_weights
class Qwen3_5EngineTest(_EngineTest, absltest.TestCase):
    MODEL_ID = "Qwen/Qwen3.5-0.8B"
    HF_CLASS = Qwen3_5ForConditionalGeneration
    load = staticmethod(_qwen3_5_causal_lm)


class Qwen3_5TP2EngineTest(Qwen3_5EngineTest):
    TP = 2


class Qwen3_5TP4EngineTest(Qwen3_5EngineTest):
    MODEL_ID = "Qwen/Qwen3.5-4B"  # 0.8B has 2 KV heads.
    TP = 4


if __name__ == "__main__":
    absltest.main()
