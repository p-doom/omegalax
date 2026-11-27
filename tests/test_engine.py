"""Rollout engine against the trainer forward and HF transformers (GPU, real weights)."""

import logging
import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl.testing import absltest
from flax import nnx
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from omegalax.distributed.mesh import mesh_rules
from omegalax.engine import Engine, EngineConfig, Request
from omegalax.text import api as text_api
from omegalax.models.qwen3.loader import create_qwen3_from_safetensors
from omegalax.models.qwen3_5.loader import create_qwen3_5_from_safetensors
from omegalax.models.qwen3_5.model import Qwen3_5ForCausalLM
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


def _qwen3_5_causal_lm(path):
    vlm, cfg = create_qwen3_5_from_safetensors(path, tp_size=1, fsdp_size=1, dp_size=1)
    mesh = jax.tree.leaves(nnx.state(vlm))[0].sharding.mesh
    text_cfg = text_api.align_config_to_mesh(cfg.text_config, mesh)
    with mesh_rules(mesh):
        abstract = nnx.eval_shape(lambda: Qwen3_5ForCausalLM(text_cfg, rngs=nnx.Rngs(0)))
    state = nnx.state(abstract)
    nnx.replace_by_pure_dict(
        state,
        {"text": nnx.to_pure_dict(nnx.state(vlm.text)), "lm_head": nnx.to_pure_dict(nnx.state(vlm.lm_head))},
    )
    return nnx.merge(nnx.graphdef(abstract), state)


class _EngineTest:
    MODEL_ID: str

    @classmethod
    def setUpClass(cls):
        cls.path = snapshot_download(cls.MODEL_ID)
        cls.tok = AutoTokenizer.from_pretrained(cls.path)
        cls.model = cls.load(cls.path)
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
        cls.engine = Engine(cls.model, cls.cfg)
        cls.engine.warmup()
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
        print(f"logprob |engine - trainer| over {errs.size} tokens: "
              f"mean {mean:.4f} p99 {p99:.4f} p99.9 {p999:.4f} max {errs.max():.4f}")
        self.assertLess(mean, LOGP_MEAN)
        self.assertLess(p99, LOGP_P99)
        self.assertLess(p999, LOGP_P999)

    def test_logprobs_match_trainer_forward(self):
        self.engine.stats.clear()
        seqs, scored = self._rollouts(np.random.default_rng(0), turns=3)
        print(dict(self.engine.stats))
        self.assertGreater(self.engine.stats["evict_idle"] + self.engine.stats["evict_running"], 0)
        self._assert_logp_close(self._logp_errors(seqs, scored))
        for sid in seqs:
            self.engine.release(sid)

    def test_update_weights_no_recompile(self):
        old_state = nnx.state(self.model)

        def perturb(x):
            noise = jax.random.normal(jax.random.key(0), x.shape, jnp.float32) * 0.05 * jnp.std(x)
            return jax.device_put((x + noise).astype(x.dtype), x.sharding)

        new_state = jax.tree.map(perturb, old_state)
        jax.block_until_ready(new_state)
        counter = _CompileLog()
        logging.getLogger("jax").addHandler(counter)
        jax.config.update("jax_log_compiles", True)
        try:
            self.engine.update_weights(new_state, version=7).result()
            seqs, scored = self._rollouts(np.random.default_rng(1), turns=1)
        finally:
            jax.config.update("jax_log_compiles", False)
            logging.getLogger("jax").removeHandler(counter)
        self.assertEqual(counter.compiles, [])
        new_errs = self._logp_errors(seqs, scored)
        self._assert_logp_close(new_errs)
        self.engine.update_weights(old_state, version=0).result()
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


@requires_real_weights
class Qwen3EngineTest(_EngineTest, absltest.TestCase):
    MODEL_ID = "Qwen/Qwen3-1.7B"
    HF_CLASS = AutoModelForCausalLM

    @staticmethod
    def load(path):
        return create_qwen3_from_safetensors(path, tp_size=1, fsdp_size=1, dp_size=1)


@requires_real_weights
class Qwen3_5EngineTest(_EngineTest, absltest.TestCase):
    MODEL_ID = "Qwen/Qwen3.5-0.8B"
    HF_CLASS = Qwen3_5ForConditionalGeneration
    load = staticmethod(_qwen3_5_causal_lm)


if __name__ == "__main__":
    absltest.main()
