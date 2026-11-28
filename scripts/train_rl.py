"""Single-controller asynchronous GRPO in one process.

The trainer owns a device sub-mesh; the remaining devices run rollout engines (TP each). An asyncio
controller keeps at most ``(rollout_ahead + 1) * groups_per_step`` prompt groups outstanding,
the trainer thread consumes completed groups, takes one optimizer step per batch and pushes the
new weights to every engine, which apply them at their next decode-call boundary.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from omegalax.engine import Engine, EngineConfig
from omegalax.rl import loop
from omegalax.rl.envs import MathEnv, PythonToolEnv
from omegalax.rl.rollout import RolloutConfig
from omegalax.rl.trainer import TrainConfig, Trainer
from omegalax.rl.weights import WeightSync, engine_replica, load_causal_lm, sub_mesh

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", None, "HF safetensors directory.", required=True)
flags.DEFINE_string("data_path", None, "JSONL of {prompt: messages, label}.", required=True)
flags.DEFINE_string("out_dir", None, "Run directory (metrics.jsonl, config.json).", required=True)
flags.DEFINE_integer("trainer_devices", None, "Leading devices forming the trainer's FSDP mesh.", required=True)
flags.DEFINE_integer("engine_devices", None, "Following devices, split into engines of engine_tp each.", required=True)
flags.DEFINE_integer("engine_tp", 1, "Tensor-parallel devices per engine.")
flags.DEFINE_integer("num_steps", None, "Optimizer steps.", required=True)
flags.DEFINE_integer("groups_per_step", 32, "Prompt groups per optimizer step.")
flags.DEFINE_integer("group_size", 8, "Samples per prompt.")
flags.DEFINE_integer("max_response", 8192, "Tokens after the prompt (generated and observed).")
flags.DEFINE_float("temperature", 1.0, "Sampling temperature (no top-p/top-k).")
flags.DEFINE_integer("rollout_ahead", 1, "Batches generated beyond the one being trained.")
flags.DEFINE_integer("max_staleness", 4, "Drop groups with a token sampled > this many versions ago.")
flags.DEFINE_float("lr", 1e-6, "Constant learning rate.")
flags.DEFINE_float("weight_decay", 0.1, "AdamW weight decay (matrices only).")
flags.DEFINE_float("adam_b1", 0.9, "Adam beta1.")
flags.DEFINE_float("adam_b2", 0.98, "Adam beta2.")
flags.DEFINE_float("adam_eps", 1e-8, "Adam epsilon.")
flags.DEFINE_float("max_grad_norm", 1.0, "Global gradient-norm clip.")
flags.DEFINE_float("eps_low", 0.2, "PPO clip below 1.")
flags.DEFINE_float("eps_high", 0.28, "PPO clip above 1.")
flags.DEFINE_integer("tokens_per_microbatch", 32768, "Tokens per trainer microbatch (rows x row_tokens).")
flags.DEFINE_integer("row_tokens", 8192, "Trainer row length; samples are packed into rows.")
flags.DEFINE_integer("logit_tile_tokens", 4096, "Positions per float32 logits tile in the trainer.")
flags.DEFINE_enum("compute_dtype", "bfloat16", ["bfloat16", "float16"], "Trainer and engine compute dtype.")
flags.DEFINE_bool("host_optimizer", False, "Keep the float32 master and Adam state in host memory (large models).")
flags.DEFINE_enum("objective", "ppo", ["ppo", "pg_sc", "ppo_sc"], "PPO clip on p/q; plain PG + score centering; PPO + score centering.")
flags.DEFINE_integer("drift_every", 0, "Steps between measurements of the score-centering drift gradient (0: never).")
flags.DEFINE_integer("topk", 1, "Sampler top-k candidates recorded per token (score centering head).")
flags.DEFINE_integer("num_pages", 32768, "KV pages per engine (64 tokens each).")
flags.DEFINE_integer("max_sessions", 256, "Concurrent sessions per engine.")
flags.DEFINE_integer("decode_steps", 8, "Decode steps per engine device call.")
flags.DEFINE_list("decode_buckets", ["16", "32", "64", "128", "256"], "Engine decode batch buckets.")
flags.DEFINE_list("extend_buckets", ["512", "1024", "2048", "4096", "8192"], "Engine extend token buckets.")
flags.DEFINE_enum("env", "math", ["math", "python_tool"], "Single-turn math, or math with a Python tool.")
flags.DEFINE_bool("thinking", None, "Thinking mode (deepscaler grading) or not (last \\boxed).", required=True)
flags.DEFINE_integer("max_turns", 16, "python_tool: policy turns per episode.")
flags.DEFINE_float("tool_timeout_s", 30.0, "python_tool: execution timeout (reported to the policy).")
flags.DEFINE_integer("tool_memory_gb", 4, "python_tool: address-space limit per execution.")
flags.DEFINE_integer("tool_output_chars", 4000, "python_tool: output kept per execution.")
flags.DEFINE_integer("max_tool_procs", 64, "python_tool: concurrent executions.")
flags.DEFINE_integer("grade_workers", 16, "Reward grading processes.")
flags.DEFINE_float("grade_timeout_s", 30.0, "Grading timeout; a timed-out sample is missing.")
flags.DEFINE_integer("seed", 0, "Seed for prompt order and sampling keys.")
flags.DEFINE_string("eval_path", None, "JSONL evaluated with one sample per row.", required=True)
flags.DEFINE_integer("eval_every", 10, "Steps between evaluations (also one before step 1).")
flags.DEFINE_integer("eval_samples", 1, "Samples per evaluation task.")
flags.DEFINE_integer("eval_concurrency", 64, "Evaluation samples in flight.")
flags.DEFINE_integer("dump_every", 50, "Steps between dumps of the trained samples (samples/step_*.jsonl).")
flags.DEFINE_integer("save_every", 0, "Steps between policy snapshots in inference dtypes (checkpoints/step_*.npz; 0: never).")


def main(argv):
    del argv
    out = Path(FLAGS.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(FLAGS.flag_values_dict(), indent=1, default=str))
    devices = jax.devices()
    nt, tp = FLAGS.trainer_devices, FLAGS.engine_tp
    if nt + FLAGS.engine_devices > len(devices) or FLAGS.engine_devices % tp:
        raise ValueError(f"{nt} trainer + {FLAGS.engine_devices} engine devices (tp={tp}) on {len(devices)} visible.")
    ne = FLAGS.engine_devices // tp
    per_engine = (-(-(FLAGS.rollout_ahead + 1) * FLAGS.groups_per_step // ne) * FLAGS.group_size
                  + -(-FLAGS.eval_concurrency // ne))
    if FLAGS.eval_samples > FLAGS.group_size:
        raise ValueError("eval_samples must not exceed group_size (the engine's max_group).")
    if FLAGS.eval_concurrency < FLAGS.eval_samples * ne:
        raise ValueError("eval_concurrency must allow one evaluation group per engine.")
    if per_engine > FLAGS.max_sessions:
        raise ValueError(f"Up to {per_engine} sessions per engine exceed max_sessions={FLAGS.max_sessions}.")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_path)
    eos = (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    pad = eos[1]
    grader = ProcessPoolExecutor(FLAGS.grade_workers, mp_context=multiprocessing.get_context("forkserver"))
    if FLAGS.env == "math":
        env = MathEnv(tokenizer, grader, FLAGS.grade_timeout_s, thinking=FLAGS.thinking)
    else:
        env = PythonToolEnv(
            tokenizer, grader, FLAGS.grade_timeout_s, thinking=FLAGS.thinking, max_turns=FLAGS.max_turns,
            tool_timeout_s=FLAGS.tool_timeout_s, tool_memory_bytes=FLAGS.tool_memory_gb << 30,
            tool_output_chars=FLAGS.tool_output_chars, max_tool_procs=FLAGS.max_tool_procs,
        )
    eval_tasks = [json.loads(line) for line in open(FLAGS.eval_path)]
    rows = [json.loads(line) for line in open(FLAGS.data_path)] + eval_tasks
    max_context = max(len(env.prompt(r)) for r in rows) + FLAGS.max_response
    if max_context > FLAGS.row_tokens:
        raise ValueError(f"Longest prompt + max_response = {max_context} exceeds row_tokens={FLAGS.row_tokens}.")

    t0 = time.perf_counter()
    model, cfg = load_causal_lm(FLAGS.model_path, sub_mesh(devices[:nt], 1, nt, 1), getattr(jnp, FLAGS.compute_dtype))
    trainer = Trainer(
        model,
        cfg,
        TrainConfig(
            lr=FLAGS.lr, weight_decay=FLAGS.weight_decay, adam_b1=FLAGS.adam_b1,
            adam_b2=FLAGS.adam_b2, adam_eps=FLAGS.adam_eps, max_grad_norm=FLAGS.max_grad_norm,
            eps_low=FLAGS.eps_low, eps_high=FLAGS.eps_high,
            tokens_per_microbatch=FLAGS.tokens_per_microbatch,
            row_tokens=FLAGS.row_tokens,
            logit_tile_tokens=FLAGS.logit_tile_tokens, pad_id=pad,
            host_optimizer=FLAGS.host_optimizer,
            loss_scale={"bfloat16": 1.0, "float16": 2.0**16}[FLAGS.compute_dtype],
            objective=FLAGS.objective, topk=FLAGS.topk, drift_every=FLAGS.drift_every,
        ),
    )
    del model
    ecfg = EngineConfig(
        num_pages=FLAGS.num_pages,
        max_context=max_context,
        decode_steps=FLAGS.decode_steps,
        decode_buckets=tuple(int(x) for x in FLAGS.decode_buckets),
        extend_buckets=tuple(int(x) for x in FLAGS.extend_buckets),
        eos_token_ids=eos,
        pad_id=pad,
        max_group=FLAGS.group_size,
        max_sessions=FLAGS.max_sessions,
        topk=FLAGS.topk,
    )
    sync = WeightSync(trainer.graphdef, trainer.params)
    params = sync.gather(trainer.params)
    for i in range(ne):
        mesh = sub_mesh(devices[nt + i * tp : nt + (i + 1) * tp], tp, 1, 1)
        replica, shardings = engine_replica(params, cfg, mesh)
        sync.add(Engine(replica, ecfg), shardings)
        del replica
    del params
    engines = sync.engines
    t1 = time.perf_counter()
    with ThreadPoolExecutor(len(engines)) as pool:
        warm = list(pool.map(lambda e: e.start(), engines))
    print(json.dumps({"time": time.time(), "load_s": t1 - t0, "engine_warmup_s": time.perf_counter() - t1,
                      "warmup_buckets_s": warm[0]}), flush=True)

    metrics_file = open(out / "metrics.jsonl", "a")

    (out / "samples").mkdir(exist_ok=True)

    def dump(step, samples):
        with open(out / "samples" / f"step_{step:06d}.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps({k: v for k, v in dataclasses.asdict(s).items() if k != "heads"}) + "\n")

    (out / "checkpoints").mkdir(exist_ok=True)
    writer = ThreadPoolExecutor(1)

    def save(step, params):
        # bfloat16 has no numpy dtype: stored as its uint16 bits under a "|bfloat16" key.
        arrays = {}
        for path, x in jax.tree_util.tree_flatten_with_path(jax.device_get(params))[0]:
            key = jax.tree_util.keystr(path)
            arrays[key + "|bfloat16" if x.dtype == jnp.bfloat16 else key] = x.view(np.uint16) if x.dtype == jnp.bfloat16 else x
        writer.submit(np.savez, out / "checkpoints" / f"step_{step:06d}.npz", **arrays)

    def log(m):
        if "step" in m:
            m["mem/peak_gb"] = [round(d.memory_stats()["peak_bytes_in_use"] / 2**30, 1) for d in devices]
        line = json.dumps({"time": time.time(), **m})
        print(line, flush=True)
        metrics_file.write(line + "\n")
        metrics_file.flush()

    lcfg = loop.LoopConfig(
        num_steps=FLAGS.num_steps,
        groups_per_step=FLAGS.groups_per_step,
        rollout_ahead=FLAGS.rollout_ahead,
        max_staleness=FLAGS.max_staleness,
        rollout=RolloutConfig(FLAGS.group_size, FLAGS.max_response, FLAGS.temperature),
        seed=FLAGS.seed,
        eval_every=FLAGS.eval_every,
        eval_samples=FLAGS.eval_samples,
        eval_concurrency=FLAGS.eval_concurrency,
        dump_every=FLAGS.dump_every,
        save_every=FLAGS.save_every,
    )
    tasks = loop.jsonl_tasks(FLAGS.data_path, FLAGS.seed)
    try:
        asyncio.run(loop.run(trainer, engines, sync, env, tasks, eval_tasks, lcfg, log, dump, save))
    finally:
        writer.shutdown()
        for e in engines:
            e.stop()
        grader.shutdown(cancel_futures=True)


if __name__ == "__main__":
    app.run(main)
