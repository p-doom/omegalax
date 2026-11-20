"""Common-update-plan path of rollout distillation through the real planner, iterator, model and trainer."""

import os
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import functools
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from absl.testing import absltest
from flax import nnx

from omegalax.data import rollout_teacher as rt
from omegalax.data.rollout_teacher_pipeline import (
    PLANNED_MICROBATCH_CHAINS,
    ROLLOUT_UPDATE_PLAN_FORMAT,
    TEACHER_BATCH_KEYS,
    _PlannedTeacherBatchBuilder,
    _json_sha256,
    _source_hashes,
    build_rollout_update_plan,
    load_rollout_update_plan,
    make_planned_rollout_teacher_iterator,
)
from omegalax.distributed.mesh import ensure_mesh
from omegalax.models.qwen3_5.config import Qwen3_5Config
from omegalax.models.qwen3_5.params import (
    create_qwen3_5_from_safetensors,
    export_qwen3_5_to_safetensors,
)
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.trainers import pretrain as trainer
from omegalax.trainers.checkpoint_utils import ResumeMode
from omegalax.trainers.loss import chunked_teacher_kl_stats
from omegalax.trainers.lr_schedule import build_common_token_schedule
from omegalax.trainers.text import TrainConfig, _make_checkpoint_manager, build_optimizer
from tests.test_rollout_teacher_pipeline import write_fixture_root, write_json
from tests.test_rollout_teacher_training import _tiny_teacher_config

REPO = Path(__file__).resolve().parents[1]
MODE = trainer.PretrainMode.STATEPASSING_BPTT
TEACHER = trainer.TrainingObjective.TEACHER_TOP512_SPECIAL_TAIL_KL
SIZES = [8] * 9 + [2] * 3
TRAIN = [(f"c{i:02d}", 600 + 40 * i, [10 + i, 511, 512, 590 + 40 * i]) for i in range(10)]
VAL = [(f"v{i}", 700 + 50 * i, [20, 512, 600 + 50 * i]) for i in range(5)]
_ROOT = None


def setUpModule():
    global _ROOT
    _ROOT = tempfile.TemporaryDirectory()


def tearDownModule():
    _ROOT.cleanup()


def _dir(name):
    path = Path(_ROOT.name) / name
    path.mkdir(parents=True)
    return path


def _write_dataset(name, train, val=()):
    samples = [
        dict(
            sequence_id=sid,
            split=split,
            status="completed",
            correct=True,
            token_count=n,
            target_positions=positions,
        )
        for split, group in (("train", train), ("val", val))
        for sid, n, positions in group
    ]
    return rt.TeacherDataset(write_fixture_root(_dir(name), samples))


def _plan(dataset, update_sizes, checkpoint_updates, max_context_tokens=1024, target_capacity=16):
    """Train chains are taken cyclically in catalog order; all chains are labelled correct."""
    records = [{**r, "label": "correct"} for r in dataset.sequences]
    train = [i for i, r in enumerate(records) if r["split"] == "train"]
    updates, cursor, cumulative = [], 0, 0
    for size in update_sizes:
        members = [train[(cursor + k) % len(train)] for k in range(size)]
        cursor += size
        tokens = sum(records[i]["token_count"] for i in members)
        cumulative += tokens
        updates.append(
            {
                "epoch": (cursor - 1) // len(train),
                "members": members,
                "context_tokens": tokens,
                "cumulative_tokens": cumulative,
            }
        )
    plan = {
        "format": ROLLOUT_UPDATE_PLAN_FORMAT,
        "source": {
            "dataset_root": str(dataset.root),
            **_source_hashes(dataset.root)[1],
            "scoring_binding": dataset.binding,
        },
        "records": records,
        "updates": updates,
        "validation_members": [i for i, r in enumerate(records) if r["split"] == "val"],
        "chains_per_update": max(update_sizes),
        "max_context_tokens": max_context_tokens,
        "target_capacity": target_capacity,
        "total_context_tokens": cumulative,
        "warmup_end_update": next(
            i + 1 for i, u in enumerate(updates) if u["cumulative_tokens"] >= 0.02 * cumulative
        ),
        "preanneal_update": next(
            i for i, u in enumerate(updates) if u["cumulative_tokens"] > 0.9 * cumulative
        ),
        "checkpoint_updates": list(checkpoint_updates),
        "lr": {
            "peak": 2e-5,
            "warmup_token_fraction": 0.02,
            "decay_token_fraction": 0.1,
            "position": "completed_update_cumulative_real_tokens",
            "final": 0.0,
        },
    }
    plan["plan_sha256"] = _json_sha256(plan)
    return plan


@functools.cache
def _pretrained():
    model, cfg = text_api.init_model(
        _tiny_teacher_config(), jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1
    )
    path = _dir("pretrained")
    export_qwen3_5_to_safetensors(model, Qwen3_5Config(text_config=cfg), path)
    return path


@functools.cache
def _main_dataset():
    return _write_dataset("main_dataset", TRAIN, VAL)


class Recorder:
    """W&B stand-in; optionally sends SIGUSR1 to this process while logging a given step."""

    def __init__(self, signal_when_logging_step=None):
        self.logs = []
        self.signal_when_logging_step = signal_when_logging_step

    def log(self, values, step):
        self.logs.append((int(step), dict(values)))
        if self.signal_when_logging_step is not None and int(step) == self.signal_when_logging_step:
            self.signal_when_logging_step = None
            os.kill(os.getpid(), signal.SIGUSR1)

    def steps(self, key):
        return [step for step, values in self.logs if key in values]

    def finish(self):
        pass


def _run(plan, save_dir, recorder=None, train=None, **kwargs):
    C = -(-plan["max_context_tokens"] // 512)
    cfg = TrainConfig(
        **{
            "seed": 0,
            "batch_size": PLANNED_MICROBATCH_CHAINS * C,
            "seq_len": 512,
            "num_steps": len(plan["updates"]),
            "lr_schedule_steps": len(plan["updates"]),
            "learning_rate": 2e-5,
            "max_grad_norm": 1.0,
            "grad_accum_steps": -(-plan["chains_per_update"] // PLANNED_MICROBATCH_CHAINS),
            **(train or {}),
        }
    )
    if plan["validation_members"]:
        kwargs = dict(
            val_data_iter=make_planned_rollout_teacher_iterator(
                plan, segment_length=512, split="val"
            ),
            val_every=2,
            val_steps=-(-len(plan["validation_members"]) // PLANNED_MICROBATCH_CHAINS),
            **kwargs,
        )
    pretrained = _pretrained()
    with mock.patch.dict(os.environ):
        os.environ.pop("SLURM_JOB_ID")  # a signalled run must never reach `scontrol requeue`
        return trainer.run_pretrain(
            str(pretrained),
            cfg,
            make_planned_rollout_teacher_iterator(plan, segment_length=512),
            save_dir=save_dir,
            wandb_run=recorder,
            pretrain_mode=MODE,
            training_objective=TEACHER,
            teacher_segment_buckets=[C],
            common_update_plan=plan,
            init_hf_model_path=pretrained,
            tp_size=1,
            dp_size=1,
            text_attn_backend="xla",
            log_every=1,
            save_every=0,
            keep_latest=None,
            **{"fsdp_size": 1, **kwargs},
        )


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _flat(tree):
    return {
        jax.tree_util.keystr(p): np.asarray(
            jax.random.key_data(v) if jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key) else v
        )
        for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def _adam_gradient(optimizer):
    """Gradient Adam received in the first update (adam_beta1=0.9: mu = 0.1 * g)."""
    return {
        k.split("['mu']", 1)[1]: v / 0.1
        for k, v in _flat(nnx.state(optimizer)).items()
        if "['inner_opt_state']" in k and "['mu']" in k
    }


def _norm(tree):
    return float(np.sqrt(sum(np.sum(v.astype(np.float64) ** 2) for v in tree.values())))


def _checkpoint_params(save_dir, step):
    manager = _make_checkpoint_manager(Path(save_dir), None, max_to_keep=None)
    restored = manager.restore(step, args=ocp.args.Composite(train_state=ocp.args.PyTreeRestore()))
    manager.close()
    return {
        k: v
        for k, v in _flat(restored["train_state"]).items()
        if k.startswith("['optimizer']['model']")
    }


def _load_model():
    pretrained = _pretrained()
    mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
    model, _ = create_qwen3_5_from_safetensors(
        str(pretrained), tp_size=1, fsdp_size=1, dp_size=1, text_only=True
    )
    set_attn_backend(model, text_backend="xla")
    return (
        model,
        text_api.align_config_to_mesh(text_api.resolve_config(str(pretrained)), mesh),
        mesh,
    )


@functools.cache
def _base_run():
    plan = _plan(_main_dataset(), SIZES, range(1, 13))
    recorder = Recorder()
    save_dir = _dir("base_run")
    optimizer, _ = _run(plan, save_dir, recorder)
    return plan, save_dir, recorder, _flat(nnx.state(optimizer))


@functools.cache
def _interrupted_run(scenario):
    plan = _plan(_main_dataset(), SIZES, [4, 8, 12])
    assert plan["preanneal_update"] == 8, plan["preanneal_update"]
    save_dir = _dir(f"interrupted_{scenario}")
    first = Recorder({"signal_early": 1, "signal_decay": 9}.get(scenario))
    _run(
        plan, save_dir, first, **({"stop_after_local_step": 10} if scenario == "stop_decay" else {})
    )
    second = Recorder()
    optimizer, _ = _run(plan, save_dir, second, resume=ResumeMode.REQUIRED)
    return {
        "metrics": [r["step"] for r in _read_jsonl(save_dir / "update_metrics.jsonl")],
        "identities": [r["step"] for r in _read_jsonl(save_dir / "update_identities.jsonl")],
        "wandb": first.steps("train/teacher_kl") + second.steps("train/teacher_kl"),
        "state": _flat(nnx.state(optimizer)),
    }


class LearningRateTest(absltest.TestCase):
    # Pins: under MultiSteps(k=4) the optimizer applies schedule(i) at update i, nothing in between, 0 at the last update, same after restoring its state.
    def test_applied_lr_equals_schedule_under_accumulation_and_resume(self):
        class Toy(nnx.Module):
            def __init__(self):
                self.w = nnx.Param(jnp.zeros((2,), jnp.float32))

        cumulative = np.cumsum([500] * 100)
        plan = {
            "updates": [{"cumulative_tokens": int(c)} for c in cumulative],
            "total_context_tokens": int(cumulative[-1]),
            "lr": {"peak": 2e-5},
        }
        schedule = build_common_token_schedule(plan)
        cfg = TrainConfig(
            num_steps=100,
            lr_schedule_steps=100,
            weight_decay=0.0,
            max_grad_norm=0.0,
            grad_accum_steps=4,
        )

        def grad(model):
            return nnx.grad(lambda m: m.w[...].sum())(model)

        def train(model, optimizer, updates):
            deltas = []
            for _ in updates:
                before = np.asarray(model.w[...]).copy()
                for micro in range(4):
                    optimizer.update(grad(model))
                    if micro < 3:
                        np.testing.assert_array_equal(np.asarray(model.w[...]), before)
                deltas.append(before - np.asarray(model.w[...]))
            return deltas

        model = Toy()
        optimizer = build_optimizer(model, cfg, lr_schedule=schedule)
        deltas = train(model, optimizer, range(50))
        saved = jax.tree.map(np.array, nnx.state(optimizer))
        deltas += train(model, optimizer, range(50, 100))
        for i, delta in enumerate(deltas[:-1]):
            np.testing.assert_allclose(delta, float(schedule(i)), rtol=1e-4, err_msg=f"update {i}")
        self.assertGreater(float(schedule(98)), 0.0)
        np.testing.assert_array_equal(deltas[-1], 0.0)
        resumed_model = Toy()
        resumed = build_optimizer(resumed_model, cfg, lr_schedule=schedule)
        nnx.update(resumed, jax.tree.map(jnp.asarray, saved))
        resumed_deltas = train(resumed_model, resumed, range(50, 100))
        for got, want in zip(resumed_deltas, deltas[50:]):
            np.testing.assert_array_equal(got, want)

    # Pins: run_pretrain logs lr == schedule(step-1) at every update and the last update leaves the model weights unchanged.
    def test_trainer_logged_lr_and_last_update_is_noop(self):
        plan, save_dir, _, _ = _base_run()
        schedule = build_common_token_schedule(plan)
        rows = _read_jsonl(save_dir / "update_metrics.jsonl")
        self.assertEqual([r["step"] for r in rows], list(range(1, 13)))
        for row in rows:
            self.assertEqual(row["lr"], float(schedule(row["step"] - 1)), row["step"])
        p10, p11, p12 = (_checkpoint_params(save_dir, s) for s in (10, 11, 12))
        self.assertTrue(any(not np.array_equal(p10[k], p11[k]) for k in p11))
        for key in p11:
            np.testing.assert_array_equal(p11[key], p12[key], err_msg=key)


class AccumulatedGradientTest(absltest.TestCase):
    # Pins: an update of 2 real chains plus a padding-only microbatch hands Adam grad(sum KL / N_update); with clipping its norm is the clip value.
    def test_short_update_gradient_and_clipping(self):
        plan = _plan(_main_dataset(), [2, 8, 8], [1, 2, 3])
        train = dict(weight_decay=0.0, adam_beta1=0.9)
        save_dir = _dir("short_noclip")
        optimizer, _ = _run(
            plan, save_dir, stop_after_local_step=1, train=dict(train, max_grad_norm=0.0)
        )
        got = _adam_gradient(optimizer)
        row = _read_jsonl(save_dir / "update_metrics.jsonl")[0]
        self.assertEqual(row["teacher_targets"], 8)
        self.assertEqual(row["context_tokens"], TRAIN[0][1] + TRAIN[1][1])

        model, cfg, mesh = _load_model()
        batch, _, _ = trainer.prepare_pretrain_batch(
            next(make_planned_rollout_teacher_iterator(plan, segment_length=512)),
            MODE,
            cfg,
            mesh,
            global_batch_size=2 * PLANNED_MICROBATCH_CHAINS,
            training_objective=TEACHER,
        )

        def mean_kl(model):
            kl_sum, _, metrics = trainer._teacher_loss_stats(model, batch, cfg, 0, MODE)
            return kl_sum / metrics["teacher_targets"]

        want = _flat(nnx.jit(nnx.grad(mean_kl))(model))
        self.assertEqual(len(got), len(want))
        for key, value in got.items():
            matches = [k for k in want if key.endswith(k) or k.endswith(key)]
            self.assertLen(matches, 1, key)
            ref = want[matches[0]]
            np.testing.assert_allclose(
                value, ref, rtol=1e-5, atol=1e-5 * float(np.max(np.abs(ref))), err_msg=key
            )
        self.assertGreater(_norm(got), 1.0)

        optimizer, _ = _run(
            plan, _dir("short_clip"), stop_after_local_step=1, train=dict(train, max_grad_norm=1e-3)
        )
        self.assertAlmostEqual(_norm(_adam_gradient(optimizer)) / 1e-3, 1.0, delta=1e-4)


class ValidationTest(absltest.TestCase):
    # Pins: every planned validation pass covers each val chain once, identically each time, and reports KL = summed KL / summed targets.
    def test_planned_validation_pass(self):
        plan, _, recorder, _ = _base_run()
        self.assertEqual(recorder.steps("val/teacher_kl"), [2, 4, 6, 8, 10, 12])
        for step, values in recorder.logs:
            if "val/teacher_kl" in values:
                self.assertEqual(values["val/teacher_targets"], sum(len(v[2]) for v in VAL), step)
                self.assertEqual(values["val/context_tokens"], sum(v[1] for v in VAL), step)
                self.assertAlmostEqual(
                    values["val/teacher_kl"],
                    values["val/teacher_kl_sum"] / values["val/teacher_targets"],
                    places=5,
                )
        iterator = make_planned_rollout_teacher_iterator(plan, segment_length=512, split="val")
        ids = [next(iterator)["metadata"]["sequence_ids"] for _ in range(2)]
        self.assertNotIn(None, ids[0])
        self.assertIn(None, ids[1])
        self.assertCountEqual(
            [s for batch in ids for s in batch if s is not None], [v[0] for v in VAL]
        )


class BoundaryRowsTest(absltest.TestCase):
    # Pins: all KL/NLL and boundary sums and counts are equal for T=8192/2048/512 and match a float64 reference.
    def test_rows_identical_across_segment_lengths(self):
        x = ("x", 17000, [*range(8190, 8449), *range(16384, 16584)])
        y = ("y", 16700, list(range(16384, 16640)))
        z = ("z", 5000, list(range(100, 111)))
        dataset = _write_dataset("boundary_dataset", [x, y, z])
        plan = _plan(dataset, [3], [1], max_context_tokens=17000, target_capacity=1024)
        rng = np.random.default_rng(0)
        hidden = rng.standard_normal((4, 24576, 8), dtype=np.float32)
        kernel = (0.5 * rng.standard_normal((8, rt.VOCAB_SIZE))).astype(np.float32)
        ref = dict(kl=0.0, nll=0.0, bkl=0.0, bnll=0.0)
        for b, (sid, lo, hi) in enumerate([("x", 8192, 8448), ("y", 16384, 16640), ("z", 0, 0)]):
            tokens = dataset.get_sequence(sid)[1]["token_ids"]
            rows = dataset.get_targets(sid)
            positions = rows["target_positions"]
            for start in range(0, len(positions), 64):
                part = {k: v[start : start + 64] for k, v in rows.items()}
                pos = part["target_positions"]
                logits = hidden[b, pos - 1].astype(np.float64) @ kernel.astype(np.float64)
                peak = logits.max(1)
                nll = (
                    peak
                    + np.log(np.exp(logits - peak[:, None]).sum(1))
                    - logits[np.arange(len(pos)), tokens[pos]]
                )
                kl = np.asarray(
                    rt.top512_special_tail_kl(logits.astype(np.float32), part), np.float64
                )
                window = (pos >= lo) & (pos < hi)
                ref["kl"] += kl.sum()
                ref["nll"] += nll.sum()
                ref["bkl"] += kl[window].sum()
                ref["bnll"] += nll[window].sum()
        stats = jax.jit(
            lambda h, w, t, ids: chunked_teacher_kl_stats(h, w, t, boundary_token_ids_BT=ids)
        )
        for T in (8192, 2048, 512):
            batch = _PlannedTeacherBatchBuilder(plan, T, "train", 0)(0)
            B, C, _ = batch["token_ids_BCT"].shape
            kl, count, m = stats(
                jnp.asarray(hidden[:, : C * T]),
                jnp.asarray(kernel),
                {k: jnp.asarray(batch[k]) for k in TEACHER_BATCH_KEYS},
                jnp.asarray(batch["token_ids_BCT"].reshape(B, C * T)),
            )
            m = {k: float(v) for k, v in m.items()}
            self.assertEqual(float(count), 726, T)
            self.assertEqual(m["teacher_boundary_windows"], 2, T)
            self.assertEqual(m["teacher_boundary_rollouts"], 2, T)
            self.assertEqual(m["teacher_boundary_nll_tokens"], 512, T)
            self.assertEqual(m["teacher_boundary_kl_tokens"], 512, T)
            for got, want in (
                (float(kl), ref["kl"]),
                (m["teacher_nll_sum"], ref["nll"]),
                (m["teacher_boundary_kl_sum"], ref["bkl"]),
                (m["teacher_boundary_nll_sum"], ref["bnll"]),
            ):
                self.assertAlmostEqual(got / want, 1.0, delta=1e-6, msg=T)


class PaddingNeutralityTest(absltest.TestCase):
    # Pins: padding rows, row order and batch composition do not change real-chain sums through the model, for the GDN-state, no-state and conv-state arms, which differ.
    def test_composition_invariance_per_arm(self):
        chains = [
            (f"p{i}", 600 + 40 * i, [10 + i, 511, 512, 513, 514, 590 + 40 * i]) for i in range(6)
        ]
        plan = _plan(_write_dataset("padding_dataset", chains), [6], [1])
        model, cfg, mesh = _load_model()
        planned = _PlannedTeacherBatchBuilder(plan, 512, "train", 0)

        def batch_of(members):
            records = [
                None
                if m is None
                else {
                    **plan["records"][m],
                    "sample_idx": m,
                    "bucket_idx": 0,
                    "num_segments": -(-plan["records"][m]["token_count"] // 512),
                }
                for m in members
            ]
            raw = planned.builder._make_batch(records, 0)
            return trainer.prepare_pretrain_batch(
                raw, MODE, cfg, mesh, global_batch_size=8, training_objective=TEACHER
            )[0]

        P = None
        compositions = [
            [[0, 1, 2, 3], [4, 5, P, P]],
            [[0, 1, 4, 5], [2, 3, P, P]],
            [[3, 2, 1, 0], [P, 5, P, 4]],
            [[m, P, P, P] for m in range(6)],
        ]
        keys = ("teacher_kl_sum", "teacher_nll_sum", "teacher_targets", "context_tokens")
        arms = {}
        for arm, flags in (
            ("gdn", {}),
            ("none", dict(pass_gdn_state=False)),
            ("conv", dict(pass_conv_state=True)),
        ):
            step = trainer.make_pretrain_eval_step(
                MODE, cfg, pad_id=0, training_objective=TEACHER, **flags
            )
            totals = []
            for batches in compositions:
                total = dict.fromkeys(keys, 0.0)
                for members in batches:
                    out = step(model, batch_of(members))
                    for key in keys:
                        total[key] += float(out[key])
                totals.append(total)
            for total in totals[1:]:
                self.assertEqual(total["teacher_targets"], totals[0]["teacher_targets"], arm)
                self.assertEqual(total["context_tokens"], totals[0]["context_tokens"], arm)
                for key in keys[:2]:
                    self.assertAlmostEqual(total[key] / totals[0][key], 1.0, delta=1e-5, msg=arm)
            empty = step(model, batch_of([P, P, P, P]))
            self.assertEqual([float(empty[k]) for k in keys], [0.0] * 4, arm)
            arms[arm] = totals[0]["teacher_kl_sum"]
        for a, b in (("gdn", "none"), ("gdn", "conv"), ("none", "conv")):
            self.assertGreater(abs(arms[a] - arms[b]) / abs(arms[a]), 1e-4, (a, b))


_EIGHT_DEVICE_DRIVER = """
import json, sys
from pathlib import Path
import jax, numpy as np
from flax import nnx
import omegalax.data.rollout_teacher_pipeline as pipeline
from omegalax.trainers import pretrain as trainer
assert jax.device_count() == 8, jax.devices()
pipeline.PLANNED_MICROBATCH_CHAINS = 8
trainer.PLANNED_MICROBATCH_CHAINS = 8
root = Path(sys.argv[1])
plan = json.loads((root / "plan.json").read_text())
cfg = trainer.TrainConfig(seed=0, batch_size=16, seq_len=512, num_steps=2, lr_schedule_steps=2,
                          learning_rate=2e-5, grad_accum_steps=2, max_grad_norm=0.0,
                          weight_decay=0.0, adam_beta1=0.9)
optimizer, _ = trainer.run_pretrain(
    str(root / "pretrained"), cfg, pipeline.make_planned_rollout_teacher_iterator(plan, segment_length=512),
    save_dir=root / "eight", stop_after_local_step=1, pretrain_mode=trainer.PretrainMode.STATEPASSING_BPTT,
    training_objective=trainer.TrainingObjective.TEACHER_TOP512_SPECIAL_TAIL_KL, teacher_segment_buckets=[2],
    common_update_plan=plan, init_hf_model_path=root / "pretrained", tp_size=1, fsdp_size=8, dp_size=1,
    text_attn_backend="xla", log_every=1, save_every=0, keep_latest=None)
state = {jax.tree_util.keystr(p): np.asarray(v) for p, v in
         jax.tree_util.tree_flatten_with_path(nnx.state(optimizer))[0]
         if "['inner_opt_state']" in jax.tree_util.keystr(p) and "['mu']" in jax.tree_util.keystr(p)}
np.savez(root / "mu_eight.npz", **{f"g{i}": state[k] for i, k in enumerate(sorted(state))})
"""


class EightDeviceTest(absltest.TestCase):
    # Pins: one 16-chain update as 2x8-chain microbatches on 8 devices equals the same update as 4x4-chain microbatches on 1 device.
    def test_eight_chain_microbatches_on_eight_devices(self):
        root = _dir("eight_devices")
        chains = [(f"e{i:02d}", 600 + 25 * i, [10 + i, 511, 512, 590 + 25 * i]) for i in range(16)]
        plan = _plan(_write_dataset("eight_dataset", chains), [16, 16], [1, 2])
        (root / "plan.json").write_text(json.dumps(plan))
        (root / "pretrained").symlink_to(_pretrained())
        optimizer, _ = _run(
            plan,
            root / "ref",
            stop_after_local_step=1,
            train=dict(max_grad_norm=0.0, weight_decay=0.0, adam_beta1=0.9),
        )
        ref_mu = {
            k: v
            for k, v in _flat(nnx.state(optimizer)).items()
            if "['inner_opt_state']" in k and "['mu']" in k
        }
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO),
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
        }
        del env["SLURM_JOB_ID"]
        run = subprocess.run(
            [sys.executable, "-c", _EIGHT_DEVICE_DRIVER, str(root)],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        self.assertEqual(run.returncode, 0, run.stdout[-3000:] + run.stderr[-5000:])
        ref_row, row = (_read_jsonl(root / d / "update_metrics.jsonl")[0] for d in ("ref", "eight"))
        self.assertEqual(row["teacher_targets"], ref_row["teacher_targets"])
        self.assertEqual(row["context_tokens"], ref_row["context_tokens"])
        self.assertAlmostEqual(row["teacher_kl_sum"] / ref_row["teacher_kl_sum"], 1.0, delta=1e-5)
        self.assertEqual(
            *(_read_jsonl(root / d / "update_identities.jsonl")[0] for d in ("ref", "eight"))
        )
        contracts = [
            json.loads((root / d / "common_update_contract.json").read_text())
            for d in ("ref", "eight")
        ]
        self.assertEqual([c["microbatch_chains"] for c in contracts], [4, 8])
        with np.load(root / "mu_eight.npz") as eight:
            self.assertLen(eight.files, len(ref_mu))
            for i, key in enumerate(sorted(ref_mu)):
                want = ref_mu[key]
                np.testing.assert_allclose(
                    eight[f"g{i}"],
                    want,
                    rtol=1e-4,
                    atol=1e-5 * float(np.max(np.abs(want))),
                    err_msg=key,
                )


def _write_planner_dataset(name, train, val, finish_length=(), final_truncated=()):
    """Fixture root with the metadata fields the planner reads (target_counts, content_key, attempt result)."""
    root = write_fixture_root(
        _dir(name),
        [
            dict(
                sequence_id=sid,
                split=split,
                status="completed",
                correct=(i % 3 != 0),
                token_count=n,
                target_positions=[5, 6],
            )
            for split, group in (("train", train), ("val", val))
            for i, (sid, n) in enumerate(group)
        ],
    )
    snapshot = json.loads((root / "prepared/snapshot.json").read_text())
    for attempt in snapshot["attempts"]:
        sid = attempt["attempt_id"].removeprefix("attempt-")
        attempt["result"].update(
            error=None,
            token_count_complete=True,
            finish_reason="length" if sid in finish_length else "stop",
        )
    prepared = json.loads((root / "prepared.json").read_text())
    catalog = json.loads((root / rt.CATALOG_FILENAME).read_text())
    for entries in (prepared["sequences"], catalog["sequences"]):
        for entry in entries:
            entry["content_key"] = f"content-{entry['sequence_id']}"
            entry["target_counts"] = (
                {"complete": entry["target_count"] - 1, "final_truncated": 1}
                if entry["sequence_id"] in final_truncated
                else {"complete": entry["target_count"]}
            )
    prepared["snapshot_sha256"] = write_json(root / "prepared/snapshot.json", snapshot)
    manifest = json.loads((root / rt.MANIFEST_FILENAME).read_text())
    manifest["snapshot"]["sha256"] = prepared["snapshot_sha256"]
    manifest["prepared"]["sha256"] = write_json(root / "prepared.json", prepared)
    write_json(root / rt.MANIFEST_FILENAME, manifest)
    catalog["dataset_binding"] = rt.dataset_source_binding(manifest)
    write_json(root / rt.CATALOG_FILENAME, catalog)
    return root


class UpdatePlanBuilderTest(absltest.TestCase):
    # Pins: build_rollout_update_plan selection, two-epoch membership, token bookkeeping, warmup/preanneal/checkpoint arithmetic, determinism and tamper detection.
    def test_build_rollout_update_plan(self):
        train = [(f"t{i:02d}", 20000 + 850 * i) for i in range(40)]
        excluded = [("short", 19999), ("long", 55001), ("trunc", 30000), ("length", 30000)]
        val = [(f"val{i}", 21000 + 5000 * i) for i in range(6)]
        root = _write_planner_dataset(
            "planner_dataset",
            train + excluded,
            val,
            finish_length={"length"},
            final_truncated={"trunc"},
        )
        plan = build_rollout_update_plan(root, root.parent / "plan_a.json")
        build_rollout_update_plan(root, root.parent / "plan_b.json")
        self.assertEqual(
            (root.parent / "plan_a.json").read_bytes(), (root.parent / "plan_b.json").read_bytes()
        )
        self.assertEqual(load_rollout_update_plan(root.parent / "plan_a.json"), plan)

        records = plan["records"]
        self.assertCountEqual([r["sequence_id"] for r in records], [s for s, _ in train + val])
        self.assertEqual({r["label"] for r in records}, {"correct", "incorrect"})
        train_idx = {i for i, r in enumerate(records) if r["split"] == "train"}
        self.assertEqual(
            sorted(i for i, r in enumerate(records) if r["split"] == "val"),
            sorted(plan["validation_members"]),
        )
        self.assertEqual(plan["chains_per_update"], 12)
        orders = []
        for epoch in (0, 1):
            updates = [u for u in plan["updates"] if u["epoch"] == epoch]
            members = [m for u in updates for m in u["members"]]
            self.assertCountEqual(members, train_idx)
            self.assertEqual([len(u["members"]) for u in updates], [12, 12, 12, 4])
            orders.append(members)
        self.assertNotEqual(orders[0], orders[1])
        self.assertEqual([u["epoch"] for u in plan["updates"]], [0] * 4 + [1] * 4)

        cumulative = 0
        for update in plan["updates"]:
            self.assertEqual(
                update["context_tokens"], sum(records[i]["token_count"] for i in update["members"])
            )
            cumulative += update["context_tokens"]
            self.assertEqual(update["cumulative_tokens"], cumulative)
        total = plan["total_context_tokens"]
        self.assertEqual(cumulative, total)
        ends = [u["cumulative_tokens"] for u in plan["updates"]]
        self.assertEqual(
            plan["warmup_end_update"], 1 + min(i for i, c in enumerate(ends) if c >= 0.02 * total)
        )
        preanneal = plan["preanneal_update"]
        self.assertLessEqual(ends[preanneal - 1], 0.9 * total)
        self.assertGreater(ends[preanneal], 0.9 * total)
        schedule = build_common_token_schedule(plan)
        self.assertEqual(float(schedule(preanneal - 1)), np.float32(2e-5))
        self.assertLess(float(schedule(preanneal)), np.float32(2e-5))
        self.assertEqual(float(schedule(len(ends) - 1)), 0.0)
        self.assertEqual(plan["checkpoint_updates"], sorted({preanneal, len(ends)}))
        self.assertEqual(plan["target_capacity"], 1024)

        text = (root.parent / "plan_a.json").read_text()
        self.assertIn('"seed":0', text)
        (root.parent / "plan_c.json").write_text(text.replace('"seed":0', '"seed":1', 1))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            load_rollout_update_plan(root.parent / "plan_c.json")


class ResumeSideEffectsTest(absltest.TestCase):
    # Pins: after a signal before the pre-anneal point and resume, both side files hold steps 1..12 exactly once.
    def test_signal_before_preanneal_side_files_once(self):
        run = _interrupted_run("signal_early")
        self.assertEqual(run["metrics"], list(range(1, 13)))
        self.assertEqual(run["identities"], list(range(1, 13)))

    # Pins: after a signal before the pre-anneal point and resume, every step reaches W&B exactly once (regression D5: the step during which the signal arrived was never logged).
    def test_signal_before_preanneal_every_step_reaches_wandb(self):
        self.assertEqual(sorted(set(_interrupted_run("signal_early")["wandb"])), list(range(1, 13)))

    # Pins: after a signal in the decay phase and resume, both side files hold steps 1..12 exactly once (regression D3: no checkpoint, replayed steps appended twice).
    def test_signal_after_preanneal_side_files_once(self):
        run = _interrupted_run("signal_decay")
        self.assertEqual(run["metrics"], list(range(1, 13)))
        self.assertEqual(run["identities"], list(range(1, 13)))

    # Pins: after a stop without checkpoint in the decay phase and resume, both side files hold steps 1..12 exactly once (regression D4: append-only side files).
    def test_stop_without_checkpoint_side_files_once(self):
        run = _interrupted_run("stop_decay")
        self.assertEqual(run["metrics"], list(range(1, 13)))
        self.assertEqual(run["identities"], list(range(1, 13)))

    # Pins: after a decay-phase signal or a stop without checkpoint, every step 1..12 reaches the W&B stand-in.
    def test_decay_phase_interruptions_reach_wandb(self):
        for scenario in ("signal_decay", "stop_decay"):
            self.assertEqual(
                sorted(set(_interrupted_run(scenario)["wandb"])), list(range(1, 13)), scenario
            )

    # Pins: all three interruptions end in an optimizer state bitwise equal to the uninterrupted run.
    def test_resumed_state_bitwise_equals_uninterrupted(self):
        reference = _base_run()[3]
        for scenario in ("signal_early", "signal_decay", "stop_decay"):
            state = _interrupted_run(scenario)["state"]
            self.assertEqual(sorted(state), sorted(reference), scenario)
            for key, value in reference.items():
                np.testing.assert_array_equal(state[key], value, err_msg=f"{scenario} {key}")


if __name__ == "__main__":
    absltest.main()
