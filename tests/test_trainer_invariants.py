"""Invariant tests for the state-passing training loop (scan path, accumulation, resume, CE)."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import flags
from absl.testing import absltest
from flax import nnx

from omegalax.data.pretrain_data_set import (
    BATCH_PRETRAIN_METADATA_KEY,
    DOC_CHAIN_FORMAT,
    write_json_arrayrecord_dataset,
)
from omegalax.data.pretrain_statepassing import (
    build_statepassing_window_index,
    make_statepassing_iterator,
)
from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.trainers import checkpoint_utils
from omegalax.trainers import pretrain as pretrain_trainer
from omegalax.trainers import text as text_trainer
from omegalax.trainers.loss import chunked_cross_entropy_multi_stats, chunked_cross_entropy_stats
from omegalax.trainers.lr_schedule import build_lr_schedule
from tests.test_pretrain_training import _tiny_qwen3_5_config

REPO_ROOT = Path(__file__).resolve().parents[1]
BPTT = pretrain_trainer.PretrainMode.STATEPASSING_BPTT
NO_BPTT = pretrain_trainer.PretrainMode.STATEPASSING_NO_BPTT


class _WandbRecorder:
    def __init__(self):
        self.rows = []

    def log(self, values, step=None):
        self.rows.append((step, dict(values)))

    def series(self, key):
        return {step: values[key] for step, values in self.rows if key in values}


def _write_doc(root: Path, split: str, n_tokens: int, seed: int) -> None:
    rng = np.random.RandomState(seed)
    token_ids = [int(x) for x in rng.randint(1, 63, size=n_tokens)]
    write_json_arrayrecord_dataset(
        (
            {
                "dataset_format": DOC_CHAIN_FORMAT,
                "doc_id": f"{split}-doc",
                "token_ids": token_ids,
                "doc_token_count": len(token_ids),
            },
        ),
        root / split / "bucket_2k",
        records_per_shard=16,
        overwrite=False,
        metadata={"dataset_format": DOC_CHAIN_FORMAT},
    )


def _build_index(tmp: Path, split: str, n_tokens: int, num_segments: int, seed: int) -> Path:
    data_root = tmp / f"docs_{split}_{n_tokens}"
    _write_doc(data_root, split, n_tokens, seed)
    return build_statepassing_window_index(
        data_root,
        tmp / f"index_{split}_{n_tokens}_c{num_segments}",
        chunk_length=4,
        num_segments=num_segments,
        split=split,
        records_per_shard=16,
    )


def _make_iter(index_path: Path, batch_size: int, *, shuffle: bool, seed: int = 17):
    return make_statepassing_iterator(
        index_path,
        batch_size=batch_size,
        chunk_length=4,
        shuffle=shuffle,
        seed=seed,
        num_epochs=1,
        dp_size=1,
        fsdp_size=1,
        process_index=0,
        grain_workers=0,
        grain_read_threads=1,
        grain_read_prefetch_buffer_size=1,
    )


_real_install_handler = pretrain_trainer._install_requeue_signal_handler


def _install_handler_and_signal():
    requested = _real_install_handler()
    os.kill(os.getpid(), signal.SIGUSR1)
    return requested


def _signal_patches():
    return (
        mock.patch.object(
            pretrain_trainer,
            "_install_requeue_signal_handler",
            side_effect=_install_handler_and_signal,
        ),
        mock.patch.object(pretrain_trainer, "_request_slurm_requeue_if_primary"),
    )


def _flat(tree) -> np.ndarray:
    return np.concatenate([np.asarray(x, np.float64).ravel() for x in jax.tree.leaves(tree)])


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _opt_counters(optimizer) -> dict[str, int]:
    return {
        jax.tree_util.keystr(path): int(np.asarray(leaf))
        for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(optimizer))[0]
        if jax.tree_util.keystr(path)
        .split("[")[-1]
        .startswith(("'mini_step'", "'gradient_step'", "'count'"))
    }


def _assert_states_equal(testcase, optimizer_a, optimizer_b) -> None:
    state_a, state_b = nnx.state(optimizer_a), nnx.state(optimizer_b)
    testcase.assertEqual(jax.tree.structure(state_a), jax.tree.structure(state_b))
    for leaf_a, leaf_b in zip(jax.tree.leaves(state_a), jax.tree.leaves(state_b), strict=True):
        np.testing.assert_array_equal(np.asarray(leaf_a), np.asarray(leaf_b))


def _three_layer_config():
    return dataclasses.replace(
        _tiny_qwen3_5_config(),
        num_hidden_layers=3,
        layer_types=("linear_attention", "full_attention", "linear_attention"),
    )


def _scan_batch(C: int, T: int, padded_tail_segments: int, seed: int):
    B = 3
    rng = np.random.RandomState(seed)
    token_ids = rng.randint(1, 64, size=(B, C, T)).astype(np.int32)
    attn = np.ones((B, C, T), dtype=np.int32)
    attn[1, -1, T // 2 :] = 0
    attn[2, C - padded_tail_segments :, :] = 0
    attn[2, C - padded_tail_segments - 1, T - 1 :] = 0
    token_ids[attn == 0] = 0
    reset = np.zeros((B, C), dtype=bool)
    reset[:, 0] = True
    return {
        "token_ids_BCT": jnp.asarray(token_ids),
        "attention_mask_BCT": jnp.asarray(attn),
        "loss_mask_BCT": jnp.asarray(attn),
        "reset_state_BC": jnp.asarray(reset),
    }


def _scan_value_and_grad(cfg, batch, *, gdn_layer_limit, pass_gdn_state, pass_conv_state):
    model, _ = text_api.init_model(cfg, jax.random.PRNGKey(0), tp_size=1, fsdp_size=1, dp_size=1)
    set_attn_backend(model, text_backend="xla")

    @nnx.jit
    def run(model):
        def loss_fn(m):
            return pretrain_trainer._statepassing_loss_stats(
                m,
                batch,
                cfg,
                0,
                BPTT,
                gdn_layer_limit=gdn_layer_limit,
                pass_gdn_state=pass_gdn_state,
                pass_conv_state=pass_conv_state,
            )

        return nnx.value_and_grad(loss_fn, has_aux=True)(model)

    (loss, metrics), grads = run(model)
    return float(loss), {k: float(v) for k, v in metrics.items()}, grads


def _sp_batch(tok, attn):
    reset = np.zeros(tok.shape[:2], dtype=bool)
    reset[:, 0] = True
    return {
        "token_ids_BCT": jnp.asarray(tok, jnp.int32),
        "attention_mask_BCT": jnp.asarray(attn, jnp.int32),
        "loss_mask_BCT": jnp.asarray(attn, jnp.int32),
        "reset_state_BC": jnp.asarray(reset),
    }


def _accumulation_vs_one_batch(unequal: bool):
    """Returns (update_accum, update_one_batch, metrics_accum, metrics_one_batch)."""
    cfg = _tiny_qwen3_5_config()
    rng = np.random.RandomState(0)
    tok1 = rng.randint(1, 64, size=(2, 2, 6))
    tok2 = rng.randint(1, 64, size=(2, 2, 6))
    attn1 = np.ones_like(tok1)
    attn2 = np.ones_like(tok2)
    if unequal:
        attn2[1, 0, 3:] = 0
        attn2[1, 1, :] = 0
        tok2 = tok2 * attn2
    micro = [_sp_batch(tok1, attn1), _sp_batch(tok2, attn2)]
    one = [_sp_batch(np.concatenate([tok1, tok2]), np.concatenate([attn1, attn2]))]

    def run(batches, accum):
        model, _ = text_api.init_model(
            cfg, jax.random.PRNGKey(0), tp_size=1, fsdp_size=1, dp_size=1
        )
        optimizer = text_trainer.build_optimizer(
            model,
            text_trainer.TrainConfig(
                batch_size=2,
                seq_len=6,
                num_steps=1,
                learning_rate=1e5,
                adam_eps=1e6,
                weight_decay=0.0,
                max_grad_norm=0.0,
                lr_schedule="linear",
                grad_accum_steps=accum,
            ),
        )
        before = _flat(nnx.state(optimizer.model, nnx.Param))
        step = pretrain_trainer.make_pretrain_train_step(BPTT, cfg)
        acc = {}
        for batch in batches:
            _, metrics = step(optimizer, batch)
            pretrain_trainer._accumulate_metric_sums(acc, metrics)
        after = _flat(nnx.state(optimizer.model, nnx.Param))
        return after - before, pretrain_trainer._finalize_accumulated_metrics(acc)

    with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
        update_accum, metrics_accum = run(micro, accum=2)
        update_one, metrics_one = run(one, accum=1)
    return update_accum, update_one, metrics_accum, metrics_one


_T9_SCRIPT = r"""
import json, os, sys
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"
F, D = int(sys.argv[1]), int(sys.argv[2])
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from omegalax.data.pretrain_data_set import BATCH_PRETRAIN_METADATA_KEY
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.trainers import pretrain as P
cfg = Qwen3_5TextConfig(
    vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
    num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6, layer_types=("linear_attention",),
    rope_theta=10_000, partial_rotary_factor=0.25, mrope_section=(1, 1, 0),
    linear_conv_kernel_dim=3, linear_key_head_dim=4, linear_num_key_heads=1,
    linear_num_value_heads=2, linear_value_head_dim=4, intermediate_size=32, dtype=jnp.float32,
)
rng = np.random.RandomState(0)
tok = rng.randint(1, 63, size=(4, 2, 4)).astype(np.int32)
attn = np.ones((4, 2, 4), np.int32)
attn[1, 1, 2:] = 0
attn[3, 1, :] = 0
attn[3, 0, 3:] = 0
tok = tok * attn
reset = np.zeros((4, 2), bool)
reset[:, 0] = True
batch = {
    "token_ids_BCT": tok, "attention_mask_BCT": attn, "loss_mask_BCT": attn.copy(),
    "chunk_idx_BC": np.tile(np.arange(2, dtype=np.int32), (4, 1)), "reset_state_BC": reset,
    BATCH_PRETRAIN_METADATA_KEY: {"doc_ids": list("abcd")},
}
def batches():
    for _ in range(4):
        yield {k: (v.copy() if hasattr(v, "copy") else v) for k, v in batch.items()}
tc = P.TrainConfig(seed=0, batch_size=8, seq_len=4, num_steps=2, learning_rate=1e-3,
                   weight_decay=0.0, print_every=0)
opt, m = P.run_pretrain(cfg, tc, batches(), pretrain_mode=P.PretrainMode.STATEPASSING_BPTT,
                        log_every=0, tp_size=1, fsdp_size=F, dp_size=D)
leaves = [np.asarray(x, np.float64) for x in jax.tree.leaves(nnx.state(opt.model, nnx.Param))]
out = {k: float(v) for k, v in m.items() if k == "nll" or k.endswith("_nll") or k.endswith("_tokens")}
out["devices"] = jax.device_count()
out["param_sum"] = float(sum(x.sum() for x in leaves))
out["param_sumsq"] = float(sum((x ** 2).sum() for x in leaves))
print("RESULT " + json.dumps(out))
"""


class TrainerInvariantsTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self._usr1 = signal.getsignal(signal.SIGUSR1)
        self._term = signal.getsignal(signal.SIGTERM)

    def tearDown(self):
        signal.signal(signal.SIGUSR1, self._usr1)
        signal.signal(signal.SIGTERM, self._term)
        super().tearDown()

    # Pins: nnx.scan path equals the unrolled segment loop (value, metrics, gradients, finite)
    # with fully padded trailing segments and with GDN state off / conv state on.
    def test_scan_matches_unrolled_loop_with_padded_tail_segments(self):
        if not flags.FLAGS.is_parsed():
            flags.FLAGS.mark_as_parsed()
        cfg = _three_layer_config()
        configs = (
            (4, True, False, 2),
            (4, True, True, 2),
            (4, False, True, 2),
            (4, False, True, 1),
            (4, False, False, 1),
            (4, True, True, 1),
            (2, False, True, 1),
            (2, True, True, 1),
        )
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            for C, pass_gdn, pass_conv, padded in configs:
                with self.subTest(C=C, pass_gdn=pass_gdn, pass_conv=pass_conv, padded=padded):
                    batch = _scan_batch(C, 6, padded, seed=C)
                    kw = dict(pass_gdn_state=pass_gdn, pass_conv_state=pass_conv)
                    scan_loss, scan_m, scan_g = _scan_value_and_grad(
                        cfg, batch, gdn_layer_limit=None, **kw
                    )
                    loop_loss, loop_m, loop_g = _scan_value_and_grad(
                        cfg, batch, gdn_layer_limit=2, **kw
                    )
                    self.assertTrue(np.isfinite(scan_loss))
                    self.assertLessEqual(abs(scan_loss - loop_loss), 1e-6 * abs(loop_loss))
                    self.assertEqual(scan_m.keys(), loop_m.keys())
                    for key in scan_m:
                        np.testing.assert_allclose(
                            scan_m[key], loop_m[key], rtol=1e-5, atol=1e-6, err_msg=key
                        )
                    for a, b in zip(jax.tree.leaves(scan_g), jax.tree.leaves(loop_g), strict=True):
                        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
                        self.assertTrue(np.all(np.isfinite(a)))
                        self.assertTrue(np.all(np.isfinite(b)))
                        self.assertLessEqual(
                            float(np.linalg.norm(a - b)), 1e-4 * float(np.linalg.norm(b)) + 1e-7
                        )

    # Pins: signal checkpoint + resume with grad accumulation on the scan path equals the
    # uninterrupted run, bit-identically, and consumes the same batch sequence.
    def test_resume_with_accumulation_on_scan_path_matches_uninterrupted(self):
        cfg = _tiny_qwen3_5_config()
        train_cfg = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=4,
            num_steps=4,
            lr_schedule_steps=4,
            grad_accum_steps=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            warmup_steps=1,
            lr_schedule="cosine",
            print_every=0,
        )
        run_kwargs = dict(
            pretrain_mode=BPTT,
            save_every=0,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
            bptt_chunks=None,
            pass_conv_state=True,
            pass_rope_positions=False,
        )
        consumed = []
        real_prepare = pretrain_trainer.prepare_pretrain_batch

        def record_prepare(raw_batch, *args, **kwargs):
            consumed.append(
                np.asarray(raw_batch[BATCH_PRETRAIN_METADATA_KEY]["window_idx_B"]).tolist()
            )
            return real_prepare(raw_batch, *args, **kwargs)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            index = _build_index(tmp, "train", 120, num_segments=2, seed=1)
            with mock.patch.object(
                pretrain_trainer, "prepare_pretrain_batch", side_effect=record_prepare
            ):
                uninterrupted, uninterrupted_metrics = pretrain_trainer.run_pretrain(
                    cfg, train_cfg, _make_iter(index, 2, shuffle=True), **run_kwargs
                )
                uninterrupted_windows = list(consumed)
                consumed.clear()
                install_patch, requeue_patch = _signal_patches()
                with install_patch, requeue_patch as request_requeue:
                    pretrain_trainer.run_pretrain(
                        cfg,
                        train_cfg,
                        _make_iter(index, 2, shuffle=True),
                        save_dir=tmp / "ckpt",
                        resume=checkpoint_utils.ResumeMode.NEVER,
                        **run_kwargs,
                    )
                request_requeue.assert_called_once_with(True)
                self.assertTrue((tmp / "ckpt" / "000001").is_dir())
                resumed, resumed_metrics = pretrain_trainer.run_pretrain(
                    cfg,
                    train_cfg,
                    _make_iter(index, 2, shuffle=True),
                    save_dir=tmp / "ckpt",
                    resume=checkpoint_utils.ResumeMode.REQUIRED,
                    **run_kwargs,
                )
                interrupted_windows = list(consumed)

        self.assertLen(uninterrupted_windows, 8)
        self.assertLen(set(map(tuple, uninterrupted_windows)), 8)
        self.assertEqual(interrupted_windows, uninterrupted_windows)
        self.assertEqual(uninterrupted_metrics["step"], 4)
        self.assertEqual(resumed_metrics["step"], 4)
        _assert_states_equal(self, uninterrupted, resumed)
        counters = _opt_counters(resumed)
        self.assertEqual(counters["['opt_state']['mini_step'].value"], 0)
        self.assertEqual(counters["['opt_state']['gradient_step'].value"], 4)

    # Pins: logged train/lr equals the schedule at the optimizer-update index under accumulation,
    # also after a resume that extends the stop step; the stored horizon is enforced.
    def test_logged_lr_equals_applied_lr_with_accumulation_across_resume(self):
        cfg = _tiny_qwen3_5_config()
        train_cfg = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=4,
            num_steps=3,
            lr_schedule_steps=6,
            grad_accum_steps=3,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_steps=2,
            lr_schedule="cosine",
            lr_end_factor=0.1,
            print_every=0,
        )
        run_kwargs = dict(
            pretrain_mode=BPTT, save_every=1, log_every=1, tp_size=1, fsdp_size=1, dp_size=1
        )
        recorder = _WandbRecorder()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            index = _build_index(tmp, "train", 200, num_segments=2, seed=1)
            pretrain_trainer.run_pretrain(
                cfg,
                train_cfg,
                _make_iter(index, 2, shuffle=True),
                save_dir=tmp / "ckpt",
                wandb_run=recorder,
                **run_kwargs,
            )
            optimizer, _ = pretrain_trainer.run_pretrain(
                cfg,
                dataclasses.replace(train_cfg, num_steps=6),
                _make_iter(index, 2, shuffle=True),
                save_dir=tmp / "ckpt",
                resume=checkpoint_utils.ResumeMode.REQUIRED,
                lr_contract_explicit_fields=(),
                wandb_run=recorder,
                **run_kwargs,
            )
            with self.assertRaises(ValueError):
                pretrain_trainer.run_pretrain(
                    cfg,
                    dataclasses.replace(train_cfg, num_steps=7),
                    _make_iter(index, 2, shuffle=True),
                    save_dir=tmp / "ckpt",
                    resume=checkpoint_utils.ResumeMode.REQUIRED,
                    lr_contract_explicit_fields=(),
                    **run_kwargs,
                )

        schedule = build_lr_schedule(
            peak_lr=1e-3, num_steps=6, warmup_steps=2, schedule="cosine", end_factor=0.1
        )
        logged = recorder.series("train/lr")
        self.assertEqual(sorted(logged), [1, 2, 3, 4, 5, 6])
        for step in range(1, 7):
            self.assertAlmostEqual(logged[step], float(schedule(step - 1)), places=9)
        counters = _opt_counters(optimizer)
        self.assertEqual(counters.pop("['opt_state']['mini_step'].value"), 0)
        self.assertEqual(counters.pop("['opt_state']['gradient_step'].value"), 6)
        self.assertNotEmpty(counters)
        self.assertEqual(set(counters.values()), {6})

    # Pins: chunked CE (single and multi mask) equals an independent full-logits optax reference
    # for tiling with padding, several masks, float32 and bfloat16, including d/d hidden.
    def test_chunked_cross_entropy_matches_full_logits_reference(self):
        rng = np.random.RandomState(0)
        B, T, D, V = 3, 23, 16, 97
        hidden32 = jnp.asarray(rng.randn(B, T, D), jnp.float32)
        kernel32 = jnp.asarray(rng.randn(D, V) / 4.0, jnp.float32)
        targets = jnp.asarray(rng.randint(0, V, size=(B, T)), jnp.int32)
        masks = jnp.asarray(rng.randint(0, 2, size=(3, B, T)), jnp.int32)
        weights = jnp.asarray([1.0, 2.0, 3.0], jnp.float32)

        def reference(hidden, kernel):
            logits = jnp.einsum("BTD,DV->BTV", hidden[:, :-1], kernel).astype(jnp.float32)
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets[:, 1:])
            mask_f = masks[:, :, 1:].astype(jnp.float32)
            return jnp.sum(mask_f * nll[None], axis=(1, 2)), jnp.sum(mask_f, axis=(1, 2))

        ref_grad = jax.grad(lambda h: jnp.dot(reference(h, kernel32)[0], weights))(hidden32)
        for dtype in (jnp.float32, jnp.bfloat16):
            hidden, kernel = hidden32.astype(dtype), kernel32.astype(dtype)
            ref_sums, ref_counts = reference(hidden, kernel)
            for num_tiles in (1, 4, 8):
                with self.subTest(dtype=jnp.dtype(dtype).name, num_tiles=num_tiles):
                    sums, counts = chunked_cross_entropy_multi_stats(
                        hidden, kernel, targets, masks, num_tiles=num_tiles
                    )
                    np.testing.assert_allclose(sums, ref_sums, rtol=1e-5)
                    np.testing.assert_array_equal(counts, ref_counts)
                    for n in range(masks.shape[0]):
                        s, c = chunked_cross_entropy_stats(
                            hidden, kernel, targets, masks[n], num_tiles=num_tiles
                        )
                        np.testing.assert_allclose(s, ref_sums[n], rtol=1e-5)
                        self.assertEqual(float(c), float(ref_counts[n]))
                    if dtype == jnp.float32:
                        grad = jax.grad(
                            lambda h: jnp.dot(
                                chunked_cross_entropy_multi_stats(
                                    h, kernel, targets, masks, num_tiles=num_tiles
                                )[0],
                                weights,
                            )
                        )(hidden)
                        np.testing.assert_allclose(grad, ref_grad, rtol=1e-4, atol=1e-6)
                        grad_single = jax.grad(
                            lambda h: chunked_cross_entropy_stats(
                                h, kernel, targets, masks[0], num_tiles=num_tiles
                            )[0]
                        )(hidden)
                        ref_single = jax.grad(lambda h: reference(h, kernel)[0][0])(hidden)
                        np.testing.assert_allclose(grad_single, ref_single, rtol=1e-4, atol=1e-6)

    # Pins: a 2-step run on 4 emulated devices (dp=4, fsdp=2 x dp=2) equals the 1-device run.
    def test_multi_device_step_matches_single_device(self):
        meshes = ((1, 1, 1), (4, 1, 4), (4, 2, 2))
        procs = {}
        for devices, F, D in meshes:
            env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(REPO_ROOT))
            env.pop("XLA_FLAGS", None)
            if devices > 1:
                env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
            procs[(devices, F, D)] = subprocess.Popen(
                [sys.executable, "-c", _T9_SCRIPT, str(F), str(D)],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        results = {}
        for key, proc in procs.items():
            stdout, stderr = proc.communicate(timeout=900)
            self.assertEqual(proc.returncode, 0, msg=stderr[-3000:])
            (line,) = [row for row in stdout.splitlines() if row.startswith("RESULT ")]
            results[key] = json.loads(line[len("RESULT ") :])
        base = results.pop((1, 1, 1))
        self.assertEqual(base["devices"], 1)
        self.assertEqual(base["supervised_tokens"], 21.0)
        self.assertIn("segment1_nll", base)
        for key, result in results.items():
            with self.subTest(mesh=key):
                self.assertEqual(result.pop("devices"), 4)
                self.assertEqual(result.keys(), base.keys() - {"devices"})
                for name, value in result.items():
                    np.testing.assert_allclose(value, base[name], rtol=1e-5, err_msg=name)

    # Pins: with equal supervised-token counts, two accumulated microbatches give the same
    # update and the same logged NLL as one batch of twice the size.
    def test_accumulation_equals_one_batch_with_equal_token_counts(self):
        update_accum, update_one, metrics_accum, metrics_one = _accumulation_vs_one_batch(False)
        self.assertLess(_rel(update_accum, update_one), 1e-5)
        self.assertEqual(metrics_accum.keys(), metrics_one.keys())
        for key in metrics_one:
            np.testing.assert_allclose(metrics_accum[key], metrics_one[key], rtol=1e-5, err_msg=key)

    # Pins: with unequal token counts the logged NLL, per-segment NLL and token counts of the
    # accumulated window stay token-weighted, i.e. equal those of the one-batch step.
    def test_accumulated_logged_nll_is_token_weighted_with_unequal_counts(self):
        _, _, metrics_accum, metrics_one = _accumulation_vs_one_batch(True)
        self.assertEqual(metrics_one["supervised_tokens"], 35.0)
        self.assertEqual(metrics_accum.keys(), metrics_one.keys())
        for key in metrics_one:
            if key == "nll" or key.endswith(("_nll", "_tokens", "_sum")):
                np.testing.assert_allclose(
                    metrics_accum[key], metrics_one[key], rtol=1e-5, err_msg=key
                )

    # Pins: with unequal token counts the accumulated update equals the one-batch update.
    # Defect: each microbatch loss is normalised by its own token count and MultiSteps averages
    # the microbatch gradients, so tokens in sparse microbatches get a larger weight.
    @unittest.expectedFailure
    def test_accumulated_update_equals_one_batch_with_unequal_counts(self):
        update_accum, update_one, _, _ = _accumulation_vs_one_batch(True)
        self.assertLess(_rel(update_accum, update_one), 1e-5)

    # Pins: validation evaluates the same val batches every time and after a resume, and a
    # one-epoch val iterator shorter than all validations together does not end the run.
    # Defect: the NTP val iterator is never reset and its position is not checkpointed, so each
    # validation reads the next slice, a resume restarts at the first slice, and it can run dry.
    @unittest.expectedFailure
    def test_validation_reads_same_data_every_time_and_after_resume(self):
        cfg = _tiny_qwen3_5_config()
        train_cfg = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=4,
            num_steps=3,
            learning_rate=0.0,
            weight_decay=0.0,
            print_every=0,
        )
        run_kwargs = dict(
            pretrain_mode=BPTT,
            save_every=1,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
            val_every=1,
            val_steps=2,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            train_index = _build_index(tmp, "train", 96, num_segments=2, seed=1)
            val_index = _build_index(tmp, "val", 64, num_segments=2, seed=2)
            short_val_index = _build_index(tmp, "val", 32, num_segments=2, seed=3)

            uninterrupted = _WandbRecorder()
            pretrain_trainer.run_pretrain(
                cfg,
                train_cfg,
                _make_iter(train_index, 2, shuffle=True),
                val_data_iter=_make_iter(val_index, 2, shuffle=False),
                wandb_run=uninterrupted,
                **run_kwargs,
            )
            val_loss = uninterrupted.series("val/loss")
            self.assertEqual(sorted(val_loss), [1, 2, 3])
            self.assertEqual(val_loss[2], val_loss[1])
            self.assertEqual(val_loss[3], val_loss[1])

            install_patch, requeue_patch = _signal_patches()
            with install_patch, requeue_patch:
                pretrain_trainer.run_pretrain(
                    cfg,
                    train_cfg,
                    _make_iter(train_index, 2, shuffle=True),
                    save_dir=tmp / "ckpt",
                    val_data_iter=_make_iter(val_index, 2, shuffle=False),
                    wandb_run=_WandbRecorder(),
                    **run_kwargs,
                )
            resumed = _WandbRecorder()
            pretrain_trainer.run_pretrain(
                cfg,
                train_cfg,
                _make_iter(train_index, 2, shuffle=True),
                save_dir=tmp / "ckpt",
                resume=checkpoint_utils.ResumeMode.REQUIRED,
                val_data_iter=_make_iter(val_index, 2, shuffle=False),
                wandb_run=resumed,
                **run_kwargs,
            )
            resumed_val = resumed.series("val/loss")
            self.assertEqual(resumed_val, {2: val_loss[2], 3: val_loss[3]})

            pretrain_trainer.run_pretrain(
                cfg,
                train_cfg,
                _make_iter(train_index, 2, shuffle=True),
                val_data_iter=_make_iter(short_val_index, 2, shuffle=False),
                **run_kwargs,
            )

    # Pins: "no BPTT" spelled as the mode default or as an explicit span supervises the same
    # targets, and a mode that contradicts the explicit span is rejected.
    # Defect: the target masks receive the unresolved bptt_chunks=None, so the mode default keeps
    # the boundary targets that an explicit span of 0 or 1 drops; the mode is only a default.
    @unittest.expectedFailure
    def test_both_spellings_of_no_bptt_supervise_same_targets(self):
        cfg = _tiny_qwen3_5_config()
        tok = (np.arange(24).reshape(2, 3, 4) % 60) + 1
        batch = _sp_batch(tok, np.ones_like(tok))
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model, _ = text_api.init_model(
                cfg, jax.random.PRNGKey(0), tp_size=1, fsdp_size=1, dp_size=1
            )

            def stats(mode, bptt_chunks):
                loss, metrics = pretrain_trainer._statepassing_loss_stats(
                    model, batch, cfg, 0, mode, bptt_chunks=bptt_chunks
                )
                return (
                    float(metrics["supervised_tokens"]),
                    float(metrics["boundary_tokens"]),
                    float(loss),
                )

            for group in (
                ((NO_BPTT, None), (BPTT, 0), (BPTT, 1)),
                ((BPTT, None), (BPTT, 3)),
            ):
                reference = stats(*group[0])
                for spelling in group[1:]:
                    np.testing.assert_allclose(stats(*spelling), reference, rtol=1e-6)
            with self.assertRaises(ValueError):
                pretrain_trainer.make_pretrain_train_step(NO_BPTT, cfg, bptt_chunks=3)

    def _run_curriculum(self, index, phase_steps, num_steps, save_dir, resume):
        train_cfg = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=6,
            seq_len=4,
            num_steps=num_steps,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )
        return pretrain_trainer.run_statepassing_curriculum(
            _tiny_qwen3_5_config(),
            train_cfg,
            lambda c: _make_iter(index[int(c)], 6, shuffle=True, seed=3),
            train_order=[2, 3],
            phase_steps=phase_steps,
            pretrain_mode=BPTT,
            save_dir=save_dir,
            save_every=0,
            log_every=0,
            resume=resume,
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )

    def _curriculum_resume_matches(self, phase_steps, num_steps):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            data_root = tmp / "docs"
            _write_doc(data_root, "train", 240, seed=1)
            index = {
                c: build_statepassing_window_index(
                    data_root,
                    tmp / f"index_c{c}",
                    chunk_length=4,
                    num_segments=c,
                    split="train",
                    records_per_shard=16,
                )
                for c in (2, 3)
            }
            control, control_metrics = self._run_curriculum(
                index, phase_steps, num_steps, tmp / "control", checkpoint_utils.ResumeMode.NEVER
            )
            install_patch, requeue_patch = _signal_patches()
            with install_patch, requeue_patch as request_requeue:
                self._run_curriculum(
                    index, phase_steps, num_steps, tmp / "signal", checkpoint_utils.ResumeMode.NEVER
                )
            request_requeue.assert_called_once_with(True)
            resumed, resumed_metrics = self._run_curriculum(
                index,
                phase_steps,
                num_steps,
                tmp / "signal",
                checkpoint_utils.ResumeMode.REQUIRED,
            )
        for key in ("step", "phase_C", "phase_step"):
            self.assertEqual(resumed_metrics[key], control_metrics[key], msg=key)
        self.assertEqual(control_metrics["step"], num_steps)
        _assert_states_equal(self, control, resumed)

    # Pins: curriculum resume with real checkpoints restores phase, phase step, iterator and
    # optimizer when the signal lands in the middle of a phase.
    def test_curriculum_resume_mid_phase_matches_uninterrupted(self):
        self._curriculum_resume_matches({2: 2, 3: 2}, num_steps=4)

    # Pins: a signal on the last step of a curriculum phase is resumable and matches.
    # Defect: the resume writes the phase-boundary checkpoint again at the already saved step,
    # which orbax rejects, so every resume of that run fails.
    @unittest.expectedFailure
    def test_curriculum_resume_after_signal_on_last_step_of_phase(self):
        self._curriculum_resume_matches({2: 1, 3: 2}, num_steps=3)

    # Pins: a non-finite loss stops the run with FloatingPointError before a non-finite
    # checkpoint is written.
    # Defect: outside the common-plan branch the loop has no finiteness check, so NaN updates
    # are applied, logged and checkpointed until the last finite checkpoint rotates out.
    @unittest.expectedFailure
    def test_nonfinite_loss_stops_run_before_next_checkpoint(self):
        cfg = _tiny_qwen3_5_config()
        train_cfg = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=4,
            num_steps=4,
            learning_rate=1e30,
            weight_decay=0.0,
            print_every=0,
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            index = _build_index(tmp, "train", 96, num_segments=2, seed=1)
            try:
                with self.assertRaises(FloatingPointError):
                    pretrain_trainer.run_pretrain(
                        cfg,
                        train_cfg,
                        _make_iter(index, 2, shuffle=True),
                        pretrain_mode=BPTT,
                        save_dir=tmp / "ckpt",
                        save_every=1,
                        keep_latest=2,
                        log_every=0,
                        tp_size=1,
                        fsdp_size=1,
                        dp_size=1,
                    )
            finally:
                checkpoints = [int(p.name) for p in (tmp / "ckpt").iterdir() if p.name.isdigit()]
        self.assertLessEqual(max(checkpoints, default=0), 1)


if __name__ == "__main__":
    absltest.main()
