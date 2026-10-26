"""Teacher-distillation training of a real tiny Qwen3.5 through the production trainer.

Every test reads batches from a real rollout-teacher index built by the production
builder over a fixture root written with the real record serializer, and runs the real
model, blockwise KL loss, optimizer, checkpoint and CLI paths on CPU.
"""
import os
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

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
from absl.testing import absltest
from flax import nnx

import omegalax
from omegalax.data import rollout_teacher as RT
from omegalax.data.rollout_teacher_pipeline import (
    TEACHER_BATCH_KEYS,
    build_rollout_teacher_index,
    load_rollout_teacher_index,
    make_rollout_teacher_iterator,
)
from omegalax.distributed.mesh import ensure_mesh
from omegalax.export import model_config_to_hf_dict
from omegalax.models.params_utils import save_hf_config
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.shard_config import ShardConfig
from omegalax.text import api as text_api
from omegalax.trainers import checkpoint_utils
from omegalax.trainers import pretrain as pretrain_trainer
from omegalax.trainers.text import _model_output_weight
from tests.test_rollout_teacher_pipeline import BUCKETS, SAMPLES, T, write_fixture_root

REPO = Path(__file__).resolve().parents[1]
assert Path(omegalax.__file__).resolve() == REPO / "omegalax/__init__.py", omegalax.__file__
TEACHER = pretrain_trainer.TrainingObjective.TEACHER_TOP512_SPECIAL_TAIL_KL
MODE = pretrain_trainer.PretrainMode.STATEPASSING_BPTT
STATE_FLAGS = dict(pass_gdn_state=True, pass_conv_state=True, pass_rope_positions=True)
# Global segment batch divisible by every fixture bucket C in (1, 2, 4): B = 4, 2, 1 chains.
SEGMENT_BATCH = 4
# Fixture targets per batch in unshuffled bucket order: [a,i,j,pad], [b,k], [c], [d].
UNSHUFFLED_TRAIN_TARGETS = [14, 20, 10, 27]
VAL_TARGETS = 10


def _tiny_teacher_config(shd_cfg: ShardConfig | None = None) -> Qwen3_5TextConfig:
    # GDN-only, so the segmented statepassing forward equals one unsegmented forward.
    return Qwen3_5TextConfig(
        vocab_size=RT.VOCAB_SIZE,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        layer_types=("linear_attention", "linear_attention"),
        rope_theta=10_000,
        partial_rotary_factor=0.25,
        mrope_section=(1, 1, 0),
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        intermediate_size=32,
        shd_cfg=shd_cfg or ShardConfig.no_sharding(),
        dtype=jnp.float32,
    )


def _train_config(**overrides) -> pretrain_trainer.TrainConfig:
    values = dict(
        seed=0,
        batch_size=SEGMENT_BATCH,
        seq_len=T,
        num_steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=0.0,
        print_every=0,
    )
    values.update(overrides)
    return pretrain_trainer.TrainConfig(**values)


def _iterator(index, split="train", *, shuffle, num_epochs=None, seed=0, batch_size=SEGMENT_BATCH):
    return make_rollout_teacher_iterator(
        index,
        split=split,
        selection="graded",
        segment_length=T,
        segment_buckets=BUCKETS,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        dp_size=1,
        fsdp_size=1,
        process_index=0,
        grain_workers=0,
        grain_read_threads=1,
        grain_read_prefetch_buffer_size=1,
    )


def _run_kwargs(**overrides) -> dict:
    values = dict(
        pretrain_mode=MODE,
        training_objective=TEACHER,
        bptt_chunks=1,
        log_every=0,
        tp_size=1,
        fsdp_size=1,
        dp_size=1,
        text_attn_backend="xla",
        teacher_segment_buckets=BUCKETS,
        **STATE_FLAGS,
    )
    values.update(overrides)
    return values


def _flat(tree) -> dict[str, np.ndarray]:
    return {
        jax.tree_util.keystr(path): np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


class _Recorder:
    """Stand-in W&B run that keeps the logged dictionaries."""

    def __init__(self):
        self.logs: list[tuple[int, dict]] = []

    def log(self, values, step):
        self.logs.append((int(step), dict(values)))

    def values(self, key):
        return {step: values[key] for step, values in self.logs if key in values}


def _recording_prepare(records: list):
    real_prepare = pretrain_trainer.prepare_pretrain_batch

    def prepare(raw_batch, *args, **kwargs):
        batch, metadata, debug = real_prepare(raw_batch, *args, **kwargs)
        records.append(
            (
                metadata["split"],
                metadata["global_batch_idx"],
                tuple(metadata["sequence_ids"]),
                metadata["teacher_target_positions_BR"].tolist(),
            )
        )
        return batch, metadata, debug

    return mock.patch.object(pretrain_trainer, "prepare_pretrain_batch", side_effect=prepare)


def _first_update_gradient(optimizer) -> dict[str, np.ndarray]:
    """Gradient of the first update from Adam's first moment (1 - beta1) * g; no clipping."""
    return {
        key.split("['mu']", 1)[1]: value / (1.0 - 0.9)
        for key, value in _flat(nnx.state(optimizer)).items()
        if "['inner_opt_state'][0][0]['mu']" in key
    }


def _dp_worker(index_path: str, out_dir: str, dp_size: int) -> None:
    """One accumulated update on dp_size CPU devices; runs in a fresh process."""
    assert jax.device_count() == dp_size, jax.devices()
    optimizer, metrics = pretrain_trainer.run_pretrain(
        _tiny_teacher_config(ShardConfig.default()),
        _train_config(batch_size=8, grad_accum_steps=2),
        _iterator(index_path, shuffle=False, batch_size=8),
        **_run_kwargs(dp_size=dp_size),
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "gradient.npz", **_first_update_gradient(optimizer))
    (out / "metrics.json").write_text(json.dumps(metrics))


def _bucket_sharding_worker(index_path: str, out_path: str) -> None:
    """Starts training on dp=4 CPU devices; runs in a fresh process."""
    assert jax.device_count() == 4, jax.devices()
    iterator = _iterator(index_path, shuffle=False)
    initial_state = iterator.get_state()
    try:
        pretrain_trainer.run_pretrain(
            _tiny_teacher_config(ShardConfig.default()),
            _train_config(),
            iterator,
            **_run_kwargs(dp_size=4),
        )
        error = None
    except ValueError as exc:
        error = str(exc)
    Path(out_path).write_text(
        json.dumps({"error": error, "iterator_unread": iterator.get_state() == initial_state})
    )


class RolloutTeacherTrainingTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("TMPDIR"), ignore_cleanup_errors=True
        )
        cls.tmp = Path(cls._tmp.name)
        write_fixture_root(cls.tmp / "rollout_teacher_v1", SAMPLES)
        cls.index = build_rollout_teacher_index(
            cls.tmp / "rollout_teacher_v1",
            cls.tmp / "index",
            selection="graded",
            segment_length=T,
            segment_buckets=BUCKETS,
        )
        metadata = load_rollout_teacher_index(
            cls.index, selection="graded", segment_length=T, segment_buckets=BUCKETS
        )
        assert metadata["splits"]["val"]["num_targets"] == VAL_TARGETS
        assert metadata["splits"]["train"]["num_targets"] == sum(UNSHUFFLED_TRAIN_TARGETS)
        cls.val_samples = metadata["splits"]["val"]["num_samples"]
        cls.model_dir = cls.tmp / "tiny_model"
        cls.model_dir.mkdir()
        save_hf_config(model_config_to_hf_dict(_tiny_teacher_config()), cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_prepare_keeps_teacher_rows_and_checks_global_segment_batch(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        cfg = text_api.align_config_to_mesh(_tiny_teacher_config(), mesh)
        raw = next(_iterator(self.index, shuffle=False))
        batch, metadata, _ = pretrain_trainer.prepare_pretrain_batch(
            dict(raw),
            MODE,
            cfg,
            mesh,
            global_batch_size=SEGMENT_BATCH,
            pass_rope_positions=True,
            training_objective=TEACHER,
        )
        self.assertEqual(
            sorted(batch),
            sorted(
                ("token_ids_BCT", "attention_mask_BCT", "reset_state_BC", "chunk_idx_BC")
                + TEACHER_BATCH_KEYS
            ),
        )
        for key in TEACHER_BATCH_KEYS:
            np.testing.assert_array_equal(np.asarray(batch[key]), raw[key], err_msg=key)
        self.assertEqual(metadata["num_segments"], 1)

        with self.assertRaisesRegex(ValueError, "expected global segment batch 8"):
            pretrain_trainer.prepare_pretrain_batch(
                dict(raw), MODE, cfg, mesh, global_batch_size=8, training_objective=TEACHER
            )
        missing = {key: value for key, value in raw.items() if key != "teacher_logz_tail_BR"}
        with self.assertRaisesRegex(KeyError, "teacher_logz_tail_BR"):
            pretrain_trainer.prepare_pretrain_batch(
                missing, MODE, cfg, mesh, global_batch_size=SEGMENT_BATCH, training_objective=TEACHER
            )
        with self.assertRaisesRegex(ValueError, "statepassing"):
            pretrain_trainer.prepare_pretrain_batch(
                dict(raw),
                pretrain_trainer.PretrainMode.IID_BASELINE,
                cfg,
                mesh,
                global_batch_size=SEGMENT_BATCH,
                training_objective=TEACHER,
            )

    def test_eval_kl_matches_unsegmented_reference_with_boundary_targets(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        model, cfg = text_api.init_model(
            _tiny_teacher_config(), jax.random.key(3), tp_size=1, fsdp_size=1, dp_size=1
        )
        # bptt_chunks=1 cuts gradients at every segment handoff; targets must stay.
        eval_step = pretrain_trainer.make_pretrain_eval_step(
            MODE, cfg, bptt_chunks=1, training_objective=TEACHER, **STATE_FLAGS
        )
        seen_boundary = False
        raw_batches = list(_iterator(self.index, shuffle=False, num_epochs=1)) + list(
            _iterator(self.index, "val", shuffle=False, num_epochs=1)
        )
        self.assertLen(raw_batches, 6)
        for raw in raw_batches:
            arrays = {key: value for key, value in raw.items() if key != "metadata"}
            batch, metadata, _ = pretrain_trainer.prepare_pretrain_batch(
                dict(raw),
                MODE,
                cfg,
                mesh,
                global_batch_size=SEGMENT_BATCH,
                pass_rope_positions=True,
                training_objective=TEACHER,
            )
            metrics = eval_step(model, batch)

            B, C, _ = arrays["token_ids_BCT"].shape
            hidden_BTD, _ = text_api.forward(
                model,
                jnp.asarray(arrays["token_ids_BCT"].reshape(B, C * T)),
                0,
                cfg,
                attention_mask_BT=jnp.asarray(arrays["attention_mask_BCT"].reshape(B, C * T)),
            )
            logits_BTV = jnp.einsum(
                "BTD,DV->BTV", hidden_BTD, _model_output_weight(model)
            ).astype(jnp.float32)
            reference = 0.0
            for row, count in enumerate(metadata["target_count_B"]):
                positions = metadata["teacher_target_positions_BR"][row, :count]
                seen_boundary |= bool(np.any((positions % T == 0) & (positions > 0)))
                kl_rows = RT.top512_special_tail_kl(
                    logits_BTV[row, positions - 1],
                    {
                        "top512_ids": arrays["teacher_top512_ids_BRK"][row, :count],
                        "top512_logits": arrays["teacher_top512_logits_BRK"][row, :count],
                        "special_logits": arrays["teacher_special_logits_BRS"][row, :count],
                        "logz_all": arrays["teacher_logz_all_BR"][row, :count],
                        "logz_tail": arrays["teacher_logz_tail_BR"][row, :count],
                    },
                )
                reference += float(jnp.sum(kl_rows))

            self.assertEqual(float(metrics["teacher_targets"]), float(metadata["target_count_B"].sum()))
            self.assertEqual(float(metrics["context_tokens"]), float(metadata["token_count_B"].sum()))
            self.assertGreater(abs(reference), 1e-2)
            np.testing.assert_allclose(float(metrics["teacher_kl_sum"]), reference, rtol=2e-5)
        self.assertTrue(seen_boundary)

    def test_accumulated_update_weights_microbatches_by_update_target_count(self):
        train_cfg = _train_config(grad_accum_steps=2, adam_beta1=0.9)
        recorder = _Recorder()
        optimizer, metrics = pretrain_trainer.run_pretrain(
            _tiny_teacher_config(),
            train_cfg,
            _iterator(self.index, shuffle=False),
            wandb_run=recorder,
            **_run_kwargs(log_every=1),
        )
        # The first Adam moment after one update is (1 - beta1) * accumulated gradient.
        accumulated = _first_update_gradient(optimizer)

        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        init_rng, _ = jax.random.split(jax.random.key(train_cfg.seed))
        model, cfg = text_api.init_model(
            _tiny_teacher_config(), init_rng, tp_size=1, fsdp_size=1, dp_size=1
        )
        batches = []
        iterator = _iterator(self.index, shuffle=False)
        for raw in (next(iterator), next(iterator)):
            batch, _, _ = pretrain_trainer.prepare_pretrain_batch(
                raw,
                MODE,
                cfg,
                mesh,
                global_batch_size=SEGMENT_BATCH,
                pass_rope_positions=True,
                training_objective=TEACHER,
            )
            batches.append(batch)

        def stats(model):
            return [
                pretrain_trainer._teacher_loss_stats(
                    model, batch, cfg, 0, MODE, bptt_chunks=1, **STATE_FLAGS
                )
                for batch in batches
            ]

        def token_mean(model):
            (kl0, _, m0), (kl1, _, m1) = stats(model)
            return (kl0 + kl1) / (m0["teacher_targets"] + m1["teacher_targets"])

        def microbatch_mean(model):
            (kl0, _, m0), (kl1, _, m1) = stats(model)
            return 0.5 * (kl0 / m0["teacher_targets"] + kl1 / m1["teacher_targets"])

        @nnx.jit
        def reference(model):
            (kl0, _, _), (kl1, _, _) = stats(model)
            return (
                kl0 + kl1,
                token_mean(model),
                nnx.grad(token_mean)(model),
                nnx.grad(microbatch_mean)(model),
            )

        kl_sum, kl_mean, expected_grads, naive_grads = reference(model)
        expected, naive = _flat(expected_grads), _flat(naive_grads)
        self.assertEqual(sorted(accumulated), sorted(expected))
        for key, value in expected.items():
            np.testing.assert_allclose(accumulated[key], value, rtol=2e-4, atol=1e-9, err_msg=key)
        lm_head = "['lm_head']['kernel'].value"
        self.assertGreater(
            np.abs(naive[lm_head] - expected[lm_head]).max(), 1e-3 * np.abs(expected[lm_head]).max()
        )

        self.assertEqual(metrics["teacher_targets"], sum(UNSHUFFLED_TRAIN_TARGETS[:2]))
        np.testing.assert_allclose(metrics["teacher_kl"], float(kl_mean), rtol=1e-5)
        np.testing.assert_allclose(metrics["teacher_kl_sum"], float(kl_sum), rtol=1e-5)
        self.assertNotIn("nll", metrics)
        self.assertNotIn("ppl", metrics)
        self.assertEqual(recorder.values("train/teacher_targets"), {1: 34.0})

    def test_tokamax_priming_reuses_first_microbatch_and_update_denominator(self):
        real_make_train_step = pretrain_trainer.make_pretrain_train_step
        calls = []

        def make_spy_train_step(*args, **kwargs):
            real_train_step = real_make_train_step(*args, **kwargs)

            def spy_train_step(optimizer, batch, loss_denominator):
                calls.append((batch, loss_denominator))
                return real_train_step(optimizer, batch, loss_denominator)

            return spy_train_step

        # Tokamax autotuning needs a GPU; only the trainer's priming bookkeeping is real here.
        autotune_result = mock.MagicMock()
        with (
            mock.patch.object(
                pretrain_trainer, "make_pretrain_train_step", side_effect=make_spy_train_step
            ),
            mock.patch.object(pretrain_trainer.tokamax_cache_lib, "try_load", return_value=None),
            mock.patch.object(
                pretrain_trainer.tokamax_cache_lib,
                "autotune_and_save",
                return_value=autotune_result,
            ) as autotune_and_save,
        ):
            _, metrics = pretrain_trainer.run_pretrain(
                _tiny_teacher_config(),
                _train_config(grad_accum_steps=2),
                _iterator(self.index, shuffle=False),
                tokamax_cache_dir=str(self.tmp / "tokamax"),
                **_run_kwargs(),
            )

        *_, tuned_batch, tuned_denominator = autotune_and_save.call_args.args
        self.assertLen(calls, 2)
        self.assertIs(calls[0][0], tuned_batch)
        self.assertIs(calls[0][1], tuned_denominator)
        self.assertIs(calls[1][1], tuned_denominator)
        # Both microbatches of the update share (14 + 20) / grad_accum_steps.
        self.assertEqual(float(tuned_denominator), 17.0)
        self.assertEqual(metrics["teacher_targets"], 34.0)
        autotune_result.__exit__.assert_called_once_with(None, None, None)

    def test_signal_resume_matches_uninterrupted_batches_state_and_validation(self):
        train_cfg = _train_config(
            num_steps=6,
            lr_schedule_steps=6,
            warmup_steps=1,
            lr_schedule="cosine",
            weight_decay=0.01,
            max_grad_norm=1.0,
            grad_accum_steps=2,
        )

        def run(records, recorder, **kwargs):
            with _recording_prepare(records):
                return pretrain_trainer.run_pretrain(
                    _tiny_teacher_config(),
                    train_cfg,
                    _iterator(self.index, shuffle=True, seed=3),
                    val_data_iter=_iterator(self.index, "val", shuffle=False, num_epochs=1),
                    val_every=2,
                    val_steps=10,
                    wandb_run=recorder,
                    **_run_kwargs(log_every=1, save_every=2, **kwargs),
                )

        uninterrupted_records, uninterrupted_log = [], _Recorder()
        uninterrupted_optimizer, uninterrupted_metrics = run(
            uninterrupted_records, uninterrupted_log
        )

        checkpoint_dir = self.tmp / "signal_checkpoint"
        install_handler = pretrain_trainer._install_requeue_signal_handler

        def install_handler_and_request_requeue():
            requeue_requested = install_handler()
            os.kill(os.getpid(), signal.SIGUSR1)
            return requeue_requested

        previous_usr1 = signal.getsignal(signal.SIGUSR1)
        previous_term = signal.getsignal(signal.SIGTERM)
        interrupted_records, interrupted_log = [], _Recorder()
        try:
            with (
                mock.patch.object(
                    pretrain_trainer,
                    "_install_requeue_signal_handler",
                    side_effect=install_handler_and_request_requeue,
                ),
                mock.patch.object(
                    pretrain_trainer, "_request_slurm_requeue_if_primary"
                ) as request_requeue,
            ):
                run(interrupted_records, interrupted_log, save_dir=checkpoint_dir)
            request_requeue.assert_called_once_with(True)
            self.assertTrue((checkpoint_dir / "000001").is_dir())
            # The interrupted update saved after reading both of its microbatches.
            saved_iterator = checkpoint_dir / "000001" / "input_iter" / "process_0-of-1.json"
            self.assertEqual(json.loads(saved_iterator.read_text()), {"next_index": 2})
            resumed_optimizer, resumed_metrics = run(
                interrupted_records,
                interrupted_log,
                save_dir=checkpoint_dir,
                resume=checkpoint_utils.ResumeMode.REQUIRED,
            )
        finally:
            signal.signal(signal.SIGUSR1, previous_usr1)
            signal.signal(signal.SIGTERM, previous_term)

        def train_batches(records):
            return [record for record in records if record[0] == "train"]

        # Same sample and target identities in the same order: 6 updates of 2 microbatches
        # cross two epoch boundaries of the 4-batch train split.
        self.assertLen(train_batches(uninterrupted_records), 12)
        self.assertEqual(train_batches(interrupted_records), train_batches(uninterrupted_records))
        self.assertEqual(
            {record[1] for record in train_batches(uninterrupted_records)}, set(range(12))
        )
        self.assertEqual(resumed_metrics["step"], 6)
        np.testing.assert_allclose(
            resumed_metrics["teacher_kl"], uninterrupted_metrics["teacher_kl"], rtol=1e-6
        )
        for uninterrupted_leaf, resumed_leaf in zip(
            jax.tree.leaves(nnx.state(uninterrupted_optimizer)),
            jax.tree.leaves(nnx.state(resumed_optimizer)),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(uninterrupted_leaf), np.asarray(resumed_leaf))

        # Every validation replays the full finite val split from its start.
        for log in (uninterrupted_log, interrupted_log):
            self.assertEqual(log.values("val/teacher_targets"), {2: 10.0, 4: 10.0, 6: 10.0})
        self.assertEqual(
            interrupted_log.values("val/teacher_kl"), uninterrupted_log.values("val/teacher_kl")
        )
        self.assertNotIn("val/ppl", {key for _, values in uninterrupted_log.logs for key in values})

    def test_two_device_data_parallel_update_matches_single_device(self):
        outputs = {}
        for dp_size in (1, 2):
            out_dir = self.tmp / f"dp{dp_size}"
            env = dict(
                os.environ,
                JAX_PLATFORMS="cpu",
                XLA_FLAGS=f"--xla_force_host_platform_device_count={dp_size}",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from tests.test_rollout_teacher_training import _dp_worker; "
                    "_dp_worker(sys.argv[1], sys.argv[2], int(sys.argv[3]))",
                    str(self.index),
                    str(out_dir),
                    str(dp_size),
                ],
                check=True,
                cwd=REPO,
                env=env,
            )
            with np.load(out_dir / "gradient.npz") as gradient:
                outputs[dp_size] = (
                    {name: gradient[name] for name in gradient.files},
                    json.loads((out_dir / "metrics.json").read_text()),
                )
        (single_gradient, single_metrics), (dp_gradient, dp_metrics) = outputs[1], outputs[2]
        # [a,i,j,5 pads] with C=1 and [b,k,2 pads] with C=2 in one update.
        self.assertEqual(single_metrics["teacher_targets"], 34.0)
        self.assertEqual(dp_metrics["teacher_targets"], 34.0)
        np.testing.assert_allclose(
            dp_metrics["teacher_kl_sum"], single_metrics["teacher_kl_sum"], rtol=1e-5
        )
        # Compare the update gradient itself; Adam's first step g / (|g| + eps) would amplify
        # reduction-order noise on near-zero coordinates.
        self.assertEqual(sorted(dp_gradient), sorted(single_gradient))
        for key, single in single_gradient.items():
            np.testing.assert_allclose(
                dp_gradient[key], single, rtol=1e-4, atol=1e-5 * np.abs(single).max(), err_msg=key
            )

    def test_every_bucket_is_checked_against_the_device_mesh_before_training(self):
        # One loading process, dp=4 devices, 4 global segments: C=1 gives 4 chains but C=2
        # only 2. The unshuffled first batch is C=1, so a per-batch check alone would train it.
        out_path = self.tmp / "bucket_sharding.json"
        env = dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=4",
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from tests.test_rollout_teacher_training import "
                "_bucket_sharding_worker; _bucket_sharding_worker(sys.argv[1], sys.argv[2])",
                str(self.index),
                str(out_path),
            ],
            check=True,
            cwd=REPO,
            env=env,
        )
        result = json.loads(out_path.read_text())
        self.assertEqual(
            result["error"],
            "Global statepassing window batch size 2 must be divisible by batch sharding "
            "multiple 4.",
        )
        self.assertTrue(result["iterator_unread"])

    def _cli_args(self, save_dir: Path, *extra: str) -> list[str]:
        return [
            "--pretrain_mode=statepassing_bptt",
            "--training_objective=teacher_top512_special_tail_kl",
            "--teacher_selection=graded",
            f"--teacher_segment_buckets={','.join(map(str, BUCKETS))}",
            f"--train_index_path={self.index}",
            f"--model_id={self.model_dir}",
            f"--seq_len={T}",
            f"--batch_size={SEGMENT_BATCH}",
            "--lr_schedule_steps=4",
            "--warmup_tokens=0",
            f"--save_dir={save_dir}",
            "--save_every=1",
            "--log_every=1",
            "--val_every=2",
            "--val_steps=4",
            "--bptt_chunks=1",
            "--pass_conv_state",
            "--pass_rope_positions",
            "--grain_workers=0",
            "--grain_read_threads=1",
            "--grain_read_prefetch_buffer_size=1",
            "--tp_size=1",
            "--fsdp_size=1",
            "--dp_size=1",
            "--text_attn_backend=xla",
            f"--jax_cache_dir={self.tmp / 'jax_cache'}",
            *extra,
        ]

    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess:
        with socket.socket() as probe:
            probe.bind(("", 0))
            coordinator_port = probe.getsockname()[1]
        return subprocess.run(
            [sys.executable, "scripts/train_text_pretrain.py", *args],
            cwd=REPO,
            env=dict(os.environ, JAX_COORDINATOR_PORT=str(coordinator_port)),
            text=True,
            capture_output=True,
        )

    def test_cli_trains_resumes_and_rejects_changed_teacher_contract(self):
        save_dir = self.tmp / "cli_checkpoint"

        def train(*extra):
            return self._run_cli(self._cli_args(save_dir, *extra))

        fresh = train("--num_steps=2")
        self.assertEqual(fresh.returncode, 0, fresh.stderr[-4000:])
        self.assertIn("finished step=2 teacher_kl=", fresh.stdout)
        contract = json.loads((save_dir / "training_contract.json").read_text())
        teacher = contract["teacher_distillation"]
        self.assertEqual(teacher["objective"], TEACHER.value)
        self.assertEqual(
            (teacher["selection"], teacher["segment_length"], teacher["segment_buckets"]),
            ("graded", T, BUCKETS),
        )
        self.assertEqual((teacher["bptt_chunks"], teacher["seed"], teacher["batch_size"]), (1, 0, 4))
        self.assertEqual((teacher["grad_accum_steps"], teacher["iterator_data_shards"]), (1, 1))
        self.assertEqual(contract["horizon_by_step"][0]["c_train"], max(BUCKETS))
        # The logged host size equals the arrays of the largest real batch of this layout.
        logged_bytes = int(
            fresh.stdout.split("teacher largest per-process host batch bytes=", 1)[1].split()[0]
        )
        self.assertEqual(
            logged_bytes,
            max(
                sum(
                    value.nbytes
                    for value in (*raw.values(), *raw["metadata"].values())
                    if isinstance(value, np.ndarray)
                )
                for raw in _iterator(self.index, shuffle=False, num_epochs=1)
            ),
        )
        # --val_steps=4 covers both val batches: no partial-validation warning.
        self.assertIn(
            f"teacher val split batches=2 samples={self.val_samples} targets={VAL_TARGETS} "
            "val_steps=4",
            fresh.stdout,
        )
        self.assertNotIn("WARNING", fresh.stdout)

        # val_steps is not part of the contract; a one-batch prefix is reported as partial.
        resumed = train(
            "--num_steps=3", "--resume=required", "--resume_expected_step=2", "--val_steps=1"
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr[-4000:])
        self.assertIn("finished step=3 teacher_kl=", resumed.stdout)
        self.assertIn(
            "WARNING: --val_steps=1 evaluates only the first 1 of 2 val batches.", resumed.stdout
        )
        self.assertIn(
            "NOT whole-split validation; the full val split needs --val_steps=2.", resumed.stdout
        )

        changed_seed = train("--num_steps=4", "--resume=required", "--seed=1")
        self.assertNotEqual(changed_seed.returncode, 0)
        self.assertIn("Training contract conflicts", changed_seed.stderr)
        changed_selection = train("--num_steps=4", "--resume=required", "--teacher_selection=correct")
        self.assertNotEqual(changed_selection.returncode, 0)
        self.assertIn("selection is 'graded'", changed_selection.stderr)
        self.assertEqual(sorted(path.name for path in save_dir.glob("00000*")), ["000001", "000002", "000003"])

    def test_cli_resume_rejects_changed_accumulation(self):
        save_dir = self.tmp / "cli_accumulation"
        initial = self._run_cli(self._cli_args(save_dir, "--num_steps=1", "--grad_accum_steps=2"))
        self.assertEqual(initial.returncode, 0, initial.stderr[-4000:])
        teacher = json.loads((save_dir / "training_contract.json").read_text())[
            "teacher_distillation"
        ]
        self.assertEqual(teacher["grad_accum_steps"], 2)
        # Same explicit LR horizon; MultiSteps state of accumulation 2 and 3 has one schema.
        changed = self._run_cli(
            self._cli_args(
                save_dir,
                "--num_steps=2",
                "--grad_accum_steps=3",
                "--resume=required",
                "--resume_expected_step=1",
            )
        )
        self.assertNotEqual(changed.returncode, 0)
        self.assertIn("Training contract conflicts", changed.stderr)
        self.assertEqual([path.name for path in save_dir.glob("00000*")], ["000001"])

    def test_cli_rejects_unloaded_iterator_shard_and_implicit_grain_workers(self):
        save_dir = self.tmp / "cli_rejected"
        # A second iterator shard has no loading process, so its chains would never be read.
        extra_shard = self._run_cli(
            self._cli_args(save_dir, "--num_steps=1", "--iterator_dp_size=2")
        )
        self.assertNotEqual(extra_shard.returncode, 0)
        self.assertIn(
            "iterator_dp_size=2 x iterator_fsdp_size=1 must equal the 1 loading processes",
            extra_shard.stderr,
        )
        implicit_workers = self._run_cli(
            [
                arg
                for arg in self._cli_args(save_dir, "--num_steps=1")
                if not arg.startswith("--grain_workers=")
            ]
        )
        self.assertNotEqual(implicit_workers.returncode, 0)
        self.assertIn(
            "requires explicit --grain_workers and --grain_read_prefetch_buffer_size",
            implicit_workers.stderr,
        )
        self.assertFalse(save_dir.exists())


if __name__ == "__main__":
    absltest.main()
