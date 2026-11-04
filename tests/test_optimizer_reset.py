"""Tests for ``--init_from`` / ``--reset_optimizer`` and the array-free restore.

Runs on whatever platform JAX is given, so the same suite doubles as the single-GPU smoke
test that a CPU-only run cannot replace:

    JAX_PLATFORMS=cpu pytest tests/test_optimizer_reset.py
    JAX_PLATFORMS=cuda pytest tests/test_optimizer_reset.py
"""

import gc
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

from absl.testing import absltest
import grain
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding

from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.trainers import vlm as vlm_trainer
from omegalax.trainers.checkpoint_utils import ResumeMode
from omegalax.trainers.lr_schedule import build_lr_schedule
from omegalax.vlm import api as vlm_api

_MODEL_ID = "qwen3-vl-smoke"
# 'xla' keeps the suite platform-agnostic (mosaic_gpu has no CPU kernel); the restore and
# sharding behaviour under test is independent of the attention backend.
_ATTN_BACKEND = "xla"
_SEQ_LEN = 4
_VOCAB = 1024
_RUN_KWARGS = dict(save_every=1, keep_latest=2, log_every=0, tp_size=1, dp_size=1)


def _fsdp_size() -> int:
    return 2 if len(jax.devices()) >= 2 else 1


def _run_kwargs() -> dict:
    return dict(_RUN_KWARGS, fsdp_size=_fsdp_size(), text_attn_backend=_ATTN_BACKEND)


def _train_cfg(**overrides) -> vlm_trainer.TrainConfig:
    kwargs = dict(
        seed=0,
        batch_size=_fsdp_size(),
        seq_len=_SEQ_LEN,
        num_steps=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=1,
        max_grad_norm=1.0,
        grad_accum_steps=2,
        print_every=0,
        num_loss_tiles=1,
    )
    kwargs.update(overrides)
    return vlm_trainer.TrainConfig(**kwargs)


def _grain_iter(num_records: int = 16):
    """A checkpointable Grain iterator whose element k is tagged with k."""
    batch_size = _fsdp_size()
    rng = np.random.RandomState(0)
    loss_mask = np.zeros((batch_size, _SEQ_LEN), dtype=np.int32)
    loss_mask[:, _SEQ_LEN // 2 :] = 1
    batch = {
        "token_ids_BT": rng.randint(1, _VOCAB, size=(batch_size, _SEQ_LEN)).astype(np.int32),
        "attention_mask_BT": np.ones((batch_size, _SEQ_LEN), dtype=np.int32),
        "loss_mask_BT": loss_mask,
    }
    records = [
        dict(batch, record_index=np.full((batch_size,), index, dtype=np.int32))
        for index in range(num_records)
    ]
    return iter(grain.MapDataset.source(records).repeat(None).to_iter_dataset())


def _record_index(batch) -> int:
    return int(np.asarray(batch["record_index"]).reshape(-1)[0])


def _build_optimizer(mesh, train_cfg):
    """Build model + optimizer the way ``run_sft`` does, i.e. on mesh shardings."""
    replicated = NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng = jax.device_put(jax.random.key(train_cfg.seed), replicated)
    model_cfg = vlm_api.align_config_to_mesh(vlm_api.resolve_config(_MODEL_ID), mesh)
    model, model_cfg = vlm_api.init_model(
        model_cfg,
        rng,
        tp_size=int(mesh.shape["tp"]),
        fsdp_size=int(mesh.shape["fsdp"]),
        dp_size=int(mesh.shape["dp"]),
    )
    set_attn_backend(model, text_backend=_ATTN_BACKEND)
    lr_schedule_fn = build_lr_schedule(
        peak_lr=train_cfg.learning_rate,
        num_steps=train_cfg.num_steps,
        warmup_steps=train_cfg.warmup_steps,
        schedule=train_cfg.lr_schedule,
        end_factor=train_cfg.lr_end_factor,
        stable_fraction=train_cfg.lr_stable_fraction,
    )
    with mesh_rules(mesh):
        optimizer = vlm_trainer.build_optimizer(model, lr_schedule_fn, train_cfg, wrt=nnx.Param)
    return optimizer, model_cfg, rng


def _train_steps(optimizer, model_cfg, train_cfg, data_iter, num_micro_steps: int) -> None:
    step_fn = vlm_trainer.make_sft_train_step(
        model_cfg, pad_id=0, wrt=nnx.Param, num_loss_tiles=train_cfg.num_loss_tiles
    )
    for _ in range(num_micro_steps):
        batch = dict(next(data_iter))
        batch.pop("record_index", None)
        step_fn(optimizer, {key: jnp.asarray(value) for key, value in batch.items()})


def _reset_leaves(optimizer) -> dict[str, tuple]:
    """(shape, dtype, sharding, nonzero) for every leaf the reset is meant to touch."""
    return {
        jax.tree_util.keystr(path): (
            tuple(leaf.shape),
            leaf.dtype,
            leaf.sharding,
            int(jnp.count_nonzero(leaf)),
        )
        for path, leaf in jax.tree_util.tree_leaves_with_path(nnx.state(optimizer))
        if path[0].key in vlm_trainer._RESET_BRANCHES
    }


class FlagValidationTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.tmpdir = Path(tempfile.mkdtemp())
        self.source = self.tmpdir / "orbax" / "000900"
        (self.source / "train_state").mkdir(parents=True)

    def _problems(self, **overrides) -> list[str]:
        kwargs = dict(
            resume=ResumeMode.NEVER,
            init_from=str(self.source),
            reset_optimizer=True,
            save_dir=str(self.tmpdir / "out"),
        )
        kwargs.update(overrides)
        return vlm_trainer.optimizer_reset_problems(**kwargs)

    def test_valid_combination_has_no_problems(self):
        self.assertEqual(self._problems(), [])

    def test_both_flags_off_is_valid_for_every_resume_mode(self):
        for mode in ("never", "if_present", "required"):
            self.assertEqual(
                vlm_trainer.optimizer_reset_problems(
                    resume=mode, init_from=None, reset_optimizer=False, save_dir="/out"
                ),
                [],
            )

    def test_reset_and_resume_are_mutually_exclusive(self):
        for mode in ("if_present", "required"):
            problems = self._problems(resume=mode)
            self.assertLen(problems, 1)
            self.assertIn("requires resume=never", problems[0])

    def test_reset_requires_init_from(self):
        problems = self._problems(init_from=None)
        self.assertLen(problems, 1)
        self.assertIn("init_from (required", problems[0])

    def test_reset_requires_save_dir(self):
        problems = self._problems(save_dir=None)
        self.assertLen(problems, 1)
        self.assertIn("save_dir (required", problems[0])

    def test_init_from_requires_reset(self):
        self.assertEqual(
            self._problems(reset_optimizer=False), ["init_from requires reset_optimizer"]
        )

    def test_run_root_is_rejected(self):
        problems = self._problems(init_from=str(self.source.parent))
        self.assertLen(problems, 1)
        self.assertIn("must name one checkpoint step directory", problems[0])

    def test_step_directory_without_train_state_is_rejected(self):
        (self.source / "train_state").rmdir()
        problems = self._problems()
        self.assertLen(problems, 1)
        self.assertIn("no train_state item", problems[0])

    def test_init_from_inside_save_dir_is_rejected(self):
        problems = self._problems(save_dir=str(self.source.parent))
        self.assertLen(problems, 1)
        self.assertIn("must lie outside save_dir", problems[0])

    def test_run_sft_raises_before_touching_devices(self):
        with self.assertRaisesRegex(ValueError, "requires resume=never"):
            vlm_trainer.run_sft(
                _MODEL_ID,
                _train_cfg(),
                iter([]),
                save_dir=str(self.tmpdir / "out"),
                resume=ResumeMode.REQUIRED,
                init_from=str(self.source),
                reset_optimizer=True,
            )


class IteratorNextIndexTest(absltest.TestCase):
    def test_worker_count_is_not_a_read_position(self):
        state = {
            "next_index_in_cycle": 0,
            "next_index_in_datasets": 2,
            "iterators_in_use_states": [{"next_index": 0}, {"next_index": 0}],
        }
        self.assertEqual(vlm_trainer._iterator_next_indices(state), [0, 0])

    def test_reports_advanced_positions(self):
        iterator = _grain_iter()
        self.assertEqual(vlm_trainer._iterator_next_indices(iterator.get_state()), [0])
        for _ in range(3):
            next(iterator)
        self.assertEqual(vlm_trainer._iterator_next_indices(iterator.get_state()), [3])


class RestoreSpecTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mesh = ensure_mesh(tp_size=1, fsdp_size=_fsdp_size(), dp_size=1)

    def test_restore_args_pin_the_trainer_sharding_and_leave_dtype_open(self):
        optimizer, _, rng = _build_optimizer(self.mesh, _train_cfg())
        spec = vlm_trainer._restore_spec(optimizer, rng)
        wanted = jax.tree.leaves(
            spec.item, is_leaf=lambda value: isinstance(value, jax.ShapeDtypeStruct)
        )
        got = jax.tree.leaves(
            spec.restore_args, is_leaf=lambda value: hasattr(value, "restore_type")
        )
        self.assertGreater(len(wanted), 0)
        self.assertLen(got, len(wanted))
        for want, arg in zip(wanted, got):
            self.assertIsInstance(arg.sharding, NamedSharding)
            self.assertEqual(arg.sharding, want.sharding)
            # dtype must stay unset, or orbax rounds a trained fp32 moment into a
            # freshly-initialized bf16 leaf.
            self.assertIsNone(arg.dtype)

    def test_a_surviving_alias_keeps_the_fresh_arrays_alive(self):
        optimizer, _, rng = _build_optimizer(self.mesh, _train_cfg())
        spec = vlm_trainer._restore_spec(optimizer, rng)
        leaked_model_alias = optimizer.model
        del optimizer
        gc.collect()
        self.assertGreater(sum(ref() is not None for ref in spec.fresh_arrays), 0)
        del leaked_model_alias
        gc.collect()
        self.assertEqual(sum(ref() is not None for ref in spec.fresh_arrays), 0)


class OptimizerResetTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mesh = ensure_mesh(tp_size=1, fsdp_size=_fsdp_size(), dp_size=1)

    def _trained(self):
        train_cfg = _train_cfg()
        optimizer, model_cfg, _ = _build_optimizer(self.mesh, train_cfg)
        _train_steps(optimizer, model_cfg, train_cfg, _grain_iter(), num_micro_steps=3)
        return optimizer

    def test_zeroing_inherits_sharding_and_dtype(self):
        optimizer = self._trained()
        before = _reset_leaves(optimizer)
        self.assertGreater(sum(nonzero for *_, nonzero in before.values()), 0)

        zeroed, nonzero_before = vlm_trainer._reset_optimizer_state_in_place(optimizer)
        after = _reset_leaves(optimizer)

        self.assertEqual(set(before), set(after))
        self.assertEqual(zeroed, len(after))
        self.assertGreater(nonzero_before, 0)
        partitioned = 0
        for path, (shape, dtype, sharding, nonzero) in after.items():
            self.assertEqual(nonzero, 0, msg=path)
            self.assertEqual(shape, before[path][0], msg=path)
            self.assertEqual(dtype, before[path][1], msg=path)
            # Must be the same mesh NamedSharding, not merely a concrete one.
            self.assertEqual(sharding, before[path][2], msg=path)
            self.assertIsInstance(sharding, NamedSharding, msg=path)
            self.assertEqual(sharding.mesh, self.mesh, msg=path)
            if any(axis is not None for axis in sharding.spec):
                partitioned += 1
        if len(jax.devices()) >= 2:
            self.assertGreater(partitioned, 0, "expected genuinely partitioned reset leaves")

    def test_step_and_optax_counters_reset_to_zero(self):
        optimizer = self._trained()
        scalars = {
            path: nonzero
            for path, (shape, _, _, nonzero) in _reset_leaves(optimizer).items()
            if shape == ()
        }
        self.assertGreater(len(scalars), 1, "expected step plus optax counters")
        self.assertGreater(sum(scalars.values()), 0, "expected advanced counters")

        vlm_trainer._reset_optimizer_state_in_place(optimizer)

        self.assertEqual(int(jnp.asarray(nnx.state(optimizer)["step"].value)), 0)
        for path, (_, _, _, nonzero) in _reset_leaves(optimizer).items():
            self.assertEqual(nonzero, 0, msg=path)


class RunSftEndToEndTest(absltest.TestCase):
    def test_reset_optimizer_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            output = Path(tmp) / "output"
            train_cfg = _train_cfg(num_steps=2, grad_accum_steps=1)
            # A config object (not the model id) keeps the source run on the random-init
            # path: no HF download, and the weights still land in the checkpoint.
            model_cfg = vlm_api.resolve_config(_MODEL_ID)

            _, metrics = vlm_trainer.run_sft(
                model_cfg,
                train_cfg,
                _grain_iter(),
                save_dir=source,
                resume=ResumeMode.NEVER,
                **_run_kwargs(),
            )
            self.assertEqual(int(metrics["step"]), 2)

            reset_iter = _grain_iter()
            optimizer, reset_metrics = vlm_trainer.run_sft(
                model_cfg,
                train_cfg,
                reset_iter,
                save_dir=output,
                resume=ResumeMode.NEVER,
                init_from=source / "000002",
                reset_optimizer=True,
                **_run_kwargs(),
            )
            self.assertEqual(int(reset_metrics["step"]), 2)

            # weights came from the source checkpoint: a warm start, not a re-init
            self.assertGreater(
                float(
                    jnp.abs(
                        nnx.state(optimizer)["model"]["text"]["embedder"]["embedding"].value
                    ).sum()
                ),
                0.0,
            )
            for path, (_, _, sharding, _) in _reset_leaves(optimizer).items():
                self.assertIsInstance(sharding, NamedSharding, msg=path)

            receipt = json.loads((output / "optimizer_reset_receipt.json").read_text())
            self.assertEqual(receipt["init_from"], str((source / "000002").resolve()))
            self.assertEqual(receipt["continued_from_step"], 2)
            self.assertGreater(receipt["zeroed_leaves"], 0)
            self.assertGreater(receipt["nonzero_elements_before_reset"], 0)

            # the run owns its own step lineage, and the source iterator was never resumed
            self.assertTrue((output / "000001").is_dir())
            self.assertFalse((output / "000000").exists())
            self.assertTrue((output / "config.json").exists())
            self.assertTrue((output / "lora_metadata.json").exists())
            self.assertTrue((source / "000002").is_dir())
            self.assertEqual(_record_index(next(reset_iter)), 2)

            with self.assertRaisesRegex(ValueError, "already holds"):
                vlm_trainer.run_sft(
                    model_cfg,
                    train_cfg,
                    _grain_iter(),
                    save_dir=output,
                    resume=ResumeMode.NEVER,
                    init_from=source / "000002",
                    reset_optimizer=True,
                    **_run_kwargs(),
                )

    def test_resume_keeps_optimizer_state_on_trainer_shardings(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "run"
            model_cfg = vlm_api.resolve_config(_MODEL_ID)
            vlm_trainer.run_sft(
                model_cfg,
                _train_cfg(num_steps=2, grad_accum_steps=1),
                _grain_iter(),
                save_dir=save_dir,
                resume=ResumeMode.NEVER,
                **_run_kwargs(),
            )
            resume_iter = _grain_iter()
            optimizer, metrics = vlm_trainer.run_sft(
                model_cfg,
                _train_cfg(num_steps=4, grad_accum_steps=1),
                resume_iter,
                save_dir=save_dir,
                resume=ResumeMode.REQUIRED,
                **_run_kwargs(),
            )
            self.assertEqual(int(metrics["step"]), 4)
            # unlike the reset path, resume keeps the optimizer state it restored
            leaves = _reset_leaves(optimizer)
            self.assertGreater(sum(nonzero for *_, nonzero in leaves.values()), 0)
            for path, (_, _, sharding, _) in leaves.items():
                self.assertIsInstance(sharding, NamedSharding, msg=path)


if __name__ == "__main__":
    absltest.main()
