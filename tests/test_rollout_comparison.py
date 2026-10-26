"""Real common-plan loader, token schedule, pretrained optimizer and checkpoint regression."""

import os
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import json
from pathlib import Path
import unittest

import jax
import numpy as np
from flax import nnx

from omegalax.data import rollout_teacher as rt
from omegalax.data.rollout_teacher_pipeline import (
    _json_sha256,
    make_planned_rollout_teacher_iterator,
)
from omegalax.models.qwen3_5.config import Qwen3_5Config
from omegalax.models.qwen3_5.params import export_qwen3_5_to_safetensors
from omegalax.text import api as text_api
from omegalax.trainers import pretrain as trainer
from omegalax.trainers.checkpoint_utils import ResumeMode
from omegalax.trainers.lr_schedule import build_common_token_schedule
from tests.test_rollout_teacher_pipeline import write_fixture_root
from tests.test_rollout_teacher_training import _tiny_teacher_config


class CommonUpdateTest(unittest.TestCase):
    def test_real_pretrained_update_resume_and_padding(self):
        # Persistent project-side evidence; no fixture deletion on a full home filesystem.
        import uuid

        root = Path(os.environ["TMPDIR"]) / ("common_regression_" + uuid.uuid4().hex)
        root.mkdir()
        samples = [
            dict(
                sequence_id=sid,
                split=split,
                status="completed",
                correct=label,
                token_count=n,
                target_positions=positions,
            )
            for sid, split, label, n, positions in [
                ("a", "train", True, 700, [511, 512, 513, 699]),
                ("b", "train", False, 1000, [10, 512, 998, 999]),
                ("v", "val", True, 800, [512, 513]),
            ]
        ]
        dataset = rt.TeacherDataset(write_fixture_root(root / "dataset", samples))
        records = [
            {**r, "label": "correct" if r["sequence_id"] != "b" else "incorrect"}
            for r in dataset.sequences
        ]
        updates = [
            {"epoch": 0, "members": [0, 1], "context_tokens": 1700, "cumulative_tokens": 1700},
            {"epoch": 1, "members": [1, 0], "context_tokens": 1700, "cumulative_tokens": 3400},
            {"epoch": 1, "members": [0], "context_tokens": 700, "cumulative_tokens": 4100},
        ]
        plan = {
            "source": {"dataset_root": str(dataset.root), "scoring_binding": dataset.binding},
            "records": records,
            "updates": updates,
            "validation_members": [2],
            "chains_per_update": 2,
            "max_context_tokens": 1024,
            "target_capacity": 1024,
            "total_context_tokens": 4100,
            "warmup_end_update": 1,
            "preanneal_update": 2,
            "checkpoint_updates": [1, 2, 3],
            "lr": {"peak": 2e-5},
        }
        plan["plan_sha256"] = _json_sha256(plan)
        iterator = lambda: make_planned_rollout_teacher_iterator(plan, segment_length=512)
        # Same true tokens/targets at all supported segment lengths, regardless of padding.
        reference = None
        for T in (8192, 2048, 512):
            batch = next(make_planned_rollout_teacher_iterator(plan, segment_length=T))
            tokens = batch["token_ids_BCT"].reshape(-1)[
                batch["attention_mask_BCT"].reshape(-1).astype(bool)
            ]
            if reference is None:
                reference = tokens
            np.testing.assert_array_equal(tokens, reference)
            self.assertEqual(int(batch["teacher_mask_BR"].sum()), 8)
        rates = build_common_token_schedule(plan)
        self.assertAlmostEqual(float(rates(0)), 2e-5, places=10)
        self.assertEqual(float(rates(2)), 0.0)
        model, cfg = text_api.init_model(
            _tiny_teacher_config(),
            jax.random.key(0),
            tp_size=1,
            fsdp_size=jax.device_count(),
            dp_size=1,
        )
        from scripts.gate_rollout_state_gradient import measure_state_gradient

        batch = next(iterator())
        # Probe input has one token ID exclusively before the handoff and another after it.
        batch["token_ids_BCT"][:, 0, :] = 100
        batch["token_ids_BCT"][:, 1, :] = 200
        gradient = measure_state_gradient(model, cfg, batch)
        self.assertNotEqual(gradient["full"]["early_embedding_to_late_hidden_gradient"], 0.0)
        self.assertEqual(
            gradient["detached_control"]["early_embedding_to_late_hidden_gradient"], 0.0
        )
        long_probe = {
            "token_ids_BCT": np.full((4, 33, 512), 200, dtype=np.int32),
            "attention_mask_BCT": np.ones((4, 33, 512), dtype=np.int32),
            "reset_state_BC": np.zeros((4, 33), dtype=np.bool_),
        }
        long_probe["token_ids_BCT"][:, 31, :] = 100
        long_probe["reset_state_BC"][:, 0] = True
        long_gradient = measure_state_gradient(model, cfg, long_probe)
        self.assertNotEqual(long_gradient["full"]["early_embedding_to_late_hidden_gradient"], 0.0)
        self.assertEqual(
            long_gradient["detached_control"]["early_embedding_to_late_hidden_gradient"], 0.0
        )
        export_qwen3_5_to_safetensors(model, Qwen3_5Config(text_config=cfg), root / "pretrained")
        train_cfg = trainer.TrainConfig(
            batch_size=8,
            seq_len=512,
            num_steps=3,
            lr_schedule_steps=3,
            learning_rate=2e-5,
            grad_accum_steps=1,
        )
        kwargs = dict(
            pretrain_mode=trainer.PretrainMode.STATEPASSING_BPTT,
            training_objective=trainer.TrainingObjective.TEACHER_TOP512_SPECIAL_TAIL_KL,
            teacher_segment_buckets=[2],
            common_update_plan=plan,
            init_hf_model_path=root / "pretrained",
            tp_size=1,
            fsdp_size=jax.device_count(),
            dp_size=1,
            text_attn_backend="xla",
            log_every=1,
            save_every=0,
            keep_latest=None,
        )
        continuous, cm = trainer.run_pretrain(
            str(root / "pretrained"), train_cfg, iterator(), save_dir=root / "continuous", **kwargs
        )
        trainer.run_pretrain(
            str(root / "pretrained"),
            train_cfg,
            iterator(),
            save_dir=root / "resumed",
            stop_after_local_step=1,
            **kwargs,
        )
        changed = {**plan, "plan_sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "contract differs"):
            trainer.run_pretrain(
                str(root / "pretrained"),
                train_cfg,
                iterator(),
                save_dir=root / "resumed",
                resume=ResumeMode.REQUIRED,
                **{**kwargs, "common_update_plan": changed},
            )
        resumed, rm = trainer.run_pretrain(
            str(root / "pretrained"),
            train_cfg,
            iterator(),
            save_dir=root / "resumed",
            resume=ResumeMode.REQUIRED,
            **kwargs,
        )
        for left, right in zip(
            jax.tree.leaves(nnx.state(continuous)), jax.tree.leaves(nnx.state(resumed))
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        self.assertEqual(cm["total_tokens"], 4100)
        self.assertEqual(rm["teacher_targets"], 4)
        self.assertEqual(rm["teacher_boundary_windows"], 0)
        self.assertNotIn("teacher_boundary_nll", rm)
        self.assertEqual(int(resumed.step[...]), 3)
        (root / "PASS.json").write_text(
            json.dumps(
                {"status": "PASS", "metrics": rm, "gradient_at_16384": long_gradient}, indent=2
            )
        )
        print("common regression evidence:", root)


if __name__ == "__main__":
    unittest.main()
