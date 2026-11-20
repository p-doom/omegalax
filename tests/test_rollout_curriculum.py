"""Block-local curriculum through the real common-plan trainer and CLI (tiny model, CPU)."""

import os
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

from omegalax.data import rollout_teacher as rt  # noqa: E402
from omegalax.data.rollout_teacher_pipeline import (  # noqa: E402
    _json_sha256,
    _source_hashes,
    make_planned_rollout_teacher_iterator,
    ROLLOUT_UPDATE_PLAN_FORMAT,
)
from omegalax.models.qwen3_5.config import Qwen3_5Config, Qwen3_5TextConfig  # noqa: E402
from omegalax.models.qwen3_5.params import export_qwen3_5_to_safetensors  # noqa: E402
from omegalax.models.shard_config import ShardConfig  # noqa: E402
from omegalax.text import api as text_api  # noqa: E402
from omegalax.trainers import pretrain as trainer  # noqa: E402
from omegalax.trainers.checkpoint_utils import ResumeMode  # noqa: E402
from tests.test_rollout_teacher_pipeline import write_fixture_root  # noqa: E402

PHASES = [(64, 1), (128, 2), (512, 3)]
METRICS = ("teacher_kl", "teacher_nll", "teacher_boundary_nll", "teacher_boundary_kl")


def _cfg() -> Qwen3_5TextConfig:
    # Qwen3.5 layout in miniature with one softmax-attention layer for the block mask.
    return Qwen3_5TextConfig(
        vocab_size=rt.VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        layer_types=("linear_attention",) * 3 + ("full_attention",),
        rope_theta=10_000_000,
        partial_rotary_factor=0.25,
        mrope_section=(1, 1, 0),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_value_head_dim=8,
        intermediate_size=64,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=jnp.float32,
    )


def _rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


class RolloutCurriculumTest(unittest.TestCase):
    def test_phase_plan_validation_checkpoints_resume_and_cli(self):
        root = Path(os.environ["TMPDIR"]) / ("curriculum_regression_" + uuid.uuid4().hex)
        root.mkdir()
        window = list(range(8192, 8448))  # one complete boundary window after 8192
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
                ("a", "train", True, 8500, [5, 100, *window, 8499]),
                ("b", "train", False, 3000, [10, 512, 2999]),
                ("v", "val", True, 8460, window),
            ]
        ]
        dataset = rt.TeacherDataset(write_fixture_root(root / "dataset", samples))
        records = [
            {**r, "label": "correct" if r["sequence_id"] != "b" else "incorrect"}
            for r in dataset.sequences
        ]
        plan = {
            "format": ROLLOUT_UPDATE_PLAN_FORMAT,
            "source": {
                "dataset_root": str(dataset.root),
                "scoring_binding": dataset.binding,
                **_source_hashes(dataset.root)[1],
            },
            "records": records,
            "updates": [
                {
                    "epoch": 0,
                    "members": [0, 1],
                    "context_tokens": 11500,
                    "cumulative_tokens": 11500,
                },
                {
                    "epoch": 1,
                    "members": [1, 0],
                    "context_tokens": 11500,
                    "cumulative_tokens": 23000,
                },
                {"epoch": 1, "members": [0], "context_tokens": 8500, "cumulative_tokens": 31500},
            ],
            "validation_members": [2],
            "chains_per_update": 2,
            "max_context_tokens": 17 * 512,
            "target_capacity": 1024,
            "total_context_tokens": 31500,
            "warmup_end_update": 1,
            "preanneal_update": 2,
            "checkpoint_updates": [3],
            "lr": {"peak": 2e-5, "warmup_token_fraction": 0.02, "decay_token_fraction": 0.1},
        }
        plan["plan_sha256"] = _json_sha256(plan)

        def iterator(split="train"):
            return make_planned_rollout_teacher_iterator(plan, segment_length=512, split=split)

        model, cfg = text_api.init_model(
            _cfg(), jax.random.key(0), tp_size=1, fsdp_size=jax.device_count(), dp_size=1
        )
        export_qwen3_5_to_safetensors(model, Qwen3_5Config(text_config=cfg), root / "pretrained")

        # Pieces of one chain only: the block mask exists only in the segment scan.
        batch = next(iterator())
        one_piece = {
            "token_ids_BCT": jnp.asarray(batch["token_ids_BCT"][:, :1]),
            "attention_mask_BCT": jnp.asarray(batch["attention_mask_BCT"][:, :1]),
            "reset_state_BC": jnp.asarray(batch["reset_state_BC"][:, :1]),
            "block_pos_BCT": jnp.zeros((4, 1, 512), dtype=jnp.int32),
        }
        with self.assertRaisesRegex(ValueError, "C > 1 pieces"):
            trainer._statepassing_hidden_states(
                model, one_piece, cfg, 0, trainer.PretrainMode.STATEPASSING_BPTT
            )

        train_cfg = trainer.TrainConfig(
            batch_size=4 * 17,
            seq_len=512,
            num_steps=3,
            lr_schedule_steps=3,
            learning_rate=2e-5,
            grad_accum_steps=1,
        )
        kwargs = dict(
            pretrain_mode=trainer.PretrainMode.STATEPASSING_BPTT,
            training_objective=trainer.TrainingObjective.TEACHER_TOP512_SPECIAL_TAIL_KL,
            teacher_segment_buckets=[17],
            common_update_plan=plan,
            init_hf_model_path=root / "pretrained",
            tp_size=1,
            fsdp_size=jax.device_count(),
            dp_size=1,
            text_attn_backend="xla",
            log_every=1,
            save_every=0,
            keep_latest=None,
            pass_conv_state=True,
            curriculum_phases=PHASES,
            checkpoint_updates=[1, 3],
            val_steps=1,
        )

        def run(out, **extra):
            return trainer.run_pretrain(
                str(root / "pretrained"),
                train_cfg,
                iterator(),
                save_dir=root / out,
                val_data_iter=iterator("val"),
                **{**kwargs, **extra},
            )

        for name, extra in [
            ("truncated_bptt", dict(bptt_chunks=2)),
            ("no_state", dict(pass_gdn_state=False)),
            ("short_phases", dict(curriculum_phases=[(64, 1), (128, 2)])),
            ("block_not_dividing_piece", dict(curriculum_phases=[(64, 1), (96, 3)])),
        ]:
            with self.subTest(name), self.assertRaisesRegex(ValueError, "urriculum"):
                run(name, **extra)

        continuous, cm = run("continuous")
        updates = _rows(root / "continuous" / "update_metrics.jsonl")
        self.assertEqual([r["block_len"] for r in updates], [64, 128, 512])
        vals = _rows(root / "continuous" / "val_metrics.jsonl")
        self.assertEqual(
            [(r["step"], r["block_len"]) for r in vals],
            [(1, 512), (1, 64), (2, 512), (2, 128), (3, 512)],
        )
        for row in updates + vals:
            self.assertEqual(row["teacher_boundary_windows"], 1.0)
            for key in METRICS:
                self.assertTrue(math.isfinite(row[key]), (key, row))
        # The phase T reaches validation: the same model scores differently at T=64.
        self.assertNotEqual(vals[0]["teacher_kl"], vals[1]["teacher_kl"])
        self.assertEqual(
            sorted(p.name for p in (root / "continuous").iterdir() if p.name.isdigit()),
            ["000001", "000003"],
        )
        contract = json.loads((root / "continuous" / "common_update_contract.json").read_text())
        self.assertEqual(contract["checkpoint_updates"], [1, 3])

        run("resumed", stop_after_local_step=1)
        resumed, rm = run("resumed", resume=ResumeMode.REQUIRED)
        for left, right in zip(
            jax.tree.leaves(nnx.state(continuous)), jax.tree.leaves(nnx.state(resumed))
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

        def numbers(rows):
            keys = ("step", "block_len", "grad_norm", "lr", *METRICS)
            return [{k: r[k] for k in keys} for r in rows]

        self.assertEqual(
            numbers(_rows(root / "resumed" / "update_metrics.jsonl")), numbers(updates)
        )
        self.assertEqual(_rows(root / "resumed" / "val_metrics.jsonl"), vals)
        self.assertEqual(rm["teacher_kl"], cm["teacher_kl"])

        # CLI with the production options; four CPU devices stand in for the four GPUs.
        (root / "plan.json").write_text(json.dumps(plan))
        driver = """
import sys
import jax
import scripts.train_rollout_comparison as cli
devices = jax.devices
gpu = type("Gpu", (), {"platform": "gpu"})()
jax.devices = lambda *a, **k: [gpu] * 4 if sys._getframe(1).f_code is cli.main.__code__ else devices(*a, **k)
cli.main()
"""
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        }
        cli_args = [
            *("--plan", str(root / "plan.json"), "--segment-length", "512"),
            *("--init-hf-model-path", str(root / "pretrained"), "--output", str(root / "cli")),
            *("--text-attn-backend", "xla", "--pass-conv-state", "--stop-after-update", "2"),
            *("--curriculum-phases", "64:1,128:2,512:3", "--checkpoint-updates", "1,3"),
        ]
        proc = subprocess.run(
            [sys.executable, "-c", driver, *cli_args], env=env, capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        cli_contract = json.loads((root / "cli" / "training_contract.json").read_text())
        self.assertEqual(cli_contract["curriculum_phases"], [list(p) for p in PHASES])
        self.assertEqual(cli_contract["checkpoint_updates"], [1, 3])
        self.assertTrue(cli_contract["eval_statepassing_config"]["pass_conv_state"])
        cli_updates = _rows(root / "cli" / "update_metrics.jsonl")
        self.assertEqual([r["block_len"] for r in cli_updates], [64, 128])
        self.assertEqual(
            [(r["step"], r["block_len"]) for r in _rows(root / "cli" / "val_metrics.jsonl")],
            [(1, 512), (1, 64), (2, 512), (2, 128)],
        )
        (root / "PASS.json").write_text(
            json.dumps({"status": "PASS", "updates": updates, "vals": vals}, indent=2)
        )
        print("curriculum regression evidence:", root)


if __name__ == "__main__":
    unittest.main()
