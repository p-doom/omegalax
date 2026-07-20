"""Tests for immutable training contracts and evaluation resolution."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest

from omegalax.data.pretrain_statepassing import (
    STATEPASSING_CURRICULUM_INDEX_FORMAT,
    STATEPASSING_FIXED_C_INDEX_FORMAT,
)
from omegalax.evals import runner
from omegalax.evals.runner import CheckpointEvalRequest
from omegalax.training_contract import (
    ManualEvalConfig,
    build_training_contract,
    ensure_training_contract,
    resolve_eval_config,
    training_contract_hash,
)
from tests.pretrain_real_data_test_utils import test_temp_dir


def _state_config(**overrides) -> dict:
    values = {
        "pass_gdn_state": True,
        "gdn_layer_limit": None,
        "pass_conv_state": False,
        "pass_rope_positions": True,
        "pad_id": 0,
        "eos_id": 2,
    }
    values.update(overrides)
    return values


def _manual(c_train: int = 4, **overrides) -> ManualEvalConfig:
    return ManualEvalConfig(c_train=c_train, **_state_config(**overrides))


def _write_metadata(path: Path, metadata: dict) -> None:
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


class TrainingContractTest(absltest.TestCase):
    def test_fixed_c_contract_resolves_actual_num_segments(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            index = tmpdir / "index"
            _write_metadata(
                index,
                {
                    "format": STATEPASSING_FIXED_C_INDEX_FORMAT,
                    "num_segments": 6,
                },
            )
            contract = build_training_contract(index, **_state_config())
            ensure_training_contract(tmpdir / "checkpoint", contract)

            resolved = resolve_eval_config(tmpdir / "checkpoint", 500, None)

            self.assertEqual(resolved.c_train, 6)
            self.assertEqual(resolved.resolution_source, "training_contract")
            self.assertEqual(resolved.training_contract_hash, training_contract_hash(contract))

    def test_curriculum_horizon_is_cumulative_and_changes_after_boundary_update(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            index = tmpdir / "index"
            _write_metadata(
                index,
                {
                    "format": STATEPASSING_CURRICULUM_INDEX_FORMAT,
                    "train_order": [2, 4, 3],
                    "splits": {
                        "train": {
                            "phases": {
                                "2": {"phase_steps": 3},
                                "4": {"phase_steps": 2},
                                "3": {"phase_steps": 2},
                            }
                        }
                    },
                },
            )
            contract = build_training_contract(index, **_state_config())
            checkpoint = tmpdir / "checkpoint"
            ensure_training_contract(checkpoint, contract)

            expected = {1: 2, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 4}
            self.assertEqual(
                {step: resolve_eval_config(checkpoint, step, None).c_train for step in expected},
                expected,
            )
            with self.assertRaisesRegex(ValueError, "no C_train horizon"):
                resolve_eval_config(checkpoint, 8, None)

    def test_contract_first_write_is_atomic_and_resume_requires_exact_match(self):
        with test_temp_dir() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            contract = {
                "schema_version": 1,
                "training_index": {"path": "/index", "metadata_hash": "sha256:index"},
                "eval_statepassing_config": _state_config(),
                "horizon_by_step": [{"start_step": 1, "end_step": None, "c_train": 4}],
            }
            with mock.patch(
                "omegalax.training_contract.os.replace",
                wraps=__import__("os").replace,
            ) as replace:
                first_hash = ensure_training_contract(checkpoint, contract)
            self.assertEqual(replace.call_count, 1)
            self.assertEqual(first_hash, ensure_training_contract(checkpoint, dict(contract)))
            self.assertEmpty(tuple(checkpoint.glob(".training_contract.json.*.tmp")))

            changed = dict(contract)
            changed["eval_statepassing_config"] = _state_config(pass_conv_state=True)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                ensure_training_contract(checkpoint, changed)

    def test_existing_checkpoint_without_contract_is_rejected(self):
        with test_temp_dir() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            (checkpoint / "000100").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "no training_contract.*safe resume"):
                ensure_training_contract(checkpoint, {"schema_version": 1})

    def test_legacy_and_contract_manual_resolution_rules(self):
        with test_temp_dir() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            checkpoint.mkdir()
            with self.assertRaisesRegex(ValueError, "all seven legacy"):
                resolve_eval_config(checkpoint, 10, None)

            manual = _manual()
            legacy = resolve_eval_config(checkpoint, 10, manual)
            self.assertEqual(legacy.resolution_source, "manual_flags")
            self.assertIsNone(legacy.training_contract_hash)

            contract = {
                "schema_version": 1,
                "training_index": {"path": "/index", "metadata_hash": "sha256:index"},
                "eval_statepassing_config": _state_config(),
                "horizon_by_step": [{"start_step": 1, "end_step": None, "c_train": 4}],
            }
            ensure_training_contract(checkpoint, contract)
            self.assertEqual(
                resolve_eval_config(checkpoint, 10, manual).resolution_source,
                "training_contract",
            )
            with self.assertRaisesRegex(ValueError, "conflict.*c_train"):
                resolve_eval_config(checkpoint, 10, _manual(c_train=6))

    def test_runner_request_resolves_contract_before_production_spec(self):
        with test_temp_dir() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            contract = {
                "schema_version": 1,
                "training_index": {"path": "/index", "metadata_hash": "sha256:index"},
                "eval_statepassing_config": _state_config(),
                "horizon_by_step": [{"start_step": 1, "end_step": None, "c_train": 4}],
            }
            ensure_training_contract(checkpoint, contract)
            resolved_checkpoint = SimpleNamespace(root=checkpoint.resolve(), step=12)
            with mock.patch.object(
                runner,
                "resolve_checkpoint",
                return_value=resolved_checkpoint,
            ):
                spec = runner.resolve_checkpoint_eval_request(CheckpointEvalRequest(checkpoint))

            self.assertEqual(spec.c_train, 4)
            self.assertEqual(spec.resolution_source, "training_contract")
            self.assertEqual(spec.training_contract_hash, training_contract_hash(contract))


if __name__ == "__main__":
    absltest.main()
