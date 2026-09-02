"""Focused coverage for gate-manifest exclusion from final memory eval holdouts."""

import dataclasses
import hashlib
import json
from pathlib import Path
import tempfile
from unittest import mock

from absl.testing import absltest

from omegalax.data.memory_tasks import MemoryTaskConfig, MemoryTaskSplit
from omegalax.evals import memory_pollution
from omegalax.evals.memory_pollution import (
    MemoryTrainingStream,
    build_pollution_free_manifest,
    canonical_json_sha256,
    load_training_stream,
    load_training_streams,
    pair_digest,
    table_digest,
    validate_pollution_free_manifest,
)
from scripts import build_memory_task_eval_manifest


_A48_V2_CONTENT_SHA256 = "971f564351b87e962092de318fcb260fa1c1552b6b1870ebc1c90834a6693610"


def _config(split: MemoryTaskSplit) -> MemoryTaskConfig:
    return MemoryTaskConfig(
        task="exact",
        seed=2,
        segment_length=16,
        num_segments=2,
        split=split,
        generator_version=2,
    )


def _empty_training_stream() -> MemoryTrainingStream:
    return MemoryTrainingStream(
        checkpoint_root=Path("/training/run"),
        checkpoint_step=1,
        contract_sha256="a" * 64,
        config=_config(MemoryTaskSplit.TRAIN),
        train_pairs=0,
    )


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _replace_pair_index(manifest: dict[str, object], pair_index: int) -> dict[str, object]:
    updated = dict(manifest)
    updated["pair_indices"] = [pair_index]
    updated["sha256"] = canonical_json_sha256(
        {name: value for name, value in updated.items() if name != "sha256"}
    )
    return updated


class MemoryEvalHoldoutTest(absltest.TestCase):
    def test_hf_training_stream_uses_explicit_zero_offset(self):
        config = _config(MemoryTaskSplit.TRAIN)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = {
                "task": dataclasses.asdict(config),
                "source": {
                    "restore": "huggingface_safetensors",
                    "global_step_offset": 0,
                    "source_hf_model_path": "/models/Qwen3.5-0.8B",
                },
                "pairs_per_batch": 4,
                "optimizer": {"num_steps": 8, "grad_accum_steps": 1},
                "runtime": {"topology": {"tp_size": 1, "fsdp_size": 1, "dp_size": 1}},
                "revision": {"content_sha256": memory_pollution._current_training_content_sha256()},
            }
            (root / "memory_task_contract.json").write_text(json.dumps(contract))

            streams = load_training_streams(root, 8)
            self.assertLen(streams, 1)
            self.assertEqual(streams[0].train_pairs, 32)

            contract["source"]["global_step_offset"] = 1
            (root / "memory_task_contract.json").write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "must start at global step 0"):
                load_training_streams(root, 8)

    def test_builder_excludes_multiple_gate_pair_and_table_digests(self):
        eval_config = _config(MemoryTaskSplit.EVAL)
        stream = _empty_training_stream()
        gate = build_pollution_free_manifest(eval_config, num_pairs=1, training_streams=(stream,))
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            gate_paths = (tmpdir / "gate_0.json", tmpdir / "gate_1.json")
            _write_manifest(gate_paths[0], gate)
            _write_manifest(gate_paths[1], _replace_pair_index(gate, 1))

            final = build_pollution_free_manifest(
                eval_config,
                num_pairs=1,
                training_streams=(stream,),
                exclude_manifests=gate_paths,
                final_holdout=True,
            )

            self.assertEqual(final["pair_indices"], [2])
            self.assertTrue(final["pollution_guard"]["final_holdout"])
            references = final["pollution_guard"]["exclusion_manifests"]
            self.assertEqual(
                [reference["path"] for reference in references],
                [str(path) for path in gate_paths],
            )
            self.assertEqual(
                [reference["sha256"] for reference in references],
                [hashlib.sha256(path.read_bytes()).hexdigest() for path in gate_paths],
            )
            validate_pollution_free_manifest(final)

            final_pair, _ = memory_pollution.memory_tasks._paired_tokens(2, eval_config)
            for pair_index in (0, 1):
                excluded_pair, _ = memory_pollution.memory_tasks._paired_tokens(
                    pair_index, eval_config
                )
                self.assertNotEqual(
                    pair_digest(final_pair, eval_config),
                    pair_digest(excluded_pair, eval_config),
                )
                self.assertTrue(
                    {table_digest(final_pair[world], eval_config) for world in (0, 1)}.isdisjoint(
                        table_digest(excluded_pair[world], eval_config) for world in (0, 1)
                    )
                )

    def test_final_holdout_requires_an_exclusion_manifest(self):
        with self.assertRaisesRegex(ValueError, "requires at least one exclusion"):
            build_pollution_free_manifest(
                _config(MemoryTaskSplit.EVAL),
                num_pairs=1,
                training_streams=(_empty_training_stream(),),
                final_holdout=True,
            )

    def test_validator_fails_closed_when_exclusion_file_changes(self):
        eval_config = _config(MemoryTaskSplit.EVAL)
        stream = _empty_training_stream()
        gate = build_pollution_free_manifest(eval_config, num_pairs=1, training_streams=(stream,))
        with tempfile.TemporaryDirectory() as tmp:
            gate_path = Path(tmp) / "gate.json"
            _write_manifest(gate_path, gate)
            final = build_pollution_free_manifest(
                eval_config,
                num_pairs=1,
                training_streams=(stream,),
                exclude_manifests=(gate_path,),
            )
            gate_path.write_text(gate_path.read_text() + "\n")

            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                validate_pollution_free_manifest(final)

    def test_validator_rechecks_pair_and_table_disjointness(self):
        eval_config = _config(MemoryTaskSplit.EVAL)
        stream = _empty_training_stream()
        gate = build_pollution_free_manifest(eval_config, num_pairs=1, training_streams=(stream,))
        with tempfile.TemporaryDirectory() as tmp:
            gate_path = Path(tmp) / "gate.json"
            _write_manifest(gate_path, gate)
            final = build_pollution_free_manifest(
                eval_config,
                num_pairs=1,
                training_streams=(stream,),
                exclude_manifests=(gate_path,),
            )
            final["pair_indices"] = [0]
            final["sha256"] = canonical_json_sha256(
                {name: value for name, value in final.items() if name != "sha256"}
            )

            with self.assertRaisesRegex(ValueError, "exclusion manifest pair digest"):
                validate_pollution_free_manifest(final)

    def test_validator_rejects_table_only_overlap(self):
        eval_config = _config(MemoryTaskSplit.EVAL)
        stream = _empty_training_stream()
        gate = _replace_pair_index(
            build_pollution_free_manifest(eval_config, num_pairs=1, training_streams=(stream,)),
            16,
        )
        with tempfile.TemporaryDirectory() as tmp:
            gate_path = Path(tmp) / "gate.json"
            _write_manifest(gate_path, gate)
            final = build_pollution_free_manifest(
                eval_config,
                num_pairs=1,
                training_streams=(stream,),
                exclude_manifests=(gate_path,),
            )
            final["pair_indices"] = [33]
            final["sha256"] = canonical_json_sha256(
                {name: value for name, value in final.items() if name != "sha256"}
            )

            gate_pair, _ = memory_pollution.memory_tasks._paired_tokens(16, eval_config)
            final_pair, _ = memory_pollution.memory_tasks._paired_tokens(33, eval_config)
            self.assertNotEqual(
                pair_digest(gate_pair, eval_config), pair_digest(final_pair, eval_config)
            )
            self.assertFalse(
                {table_digest(gate_pair[world], eval_config) for world in (0, 1)}.isdisjoint(
                    table_digest(final_pair[world], eval_config) for world in (0, 1)
                )
            )
            with self.assertRaisesRegex(ValueError, "exclusion manifest table digest"):
                validate_pollution_free_manifest(final)

    def test_cli_accepts_repeated_exclusion_manifests(self):
        args = build_memory_task_eval_manifest._parser().parse_args(
            [
                "--checkpoint=/run",
                "--output=/final.json",
                "--task=exact",
                "--num_pairs=1",
                "--exclude_manifest=/gate_1.json",
                "--exclude_manifest=/gate_2.json",
            ]
        )

        self.assertEqual(args.exclude_manifest, [Path("/gate_1.json"), Path("/gate_2.json")])

    def test_a48_v2_hash_remains_reconstructable_after_entrypoint_changes(self):
        config = MemoryTaskConfig(
            task="associative_bank",
            seed=0,
            segment_length=64,
            num_segments=4,
            bank_value_alphabet=48,
            split=MemoryTaskSplit.TRAIN,
            generator_version=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = {
                "task": dataclasses.asdict(config),
                "source": {"model_id": "qwen3.5-memory-50m"},
                "pairs_per_batch": 128,
                "optimizer": {"num_steps": 4096, "grad_accum_steps": 2},
                "runtime": {"topology": {"tp_size": 1, "fsdp_size": 1, "dp_size": 1}},
                "revision": {"content_sha256": _A48_V2_CONTENT_SHA256},
            }
            (root / "memory_task_contract.json").write_text(json.dumps(contract))

            with mock.patch.object(
                memory_pollution,
                "_current_training_content_sha256",
                return_value="0" * 64,
            ):
                stream = load_training_stream(root, 4096)

        self.assertEqual(stream.config.bank_value_alphabet, 48)
        self.assertEqual(stream.train_pairs, 1_048_576)


if __name__ == "__main__":
    absltest.main()
