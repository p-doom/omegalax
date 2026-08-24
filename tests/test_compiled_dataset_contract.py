"""Adversarial tests for compiled dataset publication and preflight."""

import json
import tempfile
from pathlib import Path

from absl.testing import absltest
from array_record.python.array_record_module import ArrayRecordWriter

from omegalax.data.artifact_contract import (
    canonical_sha256,
    file_identity,
    validate_train_val_lineage,
    validate_training_dataset_contract,
    verify_compiled_dataset,
)
from omegalax.data.grain_pipeline import (
    _write_chat_message_lengths,
    build_records_from_chat,
)

_MEASUREMENT_CONTRACT = {
    "producer_sha": "1" * 40,
    "tokenizer": {
        "source": "test-tokenizer",
        "revision": "2" * 40,
        "behavior_sha256": "a" * 64,
        "files": [{"path": "tokenizer.json", "size_bytes": 1, "sha256": "f" * 64}],
    },
    "processor": None,
    "renderer": {"class": "test.Renderer", "config_sha256": "b" * 64},
    "preprocessor": None,
}


def _measure(message):
    return message["measurement"]


class CompiledDatasetContractTest(absltest.TestCase):
    def _build(self, root: Path, *, patches: int = 12, images: int = 1) -> Path:
        chat = root / "chat.jsonl"
        rows = [
            {
                "recording_id": "recording-1",
                "messages": [
                    {
                        "role": "user",
                        "content": "do it",
                        "measurement": {
                            "length": 2,
                            "supervised_tokens": 0,
                            "vision_tokens": patches // 4,
                            "vision_patches": patches,
                            "num_images": images,
                            "image_grid_thw": [[1, 2, patches // 2]] if images else [],
                        },
                    },
                    {
                        "role": "assistant",
                        "content": "ok",
                        "measurement": {
                            "length": 1,
                            "supervised_tokens": 1,
                            "vision_tokens": 0,
                            "vision_patches": 0,
                            "num_images": 0,
                            "image_grid_thw": [],
                        },
                    },
                ],
            }
        ]
        chat.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return build_records_from_chat(
            chat,
            root / "records",
            max_length=8,
            measure_message=_measure,
            measurement_contract=_MEASUREMENT_CONTRACT,
            records_per_shard=8,
        )

    def _external_fixture(self, root: Path) -> tuple[Path, dict]:
        external_root = root / "pixels"
        shard = external_root / "frames" / "segment" / "images.array_record"
        shard.parent.mkdir(parents=True)
        writer = ArrayRecordWriter(str(shard), "group_size:1")
        writer.write(b"pixel-bytes")
        writer.close()
        identity = file_identity(shard)
        shard.chmod(0o440)
        shard.parent.chmod(0o550)
        shard.parent.parent.chmod(0o550)
        external_root.chmod(0o550)
        shards = [
            {
                "path": "frames/segment/images.array_record",
                **identity,
                "max_record_index": 0,
            }
        ]
        inventory_identity = {
            "schema_version": 1,
            "retention_pin_sha256": "9" * 64,
            "shards": shards,
        }
        return shard, {
            **inventory_identity,
            "root": str(external_root),
            "inventory_sha256": canonical_sha256(inventory_identity),
        }

    def _build_external(self, root: Path, shard: Path, inventory: dict) -> Path:
        chat = root / "external_chat.jsonl"
        chat.write_text(
            json.dumps(
                {
                    "recording_id": "recording-external",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": f"ar://{shard}#0"}],
                        },
                        {"role": "assistant", "content": "ok"},
                    ],
                }
            )
            + "\n"
        )
        cache = root / "external_message_lengths.jsonl"
        _write_chat_message_lengths(
            cache,
            {
                (0, 0): {
                    "length": 2,
                    "supervised_tokens": 0,
                    "vision_tokens": 1,
                    "vision_patches": 4,
                    "num_images": 1,
                    "image_grid_thw": [[1, 2, 2]],
                },
                (0, 1): {
                    "length": 1,
                    "supervised_tokens": 1,
                    "vision_tokens": 0,
                    "vision_patches": 0,
                    "num_images": 0,
                    "image_grid_thw": [],
                },
            },
            chat,
            _MEASUREMENT_CONTRACT,
        )
        return build_records_from_chat(
            chat,
            root / "external_records",
            max_length=8,
            measure_message=_measure,
            measurement_contract=_MEASUREMENT_CONTRACT,
            message_lengths_path=cache,
            external_artifact_inventory=inventory,
            records_per_shard=8,
        )

    def test_published_inventory_and_observed_maxima_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = self._build(Path(tmpdir))

            metadata = verify_compiled_dataset(records)

            profile = metadata["artifact_contract"]["record_profile"]
            self.assertEqual(profile["max_vision_patches"], 12)
            self.assertEqual(profile["max_images"], 1)
            self.assertEqual(profile["max_measured_length"], 3)

    def test_same_size_shard_mutation_rejects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = self._build(Path(tmpdir))
            metadata = verify_compiled_dataset(records)
            shard = records / metadata["artifact_contract"]["shards"][0]["path"]
            payload = bytearray(shard.read_bytes())
            payload[-1] ^= 1
            shard.chmod(0o640)
            shard.write_bytes(payload)
            shard.chmod(0o440)

            with self.assertRaisesRegex(ValueError, "content identity"):
                verify_compiled_dataset(records)

    def test_external_arrayrecord_mutation_rejects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard, inventory = self._external_fixture(root)
            records = self._build_external(root, shard, inventory)
            verify_compiled_dataset(records)
            shard.chmod(0o640)
            payload = bytearray(shard.read_bytes())
            payload[-1] ^= 1
            shard.write_bytes(payload)
            shard.chmod(0o440)

            with self.assertRaisesRegex(ValueError, "content identity"):
                verify_compiled_dataset(records)

    def test_external_arrayrecord_requires_inventory_before_measurement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard, _inventory = self._external_fixture(root)
            chat = root / "missing_inventory.jsonl"
            chat.write_text(
                json.dumps(
                    {
                        "recording_id": "recording-external",
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "image", "image": f"ar://{shard}#0"}],
                            },
                            {"role": "assistant", "content": "ok"},
                        ],
                    }
                )
                + "\n"
            )

            with self.assertRaisesRegex(ValueError, "no external_artifact_inventory"):
                build_records_from_chat(
                    chat,
                    root / "records",
                    max_length=8,
                    measure_message=_measure,
                    measurement_contract=_MEASUREMENT_CONTRACT,
                    records_per_shard=8,
                )

    def test_single_source_processor_mismatch_rejects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = self._build(Path(tmpdir))
            changed = json.loads(json.dumps(_MEASUREMENT_CONTRACT))
            changed["tokenizer"]["behavior_sha256"] = "e" * 64

            with self.assertRaisesRegex(ValueError, "measurement contract"):
                validate_training_dataset_contract(
                    [records],
                    val_path=None,
                    measurement_contract=changed,
                    max_length=8,
                    max_vision_patches_per_sample=12,
                    max_vision_images_per_sample=1,
                )

    def test_observed_patch_maximum_rejects_before_collation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = self._build(Path(tmpdir), patches=12)

            with self.assertRaisesRegex(ValueError, "observed max_vision_patches=12"):
                validate_training_dataset_contract(
                    [records],
                    val_path=None,
                    measurement_contract=_MEASUREMENT_CONTRACT,
                    max_length=8,
                    max_vision_patches_per_sample=11,
                    max_vision_images_per_sample=1,
                )

    def test_train_validation_lineage_overlap_rejects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = verify_compiled_dataset(self._build(Path(tmpdir)))
            train = json.loads(json.dumps(metadata))
            val = json.loads(json.dumps(metadata))
            train["artifact_contract"]["lineage"]["split"] = "train"
            val["artifact_contract"]["lineage"]["split"] = "val"

            with self.assertRaisesRegex(ValueError, "lineage overlap"):
                validate_train_val_lineage([train], val)

    def test_legacy_metadata_rejects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps({"version": 1, "shard_paths": ["part-00000.array_record"]})
            )
            metadata_path.chmod(0o440)

            with self.assertRaisesRegex(ValueError, "compiled dataset schema"):
                verify_compiled_dataset(root)


if __name__ == "__main__":
    absltest.main()
