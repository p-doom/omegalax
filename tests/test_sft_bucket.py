import importlib.util
import json
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_sft_bucket_module = _load_script("build_sft_bucket")
count_sft_tokens_module = _load_script("count_sft_tokens")


def _measure(length: int) -> dict:
    return {
        "length": length,
        "vision_tokens": 0,
        "vision_patches": 0,
        "num_images": 0,
        "image_grid_thw": [],
    }


def _record(sample_id: str, split: str, length: int, message_lengths: list[int]) -> dict:
    messages = []
    for idx, message_length in enumerate(message_lengths):
        role = "assistant" if idx % 2 else "user"
        messages.append(
            {
                "role": role,
                "content": f"{sample_id}:{idx}",
                "_omegalax_token_measure": _measure(message_length),
            }
        )
    return {
        "sample_id": sample_id,
        "split": split,
        "messages": messages,
        "_omegalax_token_length": length,
    }


class SftBucketTest(absltest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_count_tokens_writes_reusable_precomputed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical = root / "canonical"
            self._write_jsonl(
                canonical / "chat.jsonl",
                [
                    {
                        "sample_id": "sample_1",
                        "split": "train",
                        "messages": [
                            {"role": "system", "content": "system"},
                            {"role": "user", "content": "hello"},
                            {"role": "assistant", "content": "world"},
                        ],
                    }
                ],
            )

            manifest = count_sft_tokens_module.count_sft_tokens(
                canonical_root=canonical,
                out_dir=root / "token_counts",
                model_id="test-model",
                measure_message=lambda message: _measure(len(message["content"])),
                num_workers=1,
            )

            self.assertEqual(manifest["artifact_type"], "omegalax_sft_token_counts")
            self.assertEqual(manifest["n_valid"], 1)
            records = [
                json.loads(line)
                for line in (root / "token_counts" / "records.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[0]["_omegalax_token_length"], len("systemhelloworld"))
            self.assertIn("_omegalax_token_measure", records[0]["messages"][0])

    def test_build_bucket_uses_precomputed_counts_with_min_exclusive_max_inclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            token_counts = root / "token_counts"
            token_counts.mkdir()
            (token_counts / "manifest.json").write_text(
                json.dumps(
                    {
                        "artifact_type": "omegalax_sft_token_counts",
                        "schema_version": 1,
                        "model_id": "test-model",
                        "tokenizer": "test-tokenizer",
                        "processor": None,
                        "preprocessor_config": None,
                        "system_message": None,
                        "system_message_measure": None,
                    }
                )
                + "\n"
            )
            self._write_jsonl(
                token_counts / "records.jsonl",
                [
                    _record("too_short", "train", 8, [4, 4]),
                    _record("train_ok", "train", 9, [4, 5]),
                    _record("val_ok", "val", 16, [8, 8]),
                    _record("too_long", "val", 17, [8, 9]),
                ],
            )

            def fake_compile(src_path, out_dir, **_kwargs):
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "metadata.json").write_text(json.dumps({"source_path": str(src_path)}))

            def fake_chunk_index(payload_path, out_dir, *, measure_message, **_kwargs):
                out_dir.mkdir(parents=True, exist_ok=True)
                probe = {"role": "user", "content": "x", "_omegalax_token_measure": _measure(3)}
                self.assertEqual(measure_message(probe)["length"], 3)
                (out_dir / "metadata.json").write_text(
                    json.dumps({"payload_path": str(payload_path)})
                )

            with (
                mock.patch.object(
                    build_sft_bucket_module,
                    "compile_jsonl_to_arrayrecord",
                    side_effect=fake_compile,
                ),
                mock.patch.object(
                    build_sft_bucket_module,
                    "build_chunk_index",
                    side_effect=fake_chunk_index,
                ),
            ):
                manifest = build_sft_bucket_module.build_sft_bucket(
                    token_count_root=token_counts,
                    out_dir=root / "bucket",
                    min_length=8,
                    max_length=16,
                    messages_per_record=8,
                    records_per_shard_payload=8,
                    records_per_shard_index=8,
                    num_workers=1,
                )

            self.assertEqual(manifest["n_train"], 1)
            self.assertEqual(manifest["n_val"], 1)
            self.assertEqual(manifest["n_rejected"], 2)

            train_rows = [
                json.loads(line)
                for line in (root / "bucket" / "source" / "train.jsonl").read_text().splitlines()
            ]
            val_rows = [
                json.loads(line)
                for line in (root / "bucket" / "source" / "val.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["sample_id"] for row in train_rows], ["train_ok"])
            self.assertEqual([row["sample_id"] for row in val_rows], ["val_ok"])

            rejected = [
                json.loads(line)
                for line in (root / "bucket" / "source" / "rejected.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                {row["sample_id"]: row["reason"] for row in rejected},
                {"too_short": "too_short_for_bucket", "too_long": "too_long_for_bucket"},
            )
            self.assertTrue((root / "bucket" / "train" / "metadata.json").is_file())
            self.assertTrue((root / "bucket" / "val" / "metadata.json").is_file())


if __name__ == "__main__":
    absltest.main()
