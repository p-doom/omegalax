import json
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from omegalax.registry import resolve_hf_model_source


class ModelSourceTest(absltest.TestCase):
    def test_local_model_directory_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            (source / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))

            self.assertEqual(resolve_hf_model_source(str(source), None), source.resolve())

    def test_local_model_directory_rejects_revision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            (source / "config.json").write_text("{}")

            with self.assertRaisesRegex(ValueError, "invalid when model_id is a local path"):
                resolve_hf_model_source(str(source), "a" * 40)

    def test_remote_model_requires_exact_revision(self):
        with self.assertRaisesRegex(ValueError, "exact 40-character model_revision"):
            resolve_hf_model_source("Qwen/example", None)
        with self.assertRaisesRegex(ValueError, "exact 40-character model_revision"):
            resolve_hf_model_source("Qwen/example", "main")

    def test_remote_model_is_resolved_once_from_local_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            (source / "config.json").write_text("{}")
            with mock.patch(
                "omegalax.registry.snapshot_download",
                return_value=str(source),
            ) as download:
                resolved = resolve_hf_model_source("Qwen/example", "a" * 40)

            self.assertEqual(resolved, source.resolve())
            download.assert_called_once_with(
                "Qwen/example",
                revision="a" * 40,
                local_files_only=True,
            )


if __name__ == "__main__":
    absltest.main()
