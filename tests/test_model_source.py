import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from omegalax.registry import Arch, resolve, resolve_hf_model_source


class ModelSourceTest(absltest.TestCase):
    def test_qwen3_vl_thinking_models_resolve_as_vlm(self):
        model_ids = (
            "Qwen/Qwen3-VL-4B-Thinking",
            "Qwen/Qwen3-VL-8B-Thinking",
        )
        with mock.patch(
            "omegalax.registry.load_hf_config_from_source",
            return_value={"model_type": "qwen3_vl"},
        ):
            for model_id in model_ids:
                with self.subTest(model_id=model_id):
                    self.assertEqual(resolve(model_id), Arch.VLM)

    def test_export_load_model_accepts_resolved_local_snapshot(self):
        from scripts import export_to_hf

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            (source / "config.json").write_text('{"model_type": "qwen3_vl"}')
            with mock.patch("omegalax.registry.snapshot_download") as download:
                resolved = resolve_hf_model_source(str(source))

            loaded = object()
            with mock.patch.object(
                export_to_hf,
                "_load_vlm_model",
                return_value=loaded,
            ) as load_vlm:
                self.assertIs(export_to_hf.load_model(resolved), loaded)

            download.assert_not_called()
            load_vlm.assert_called_once_with(source.resolve())

    def test_invalid_local_model_sources_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file_source = root / "model.bin"
            file_source.write_bytes(b"")
            directory_source = root / "model"
            directory_source.mkdir()

            for source, error in (
                (file_source, "must be a HuggingFace model directory"),
                (directory_source, "has no config.json"),
            ):
                with self.subTest(source=source), self.assertRaisesRegex(ValueError, error):
                    resolve_hf_model_source(str(source))

    def test_remote_model_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            (source / "config.json").write_text("{}")
            with mock.patch(
                "omegalax.registry.snapshot_download",
                return_value=str(source),
            ) as download:
                resolved = resolve_hf_model_source("Qwen/Qwen3-VL-4B-Thinking")

            self.assertEqual(resolved, source.resolve())
            download.assert_called_once_with("Qwen/Qwen3-VL-4B-Thinking")


if __name__ == "__main__":
    absltest.main()
