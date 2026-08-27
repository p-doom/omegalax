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

    def test_remote_model_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            (source / "config.json").write_text("{}")
            with mock.patch(
                "omegalax.registry.snapshot_download",
                return_value=str(source),
            ) as download:
                resolved = resolve_hf_model_source("Qwen/example")

            self.assertEqual(resolved, source.resolve())
            download.assert_called_once_with("Qwen/example")


if __name__ == "__main__":
    absltest.main()
