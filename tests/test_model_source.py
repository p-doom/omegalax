import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from omegalax.registry import resolve_hf_model_source


class ModelSourceTest(absltest.TestCase):
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
