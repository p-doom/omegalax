from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from absl.testing import absltest
from safetensors.numpy import save_file

from omegalax.vlm import local_snapshot
from scripts.seal_vlm_snapshot import seal_snapshot


def _snapshot_files() -> dict[str, bytes]:
    return {
        "chat_template.json": b'{"chat_template":"{{ messages }}"}\n',
        "config.json": b'{"model_type":"qwen3_vl"}\n',
        "preprocessor_config.json": b'{"image_processor_type":"Qwen3VLImageProcessor"}\n',
        "tokenizer.json": b'{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"x":0},"unk_token":"x"}}\n',
        "tokenizer_config.json": b'{"model_max_length":128,"tokenizer_class":"PreTrainedTokenizerFast"}\n',
    }


def _make_snapshot(root: Path) -> Path:
    snapshot = root / "snapshot"
    snapshot.mkdir()
    for name, data in _snapshot_files().items():
        (snapshot / name).write_bytes(data)
    save_file({"weight": np.arange(4, dtype=np.float32)}, snapshot / "model.safetensors")
    local_snapshot.write_local_vlm_snapshot_manifest(snapshot)
    return snapshot


def _rewrite_manifest(snapshot: Path, update) -> None:
    manifest_path = snapshot / "omegalax-vlm-snapshot.json"
    os.chmod(snapshot, 0o750)
    os.chmod(manifest_path, 0o640)
    value = json.loads(manifest_path.read_bytes())
    update(value)
    manifest_path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    os.chmod(manifest_path, 0o440)
    os.chmod(snapshot, 0o550)


class LocalVLMSnapshotTest(absltest.TestCase):
    def test_sealer_creates_the_only_accepted_artifact_shape(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            source = root_path / "source"
            source.mkdir()
            for name, data in _snapshot_files().items():
                (source / name).write_bytes(data)
            save_file({"weight": np.arange(4, dtype=np.float32)}, source / "model.safetensors")
            (source / "model.safetensors.index.json").write_text("{}")
            destination = root_path / "sealed"
            self.assertEqual(seal_snapshot(str(source), str(destination)), destination)
            self.assertFalse((destination / "model.safetensors.index.json").exists())
            with local_snapshot.open_local_vlm_snapshot(destination):
                pass

    def test_sealer_rejects_non_regular_source_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            source = root_path / "source"
            source.mkdir()
            for name, data in _snapshot_files().items():
                (source / name).write_bytes(data)
            save_file({"weight": np.arange(4, dtype=np.float32)}, source / "model.safetensors")
            (source / "link").symlink_to(source / "config.json")
            destination = root_path / "sealed"
            with self.assertRaisesRegex(ValueError, "regular file"):
                seal_snapshot(str(source), str(destination))
            self.assertFalse(destination.exists())
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                seal_snapshot(str(source), str(destination))

    def test_pins_every_consumer_file_and_closes_cleanly(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot_path = _make_snapshot(Path(root))
            with local_snapshot.open_local_vlm_snapshot(snapshot_path) as snapshot:
                self.assertEqual(snapshot.path, snapshot_path)
                with snapshot.consume() as consumer:
                    self.assertEqual(
                        (Path(consumer) / "config.json").read_bytes(),
                        _snapshot_files()["config.json"],
                    )
                    self.assertTrue((Path(consumer) / "model.safetensors").is_file())
                snapshot.assert_unchanged()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                snapshot.assert_unchanged()

    def test_constructor_is_not_a_bypass(self):
        with self.assertRaisesRegex(TypeError, "open_local_vlm_snapshot"):
            local_snapshot.LocalVLMSnapshot(None, Path("/tmp/x"), -1, {}, {})

    def test_rejects_relative_and_symlinked_snapshot_paths(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            local_snapshot.open_local_vlm_snapshot("relative")
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            snapshot = _make_snapshot(root_path)
            link = root_path / "link"
            link.symlink_to(snapshot, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                local_snapshot.open_local_vlm_snapshot(link)

    def test_rejects_writable_artifacts_and_directories(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            os.chmod(snapshot, 0o750)
            with self.assertRaisesRegex(ValueError, "directory must be read-only"):
                local_snapshot.open_local_vlm_snapshot(snapshot)
            os.chmod(snapshot, 0o550)
            os.chmod(snapshot / "config.json", 0o640)
            with self.assertRaisesRegex(ValueError, "child must be read-only"):
                local_snapshot.open_local_vlm_snapshot(snapshot)

    def test_rejects_symlink_and_special_children(self):
        for kind in ("symlink", "fifo"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                snapshot = _make_snapshot(Path(root))
                os.chmod(snapshot, 0o750)
                extra = snapshot / "extra"
                if kind == "symlink":
                    extra.symlink_to(snapshot / "config.json")
                else:
                    os.mkfifo(extra)
                os.chmod(snapshot, 0o550)
                with self.assertRaisesRegex(ValueError, "inventory"):
                    local_snapshot.open_local_vlm_snapshot(snapshot)

    def test_rejects_missing_extra_and_mismatched_files(self):
        cases = ("missing", "extra", "content")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                snapshot = _make_snapshot(Path(root))
                os.chmod(snapshot, 0o750)
                if case == "missing":
                    (snapshot / "tokenizer.json").unlink()
                elif case == "extra":
                    (snapshot / "extra.json").write_text("{}")
                else:
                    os.chmod(snapshot / "config.json", 0o640)
                    (snapshot / "config.json").write_text('{"model_type":"qwen3_5"}\n')
                    os.chmod(snapshot / "config.json", 0o440)
                os.chmod(snapshot, 0o550)
                match = "inventory" if case != "content" else "identity mismatch"
                with self.assertRaisesRegex(ValueError, match):
                    local_snapshot.open_local_vlm_snapshot(snapshot)

    def test_rejects_invalid_manifest_json_and_schema(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            manifest = snapshot / "omegalax-vlm-snapshot.json"
            os.chmod(snapshot, 0o750)
            os.chmod(manifest, 0o640)
            manifest.write_bytes(b'{"format":"x","format":"y","files":{}}')
            os.chmod(manifest, 0o440)
            os.chmod(snapshot, 0o550)
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                local_snapshot.open_local_vlm_snapshot(snapshot)

        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            _rewrite_manifest(snapshot, lambda value: value.update(extra=True))
            with self.assertRaisesRegex(ValueError, "invalid schema"):
                local_snapshot.open_local_vlm_snapshot(snapshot)

    def test_rejects_oversized_manifest_before_parsing(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            manifest = snapshot / "omegalax-vlm-snapshot.json"
            os.chmod(snapshot, 0o750)
            os.chmod(manifest, 0o640)
            with manifest.open("wb") as stream:
                stream.truncate((8 << 20) + 1)
            os.chmod(manifest, 0o440)
            os.chmod(snapshot, 0o550)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                local_snapshot.open_local_vlm_snapshot(snapshot)

    def test_rejects_unsupported_config_and_invalid_safetensors(self):
        for case in ("config", "weights"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                snapshot = _make_snapshot(Path(root))
                os.chmod(snapshot, 0o750)
                target = snapshot / ("config.json" if case == "config" else "model.safetensors")
                os.chmod(target, 0o640)
                if case == "config":
                    target.write_text('{"model_type":"qwen3"}\n')
                else:
                    target.write_bytes(b"not safetensors")
                os.chmod(target, 0o440)
                manifest = snapshot / "omegalax-vlm-snapshot.json"
                os.chmod(manifest, 0o640)
                value = json.loads(manifest.read_bytes())
                data = target.read_bytes()
                value["files"][target.name] = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
                manifest.write_text(json.dumps(value, separators=(",", ":")) + "\n")
                os.chmod(manifest, 0o440)
                os.chmod(snapshot, 0o550)
                match = "model_type" if case == "config" else "Invalid safetensors"
                with self.assertRaisesRegex(ValueError, match):
                    local_snapshot.open_local_vlm_snapshot(snapshot)

    def test_replacement_uses_pinned_file_but_fails_the_lease(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot_path = _make_snapshot(Path(root))
            with local_snapshot.open_local_vlm_snapshot(snapshot_path) as snapshot:
                os.chmod(snapshot_path, 0o750)
                replacement = snapshot_path / "replacement"
                replacement.write_bytes(b'{"model_type":"qwen3_5"}\n')
                os.chmod(replacement, 0o440)
                replacement.replace(snapshot_path / "config.json")
                os.chmod(snapshot_path, 0o550)
                with (
                    self.assertRaisesRegex(RuntimeError, "changed after validation"),
                    snapshot.consume() as consumer,
                ):
                    self.assertEqual(
                        (Path(consumer) / "config.json").read_bytes(),
                        _snapshot_files()["config.json"],
                    )

    def test_in_place_mutation_is_detected_before_consumer_returns(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot_path = _make_snapshot(Path(root))
            with local_snapshot.open_local_vlm_snapshot(snapshot_path) as snapshot:
                config = snapshot_path / "config.json"
                with (
                    self.assertRaisesRegex(RuntimeError, "changed after validation"),
                    snapshot.consume(),
                ):
                    os.chmod(config, 0o640)
                    config.write_text('{"model_type":"qwen3_5"}\n')

    def test_contract_is_identical_under_python_optimization(self):
        source = """
import tempfile
from pathlib import Path
from tests.test_vlm_local_snapshot import _make_snapshot
from omegalax.vlm.local_snapshot import open_local_vlm_snapshot
with tempfile.TemporaryDirectory() as root:
    snapshot_path = _make_snapshot(Path(root))
    with open_local_vlm_snapshot(snapshot_path) as snapshot:
        with snapshot.consume() as consumer:
            assert Path(consumer, 'config.json').is_file()
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd()
        for optimized in (False, True):
            command = [sys.executable]
            if optimized:
                command.append("-O")
            command.extend(["-c", source])
            with self.subTest(optimized=optimized):
                subprocess.run(command, check=True, env=env, timeout=180)

    def test_manifest_read_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            os.chmod(snapshot, 0o750)
            manifest = snapshot / "omegalax-vlm-snapshot.json"
            os.chmod(manifest, 0o640)
            with manifest.open("ab") as stream:
                stream.write(b"x")
            os.chmod(manifest, 0o440)
            os.chmod(snapshot, 0o550)
            with (
                mock.patch.object(local_snapshot.os, "read", side_effect=AssertionError),
                self.assertRaisesRegex(ValueError, "Invalid JSON"),
            ):
                local_snapshot.open_local_vlm_snapshot(snapshot)


if __name__ == "__main__":
    absltest.main()
