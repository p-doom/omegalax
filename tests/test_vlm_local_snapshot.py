from __future__ import annotations

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

from omegalax.vlm import api as vlm_api
from omegalax.vlm import local_snapshot
from scripts.seal_vlm_snapshot import _remove_owned_output, seal_snapshot


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


def _rewrite_file_and_identity(snapshot: Path, name: str, data: bytes) -> None:
    target = snapshot / name
    manifest_path = snapshot / "omegalax-vlm-snapshot.json"
    os.chmod(snapshot, 0o750)
    os.chmod(target, 0o640)
    target.write_bytes(data)
    os.chmod(target, 0o440)
    os.chmod(manifest_path, 0o640)
    value = json.loads(manifest_path.read_bytes())
    value["files"][name] = {"size_bytes": len(data)}
    manifest_path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    os.chmod(manifest_path, 0o440)
    os.chmod(snapshot, 0o550)


class LocalVLMSnapshotTest(absltest.TestCase):
    def test_pretrained_loader_rejects_ids_and_raw_paths(self):
        for value in ("Qwen/Qwen3-VL-8B-Instruct", "/local/model"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    TypeError,
                    "LocalVLMSnapshot",
                ),
            ):
                vlm_api.load_pretrained(value)

    def test_sealer_creates_the_only_accepted_artifact_shape(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            source = root_path / "source"
            source.mkdir()
            for name, data in _snapshot_files().items():
                (source / name).write_bytes(data)
            (source / "generation_config.json").write_text('{"temperature":0.7}\n')
            (source / "README.md").write_text("not model identity\n")
            save_file({"weight": np.arange(4, dtype=np.float32)}, source / "model.safetensors")
            (source / "model.safetensors.index.json").write_text("{}")
            destination = root_path / "sealed"
            self.assertEqual(seal_snapshot(str(source), str(destination)), destination)
            self.assertFalse((destination / "model.safetensors.index.json").exists())
            self.assertFalse((destination / "README.md").exists())
            with local_snapshot.open_local_vlm_snapshot(destination) as snapshot:
                self.assertIn("generation_config.json", snapshot.identity_assets)
                self.assertNotIn("omegalax-vlm-snapshot.json", snapshot.names)
                self.assertNotIn("model.safetensors", snapshot.identity_assets)

    def test_sealer_rejects_non_regular_source_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            source = root_path / "source"
            source.mkdir()
            for name, data in _snapshot_files().items():
                (source / name).write_bytes(data)
            save_file({"weight": np.arange(4, dtype=np.float32)}, source / "model.safetensors")
            (source / "tokenizer.model").symlink_to(source / "config.json")
            destination = root_path / "sealed"
            with self.assertRaisesRegex(ValueError, "regular file"):
                seal_snapshot(str(source), str(destination))
            self.assertFalse(destination.exists())
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                seal_snapshot(str(source), str(destination))
            with self.assertRaisesRegex(ValueError, "canonical"):
                seal_snapshot(
                    str(source),
                    str(root_path / "nested" / ".." / "escaped"),
                )

    def test_failed_sealer_cleanup_never_removes_a_replaced_destination(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            destination = root_path / "sealed"
            destination.mkdir()
            (destination / "partial").write_text("partial")
            parent_fd = os.open(root_path, os.O_RDONLY | os.O_DIRECTORY)
            destination_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
            metadata = os.fstat(destination_fd)
            destination.rename(root_path / "moved")
            destination.mkdir()
            sentinel = destination / "sentinel"
            sentinel.write_text("keep")
            try:
                with self.assertRaisesRegex(RuntimeError, "path changed"):
                    _remove_owned_output(
                        parent_fd,
                        destination_fd,
                        destination.name,
                        (metadata.st_dev, metadata.st_ino),
                    )
                self.assertEqual(sentinel.read_text(), "keep")
            finally:
                os.close(destination_fd)
                os.close(parent_fd)

    def test_pins_every_consumer_file_and_closes_cleanly(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot_path = _make_snapshot(Path(root))
            with local_snapshot.open_local_vlm_snapshot(snapshot_path) as snapshot:
                self.assertEqual(snapshot.path, snapshot_path)
                self.assertRegex(snapshot.sha256, r"[0-9a-f]{64}")
                with snapshot.files() as files:
                    self.assertEqual(
                        Path(files["config.json"]).read_bytes(),
                        _snapshot_files()["config.json"],
                    )
                    self.assertTrue(Path(files["model.safetensors"]).is_file())
                    with self.assertRaises(TypeError):
                        files["config.json"] = "/tmp/replacement"
                snapshot.assert_unchanged()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                snapshot.assert_unchanged()

    def test_identity_copy_uses_the_pinned_source_fd(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            snapshot_path = _make_snapshot(root_path)
            with local_snapshot.open_local_vlm_snapshot(snapshot_path) as snapshot:
                output_path = root_path / "config-copy.json"
                output_fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    snapshot.copy_identity_asset_to("config.json", output_fd)
                finally:
                    os.close(output_fd)
                self.assertEqual(output_path.read_bytes(), _snapshot_files()["config.json"])
                with self.assertRaisesRegex(ValueError, "allowed identity asset"):
                    snapshot.copy_identity_asset_to("model.safetensors", -1)

    def test_close_can_retry_after_a_descriptor_close_failure(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = local_snapshot.open_local_vlm_snapshot(_make_snapshot(Path(root)))
            failed_fd = next(iter(snapshot._open_fds))
            real_close = os.close
            failed = False

            def close_once(fd):
                nonlocal failed
                if fd == failed_fd and not failed:
                    failed = True
                    raise OSError("injected close failure")
                real_close(fd)

            with (
                mock.patch.object(local_snapshot.os, "close", side_effect=close_once),
                self.assertRaisesRegex(OSError, "injected close failure"),
            ):
                snapshot.close()
            self.assertFalse(snapshot._closed)
            snapshot.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                snapshot.assert_unchanged()

    def test_constructor_is_not_a_bypass(self):
        with self.assertRaisesRegex(TypeError, "open_local_vlm_snapshot"):
            local_snapshot.LocalVLMSnapshot(
                None,
                Path("/tmp/x"),
                -1,
                -1,
                "x",
                -1,
                "0" * 32,
                {},
            )

    def test_rejects_relative_and_symlinked_snapshot_paths(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            local_snapshot.open_local_vlm_snapshot("relative")
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            snapshot = _make_snapshot(root_path)
            link = root_path / "link"
            link.symlink_to(snapshot, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "no-follow"):
                local_snapshot.open_local_vlm_snapshot(link)
            escaped = snapshot / ".." / snapshot.name
            with self.assertRaisesRegex(ValueError, "canonical"):
                local_snapshot.open_local_vlm_snapshot(escaped)

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
                match = "inventory" if case != "content" else "size mismatch"
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

        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            _rewrite_manifest(
                snapshot,
                lambda value: value["files"].update(
                    {"omegalax-vlm-snapshot.json": {"size_bytes": 0}}
                ),
            )
            with self.assertRaisesRegex(ValueError, "manifest child name"):
                local_snapshot.open_local_vlm_snapshot(snapshot)

    def test_rejects_invalid_and_oversized_identity_json(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            _rewrite_file_and_identity(
                snapshot,
                "tokenizer_config.json",
                b'{"x":1,"x":2}',
            )
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                local_snapshot.open_local_vlm_snapshot(snapshot)

    def test_rejects_manifest_fd_and_aggregate_json_bounds(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            with (
                mock.patch.object(local_snapshot, "_MAX_FILES", 1),
                self.assertRaisesRegex(ValueError, "exceeds 1 files"),
            ):
                local_snapshot.open_local_vlm_snapshot(snapshot)

        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            total = sum(
                len(data) for name, data in _snapshot_files().items() if name.endswith(".json")
            )
            with (
                mock.patch.object(local_snapshot, "_MAX_TOTAL_JSON_ASSET_BYTES", total - 1),
                self.assertRaisesRegex(ValueError, "aggregate bytes"),
            ):
                local_snapshot.open_local_vlm_snapshot(snapshot)

        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            with (
                mock.patch.object(local_snapshot, "_MAX_PATH_COMPONENTS", 1),
                self.assertRaisesRegex(ValueError, "path components"),
            ):
                local_snapshot.open_local_vlm_snapshot(snapshot)

        with tempfile.TemporaryDirectory() as root:
            snapshot = _make_snapshot(Path(root))
            with (
                mock.patch.object(local_snapshot, "_MAX_JSON_ASSET_BYTES", 1),
                self.assertRaisesRegex(ValueError, "exceeds 1 byte"),
            ):
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

    def test_snapshot_primitive_is_model_agnostic_and_rejects_invalid_safetensors(self):
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
                value["files"][target.name] = {"size_bytes": len(data)}
                manifest.write_text(json.dumps(value, separators=(",", ":")) + "\n")
                os.chmod(manifest, 0o440)
                os.chmod(snapshot, 0o550)
                if case == "config":
                    with local_snapshot.open_local_vlm_snapshot(snapshot):
                        pass
                else:
                    with self.assertRaisesRegex(ValueError, "Invalid safetensors"):
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
                    snapshot.files() as files,
                ):
                    self.assertEqual(
                        Path(files["config.json"]).read_bytes(),
                        _snapshot_files()["config.json"],
                    )

    def test_in_place_mutation_is_detected_before_consumer_returns(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot_path = _make_snapshot(Path(root))
            with local_snapshot.open_local_vlm_snapshot(snapshot_path) as snapshot:
                config = snapshot_path / "config.json"
                with (
                    self.assertRaisesRegex(RuntimeError, "changed after validation"),
                    snapshot.files(),
                ):
                    os.chmod(config, 0o640)
                    config.write_text('{"model_type":"qwen3_5"}\n')

    def test_manifest_mutation_is_detected_before_consumer_returns(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot_path = _make_snapshot(Path(root))
            with local_snapshot.open_local_vlm_snapshot(snapshot_path) as snapshot:
                manifest = snapshot_path / "omegalax-vlm-snapshot.json"
                with (
                    self.assertRaisesRegex(RuntimeError, "manifest changed"),
                    snapshot.files(),
                ):
                    os.chmod(manifest, 0o640)
                    manifest.write_bytes(manifest.read_bytes() + b" ")

    def test_parent_path_rebind_cannot_substitute_exact_files(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            parent = root_path / "parent"
            parent.mkdir()
            snapshot_path = _make_snapshot(parent)
            moved = root_path / "moved"
            with local_snapshot.open_local_vlm_snapshot(snapshot_path) as snapshot:
                with snapshot.files() as files:
                    parent.rename(moved)
                    parent.mkdir()
                    self.assertEqual(
                        Path(files["config.json"]).read_bytes(),
                        _snapshot_files()["config.json"],
                    )
                snapshot.assert_unchanged()

    def test_contract_is_identical_under_python_optimization(self):
        source = """
import os
import tempfile
from pathlib import Path
from tests.test_vlm_local_snapshot import _make_snapshot, _snapshot_files
from omegalax.vlm.local_snapshot import open_local_vlm_snapshot
with tempfile.TemporaryDirectory() as root:
    root_path = Path(root)
    parent = root_path / 'parent'
    parent.mkdir()
    snapshot_path = _make_snapshot(parent)
    with open_local_vlm_snapshot(snapshot_path) as snapshot:
        if hasattr(snapshot, 'consume'):
            raise RuntimeError('mutable directory consumer still exists')
        with snapshot.files() as files:
            if any(not path.startswith('/proc/self/fd/') for path in files.values()):
                raise RuntimeError('consumer file is not descriptor rooted')
            parent.rename(root_path / 'moved')
            parent.mkdir()
            if Path(files['config.json']).read_bytes() != _snapshot_files()['config.json']:
                raise RuntimeError('validated config is unavailable')

with tempfile.TemporaryDirectory() as root:
    snapshot_path = _make_snapshot(Path(root))
    with open_local_vlm_snapshot(snapshot_path) as snapshot:
        try:
            with snapshot.files() as files:
                os.chmod(snapshot_path, 0o750)
                replacement = snapshot_path / 'replacement'
                replacement.write_bytes(b'{"model_type":"replacement"}\\n')
                os.chmod(replacement, 0o440)
                replacement.replace(snapshot_path / 'config.json')
                os.chmod(snapshot_path, 0o550)
                if Path(files['config.json']).read_bytes() != _snapshot_files()['config.json']:
                    raise RuntimeError('exact descriptor was substituted')
        except RuntimeError as error:
            if 'changed after validation' not in str(error):
                raise
        else:
            raise RuntimeError('snapshot replacement was not detected')
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
