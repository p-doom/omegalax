"""Local VLM snapshot custody contract."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from absl.testing import absltest

from omegalax.vlm.local_snapshot import (
    MANIFEST_NAME,
    REQUIRED_IDENTITY_ASSETS,
    SNAPSHOT_FORMAT,
    open_local_vlm_snapshot,
)
from scripts.seal_vlm_snapshot import seal_vlm_snapshot


def _write_source(path: Path) -> None:
    for name in REQUIRED_IDENTITY_ASSETS:
        payload = b"{}\n" if name.endswith(".json") else b"token-data\n"
        (path / name).write_bytes(payload)
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "README.md").write_text("not part of the runtime snapshot")


class LocalVLMSnapshotTest(absltest.TestCase):
    def test_seal_open_and_copy_exact_identity_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            _write_source(source)
            destination = seal_vlm_snapshot(source, root / "snapshot")

            with open_local_vlm_snapshot(destination) as snapshot:
                self.assertLen(snapshot.sha256, 64)
                self.assertIn("model.safetensors", snapshot.names)
                self.assertNotIn("README.md", snapshot.names)
                copied = root / "copied"
                copied.mkdir()
                snapshot.copy_identity_assets(copied)
                self.assertEqual(
                    {path.name for path in copied.iterdir()}, set(snapshot.identity_assets)
                )
                for name in snapshot.identity_assets:
                    self.assertEqual((copied / name).read_bytes(), (source / name).read_bytes())

    def test_manifest_mismatch_and_path_replacement_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            _write_source(source)
            destination = seal_vlm_snapshot(source, root / "snapshot")
            os.chmod(destination, 0o755)
            os.chmod(destination / "tokenizer.json", 0o644)
            (destination / "tokenizer.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "does not match"):
                open_local_vlm_snapshot(destination)

            payloads = {
                name: (b"{}\n" if name.endswith(".json") else b"token-data\n")
                for name in REQUIRED_IDENTITY_ASSETS
            }
            payloads["model.safetensors"] = b"weights"
            replacement = root / "replacement"
            replacement.mkdir()
            entries = {}
            for name, payload in payloads.items():
                (replacement / name).write_bytes(payload)
                entries[name] = {
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            manifest = {"format": SNAPSHOT_FORMAT, "files": entries}
            (replacement / MANIFEST_NAME).write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            )
            snapshot = open_local_vlm_snapshot(replacement)
            moved = root / "moved"
            replacement.rename(moved)
            replacement.mkdir()
            try:
                with self.assertRaisesRegex(RuntimeError, "path changed"):
                    snapshot.files()
            finally:
                snapshot.close()

    def test_consumer_contract_survives_python_optimized_mode(self):
        code = """
import sys
from omegalax.vlm.local_snapshot import open_local_vlm_snapshot
with open_local_vlm_snapshot(sys.argv[1]) as snapshot:
    if 'config.json' not in snapshot.files() or len(snapshot.sha256) != 64:
        raise RuntimeError('invalid snapshot')
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            _write_source(source)
            destination = seal_vlm_snapshot(source, root / "snapshot")
            for optimized in (False, True):
                command = [sys.executable]
                if optimized:
                    command.append("-O")
                command.extend(["-c", code, str(destination)])
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    absltest.main()
