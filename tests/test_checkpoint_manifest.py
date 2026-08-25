from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from omegalax.trainers import checkpoint_utils


def _identities() -> checkpoint_utils.CheckpointIdentities:
    return checkpoint_utils.CheckpointIdentities(
        model_sha256="1" * 64,
        dataset_sha256="2" * 64,
        source_sha256="3" * 64,
        runtime_sha256="4" * 64,
    )


def _receipt(step: int = 1) -> checkpoint_utils.ValidationReceipt:
    return checkpoint_utils.ValidationReceipt(
        step=step,
        batches=2,
        loss_sum_hex=(3.5).hex(),
        supervised_tokens=17,
        dataset_sha256="5" * 64,
    )


def _staging(root: Path, step: int = 1) -> tuple[Path, Path]:
    final = root / f"{step:06d}"
    pending = root / f".pending-{step:06d}-test"
    staging = pending / f"{step:06d}"
    (staging / "train_state").mkdir(parents=True)
    (staging / "train_state" / "state.bin").write_bytes(b"optimizer-rng-status")
    (staging / "input_iter").mkdir()
    (staging / "input_iter" / "process_0-of-1.json").write_text(
        '{"logical_shards":8,"schema_version":1,"states":[{},{},{},{},{},{},{},{}]}\n'
    )
    (staging / "schema").mkdir()
    (staging / "schema" / "schema.json").write_text("{}\n")
    return staging, final


def _publish(staging: Path, final: Path):
    return checkpoint_utils.publish_checkpoint(
        staging,
        final,
        phase={"schedule_horizon": 20, "invocation_end_step": 10},
        identities=_identities(),
        rng_sha256="6" * 64,
        validation=_receipt(int(final.name)),
    )


class CheckpointManifestTest(absltest.TestCase):
    def test_whole_directory_publication_and_verification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            verified = _publish(staging, final)

            self.assertEqual(verified.path, final)
            self.assertEqual(verified.step, 1)
            self.assertEqual(verified.identities, _identities())
            self.assertEqual(checkpoint_utils.verify_checkpoint(final), verified)
            self.assertFalse(staging.exists())
            self.assertTrue(final.is_dir())
            self.assertEqual(stat.S_IMODE(final.stat().st_mode), 0o550)
            self.assertEqual(
                stat.S_IMODE((final / "train_state" / "state.bin").stat().st_mode),
                0o440,
            )

            raw = (final / checkpoint_utils.CHECKPOINT_MANIFEST_FILENAME).read_bytes()
            manifest = json.loads(raw)
            self.assertEqual(
                raw,
                (
                    json.dumps(
                        manifest,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("ascii"),
            )
            self.assertEqual(
                [entry["path"] for entry in manifest["files"]],
                [
                    "input_iter/process_0-of-1.json",
                    "schema/schema.json",
                    "train_state/state.bin",
                ],
            )

    def test_publish_failure_never_exposes_numeric_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            real_rename = os.rename
            calls = 0

            def fail_final_rename(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected publish failure")
                return real_rename(*args, **kwargs)

            with (
                mock.patch.object(checkpoint_utils.os, "rename", side_effect=fail_final_rename),
                self.assertRaisesRegex(OSError, "injected publish failure"),
            ):
                _publish(staging, final)
            self.assertFalse(final.exists())
            self.assertTrue(staging.is_dir())

    def test_torn_and_mutated_numeric_checkpoints_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            torn = root / "000001"
            torn.mkdir()
            (torn / "payload").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "regular file"):
                checkpoint_utils.verify_checkpoint(torn)

            staging, final = _staging(root, step=2)
            _publish(staging, final)
            payload = final / "train_state" / "state.bin"
            payload.chmod(0o640)
            payload.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "exhaustive file inventory"):
                checkpoint_utils.verify_checkpoint(final)

    def test_symlink_and_stale_validation_never_publish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            (staging / "link").symlink_to(staging / "schema" / "schema.json")
            with self.assertRaisesRegex(ValueError, "non-regular entry"):
                _publish(staging, final)
            self.assertFalse(final.exists())

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root, step=2)
            with self.assertRaisesRegex(ValueError, "receipt step"):
                checkpoint_utils.publish_checkpoint(
                    staging,
                    final,
                    phase={"schedule_horizon": 20, "invocation_end_step": 10},
                    identities=_identities(),
                    rng_sha256="6" * 64,
                    validation=_receipt(1),
                )
            self.assertFalse(final.exists())

    def test_manifest_unknown_key_and_noncanonical_bytes_fail(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            _publish(staging, final)
            manifest_path = final / checkpoint_utils.CHECKPOINT_MANIFEST_FILENAME
            manifest_path.chmod(0o640)
            manifest = json.loads(manifest_path.read_text())
            manifest["unknown"] = True
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                checkpoint_utils.verify_checkpoint(final)


if __name__ == "__main__":
    absltest.main()
