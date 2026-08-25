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
    staging = root / f".pending-{step:06d}-test"
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
            self.assertEqual(stat.S_IMODE(final.stat().st_mode), 0o750)
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

            with (
                mock.patch.object(
                    checkpoint_utils,
                    "_rename_noreplace",
                    side_effect=OSError("injected publish failure"),
                ),
                self.assertRaisesRegex(OSError, "injected publish failure"),
            ):
                _publish(staging, final)
            self.assertFalse(final.exists())
            self.assertTrue(staging.is_dir())

    def test_publish_race_never_replaces_destination(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            rename_noreplace = checkpoint_utils._rename_noreplace

            def race(source, destination, **kwargs):
                final.mkdir()
                (final / "winner").write_bytes(b"other writer")
                rename_noreplace(source, destination, **kwargs)

            with (
                mock.patch.object(checkpoint_utils, "_rename_noreplace", side_effect=race),
                self.assertRaises(FileExistsError),
            ):
                _publish(staging, final)
            self.assertEqual((final / "winner").read_bytes(), b"other writer")
            self.assertTrue(staging.is_dir())

    def test_staging_name_must_bind_publication_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            mismatched = staging.parent / ".pending-000002-test"
            staging.rename(mismatched)
            with self.assertRaisesRegex(ValueError, "expected hidden sibling"):
                _publish(mismatched, final)
            self.assertFalse(final.exists())

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

    def test_requeue_receipt_binds_verified_frontier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            verified = _publish(staging, final)

            receipt_path = checkpoint_utils.write_requeue_receipt(verified)
            receipt_bytes = receipt_path.read_bytes()
            receipt = checkpoint_utils.read_requeue_receipt(root)

            self.assertEqual(receipt_path, root / checkpoint_utils.REQUEUE_RECEIPT_FILENAME)
            self.assertEqual(receipt.checkpoint_step, 1)
            self.assertEqual(receipt.checkpoint_sha256, verified.sha256)
            self.assertEqual(receipt.exit_code, checkpoint_utils.REQUEUE_EXIT_CODE)
            with self.assertRaises(FileExistsError):
                checkpoint_utils.write_requeue_receipt(verified)
            self.assertEqual(receipt_path.read_bytes(), receipt_bytes)

            staging, final = _staging(root, step=2)
            _publish(staging, final)
            with self.assertRaisesRegex(ValueError, "checkpoint frontier 2"):
                checkpoint_utils.read_requeue_receipt(root)

    def test_torn_requeue_receipt_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            _publish(staging, final)
            (root / checkpoint_utils.REQUEUE_RECEIPT_FILENAME).write_bytes(b'{"schema":')

            with self.assertRaisesRegex(ValueError, "canonical JSON"):
                checkpoint_utils.read_requeue_receipt(root)

    def test_requeue_receipt_publish_failure_has_no_visible_outcome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            verified = _publish(staging, final)
            rename_noreplace = checkpoint_utils._rename_noreplace

            def fail_receipt(source, destination, **kwargs):
                if destination == checkpoint_utils.REQUEUE_RECEIPT_FILENAME:
                    raise OSError("injected receipt failure")
                return rename_noreplace(source, destination, **kwargs)

            with (
                mock.patch.object(checkpoint_utils, "_rename_noreplace", side_effect=fail_receipt),
                self.assertRaisesRegex(OSError, "injected receipt failure"),
            ):
                checkpoint_utils.write_requeue_receipt(verified)
            self.assertFalse((root / checkpoint_utils.REQUEUE_RECEIPT_FILENAME).exists())
            self.assertEmpty(tuple(root.glob(".requeue-required.json.*.pending")))

    def test_requeue_receipt_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            verified = _publish(staging, final)
            payload = final / "train_state" / "state.bin"
            payload.chmod(0o640)
            payload.write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "exhaustive file inventory"):
                checkpoint_utils.write_requeue_receipt(verified)
            self.assertFalse((root / checkpoint_utils.REQUEUE_RECEIPT_FILENAME).exists())

    def test_open_checkpoint_is_pinned_across_whole_path_replacement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            staging, final = _staging(root)
            original = _publish(staging, final)

            with checkpoint_utils.open_verified_checkpoint(final) as opened:
                displaced = root / "original-checkpoint"
                os.rename(final, displaced)
                staging, replacement = _staging(root)
                (staging / "train_state" / "state.bin").write_bytes(b"alternate")
                alternate = _publish(staging, replacement)

                self.assertEqual(opened.verify(), original)
                self.assertNotEqual(alternate.sha256, original.sha256)
                self.assertEqual(
                    (opened.descriptor_path / "train_state" / "state.bin").read_bytes(),
                    b"optimizer-rng-status",
                )
            with self.assertRaisesRegex(RuntimeError, "closed"):
                opened.verify()


if __name__ == "__main__":
    absltest.main()
