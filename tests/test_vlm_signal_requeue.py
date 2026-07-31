"""Fail-closed tests for signal-triggered VLM checkpoint/requeue."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest

from omegalax.trainers import vlm


class SignalRequeueCheckpointTest(absltest.TestCase):
    def test_regular_checkpoint_preserves_manager_save_policy(self):
        manager = mock.Mock()
        manager.save.return_value = False
        with (
            mock.patch.object(vlm, "_train_state", return_value="state"),
            mock.patch.object(
                vlm.checkpoint_utils,
                "make_grain_save_args",
                return_value="save_args",
            ),
        ):
            saved = vlm._save_sft_checkpoint(
                manager, "optimizer", "rng", 883, "iterator"
            )

        self.assertFalse(saved)
        manager.save.assert_called_once_with(883, args="save_args")

    def test_off_cycle_checkpoint_is_forced_and_attested(self):
        manager = mock.Mock()
        manager.save.return_value = True
        manager.latest_step.return_value = 883
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "000883"
            checkpoint.mkdir()
            (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
            with (
                mock.patch.object(vlm, "_train_state", return_value="state"),
                mock.patch.object(
                    vlm.checkpoint_utils,
                    "make_grain_save_args",
                    return_value="save_args",
                ),
            ):
                vlm._save_sft_checkpoint_for_requeue(
                    manager, "optimizer", "rng", 883, "iterator", root
                )

        manager.save.assert_called_once_with(883, args="save_args", force=True)
        manager.wait_until_finished.assert_called_once_with()
        manager.latest_step.assert_called_once_with()
        manager.close.assert_called_once_with()

    def test_skipped_save_fails_before_wait_close_or_requeue(self):
        manager = mock.Mock()
        manager.save.return_value = False
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(vlm, "_train_state", return_value="state"),
            mock.patch.object(
                vlm.checkpoint_utils,
                "make_grain_save_args",
                return_value="save_args",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "save was skipped"):
                vlm._save_sft_checkpoint_for_requeue(
                    manager, "optimizer", "rng", 883, "iterator", Path(temporary)
                )

        manager.save.assert_called_once_with(883, args="save_args", force=True)
        manager.wait_until_finished.assert_not_called()
        manager.close.assert_not_called()

    def test_missing_final_marker_fails_closed(self):
        manager = mock.Mock()
        manager.save.return_value = True
        manager.latest_step.return_value = 883
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(vlm, "_train_state", return_value="state"),
            mock.patch.object(
                vlm.checkpoint_utils,
                "make_grain_save_args",
                return_value="save_args",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not finalize"):
                vlm._save_sft_checkpoint_for_requeue(
                    manager, "optimizer", "rng", 883, "iterator", Path(temporary)
                )

        manager.wait_until_finished.assert_called_once_with()
        manager.close.assert_not_called()


if __name__ == "__main__":
    absltest.main()
