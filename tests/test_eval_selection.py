"""Tests for deterministic entropy token selection and its resumable cache."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np

from omegalax.evals.selection import (
    EntropySelectionRow,
    EntropySelectorIdentity,
    EntropySelectorStore,
    top_entropy_positions,
)
from omegalax.trainers.loss import chunked_token_entropies
from tests.pretrain_real_data_test_utils import test_temp_dir


class EvalSelectionTest(absltest.TestCase):
    def _identity(self, root: Path) -> EntropySelectorIdentity:
        return EntropySelectorIdentity(
            reference_checkpoint_root=str(root.resolve()),
            reference_checkpoint_step=7,
            reference_config_hash="sha256:config",
            manifest_hash="sha256:manifest",
            dataset_hash="sha256:dataset",
            code_hash="sha256:code",
            pad_id=0,
            fraction=0.1,
            state_mode="zero_gdn_and_conv",
            position_mode="reset_each_chunk",
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
            batch_size=2,
            deltanet_kernel="xla",
            attention_backend="xla",
        )

    def test_top_ten_percent_uses_only_valid_targets_and_deterministic_ties(self):
        entropies = np.asarray([0.2, 0.9, 0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        loss_mask = np.ones(11, dtype=np.int32)
        loss_mask[2] = 0
        self.assertEqual(
            top_entropy_positions(entropies, loss_mask),
            (3,),
        )
        self.assertEmpty(top_entropy_positions(np.asarray([0.1]), np.asarray([1, 0])))

    def test_chunked_entropy_matches_direct_distribution_entropy(self):
        hidden = jnp.asarray([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        lm_head = jnp.asarray([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
        actual = chunked_token_entropies(hidden, lm_head, num_tiles=2)
        logits = hidden[:, :-1] @ lm_head
        log_probs = jnp.log(
            jnp.asarray(jnp.exp(logits) / jnp.exp(logits).sum(axis=-1, keepdims=True))
        )
        expected = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)

    def test_selector_store_creates_directories_before_files_and_roundtrips_masks(self):
        with test_temp_dir() as tmp:
            root = Path(tmp) / "reference"
            identity = self._identity(root)
            store = EntropySelectorStore.open(root, identity)
            rows = (
                EntropySelectionRow(1, 10, (2,)),
                EntropySelectionRow(2, 4, (1,)),
            )
            store.write_shard(3, rows)

            self.assertTrue((store.run_dir / "raw").is_dir())
            self.assertEqual(store.read_shard(3), rows)
            mask = store.selection_mask(3, num_chunks=2, chunk_length=5)
            np.testing.assert_array_equal(
                mask,
                np.asarray([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=np.int32),
            )
            self.assertEqual(
                asdict(EntropySelectorStore.open(root, identity).identity),
                asdict(identity),
            )


if __name__ == "__main__":
    absltest.main()
