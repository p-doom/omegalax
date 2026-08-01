"""Tests for full-context evaluation mask and comparison contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from absl.testing import absltest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from omegalax.evals.full_context import (
    compare_relative_nll,
    full_context_arrays,
    validate_full_context_arrays,
)
from scripts import run_full_context_eval


class FullContextEvalTest(absltest.TestCase):
    def test_flatten_preserves_chunk_masks_and_excludes_boundaries(self):
        document = SimpleNamespace(
            token_ids_CT=np.arange(12, dtype=np.int32).reshape(3, 4),
            attention_mask_CT=np.ones((3, 4), dtype=np.int32),
            loss_mask_CT=np.asarray(
                (
                    (1, 1, 1, 1),
                    (1, 1, 0, 1),
                    (1, 1, 1, 0),
                ),
                dtype=np.int32,
            ),
        )

        tokens, attention, loss_masks = full_context_arrays(document)
        validate_full_context_arrays(document, loss_masks)

        np.testing.assert_array_equal(tokens, np.arange(12).reshape(1, 12))
        np.testing.assert_array_equal(attention, np.ones((1, 12), dtype=np.int32))
        self.assertEqual(loss_masks.shape, (3, 1, 12))
        np.testing.assert_array_equal(loss_masks[:, 0, (0, 4, 8)], np.zeros((3, 3)))
        np.testing.assert_array_equal(loss_masks.sum(axis=(1, 2)), (3, 2, 2))

    def test_rejects_materialized_cross_boundary_target(self):
        document = SimpleNamespace(
            token_ids_CT=np.arange(8, dtype=np.int32).reshape(2, 4),
            attention_mask_CT=np.ones((2, 4), dtype=np.int32),
            loss_mask_CT=np.asarray(((1, 1, 1, 1), (1, 1, 1, 1)), dtype=np.int32),
        )
        _, _, loss_masks = full_context_arrays(document)
        loss_masks[1, 0, 4] = 1
        with self.assertRaisesRegex(ValueError, "cross-boundary target for chunk 2"):
            validate_full_context_arrays(document, loss_masks)

    def test_cli_consumes_argparse_flags_before_runtime(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "Summary metrics do not exist"):
                run_full_context_eval.main(
                    (
                        "compare",
                        "--baseline_result",
                        "/does/not/exist",
                        "--candidate",
                        "candidate=/also/missing",
                        "--output_dir",
                        output_dir,
                    )
                )

    def test_relative_comparison_writes_joint_exact_length_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidates = (root / "full", root / "lora")
            points = []
            for view, length in (("in_horizon", None), ("beyond_horizon", None)):
                points.append(
                    {
                        "experiment": "gdn",
                        "metric": "nll",
                        "view": view,
                        "doc_num_chunks": length,
                        "chunk_position": 1,
                        "condition": "true_gdn",
                        "value": 2.0,
                        "token_count": 100.0,
                    }
                )
            for position in (1, 2):
                points.append(
                    {
                        "experiment": "gdn",
                        "metric": "nll",
                        "view": "exact_length",
                        "doc_num_chunks": 2,
                        "chunk_position": position,
                        "condition": "true_gdn",
                        "value": 2.0,
                        "token_count": 100.0,
                    }
                )
            for result, value in ((baseline, 2.0), (candidates[0], 1.5), (candidates[1], 1.75)):
                summary = result / "summary" / "metrics.parquet"
                summary.parent.mkdir(parents=True)
                pq.write_table(
                    pa.Table.from_pylist([dict(point, value=value) for point in points]),
                    summary,
                )
                (result / "eval_config.json").write_text(json.dumps({"c_train": 4}))

            output = root / "comparison"
            compare_relative_nll(
                baseline_result=baseline,
                candidate_results=(("Full-FT", candidates[0]), ("LoRA", candidates[1])),
                output_dir=output,
            )

            relative = pq.read_table(output / "summary/metrics.parquet").to_pylist()
            self.assertAlmostEqual(relative[0]["relative_nll_reduction"], 0.25)
            self.assertTrue(
                (output / "plots/pdfs/relative_exact_length_L2_nll_reduction.pdf").is_file()
            )
            self.assertTrue(
                (
                    output / "plots/exact_length_svgs/relative_exact_length_L2_nll_reduction.svg"
                ).is_file()
            )


if __name__ == "__main__":
    absltest.main()
