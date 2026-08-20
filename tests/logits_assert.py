"""Shared helper for comparing JAX vs HF logits in correctness tests.

``DEFAULT_ATOL`` is NOT a measured tolerance. Every one of the 17 call sites is
currently unexecutable -- the nine CPU smoke ones die in ``setUpClass`` on
``params_utils._place_like`` device_put-ing against an ``eval_shape`` leaf's
AbstractMesh sharding, and the eight real-weight ones need CUDA plus
``OMEGALAX_RUN_REAL_WEIGHTS_TESTS`` -- so nothing has exercised these numbers and
they cannot be calibrated from a run. Every model config is bf16 on both sides,
so a max-abs of 2.0 may well be near the real floor for the 8B/30B comparisons
and far above it for the smoke models: one shared default cannot be right for
both. Calibrate per tier once the loader places abstract leaves, and split the
default then; do not tighten it on a guess.
"""

import numpy as np

DEFAULT_ATOL = 2.0
DEFAULT_MEDIAN_ATOL = 0.2
DEFAULT_TOP1_MIN_MATCH = 0.9


def assert_logits_close(
    test_case,
    jax_logits,
    hf_logits,
    mask=None,
    *,
    atol=DEFAULT_ATOL,
    median_atol=DEFAULT_MEDIAN_ATOL,
    top1_min_match=DEFAULT_TOP1_MIN_MATCH,
):
    """Assert JAX and HF logits agree on max abs diff, median abs diff, and top-1 rate.

    ``mask`` is an optional boolean (B, T) selecting the positions to compare.
    """
    if mask is not None:
        jax_masked = jax_logits[mask]
        hf_masked = hf_logits[mask]
    else:
        jax_masked = jax_logits
        hf_masked = hf_logits

    abs_diff = np.abs(jax_masked - hf_masked)
    max_abs = np.max(abs_diff)
    median_abs = np.median(abs_diff)
    test_case.assertLess(
        max_abs,
        atol,
        f"max abs diff {max_abs:.4f} >= {atol} (median={median_abs:.4f})",
    )
    test_case.assertLess(
        median_abs,
        median_atol,
        f"median abs diff {median_abs:.4f} >= {median_atol} (max={max_abs:.4f})",
    )
    jax_top1 = np.argmax(jax_masked, axis=-1)
    hf_top1 = np.argmax(hf_masked, axis=-1)
    match_rate = np.mean(jax_top1 == hf_top1)
    test_case.assertGreater(
        match_rate,
        top1_min_match,
        f"top-1 match rate {match_rate:.2%} <= {top1_min_match:.0%}",
    )
