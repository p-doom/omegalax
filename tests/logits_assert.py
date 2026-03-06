"""Shared helper for comparing JAX vs HF logits in correctness tests."""

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
    """Assert JAX and HF logits match within tolerance (max/median abs diff and top-1 match).

    Args:
        test_case: unittest.TestCase instance (for assertLess/assertGreater).
        jax_logits: JAX logits array (B,T,V) or already masked (N,V).
        hf_logits: HF logits array, same shape as jax_logits.
        mask: Optional boolean mask (B,T). If provided, only masked positions are compared.
        atol: Max allowed absolute difference.
        median_atol: Max allowed median absolute difference.
        top1_min_match: Min fraction of positions where argmax must match (e.g. 0.9).
            Pass None to skip the top-1 check.
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
        max_abs, atol,
        f"max abs diff {max_abs:.4f} >= {atol} (median={median_abs:.4f})",
    )
    test_case.assertLess(
        median_abs, median_atol,
        f"median abs diff {median_abs:.4f} >= {median_atol} (max={max_abs:.4f})",
    )
    if top1_min_match is not None:
        jax_top1 = np.argmax(jax_masked, axis=-1)
        hf_top1 = np.argmax(hf_masked, axis=-1)
        match_rate = np.mean(jax_top1 == hf_top1)
        test_case.assertGreater(
            match_rate, top1_min_match,
            f"top-1 match rate {match_rate:.2%} <= {top1_min_match:.0%}",
        )
