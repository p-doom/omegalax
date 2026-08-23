"""Shared helper for comparing JAX vs HF logits in correctness tests.

The default is calibrated to seeded smoke models on right-padded inputs, the
shape emitted by both production collators. Across nine single and batched call
sites, max error is at most 0.00928, median error at most 0.00122, and top-1
agreement at least 0.875. A larger B=4, T=64 dense/MoE probe measured max error
0.01011, median error 0.00124, top-1 agreement at least 0.977, mean top-8 overlap
at least 0.984, KL at most 3.65e-6 nats, and next-token cross-entropy delta at
most 0.0021%.

Real-weight tests need their own measured bounds: error accumulates over many
more layers. Qwen3-0.6B's explicit tolerance is based on its real-weight probe;
do not reuse the smoke default for an unmeasured real model.
"""

import numpy as np

DEFAULT_ATOL = 0.02
DEFAULT_MEDIAN_ATOL = 0.005
DEFAULT_TOP1_MIN_MATCH = 0.8


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
