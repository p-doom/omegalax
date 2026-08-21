"""Shared helper for comparing JAX vs HF logits in correctness tests.

The smoke tier is measured now that the loader places abstract leaves and the
smoke modules select a backend their CPU pin can reach: nine call sites execute.
The numbers below are what they report, not a guess.

Agreeing call sites, best case (CPU/xla and GPU/mosaic_gpu agree):

    max <= 0.00977    median <= 0.00146

But repeat runs of one such case on a contended CPU gave max 0.0078, 0.0930, and
0.0078 again for byte-identical code and inputs -- a >10x spread on a comparison
that should be deterministic. So the agreeing tier's real spread reaches ~0.093,
and a tolerance calibrated to the best case (0.05 was tried) fails intermittently.
Whatever causes that drift is not understood; until it is, the floor is the
observed spread and not the best reading.

Disagreeing call sites, all well clear of it: the four Qwen3 dense/MoE ones at max
0.698-1.094, median 0.092-0.131, and the left-padded batched ones at max
0.660-1.293. So 0.2 / 0.02 separates every measured defect from every measured
healthy run by a factor of ~3 on both sides. The previous 2.0 / 0.2 passed all of
them on magnitude and left only top-1 doing any work -- which is how a family that
agrees with the reference on 13% of positions read as a tolerance question.

Do not tighten past the spread above on a single quiet reading; take several,
uncontended, and find out what moves first.

Top-1 is the weakest of the three here: at 16 positions its resolution is 0.0625
and a single bf16 tie flips it (0.9375 on CPU vs 1.0000 on GPU for the same code).
Prefer max/median when judging a change.

The eight real-weight call sites are still unmeasured -- they need CUDA plus
``OMEGALAX_RUN_REAL_WEIGHTS_TESTS`` -- and 8B/30B bf16 error accumulates over far
more layers than a 4-layer smoke model. Expect to pass an explicit, wider ``atol``
there on first run, and measure it rather than guessing.
"""

import numpy as np

DEFAULT_ATOL = 0.2
DEFAULT_MEDIAN_ATOL = 0.02
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
