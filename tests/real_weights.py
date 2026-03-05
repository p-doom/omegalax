"""Shared gate for tests that require downloading/loading real model weights."""

import os
import unittest

_ENV_VAR = "OMEGALAX_RUN_REAL_WEIGHTS_TESTS"
_TRUTHY = {"1", "true", "yes", "on"}


def _real_weights_enabled() -> bool:
    value = os.environ.get(_ENV_VAR, "")
    return value.strip().lower() in _TRUTHY


def requires_real_weights(obj):
    reason = (
        "Requires real model weights/checkpoint downloads. "
        f"Set {_ENV_VAR}=1 to run."
    )
    return unittest.skipUnless(_real_weights_enabled(), reason)(obj)
