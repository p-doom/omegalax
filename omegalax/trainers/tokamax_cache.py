"""Persistent tokamax autotuning cache (per https://github.com/openxla/tokamax/issues/792)."""

from __future__ import annotations

from pathlib import Path

import jax
import tokamax

from omegalax.trainers.text import startup_log


_FILENAME = "tokamax_autotuning.json"


def cache_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / _FILENAME


def try_load(cache_dir: str | Path) -> tokamax.AutotuningResult | None:
    """Load AutotuningResult from cache_dir if present, else None."""
    path = cache_path(cache_dir)
    if not path.exists():
        return None
    startup_log(f"loading tokamax autotuning cache from {path}")
    with open(path, "r") as f:
        return tokamax.AutotuningResult.load(f)


def autotune_and_save(
    cache_dir: str | Path,
    callable_,
    *args,
) -> tokamax.AutotuningResult:
    """Run `tokamax.autotune` and persist the result on process 0."""
    path = cache_path(cache_dir)
    startup_log(f"running tokamax autotuning -> {path}")
    result = tokamax.autotune(callable_, *args)
    if jax.process_index() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            result.dump(f)
        startup_log(f"saved tokamax autotuning cache ({len(result.data)} ops) to {path}")
    return result
