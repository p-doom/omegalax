"""Persistent tokamax autotuning cache (per https://github.com/openxla/tokamax/issues/792)."""

from __future__ import annotations

from pathlib import Path

import jax
import tokamax
from tokamax._src.autotuning.autotuner import AutotuningData
from tokamax._src.benchmarking import BenchmarkData

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
    """Run `tokamax.autotune` and persist the result on process 0.

    nnx.jit returns a `flax.nnx.transforms.compilation.Lowered` from `.lower()`,
    which tokamax does not recognize as `jax.stages.Lowered`. Lower manually
    and pass the inner `jax.stages.Lowered` to keep tokamax's HLO walk happy.
    """
    path = cache_path(cache_dir)
    startup_log(f"running tokamax autotuning -> {path}")
    target = callable_
    if args and hasattr(callable_, "lower"):
        lowered = callable_.lower(*args)
        target = getattr(lowered, "lowered", lowered)
        args = ()
    result = tokamax.autotune(target, *args)
    # AutotuningData also stores Exception entries for configs that failed at
    # compile/benchmark time; pydantic can't serialize those, so drop them.
    pruned = tokamax.AutotuningResult(
        device_kind=result.device_kind,
        data=tuple(
            (
                ba,
                AutotuningData({k: v for k, v in d.items() if isinstance(v, BenchmarkData)}),
            )
            for ba, d in result.data
        ),
    )
    if jax.process_index() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a mid-dump crash can't leave a 0-byte file that
        # would later be loaded as an empty (= invalid) AutotuningResult.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            pruned.dump(f)
        tmp_path.replace(path)
        startup_log(f"saved tokamax autotuning cache ({len(pruned.data)} ops) to {path}")
    return pruned
