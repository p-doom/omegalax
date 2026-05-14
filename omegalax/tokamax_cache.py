"""Persistent autotuning cache for tokamax, implemented in user-space.

Tokamax ships baked-in autotuning caches but has no on-disk cache for results
discovered at runtime. This module wraps the public ``tokamax.AutotuningResult``
API (``dump``/``load``/``__enter__``) to persist results across runs.

For a known-up-front workload, the simple form is:

    from omegalax.tokamax_cache import persistent_cache

    cache_path = f"autotune_cache/{jax.devices()[0].device_kind}.json"
    with persistent_cache(cache_path, train_step, *example_args):
        for batch in dataloader:
            train_step(batch)

For training where ops are autotuned lazily inside ``Op.__call__`` (via the
``tokamax_autotuning_cache_miss_fallback="autotune"`` config), use the
session-scoped helper, which on entry loads existing entries as an overlay and
on exit captures everything currently in tokamax's in-memory cache:

    from omegalax.tokamax_cache import session_persistent_cache

    with session_persistent_cache(cache_path):
        run_training_loop(...)

Limitations:
  * ``AutotuningResult.device_kind`` is a single string per file; put the
    device kind in the file name when running on heterogeneous hardware.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
from collections.abc import Iterator
from typing import Any, Callable

from absl import logging

import tokamax


PathLike = str | os.PathLike[str]


def load(path: PathLike) -> tokamax.AutotuningResult | None:
  """Loads an ``AutotuningResult`` from ``path``, or returns None if missing."""
  path = pathlib.Path(path)
  if not path.is_file():
    return None
  with open(path) as fp:
    return tokamax.AutotuningResult.load(fp)


def save(path: PathLike, result: tokamax.AutotuningResult) -> None:
  """Atomically writes ``result`` to ``path``."""
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
  try:
    tmp.write_text(result.dumps())
    os.replace(tmp, path)
  except BaseException:
    if tmp.exists():
      try:
        tmp.unlink()
      except OSError:
        pass
    raise


def merge_save(
    path: PathLike, result: tokamax.AutotuningResult
) -> tokamax.AutotuningResult:
  """Merges ``result`` into the file at ``path`` and writes it back.

  If the file exists, its contents are loaded and combined with ``result`` via
  ``AutotuningResult.__or__``; entries in ``result`` take precedence on key
  collisions. Useful for accumulating new entries across runs.
  """
  existing = load(path)
  merged = result if existing is None else (existing | result)
  save(path, merged)
  return merged


def load_or_autotune(
    path: PathLike,
    f: Callable[..., Any],
    *args: Any,
    **autotune_kwargs: Any,
) -> tokamax.AutotuningResult:
  """Returns the cached ``AutotuningResult`` at ``path``, autotuning if absent.

  Args:
    path: File path for the persistent cache.
    f: A callable or a lowered JAX function, as accepted by ``tokamax.autotune``.
    *args: Positional args to forward to ``tokamax.autotune`` (only valid if
      ``f`` is callable).
    **autotune_kwargs: Forwarded to ``tokamax.autotune`` (``all_implementations``,
      ``progress_bar``, etc.).
  """
  if (result := load(path)) is not None:
    logging.info("Loaded tokamax autotuning cache from %s", path)
    return result
  logging.info("Autotuning tokamax ops; will write cache to %s", path)
  result = tokamax.autotune(f, *args, **autotune_kwargs)
  save(path, result)
  return result


@contextlib.contextmanager
def persistent_cache(
    path: PathLike,
    f: Callable[..., Any],
    *args: Any,
    **autotune_kwargs: Any,
) -> Iterator[tokamax.AutotuningResult]:
  """Context manager: load-or-autotune, then enter the result as an overlay.

  Inside the ``with`` block, ``BoundArguments.cached_autotuning_data`` returns
  cached results from ``path`` before falling back to the packaged caches.
  """
  result = load_or_autotune(path, f, *args, **autotune_kwargs)
  with result:
    yield result


@contextlib.contextmanager
def session_persistent_cache(
    path: PathLike,
    *,
    write: bool = True,
) -> Iterator[tokamax.AutotuningResult | None]:
  """Context manager: load on entry, capture-and-write on exit.

  On entry, an existing ``AutotuningResult`` at ``path`` (if any) is loaded
  and entered as an overlay so cached configs take precedence over the
  packaged caches.

  While the context is active, every call to ``BoundArguments.autotune`` is
  intercepted to record the result. Both call paths produce entries:
    * Explicit ``tokamax.autotune(f, *args)`` calls.
    * Lazy autotune inside ``Op.__call__`` (triggered by
      ``tokamax_autotuning_cache_miss_fallback="autotune"``).

  On exit, captured entries are grouped by inferred device kind and merged
  into ``path``. If multiple device kinds are observed, nothing is written
  (use one file per device kind in that case).

  Args:
    path: Persistent cache file.
    write: If False, only load — useful for non-rank-0 processes in multi-host
      training where only one process should write.
  """
  from tokamax._src.ops import op as _op_lib  # pylint: disable=import-outside-toplevel
  from jax.extend import backend  # pylint: disable=import-outside-toplevel

  existing = load(path)
  captured: list[tuple[Any, Any]] = []
  _orig_autotune = _op_lib.BoundArguments.autotune

  def _patched_autotune(self, *args, **kwargs):
    data = _orig_autotune(self, *args, **kwargs)
    if data:
      captured.append((self, data))
    return data

  if write:
    _op_lib.BoundArguments.autotune = _patched_autotune  # type: ignore[method-assign]

  ctx = contextlib.nullcontext() if existing is None else existing
  try:
    with ctx:
      yield existing
  finally:
    if write:
      _op_lib.BoundArguments.autotune = _orig_autotune  # type: ignore[method-assign]
      try:
        _flush_captured(path, captured)
      except Exception:  # pylint: disable=broad-except
        logging.exception("Failed to persist tokamax cache to %s", path)


def _flush_captured(
    path: PathLike, captured: list[tuple[Any, Any]]
) -> tokamax.AutotuningResult | None:
  """Merges captured ``(bound_args, data)`` pairs into ``path``."""
  from tokamax._src.ops import op as _op_lib  # pylint: disable=import-outside-toplevel
  from jax.extend import backend  # pylint: disable=import-outside-toplevel

  if not captured:
    return None

  groups: dict[str, list[tuple[Any, Any]]] = {}
  for ba, data in captured:
    dk = _op_lib.infer_device_kind(ba) or backend.get_default_device().device_kind
    groups.setdefault(dk, []).append((ba, data))

  if len(groups) > 1:
    logging.warning(
        "Captured autotune entries for multiple device kinds %s; not writing"
        " single-file cache to %s. Use one file per device kind.",
        list(groups), path,
    )
    return None

  device_kind, items = next(iter(groups.items()))
  result = tokamax.AutotuningResult(device_kind, tuple(items))
  return merge_save(path, result)
