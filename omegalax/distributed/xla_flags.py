"""GPU XLA performance flags for comm/compute overlap (latency hiding).

On TPU, XLA overlaps collectives (FSDP all-gather / reduce-scatter, DP all-reduce,
future EP all-to-all) with compute automatically. On **GPU** this overlap must be
turned on explicitly via ``XLA_FLAGS``; by default the GPU backend schedules
collectives largely synchronously, so communication stalls the pipeline.

This module builds a conservative, well-justified GPU flag string and installs it
into ``os.environ["XLA_FLAGS"]``. It is the single source of truth for those flags.

Ordering requirement (IMPORTANT)
--------------------------------
XLA reads ``XLA_FLAGS`` from the process environment **once, lazily, when the XLA
backend is first created** -- i.e. at the first jax computation, ``jax.devices()``,
or ``jax.distributed.initialize()``. Merely doing ``import jax`` does NOT create the
backend, so it is sufficient (and required) to set ``XLA_FLAGS`` *before the first
such call*. In these entrypoints the first backend-creating call is
``jax.distributed.initialize()`` inside ``main()``, and there are no module-level
jax device/computation calls, so calling :func:`configure_gpu_xla_flags` as the very
first statement of ``main()`` (before ``jax.distributed.initialize()``) is correct.
Setting it after the backend is created is a silent no-op.

Behavior
--------
* **Append, never clobber.** Any pre-existing ``XLA_FLAGS`` are preserved. Flags the
  user already specified are NOT overridden -- our defaults are only added for keys
  the user did not set, and the user's original string is appended *last* so that on
  a genuine duplicate the user's value still wins (XLA honors the last occurrence).
* **Idempotent.** Calling it twice does not duplicate flags (guarded by a sentinel).
* **Opt-out.** Set ``OMEGALAX_DISABLE_XLA_PERF_FLAGS=1`` (or pass ``enable=False``) to
  skip entirely. Users can also just set their own ``XLA_FLAGS`` -- those win.
* **GPU-only.** On non-GPU platforms this is a no-op (the flags are GPU-specific and
  harmless-but-pointless elsewhere). Platform is detected *without* importing jax /
  creating a backend, so the ordering guarantee above is not violated.

Flag validation
----------------
Every flag below was parse-checked against the installed jaxlib 0.9.2 CUDA plugin on
CPU (XLA rejects unknown GPU flags at backend init even on the CPU backend, via
``parse_flags_from_env.cc`` "Unknown flag in XLA_FLAGS"). Flags that this XLA build
rejected were dropped -- see the module docstring notes and ``tests/test_xla_flags.py``.
Notably the per-collective ``--xla_gpu_enable_async_{all_gather,reduce_scatter,
all_reduce,collective_permute}`` toggles are **not** registered in this build (async
collectives are driven by the latency-hiding scheduler here), so they are omitted.
"""

from __future__ import annotations

import os

# Env var that fully disables the perf flags (opt-out escape hatch).
DISABLE_ENV_VAR = "OMEGALAX_DISABLE_XLA_PERF_FLAGS"

# Idempotency marker: XLA rejects unknown flags, so we can't add a private sentinel
# flag. Instead we detect a prior application by scanning XLA_FLAGS for a flag we
# always set (the latency-hiding scheduler). If it's already present, we assume our
# defaults were installed and skip re-applying.
_IDEMPOTENCY_MARKER = "--xla_gpu_enable_latency_hiding_scheduler"

# Combine thresholds batch many small collectives into fewer, larger NCCL calls to
# amortize per-launch overhead. 32 MiB is a widely used, conservative value (matches
# common MaxText GPU configs); large enough to coalesce FSDP shards without creating
# monster collectives that hurt overlap granularity. Bytes.
_COMBINE_THRESHOLD_BYTES = 32 * 1024 * 1024  # 33554432


def _default_flags() -> dict[str, str]:
    """The GPU perf flags we add, as an ordered {flag_name: value} mapping.

    Each was validated (name+value parse-accepted) against the installed jaxlib 0.9.2
    CUDA plugin. Keep this list small, safe, and well-understood.
    """
    return {
        # Latency-hiding scheduler: the master switch. Reorders the HLO schedule to
        # start collectives early (async-start) and overlap them with independent
        # compute (async-done later). Without this, GPU collectives are effectively
        # synchronous. This is the single highest-value flag and gates async behavior
        # for all-gather / reduce-scatter / all-reduce in this XLA build.
        "--xla_gpu_enable_latency_hiding_scheduler": "true",
        # Combine (coalesce) small collectives up to N bytes into one launch. Reduces
        # kernel-launch / NCCL setup overhead for the many small FSDP shards. Risk: too
        # large delays overlap; 32 MiB is conservative and battle-tested.
        "--xla_gpu_all_reduce_combine_threshold_bytes": str(_COMBINE_THRESHOLD_BYTES),
        "--xla_gpu_reduce_scatter_combine_threshold_bytes": str(_COMBINE_THRESHOLD_BYTES),
        "--xla_gpu_all_gather_combine_threshold_bytes": str(_COMBINE_THRESHOLD_BYTES),
        # Combine collectives across different tensor dims (not just same-dim). Lets the
        # combiner coalesce more of the FSDP all-gathers/reduce-scatters that come from
        # differently-shaped params. Safe; improves the effectiveness of the thresholds
        # above.
        "--xla_gpu_enable_all_gather_combine_by_dim": "false",
        "--xla_gpu_enable_reduce_scatter_combine_by_dim": "false",
        # Pipelined collectives inside while-loops (the training step / scan bodies):
        # software-pipeline the collective across loop iterations so iteration i's comm
        # overlaps iteration i+1's compute. Directly targets the FSDP all-gather /
        # reduce-scatter that live in the per-layer scan. Depends on the latency-hiding
        # scheduler being on.
        "--xla_gpu_enable_pipelined_all_gather": "true",
        "--xla_gpu_enable_pipelined_reduce_scatter": "true",
        "--xla_gpu_enable_pipelined_all_reduce": "true",
        # Double-buffer while-loop bodies: unrolls one extra iteration so prefetch/comm
        # of the next iteration overlaps compute of the current one. Complements the
        # pipelined collectives above. Slight compile-time / code-size cost.
        "--xla_gpu_enable_while_loop_double_buffering": "true",
    }


# Flags intentionally NOT enabled by default (documented for reviewers / future work):
#   --xla_gpu_enable_triton_gemm=true            (ACCEPTED by this XLA) -- GEMM autotuning
#       path; can help or regress depending on shapes, and interacts with tokamax/pallas
#       kernels already used here. Left to the user; higher-risk, orthogonal to comm
#       overlap.
#   --xla_gpu_enable_command_buffer=FUSION       (ACCEPTED) -- CUDA-graph capture; can cut
#       launch overhead but has known interactions with collectives/NCCL and dynamic
#       shapes. Higher-risk; opt-in only.
#   --xla_gpu_enable_highest_priority_async_stream=true (ACCEPTED) -- puts async
#       collectives on a high-priority CUDA stream. Usually beneficial with LHS but
#       occasionally causes contention; left off pending GPU validation.
#   --xla_gpu_enable_nccl_comm_splitting=true    (ACCEPTED) -- NCCL communicator splitting;
#       topology/version sensitive. Off pending multi-node validation.
#   --xla_gpu_enable_pipelined_p2p=true          (ACCEPTED) -- for send/recv (pipeline
#       parallel / EP all-to-all). Not used yet; enable when EP lands.
# REJECTED by this XLA 0.9.2 build (parse-check "Unknown flag"), so NOT usable here:
#   --xla_gpu_enable_async_all_gather / _reduce_scatter / _all_reduce / _collective_permute
#   --xla_gpu_enable_pipelined_collectives
#   --xla_gpu_graph_level
#   --xla_gpu_lhs_enable_gpu_async_tracker


def _is_gpu_platform() -> bool:
    """Best-effort GPU detection that does NOT create the XLA backend.

    We must not import jax and trigger backend init here (that would consume XLA_FLAGS
    before we set them). We therefore detect GPU cheaply from the environment /
    filesystem. Heuristics, in order:
      * ``JAX_PLATFORMS`` explicitly requesting cpu/tpu -> not GPU.
      * ``JAX_PLATFORMS`` explicitly requesting cuda/gpu -> GPU.
      * ``CUDA_VISIBLE_DEVICES`` set to a non-empty, non-"-1" value -> GPU.
      * presence of an NVIDIA device node (/dev/nvidia0) -> GPU.
    Defaults to False (safe no-op) when nothing indicates a GPU.
    """
    platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if platforms:
        first = platforms.split(",")[0]
        if first in ("cpu", "tpu"):
            return False
        if first in ("cuda", "gpu", "rocm"):
            return True
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return cvd.strip() not in ("", "-1")
    # Fall back to hardware presence.
    return os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidiactl")


def build_gpu_xla_flags(existing: str | None = None) -> str:
    """Return the merged XLA_FLAGS string (defaults + preserved user flags).

    Pure/testable: does not touch ``os.environ``. User-provided ``existing`` flags are
    preserved and take precedence -- we only add defaults for keys the user did not set,
    and we append the user's original string *last* so a duplicate key resolves to the
    user's value (XLA honors the last occurrence of a repeated flag).
    """
    existing = (existing or "").strip()

    # Parse the keys the user already set (by flag name, ignoring value) so we don't
    # override the user's explicit choices.
    user_keys: set[str] = set()
    for tok in existing.split():
        if tok.startswith("--"):
            user_keys.add(tok.split("=", 1)[0])

    parts: list[str] = []
    for name, value in _default_flags().items():
        if name in user_keys:
            continue  # user already set this flag; leave it to them
        parts.append(f"{name}={value}")

    if existing:
        parts.append(existing)  # user flags last -> win on any duplicate

    return " ".join(parts).strip()


def _already_applied(current: str) -> bool:
    """True if our default set was already installed (idempotency guard)."""
    return _IDEMPOTENCY_MARKER in current


def configure_gpu_xla_flags(
    *,
    enable: bool = True,
    force: bool = False,
) -> str | None:
    """Install GPU XLA perf flags into ``os.environ['XLA_FLAGS']`` (idempotent).

    Must be called BEFORE the XLA backend is created (before
    ``jax.distributed.initialize()`` / first jax computation) -- see module docstring.

    Args:
      enable: master switch. If False (or ``OMEGALAX_DISABLE_XLA_PERF_FLAGS`` is truthy),
        do nothing and return None. Wire a CLI flag to this for per-run control.
      force: apply the GPU flags even if GPU is not detected (for testing / unusual
        setups). Normally leave False so this is a no-op on CPU/TPU.

    Returns:
      The final ``XLA_FLAGS`` string that is now in the environment, or None if we
      opted out / were a no-op. On process 0 the trainers log this for reproducibility.
    """
    if not enable:
        return None
    if os.environ.get(DISABLE_ENV_VAR, "0") not in ("0", "", "false", "False"):
        return None
    if not force and not _is_gpu_platform():
        # CPU/TPU: GPU flags are irrelevant; stay a no-op so we don't perturb anything.
        return None

    current = os.environ.get("XLA_FLAGS", "")
    if _already_applied(current):
        # Idempotent: our defaults are already present. Return what's there.
        return current

    merged = build_gpu_xla_flags(existing=current)
    os.environ["XLA_FLAGS"] = merged
    return merged
