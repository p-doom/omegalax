"""Robust JAX distributed launch across single-/multi-process SLURM modes.

Why this module exists
======================
The entrypoints used to call a bare ``jax.distributed.initialize()`` at startup.
That is wrong in two common single-node cases:

1. **Single-node, single-process, 4 GPUs** (the interactive/dev case:
   ``salloc --gres=gpu:4`` then ``python -m ...`` once). A bare
   ``jax.distributed.initialize()`` still finds the SLURM env vars, auto-detects
   the allocation, decides it is "process 0 of N", and -- because JAX's SLURM/GPU
   auto-detect path defaults ``local_device_ids`` to **a single device per
   process** -- grabs only **one** local GPU. ``jax.devices()`` then returns 1
   device and you silently train on a quarter of the node. (See the note on
   ``local_device_ids`` in ``jax.distributed.initialize``'s docstring: "defaults
   to all local devices being visible to the process except when processes are
   launched via Slurm and Open MPI on GPUs. In that case, it will default to a
   single device per process.")

2. **Single-node, multi-process** (one task per GPU via
   ``srun --ntasks-per-node=4``). A bare ``initialize()`` frequently hangs on the
   coordinator topology exchange under a nested ``srun``/``salloc`` (each of the
   4 tasks races to be coordinator; the auto-detected coordinator address/port is
   unreliable for the nested case).

The fix: detect the launch mode from the SLURM environment (env-driven, decided
*before* any JAX backend is created -- never at runtime failure) and either skip
``initialize()`` entirely (single process -> JAX sees all local GPUs by default)
or call it with an explicit, derived ``(coordinator_address, num_processes,
process_id)`` so the exchange is deterministic.

Modes
-----
* **SINGLE_PROCESS**: exactly one process for this node/step (or no SLURM env at
  all -- a plain workstation, or ``OMEGALAX_FORCE_SINGLE_PROCESS=1``). We do NOT
  call ``jax.distributed.initialize()``; JAX exposes every local device by
  default, so ``len(jax.devices()) == GPUs-on-node``. This is the headline fix.
* **MULTI_PROCESS**: more than one task in the step. We derive the coordinator
  host from the SLURM nodelist (first host) + a fixed, job-derived port, and pass
  ``num_processes=SLURM_NTASKS``, ``process_id=SLURM_PROCID`` explicitly. This
  covers both single-node-multi-process (one task per GPU; each process then owns
  exactly one local GPU, JAX's SLURM/GPU default) and the multi-node production
  path (N nodes x T tasks/node).

Multi-node assumptions (UNTESTABLE here -- no multi-node access)
----------------------------------------------------------------
The multi-node path is the same code path as single-node-multi-process: it is
driven entirely by ``SLURM_NTASKS`` / ``SLURM_PROCID`` / the nodelist, which SLURM
sets consistently across nodes. Assumptions:

* ``SLURM_PROCID`` is a dense global range ``0..SLURM_NTASKS-1`` across all nodes
  (SLURM guarantees this for a single step). JAX requires exactly this.
* The coordinator host is the first host in ``SLURM_STEP_NODELIST`` (falling back
  to ``SLURM_NODELIST``), reachable from every node. On this cluster the compute
  hostnames (e.g. ``hkn0533``) are directly routable, which is the normal HPC
  case.
* The port is derived deterministically from ``SLURM_JOB_ID`` so every process in
  the step agrees on it without extra coordination, and distinct concurrent jobs
  pick distinct ports. This matches JAX's own SLURM port heuristic
  (``JOB_ID % 4096 + 61440``); we reuse it so behavior is unchanged from what a
  bare ``initialize()`` would have derived, only made explicit.
* ``local_device_ids`` is left ``None`` so JAX applies its SLURM/GPU default of
  one local device per task -- correct when launching one task per GPU (the
  documented omegalax multi-node layout). We do NOT force all local GPUs onto a
  single multi-node process.

Ordering
--------
Like :mod:`omegalax.distributed.xla_flags`, this must run BEFORE the XLA backend
is created (before the first ``jax.devices()`` / computation). ``initialize()``
itself raises if the backend already exists. :func:`init_distributed` is
idempotent and safe to call after JAX is already initialized.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import os

logger = logging.getLogger(__name__)

# --- Env vars (SLURM + omegalax overrides) -----------------------------------
# JAX's own SlurmCluster reads SLURM_STEP_NODELIST / SLURM_NTASKS / SLURM_PROCID /
# SLURM_LOCALID; we mirror those and add a couple of fallbacks so detection is
# robust across srun/salloc/sbatch step-vs-job scoping.
_ENV_JOB_ID = "SLURM_JOB_ID"
_ENV_STEP_NODELIST = "SLURM_STEP_NODELIST"
_ENV_NODELIST = "SLURM_NODELIST"
_ENV_NTASKS = "SLURM_NTASKS"
_ENV_STEP_NUM_TASKS = "SLURM_STEP_NUM_TASKS"
_ENV_PROCID = "SLURM_PROCID"
_ENV_LOCALID = "SLURM_LOCALID"
_ENV_NNODES = "SLURM_NNODES"
_ENV_STEP_NUM_NODES = "SLURM_STEP_NUM_NODES"

# omegalax escape hatch: force SINGLE_PROCESS regardless of SLURM env (e.g. a
# 1-process debug run inside a larger allocation that should see all local GPUs).
FORCE_SINGLE_PROCESS_ENV = "OMEGALAX_FORCE_SINGLE_PROCESS"
# Optional explicit coordinator port override (else derived from SLURM_JOB_ID).
COORDINATOR_PORT_ENV = "OMEGALAX_COORDINATOR_PORT"

# Same ephemeral-range heuristic JAX's SlurmCluster uses: port in
# [65535 - 2**12 + 1, 65535] == [61440, 65535], keyed on the job id so all tasks
# of a step agree and distinct jobs rarely collide.
_PORT_BASE = 65535 - 2**12 + 1  # 61440
_PORT_SPAN = 2**12  # 4096
_DEFAULT_PORT = 61440  # used when SLURM_JOB_ID is absent/unparseable


class LaunchMode(str, enum.Enum):
    """Detected launch mode."""

    SINGLE_PROCESS = "single_process"
    MULTI_PROCESS = "multi_process"


@dataclasses.dataclass(frozen=True)
class LaunchInfo:
    """Result of mode detection / distributed init.

    Attributes:
      mode: the detected :class:`LaunchMode`.
      num_processes: number of JAX processes (1 for SINGLE_PROCESS).
      process_id: this process' id (0 for SINGLE_PROCESS).
      coordinator_address: ``host:port`` for the coordinator, or None in
        SINGLE_PROCESS (no coordinator is created).
      local_process_id: SLURM_LOCALID if known (local rank within the node), else
        None. Informational only.
      num_nodes: number of nodes in the step if known, else None. Informational.
      initialized: True if this call actually invoked
        ``jax.distributed.initialize()`` (False if it was skipped -- single
        process -- or was already initialized).
      reason: short human-readable explanation of the decision (for logging).
    """

    mode: LaunchMode
    num_processes: int
    process_id: int
    coordinator_address: str | None
    local_process_id: int | None
    num_nodes: int | None
    initialized: bool
    reason: str


def _int_env(env: dict[str, str], key: str) -> int | None:
    val = env.get(key)
    if val is None or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() not in ("", "0", "false", "no", "off")


def parse_nodelist_first_host(nodelist: str) -> str:
    """Return the first concrete hostname from a SLURM nodelist string.

    Handles the compact SLURM forms, e.g.::

        'hkn0533'                         -> 'hkn0533'
        'hkn0533,hkn0534'                 -> 'hkn0533'
        'hkn[0533-0536]'                  -> 'hkn0533'
        'hkn[0533,0535-0537]'             -> 'hkn0533'
        'hkn[0533-0536],gpu[01-02]'       -> 'hkn0533'

    Mirrors JAX's own SlurmCluster.get_coordinator_address host parsing so the
    coordinator host is identical to what a bare ``initialize()`` would derive.
    """
    nodelist = nodelist.strip()
    if not nodelist:
        raise ValueError("empty SLURM nodelist")
    # Find the first delimiter that ends the first host token: ',' (separates
    # top-level hosts) or '[' (start of a numeric range for the first prefix).
    ind = next((i for i, ch in enumerate(nodelist) if ch in {",", "["}), len(nodelist))
    if ind == len(nodelist) or nodelist[ind] == ",":
        # 'host' or 'host,rest' -> the bare first host.
        return nodelist[:ind]
    # 'prefix[range],rest' -> prefix + first number inside the bracket.
    prefix = nodelist[:ind]
    suffix = nodelist[ind + 1 :]  # after '['
    # First number token ends at the first ',' , '-' or ']'.
    ind2 = next((i for i, ch in enumerate(suffix) if ch in {",", "-", "]"}), None)
    first_num = suffix if ind2 is None else suffix[:ind2]
    return f"{prefix}{first_num}"


def _derive_port(env: dict[str, str]) -> int:
    override = env.get(COORDINATOR_PORT_ENV)
    if override and override.strip():
        try:
            return int(override.strip())
        except ValueError:
            logger.warning(
                "%s=%r is not an int; falling back to SLURM-derived port.",
                COORDINATOR_PORT_ENV,
                override,
            )
    job_id = _int_env(env, _ENV_JOB_ID)
    if job_id is None:
        return _DEFAULT_PORT
    return job_id % _PORT_SPAN + _PORT_BASE


def _coordinator_address(env: dict[str, str]) -> str:
    # Prefer SLURM_STEP_NODELIST (scoped to the current step, what JAX uses),
    # fall back to SLURM_NODELIST (job allocation) for salloc-without-step cases.
    nodelist = env.get(_ENV_STEP_NODELIST) or env.get(_ENV_NODELIST)
    if not nodelist:
        raise ValueError(
            "multi-process launch detected but neither SLURM_STEP_NODELIST nor "
            "SLURM_NODELIST is set; cannot derive coordinator address."
        )
    host = parse_nodelist_first_host(nodelist)
    if not host:
        raise ValueError(f"could not parse a coordinator host from nodelist {nodelist!r}.")
    return f"{host}:{_derive_port(env)}"


def _ntasks(env: dict[str, str]) -> int | None:
    """Number of tasks in the step. Prefer step-scoped, fall back to job-scoped."""
    return _int_env(env, _ENV_STEP_NUM_TASKS) or _int_env(env, _ENV_NTASKS)


def detect_mode(env: dict[str, str] | None = None) -> LaunchInfo:
    """Pure mode detection from the (SLURM) environment -- no JAX calls.

    Decides SINGLE_PROCESS vs MULTI_PROCESS and, for the latter, derives the
    coordinator address / num_processes / process_id. Split out from
    :func:`init_distributed` so it is unit-testable on CPU by monkeypatching
    ``os.environ`` -- it never touches JAX.

    ``initialized`` is always False here (this function does not call JAX); it is
    a placeholder the caller fills in.
    """
    if env is None:
        env = dict(os.environ)

    # Escape hatch: force single process regardless of SLURM env.
    if _truthy(env.get(FORCE_SINGLE_PROCESS_ENV)):
        return LaunchInfo(
            mode=LaunchMode.SINGLE_PROCESS,
            num_processes=1,
            process_id=0,
            coordinator_address=None,
            local_process_id=_int_env(env, _ENV_LOCALID),
            num_nodes=_int_env(env, _ENV_NNODES) or _int_env(env, _ENV_STEP_NUM_NODES),
            initialized=False,
            reason=f"{FORCE_SINGLE_PROCESS_ENV} set -> forced single process (all local GPUs).",
        )

    ntasks = _ntasks(env)
    procid = _int_env(env, _ENV_PROCID)
    num_nodes = _int_env(env, _ENV_NNODES) or _int_env(env, _ENV_STEP_NUM_NODES)
    localid = _int_env(env, _ENV_LOCALID)

    # No usable SLURM task info -> plain workstation / bare python. Single process.
    if ntasks is None:
        return LaunchInfo(
            mode=LaunchMode.SINGLE_PROCESS,
            num_processes=1,
            process_id=0,
            coordinator_address=None,
            local_process_id=localid,
            num_nodes=num_nodes,
            initialized=False,
            reason="no SLURM_NTASKS/SLURM_STEP_NUM_TASKS -> single process (all local GPUs).",
        )

    # One task total -> single process even inside a SLURM allocation. This is the
    # headline fix: a bare initialize() here would auto-detect SLURM+GPU and grab
    # only ONE local GPU. Skipping it lets JAX expose ALL local GPUs.
    if ntasks <= 1:
        return LaunchInfo(
            mode=LaunchMode.SINGLE_PROCESS,
            num_processes=1,
            process_id=0,
            coordinator_address=None,
            local_process_id=localid,
            num_nodes=num_nodes,
            initialized=False,
            reason=(
                f"SLURM task count == {ntasks} -> single process; skipping "
                "jax.distributed.initialize() so JAX sees ALL local GPUs "
                "(avoids the SLURM/GPU 1-device-per-process default)."
            ),
        )

    # Multi-process: derive coordinator + ids explicitly (never rely on JAX's
    # auto-detect, which races/hangs under nested srun).
    if procid is None:
        raise ValueError(
            f"multi-process launch (tasks={ntasks}) but SLURM_PROCID is unset; "
            "cannot assign a JAX process_id."
        )
    if not (0 <= procid < ntasks):
        raise ValueError(
            f"SLURM_PROCID={procid} is out of range for SLURM_NTASKS={ntasks}; "
            "expected a dense 0..ntasks-1 process id."
        )
    coordinator = _coordinator_address(env)
    return LaunchInfo(
        mode=LaunchMode.MULTI_PROCESS,
        num_processes=ntasks,
        process_id=procid,
        coordinator_address=coordinator,
        local_process_id=localid,
        num_nodes=num_nodes,
        initialized=False,
        reason=(
            f"SLURM task count == {ntasks} -> multi-process; "
            f"coordinator={coordinator} process_id={procid}."
        ),
    )


def init_distributed(
    env: dict[str, str] | None = None,
    *,
    initialization_timeout: int | None = None,
) -> LaunchInfo:
    """Detect the launch mode and initialize JAX distributed accordingly.

    * SINGLE_PROCESS -> does NOT call ``jax.distributed.initialize()`` (JAX then
      exposes every local device -> ``len(jax.devices()) == GPUs-on-node``).
    * MULTI_PROCESS -> calls ``jax.distributed.initialize(coordinator_address=...,
      num_processes=..., process_id=...)`` with values derived from SLURM env. In
      this mode JAX's SLURM/GPU default gives each process exactly one local
      device -- correct for one-task-per-GPU launches.

    Idempotent / safe: if JAX distributed is already initialized, it is not
    re-initialized (returns ``initialized=False``). Must be called before the XLA
    backend is created (before the first ``jax.devices()`` / computation) in the
    MULTI_PROCESS case; ``jax.distributed.initialize`` enforces this and raises
    otherwise.

    Args:
      env: environment mapping (defaults to ``os.environ``); injectable for tests.
      initialization_timeout: optional override for JAX's coordinator connection
        timeout (seconds). None uses JAX's default (300s).

    Returns:
      A :class:`LaunchInfo` describing the detected mode and the actual action
      taken.
    """
    # Import jax lazily so importing this module (e.g. in the pure-CPU unit test)
    # does not pull in / initialize a backend.
    import jax

    info = detect_mode(env)

    if info.mode is LaunchMode.SINGLE_PROCESS:
        logger.info("omegalax launch: %s", info.reason)
        # Nothing to initialize; JAX will expose all local devices by default.
        return info

    # MULTI_PROCESS. Guard against double init (idempotent).
    if jax.distributed.is_initialized():
        logger.info(
            "omegalax launch: jax.distributed already initialized; not re-initializing (%s).",
            info.reason,
        )
        return dataclasses.replace(info, initialized=False)

    kwargs: dict = {
        "coordinator_address": info.coordinator_address,
        "num_processes": info.num_processes,
        "process_id": info.process_id,
    }
    if initialization_timeout is not None:
        kwargs["initialization_timeout"] = initialization_timeout
    logger.info("omegalax launch: %s -> jax.distributed.initialize(%s)", info.reason, kwargs)
    jax.distributed.initialize(**kwargs)
    return dataclasses.replace(info, initialized=True)


__all__ = [
    "LaunchMode",
    "LaunchInfo",
    "detect_mode",
    "init_distributed",
    "parse_nodelist_first_host",
    "FORCE_SINGLE_PROCESS_ENV",
    "COORDINATOR_PORT_ENV",
]
