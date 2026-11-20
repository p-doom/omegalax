"""Launcher: run the unmodified production train script, optionally under the profiler.

Same trick as the earlier GDN work: import ``scripts/train_text_pretrain.py`` as a
module so its absl flags get defined, then hand ``main`` to ``app.run``. That gives
a patch point between "flags exist" and "training starts" without editing anything
under ``scripts/`` or ``omegalax/``.

Environment:
  MFU_TRACE_DIR     if set, capture a profiler trace
  MFU_TRACE_START   first step to include in the trace (default 5, i.e. after
                    compilation and after the two warmup steps)
  MFU_TRACE_STEPS   how many steps to trace (default 3)
  MFU_RELAX_INDEX   set to 1 to disable the index segments-per-update contract,
                    which is irrelevant for a short throughput run

The trace is started and stopped from inside ``StepTimer.step``, which the training
loop calls exactly once per step. That makes the step boundaries of the trace line
up with the logged steps instead of with wall-clock guesses.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_TRAIN = _REPO / "scripts" / "train_text_pretrain.py"



def _maybe_log_memory(step: int) -> None:
    """Report allocator peak once, after the step given by MFU_MEMLOG.

    perf.log_device_memory exists but is not called on the benchmark path, so the
    peak never reaches the log. Peak matters here because the whole cascade-vs-FSDP
    question is a memory-pressure question: two configurations that differ in speed
    but not in peak would refute that explanation.
    """
    want = os.environ.get("MFU_MEMLOG")
    if not want or step != int(want):
        return
    import jax

    for d in jax.local_devices():
        try:
            st = d.memory_stats() or {}
        except Exception as exc:
            print(f"[mfu-mem] dev={d.id}: nicht verfuegbar ({exc})", flush=True)
            continue
        g = lambda k: st.get(k, 0) / 1e9
        print(
            f"[mfu-mem] step={step} dev={d.id} "
            f"peak={g('peak_bytes_in_use'):.2f}GB in_use={g('bytes_in_use'):.2f}GB "
            f"limit={g('bytes_limit'):.2f}GB largest_free={g('largest_free_block_bytes'):.2f}GB",
            flush=True,
        )


def install_profiler() -> None:
    trace_dir = os.environ.get("MFU_TRACE_DIR")
    if not trace_dir and not os.environ.get("MFU_MEMLOG"):
        return
    import jax

    from omegalax.trainers import perf

    start = int(os.environ.get("MFU_TRACE_START", "5"))
    count = int(os.environ.get("MFU_TRACE_STEPS", "3"))
    original = perf.StepTimer.step
    state = {"n": 0, "active": False}

    def patched(self, block_on=None):
        state["n"] += 1
        _maybe_log_memory(state["n"])
        if trace_dir and state["n"] == start:
            os.makedirs(trace_dir, exist_ok=True)
            jax.profiler.start_trace(trace_dir, create_perfetto_trace=True)
            state["active"] = True
            print(f"[mfu] profiler trace gestartet bei Step {state['n']}", flush=True)
        elif trace_dir and state["active"] and state["n"] >= start + count:
            jax.profiler.stop_trace()
            state["active"] = False
            print(f"[mfu] profiler trace gestoppt bei Step {state['n']} -> {trace_dir}", flush=True)
        return original(self, block_on)

    perf.StepTimer.step = patched
    if trace_dir:
        print(f"[mfu] Profiler aktiv: Steps {start}..{start + count - 1} -> {trace_dir}", flush=True)
    if os.environ.get("MFU_MEMLOG"):
        print(f"[mfu] Speicher-Report bei Step {os.environ['MFU_MEMLOG']}", flush=True)


# grain spawns data-loader workers by re-importing this module; without the guard
# every worker would re-run the patching and re-exec the train script.

def install_budget_report() -> None:
    """Print the parts of the step budget the default log line omits.

    The trainer computes data_wait_s and data_wait_frac (pretrain.py, around line
    1060) but only puts them into window_metrics; with wandb disabled they never
    reach the log. Without them the step budget cannot be closed: a profiler trace
    of the baseline accounted for ~2437 ms of GPU kernel time against ~2840 ms of
    wall clock, leaving ~14 % unattributed.
    """
    if os.environ.get("MFU_BUDGET", "0") != "1":
        return
    # Patch the name in pretrain's namespace, not in perf. pretrain does
    # "from omegalax.trainers.perf import maybe_log_step_metrics" at module level,
    # so by the time we get here the name is already bound there and patching perf
    # has no effect — the first version of this hook printed "aktiv" and then never
    # fired, which is exactly that mistake.
    from omegalax.trainers import perf, pretrain as _P

    original = _P.maybe_log_step_metrics

    def patched(step_to_log, metrics_to_log, step_delta, **kw):
        out = original(step_to_log, metrics_to_log, step_delta, **kw)
        try:
            secs = step_delta.total_seconds()
            dw = float(metrics_to_log.get("data_wait_s", 0.0))
            print(
                f"[mfu-budget] step={step_to_log} step_s={secs:.3f} "
                f"data_wait_s={dw:.4f} data_wait_frac={dw / secs if secs else 0:.4f}",
                flush=True,
            )
        except Exception as exc:
            print(f"[mfu-budget] nicht verfuegbar ({exc})", flush=True)
        return out

    _P.maybe_log_step_metrics = patched
    perf.maybe_log_step_metrics = patched
    print("[mfu] Budget-Report aktiv", flush=True)


if __name__ == "__main__":
    install_profiler()
    install_budget_report()

    _spec = importlib.util.spec_from_file_location("train_text_pretrain_mfu", _TRAIN)
    _train = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _train
    _spec.loader.exec_module(_train)

    if os.environ.get("MFU_RELAX_INDEX") == "1":
        _train._validate_trimmed_index_runtime_config = lambda *a, **k: None
        print("[mfu] index segments-per-update contract check disabled", flush=True)

    from absl import app

    sys.argv[0] = str(_TRAIN)
    app.run(_train.main)
