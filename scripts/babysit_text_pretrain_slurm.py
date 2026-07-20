"""Monitor resumed pretraining jobs and activate Codex on anomalies."""

from __future__ import annotations

import datetime as dt
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "babysit_mode_job",
    [],
    "Mode mapping as mode:job_id:sbatch_path. Repeat once per mode.",
)
flags.DEFINE_multi_string(
    "babysit_resume_step",
    [],
    "Resume optimizer step as mode:step. Repeat once per monitored mode.",
)
flags.DEFINE_string("babysit_log_dir", None, "Slurm log directory.", required=True)
flags.DEFINE_string("babysit_wiki_path", None, "Progress log to append.", required=True)
flags.DEFINE_integer(
    "babysit_startup_steps", 100, "New optimizer steps in the frequent-check phase."
)
flags.DEFINE_integer(
    "babysit_startup_poll_seconds", 60, "Check interval during the frequent-check phase."
)
flags.DEFINE_integer("babysit_poll_seconds", 1200, "Check interval after the startup phase.")
flags.DEFINE_integer(
    "babysit_stall_seconds", 900, "Alert if a running job logs no new step for this long."
)
flags.DEFINE_integer("babysit_tail_lines", 120, "Log lines included in a Codex alert.")
flags.DEFINE_string(
    "babysit_codex_session_id",
    os.environ.get("CODEX_THREAD_ID"),
    "Codex session/thread id activated on an anomaly.",
)
flags.DEFINE_string("babysit_codex_model", None, "Optional Codex model override.")
flags.DEFINE_string(
    "babysit_codex_reasoning_effort", None, "Optional Codex reasoning effort override."
)
flags.DEFINE_integer("babysit_codex_timeout_seconds", 1800, "Codex alert timeout.")

_TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}
_METRIC_LINE_RE = re.compile(
    r"\bstep=(\d+)\b.*\bloss=([^\s]+).*\bgrad_norm=([^\s]+).*\btrain/lr=([^\s]+)"
)


def _timestamp() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _append(path: Path, text: str) -> None:
    with path.open("a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    print(text, flush=True)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _flag_value(name: str):
    try:
        return getattr(FLAGS, name)
    except flags.UnparsedFlagAccessError:
        return FLAGS[name].value


def job_status(job_id: str) -> tuple[str, str]:
    proc = _run(["sacct", "-j", job_id, "-X", "-n", "-o", "JobIDRaw,State"])
    for line in proc.stdout.splitlines():
        row = line.strip().split()
        if row and row[0] == job_id and len(row) >= 2:
            return row[1].split("+", 1)[0], "sacct"

    proc = _run(["squeue", "-j", job_id, "-h", "-o", "%T"])
    states = proc.stdout.strip().splitlines()
    if states:
        return states[0].strip(), "squeue"
    return "UNKNOWN", "unknown"


def _log_paths(log_dir: Path, job_id: str) -> list[Path]:
    return sorted(log_dir.glob(f"*_{job_id}.log"))


def _tail(path: Path, lines: int) -> str:
    proc = _run(["tail", "-n", str(lines), str(path)])
    return proc.stdout if proc.stdout else proc.stderr


def _metrics_after(log_dir: Path, job_id: str, after_step: int | None) -> list[tuple[int, str]]:
    by_step = {}
    for path in _log_paths(log_dir, job_id):
        with path.open(errors="replace") as f:
            for line in f:
                match = _METRIC_LINE_RE.search(line)
                if match is None:
                    continue
                step = int(match.group(1))
                if after_step is None or step > after_step:
                    by_step[step] = line.rstrip()
    return sorted(by_step.items())


def _metric_has_non_finite_value(line: str) -> bool:
    match = _METRIC_LINE_RE.search(line)
    if match is None:
        return False
    try:
        loss, grad_norm, learning_rate = (float(value) for value in match.groups()[1:])
    except ValueError:
        return False
    return not all(math.isfinite(value) for value in (loss, grad_norm, learning_rate))


def _tail_logs(log_dir: Path, job_id: str, lines: int) -> str:
    tails = [
        f"### `{path}`\n\n```text\n{_tail(path, lines)}\n```"
        for path in _log_paths(log_dir, job_id)
    ]
    return "\n\n".join(tails) if tails else "No Slurm log file found yet."


def _parse_mode_jobs() -> dict[str, dict[str, object]]:
    modes = {}
    for raw in _flag_value("babysit_mode_job"):
        mode, job_id, script = raw.split(":", 2)
        if mode in modes:
            raise ValueError(f"Duplicate babysit mode: {mode}")
        modes[mode] = {
            "job_id": job_id,
            "script": Path(script).expanduser().resolve(),
            "done": False,
        }
    if not modes:
        raise ValueError("Provide at least one --babysit_mode_job")
    return modes


def _parse_resume_steps() -> dict[str, int]:
    resume_steps = {}
    for raw in _flag_value("babysit_resume_step"):
        mode, step = raw.split(":", 1)
        if mode in resume_steps:
            raise ValueError(f"Duplicate babysit resume step for mode: {mode}")
        resume_steps[mode] = int(step)
    return resume_steps


def _requeue_self(_signum, _frame) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise SystemExit("Received requeue signal outside a Slurm job.")
    proc = _run(["scontrol", "requeue", job_id])
    if proc.returncode != 0:
        raise SystemExit(f"Failed to requeue babysitter job {job_id}: {proc.stderr}")
    print(f"Requeued babysitter job {job_id} before time limit.", flush=True)
    raise SystemExit(0)


def _prompt_codex(
    *,
    reason: str,
    mode: str,
    job_id: str,
    state: str,
    script_path: Path,
    log_dir: Path,
    wiki_path: Path,
    tail_text: str,
) -> tuple[bool, str]:
    session_id = _flag_value("babysit_codex_session_id")
    if not session_id:
        return False, "No Codex session id configured."

    prompt = f"""Statepassing pretraining babysitter alert.

Reason: {reason}
Mode: {mode}
Training job: {job_id}
Training state: {state}
Training sbatch: {script_path}
Log directory: {log_dir}
Progress wiki: {wiki_path}
Babysitter job: {os.environ.get("SLURM_JOB_ID", "unknown")}

Inspect the full current state and diagnose the cause. If intervention is needed,
make the smallest justified fix and run focused checks. Resubmit only after the
cause is understood. A resubmit must use the latest valid checkpoint with
resume=required, the identical W&B run id with wandb_resume=must, and unchanged
data, effective tokens per optimizer step, optimizer, and LR-schedule arguments.
Verify the next optimizer step and its LR against the checkpoint before launch.
Document a replacement run in runs.md before it starts. If the training job id
changes, submit a replacement babysitter for the new id. Do not blindly resubmit.

Recent log tail:

{tail_text}
"""
    cmd = ["codex", "exec"]
    model = _flag_value("babysit_codex_model")
    if model:
        cmd.extend(["-m", str(model)])
    reasoning_effort = _flag_value("babysit_codex_reasoning_effort")
    if reasoning_effort:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    cmd.extend(
        [
            "-C",
            str(Path(__file__).resolve().parents[1]),
            "resume",
            str(session_id),
            "-",
        ]
    )
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            input=prompt,
            timeout=_flag_value("babysit_codex_timeout_seconds"),
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"Codex alert timed out after {exc.timeout} seconds."
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    return proc.returncode == 0, output or f"Codex exited with {proc.returncode}."


def _alert_codex(
    *,
    reason: str,
    mode: str,
    info: dict[str, object],
    state: str,
    log_dir: Path,
    wiki_path: Path,
) -> None:
    job_id = str(info["job_id"])
    tail_text = _tail_logs(log_dir, job_id, int(_flag_value("babysit_tail_lines")))
    ok, output = _prompt_codex(
        reason=reason,
        mode=mode,
        job_id=job_id,
        state=state,
        script_path=Path(info["script"]),
        log_dir=log_dir,
        wiki_path=wiki_path,
        tail_text=tail_text,
    )
    _append(
        wiki_path,
        f"\n### Codex Babysitter Alert\n\nTime: {_timestamp()}\n\n"
        f"Mode: `{mode}`\n\nJob: `{job_id}`\n\nReason: `{reason}`\n\n"
        f"Session: `{_flag_value('babysit_codex_session_id')}`\n\nSuccess: `{ok}`\n\n"
        f"```text\n{output}\n```\n",
    )


def main(_) -> None:
    signal.signal(signal.SIGUSR1, _requeue_self)
    log_dir = Path(_flag_value("babysit_log_dir")).expanduser().resolve()
    wiki_path = Path(_flag_value("babysit_wiki_path")).expanduser().resolve()
    modes = _parse_mode_jobs()
    resume_steps = _parse_resume_steps()
    if set(modes) != set(resume_steps):
        raise ValueError("Provide exactly one --babysit_resume_step for every monitored mode")

    now = time.time()
    for mode, info in modes.items():
        resume_step = resume_steps[mode]
        info["startup_target"] = resume_step + int(_flag_value("babysit_startup_steps"))
        info["startup_done"] = False
        info["last_metric_step"] = resume_step
        info["last_progress_at"] = now
        info["was_running"] = False
    seen_states = {}

    _append(
        wiki_path,
        "\n## Slurm Babysitter Started\n\n"
        f"Time: {_timestamp()}\n\n"
        + "\n".join(
            f"- `{mode}` job `{info['job_id']}` through step `{info['startup_target']}`"
            for mode, info in modes.items()
        )
        + "\n",
    )

    while True:
        for mode, info in modes.items():
            if info["done"]:
                continue
            job_id = str(info["job_id"])
            state, source = job_status(job_id)
            if seen_states.get(mode) != state:
                _append(
                    wiki_path,
                    f"- `{_timestamp()}` mode `{mode}` job `{job_id}` state `{state}` via `{source}`.",
                )
                seen_states[mode] = state

            now = time.time()
            if state == "RUNNING" and not info["was_running"]:
                info["was_running"] = True
                info["last_progress_at"] = now
            metrics = _metrics_after(log_dir, job_id, int(info["last_metric_step"]))
            non_finite = []
            if metrics:
                info["last_metric_step"] = metrics[-1][0]
                info["last_progress_at"] = now
                non_finite = [line for _, line in metrics if _metric_has_non_finite_value(line)]
                print(
                    f"{_timestamp()} checked {mode} through step {metrics[-1][0]} "
                    f"({len(metrics)} new metric records).",
                    flush=True,
                )

            if state == "COMPLETED":
                if non_finite:
                    _alert_codex(
                        reason=f"completed with non-finite metric: {non_finite[-1]}",
                        mode=mode,
                        info=info,
                        state=state,
                        log_dir=log_dir,
                        wiki_path=wiki_path,
                    )
                info["done"] = True
                continue
            if state in _TERMINAL_STATES:
                reason = f"training job ended with state {state}"
                if non_finite:
                    reason += f" and non-finite metric: {non_finite[-1]}"
                _alert_codex(
                    reason=reason,
                    mode=mode,
                    info=info,
                    state=state,
                    log_dir=log_dir,
                    wiki_path=wiki_path,
                )
                info["done"] = True
                continue
            if state != "RUNNING":
                if state != "UNKNOWN":
                    info["was_running"] = False
                continue
            if non_finite:
                _alert_codex(
                    reason=f"non-finite metric: {non_finite[-1]}",
                    mode=mode,
                    info=info,
                    state=state,
                    log_dir=log_dir,
                    wiki_path=wiki_path,
                )

            if not info["startup_done"] and int(info["last_metric_step"]) >= int(
                info["startup_target"]
            ):
                info["startup_done"] = True
                _append(
                    wiki_path,
                    f"- `{_timestamp()}` mode `{mode}` passed its first "
                    f"`{_flag_value('babysit_startup_steps')}` resumed optimizer steps at "
                    f"step `{info['last_metric_step']}`.",
                )

            if now - float(info["last_progress_at"]) >= int(_flag_value("babysit_stall_seconds")):
                _alert_codex(
                    reason=(
                        f"no new logged optimizer step for at least "
                        f"{_flag_value('babysit_stall_seconds')} seconds"
                    ),
                    mode=mode,
                    info=info,
                    state=state,
                    log_dir=log_dir,
                    wiki_path=wiki_path,
                )
                info["last_progress_at"] = time.time()

        if all(bool(info["done"]) for info in modes.values()):
            break
        startup_active = any(
            not info["done"] and not info["startup_done"] for info in modes.values()
        )
        time.sleep(
            int(_flag_value("babysit_startup_poll_seconds"))
            if startup_active
            else int(_flag_value("babysit_poll_seconds"))
        )

    _append(wiki_path, f"\n## Slurm Babysitter Complete\n\nTime: {_timestamp()}\n")


if __name__ == "__main__":
    app.run(main)
