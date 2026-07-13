"""Poll Slurm state for the Statepassing pretraining jobs and append wiki notes."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import subprocess
import time

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_list(
    "monitor_job_ids", None, "Comma-separated Slurm job ids to monitor.", required=True
)
flags.DEFINE_string("monitor_log_dir", None, "Slurm log directory for the run.", required=True)
flags.DEFINE_string(
    "monitor_wiki_path", None, "Progress documentation file to append.", required=True
)
flags.DEFINE_integer("monitor_poll_seconds", 1200, "Polling interval.")
flags.DEFINE_integer("monitor_max_polls", 0, "Maximum polls; 0 means until all jobs finish.")
flags.DEFINE_integer("monitor_tail_lines", 80, "Log tail lines to record for failed jobs.")
flags.DEFINE_string(
    "monitor_codex_session_id",
    os.environ.get("CODEX_THREAD_ID"),
    "Codex session/thread id to prompt for failures and periodic health checks.",
)
flags.DEFINE_integer(
    "monitor_codex_interval_seconds",
    0,
    "If >0, prompt Codex every N seconds after any monitored job starts running.",
)
flags.DEFINE_bool("monitor_codex_on_failure", False, "Prompt Codex immediately on failures.")
flags.DEFINE_integer("monitor_codex_timeout_seconds", 1800, "Codex prompt timeout.")

_DONE_STATES = {
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
_FAIL_STATES = _DONE_STATES - {"COMPLETED"}


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


def _append(path: Path, text: str) -> None:
    with path.open("a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _flag_value(name: str):
    try:
        return getattr(FLAGS, name)
    except flags.UnparsedFlagAccessError:
        return FLAGS[name].value


def job_status(job_id: str) -> tuple[str, str]:
    proc = _run(["sacct", "-j", job_id, "-X", "-n", "-o", "JobIDRaw,State"])
    rows = [line.strip().split() for line in proc.stdout.splitlines() if line.strip()]
    for row in rows:
        if row and row[0] == job_id and len(row) >= 2:
            return row[1].split("+", 1)[0], "sacct"

    proc = _run(["squeue", "-j", job_id, "-h", "-o", "%T"])
    state = proc.stdout.strip().splitlines()
    if state:
        return state[0].strip(), "squeue"
    return "UNKNOWN", "unknown"


def _log_paths(log_dir: Path, job_id: str) -> list[Path]:
    return sorted(log_dir.rglob(f"*_{job_id}.log"))


def _tail(path: Path, lines: int) -> str:
    proc = _run(["tail", "-n", str(lines), str(path)])
    return proc.stdout if proc.stdout else proc.stderr


def _tail_logs_for_jobs(log_dir: Path, job_ids: list[str], lines: int) -> str:
    tails = []
    for job_id in job_ids:
        for path in _log_paths(log_dir, job_id):
            tails.append(f"### `{path}`\n\n```text\n{_tail(path, lines)}\n```")
    return "\n\n".join(tails) if tails else "No Slurm log file found yet."


def _prompt_codex(
    *,
    reason: str,
    job_ids: list[str],
    states: dict[str, str],
    log_dir: Path,
    wiki_path: Path,
    tail_text: str,
) -> tuple[bool, str]:
    session_id = _flag_value("monitor_codex_session_id")
    if not session_id:
        return False, "No --monitor_codex_session_id and CODEX_THREAD_ID was unset."

    prompt = f"""Statepassing pretraining monitor update.

Reason: {reason}
Jobs: {", ".join(job_ids)}
States: {states}
Log directory: {log_dir}
Progress wiki: {wiki_path}

Inspect the current state, check whether the run looks healthy, and intervene only if needed.
If there is a failure, debug before any resubmit and document the cause/fix.

Recent log tail:

{tail_text}
"""
    try:
        proc = subprocess.run(
            [
                "codex",
                "exec",
                "-C",
                str(Path(__file__).resolve().parents[1]),
                "resume",
                str(session_id),
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            input=prompt,
            timeout=_flag_value("monitor_codex_timeout_seconds"),
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"codex prompt timed out after {exc.timeout} seconds"
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    return proc.returncode == 0, output or f"codex exited with {proc.returncode}"


def _append_codex_prompt_result(
    wiki_path: Path,
    *,
    reason: str,
    ok: bool,
    output: str,
) -> None:
    _append(
        wiki_path,
        f"\n### Codex Monitor Prompt\n\nTime: {_timestamp()}\n\n"
        f"Reason: `{reason}`\n\n"
        f"Session: `{_flag_value('monitor_codex_session_id')}`\n\n"
        f"Success: `{ok}`\n\n"
        f"```text\n{output}\n```\n",
    )


def main(_) -> None:
    job_ids = [job_id.strip() for job_id in FLAGS.monitor_job_ids if job_id.strip()]
    log_dir = Path(FLAGS.monitor_log_dir).expanduser().resolve()
    wiki_path = Path(FLAGS.monitor_wiki_path).expanduser().resolve()
    poll = 0
    seen_states: dict[str, str] = {}
    failures: dict[str, str] = {}
    first_running_at: float | None = None
    last_codex_prompt_at: float | None = None

    _append(
        wiki_path,
        f"\n## Slurm Monitor Started\n\nTime: {_timestamp()}\n\nJobs: {', '.join(job_ids)}\n",
    )

    while True:
        poll += 1
        states = {}
        for job_id in job_ids:
            state, source = job_status(job_id)
            states[job_id] = state
            if seen_states.get(job_id) != state:
                _append(
                    wiki_path,
                    f"- `{_timestamp()}` job `{job_id}` state `{state}` via `{source}`.",
                )
                seen_states[job_id] = state
            if state in _FAIL_STATES and job_id not in failures:
                failures[job_id] = state
                tail_text = _tail_logs_for_jobs(log_dir, [job_id], FLAGS.monitor_tail_lines)
                _append(
                    wiki_path,
                    f"\n## Slurm Job Failure\n\nTime: {_timestamp()}\n\n"
                    f"Job `{job_id}` ended with state `{state}`.\n\n{tail_text}\n",
                )
                if FLAGS.monitor_codex_on_failure:
                    ok, output = _prompt_codex(
                        reason=f"job {job_id} failed with {state}",
                        job_ids=job_ids,
                        states=states,
                        log_dir=log_dir,
                        wiki_path=wiki_path,
                        tail_text=tail_text,
                    )
                    _append_codex_prompt_result(
                        wiki_path,
                        reason=f"job {job_id} failed with {state}",
                        ok=ok,
                        output=output,
                    )

        now = time.monotonic()
        if states and any(state == "RUNNING" for state in states.values()):
            if first_running_at is None:
                first_running_at = now
                last_codex_prompt_at = now
            interval = int(FLAGS.monitor_codex_interval_seconds or 0)
            if (
                interval > 0
                and last_codex_prompt_at is not None
                and now - last_codex_prompt_at >= interval
            ):
                tail_text = _tail_logs_for_jobs(log_dir, job_ids, FLAGS.monitor_tail_lines)
                ok, output = _prompt_codex(
                    reason=f"periodic health check after {interval} seconds",
                    job_ids=job_ids,
                    states=states,
                    log_dir=log_dir,
                    wiki_path=wiki_path,
                    tail_text=tail_text,
                )
                _append_codex_prompt_result(
                    wiki_path,
                    reason=f"periodic health check after {interval} seconds",
                    ok=ok,
                    output=output,
                )
                last_codex_prompt_at = now

        if states and all(state in _DONE_STATES for state in states.values()):
            break
        if FLAGS.monitor_max_polls and poll >= FLAGS.monitor_max_polls:
            _append(
                wiki_path,
                f"\n## Slurm Monitor Stopped\n\nTime: {_timestamp()}\n\n"
                f"Stopped after {poll} polls. Current states: {states}\n",
            )
            break
        time.sleep(FLAGS.monitor_poll_seconds)

    if failures:
        raise SystemExit(f"Monitored job failures: {failures}")
    _append(
        wiki_path,
        f"\n## Slurm Monitor Complete\n\nTime: {_timestamp()}\n\nFinal states: {states}\n",
    )


if __name__ == "__main__":
    app.run(main)
