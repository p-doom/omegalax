"""Poll Slurm state for the Statepassing pretraining jobs and append wiki notes."""

from __future__ import annotations

import datetime as dt
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
    return sorted(log_dir.glob(f"*_{job_id}.log"))


def _tail(path: Path, lines: int) -> str:
    proc = _run(["tail", "-n", str(lines), str(path)])
    return proc.stdout if proc.stdout else proc.stderr


def main(_) -> None:
    job_ids = [job_id.strip() for job_id in FLAGS.monitor_job_ids if job_id.strip()]
    log_dir = Path(FLAGS.monitor_log_dir).expanduser().resolve()
    wiki_path = Path(FLAGS.monitor_wiki_path).expanduser().resolve()
    poll = 0
    seen_states: dict[str, str] = {}
    failures: dict[str, str] = {}

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
                tails = []
                for path in _log_paths(log_dir, job_id):
                    tails.append(
                        f"### `{path}`\n\n```text\n{_tail(path, FLAGS.monitor_tail_lines)}\n```"
                    )
                tail_text = "\n\n".join(tails) if tails else "No Slurm log file found yet."
                _append(
                    wiki_path,
                    f"\n## Slurm Job Failure\n\nTime: {_timestamp()}\n\n"
                    f"Job `{job_id}` ended with state `{state}`.\n\n{tail_text}\n",
                )

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
