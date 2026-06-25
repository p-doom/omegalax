"""Monitor Statepassing pretraining jobs and prompt Codex on failures."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import re
import subprocess
import time

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "babysit_mode_job",
    [],
    "Mode mapping as mode:job_id:sbatch_path. Repeat once per mode.",
)
flags.DEFINE_string("babysit_log_dir", None, "Slurm log directory for the run.", required=True)
flags.DEFINE_string(
    "babysit_wiki_path", None, "Progress documentation file to append.", required=True
)
flags.DEFINE_integer("babysit_poll_seconds", 1200, "Polling interval.")
flags.DEFINE_integer("babysit_tail_lines", 80, "Log tail lines to record for failed jobs.")
flags.DEFINE_integer("babysit_max_restarts", 3, "Maximum resubmits per mode.")
flags.DEFINE_bool(
    "babysit_auto_resubmit",
    False,
    "If true, resubmit after Codex is prompted. Default is to stop and wait for intervention.",
)
flags.DEFINE_string(
    "babysit_codex_session_id",
    os.environ.get("CODEX_THREAD_ID"),
    "Codex session/thread id to prompt on failed jobs.",
)
flags.DEFINE_integer("babysit_codex_timeout_seconds", 1800, "Codex prompt timeout.")

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
_RESTART_STATES = {
    "FAILED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


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


def _submit(script_path: Path) -> str:
    proc = _run(["sbatch", str(script_path)])
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed for {script_path}:\n{proc.stderr}\n{proc.stdout}")
    match = re.search(r"Submitted batch job (\d+)", proc.stdout)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output for {script_path}: {proc.stdout}")
    return match.group(1)


def _prompt_codex(
    *,
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
        return False, "No --babysit_codex_session_id and CODEX_THREAD_ID was unset."

    prompt = f"""A Statepassing pretraining Slurm job failed and needs debugging before any resubmit.

Mode: {mode}
Job id: {job_id}
State: {state}
Sbatch script: {script_path}
Log directory: {log_dir}
Progress wiki: {wiki_path}

Do not blindly resubmit. Inspect the failure, patch code or scripts if needed,
document the cause/fix in the progress wiki, run focused checks, and only then
decide whether and how to resubmit.

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
            timeout=_flag_value("babysit_codex_timeout_seconds"),
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"codex prompt timed out after {exc.timeout} seconds"
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    return proc.returncode == 0, output or f"codex exited with {proc.returncode}"


def _should_auto_resubmit_after_codex(codex_ok: bool) -> bool:
    return codex_ok and bool(_flag_value("babysit_auto_resubmit"))


def _parse_mode_jobs() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for raw in FLAGS.babysit_mode_job:
        mode, job_id, script = raw.split(":", 2)
        if mode in out:
            raise ValueError(f"Duplicate babysit mode: {mode}")
        out[mode] = {
            "job_id": job_id,
            "script": Path(script).expanduser().resolve(),
            "restarts": 0,
            "done": False,
        }
    if not out:
        raise ValueError("Provide at least one --babysit_mode_job")
    return out


def main(_) -> None:
    log_dir = Path(FLAGS.babysit_log_dir).expanduser().resolve()
    wiki_path = Path(FLAGS.babysit_wiki_path).expanduser().resolve()
    modes = _parse_mode_jobs()
    seen_states: dict[str, str] = {}

    _append(
        wiki_path,
        "\n## Slurm Babysitter Started\n\n"
        f"Time: {_timestamp()}\n\n"
        + "\n".join(
            f"- `{mode}` job `{info['job_id']}` script `{info['script']}`"
            for mode, info in modes.items()
        )
        + "\n",
    )

    while True:
        all_done = True
        for mode, info in modes.items():
            if info["done"]:
                continue
            all_done = False
            job_id = str(info["job_id"])
            state, source = job_status(job_id)
            state_key = f"{mode}:{job_id}"
            if seen_states.get(state_key) != state:
                _append(
                    wiki_path,
                    f"- `{_timestamp()}` mode `{mode}` job `{job_id}` state `{state}` via `{source}`.",
                )
                seen_states[state_key] = state

            if state == "COMPLETED":
                info["done"] = True
                _append(wiki_path, f"- `{_timestamp()}` mode `{mode}` completed as job `{job_id}`.")
                continue

            if state in _RESTART_STATES:
                tails = []
                for path in _log_paths(log_dir, job_id):
                    tails.append(
                        f"### `{path}`\n\n```text\n{_tail(path, FLAGS.babysit_tail_lines)}\n```"
                    )
                tail_text = "\n\n".join(tails) if tails else "No Slurm log file found yet."
                _append(
                    wiki_path,
                    f"\n## Babysitter Failure Intervention\n\nTime: {_timestamp()}\n\n"
                    f"Mode `{mode}` job `{job_id}` ended with state `{state}`.\n\n{tail_text}\n",
                )
                codex_ok, codex_output = _prompt_codex(
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
                    f"\n### Codex Failure Prompt\n\nTime: {_timestamp()}\n\n"
                    f"Session: `{_flag_value('babysit_codex_session_id')}`\n\n"
                    f"Success: `{codex_ok}`\n\n"
                    f"```text\n{codex_output}\n```\n",
                )
                if not _should_auto_resubmit_after_codex(codex_ok):
                    info["done"] = True
                    reason = (
                        "Codex prompt failed"
                        if not codex_ok
                        else "requires Codex/manual intervention"
                    )
                    _append(
                        wiki_path,
                        f"- `{_timestamp()}` mode `{mode}` {reason}; "
                        "not resubmitting automatically.",
                    )
                    continue
                restarts = int(info["restarts"])
                if restarts >= FLAGS.babysit_max_restarts:
                    info["done"] = True
                    _append(
                        wiki_path,
                        f"- `{_timestamp()}` mode `{mode}` reached restart limit "
                        f"{FLAGS.babysit_max_restarts}; not resubmitting.",
                    )
                    continue
                new_job_id = _submit(Path(info["script"]))
                info["job_id"] = new_job_id
                info["restarts"] = restarts + 1
                _append(
                    wiki_path,
                    f"- `{_timestamp()}` mode `{mode}` resubmitted as job `{new_job_id}` "
                    f"(restart {info['restarts']}/{FLAGS.babysit_max_restarts}).",
                )

            if state in _DONE_STATES and state not in _RESTART_STATES:
                info["done"] = True
                _append(
                    wiki_path,
                    f"- `{_timestamp()}` mode `{mode}` ended with state `{state}`; not resubmitting.",
                )

        if all_done:
            break
        time.sleep(FLAGS.babysit_poll_seconds)

    _append(wiki_path, f"\n## Slurm Babysitter Complete\n\nTime: {_timestamp()}\n")


if __name__ == "__main__":
    app.run(main)
