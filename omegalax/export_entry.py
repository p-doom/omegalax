"""Slurm topology validation for the HuggingFace exporter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_STEP_CHILD_JOB_ID = "OMEGALAX_EXPORT_STEP_JOB_ID"
_STEP_RANK_VARIABLES = {
    "SLURM_GTIDS",
    "SLURM_LOCALID",
    "SLURM_NNODES",
    "SLURM_NODEID",
    "SLURM_NPROCS",
    "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_PROCID",
    "SLURM_SRUN_COMM_HOST",
    "SLURM_SRUN_COMM_PORT",
    "SLURM_TASK_PID",
    "SLURM_TASKS_PER_NODE",
}


def _required_int(env: Mapping[str, str], name: str) -> int:
    value = env.get(name)
    if value is None:
        raise ValueError(f"{name} must be an integer, got None")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def resolve_export_step(
    env: Mapping[str, str],
    argv: Sequence[str],
    executable: str,
    script: str,
    hostname: str,
) -> tuple[list[str], dict[str, str]] | None:
    job_id = env.get("SLURM_JOB_ID")
    if not job_id:
        raise ValueError("export_to_hf.py requires a Slurm allocation")

    child_job_id = env.get(_STEP_CHILD_JOB_ID)
    if child_job_id is not None and child_job_id != job_id:
        raise ValueError(
            f"exporter-created step belongs to job {child_job_id}, current job is {job_id}"
        )

    step_id = env.get("SLURM_STEP_ID")
    if step_id in (None, "", "-5", "batch"):
        if child_job_id is not None:
            raise ValueError("srun did not create a Slurm step for the exporter")
        child_env = dict(env)
        for name in tuple(child_env):
            if name.startswith("SLURM_STEP_") or name in _STEP_RANK_VARIABLES:
                child_env.pop(name)
        child_env[_STEP_CHILD_JOB_ID] = job_id
        command = [
            "srun",
            "--nodes=1",
            "--ntasks=1",
            "--ntasks-per-node=1",
            "--kill-on-bad-exit=1",
            executable,
            script,
            *argv[1:],
        ]
        return command, child_env

    if _required_int(env, "SLURM_STEP_ID") < 0:
        raise ValueError(f"SLURM_STEP_ID must identify an srun step, got {step_id!r}")
    if _required_int(env, "SLURM_STEP_NUM_NODES") != 1:
        raise ValueError("export_to_hf.py requires exactly one Slurm step node")
    if _required_int(env, "SLURM_STEP_NUM_TASKS") != 1 or _required_int(env, "SLURM_NTASKS") != 1:
        raise ValueError("export_to_hf.py requires exactly one task")
    for name in ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID"):
        if _required_int(env, name) != 0:
            raise ValueError(f"{name} must be zero for the single-task exporter")

    step_node = env.get("SLURM_STEP_NODELIST")
    if not step_node:
        raise ValueError("SLURM_STEP_NODELIST is required inside the exporter step")
    if step_node.split(".", 1)[0] != hostname.split(".", 1)[0]:
        raise ValueError(f"SLURM_STEP_NODELIST={step_node!r}, but exporter runs on {hostname}")
    return None
