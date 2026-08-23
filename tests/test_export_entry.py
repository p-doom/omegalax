from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "export_to_hf", Path(__file__).parents[1] / "scripts" / "export_to_hf.py"
)
assert _SPEC is not None and _SPEC.loader is not None
export_to_hf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_to_hf)


def test_plain_batch_launches_one_clean_step():
    env = {
        "SLURM_JOB_ID": "1234",
        "SLURM_JOB_NODELIST": "hai001",
        "SLURM_STEP_ID": "-5",
        "SLURM_STEP_NODELIST": "hai009",
        "SLURM_PROCID": "7",
        "KEEP_ME": "yes",
    }

    launch = export_to_hf._step_launch(
        env,
        ["scripts/export_to_hf.py", "--model_id=x"],
        "/venv/bin/python",
        "/repo/scripts/export_to_hf.py",
        "hai001",
    )

    assert launch is not None
    argv, child_env = launch
    assert argv == [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--ntasks-per-node=1",
        "--kill-on-bad-exit=1",
        "/venv/bin/python",
        "/repo/scripts/export_to_hf.py",
        "--model_id=x",
    ]
    assert child_env["OMEGALAX_EXPORT_STEP_JOB_ID"] == "1234"
    assert child_env["KEEP_ME"] == "yes"
    assert "SLURM_STEP_NODELIST" not in child_env
    assert "SLURM_PROCID" not in child_env


def test_valid_single_task_step_runs_export_directly():
    env = {
        "SLURM_JOB_ID": "1234",
        "SLURM_STEP_ID": "0",
        "SLURM_STEP_NUM_NODES": "1",
        "SLURM_STEP_NUM_TASKS": "1",
        "SLURM_STEP_NODELIST": "hai001",
        "SLURM_NTASKS": "1",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NODEID": "0",
    }

    assert (
        export_to_hf._step_launch(
            env,
            ["scripts/export_to_hf.py"],
            "/venv/bin/python",
            "/repo/scripts/export_to_hf.py",
            "hai001",
        )
        is None
    )


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"SLURM_STEP_NUM_TASKS": "2", "SLURM_NTASKS": "2"}, "exactly one task"),
        ({"SLURM_PROCID": "x"}, "SLURM_PROCID"),
        ({"SLURM_STEP_NODELIST": "hai009"}, "runs on hai001"),
        ({"SLURM_STEP_NUM_NODES": ""}, "SLURM_STEP_NUM_NODES"),
    ],
)
def test_malformed_or_mismatched_step_fails(update, match):
    env = {
        "SLURM_JOB_ID": "1234",
        "SLURM_STEP_ID": "0",
        "SLURM_STEP_NUM_NODES": "1",
        "SLURM_STEP_NUM_TASKS": "1",
        "SLURM_STEP_NODELIST": "hai001",
        "SLURM_NTASKS": "1",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NODEID": "0",
    }
    env.update(update)

    with pytest.raises(ValueError, match=match):
        export_to_hf._step_launch(
            env,
            ["scripts/export_to_hf.py"],
            "/venv/bin/python",
            "/repo/scripts/export_to_hf.py",
            "hai001",
        )


def test_exporter_created_child_must_belong_to_same_job():
    env = {
        "SLURM_JOB_ID": "5678",
        "OMEGALAX_EXPORT_STEP_JOB_ID": "1234",
    }

    with pytest.raises(ValueError, match="1234.*5678"):
        export_to_hf._step_launch(
            env,
            ["scripts/export_to_hf.py"],
            "/venv/bin/python",
            "/repo/scripts/export_to_hf.py",
            "hai001",
        )
