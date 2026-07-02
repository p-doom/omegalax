"""Unit tests for the robust distributed-launch mode detection (CPU-only, no jax).

These tests exercise ONLY the pure mode-detection logic
(:func:`omegalax.distributed.launch.detect_mode` and
:func:`parse_nodelist_first_host`) by feeding in synthetic SLURM environments.
They deliberately do NOT call ``jax.distributed.initialize()`` and do NOT create
any XLA backend, so they are fast and safe under ``JAX_PLATFORMS=cpu``.

Covered:
  * SINGLE_PROCESS: no SLURM env (plain workstation), one-task allocation, and the
    OMEGALAX_FORCE_SINGLE_PROCESS escape hatch -> skip init / all local GPUs.
  * MULTI_PROCESS single-node (one task per GPU): 4 tasks / 1 node -> correct
    num_processes/process_id and a derived coordinator host:port.
  * MULTI_PROCESS multi-node (production): e.g. 4 nodes x 4 tasks -> dense global
    process_id, num_processes == total tasks, coordinator == first host.
  * Coordinator host/port derivation from a compact ``SLURM_NODELIST`` like
    ``hkn[0533-0536]`` (and other SLURM nodelist forms).
"""

import os
import unittest
from unittest import mock

from omegalax.distributed.launch import (
    COORDINATOR_PORT_ENV,
    FORCE_SINGLE_PROCESS_ENV,
    LaunchMode,
    detect_mode,
    parse_nodelist_first_host,
)


class ParseNodelistTest(unittest.TestCase):
    def test_single_host(self):
        self.assertEqual(parse_nodelist_first_host("hkn0533"), "hkn0533")

    def test_comma_list(self):
        self.assertEqual(parse_nodelist_first_host("hkn0533,hkn0534"), "hkn0533")

    def test_bracket_range(self):
        # The headline case from the task description.
        self.assertEqual(parse_nodelist_first_host("hkn[0533-0536]"), "hkn0533")

    def test_bracket_mixed_range(self):
        self.assertEqual(parse_nodelist_first_host("hkn[0533,0535-0537]"), "hkn0533")

    def test_bracket_range_with_trailing_group(self):
        self.assertEqual(parse_nodelist_first_host("hkn[0533-0536],gpu[01-02]"), "hkn0533")

    def test_whitespace_stripped(self):
        self.assertEqual(parse_nodelist_first_host("  hkn0533  "), "hkn0533")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_nodelist_first_host("")


class DetectSingleProcessTest(unittest.TestCase):
    def test_no_slurm_env_is_single_process(self):
        # Plain workstation / bare python: no SLURM vars at all.
        info = detect_mode(env={})
        self.assertIs(info.mode, LaunchMode.SINGLE_PROCESS)
        self.assertEqual(info.num_processes, 1)
        self.assertEqual(info.process_id, 0)
        self.assertIsNone(info.coordinator_address)
        self.assertFalse(info.initialized)

    def test_one_task_allocation_is_single_process(self):
        # salloc --gres=gpu:4 then `python -m ...` ONCE: SLURM env present but a
        # single task. This is the headline "sees only 1 GPU" bug -> must be
        # SINGLE_PROCESS (skip init, expose all local GPUs).
        env = {
            "SLURM_JOB_ID": "4257886",
            "SLURM_NTASKS": "1",
            "SLURM_PROCID": "0",
            "SLURM_LOCALID": "0",
            "SLURM_NNODES": "1",
            "SLURM_NODELIST": "hkn0533",
            "SLURM_STEP_NODELIST": "hkn0533",
        }
        info = detect_mode(env=env)
        self.assertIs(info.mode, LaunchMode.SINGLE_PROCESS)
        self.assertEqual(info.num_processes, 1)
        self.assertIsNone(info.coordinator_address)
        self.assertFalse(info.initialized)

    def test_step_num_tasks_one_is_single_process(self):
        # A step scoped to one task (SLURM_STEP_NUM_TASKS) inside a bigger alloc.
        env = {
            "SLURM_JOB_ID": "999",
            "SLURM_NTASKS": "4",  # job-wide, but the step is 1 task
            "SLURM_STEP_NUM_TASKS": "1",
            "SLURM_PROCID": "0",
            "SLURM_NODELIST": "hkn0533",
        }
        info = detect_mode(env=env)
        self.assertIs(info.mode, LaunchMode.SINGLE_PROCESS)

    def test_force_single_process_override(self):
        # Even a genuine 4-task multi-process env must collapse to SINGLE_PROCESS
        # when the escape hatch is set.
        env = {
            "SLURM_JOB_ID": "4257886",
            "SLURM_NTASKS": "4",
            "SLURM_PROCID": "2",
            "SLURM_NNODES": "1",
            "SLURM_NODELIST": "hkn0533",
            "SLURM_STEP_NODELIST": "hkn0533",
            FORCE_SINGLE_PROCESS_ENV: "1",
        }
        info = detect_mode(env=env)
        self.assertIs(info.mode, LaunchMode.SINGLE_PROCESS)
        self.assertEqual(info.num_processes, 1)
        self.assertEqual(info.process_id, 0)
        self.assertIsNone(info.coordinator_address)

    def test_force_single_process_falsey_ignored(self):
        env = {
            "SLURM_JOB_ID": "1",
            "SLURM_NTASKS": "4",
            "SLURM_PROCID": "0",
            "SLURM_NODELIST": "hkn0533",
            FORCE_SINGLE_PROCESS_ENV: "0",
        }
        info = detect_mode(env=env)
        self.assertIs(info.mode, LaunchMode.MULTI_PROCESS)


class DetectMultiProcessSingleNodeTest(unittest.TestCase):
    """One task per GPU on a single node: srun --ntasks-per-node=4 --gres=gpu:4."""

    def _env(self, procid: int) -> dict:
        return {
            "SLURM_JOB_ID": "4257886",
            "SLURM_NTASKS": "4",
            "SLURM_STEP_NUM_TASKS": "4",
            "SLURM_PROCID": str(procid),
            "SLURM_LOCALID": str(procid),
            "SLURM_NNODES": "1",
            "SLURM_STEP_NUM_NODES": "1",
            "SLURM_NODELIST": "hkn0533",
            "SLURM_STEP_NODELIST": "hkn0533",
        }

    def test_four_processes_distinct_ids(self):
        infos = [detect_mode(env=self._env(p)) for p in range(4)]
        for p, info in enumerate(infos):
            self.assertIs(info.mode, LaunchMode.MULTI_PROCESS)
            self.assertEqual(info.num_processes, 4)
            self.assertEqual(info.process_id, p)
            self.assertEqual(info.num_nodes, 1)
        # Coordinator is identical across all processes (same host:port).
        addrs = {i.coordinator_address for i in infos}
        self.assertEqual(len(addrs), 1)
        addr = addrs.pop()
        self.assertTrue(addr.startswith("hkn0533:"))

    def test_coordinator_port_is_job_derived_and_stable(self):
        info = detect_mode(env=self._env(0))
        host, port = info.coordinator_address.rsplit(":", 1)
        self.assertEqual(host, "hkn0533")
        # JAX's own SLURM heuristic: JOB_ID % 4096 + 61440.
        expected = 4257886 % 4096 + 61440
        self.assertEqual(int(port), expected)
        self.assertTrue(61440 <= int(port) <= 65535)


class DetectMultiNodeTest(unittest.TestCase):
    """Production multi-node path (untestable on GPU here; logic verified on CPU)."""

    def _env(self, global_procid: int) -> dict:
        # 4 nodes x 4 tasks/node = 16 processes. SLURM_PROCID is the dense GLOBAL
        # rank 0..15; the nodelist is the compact bracket form.
        return {
            "SLURM_JOB_ID": "4238678",
            "SLURM_NTASKS": "16",
            "SLURM_STEP_NUM_TASKS": "16",
            "SLURM_PROCID": str(global_procid),
            "SLURM_LOCALID": str(global_procid % 4),
            "SLURM_NNODES": "4",
            "SLURM_STEP_NUM_NODES": "4",
            "SLURM_NODELIST": "hkn[0533-0536]",
            "SLURM_STEP_NODELIST": "hkn[0533-0536]",
        }

    def test_dense_global_ranks(self):
        for p in (0, 1, 5, 15):
            info = detect_mode(env=self._env(p))
            self.assertIs(info.mode, LaunchMode.MULTI_PROCESS)
            self.assertEqual(info.num_processes, 16)
            self.assertEqual(info.process_id, p)
            self.assertEqual(info.num_nodes, 4)

    def test_coordinator_is_first_node_of_bracket(self):
        info = detect_mode(env=self._env(7))
        host, port = info.coordinator_address.rsplit(":", 1)
        self.assertEqual(host, "hkn0533")
        self.assertEqual(int(port), 4238678 % 4096 + 61440)

    def test_all_ranks_agree_on_coordinator(self):
        addrs = {detect_mode(env=self._env(p)).coordinator_address for p in range(16)}
        self.assertEqual(len(addrs), 1)

    def test_prefers_step_nodelist_over_job_nodelist(self):
        env = self._env(3)
        # If the step is scoped to a subset of nodes, prefer that.
        env["SLURM_STEP_NODELIST"] = "hkn[0535-0536]"
        env["SLURM_NODELIST"] = "hkn[0533-0536]"
        info = detect_mode(env=env)
        self.assertTrue(info.coordinator_address.startswith("hkn0535:"))

    def test_falls_back_to_job_nodelist(self):
        env = self._env(3)
        del env["SLURM_STEP_NODELIST"]
        info = detect_mode(env=env)
        self.assertTrue(info.coordinator_address.startswith("hkn0533:"))


class DetectMultiProcessValidationTest(unittest.TestCase):
    def test_procid_out_of_range_raises(self):
        env = {
            "SLURM_JOB_ID": "1",
            "SLURM_NTASKS": "4",
            "SLURM_PROCID": "4",  # invalid: must be < ntasks
            "SLURM_NODELIST": "hkn0533",
        }
        with self.assertRaises(ValueError):
            detect_mode(env=env)

    def test_missing_procid_raises(self):
        env = {
            "SLURM_JOB_ID": "1",
            "SLURM_NTASKS": "4",
            "SLURM_NODELIST": "hkn0533",
        }
        with self.assertRaises(ValueError):
            detect_mode(env=env)

    def test_missing_nodelist_raises(self):
        env = {
            "SLURM_JOB_ID": "1",
            "SLURM_NTASKS": "4",
            "SLURM_PROCID": "0",
        }
        with self.assertRaises(ValueError):
            detect_mode(env=env)

    def test_explicit_port_override(self):
        env = {
            "SLURM_JOB_ID": "1",
            "SLURM_NTASKS": "2",
            "SLURM_PROCID": "1",
            "SLURM_NODELIST": "hkn0533",
            COORDINATOR_PORT_ENV: "12345",
        }
        info = detect_mode(env=env)
        self.assertEqual(info.coordinator_address, "hkn0533:12345")


class DetectFromRealEnvironTest(unittest.TestCase):
    """detect_mode() with env=None reads os.environ; verify via monkeypatch."""

    def test_reads_os_environ_when_env_none(self):
        fake = {
            "SLURM_JOB_ID": "77",
            "SLURM_NTASKS": "2",
            "SLURM_PROCID": "1",
            "SLURM_NODELIST": "hkn[0100-0101]",
        }
        with mock.patch.dict(os.environ, fake, clear=True):
            info = detect_mode()
        self.assertIs(info.mode, LaunchMode.MULTI_PROCESS)
        self.assertEqual(info.process_id, 1)
        self.assertEqual(info.num_processes, 2)
        self.assertTrue(info.coordinator_address.startswith("hkn0100:"))

    def test_clean_environ_is_single_process(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            info = detect_mode()
        self.assertIs(info.mode, LaunchMode.SINGLE_PROCESS)


if __name__ == "__main__":
    unittest.main()
