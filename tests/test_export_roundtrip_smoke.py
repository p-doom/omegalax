"""Round-trip export/import smoke tests for all supported families."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from absl.testing import absltest
from flax import nnx

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.export import model_config_to_hf_dict, param_fingerprint
from omegalax.export_entry import resolve_export_step
from omegalax.models.params_utils import flatten_pure_state
from omegalax.models.qwen3.config import make_config as make_qwen3_config
from omegalax.models.qwen3.loader import create_qwen3_from_safetensors
from omegalax.models.qwen3.model import Qwen3
from omegalax.models.qwen3.params import export_qwen3_to_safetensors
from omegalax.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration, make_config
from omegalax.models.qwen3_5.params import (
    create_qwen3_5_from_safetensors,
    export_qwen3_5_to_safetensors,
)
from omegalax.models.qwen3_vl import Qwen3VL, make_vl_config
from omegalax.models.qwen3_vl.params import (
    create_qwen3_vl_from_safetensors,
    export_qwen3_vl_to_safetensors,
)


def _valid_export_step_env():
    return {
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


class ExportEntryTopologyTest(absltest.TestCase):
    def test_snapshot_validation_precedes_jax_export_entrypoint(self):
        code = """
from unittest import mock
from scripts import export_to_hf as script
events = []
snapshot = mock.Mock()
snapshot_context = mock.MagicMock()
snapshot_context.__enter__.side_effect = lambda: events.append("open") or snapshot
snapshot_context.__exit__.side_effect = lambda *_: events.append("close")
with (
    mock.patch.object(script, "FLAGS") as flag_values,
    mock.patch.object(script, "open_local_vlm_snapshot", return_value=snapshot_context),
    mock.patch.object(
        script.vlm_api,
        "resolve_config",
        side_effect=lambda *_: events.append("config"),
    ),
    mock.patch.object(script, "_run", side_effect=lambda *_: events.append("jax")),
):
    flag_values.model_snapshot = "/sealed/model"
    script.main(None)
if events != ["open", "config", "jax", "close"]:
    raise AssertionError(events)
"""
        env = dict(os.environ)
        env["JAX_PLATFORMS"] = "cpu"
        for optimized in (False, True):
            command = [sys.executable]
            if optimized:
                command.append("-O")
            command.extend(["-c", code])
            with self.subTest(optimized=optimized):
                subprocess.run(command, env=env, check=True, timeout=180)

    def test_export_input_has_no_network_fallback(self):
        source = (Path(__file__).parents[1] / "scripts" / "export_to_hf.py").read_text()
        self.assertNotIn("snapshot_download", source)
        self.assertIn("open_local_vlm_snapshot", source)

    def test_plain_batch_launches_one_clean_step(self):
        env = {
            "SLURM_JOB_ID": "1234",
            "SLURM_JOB_NODELIST": "hai001",
            "SLURM_STEP_ID": "-5",
            "SLURM_STEP_NODELIST": "hai009",
            "SLURM_PROCID": "7",
            "KEEP_ME": "yes",
        }

        argv, child_env = resolve_export_step(
            env,
            ["scripts/export_to_hf.py", "--model_snapshot=/sealed/x"],
            "/venv/bin/python",
            "/repo/scripts/export_to_hf.py",
            "hai001",
        )

        self.assertEqual(
            argv,
            [
                "srun",
                "--nodes=1",
                "--ntasks=1",
                "--ntasks-per-node=1",
                "--kill-on-bad-exit=1",
                "/venv/bin/python",
                "/repo/scripts/export_to_hf.py",
                "--model_snapshot=/sealed/x",
            ],
        )
        self.assertEqual(child_env["OMEGALAX_EXPORT_STEP_JOB_ID"], "1234")
        self.assertEqual(child_env["KEEP_ME"], "yes")
        self.assertNotIn("SLURM_STEP_NODELIST", child_env)
        self.assertNotIn("SLURM_PROCID", child_env)

    def test_valid_single_task_step_runs_export_directly(self):
        self.assertIsNone(
            resolve_export_step(
                _valid_export_step_env(),
                ["scripts/export_to_hf.py"],
                "/venv/bin/python",
                "/repo/scripts/export_to_hf.py",
                "hai001",
            )
        )

    def test_malformed_or_mismatched_step_fails(self):
        cases = [
            ({"SLURM_STEP_NUM_TASKS": "2", "SLURM_NTASKS": "2"}, "exactly one task"),
            ({"SLURM_PROCID": "x"}, "SLURM_PROCID"),
            ({"SLURM_STEP_NODELIST": "hai009"}, "runs on hai001"),
            ({"SLURM_STEP_NUM_NODES": ""}, "SLURM_STEP_NUM_NODES"),
        ]
        for update, match in cases:
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, match):
                resolve_export_step(
                    {**_valid_export_step_env(), **update},
                    ["scripts/export_to_hf.py"],
                    "/venv/bin/python",
                    "/repo/scripts/export_to_hf.py",
                    "hai001",
                )

    def test_exporter_created_child_must_belong_to_same_job(self):
        env = {
            "SLURM_JOB_ID": "5678",
            "OMEGALAX_EXPORT_STEP_JOB_ID": "1234",
        }

        with self.assertRaisesRegex(ValueError, "1234.*5678"):
            resolve_export_step(
                env,
                ["scripts/export_to_hf.py"],
                "/venv/bin/python",
                "/repo/scripts/export_to_hf.py",
                "hai001",
            )


def _flatten_model(model):
    _, state = nnx.split(model)
    pure = nnx.to_pure_dict(state)
    return flatten_pure_state(pure)


def _assert_params_equal(testcase: absltest.TestCase, model_a, model_b):
    flat_a = _flatten_model(model_a)
    flat_b = _flatten_model(model_b)
    testcase.assertSetEqual(set(flat_a.keys()), set(flat_b.keys()))
    for key in flat_a:
        a = np.asarray(jax.device_get(flat_a[key]))
        b = np.asarray(jax.device_get(flat_b[key]))
        testcase.assertEqual(a.shape, b.shape, f"Shape mismatch at {key}")
        np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=key)


class ExportRoundTripTest(absltest.TestCase):
    def test_fingerprint_does_not_cancel_equal_and_opposite_changes(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg = make_qwen3_config("qwen3-smoke")
            model = Qwen3(cfg, rngs=nnx.Rngs(params=jax.random.key(0)))
        before = param_fingerprint(model)["lm_head.kernel"]
        model.lm_head.kernel[0, 0] += 1.0
        model.lm_head.kernel[0, 1] -= 1.0

        self.assertNotEqual(param_fingerprint(model)["lm_head.kernel"], before)

    def test_qwen3_dense_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg = make_qwen3_config("qwen3-smoke")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_to_safetensors(model, cfg, tmpdir)
                loaded = create_qwen3_from_safetensors(
                    tmpdir, "qwen3-smoke", tp_size=1, fsdp_size=1, dp_size=1
                )
        _assert_params_equal(self, model, loaded)

    def test_qwen3_moe_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg = make_qwen3_config("qwen3-smoke-moe")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_to_safetensors(model, cfg, tmpdir)
                loaded = create_qwen3_from_safetensors(
                    tmpdir, "qwen3-smoke-moe", tp_size=1, fsdp_size=1, dp_size=1
                )
        _assert_params_equal(self, model, loaded)

    def test_qwen3_vl_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg = make_vl_config("qwen3-vl-smoke")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3VL(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_vl_to_safetensors(model, cfg, tmpdir)
                loaded, _ = create_qwen3_vl_from_safetensors(
                    tmpdir,
                    "qwen3-vl-smoke",
                    tp_size=1,
                    fsdp_size=1,
                    dp_size=1,
                )
        _assert_params_equal(self, model, loaded)

    def test_qwen3_vl_moe_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg = make_vl_config("qwen3-vl-smoke-moe")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3VL(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_vl_to_safetensors(model, cfg, tmpdir)
                loaded, _ = create_qwen3_vl_from_safetensors(
                    tmpdir,
                    "qwen3-vl-smoke-moe",
                    tp_size=1,
                    fsdp_size=1,
                    dp_size=1,
                )
        _assert_params_equal(self, model, loaded)

    def test_qwen3_5_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg: Qwen3_5Config = make_config("qwen3.5-smoke")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3_5ForConditionalGeneration(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_5_to_safetensors(model, cfg, tmpdir)
                loaded, _ = create_qwen3_5_from_safetensors(
                    tmpdir,
                    "qwen3.5-smoke",
                    tp_size=1,
                    fsdp_size=1,
                    dp_size=1,
                )
        _assert_params_equal(self, model, loaded)

    def test_qwen3_5_dense_round_trip(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            cfg: Qwen3_5Config = make_config("qwen3.5-smoke-dense")
            rngs = nnx.Rngs(params=jax.random.key(0))
            model = Qwen3_5ForConditionalGeneration(cfg, rngs=rngs)
            with tempfile.TemporaryDirectory() as tmpdir:
                export_qwen3_5_to_safetensors(model, cfg, tmpdir)
                loaded, _ = create_qwen3_5_from_safetensors(
                    tmpdir,
                    "qwen3.5-smoke-dense",
                    tp_size=1,
                    fsdp_size=1,
                    dp_size=1,
                )
        _assert_params_equal(self, model, loaded)


# The top-level keys a serving stack dereferences, per model_type, taken from the
# published Qwen configs these exports stand in for. `architectures` alone is not
# enough to assert: the export that killed an eval was missing it *and*
# `vision_end_token_id`, and a check for one would have passed the other.
_SERVABLE_KEYS = {
    "qwen3": {
        "architectures",
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "tie_word_embeddings",
    },
    "qwen3_moe": {
        "architectures",
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "tie_word_embeddings",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
    },
    "qwen3_vl": {
        "architectures",
        "model_type",
        "text_config",
        "vision_config",
        "tie_word_embeddings",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    },
    "qwen3_vl_moe": {
        "architectures",
        "model_type",
        "text_config",
        "vision_config",
        "tie_word_embeddings",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    },
    "qwen3_5": {
        "architectures",
        "model_type",
        "text_config",
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    },
    "qwen3_5_moe": {
        "architectures",
        "model_type",
        "text_config",
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    },
}


class ExportedConfigIsServableTest(absltest.TestCase):
    """Every family's exported config.json must name a resolvable architecture and
    carry the keys a serving stack reads."""

    def _check(self, cfg):
        import transformers

        hf_cfg = model_config_to_hf_dict(cfg)
        model_type = hf_cfg["model_type"]
        self.assertIn(model_type, _SERVABLE_KEYS, f"no servable key set declared for {model_type}")

        missing = _SERVABLE_KEYS[model_type] - set(hf_cfg)
        self.assertEqual(
            set(), missing, f"{model_type} export omits servable keys: {sorted(missing)}"
        )

        arch = hf_cfg["architectures"]
        self.assertIsInstance(arch, list)
        self.assertLen(arch, 1)
        # sglang does `hf_config.architectures[0] in ...`; transformers instantiates
        # the class by this name. A plausible-looking string that names nothing is
        # exactly as unservable as None.
        self.assertTrue(
            hasattr(transformers, arch[0]),
            f"{model_type} declares architecture {arch[0]!r}, which transformers "
            f"{transformers.__version__} does not define",
        )

    def test_qwen3_dense_config_servable(self):
        self._check(make_qwen3_config("qwen3-smoke"))

    def test_qwen3_moe_config_servable(self):
        self._check(make_qwen3_config("qwen3-smoke-moe"))

    def test_qwen3_vl_config_servable(self):
        self._check(make_vl_config("qwen3-vl-smoke"))

    def test_qwen3_vl_moe_config_servable(self):
        self._check(make_vl_config("qwen3-vl-smoke-moe"))

    def test_qwen3_5_config_servable(self):
        self._check(make_config("qwen3.5-smoke"))

    def test_qwen3_5_dense_config_servable(self):
        self._check(make_config("qwen3.5-smoke-dense"))


if __name__ == "__main__":
    absltest.main()
