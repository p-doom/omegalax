"""Production-path tests for Memory-task HuggingFace initialization."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

from absl.testing import absltest
from flax import nnx
import safetensors

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_5.config import make_config
from omegalax.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from omegalax.models.qwen3_5.params import export_qwen3_5_to_safetensors
from omegalax.text.checkpoint import resolve_checkpoint
from omegalax.trainers import tail_pretrain
from scripts import train_memory_tasks
from tests.pretrain_real_data_test_utils import test_temp_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_safetensors_index_and_eos(source: Path, *, eos_token_id: int) -> None:
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["eos_token_id"] = eos_token_id
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    tensor_path = source / "model.safetensors"
    with safetensors.safe_open(tensor_path, framework="numpy") as stream:
        weight_map = {key: tensor_path.name for key in stream.keys()}
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, indent=2, sort_keys=True) + "\n"
    )


def _write_tiny_hf_source(source: Path) -> None:
    with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
        config = make_config("qwen3.5-smoke-dense")
        config = dataclasses.replace(
            config,
            text_config=dataclasses.replace(
                config.text_config,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                layer_types=("linear_attention",),
                mrope_section=(1, 1, 0),
                linear_conv_kernel_dim=3,
                linear_key_head_dim=4,
                linear_num_key_heads=1,
                linear_num_value_heads=2,
                linear_value_head_dim=4,
                intermediate_size=32,
            ),
        )
        model = Qwen3_5ForConditionalGeneration(config, rngs=nnx.Rngs(params=0))
        export_qwen3_5_to_safetensors(model, config, source)
    _add_safetensors_index_and_eos(source, eos_token_id=1023)


class MemoryTaskHfInitializationTest(absltest.TestCase):
    def test_tail_default_keeps_legacy_hf_initialization_metadata(self):
        metadata = tail_pretrain._initialization_metadata(
            None,
            "/tmp/legacy-hf-source",
            global_step_offset=40000,
            hf_provenance=None,
        )

        self.assertEqual(
            metadata,
            {
                "source_hf_model_path": "/tmp/legacy-hf-source",
                "restore": "huggingface_safetensors",
            },
        )

    def test_hf_provenance_hashes_config_index_and_every_referenced_shard(self):
        with test_temp_dir() as tmp:
            source = Path(tmp) / "hf"
            source.mkdir()
            config = {
                "tie_word_embeddings": True,
                "text_config": {
                    "vocab_size": 32,
                    "tie_word_embeddings": True,
                    "eos_token_id": 31,
                },
            }
            (source / "config.json").write_text(json.dumps(config))
            (source / "a.safetensors").write_bytes(b"first shard")
            (source / "b.safetensors").write_bytes(b"second shard")
            index = {
                "weight_map": {
                    "model.a": "b.safetensors",
                    "model.b": "a.safetensors",
                    "model.c": "b.safetensors",
                }
            }
            (source / "model.safetensors.index.json").write_text(json.dumps(index))

            metadata = tail_pretrain.hf_initialization_metadata(source, global_step_offset=0)

            self.assertEqual(metadata["source_hf_model_path"], str(source.resolve()))
            self.assertEqual(metadata["restore"], "huggingface_safetensors")
            self.assertEqual(metadata["global_step_offset"], 0)
            self.assertEqual(metadata["vocab_size"], 32)
            self.assertTrue(metadata["tie_word_embeddings"])
            self.assertEqual(metadata["eos_token_id"], 31)
            self.assertEqual(metadata["config_sha256"], _sha256(source / "config.json"))
            self.assertEqual(
                metadata["safetensors_index_sha256"],
                _sha256(source / "model.safetensors.index.json"),
            )
            self.assertEqual(
                metadata["safetensors_shards"],
                [
                    {"filename": "a.safetensors", "sha256": _sha256(source / "a.safetensors")},
                    {"filename": "b.safetensors", "sha256": _sha256(source / "b.safetensors")},
                ],
            )

    def test_hf_provenance_rejects_escaped_and_unindexed_shards(self):
        config = {
            "tie_word_embeddings": True,
            "text_config": {
                "vocab_size": 32,
                "tie_word_embeddings": True,
                "eos_token_id": 31,
            },
        }
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "escaped"
            source.mkdir()
            (source / "config.json").write_text(json.dumps(config))
            (tmpdir / "outside.safetensors").write_bytes(b"outside")
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.a": "../outside.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "Missing or invalid indexed"):
                tail_pretrain.hf_initialization_metadata(source, global_step_offset=0)

            source = tmpdir / "unindexed"
            source.mkdir()
            (source / "config.json").write_text(json.dumps(config))
            (source / "model.safetensors").write_bytes(b"indexed")
            (source / "extra.safetensors").write_bytes(b"not indexed")
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.a": "model.safetensors"}})
            )
            with self.assertRaisesRegex(ValueError, "do not match the shards referenced"):
                tail_pretrain.hf_initialization_metadata(source, global_step_offset=0)

    def test_cli_rejects_checkpoint_and_hf_initialization_together(self):
        with self.assertRaisesRegex(ValueError, "at most one"):
            train_memory_tasks.main(
                [
                    "--task=exact",
                    "--condition=full_context",
                    "--save_dir=/tmp/not-used",
                    "--init_model_checkpoint=/tmp/checkpoint",
                    "--init_hf_model_path=/tmp/hf",
                    "--jax_cache_dir=/tmp/not-used-cache",
                    f"--run_plan_sha256={'a' * 64}",
                    "--success_criterion=test",
                ]
            )

    def test_cli_rejects_expected_step_without_resume_mode(self):
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            train_memory_tasks.main(
                [
                    "--task=exact",
                    "--condition=full_context",
                    "--save_dir=/tmp/not-used",
                    "--resume_expected_step=1",
                    "--jax_cache_dir=/tmp/not-used-cache",
                    f"--run_plan_sha256={'a' * 64}",
                    "--success_criterion=test",
                ]
            )

    def test_real_hf_load_stop_and_exact_resume_path(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "hf"
            _write_tiny_hf_source(source)

            save_dir = tmpdir / "run"
            base_argv = [
                "--task=exact",
                "--condition=state",
                f"--save_dir={save_dir}",
                f"--init_hf_model_path={source}",
                "--num_steps=2",
                "--pairs_per_batch=1",
                "--save_every=1",
                "--log_every=0",
                f"--jax_cache_dir={tmpdir / 'jax-cache'}",
                "--text_attn_backend=xla",
                f"--run_plan_sha256={'b' * 64}",
                "--success_criterion=real HF load and exact resume",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(base_argv + ["--stop_after_local_step=1"])

            contract = json.loads((save_dir / "memory_task_contract.json").read_text())
            initialization = json.loads((save_dir / "initialization.json").read_text())
            self.assertEqual(contract["source"], initialization)
            self.assertEqual(contract["source"]["global_step_offset"], 0)
            self.assertTrue((save_dir / "000001").is_dir())

            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(
                    base_argv
                    + [
                        "--resume=required",
                        "--resume_expected_step=1",
                        "--stop_after_local_step=2",
                    ]
                )

            self.assertTrue((save_dir / "000002").is_dir())

    def test_model_checkpoint_offset_uses_local_save_interval(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "hf"
            _write_tiny_hf_source(source)

            source_dir = tmpdir / "source"
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(
                    [
                        "--task=exact",
                        "--condition=state",
                        f"--save_dir={source_dir}",
                        f"--init_hf_model_path={source}",
                        "--num_steps=1",
                        "--pairs_per_batch=1",
                        "--save_every=1",
                        "--log_every=0",
                        f"--jax_cache_dir={tmpdir / 'source-jax-cache'}",
                        "--text_attn_backend=xla",
                        f"--run_plan_sha256={'d' * 64}",
                        "--success_criterion=source checkpoint for local save interval test",
                    ]
                )

            tail_dir = tmpdir / "tail"
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(
                    [
                        "--task=exact",
                        "--condition=state",
                        f"--save_dir={tail_dir}",
                        f"--init_model_checkpoint={source_dir / '000001'}",
                        "--num_steps=4",
                        "--pairs_per_batch=1",
                        "--save_every=2",
                        "--log_every=0",
                        f"--jax_cache_dir={tmpdir / 'tail-jax-cache'}",
                        "--text_attn_backend=xla",
                        f"--run_plan_sha256={'e' * 64}",
                        "--success_criterion=local save interval with non-aligned global offset",
                    ]
                )

            self.assertTrue((tail_dir / "000003").is_dir())
            self.assertTrue((tail_dir / "000005").is_dir())
            self.assertEqual(resolve_checkpoint(tail_dir).step, 5)
            cursor = json.loads((tail_dir / "000005/input_iter/process_0-of-1.json").read_text())
            self.assertEqual(cursor["next_index"], 4)

    def test_real_scratch_stop_and_exact_resume_path(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            model_source = tmpdir / "model-config"
            _write_tiny_hf_source(model_source)
            save_dir = tmpdir / "run"
            base_argv = [
                "--task=exact",
                "--condition=full_context",
                f"--save_dir={save_dir}",
                f"--model_id={model_source}",
                "--num_steps=2",
                "--pairs_per_batch=1",
                "--save_every=1",
                "--log_every=0",
                f"--jax_cache_dir={tmpdir / 'jax-cache'}",
                "--text_attn_backend=xla",
                f"--run_plan_sha256={'c' * 64}",
                "--success_criterion=real scratch stop and exact resume",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(base_argv + ["--stop_after_local_step=1"])

            self.assertTrue((save_dir / "000001").is_dir())
            self.assertFalse((save_dir / "000002").exists())

            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(
                    base_argv
                    + [
                        "--resume=required",
                        "--resume_expected_step=1",
                        "--stop_after_local_step=2",
                    ]
                )

            self.assertTrue((save_dir / "000002").is_dir())


if __name__ == "__main__":
    absltest.main()
