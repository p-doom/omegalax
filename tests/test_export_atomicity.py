"""An HF export must never expose a partially written file under its final name."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from absl.testing import absltest
from flax import nnx
from safetensors import numpy as stnp

from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models import params_utils
from omegalax.models.params_utils import (
    atomic_save_hf_weights,
    atomic_save_safetensors,
    atomic_write_json,
    save_hf_config,
    staged_replace,
)
from omegalax.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration, make_config
from omegalax.models.qwen3_5.params import export_qwen3_5_to_safetensors


class WriterExploded(RuntimeError):
    pass


def _tensors():
    rng = np.random.default_rng(0)
    return {
        "a": rng.standard_normal((16, 8)).astype(np.float32),
        "b": rng.standard_normal((4,)).astype(np.float32),
    }


def _leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in Path(directory).iterdir())


class StagedReplaceTest(absltest.TestCase):
    def test_body_writes_to_a_temp_path_in_the_destination_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final = Path(tmpdir) / "model.safetensors"
            with staged_replace(final) as staged:
                self.assertNotEqual(staged, final)
                self.assertEqual(staged.parent, final.parent)
                staged.write_bytes(b"payload")
                self.assertFalse(final.exists())
            self.assertEqual(final.read_bytes(), b"payload")
            self.assertEqual(_leftovers(tmpdir), ["model.safetensors"])

    def test_interrupted_write_leaves_no_final_named_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final = Path(tmpdir) / "model.safetensors"
            with self.assertRaises(WriterExploded), staged_replace(final) as staged:
                staged.write_bytes(b"half a checkpoint")
                raise WriterExploded
            self.assertFalse(final.exists())
            self.assertEqual(_leftovers(tmpdir), [])

    def test_interrupted_write_does_not_clobber_the_previous_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final = Path(tmpdir) / "model.safetensors"
            final.write_bytes(b"the good old weights")
            with self.assertRaises(WriterExploded), staged_replace(final) as staged:
                staged.write_bytes(b"half a checkpoint")
                raise WriterExploded
            self.assertEqual(final.read_bytes(), b"the good old weights")
            self.assertEqual(_leftovers(tmpdir), ["model.safetensors"])

    def test_keyboard_interrupt_is_cleaned_up_and_propagated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final = Path(tmpdir) / "model.safetensors"
            with self.assertRaises(KeyboardInterrupt), staged_replace(final):
                raise KeyboardInterrupt
            self.assertEqual(_leftovers(tmpdir), [])

    def test_creates_missing_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final = Path(tmpdir) / "run" / "000500" / "model.safetensors"
            with staged_replace(final) as staged:
                staged.write_bytes(b"x")
            self.assertTrue(final.is_file())


class AtomicSaveSafetensorsTest(absltest.TestCase):
    def test_bytes_match_a_direct_save_file(self):
        tensors = _tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            reference = Path(tmpdir) / "reference.safetensors"
            stnp.save_file(tensors, str(reference))
            atomic = Path(tmpdir) / "atomic.safetensors"
            atomic_save_safetensors(tensors, atomic)
            self.assertEqual(atomic.read_bytes(), reference.read_bytes())

    def test_interrupted_save_leaves_nothing_behind(self):
        tensors = _tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            final = Path(tmpdir) / "model.safetensors"

            def _explode(_tensors, filename, metadata=None):
                Path(filename).write_bytes(b"\x00" * 32)
                raise WriterExploded

            with (
                mock.patch.object(params_utils.stnp, "save_file", _explode),
                self.assertRaises(WriterExploded),
            ):
                atomic_save_safetensors(tensors, final)
            self.assertFalse(final.exists())
            self.assertEqual(_leftovers(tmpdir), [])

    def test_a_reader_never_observes_a_short_file_under_the_final_name(self):
        tensors = _tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            final = Path(tmpdir) / "model.safetensors"
            observed = []
            real_save = params_utils.stnp.save_file

            def _save_and_peek(t, filename, metadata=None):
                observed.append(final.exists())
                real_save(t, filename, metadata=metadata)
                observed.append(final.exists())

            with mock.patch.object(params_utils.stnp, "save_file", _save_and_peek):
                atomic_save_safetensors(tensors, final)
            self.assertEqual(observed, [False, False])
            self.assertTrue(final.is_file())


class AtomicWriteJsonTest(absltest.TestCase):
    def test_writes_and_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final = Path(tmpdir) / "config.json"
            final.write_text("{}")
            atomic_write_json({"model_type": "qwen3_5"}, final)
            self.assertEqual(json.loads(final.read_text()), {"model_type": "qwen3_5"})
            self.assertEqual(_leftovers(tmpdir), ["config.json"])

    def test_save_hf_config_leaves_no_partial_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            unserializable = {"bad": object()}
            with self.assertRaises(TypeError):
                save_hf_config(unserializable, tmpdir)
            self.assertEqual(_leftovers(tmpdir), [])


class AtomicSaveHfWeightsTest(absltest.TestCase):
    def test_single_shard_returns_the_weights_path_and_writes_no_index(self):
        tensors = _tensors()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = atomic_save_hf_weights(tmpdir, {"model.safetensors": tensors})
            self.assertEqual(path, Path(tmpdir) / "model.safetensors")
            self.assertEqual(_leftovers(tmpdir), ["model.safetensors"])

    def test_index_is_written_after_every_shard(self):
        rng = np.random.default_rng(1)
        shards = {
            "model-00001-of-00002.safetensors": {"a": rng.standard_normal(8).astype(np.float32)},
            "model-00002-of-00002.safetensors": {"b": rng.standard_normal(4).astype(np.float32)},
        }
        order = []
        real_replace = params_utils.os.replace

        def _record(src, dst):
            order.append(Path(dst).name)
            real_replace(src, dst)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(params_utils.os, "replace", _record):
                path = atomic_save_hf_weights(tmpdir, shards)
            self.assertEqual(path, Path(tmpdir) / "model.safetensors.index.json")
            self.assertEqual(order[-1], "model.safetensors.index.json")
            self.assertCountEqual(order[:-1], list(shards))
            index = json.loads(path.read_text())
            self.assertEqual(
                index["weight_map"],
                {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            )
            self.assertEqual(index["metadata"]["total_size"], 8 * 4 + 4 * 4)

    def test_a_failed_shard_leaves_no_index_and_no_final_named_shard(self):
        rng = np.random.default_rng(2)
        shards = {
            "model-00001-of-00002.safetensors": {"a": rng.standard_normal(8).astype(np.float32)},
            "model-00002-of-00002.safetensors": {"b": rng.standard_normal(4).astype(np.float32)},
        }
        real_save = params_utils.stnp.save_file
        calls = []

        def _explode_on_second(t, filename, metadata=None):
            calls.append(filename)
            if len(calls) == 2:
                Path(filename).write_bytes(b"\x00" * 16)
                raise WriterExploded
            real_save(t, filename, metadata=metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(params_utils.stnp, "save_file", _explode_on_second),
                self.assertRaises(WriterExploded),
            ):
                atomic_save_hf_weights(tmpdir, shards)
            self.assertEqual(_leftovers(tmpdir), ["model-00001-of-00002.safetensors"])


class ExportQwen35AtomicityTest(absltest.TestCase):
    def _model(self):
        cfg: Qwen3_5Config = make_config("qwen3.5-smoke-dense")
        rngs = nnx.Rngs(params=jax.random.key(0))
        return Qwen3_5ForConditionalGeneration(cfg, rngs=rngs), cfg

    def test_completed_export_is_byte_identical_to_a_direct_save(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model, cfg = self._model()
            with tempfile.TemporaryDirectory() as tmpdir:
                out = Path(tmpdir) / "export"
                path = export_qwen3_5_to_safetensors(model, cfg, out)
                self.assertEqual(path, out / "model.safetensors")
                exported = stnp.load_file(str(path))

                reference = Path(tmpdir) / "reference.safetensors"
                stnp.save_file(exported, str(reference))
                self.assertEqual(path.read_bytes(), reference.read_bytes())
                self.assertEqual(_leftovers(out), ["config.json", "model.safetensors"])

    def test_config_lands_before_the_weights_consumers_gate_on(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model, cfg = self._model()
            with tempfile.TemporaryDirectory() as tmpdir:
                out = Path(tmpdir) / "export"
                seen = []
                real_save = params_utils.stnp.save_file

                def _peek(t, filename, metadata=None):
                    seen.append((out / "config.json").is_file())
                    real_save(t, filename, metadata=metadata)

                with mock.patch.object(params_utils.stnp, "save_file", _peek):
                    export_qwen3_5_to_safetensors(model, cfg, out)
                self.assertEqual(seen, [True])

    def test_interrupted_export_leaves_no_model_safetensors(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model, cfg = self._model()
            with tempfile.TemporaryDirectory() as tmpdir:
                out = Path(tmpdir) / "export"

                def _explode(t, filename, metadata=None):
                    Path(filename).write_bytes(b"\x00" * 4096)
                    raise WriterExploded

                with (
                    mock.patch.object(params_utils.stnp, "save_file", _explode),
                    self.assertRaises(WriterExploded),
                ):
                    export_qwen3_5_to_safetensors(model, cfg, out)
                self.assertFalse((out / "model.safetensors").exists())
                self.assertEqual(_leftovers(out), ["config.json"])

    def test_reexport_over_a_complete_dir_stays_loadable_throughout(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model, cfg = self._model()
            with tempfile.TemporaryDirectory() as tmpdir:
                out = Path(tmpdir) / "export"
                export_qwen3_5_to_safetensors(model, cfg, out)
                first = (out / "model.safetensors").read_bytes()

                real_save = params_utils.stnp.save_file
                midwrite = {}

                def _peek(t, filename, metadata=None):
                    midwrite["bytes"] = (out / "model.safetensors").read_bytes()
                    real_save(t, filename, metadata=metadata)

                with mock.patch.object(params_utils.stnp, "save_file", _peek):
                    export_qwen3_5_to_safetensors(model, cfg, out)
                self.assertEqual(midwrite["bytes"], first)
                self.assertEqual((out / "model.safetensors").read_bytes(), first)


if __name__ == "__main__":
    absltest.main()
