"""Production-path tests for additive tail LoRA training and checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np
import orbax.checkpoint as ocp

from omegalax.data.pretrain_data_set import DOC_CHAIN_FORMAT, write_json_arrayrecord_dataset
from omegalax.data.pretrain_statepassing import (
    build_statepassing_fixed_c_indexes,
    make_statepassing_iterator,
)
from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.shard_config import ShardConfig
from omegalax.text import api as text_api
from omegalax.text.checkpoint import resolve_checkpoint, restore_model_params
from omegalax.trainers import checkpoint_utils
from omegalax.trainers.lora import LoRALinear, LoRAParam
from omegalax.trainers.pretrain import PretrainMode
from omegalax.trainers.tail_pretrain import (
    build_cosine_bridge,
    build_tail_optimizer,
    expected_lora_target_counts,
    inject_tail_lora,
    run_tail_pretrain,
    TailTrainConfig,
)
from omegalax.trainers.text import (
    _make_checkpoint_manager,
    _restore_sft_checkpoint,
    _train_state,
    _write_checkpoint_config,
    build_optimizer,
    TrainConfig,
)
from tests.pretrain_real_data_test_utils import test_temp_dir


def _tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        layer_types=("linear_attention",),
        rope_theta=10_000,
        partial_rotary_factor=0.25,
        mrope_section=(1, 1, 0),
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        intermediate_size=32,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=jnp.float32,
    )


def _save_source_checkpoint(root: Path, model, cfg) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_checkpoint_config(root, cfg)
    with mesh_rules(ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)):
        optimizer = build_optimizer(model, TrainConfig(num_steps=1))
    manager = _make_checkpoint_manager(root, save_interval=None, max_to_keep=1)
    manager.save(
        40000,
        args=ocp.args.Composite(
            train_state=ocp.args.PyTreeSave(_train_state(optimizer, jax.random.key(9)))
        ),
        force=True,
    )
    manager.wait_until_finished()
    manager.close()


def _direct_backbone_params(model) -> dict[tuple[tuple[str, ...], str], np.ndarray]:
    values = {}
    for path, module in nnx.iter_modules(model):
        normalized_path = tuple(str(part) for part in path if str(part) != "base")
        for name, value in vars(module).items():
            if isinstance(value, nnx.Param) and not isinstance(value, LoRAParam):
                values[(normalized_path, name)] = np.asarray(value[...]).copy()
    return values


def _adapter_arrays(model) -> list[np.ndarray]:
    return [
        np.asarray(value).copy() for value in jax.tree.leaves(nnx.pure(nnx.state(model, LoRAParam)))
    ]


class TailPretrainTest(absltest.TestCase):
    def test_cosine_bridge_uses_exact_first_and_last_update_values(self):
        config = TailTrainConfig(
            num_steps=4000,
            start_lr=4.19804819103e-5,
            end_lr=3e-5,
        )
        schedule = build_cosine_bridge(config)

        self.assertAlmostEqual(float(schedule(0)), config.start_lr, places=11)
        self.assertAlmostEqual(float(schedule(3999)), config.end_lr, places=11)

    def test_real_qwen_lora_forward_update_and_orbax_restore_freeze_backbone(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            cfg = _tiny_config()
            source_model, cfg = text_api.init_model(
                cfg, jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1
            )
            source_root = tmpdir / "source"
            _save_source_checkpoint(source_root, source_model, cfg)
            source_checkpoint = resolve_checkpoint(source_root / "040000")

            plain_model, _ = text_api.init_model(
                cfg, jax.random.key(1), tp_size=1, fsdp_size=1, dp_size=1
            )
            restore_model_params(plain_model, source_checkpoint)
            lora_model, _ = text_api.init_model(
                cfg, jax.random.key(2), tp_size=1, fsdp_size=1, dp_size=1
            )
            restore_model_params(lora_model, source_checkpoint)
            wrapped_counts = inject_tail_lora(lora_model, adapter_seed=0)
            self.assertEqual(wrapped_counts, expected_lora_target_counts(cfg))
            ranks = sorted(
                module.r
                for _, module in nnx.iter_modules(lora_model)
                if isinstance(module, LoRALinear)
            )
            self.assertEqual(ranks.count(32), wrapped_counts[0])
            self.assertEqual(ranks.count(6), wrapped_counts[1])

            tokens = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
            attention = jnp.ones_like(tokens)
            plain_hidden = text_api.forward_with_gdn_state(
                plain_model, tokens, 0, cfg, attention_mask_BT=attention
            )[0]
            lora_hidden = text_api.forward_with_gdn_state(
                lora_model, tokens, 0, cfg, attention_mask_BT=attention
            )[0]
            np.testing.assert_array_equal(np.asarray(plain_hidden), np.asarray(lora_hidden))

            docs = tmpdir / "docs"
            write_json_arrayrecord_dataset(
                (
                    {
                        "dataset_format": DOC_CHAIN_FORMAT,
                        "doc_id": "tail-doc",
                        "token_ids": list(range(1, 32)) + [63],
                        "doc_token_count": 32,
                    },
                ),
                docs / "train" / "bucket_2k",
                records_per_shard=10,
                overwrite=False,
                metadata={"dataset_format": DOC_CHAIN_FORMAT},
            )
            bundle = build_statepassing_fixed_c_indexes(
                docs,
                tmpdir / "bundle",
                num_segments=4,
                trim_batch_size=4,
                trim_grad_accum_steps=1,
                chunk_length=4,
                eos_id=63,
                splits=("train",),
                records_per_shard=10,
            )

            def make_iterator():
                return make_statepassing_iterator(
                    bundle / "train",
                    batch_size=4,
                    chunk_length=4,
                    eos_id=63,
                    shuffle=False,
                    num_epochs=1,
                    dp_size=1,
                    fsdp_size=1,
                    process_index=0,
                    grain_workers=0,
                )

            config = TailTrainConfig(
                seed=0,
                batch_size=4,
                seq_len=4,
                num_steps=2,
                global_step_offset=40000,
                start_lr=1e-3,
                end_lr=5e-4,
                weight_decay=0.0,
                max_grad_norm=1.0,
                grad_accum_steps=1,
            )
            save_dir = tmpdir / "tail-run"
            optimizer, metrics = run_tail_pretrain(
                source_checkpoint.step_path,
                config,
                make_iterator(),
                pretrain_mode=PretrainMode.STATEPASSING_BPTT,
                save_dir=save_dir,
                enable_lora=True,
                lora_adapter_seed=0,
                save_every=1,
                keep_latest=3,
                log_every=1,
                resume=checkpoint_utils.ResumeMode.NEVER,
                pad_id=0,
                tp_size=1,
                fsdp_size=1,
                dp_size=1,
                text_attn_backend="xla",
                bptt_chunks=4,
                pass_gdn_state=True,
                gdn_layer_limit=None,
                pass_rope_positions=False,
                pass_conv_state=True,
            )
            self.assertEqual(metrics["step"], 40002)
            self.assertTrue(np.isfinite(metrics["loss"]))
            self.assertEqual(optimizer.wrt, LoRAParam)

            source_backbone = _direct_backbone_params(plain_model)
            trained_backbone = _direct_backbone_params(optimizer.model)
            self.assertEqual(trained_backbone.keys(), source_backbone.keys())
            for key in source_backbone:
                np.testing.assert_array_equal(trained_backbone[key], source_backbone[key])
            self.assertTrue(
                any(np.any(array != 0) for array in _adapter_arrays(optimizer.model)[1::2])
            )

            restored_model, restored_cfg = text_api.init_model(
                cfg, jax.random.key(3), tp_size=1, fsdp_size=1, dp_size=1
            )
            restore_model_params(restored_model, source_checkpoint)
            inject_tail_lora(restored_model, adapter_seed=0)
            mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
            with mesh_rules(mesh):
                restored_optimizer = build_tail_optimizer(
                    restored_model,
                    build_cosine_bridge(config),
                    config,
                    wrt=LoRAParam,
                )
            manager = _make_checkpoint_manager(save_dir, save_interval=None, max_to_keep=3)
            restore_rng = jax.device_put(jax.random.key(0), NamedSharding(mesh, PartitionSpec()))
            restored_optimizer, restored_step, _, _ = _restore_sft_checkpoint(
                manager,
                restored_optimizer,
                restore_rng,
                make_iterator(),
            )
            manager.close()
            self.assertEqual(restored_step, 40002)
            for expected, restored in zip(
                _adapter_arrays(optimizer.model),
                _adapter_arrays(restored_optimizer.model),
                strict=True,
            ):
                np.testing.assert_array_equal(restored, expected)
            self.assertEqual(restored_cfg.layer_types, cfg.layer_types)

            eval_model, eval_cfg = text_api.init_model(
                cfg, jax.random.key(4), tp_size=1, fsdp_size=1, dp_size=1
            )
            restore_model_params(eval_model, resolve_checkpoint(save_dir / "040002"))
            self.assertEqual(
                tuple(
                    sum(
                        isinstance(module, LoRALinear) and module.r == rank
                        for _, module in nnx.iter_modules(eval_model)
                    )
                    for rank in (32, 6)
                ),
                wrapped_counts,
            )
            for expected, restored in zip(
                _adapter_arrays(optimizer.model),
                _adapter_arrays(eval_model),
                strict=True,
            ):
                np.testing.assert_array_equal(restored, expected)
            eval_hidden = text_api.forward_with_gdn_state(
                eval_model, tokens, 0, eval_cfg, attention_mask_BT=attention
            )[0]
            trained_hidden = text_api.forward_with_gdn_state(
                optimizer.model, tokens, 0, cfg, attention_mask_BT=attention
            )[0]
            np.testing.assert_allclose(
                np.asarray(eval_hidden), np.asarray(trained_hidden), rtol=0, atol=0
            )


if __name__ == "__main__":
    absltest.main()
