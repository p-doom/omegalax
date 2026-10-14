"""Production-path tests for Recurrent Memory Transformer (RMT) memory passing."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

from absl import flags
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.data.pretrain_data_set import DOC_CHAIN_FORMAT, write_json_arrayrecord_dataset
from omegalax.data.pretrain_statepassing import (
    build_statepassing_window_index,
    make_statepassing_iterator,
)
from omegalax.distributed.mesh import mesh_rules_for
from omegalax.models.qwen3_5.config import Qwen3_5Config, Qwen3_5TextConfig, make_config_from_hf
from omegalax.models.qwen3_5.params import (
    create_qwen3_5_from_safetensors,
    qwen3_5_to_hf_config_dict,
)
from omegalax.models.shard_config import ShardConfig
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.text.checkpoint import resolve_checkpoint, restore_model_params
from omegalax.trainers import checkpoint_utils
from omegalax.trainers import pretrain as pretrain_trainer
from scripts import train_memory_tasks
from tests.pretrain_real_data_test_utils import test_temp_dir
from tests.test_memory_tasks_hf_initialization import _write_tiny_hf_source

_HYBRID = ("linear_attention", "full_attention", "linear_attention")


def setUpModule():
    # tokamax parses sys.argv on its first attention call, which fails when other test
    # modules collected in the same process registered required flags.
    flags.FLAGS.mark_as_parsed()


def _config(
    num_rmt_tokens: int,
    layer_types: tuple[str, ...] = _HYBRID,
    *,
    heads: int = 2,
    kv_heads: int = 1,
    param_dtype=jnp.float32,
) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=len(layer_types),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=8,
        rms_norm_eps=1e-6,
        layer_types=layer_types,
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
        param_dtype=param_dtype,
        num_rmt_tokens=num_rmt_tokens,
    )


def _wrap(text_config: Qwen3_5TextConfig) -> Qwen3_5Config:
    return Qwen3_5Config(text_config=text_config)


def _init(cfg: Qwen3_5TextConfig, seed: int = 0):
    model, cfg = text_api.init_model(
        cfg, jax.random.PRNGKey(seed), tp_size=1, fsdp_size=1, dp_size=1
    )
    set_attn_backend(model, text_backend="xla")
    return model, cfg


def _flat_state(model) -> dict[str, np.ndarray]:
    leaves = jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(nnx.state(model)))[0]
    return {jax.tree_util.keystr(path): np.asarray(value) for path, value in leaves}


class RmtModelTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        mesh_context = mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1)
        mesh_context.__enter__()
        self.addCleanup(mesh_context.__exit__, None, None, None)

    def test_config_round_trips_and_baseline_config_is_unchanged(self):
        rmt_dict = qwen3_5_to_hf_config_dict(_wrap(_config(4)))
        self.assertEqual(rmt_dict["text_config"]["num_rmt_tokens"], 4)
        self.assertEqual(make_config_from_hf(rmt_dict).text_config.num_rmt_tokens, 4)
        baseline_dict = qwen3_5_to_hf_config_dict(_wrap(_config(0)))
        self.assertNotIn("num_rmt_tokens", baseline_dict["text_config"])
        self.assertEqual(make_config_from_hf(baseline_dict).text_config.num_rmt_tokens, 0)
        with self.assertRaisesRegex(ValueError, "num_rmt_tokens"):
            _config(-1)

    def test_rmt_memory_does_not_change_base_parameter_initialization(self):
        baseline, _ = _init(_config(0), seed=3)
        rmt, _ = _init(_config(5), seed=3)
        baseline_state, rmt_state = _flat_state(baseline), _flat_state(rmt)
        self.assertEqual(set(rmt_state) - set(baseline_state), {"['rmt_memory']"})
        self.assertEqual(rmt_state["['rmt_memory']"].shape, (5, 16))
        self.assertGreater(float(np.abs(rmt_state["['rmt_memory']"]).max()), 0.0)
        for key, value in baseline_state.items():
            np.testing.assert_array_equal(value, rmt_state[key], err_msg=key)

    def test_single_memory_token_matches_explicit_memory_sequence_with_carried_states(self):
        # With one memory token per side a [read | text | write] block has the same visibility
        # as plain causal attention, so the RMT model must equal a no-RMT model run on the
        # explicitly concatenated embeddings, including carried GDN/conv state and positions.
        rmt, rmt_cfg = _init(_config(1), seed=1)
        plain, plain_cfg = _init(_config(0), seed=1)
        B, T = 2, 6
        first = jnp.arange(1, 1 + B * T, dtype=jnp.int32).reshape(B, T) % 60 + 1
        second = (first * 7) % 60 + 1
        mask = jnp.ones((B, T), dtype=jnp.int32)

        def rmt_forward(tokens, gdn, conv, memory):
            return text_api.forward_with_gdn_state(
                rmt,
                tokens,
                0,
                rmt_cfg,
                attention_mask_BT=mask,
                initial_gdn_states=gdn,
                initial_conv_states=conv,
                return_conv_states=True,
                initial_rmt_memory_BMD=memory,
                return_rmt_memory=True,
            )

        _, _, gdn, conv, memory = rmt_forward(first, None, None, None)
        hidden, _, gdn_out, conv_out, memory_out = rmt_forward(second, gdn, conv, memory)

        embeddings = plain.text.embedder.embedding[...][second]
        sequence = jnp.concatenate([memory, embeddings, memory], axis=1)
        positions = jnp.broadcast_to(jnp.arange(-1, T + 1, dtype=jnp.int32), (3, B, T + 2))
        reference, _ = plain.text(
            inputs_embeds_BTD=sequence,
            position_ids_ZBT=positions,
            gdn_initial_states=gdn,
            conv_initial_states=conv,
        )
        _, _, ref_gdn, ref_conv = plain.text(
            inputs_embeds_BTD=sequence[:, : 1 + T],
            position_ids_ZBT=positions[:, :, : 1 + T],
            gdn_initial_states=gdn,
            conv_initial_states=conv,
            return_gdn_states=True,
            return_conv_states=True,
        )
        np.testing.assert_allclose(hidden, reference[:, 1:-1], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(memory_out, reference[:, -1:], rtol=1e-5, atol=1e-5)
        self.assertEqual(hidden.shape, (B, T, 16))
        self.assertEqual(memory_out.shape, (B, 1, 16))
        for carried, expected in zip(gdn_out + conv_out, ref_gdn + ref_conv, strict=True):
            np.testing.assert_allclose(carried, expected, rtol=1e-5, atol=1e-5)

    def test_rmt_attention_matches_dense_block_mask_and_never_leaks(self):
        M, T, B = 3, 5, 2
        S = T + 2 * M
        model, _ = _init(_config(M, heads=4, kv_heads=2), seed=0)
        attention = model.text.layers[1].attn
        rng = np.random.default_rng(0)
        q = rng.standard_normal((B, S, 4, 8)).astype(np.float32)
        k = rng.standard_normal((B, S, 2, 8)).astype(np.float32)
        v = rng.standard_normal((B, S, 2, 8)).astype(np.float32)
        mask = np.ones((B, S), dtype=np.float32)
        mask[1, M + 3 : M + T] = 0.0  # two right-padded text positions in row 1

        def run(q, k, v):
            return np.asarray(
                attention._rmt_attention(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), mask)
            )

        expected = np.zeros_like(q)
        for b in range(B):
            for position in range(S):
                if position < M:
                    visible = np.arange(S) < M
                elif position < S - M:
                    visible = np.arange(S) <= position
                else:
                    visible = mask[b] > 0
                for head in range(4):
                    keys, values = k[b, :, head // 2], v[b, :, head // 2]
                    logits = np.where(visible, keys @ q[b, position, head] * 8**-0.5, -np.inf)
                    weights = np.exp(logits - logits.max())
                    expected[b, position, head] = (weights / weights.sum()) @ values
        output = run(q, k, v)
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)

        write_noise = rng.standard_normal((B, M, 2, 8)).astype(np.float32)
        leaked_k, leaked_v = k.copy(), v.copy()
        leaked_k[:, -M:] += write_noise
        leaked_v[:, -M:] += write_noise
        leaked = run(q, leaked_k, leaked_v)
        np.testing.assert_array_equal(leaked[:, : S - M], output[:, : S - M])
        self.assertGreater(float(np.abs(leaked[:, -M:] - output[:, -M:]).max()), 1e-3)

        padded_k, padded_v = k.copy(), v.copy()
        padded_k[1, M + 3 : M + T] += 5.0
        padded_v[1, M + 3 : M + T] += 5.0
        np.testing.assert_array_equal(run(q, padded_k, padded_v)[1, -M:], output[1, -M:])

    def test_text_is_causal_and_write_memory_reads_all_text(self):
        model, cfg = _init(_config(3), seed=2)
        base = jnp.asarray([[3, 5, 7, 11, 13, 17]], dtype=jnp.int32)
        changed = base.at[0, 3].set(40)

        def forward(tokens):
            return text_api.forward_with_gdn_state(
                model, tokens, 0, cfg, return_conv_states=True, return_rmt_memory=True
            )

        base_out, changed_out = forward(base), forward(changed)
        np.testing.assert_array_equal(base_out[0][:, :3], changed_out[0][:, :3])
        self.assertGreater(float(jnp.abs(base_out[0][:, 3:] - changed_out[0][:, 3:]).max()), 1e-4)
        self.assertGreater(float(jnp.abs(base_out[-1] - changed_out[-1]).max()), 1e-4)

    def test_right_padding_leaves_text_and_carried_states_unchanged(self):
        model, cfg = _init(_config(3), seed=2)
        valid = jnp.asarray([[3, 5, 7, 11]], dtype=jnp.int32)
        padded = jnp.concatenate([valid, jnp.zeros((1, 3), dtype=jnp.int32)], axis=1)
        padded_mask = jnp.asarray([[1, 1, 1, 1, 0, 0, 0]], dtype=jnp.int32)

        def forward(tokens, mask):
            return text_api.forward_with_gdn_state(
                model, tokens, 0, cfg, attention_mask_BT=mask, return_conv_states=True
            )

        exact = forward(valid, jnp.ones_like(valid))
        with_padding = forward(padded, padded_mask)
        np.testing.assert_allclose(with_padding[0][:, :4], exact[0], rtol=1e-5, atol=1e-5)
        for carried, expected in zip(with_padding[2] + with_padding[3], exact[2] + exact[3]):
            np.testing.assert_allclose(carried, expected, rtol=1e-5, atol=1e-5)

    def test_memory_arguments_are_validated(self):
        plain, plain_cfg = _init(_config(0))
        rmt, rmt_cfg = _init(_config(2))
        tokens = jnp.ones((1, 4), dtype=jnp.int32)
        with self.assertRaisesRegex(ValueError, "num_rmt_tokens > 0"):
            text_api.forward_with_gdn_state(plain, tokens, 0, plain_cfg, return_rmt_memory=True)
        with self.assertRaisesRegex(ValueError, "Expected RMT memory shape"):
            text_api.forward_with_gdn_state(
                rmt, tokens, 0, rmt_cfg, initial_rmt_memory_BMD=jnp.zeros((1, 3, 16))
            )
        with self.assertRaisesRegex(ValueError, "requires rmt_memory_BMD"):
            rmt.text(token_ids_BT=tokens)


def _segment_batch(B: int = 2, C: int = 2, T: int = 4, *, shift: int = 0):
    tokens = (np.arange(B * C * T).reshape(B, C, T) * 5 + 3) % 60 + 1
    tokens[:, 0] = (tokens[:, 0] + shift) % 60 + 1
    return {
        "token_ids_BCT": jnp.asarray(tokens, dtype=jnp.int32),
        "attention_mask_BCT": jnp.ones((B, C, T), dtype=jnp.int32),
        "loss_mask_BCT": jnp.ones((B, C, T), dtype=jnp.int32),
        "chunk_idx_BC": jnp.broadcast_to(jnp.arange(C, dtype=jnp.int32), (B, C)),
        "reset_state_BC": jnp.asarray([[True] + [False] * (C - 1)] * B),
    }


def _loss_stats(model, cfg, batch, **kwargs):
    mode = pretrain_trainer.PretrainMode.STATEPASSING_BPTT
    kwargs.setdefault("pass_gdn_state", False)
    return pretrain_trainer._statepassing_loss_stats(model, batch, cfg, 0, mode, **kwargs)


class RmtTrainerTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        mesh_context = mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1)
        mesh_context.__enter__()
        self.addCleanup(mesh_context.__exit__, None, None, None)

    def test_rmt_alone_carries_memory_between_segments_and_keeps_loss_counts(self):
        plain, plain_cfg = _init(_config(0), seed=4)
        rmt, rmt_cfg = _init(_config(3), seed=4)
        batch, other_first_segment = _segment_batch(), _segment_batch(shift=17)

        _, plain_metrics = _loss_stats(plain, plain_cfg, batch)
        _, plain_other = _loss_stats(plain, plain_cfg, other_first_segment)
        _, rmt_metrics = _loss_stats(rmt, rmt_cfg, batch)
        _, rmt_other = _loss_stats(rmt, rmt_cfg, other_first_segment)

        for name in ("supervised_tokens", "iid_comparable_tokens", "boundary_tokens"):
            self.assertEqual(float(plain_metrics[name]), float(rmt_metrics[name]), name)
        self.assertEqual(float(rmt_metrics["segment1_tokens"]), 2 * 3)
        self.assertTrue(np.isfinite(float(rmt_metrics["loss"])))
        # Without any carried state a no-RMT model's second segment cannot see the first one.
        self.assertEqual(float(plain_metrics["segment1_nll"]), float(plain_other["segment1_nll"]))
        self.assertNotEqual(float(rmt_metrics["segment1_nll"]), float(rmt_other["segment1_nll"]))

        reset = dict(batch, reset_state_BC=jnp.asarray([[True, True]] * 2))
        reset_other = dict(other_first_segment, reset_state_BC=jnp.asarray([[True, True]] * 2))
        np.testing.assert_array_equal(
            float(_loss_stats(rmt, rmt_cfg, reset)[1]["segment1_nll"]),
            float(_loss_stats(rmt, rmt_cfg, reset_other)[1]["segment1_nll"]),
        )

    def test_rmt_composes_with_gdn_conv_and_rope_passing(self):
        plain, plain_cfg = _init(_config(0), seed=4)
        rmt, rmt_cfg = _init(_config(3), seed=4)
        batch = _segment_batch(C=3)
        flags = dict(pass_gdn_state=True, pass_conv_state=True, pass_rope_positions=True)
        plain_loss, plain_metrics = _loss_stats(plain, plain_cfg, batch, **flags)
        rmt_loss, rmt_metrics = _loss_stats(rmt, rmt_cfg, batch, **flags)
        self.assertTrue(np.isfinite(float(rmt_loss)))
        self.assertNotEqual(float(plain_loss), float(rmt_loss))
        self.assertEqual(
            float(plain_metrics["supervised_tokens"]), float(rmt_metrics["supervised_tokens"])
        )

    def test_memory_handoff_follows_bptt_windows(self):
        rmt, rmt_cfg = _init(_config(3), seed=4)
        batch = _segment_batch()
        batch["loss_mask_BCT"] = batch["loss_mask_BCT"].at[:, 0].set(0)

        def memory_gradient(bptt_chunks):
            @nnx.jit
            def gradients(model):
                def loss_fn(inner_model):
                    return _loss_stats(inner_model, rmt_cfg, batch, bptt_chunks=bptt_chunks)

                grads, _ = nnx.grad(loss_fn, has_aux=True)(model)
                return grads

            return np.asarray(nnx.to_pure_dict(gradients(rmt))["rmt_memory"])

        # Loss only on segment 1: the learned initial memory reaches it solely through the
        # carried memory, which a BPTT window of one segment detaches.
        np.testing.assert_array_equal(memory_gradient(1), 0.0)
        self.assertGreater(float(np.abs(memory_gradient(2)).max()), 0.0)

    def test_iid_mode_rejects_rmt(self):
        rmt, rmt_cfg = _init(_config(3))
        batch = {
            "token_ids_BT": jnp.ones((1, 4), dtype=jnp.int32),
            "attention_mask_BT": jnp.ones((1, 4), dtype=jnp.int32),
            "loss_mask_BT": jnp.ones((1, 4), dtype=jnp.int32),
        }
        with self.assertRaisesRegex(ValueError, "statepassing"):
            pretrain_trainer._iid_loss_stats(rmt, batch, rmt_cfg, 0)

    def test_num_rmt_tokens_resolution_never_overrides_a_different_value(self):
        base, with_rmt = _config(0), _config(4)
        self.assertEqual(text_api.with_num_rmt_tokens(base, 4).num_rmt_tokens, 4)
        self.assertIs(text_api.with_num_rmt_tokens(with_rmt, 4), with_rmt)
        with self.assertRaisesRegex(ValueError, "num_rmt_tokens=4"):
            text_api.with_num_rmt_tokens(with_rmt, 0)
        with self.assertRaisesRegex(ValueError, "num_rmt_tokens=0"):
            text_api.with_num_rmt_tokens(base, 4, resumed=True)

    def test_pretrain_resume_matches_uninterrupted_run_with_rmt_state(self):
        cfg = _config(3, param_dtype=jnp.bfloat16)
        train_cfg = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=4,
            num_steps=4,
            lr_schedule_steps=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_steps=1,
            lr_schedule="cosine",
            lr_end_factor=0.25,
            print_every=0,
        )
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            data_root = tmpdir / "docs"
            token_ids = [index % 63 + 1 for index in range(48)]
            write_json_arrayrecord_dataset(
                (
                    {
                        "dataset_format": DOC_CHAIN_FORMAT,
                        "doc_id": "resume-doc",
                        "token_ids": token_ids,
                        "doc_token_count": len(token_ids),
                    },
                ),
                data_root / "train" / "bucket_2k",
                records_per_shard=16,
                overwrite=False,
                metadata={"dataset_format": DOC_CHAIN_FORMAT},
            )
            index_path = build_statepassing_window_index(
                data_root,
                tmpdir / "index",
                chunk_length=4,
                num_segments=2,
                split="train",
                records_per_shard=16,
            )

            def make_iterator():
                return make_statepassing_iterator(
                    index_path,
                    batch_size=2,
                    chunk_length=4,
                    shuffle=True,
                    seed=17,
                    num_epochs=1,
                    dp_size=1,
                    fsdp_size=1,
                    process_index=0,
                    grain_workers=0,
                    grain_read_threads=1,
                    grain_read_prefetch_buffer_size=1,
                )

            run_kwargs = {
                "pretrain_mode": pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                "save_every": 1,
                "log_every": 0,
                "tp_size": 1,
                "fsdp_size": 1,
                "dp_size": 1,
                "bptt_chunks": 2,
                "pass_gdn_state": True,
                "pass_rope_positions": True,
                "pass_conv_state": True,
                "num_rmt_tokens": 3,
                "text_attn_backend": "xla",
            }
            previous_usr1 = signal.getsignal(signal.SIGUSR1)
            previous_term = signal.getsignal(signal.SIGTERM)
            try:
                uninterrupted_optimizer, uninterrupted_metrics = pretrain_trainer.run_pretrain(
                    cfg, train_cfg, make_iterator(), **run_kwargs
                )

                checkpoint_dir = tmpdir / "checkpoint"
                install_handler = pretrain_trainer._install_requeue_signal_handler

                def install_handler_and_request_requeue():
                    requeue_requested = install_handler()
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return requeue_requested

                with (
                    mock.patch.object(
                        pretrain_trainer,
                        "_install_requeue_signal_handler",
                        side_effect=install_handler_and_request_requeue,
                    ),
                    mock.patch.object(pretrain_trainer, "_request_slurm_requeue_if_primary"),
                ):
                    pretrain_trainer.run_pretrain(
                        cfg,
                        train_cfg,
                        make_iterator(),
                        save_dir=checkpoint_dir,
                        resume=checkpoint_utils.ResumeMode.NEVER,
                        **run_kwargs,
                    )
                saved_config = json.loads((checkpoint_dir / "config.json").read_text())
                self.assertEqual(saved_config["text_config"]["num_rmt_tokens"], 3)

                mismatched = dict(run_kwargs, num_rmt_tokens=0)
                with self.assertRaisesRegex(ValueError, "num_rmt_tokens=3"):
                    pretrain_trainer.run_pretrain(
                        cfg,
                        train_cfg,
                        make_iterator(),
                        save_dir=checkpoint_dir,
                        resume=checkpoint_utils.ResumeMode.REQUIRED,
                        **mismatched,
                    )

                resumed_optimizer, resumed_metrics = pretrain_trainer.run_pretrain(
                    cfg,
                    train_cfg,
                    make_iterator(),
                    save_dir=checkpoint_dir,
                    resume=checkpoint_utils.ResumeMode.REQUIRED,
                    **run_kwargs,
                )
            finally:
                signal.signal(signal.SIGUSR1, previous_usr1)
                signal.signal(signal.SIGTERM, previous_term)

        self.assertEqual(uninterrupted_metrics["step"], 4)
        self.assertEqual(resumed_metrics["step"], 4)
        self.assertAlmostEqual(uninterrupted_metrics["nll"], resumed_metrics["nll"], places=6)
        uninterrupted_state = nnx.state(uninterrupted_optimizer)
        resumed_state = nnx.state(resumed_optimizer)
        self.assertEqual(jax.tree.structure(uninterrupted_state), jax.tree.structure(resumed_state))
        paths_and_leaves = jax.tree_util.tree_flatten_with_path(resumed_state)[0]
        memory_leaves = [
            leaf for path, leaf in paths_and_leaves if "rmt_memory" in jax.tree_util.keystr(path)
        ]
        self.assertGreaterEqual(len(memory_leaves), 3)  # parameter plus both Adam moments
        for uninterrupted_leaf, resumed_leaf in zip(
            jax.tree.leaves(uninterrupted_state), jax.tree.leaves(resumed_state), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(uninterrupted_leaf), np.asarray(resumed_leaf))
        init_rng, _ = jax.random.split(jax.random.key(train_cfg.seed))
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            fresh, _ = text_api.init_model(cfg, init_rng, tp_size=1, fsdp_size=1, dp_size=1)
        trained_memory = np.asarray(uninterrupted_optimizer.model.rmt_memory[...])
        self.assertGreater(
            float(np.abs(trained_memory - np.asarray(fresh.rmt_memory[...])).max()), 0.0
        )


class RmtShardingTest(absltest.TestCase):
    def test_sharded_meshes_reproduce_the_single_device_loss(self):
        def run(dp_size: int, fsdp_size: int, tp_size: int) -> dict:
            env = os.environ.copy()
            env.update(
                JAX_PLATFORMS="cpu",
                XLA_FLAGS=f"--xla_force_host_platform_device_count={dp_size * fsdp_size * tp_size}",
                OMEGALAX_DELTANET_KERNEL="xla",
            )
            env.pop("JAX_LOCAL_DEVICE_IDS", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("rmt_multidevice_subprocess.py")),
                    str(dp_size),
                    str(fsdp_size),
                    str(tp_size),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parents[1],
                env=env,
            )
            return json.loads(completed.stdout.strip().splitlines()[-1])

        single = run(1, 1, 1)
        self.assertTrue(single["grads_finite"])
        self.assertGreater(single["memory_grad_abs_max"], 0.0)
        for layout in ((2, 1, 1), (1, 2, 1), (1, 1, 2)):
            sharded = run(*layout)
            self.assertTrue(sharded["grads_finite"], layout)
            self.assertAlmostEqual(sharded["loss"], single["loss"], places=5, msg=str(layout))


class RmtInitializationAndCliTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        slurm_environment = mock.patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "test-job", "SLURMD_NODENAME": socket.gethostname()},
        )
        slurm_environment.start()
        self.addCleanup(slurm_environment.stop)

    def _cli_argv(self, tmpdir: Path, save_dir: Path, *extra: str) -> list[str]:
        return [
            "--task=exact",
            "--condition=state",
            f"--save_dir={save_dir}",
            "--num_steps=2",
            "--pairs_per_batch=1",
            "--save_every=1",
            "--log_every=0",
            f"--jax_cache_dir={tmpdir / 'jax-cache'}",
            "--text_attn_backend=xla",
            f"--run_plan_sha256={'f' * 64}",
            "--success_criterion=rmt unit test",
            "--pass_conv_state",
            "--num_rmt_tokens=3",
            *extra,
        ]

    def test_hf_loading_initializes_memory_without_changing_loaded_weights(self):
        with test_temp_dir() as tmp:
            source = Path(tmp) / "hf"
            _write_tiny_hf_source(source)
            with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
                plain, plain_cfg = create_qwen3_5_from_safetensors(str(source), text_only=True)
                rmt, rmt_cfg = create_qwen3_5_from_safetensors(
                    str(source),
                    text_only=True,
                    num_rmt_tokens=3,
                    rmt_rng=jax.random.PRNGKey(5),
                )
                again, _ = create_qwen3_5_from_safetensors(
                    str(source), text_only=True, num_rmt_tokens=3, rmt_rng=jax.random.PRNGKey(5)
                )
                other, _ = create_qwen3_5_from_safetensors(
                    str(source), text_only=True, num_rmt_tokens=3, rmt_rng=jax.random.PRNGKey(6)
                )
                with self.assertRaisesRegex(ValueError, "rmt_rng"):
                    create_qwen3_5_from_safetensors(str(source), text_only=True, num_rmt_tokens=3)
                with self.assertRaisesRegex(ValueError, "text_only"):
                    create_qwen3_5_from_safetensors(
                        str(source), num_rmt_tokens=3, rmt_rng=jax.random.PRNGKey(5)
                    )
            self.assertEqual(rmt_cfg.text_config.num_rmt_tokens, 3)
            self.assertEqual(plain_cfg.text_config.num_rmt_tokens, 0)
            plain_state, rmt_state = _flat_state(plain), _flat_state(rmt)
            self.assertEqual(set(rmt_state) - set(plain_state), {"['rmt_memory']"})
            for key, value in plain_state.items():
                np.testing.assert_array_equal(value, rmt_state[key], err_msg=key)
            self.assertTrue(
                all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(nnx.state(rmt)))
            )
            np.testing.assert_array_equal(
                rmt_state["['rmt_memory']"], _flat_state(again)["['rmt_memory']"]
            )
            self.assertGreater(
                float(
                    np.abs(rmt_state["['rmt_memory']"] - _flat_state(other)["['rmt_memory']"]).max()
                ),
                0.0,
            )
            self.assertGreater(float(np.abs(rmt_state["['rmt_memory']"]).max()), 0.0)

    def test_real_cli_hf_initialization_trains_stops_and_resumes_rmt_state(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "hf"
            _write_tiny_hf_source(source)
            save_dir = tmpdir / "run"
            argv = self._cli_argv(tmpdir, save_dir, f"--init_hf_model_path={source}")
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(argv + ["--stop_after_local_step=1"])

            contract = json.loads((save_dir / "memory_task_contract.json").read_text())
            self.assertTrue(contract["state"]["pass_conv_state"])
            self.assertTrue(contract["state"]["pass_gdn_state"])
            self.assertEqual(contract["state"]["num_rmt_tokens"], 3)
            config = json.loads((save_dir / "config.json").read_text())
            self.assertEqual(config["text_config"]["num_rmt_tokens"], 3)
            self.assertTrue((save_dir / "000001").is_dir())

            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(
                    argv
                    + ["--resume=required", "--resume_expected_step=1", "--stop_after_local_step=2"]
                )
            self.assertTrue((save_dir / "000002").is_dir())

            resumed = text_api.resolve_config(str(save_dir))
            self.assertEqual(resumed.num_rmt_tokens, 3)
            with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
                memories = []
                for seed in (11, 12):
                    model, _ = _init(resumed, seed=seed)
                    initial = _flat_state(model)["['rmt_memory']"]
                    restored = _flat_state(
                        restore_model_params(model, resolve_checkpoint(save_dir))
                    )
                    self.assertGreater(
                        float(np.abs(restored["['rmt_memory']"] - initial).max()), 0.0
                    )
                    memories.append(restored["['rmt_memory']"])
            np.testing.assert_array_equal(memories[0], memories[1])

    def test_real_cli_scratch_run_with_and_without_rmt_keeps_the_baseline_contract(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            baseline_dir, rmt_dir = tmpdir / "baseline", tmpdir / "rmt"
            baseline_argv = [
                arg
                for arg in self._cli_argv(tmpdir, baseline_dir, "--model_id=qwen3.5-smoke-dense")
                if arg not in ("--pass_conv_state", "--num_rmt_tokens=3")
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(baseline_argv)
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                train_memory_tasks.main(
                    self._cli_argv(tmpdir, rmt_dir, "--model_id=qwen3.5-smoke-dense")
                )
            baseline_state = json.loads((baseline_dir / "memory_task_contract.json").read_text())[
                "state"
            ]
            self.assertEqual(
                baseline_state,
                {
                    "pass_gdn_state": True,
                    "bptt_chunks": 2,
                    "pass_rope_positions": False,
                    "pass_conv_state": False,
                },
            )
            self.assertNotIn(
                "num_rmt_tokens",
                json.loads((baseline_dir / "config.json").read_text())["text_config"],
            )
            metrics = json.loads(stdout.getvalue().strip().splitlines()[-1])
            self.assertTrue(np.isfinite(metrics["nll"]))

    def test_tail_from_checkpoint_restores_memory_and_requires_matching_token_count(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            source_dir, tail_dir = tmpdir / "source", tmpdir / "tail"
            wrong_dir = tmpdir / "tail_wrong_count"
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(
                    self._cli_argv(
                        tmpdir, source_dir, "--model_id=qwen3.5-smoke-dense", "--num_steps=1"
                    )
                )
            tail_args = (f"--init_model_checkpoint={source_dir}", "--num_steps=1")
            wrong_count = [
                arg
                for arg in self._cli_argv(tmpdir, wrong_dir, *tail_args)
                if arg != "--num_rmt_tokens=3"
            ]
            with self.assertRaisesRegex(ValueError, "num_rmt_tokens=3"):
                train_memory_tasks.main(wrong_count)
            self.assertFalse((wrong_dir / "config.json").exists())
            plain_dir = tmpdir / "plain_source"
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(
                    [
                        arg
                        for arg in self._cli_argv(
                            tmpdir, plain_dir, "--model_id=qwen3.5-smoke-dense", "--num_steps=1"
                        )
                        if arg not in ("--pass_conv_state", "--num_rmt_tokens=3")
                    ]
                )
            with self.assertRaisesRegex(ValueError, "num_rmt_tokens=0"):
                train_memory_tasks.main(
                    self._cli_argv(
                        tmpdir,
                        tmpdir / "tail_adds_rmt",
                        f"--init_model_checkpoint={plain_dir}",
                        "--num_steps=1",
                    )
                )
            tail_argv = self._cli_argv(tmpdir, tail_dir, *tail_args)
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(tail_argv)
            self.assertTrue((tail_dir / "000002").is_dir())
            self.assertEqual(
                json.loads((tail_dir / "config.json").read_text())["text_config"]["num_rmt_tokens"],
                3,
            )

    def test_full_context_condition_rejects_segment_state_flags(self):
        for condition in ("full_context", "reset"):
            for flag in ("--num_rmt_tokens=3", "--pass_conv_state"):
                with self.assertRaisesRegex(ValueError, "segmented condition"):
                    train_memory_tasks.main(
                        [
                            "--task=exact",
                            f"--condition={condition}",
                            "--save_dir=/tmp/not-used",
                            flag,
                            "--jax_cache_dir=/tmp/not-used-cache",
                            f"--run_plan_sha256={'a' * 64}",
                            "--success_criterion=test",
                        ]
                    )


if __name__ == "__main__":
    absltest.main()
