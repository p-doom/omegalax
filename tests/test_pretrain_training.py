"""Focused tests for text pretraining train steps and batch preparation."""

from __future__ import annotations

import dataclasses
import os
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx
from jax.sharding import PartitionSpec

from omegalax.distributed.mesh import ensure_mesh
from omegalax.distributed.mesh import mesh_rules_for
from omegalax.distributed.mesh import required_batch_multiple
from omegalax.models.qwen3_5.deltanet import GatedDeltaNet, _causal_depthwise_conv1d
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.shard_config import ShardConfig
from omegalax.text import api as text_api
from omegalax.trainers.loss import chunked_cross_entropy_multi_stats
from omegalax.trainers import pretrain as pretrain_trainer


def _tiny_qwen3_5_config() -> Qwen3_5TextConfig:
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


def _batch_iter(batch: dict[str, np.ndarray], repeats: int = 4):
    for _ in range(repeats):
        yield {
            key: (value.copy() if hasattr(value, "copy") else value) for key, value in batch.items()
        }


class PretrainBatchPrepTest(absltest.TestCase):
    def test_prepare_iid_batch_pops_metadata_and_debug_arrays(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        cfg = _tiny_qwen3_5_config()
        batch = {
            "token_ids_BT": np.ones((2, 4), dtype=np.int32),
            "attention_mask_BT": np.ones((2, 4), dtype=np.int32),
            "loss_mask_BT": np.ones((2, 4), dtype=np.int32),
            "chunk_idx_B": np.asarray([3, 4], dtype=np.int32),
            "metadata": {"doc_ids": ["a", "b"]},
        }

        device_batch, metadata, debug = pretrain_trainer.prepare_pretrain_batch(
            batch, pretrain_trainer.PretrainMode.IID_BASELINE, cfg, mesh
        )

        self.assertEqual(metadata["doc_ids"], ["a", "b"])
        np.testing.assert_array_equal(debug["chunk_idx_B"], np.asarray([3, 4], dtype=np.int32))
        self.assertNotIn("chunk_idx_B", device_batch)
        self.assertIn("attention_mask_BT", device_batch)
        self.assertEqual(device_batch["token_ids_BT"].shape, (2, 4))

    def test_prepare_statepassing_batch_keeps_reset_state_on_device(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        cfg = _tiny_qwen3_5_config()
        batch = {
            "token_ids_BCT": np.ones((1, 2, 4), dtype=np.int32),
            "attention_mask_BCT": np.ones((1, 2, 4), dtype=np.int32),
            "loss_mask_BCT": np.ones((1, 2, 4), dtype=np.int32),
            "chunk_idx_BC": np.asarray([[0, 1]], dtype=np.int32),
            "reset_state_BC": np.asarray([[True, False]], dtype=np.bool_),
            "metadata": {"doc_ids": ["doc"]},
        }

        device_batch, metadata, debug = pretrain_trainer.prepare_pretrain_batch(
            batch, pretrain_trainer.PretrainMode.STATEPASSING_NO_BPTT, cfg, mesh
        )

        self.assertEqual(metadata["doc_ids"], ["doc"])
        self.assertIn("reset_state_BC", device_batch)
        self.assertIn("attention_mask_BCT", device_batch)
        self.assertNotIn("chunk_idx_BC", device_batch)
        np.testing.assert_array_equal(debug["chunk_idx_BC"], np.asarray([[0, 1]], dtype=np.int32))

    def test_prepare_iid_batch_builds_rope_positions_from_chunk_idx(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        cfg = _tiny_qwen3_5_config()
        batch = {
            "token_ids_BT": np.ones((2, 4), dtype=np.int32),
            "attention_mask_BT": np.ones((2, 4), dtype=np.int32),
            "loss_mask_BT": np.ones((2, 4), dtype=np.int32),
            "chunk_idx_B": np.asarray([0, 3], dtype=np.int32),
        }

        device_batch, _, debug = pretrain_trainer.prepare_pretrain_batch(
            batch,
            pretrain_trainer.PretrainMode.IID_BASELINE,
            cfg,
            mesh,
            pass_rope_positions=True,
        )

        expected = np.asarray(
            [
                [[0, 1, 2, 3], [12, 13, 14, 15]],
                [[0, 1, 2, 3], [12, 13, 14, 15]],
                [[0, 1, 2, 3], [12, 13, 14, 15]],
            ],
            dtype=np.int32,
        )
        self.assertIn("position_ids_ZBT", device_batch)
        np.testing.assert_array_equal(np.asarray(device_batch["position_ids_ZBT"]), expected)
        np.testing.assert_array_equal(debug["chunk_idx_B"], np.asarray([0, 3], dtype=np.int32))

    def test_prepare_statepassing_batch_keeps_chunk_idx_for_rope_positions(self):
        mesh = ensure_mesh(tp_size=1, fsdp_size=1, dp_size=1)
        cfg = _tiny_qwen3_5_config()
        batch = {
            "token_ids_BCT": np.ones((1, 3, 4), dtype=np.int32),
            "attention_mask_BCT": np.ones((1, 3, 4), dtype=np.int32),
            "loss_mask_BCT": np.ones((1, 3, 4), dtype=np.int32),
            "chunk_idx_BC": np.asarray([[4, 5, 6]], dtype=np.int32),
            "reset_state_BC": np.asarray([[True, False, False]], dtype=np.bool_),
        }

        device_batch, _, debug = pretrain_trainer.prepare_pretrain_batch(
            batch,
            pretrain_trainer.PretrainMode.STATEPASSING_NO_BPTT,
            cfg,
            mesh,
            pass_rope_positions=True,
        )

        self.assertIn("chunk_idx_BC", device_batch)
        np.testing.assert_array_equal(
            np.asarray(device_batch["chunk_idx_BC"]), np.asarray([[4, 5, 6]], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            debug["chunk_idx_BC"], np.asarray([[4, 5, 6]], dtype=np.int32)
        )

    def test_prepare_statepassing_batch_validates_window_batch_sharding(self):
        class MeshLike:
            shape = {"dp": 2}

        cfg = _tiny_qwen3_5_config()
        cfg = dataclasses.replace(
            cfg,
            shd_cfg=ShardConfig(
                act_btd=PartitionSpec("dp", None, None),
                act_btf=PartitionSpec("dp", None, None),
                act_btnh=PartitionSpec("dp", None, None, None),
            ),
        )
        batch = {
            "token_ids_BCT": np.ones((1, 3, 4), dtype=np.int32),
            "attention_mask_BCT": np.ones((1, 3, 4), dtype=np.int32),
            "loss_mask_BCT": np.ones((1, 3, 4), dtype=np.int32),
            "reset_state_BC": np.asarray([[True, False, False]], dtype=np.bool_),
        }

        with self.assertRaisesRegex(ValueError, "Statepassing window batch size"):
            pretrain_trainer.prepare_pretrain_batch(
                batch,
                pretrain_trainer.PretrainMode.STATEPASSING_NO_BPTT,
                cfg,
                MeshLike(),
            )


class PretrainMaskAndLossTest(absltest.TestCase):
    def test_explicit_attention_mask_overrides_pad_id_fallback(self):
        token_ids = jnp.asarray([[5, 0, 7, 0]], dtype=jnp.int32)
        attention_mask = jnp.asarray([[1, 1, 1, 0]], dtype=jnp.int32)

        segment_ids = text_api.segment_ids_from_inputs(
            token_ids, pad_id=0, attention_mask_BT=attention_mask
        )

        np.testing.assert_array_equal(np.asarray(segment_ids), np.asarray([[1, 1, 1, 0]]))

    def test_statepassing_target_masks_include_boundary_token(self):
        loss_mask = jnp.ones((1, 2, 4), dtype=jnp.int32)
        masks = pretrain_trainer.statepassing_target_masks(loss_mask)

        self.assertEqual(float(jnp.sum(masks.total[:, 1:])), 7.0)
        self.assertEqual(float(jnp.sum(masks.iid_comparable[:, 1:])), 6.0)
        self.assertEqual(float(jnp.sum(masks.segments[0][:, 1:])), 3.0)
        self.assertEqual(float(jnp.sum(masks.boundary[:, 1:])), 1.0)
        self.assertEqual(float(jnp.sum(masks.segments[1][:, 1:])), 3.0)

    def test_statepassing_boundary_mask_respects_segment1_loss_mask(self):
        loss_mask = jnp.ones((1, 2, 4), dtype=jnp.int32)
        loss_mask = loss_mask.at[:, 1, 0].set(0)
        masks = pretrain_trainer.statepassing_target_masks(loss_mask)

        self.assertEqual(float(jnp.sum(masks.total[:, 1:])), 6.0)
        self.assertEqual(float(jnp.sum(masks.iid_comparable[:, 1:])), 6.0)
        self.assertEqual(float(jnp.sum(masks.boundary[:, 1:])), 0.0)

    def test_statepassing_boundary_mask_respects_reset_state(self):
        loss_mask = jnp.ones((1, 2, 4), dtype=jnp.int32)
        reset_state = jnp.asarray([[True, True]], dtype=jnp.bool_)
        masks = pretrain_trainer.statepassing_target_masks(loss_mask, reset_state)

        self.assertEqual(float(jnp.sum(masks.total[:, 1:])), 6.0)
        self.assertEqual(float(jnp.sum(masks.iid_comparable[:, 1:])), 6.0)
        self.assertEqual(float(jnp.sum(masks.boundary[:, 1:])), 0.0)

    def test_statepassing_target_masks_support_generic_c(self):
        loss_mask = jnp.ones((1, 3, 4), dtype=jnp.int32)
        reset_state = jnp.asarray([[True, False, True]], dtype=jnp.bool_)

        masks = pretrain_trainer.statepassing_target_masks(loss_mask, reset_state)

        self.assertEqual(float(jnp.sum(masks.total[:, 1:])), 10.0)
        self.assertEqual(float(jnp.sum(masks.iid_comparable[:, 1:])), 9.0)
        self.assertEqual(float(jnp.sum(masks.boundary[:, 1:])), 1.0)
        self.assertEqual(float(jnp.sum(masks.segments[0][:, 1:])), 3.0)
        self.assertEqual(float(jnp.sum(masks.segments[1][:, 1:])), 3.0)
        self.assertEqual(float(jnp.sum(masks.segments[2][:, 1:])), 3.0)

    def test_statepassing_target_masks_drop_explicit_bptt_split_boundaries(self):
        loss_mask = jnp.ones((1, 6, 2), dtype=jnp.int32)

        masks = pretrain_trainer.statepassing_target_masks(loss_mask, bptt_chunks=4)

        self.assertEqual(float(jnp.sum(masks.boundary[:, 1:])), 4.0)
        self.assertEqual(float(masks.boundary[0, 8]), 0.0)

    def test_boundary_target_alignment_is_segment0_last_to_segment1_first(self):
        token_ids_BCT = jnp.asarray([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype=jnp.int32)
        loss_mask_BCT = jnp.ones((1, 2, 4), dtype=jnp.int32)
        hidden_BTD = jnp.zeros((1, 8, 8), dtype=jnp.float32)
        for pos in range(7):
            logit = 7.0 if pos == 3 else 3.0
            hidden_BTD = hidden_BTD.at[0, pos, int(pos + 1)].set(logit)
        lm_head = jnp.eye(8, dtype=jnp.float32)
        masks = pretrain_trainer.statepassing_target_masks(loss_mask_BCT)
        nll_sums, counts = chunked_cross_entropy_multi_stats(
            hidden_BTD,
            lm_head,
            token_ids_BCT.reshape(1, 8),
            jnp.stack(
                [masks.total, masks.iid_comparable, masks.boundary, *masks.segments],
                axis=0,
            ),
            num_tiles=1,
        )

        expected_boundary = -jax.nn.log_softmax(hidden_BTD[0, 3])[4]
        np.testing.assert_allclose(nll_sums[2], expected_boundary, rtol=1e-6, atol=1e-6)
        self.assertEqual(float(counts[1]), 6.0)
        self.assertEqual(float(counts[2]), 1.0)

    def test_statepassing_loss_applies_bptt_mode_at_segment_handoff(self):
        class LmHead:
            kernel = jnp.eye(4, dtype=jnp.float32)

        class Model:
            lm_head = LmHead()

        model = Model()
        cfg = _tiny_qwen3_5_config()
        batch = {
            "token_ids_BCT": jnp.asarray([[[0, 0], [0, 1]]], dtype=jnp.int32),
            "attention_mask_BCT": jnp.ones((1, 2, 2), dtype=jnp.int32),
            "loss_mask_BCT": jnp.asarray([[[0, 0], [0, 1]]], dtype=jnp.int32),
            "reset_state_BC": jnp.asarray([[True, False]], dtype=jnp.bool_),
        }

        def loss_for_source(source, mode):
            def fake_forward(
                _model,
                token_ids_BT,
                _pad_id,
                _cfg,
                *,
                attention_mask_BT=None,
                segment_ids_BT=None,
                initial_gdn_states=None,
            ):
                del attention_mask_BT, segment_ids_BT
                B, T = token_ids_BT.shape
                hidden = jnp.zeros((B, T, 4), dtype=jnp.float32)
                if initial_gdn_states is None:
                    final_state = (jnp.broadcast_to(source.reshape(1, 1, 1, 1), (B, 1, 1, 1)),)
                    return hidden, jnp.array(0.0, dtype=jnp.float32), final_state

                carried = initial_gdn_states[0].reshape(B)
                hidden = hidden.at[:, 0, 1].set(carried)
                return hidden, jnp.array(0.0, dtype=jnp.float32), initial_gdn_states

            with mock.patch.object(
                pretrain_trainer.text_api, "forward_with_gdn_state", side_effect=fake_forward
            ):
                loss, _ = pretrain_trainer._statepassing_loss_stats(
                    model,
                    batch,
                    cfg,
                    pad_id=0,
                    pretrain_mode=mode,
                )
            return loss

        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            grad_no_bptt = jax.grad(loss_for_source, argnums=0)(
                jnp.array(0.5, dtype=jnp.float32),
                pretrain_trainer.PretrainMode.STATEPASSING_NO_BPTT,
            )
            grad_bptt = jax.grad(loss_for_source, argnums=0)(
                jnp.array(0.5, dtype=jnp.float32),
                pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
            )

        self.assertEqual(float(grad_no_bptt), 0.0)
        self.assertNotEqual(float(grad_bptt), 0.0)

    def test_statepassing_loss_applies_bptt_remainder_groups(self):
        class LmHead:
            kernel = jnp.eye(4, dtype=jnp.float32)

        class Model:
            lm_head = LmHead()

        model = Model()
        cfg = _tiny_qwen3_5_config()
        batch = {
            "token_ids_BCT": jnp.ones((1, 6, 2), dtype=jnp.int32),
            "attention_mask_BCT": jnp.ones((1, 6, 2), dtype=jnp.int32),
            "loss_mask_BCT": jnp.asarray([[[0, 1]] * 6], dtype=jnp.int32),
            "reset_state_BC": jnp.asarray([[True, False, False, False, False, False]]),
        }

        def loss_for_source(source_C):
            call_idx = 0

            def fake_forward(
                _model,
                token_ids_BT,
                _pad_id,
                _cfg,
                *,
                attention_mask_BT=None,
                segment_ids_BT=None,
                initial_gdn_states=None,
            ):
                nonlocal call_idx
                del attention_mask_BT, segment_ids_BT
                B, T = token_ids_BT.shape
                hidden = jnp.zeros((B, T, 4), dtype=jnp.float32)
                if initial_gdn_states is not None:
                    hidden = hidden.at[:, 0, 1].set(initial_gdn_states[0].reshape(B))
                final_state = (
                    jnp.broadcast_to(source_C[call_idx].reshape(1, 1, 1, 1), (B, 1, 1, 1)),
                )
                call_idx += 1
                return hidden, jnp.array(0.0, dtype=jnp.float32), final_state

            with mock.patch.object(
                pretrain_trainer.text_api, "forward_with_gdn_state", side_effect=fake_forward
            ):
                loss, _ = pretrain_trainer._statepassing_loss_stats(
                    model,
                    batch,
                    cfg,
                    pad_id=0,
                    pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                    bptt_chunks=4,
                )
            return loss

        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            grad = jax.grad(loss_for_source)(jnp.linspace(0.1, 0.6, 6, dtype=jnp.float32))

        self.assertNotEqual(float(grad[0]), 0.0)
        self.assertNotEqual(float(grad[1]), 0.0)
        self.assertNotEqual(float(grad[2]), 0.0)
        self.assertEqual(float(grad[3]), 0.0)
        self.assertNotEqual(float(grad[4]), 0.0)
        self.assertEqual(float(grad[5]), 0.0)

    def test_statepassing_loss_can_disable_gdn_state_passing(self):
        class LmHead:
            kernel = jnp.eye(4, dtype=jnp.float32)

        class Model:
            lm_head = LmHead()

        batch = {
            "token_ids_BCT": jnp.ones((1, 3, 2), dtype=jnp.int32),
            "attention_mask_BCT": jnp.ones((1, 3, 2), dtype=jnp.int32),
            "loss_mask_BCT": jnp.ones((1, 3, 2), dtype=jnp.int32),
            "reset_state_BC": jnp.asarray([[True, False, False]]),
        }
        seen_initial_states = []

        def fake_forward(
            _model,
            token_ids_BT,
            _pad_id,
            _cfg,
            *,
            attention_mask_BT=None,
            segment_ids_BT=None,
            initial_gdn_states=None,
        ):
            del attention_mask_BT, segment_ids_BT
            seen_initial_states.append(initial_gdn_states)
            B, T = token_ids_BT.shape
            hidden = jnp.zeros((B, T, 4), dtype=jnp.float32)
            state = (jnp.ones((B, 1, 1, 1), dtype=jnp.float32),)
            return hidden, jnp.array(0.0, dtype=jnp.float32), state

        with (
            mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1),
            mock.patch.object(
                pretrain_trainer.text_api, "forward_with_gdn_state", side_effect=fake_forward
            ),
        ):
            pretrain_trainer._statepassing_loss_stats(
                Model(),
                batch,
                _tiny_qwen3_5_config(),
                pad_id=0,
                pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                pass_gdn_state=False,
            )

        self.assertEqual(seen_initial_states, [None, None, None])

    def test_statepassing_loss_applies_gdn_layer_limit(self):
        class LmHead:
            kernel = jnp.eye(4, dtype=jnp.float32)

        class Model:
            lm_head = LmHead()

        batch = {
            "token_ids_BCT": jnp.ones((1, 2, 2), dtype=jnp.int32),
            "attention_mask_BCT": jnp.ones((1, 2, 2), dtype=jnp.int32),
            "loss_mask_BCT": jnp.ones((1, 2, 2), dtype=jnp.int32),
            "reset_state_BC": jnp.asarray([[True, False]]),
        }
        seen_initial_states = []

        def fake_forward(
            _model,
            token_ids_BT,
            _pad_id,
            _cfg,
            *,
            attention_mask_BT=None,
            segment_ids_BT=None,
            initial_gdn_states=None,
        ):
            del attention_mask_BT, segment_ids_BT
            seen_initial_states.append(initial_gdn_states)
            B, T = token_ids_BT.shape
            hidden = jnp.zeros((B, T, 4), dtype=jnp.float32)
            states = (
                jnp.full((B, 1, 1, 1), 3.0, dtype=jnp.float32),
                jnp.full((B, 1, 1, 1), 5.0, dtype=jnp.float32),
            )
            return hidden, jnp.array(0.0, dtype=jnp.float32), states

        with (
            mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1),
            mock.patch.object(
                pretrain_trainer.text_api, "forward_with_gdn_state", side_effect=fake_forward
            ),
        ):
            pretrain_trainer._statepassing_loss_stats(
                Model(),
                batch,
                _tiny_qwen3_5_config(),
                pad_id=0,
                pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                gdn_layer_limit=1,
            )

        self.assertIsNone(seen_initial_states[0])
        self.assertLen(seen_initial_states[1], 2)
        np.testing.assert_array_equal(
            np.asarray(seen_initial_states[1][0]),
            np.full((1, 1, 1, 1), 3.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(seen_initial_states[1][1]),
            np.zeros((1, 1, 1, 1), dtype=np.float32),
        )

    def test_select_gdn_states_for_carry_limit_edges(self):
        states = (
            jnp.full((1, 1, 1, 1), 3.0, dtype=jnp.float32),
            jnp.full((1, 1, 1, 1), 5.0, dtype=jnp.float32),
        )

        self.assertIsNone(
            pretrain_trainer._select_gdn_states_for_carry(
                states, pass_gdn_state=True, gdn_layer_limit=0
            )
        )
        with self.assertRaisesRegex(ValueError, "gdn_layer_limit"):
            pretrain_trainer._select_gdn_states_for_carry(
                states, pass_gdn_state=True, gdn_layer_limit=3
            )

    def test_statepassing_loss_passes_rope_positions_per_segment(self):
        class LmHead:
            kernel = jnp.eye(4, dtype=jnp.float32)

        class Model:
            lm_head = LmHead()

        batch = {
            "token_ids_BCT": jnp.ones((1, 3, 2), dtype=jnp.int32),
            "attention_mask_BCT": jnp.ones((1, 3, 2), dtype=jnp.int32),
            "loss_mask_BCT": jnp.ones((1, 3, 2), dtype=jnp.int32),
            "reset_state_BC": jnp.asarray([[True, False, False]]),
            "chunk_idx_BC": jnp.asarray([[4, 5, 6]], dtype=jnp.int32),
        }
        seen_positions = []

        def fake_forward(
            _model,
            token_ids_BT,
            _pad_id,
            _cfg,
            *,
            attention_mask_BT=None,
            segment_ids_BT=None,
            initial_gdn_states=None,
            position_ids_ZBT=None,
        ):
            del attention_mask_BT, segment_ids_BT, initial_gdn_states
            seen_positions.append(position_ids_ZBT)
            B, T = token_ids_BT.shape
            hidden = jnp.zeros((B, T, 4), dtype=jnp.float32)
            state = (jnp.ones((B, 1, 1, 1), dtype=jnp.float32),)
            return hidden, jnp.array(0.0, dtype=jnp.float32), state

        with (
            mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1),
            mock.patch.object(
                pretrain_trainer.text_api, "forward_with_gdn_state", side_effect=fake_forward
            ),
        ):
            pretrain_trainer._statepassing_loss_stats(
                Model(),
                batch,
                _tiny_qwen3_5_config(),
                pad_id=0,
                pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                pass_rope_positions=True,
            )

        expected = [
            np.asarray([[[8, 9]], [[8, 9]], [[8, 9]]], dtype=np.int32),
            np.asarray([[[10, 11]], [[10, 11]], [[10, 11]]], dtype=np.int32),
            np.asarray([[[12, 13]], [[12, 13]], [[12, 13]]], dtype=np.int32),
        ]
        for actual, expected_segment in zip(seen_positions, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(actual), expected_segment)

    def test_segment_handoff_stops_gradient_for_conv_state(self):
        state = jnp.ones((1, 3, 2), dtype=jnp.float32)

        def loss_for_segment(x, segment_idx):
            carried = pretrain_trainer._prepare_carried_states_for_segment(
                (x,), segment_idx=segment_idx, bptt_chunks=2
            )[0]
            return jnp.sum(carried**2)

        grad_inside_group = jax.grad(loss_for_segment, argnums=0)(state, 1)
        grad_group_split = jax.grad(loss_for_segment, argnums=0)(state, 2)

        self.assertGreater(float(jnp.linalg.norm(grad_inside_group)), 0.0)
        self.assertEqual(float(jnp.linalg.norm(grad_group_split)), 0.0)

    def test_apply_state_reset_supports_conv_state_rank(self):
        state = jnp.ones((2, 3, 2), dtype=jnp.float32)
        reset = jnp.asarray([False, True])

        (reset_state,) = pretrain_trainer.apply_state_reset((state,), reset)

        np.testing.assert_array_equal(np.asarray(reset_state[0]), np.ones((3, 2)))
        np.testing.assert_array_equal(np.asarray(reset_state[1]), np.zeros((3, 2)))

    def test_statepassing_loss_passes_conv_state_per_segment(self):
        class LmHead:
            kernel = jnp.eye(4, dtype=jnp.float32)

        class Model:
            lm_head = LmHead()

        batch = {
            "token_ids_BCT": jnp.ones((1, 3, 2), dtype=jnp.int32),
            "attention_mask_BCT": jnp.ones((1, 3, 2), dtype=jnp.int32),
            "loss_mask_BCT": jnp.ones((1, 3, 2), dtype=jnp.int32),
            "reset_state_BC": jnp.asarray([[True, False, False]]),
        }
        seen_conv_states = []

        def fake_forward(
            _model,
            token_ids_BT,
            _pad_id,
            _cfg,
            *,
            attention_mask_BT=None,
            segment_ids_BT=None,
            initial_gdn_states=None,
            initial_conv_states=None,
            return_conv_states=False,
        ):
            del attention_mask_BT, segment_ids_BT, initial_gdn_states
            seen_conv_states.append(initial_conv_states)
            B, T = token_ids_BT.shape
            hidden = jnp.zeros((B, T, 4), dtype=jnp.float32)
            gdn_state = (jnp.ones((B, 1, 1, 1), dtype=jnp.float32),)
            conv_state = (jnp.full((B, 3, 2), len(seen_conv_states), dtype=jnp.float32),)
            if return_conv_states:
                return hidden, jnp.array(0.0, dtype=jnp.float32), gdn_state, conv_state
            return hidden, jnp.array(0.0, dtype=jnp.float32), gdn_state

        with (
            mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1),
            mock.patch.object(
                pretrain_trainer.text_api, "forward_with_gdn_state", side_effect=fake_forward
            ),
        ):
            pretrain_trainer._statepassing_loss_stats(
                Model(),
                batch,
                _tiny_qwen3_5_config(),
                pad_id=0,
                pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                pass_conv_state=True,
            )

        self.assertIsNone(seen_conv_states[0])
        np.testing.assert_array_equal(
            np.asarray(seen_conv_states[1][0]),
            np.ones((1, 3, 2), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(seen_conv_states[2][0]),
            np.full((1, 3, 2), 2.0, dtype=np.float32),
        )

    def test_statepassing_loss_applies_gdn_layer_limit_to_conv_state(self):
        class LmHead:
            kernel = jnp.eye(4, dtype=jnp.float32)

        class Model:
            lm_head = LmHead()

        batch = {
            "token_ids_BCT": jnp.ones((1, 2, 2), dtype=jnp.int32),
            "attention_mask_BCT": jnp.ones((1, 2, 2), dtype=jnp.int32),
            "loss_mask_BCT": jnp.ones((1, 2, 2), dtype=jnp.int32),
            "reset_state_BC": jnp.asarray([[True, False]]),
        }
        seen_conv_states = []

        def fake_forward(
            _model,
            token_ids_BT,
            _pad_id,
            _cfg,
            *,
            attention_mask_BT=None,
            segment_ids_BT=None,
            initial_gdn_states=None,
            initial_conv_states=None,
            return_conv_states=False,
        ):
            del attention_mask_BT, segment_ids_BT, initial_gdn_states
            seen_conv_states.append(initial_conv_states)
            B, T = token_ids_BT.shape
            hidden = jnp.zeros((B, T, 4), dtype=jnp.float32)
            gdn_state = (
                jnp.full((B, 1, 1, 1), 3.0, dtype=jnp.float32),
                jnp.full((B, 1, 1, 1), 5.0, dtype=jnp.float32),
            )
            conv_state = (
                jnp.full((B, 3, 2), 7.0, dtype=jnp.float32),
                jnp.full((B, 3, 2), 11.0, dtype=jnp.float32),
            )
            if return_conv_states:
                return hidden, jnp.array(0.0, dtype=jnp.float32), gdn_state, conv_state
            return hidden, jnp.array(0.0, dtype=jnp.float32), gdn_state

        with (
            mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1),
            mock.patch.object(
                pretrain_trainer.text_api, "forward_with_gdn_state", side_effect=fake_forward
            ),
        ):
            pretrain_trainer._statepassing_loss_stats(
                Model(),
                batch,
                _tiny_qwen3_5_config(),
                pad_id=0,
                pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                pass_conv_state=True,
                gdn_layer_limit=1,
            )

        self.assertIsNone(seen_conv_states[0])
        self.assertLen(seen_conv_states[1], 2)
        np.testing.assert_array_equal(
            np.asarray(seen_conv_states[1][0]),
            np.full((1, 3, 2), 7.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(seen_conv_states[1][1]),
            np.zeros((1, 3, 2), dtype=np.float32),
        )


class PretrainTrainingSmokeTest(absltest.TestCase):
    def test_one_step_iid_pretrain(self):
        cfg = _tiny_qwen3_5_config()
        train_cfg = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=1,
            seq_len=8,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )
        batch = {
            "token_ids_BT": np.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32),
            "attention_mask_BT": np.ones((1, 8), dtype=np.int32),
            "loss_mask_BT": np.ones((1, 8), dtype=np.int32),
            "chunk_idx_B": np.asarray([0], dtype=np.int32),
            "metadata": {"doc_ids": ["iid"]},
        }

        _, metrics = pretrain_trainer.run_pretrain(
            cfg,
            train_cfg,
            _batch_iter(batch),
            pretrain_mode=pretrain_trainer.PretrainMode.IID_BASELINE,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )

        self.assertIn("nll", metrics)
        self.assertIn("ppl", metrics)
        self.assertEqual(metrics["supervised_tokens"], 7.0)

    def test_one_step_statepassing_pretrain(self):
        cfg = _tiny_qwen3_5_config()
        train_cfg = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=8,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )
        batch = {
            "token_ids_BCT": np.asarray(
                [[[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]],
                dtype=np.int32,
            ),
            "attention_mask_BCT": np.ones((1, 2, 8), dtype=np.int32),
            "loss_mask_BCT": np.ones((1, 2, 8), dtype=np.int32),
            "chunk_idx_BC": np.asarray([[0, 1]], dtype=np.int32),
            "reset_state_BC": np.asarray([[True, False]], dtype=np.bool_),
            "metadata": {"doc_ids": ["sp"]},
        }

        _, metrics = pretrain_trainer.run_pretrain(
            cfg,
            train_cfg,
            _batch_iter(batch),
            pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_NO_BPTT,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )

        self.assertIn("segment0_nll", metrics)
        self.assertIn("segment1_nll", metrics)
        self.assertIn("boundary_nll", metrics)
        self.assertIn("iid_comparable_nll", metrics)
        self.assertEqual(metrics["supervised_tokens"], 15.0)
        self.assertEqual(metrics["iid_comparable_tokens"], 14.0)

    def test_one_step_statepassing_pretrain_c3(self):
        cfg = _tiny_qwen3_5_config()
        train_cfg = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=3,
            seq_len=4,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )
        batch = {
            "token_ids_BCT": np.asarray(
                [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]],
                dtype=np.int32,
            ),
            "attention_mask_BCT": np.ones((1, 3, 4), dtype=np.int32),
            "loss_mask_BCT": np.ones((1, 3, 4), dtype=np.int32),
            "chunk_idx_BC": np.asarray([[0, 1, 2]], dtype=np.int32),
            "reset_state_BC": np.asarray([[True, False, False]], dtype=np.bool_),
            "metadata": {"doc_ids": ["sp-c3"]},
        }

        _, metrics = pretrain_trainer.run_pretrain(
            cfg,
            train_cfg,
            _batch_iter(batch),
            pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_NO_BPTT,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )

        self.assertIn("segment2_nll", metrics)
        self.assertIn("boundary_nll", metrics)
        self.assertEqual(metrics["supervised_tokens"], 11.0)
        self.assertEqual(metrics["iid_comparable_tokens"], 9.0)


class GatedDeltaNetMaskStateTest(absltest.TestCase):
    def test_causal_depthwise_conv_state_matches_full_sequence(self):
        x = jnp.arange(1 * 2 * 6, dtype=jnp.float32).reshape(1, 2, 6) / 10.0
        weight = jnp.asarray([[0.2, 0.3, 0.5], [0.7, -0.1, 0.4]], dtype=jnp.float32)

        full, full_state = _causal_depthwise_conv1d(x, weight, return_final_state=True)
        first, state = _causal_depthwise_conv1d(x[:, :, :4], weight, return_final_state=True)
        second, second_state = _causal_depthwise_conv1d(
            x[:, :, 4:], weight, initial_state_BCS=state, return_final_state=True
        )

        np.testing.assert_allclose(
            np.asarray(jnp.concatenate([first, second], axis=2)),
            np.asarray(full),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(np.asarray(second_state), np.asarray(full_state))

    def test_conv_and_gdn_state_match_full_sequence(self):
        cfg = _tiny_qwen3_5_config()
        hidden = jnp.arange(1 * 6 * cfg.hidden_size, dtype=jnp.float32).reshape(
            1, 6, cfg.hidden_size
        )
        hidden = hidden / 100.0
        attention_mask = jnp.ones((1, 6), dtype=jnp.float32)

        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            module = GatedDeltaNet(cfg, rngs=nnx.Rngs(0))
            full, full_gdn_state, full_conv_state = module(
                hidden,
                attention_mask,
                return_final_state=True,
                return_conv_state=True,
            )
            first, gdn_state, conv_state = module(
                hidden[:, :3, :],
                attention_mask[:, :3],
                return_final_state=True,
                return_conv_state=True,
            )
            second, second_gdn_state, second_conv_state = module(
                hidden[:, 3:, :],
                attention_mask[:, 3:],
                gdn_state,
                initial_conv_state_BCS=conv_state,
                return_final_state=True,
                return_conv_state=True,
            )

        np.testing.assert_allclose(
            np.asarray(jnp.concatenate([first, second], axis=1)),
            np.asarray(full),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(second_gdn_state, full_gdn_state, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(second_conv_state, full_conv_state, rtol=1e-5, atol=1e-5)

    def test_right_pads_do_not_change_conv_state(self):
        cfg = _tiny_qwen3_5_config()
        hidden = jnp.arange(1 * 5 * cfg.hidden_size, dtype=jnp.float32).reshape(
            1, 5, cfg.hidden_size
        )
        hidden = hidden / 100.0
        attention_mask = jnp.asarray([[1, 1, 1, 0, 0]], dtype=jnp.float32)

        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            module = GatedDeltaNet(cfg, rngs=nnx.Rngs(0))
            _, _, padded_conv_state = module(
                hidden,
                attention_mask,
                return_final_state=True,
                return_conv_state=True,
            )
            _, _, truncated_conv_state = module(
                hidden[:, :3, :],
                jnp.ones((1, 3), dtype=jnp.float32),
                return_final_state=True,
                return_conv_state=True,
            )

        np.testing.assert_allclose(padded_conv_state, truncated_conv_state, rtol=1e-5, atol=1e-5)

    def test_right_pads_do_not_change_final_state(self):
        cfg = _tiny_qwen3_5_config()
        hidden = jnp.arange(1 * 5 * cfg.hidden_size, dtype=jnp.float32).reshape(
            1, 5, cfg.hidden_size
        )
        hidden = hidden / 100.0
        attention_mask = jnp.asarray([[1, 1, 1, 0, 0]], dtype=jnp.float32)

        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            module = GatedDeltaNet(cfg, rngs=nnx.Rngs(0))
            _, padded_state = module(hidden, attention_mask, return_final_state=True)
            _, truncated_state = module(
                hidden[:, :3, :],
                jnp.ones((1, 3), dtype=jnp.float32),
                return_final_state=True,
            )

        np.testing.assert_allclose(padded_state, truncated_state, rtol=1e-5, atol=1e-5)


class MeshBatchMultipleTest(absltest.TestCase):
    def test_required_batch_multiple_multiplies_tuple_axes(self):
        class MeshLike:
            shape = {"dp": 2, "fsdp": 4}

        self.assertEqual(
            required_batch_multiple(PartitionSpec(("dp", "fsdp"), None), MeshLike()), 8
        )


if __name__ == "__main__":
    absltest.main()
