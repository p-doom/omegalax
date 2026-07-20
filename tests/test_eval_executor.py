"""Tests for the shared full-document eval chain executor."""

from __future__ import annotations

import dataclasses
import os
import re
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.evals import executor
from omegalax.evals.executor import (
    ChainBatch,
    SegmentInputs,
    assert_chunk1_consistent,
    run_chain,
)
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.trainers.loss import chunked_cross_entropy_stats
from tests.eval_test_utils import tiny_hybrid_config, two_document_chain_arrays


def _position_ids(offsets_B: jax.Array, seq_len: int) -> jax.Array:
    positions_BT = offsets_B[:, None] + jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    return jnp.stack([positions_BT] * 3, axis=0)


class EvalExecutorTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model, cls.cfg = text_api.init_model(
            tiny_hybrid_config(),
            jax.random.key(0),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )
        set_attn_backend(cls.model, "xla")

    def _batch(self) -> ChainBatch:
        arrays = two_document_chain_arrays()
        return ChainBatch(
            document_ids=("doc-a", "doc-b"),
            **{name: jnp.asarray(value) for name, value in arrays.items()},
        )

    def _global_inputs(self, batch, segment_idx, previous_state, *, carry):
        offsets = batch.chunk_indices_BC[:, segment_idx] * batch.token_ids_BCT.shape[-1]
        if previous_state is None or not carry:
            return SegmentInputs(position_offsets_B=offsets)
        return SegmentInputs(
            gdn_states=previous_state.gdn_states,
            conv_states=previous_state.conv_states,
            position_offsets_B=offsets,
        )

    def test_real_chain_uses_per_chunk_nll_and_preserves_partial_tail(self):
        batch = self._batch()

        with mock.patch.object(
            executor.text_api,
            "forward_with_gdn_state",
            wraps=text_api.forward_with_gdn_state,
        ) as forward_spy:
            result = run_chain(
                self.model,
                self.cfg,
                batch,
                pad_id=0,
                condition="stateful",
                segment_input_fn=lambda idx, previous: self._global_inputs(
                    batch, idx, previous, carry=True
                ),
            )

        self.assertLen(forward_spy.call_args_list, batch.token_ids_BCT.shape[1])
        for segment_idx, call in enumerate(forward_spy.call_args_list):
            offsets = batch.chunk_indices_BC[:, segment_idx] * batch.token_ids_BCT.shape[-1]
            np.testing.assert_array_equal(
                np.asarray(call.kwargs["position_ids_ZBT"]),
                np.asarray(_position_ids(offsets, batch.token_ids_BCT.shape[-1])),
            )

        counts_CB = np.stack([np.asarray(segment.token_count_B) for segment in result.segments])
        np.testing.assert_array_equal(
            counts_CB,
            np.asarray([[3, 3], [3, 3], [1, 2]], dtype=np.float32),
        )
        np.testing.assert_array_equal(np.asarray(result.token_count_B), [7, 8])
        np.testing.assert_allclose(
            np.asarray(result.nll_B),
            np.asarray(result.nll_sum_B) / np.asarray(result.token_count_B),
            rtol=1e-7,
            atol=1e-7,
        )

        expected_nll_CB = []
        expected_states = []
        gdn_states = None
        conv_states = None
        for segment_idx in range(batch.token_ids_BCT.shape[1]):
            tokens = batch.token_ids_BCT[:, segment_idx]
            mask = batch.attention_mask_BCT[:, segment_idx]
            offsets = batch.chunk_indices_BC[:, segment_idx] * tokens.shape[1]
            hidden, _, gdn_states, conv_states = text_api.forward_with_gdn_state(
                self.model,
                tokens,
                pad_id=0,
                cfg=self.cfg,
                attention_mask_BT=mask,
                initial_gdn_states=gdn_states,
                initial_conv_states=conv_states,
                position_ids_ZBT=_position_ids(offsets, tokens.shape[1]),
                return_conv_states=True,
            )
            document_nll = []
            for document_idx in range(tokens.shape[0]):
                nll_sum, _ = chunked_cross_entropy_stats(
                    hidden[document_idx : document_idx + 1],
                    self.model.output_weight(),
                    tokens[document_idx : document_idx + 1],
                    batch.loss_mask_BCT[document_idx : document_idx + 1, segment_idx],
                    num_tiles=1,
                )
                document_nll.append(nll_sum)
            expected_nll_CB.append(jnp.stack(document_nll))
            expected_states.append((gdn_states, conv_states))

        np.testing.assert_allclose(
            np.stack([np.asarray(segment.nll_sum_B) for segment in result.segments]),
            np.asarray(jnp.stack(expected_nll_CB)),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(result.nll_sum_B),
            np.sum(
                np.stack([np.asarray(segment.nll_sum_B) for segment in result.segments]),
                axis=0,
            ),
            rtol=1e-7,
            atol=1e-7,
        )
        for segment, (expected_gdn, expected_conv) in zip(
            result.segments, expected_states, strict=True
        ):
            for actual, expected in zip(segment.state.gdn_states, expected_gdn, strict=True):
                np.testing.assert_allclose(
                    np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6
                )
            for actual, expected in zip(segment.state.conv_states, expected_conv, strict=True):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_conditions_are_independent_and_chunk1_mismatch_is_hard_error(self):
        batch = self._batch()
        stateful = run_chain(
            self.model,
            self.cfg,
            batch,
            pad_id=0,
            condition="stateful",
            segment_input_fn=lambda idx, previous: self._global_inputs(
                batch, idx, previous, carry=True
            ),
        )
        stateless = run_chain(
            self.model,
            self.cfg,
            batch,
            pad_id=0,
            condition="stateless",
            segment_input_fn=lambda idx, previous: self._global_inputs(
                batch, idx, previous, carry=False
            ),
        )

        assert_chunk1_consistent(stateful, stateless, checkpoint="tiny", atol=1e-6)
        np.testing.assert_array_equal(
            np.asarray(stateful.segments[0].token_count_B),
            np.asarray(stateless.segments[0].token_count_B),
        )
        np.testing.assert_allclose(
            np.asarray(stateful.segments[0].nll_sum_B),
            np.asarray(stateless.segments[0].nll_sum_B),
            rtol=0,
            atol=1e-6,
        )

        boundary_values = jnp.asarray(
            [1.0, 2.0],
            dtype=stateful.segments[0].nll_sum_B.dtype,
        )
        boundary_reference_segment = dataclasses.replace(
            stateful.segments[0],
            nll_sum_B=boundary_values,
        )
        boundary_reference = dataclasses.replace(
            stateful,
            condition="boundary_reference",
            segments=(boundary_reference_segment, *stateful.segments[1:]),
        )
        inside_segment = dataclasses.replace(
            stateless.segments[0],
            nll_sum_B=boundary_values.at[0].add(0.9e-6),
        )
        inside = dataclasses.replace(
            stateless,
            condition="inside_atol",
            segments=(inside_segment, *stateless.segments[1:]),
        )
        outside_segment = dataclasses.replace(
            stateless.segments[0],
            nll_sum_B=boundary_values.at[0].add(1.1e-6),
        )
        outside = dataclasses.replace(
            stateless,
            condition="outside_atol",
            segments=(outside_segment, *stateless.segments[1:]),
        )
        inside_difference = float(inside_segment.nll_sum_B[0] - boundary_values[0])
        outside_difference = float(outside_segment.nll_sum_B[0] - boundary_values[0])
        self.assertLess(inside_difference, 1e-6)
        self.assertGreater(outside_difference, 1e-6)
        assert_chunk1_consistent(boundary_reference, inside, checkpoint="tiny", atol=1e-6)
        with self.assertRaisesRegex(ValueError, "nll_sum.*difference"):
            assert_chunk1_consistent(
                boundary_reference,
                outside,
                checkpoint="tiny",
                atol=1e-6,
            )

        reference_count = boundary_reference_segment.token_count_B[0]
        next_count = jnp.nextafter(
            reference_count,
            jnp.asarray(jnp.inf, dtype=reference_count.dtype),
        )
        bad_count_segment = dataclasses.replace(
            boundary_reference_segment,
            token_count_B=boundary_reference_segment.token_count_B.at[0].set(next_count),
        )
        bad_count = dataclasses.replace(
            stateless,
            condition="minimal_bad_count",
            segments=(bad_count_segment, *stateless.segments[1:]),
        )
        with self.assertRaisesRegex(ValueError, "token_count"):
            assert_chunk1_consistent(
                boundary_reference,
                bad_count,
                checkpoint="tiny",
                atol=1e-6,
            )

        bad_loss_mask = batch.loss_mask_BCT.at[0, 0, 2].set(0)
        bad_batch = dataclasses.replace(batch, loss_mask_BCT=bad_loss_mask)
        bad = run_chain(
            self.model,
            self.cfg,
            bad_batch,
            pad_id=0,
            condition="bad_mask",
            segment_input_fn=lambda idx, previous: self._global_inputs(
                bad_batch, idx, previous, carry=True
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "checkpoint=tiny.*document=doc-a.*stateful.*bad_mask",
        ):
            assert_chunk1_consistent(stateful, bad, checkpoint="tiny", atol=1e-6)

        bad_first_segment = dataclasses.replace(
            stateless.segments[0],
            nll_sum_B=stateless.segments[0].nll_sum_B.at[0].add(0.01),
        )
        bad_nll = dataclasses.replace(
            stateless,
            condition="bad_nll",
            segments=(bad_first_segment, *stateless.segments[1:]),
        )
        with self.assertRaisesRegex(
            ValueError,
            "checkpoint=tiny.*document=doc-a.*stateful.*bad_nll.*nll_sum.*difference",
        ) as error:
            assert_chunk1_consistent(stateful, bad_nll, checkpoint="tiny", atol=1e-6)
        reference_value = float(stateful.segments[0].nll_sum_B[0])
        candidate_value = float(bad_nll.segments[0].nll_sum_B[0])
        values = re.search(
            r"reference(?:_nll_sum)?=([-+0-9.eE]+).*"
            r"candidate(?:_nll_sum)?=([-+0-9.eE]+).*"
            r"difference=([-+0-9.eE]+)",
            str(error.exception),
        )
        self.assertIsNotNone(values)
        self.assertAlmostEqual(float(values.group(1)), reference_value)
        self.assertAlmostEqual(float(values.group(2)), candidate_value)
        self.assertAlmostEqual(
            float(values.group(3)),
            abs(candidate_value - reference_value),
        )

    def test_loss_mask_is_distinct_from_attention_and_one_token_tail_has_zero_targets(self):
        token_ids = jnp.asarray(
            [[[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]]],
            dtype=jnp.int32,
        )
        attention_mask = jnp.asarray(
            [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0]]],
            dtype=jnp.int32,
        )
        loss_mask = attention_mask.at[0, 1, 2].set(0)
        batch = ChainBatch(
            document_ids=("one-token-tail",),
            token_ids_BCT=token_ids,
            attention_mask_BCT=attention_mask,
            loss_mask_BCT=loss_mask,
            chunk_indices_BC=jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        )
        with mock.patch.object(
            executor.text_api,
            "forward_with_gdn_state",
            wraps=text_api.forward_with_gdn_state,
        ) as forward_spy:
            result = run_chain(
                self.model,
                self.cfg,
                batch,
                pad_id=0,
                condition="masked",
                segment_input_fn=lambda idx, previous: self._global_inputs(
                    batch, idx, previous, carry=False
                ),
            )

        self.assertLen(forward_spy.call_args_list, 3)
        for segment_idx, call in enumerate(forward_spy.call_args_list):
            np.testing.assert_array_equal(
                np.asarray(call.kwargs["attention_mask_BT"]),
                np.asarray(attention_mask[:, segment_idx]),
            )
        self.assertTrue(np.all(np.asarray(token_ids[attention_mask == 0]) != 0))
        self.assertFalse(np.array_equal(np.asarray(attention_mask), np.asarray(loss_mask)))
        np.testing.assert_array_equal(
            [float(segment.token_count_B[0]) for segment in result.segments],
            [3.0, 2.0, 0.0],
        )
        self.assertEqual(float(result.segments[-1].nll_sum_B[0]), 0.0)
        self.assertEqual(float(result.token_count_B[0]), 5.0)
        self.assertTrue(np.isfinite(float(result.nll_B[0])))


if __name__ == "__main__":
    absltest.main()
