"""Tests for isolated GDN and Conv state-usage experiments."""

from __future__ import annotations

import os
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.evals import experiments
from omegalax.evals.executor import (
    ChainBatch,
    SegmentInputs,
    assert_chunk1_consistent,
    run_chain,
)
from omegalax.evals.experiments import (
    run_conv_experiment,
    run_gdn_experiment,
)
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.text.chat import StatePassingConfig
from tests.eval_test_utils import (
    four_document_chain_arrays,
    tiny_hybrid_config,
    two_document_chain_arrays,
)


def _take_donors(states, donor_indices_B):
    if states is None:
        return None
    return tuple(jnp.take(state, donor_indices_B, axis=0) for state in states)


def _zeros(states):
    if states is None:
        return None
    return tuple(jnp.zeros_like(state) for state in states)


def _assert_same_metrics(testcase, actual, expected):
    testcase.assertLen(actual.segments, len(expected.segments))
    for actual_segment, expected_segment in zip(actual.segments, expected.segments, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_segment.nll_sum_B),
            np.asarray(expected_segment.nll_sum_B),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(actual_segment.token_count_B),
            np.asarray(expected_segment.token_count_B),
        )


def _recording_run_chain(recorded_inputs):
    def wrapped(*args, condition, segment_input_fn, **kwargs):
        condition_inputs = []
        recorded_inputs[condition] = condition_inputs

        def record_inputs(segment_idx, previous_state):
            inputs = segment_input_fn(segment_idx, previous_state)
            condition_inputs.append(inputs)
            return inputs

        return run_chain(
            *args,
            condition=condition,
            segment_input_fn=record_inputs,
            **kwargs,
        )

    return wrapped


class EvalExperimentsTest(absltest.TestCase):
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

    def _batch(self, *, four_documents=False):
        arrays = four_document_chain_arrays() if four_documents else two_document_chain_arrays()
        return ChainBatch(
            document_ids=tuple(f"doc-{idx}" for idx in range(arrays["token_ids_BCT"].shape[0])),
            **{name: jnp.asarray(value) for name, value in arrays.items()},
        )

    def _offsets(self, batch, segment_idx):
        return batch.chunk_indices_BC[:, segment_idx] * batch.token_ids_BCT.shape[-1]

    def test_gdn_controls_use_true_conv_and_true_donor_gdn_states(self):
        batch = self._batch(four_documents=True)
        donors = jnp.asarray([2, 3, 0, 1], dtype=jnp.int32)
        state_cfg = StatePassingConfig(
            pass_gdn_state=True,
            gdn_layer_limit=None,
            pass_conv_state=True,
            pass_rope_positions=True,
        )

        def true_inputs(segment_idx, previous):
            offsets = self._offsets(batch, segment_idx)
            if previous is None:
                return SegmentInputs(position_offsets_B=offsets)
            return SegmentInputs(
                gdn_states=previous.gdn_states,
                conv_states=previous.conv_states,
                position_offsets_B=offsets,
            )

        true_oracle = run_chain(
            self.model,
            self.cfg,
            batch,
            pad_id=0,
            condition="oracle_true_gdn",
            segment_input_fn=true_inputs,
        )
        experiment = run_gdn_experiment(
            self.model,
            self.cfg,
            batch,
            state_config=state_cfg,
            donor_indices_B=donors,
            pad_id=0,
            checkpoint="tiny",
        )
        self.assertEqual(
            tuple(experiment.conditions),
            ("true_gdn", "zero_gdn", "shuffled_gdn"),
        )
        _assert_same_metrics(self, experiment.conditions["true_gdn"], true_oracle)

        def expected_inputs(segment_idx, _previous, *, shuffled):
            offsets = self._offsets(batch, segment_idx)
            if segment_idx == 0:
                return SegmentInputs(position_offsets_B=offsets)
            true_state = true_oracle.segments[segment_idx - 1].state
            gdn_states = (
                _take_donors(true_state.gdn_states, donors)
                if shuffled
                else _zeros(true_state.gdn_states)
            )
            return SegmentInputs(
                gdn_states=gdn_states,
                conv_states=true_state.conv_states,
                position_offsets_B=offsets,
            )

        expected_zero = run_chain(
            self.model,
            self.cfg,
            batch,
            pad_id=0,
            condition="expected_zero_gdn",
            segment_input_fn=lambda idx, previous: expected_inputs(idx, previous, shuffled=False),
        )
        expected_shuffled = run_chain(
            self.model,
            self.cfg,
            batch,
            pad_id=0,
            condition="expected_shuffled_gdn",
            segment_input_fn=lambda idx, previous: expected_inputs(idx, previous, shuffled=True),
        )
        _assert_same_metrics(self, experiment.conditions["zero_gdn"], expected_zero)
        _assert_same_metrics(
            self,
            experiment.conditions["shuffled_gdn"],
            expected_shuffled,
        )

    def test_conv_controls_use_true_gdn_and_true_donor_conv_states(self):
        batch = self._batch(four_documents=True)
        donors = jnp.asarray([2, 3, 0, 1], dtype=jnp.int32)
        state_cfg = StatePassingConfig(True, None, True, True)

        def true_inputs(segment_idx, previous):
            offsets = self._offsets(batch, segment_idx)
            if previous is None:
                return SegmentInputs(position_offsets_B=offsets)
            return SegmentInputs(
                gdn_states=previous.gdn_states,
                conv_states=previous.conv_states,
                position_offsets_B=offsets,
            )

        true_oracle = run_chain(
            self.model,
            self.cfg,
            batch,
            pad_id=0,
            condition="oracle_true_conv",
            segment_input_fn=true_inputs,
        )
        experiment = run_conv_experiment(
            self.model,
            self.cfg,
            batch,
            state_config=state_cfg,
            donor_indices_B=donors,
            pad_id=0,
            checkpoint="tiny",
        )
        self.assertEqual(
            tuple(experiment.conditions),
            ("true_conv", "zero_conv", "shuffled_conv"),
        )
        _assert_same_metrics(self, experiment.conditions["true_conv"], true_oracle)

        def expected_inputs(segment_idx, _previous, *, shuffled):
            offsets = self._offsets(batch, segment_idx)
            if segment_idx == 0:
                return SegmentInputs(position_offsets_B=offsets)
            true_state = true_oracle.segments[segment_idx - 1].state
            conv_states = (
                _take_donors(true_state.conv_states, donors)
                if shuffled
                else _zeros(true_state.conv_states)
            )
            return SegmentInputs(
                gdn_states=true_state.gdn_states,
                conv_states=conv_states,
                position_offsets_B=offsets,
            )

        expected_zero = run_chain(
            self.model,
            self.cfg,
            batch,
            pad_id=0,
            condition="expected_zero_conv",
            segment_input_fn=lambda idx, previous: expected_inputs(idx, previous, shuffled=False),
        )
        expected_shuffled = run_chain(
            self.model,
            self.cfg,
            batch,
            pad_id=0,
            condition="expected_shuffled_conv",
            segment_input_fn=lambda idx, previous: expected_inputs(idx, previous, shuffled=True),
        )
        _assert_same_metrics(self, experiment.conditions["zero_conv"], expected_zero)
        _assert_same_metrics(
            self,
            experiment.conditions["shuffled_conv"],
            expected_shuffled,
        )

        disabled = StatePassingConfig(True, None, False, True)
        with self.assertRaisesRegex(ValueError, "Conv.*not applicable"):
            run_conv_experiment(
                self.model,
                self.cfg,
                batch,
                state_config=disabled,
                donor_indices_B=donors,
                pad_id=0,
                checkpoint="tiny",
            )

    def test_disabled_non_target_states_and_gdn_without_training_state_passing(self):
        batch = self._batch()
        donors = jnp.asarray([1, 0], dtype=jnp.int32)
        recorded_inputs = {}

        with mock.patch.object(
            experiments,
            "run_chain",
            side_effect=_recording_run_chain(recorded_inputs),
        ):
            gdn = run_gdn_experiment(
                self.model,
                self.cfg,
                batch,
                state_config=StatePassingConfig(False, None, False, True),
                donor_indices_B=donors,
                pad_id=0,
                checkpoint="tiny",
            )
            run_conv_experiment(
                self.model,
                self.cfg,
                batch,
                state_config=StatePassingConfig(False, None, True, True),
                donor_indices_B=donors,
                pad_id=0,
                checkpoint="tiny",
            )

        self.assertEqual(
            tuple(gdn.conditions),
            ("true_gdn", "zero_gdn", "shuffled_gdn"),
        )
        for condition in ("true_gdn", "zero_gdn", "shuffled_gdn"):
            self.assertTrue(
                all(inputs.conv_states is None for inputs in recorded_inputs[condition])
            )
        for condition in ("true_gdn", "shuffled_gdn"):
            self.assertTrue(
                all(inputs.gdn_states is not None for inputs in recorded_inputs[condition][1:])
            )
        for condition in ("true_conv", "zero_conv", "shuffled_conv"):
            self.assertTrue(all(inputs.gdn_states is None for inputs in recorded_inputs[condition]))

    def test_gdn_and_conv_position_offsets_follow_state_config(self):
        batch = self._batch()
        donors = jnp.asarray([1, 0], dtype=jnp.int32)
        for pass_rope_positions in (True, False):
            state_cfg = StatePassingConfig(True, None, True, pass_rope_positions)
            recorded_inputs = {}
            with self.subTest(pass_rope_positions=pass_rope_positions):
                with mock.patch.object(
                    experiments,
                    "run_chain",
                    side_effect=_recording_run_chain(recorded_inputs),
                ):
                    run_gdn_experiment(
                        self.model,
                        self.cfg,
                        batch,
                        state_config=state_cfg,
                        donor_indices_B=donors,
                        pad_id=0,
                        checkpoint="tiny",
                    )
                    run_conv_experiment(
                        self.model,
                        self.cfg,
                        batch,
                        state_config=state_cfg,
                        donor_indices_B=donors,
                        pad_id=0,
                        checkpoint="tiny",
                    )

                if pass_rope_positions:
                    expected_offsets = np.asarray(batch.chunk_indices_BC) * int(
                        batch.token_ids_BCT.shape[-1]
                    )
                else:
                    expected_offsets = np.zeros_like(
                        np.asarray(batch.chunk_indices_BC),
                        dtype=np.int32,
                    )
                for condition in (
                    "true_gdn",
                    "zero_gdn",
                    "shuffled_gdn",
                    "true_conv",
                    "zero_conv",
                    "shuffled_conv",
                ):
                    np.testing.assert_array_equal(
                        np.stack(
                            [
                                np.asarray(inputs.position_offsets_B)
                                for inputs in recorded_inputs[condition]
                            ],
                            axis=1,
                        ),
                        expected_offsets,
                    )

    def test_layer_limit_applies_to_gdn_and_conv_reference_inputs(self):
        batch = self._batch()
        donors = jnp.asarray([1, 0], dtype=jnp.int32)
        state_cfg = StatePassingConfig(True, 1, True, True)

        def limited_inputs(segment_idx, previous):
            offsets = self._offsets(batch, segment_idx)
            if previous is None:
                return SegmentInputs(position_offsets_B=offsets)
            return SegmentInputs(
                gdn_states=(
                    previous.gdn_states[0],
                    jnp.zeros_like(previous.gdn_states[1]),
                ),
                conv_states=(
                    previous.conv_states[0],
                    jnp.zeros_like(previous.conv_states[1]),
                ),
                position_offsets_B=offsets,
            )

        expected_reference = run_chain(
            self.model,
            self.cfg,
            batch,
            pad_id=0,
            condition="expected_limited_reference",
            segment_input_fn=limited_inputs,
        )
        recorded_inputs = {}
        with mock.patch.object(
            experiments,
            "run_chain",
            side_effect=_recording_run_chain(recorded_inputs),
        ):
            gdn = run_gdn_experiment(
                self.model,
                self.cfg,
                batch,
                state_config=state_cfg,
                donor_indices_B=donors,
                pad_id=0,
                checkpoint="tiny",
            )
            conv = run_conv_experiment(
                self.model,
                self.cfg,
                batch,
                state_config=state_cfg,
                donor_indices_B=donors,
                pad_id=0,
                checkpoint="tiny",
            )

        self.assertLen(gdn.conditions["true_gdn"].segments[0].state.gdn_states, 2)
        self.assertLen(conv.conditions["true_conv"].segments[0].state.conv_states, 2)
        _assert_same_metrics(self, gdn.conditions["true_gdn"], expected_reference)
        _assert_same_metrics(self, conv.conditions["true_conv"], expected_reference)

        def assert_states_close(actual_states, expected_states):
            self.assertIsNotNone(actual_states)
            for actual, expected in zip(actual_states, expected_states, strict=True):
                np.testing.assert_allclose(
                    np.asarray(actual),
                    np.asarray(expected),
                    rtol=1e-6,
                    atol=1e-6,
                )

        for segment_idx in range(1, batch.token_ids_BCT.shape[1]):
            true_state = expected_reference.segments[segment_idx - 1].state
            donor_gdn = _take_donors(true_state.gdn_states, donors)
            donor_conv = _take_donors(true_state.conv_states, donors)
            gdn_inputs = recorded_inputs["shuffled_gdn"][segment_idx]
            conv_inputs = recorded_inputs["shuffled_conv"][segment_idx]
            limited_gdn = (
                true_state.gdn_states[0],
                jnp.zeros_like(true_state.gdn_states[1]),
            )
            limited_conv = (
                true_state.conv_states[0],
                jnp.zeros_like(true_state.conv_states[1]),
            )
            shuffled_gdn = (
                (
                    donor_gdn[0],
                    jnp.zeros_like(true_state.gdn_states[1]),
                ),
                limited_conv,
            )
            shuffled_conv = (
                limited_gdn,
                (
                    donor_conv[0],
                    jnp.zeros_like(true_state.conv_states[1]),
                ),
            )
            zero_gdn_inputs = recorded_inputs["zero_gdn"][segment_idx]
            zero_conv_inputs = recorded_inputs["zero_conv"][segment_idx]
            assert_states_close(
                zero_gdn_inputs.gdn_states,
                tuple(jnp.zeros_like(state) for state in true_state.gdn_states),
            )
            assert_states_close(zero_gdn_inputs.conv_states, limited_conv)
            assert_states_close(zero_conv_inputs.gdn_states, limited_gdn)
            assert_states_close(
                zero_conv_inputs.conv_states,
                tuple(jnp.zeros_like(state) for state in true_state.conv_states),
            )
            assert_states_close(gdn_inputs.gdn_states, shuffled_gdn[0])
            assert_states_close(gdn_inputs.conv_states, shuffled_gdn[1])
            assert_states_close(conv_inputs.gdn_states, shuffled_conv[0])
            assert_states_close(conv_inputs.conv_states, shuffled_conv[1])

        for invalid_limit in (-1, 3):
            invalid = StatePassingConfig(True, invalid_limit, True, True)
            for run_experiment in (run_gdn_experiment, run_conv_experiment):
                with self.subTest(
                    invalid_limit=invalid_limit,
                    entrypoint=run_experiment.__name__,
                ):
                    with self.assertRaisesRegex(ValueError, "gdn_layer_limit"):
                        run_experiment(
                            self.model,
                            self.cfg,
                            batch,
                            state_config=invalid,
                            donor_indices_B=donors,
                            pad_id=0,
                            checkpoint="tiny",
                        )

    def test_all_six_conditions_recompute_complete_chains_and_assert_chunk1(self):
        batch = self._batch()
        donors = jnp.asarray([1, 0], dtype=jnp.int32)
        state_cfg = StatePassingConfig(True, None, True, True)

        with (
            mock.patch.object(experiments, "run_chain", wraps=run_chain) as chain_spy,
            mock.patch.object(
                experiments,
                "assert_chunk1_consistent",
                wraps=assert_chunk1_consistent,
            ) as consistency_spy,
        ):
            gdn = run_gdn_experiment(
                self.model,
                self.cfg,
                batch,
                state_config=state_cfg,
                donor_indices_B=donors,
                pad_id=0,
                checkpoint="tiny",
            )
            conv = run_conv_experiment(
                self.model,
                self.cfg,
                batch,
                state_config=state_cfg,
                donor_indices_B=donors,
                pad_id=0,
                checkpoint="tiny",
            )

        self.assertCountEqual(
            [call.kwargs["condition"] for call in chain_spy.call_args_list],
            (
                "true_gdn",
                "zero_gdn",
                "shuffled_gdn",
                "true_conv",
                "zero_conv",
                "shuffled_conv",
            ),
        )
        for experiment in (gdn, conv):
            for result in experiment.conditions.values():
                self.assertLen(result.segments, 3)
        self.assertCountEqual(
            [
                (call.args[0].condition, call.args[1].condition)
                for call in consistency_spy.call_args_list
            ],
            (
                ("true_gdn", "zero_gdn"),
                ("true_gdn", "shuffled_gdn"),
                ("true_conv", "zero_conv"),
                ("true_conv", "shuffled_conv"),
            ),
        )
        for call in consistency_spy.call_args_list:
            self.assertEqual(call.kwargs["checkpoint"], "tiny")
            self.assertEqual(call.kwargs["atol"], 1e-6)

        for run_experiment in (run_gdn_experiment, run_conv_experiment):
            with self.subTest(entrypoint=run_experiment.__name__):
                with mock.patch.object(
                    experiments,
                    "assert_chunk1_consistent",
                    side_effect=ValueError("chunk1 divergence"),
                ):
                    with self.assertRaisesRegex(ValueError, "chunk1 divergence"):
                        run_experiment(
                            self.model,
                            self.cfg,
                            batch,
                            state_config=state_cfg,
                            donor_indices_B=donors,
                            pad_id=0,
                            checkpoint="tiny",
                        )


if __name__ == "__main__":
    absltest.main()
