"""Production-contract tests for the controlled symbolic memory tasks."""

import json
from pathlib import Path
import tempfile

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from omegalax.data.memory_tasks import (
    MemoryTask,
    MemoryTaskConfig,
    MemoryTaskSplit,
    build_memory_task_batch,
    make_memory_task_iterator,
)
from omegalax.data.pretrain_data_set import BATCH_PRETRAIN_METADATA_KEY
from omegalax.evals import memory_tasks as memory_evals
from omegalax.evals.memory_tasks import (
    MemoryEvalMode,
    evaluate_memory_batch,
    swap_counterfactual_pairs,
)
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.shard_config import ShardConfig
from omegalax.text import api as text_api
from omegalax.trainers import pretrain as pretrain_trainer
from scripts.train_memory_tasks import TrainingCondition, training_layout
from scripts.eval_memory_tasks import _validate_training_contract, summarize_rows


def _tiny_gdn_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=("linear_attention",),
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
        param_dtype=jnp.float32,
    )


class MemoryTaskTest(absltest.TestCase):
    def test_eval_summary_keeps_answer_and_donor_metrics_separate(self):
        rows = [
            {
                "mode": "correct_state",
                "correct": True,
                "donor_correct": False,
                "answer_nll": 0.1,
                "donor_answer_nll": 3.0,
            },
            {
                "mode": "correct_state",
                "correct": False,
                "donor_correct": True,
                "answer_nll": 1.1,
                "donor_answer_nll": 0.5,
            },
        ]

        summary = summarize_rows(rows)["correct_state"]

        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["donor_accuracy"], 0.5)
        self.assertAlmostEqual(summary["answer_nll"], 0.6)
        self.assertAlmostEqual(summary["donor_answer_nll"], 1.75)

    def test_training_conditions_have_matched_token_budgets(self):
        layouts = {
            condition: training_layout(
                condition,
                pairs_per_batch=3,
                num_segments=2,
                segment_length=16,
            )
            for condition in TrainingCondition
        }

        self.assertEqual({layout.batch_size * layout.seq_len for layout in layouts.values()}, {192})
        self.assertTrue(layouts[TrainingCondition.FULL_CONTEXT].flatten)
        self.assertFalse(layouts[TrainingCondition.STATE].flatten)
        self.assertFalse(layouts[TrainingCondition.RESET].pass_gdn_state)
        self.assertTrue(layouts[TrainingCondition.STATE].pass_gdn_state)

    def test_exact_transport_pair_has_shared_query_and_one_answer_target(self):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=17,
            segment_length=16,
            num_segments=2,
        )
        batch = build_memory_task_batch((3,), config, flatten=False)

        self.assertEqual(batch["token_ids_BCT"].shape, (2, 2, 16))
        np.testing.assert_array_equal(
            batch["token_ids_BCT"][0, -1, :-1],
            batch["token_ids_BCT"][1, -1, :-1],
        )
        self.assertNotEqual(
            batch["token_ids_BCT"][0, 0].tolist(),
            batch["token_ids_BCT"][1, 0].tolist(),
        )
        self.assertNotEqual(
            int(batch["token_ids_BCT"][0, -1, -1]),
            int(batch["token_ids_BCT"][1, -1, -1]),
        )
        np.testing.assert_array_equal(batch["loss_mask_BCT"].sum(axis=(1, 2)), [1, 1])
        np.testing.assert_array_equal(batch["loss_mask_BCT"][:, -1, -1], [1, 1])
        np.testing.assert_array_equal(batch["reset_state_BC"], [[True, False], [True, False]])

        flat = build_memory_task_batch((3,), config, flatten=True)
        np.testing.assert_array_equal(flat["token_ids_BT"], batch["token_ids_BCT"].reshape(2, 32))
        np.testing.assert_array_equal(flat["loss_mask_BT"].sum(axis=1), [1, 1])

    def test_dominant_rule_pair_requires_the_majority_convention(self):
        config = MemoryTaskConfig(
            task=MemoryTask.DOMINANT_RULE,
            seed=23,
            segment_length=24,
            num_segments=2,
        )
        batch = build_memory_task_batch((5,), config, flatten=False)
        metadata = batch[BATCH_PRETRAIN_METADATA_KEY]
        answers = metadata["answer_token_B"]

        np.testing.assert_array_equal(
            batch["token_ids_BCT"][0, -1, :-1],
            batch["token_ids_BCT"][1, -1, :-1],
        )
        self.assertNotEqual(int(answers[0]), int(answers[1]))
        for world in (0, 1):
            history = batch["token_ids_BCT"][world, 0]
            self.assertEqual(int(np.sum(history == answers[world])), 3)
            self.assertEqual(int(np.sum(history == answers[1 - world])), 2)
            self.assertEqual(int(batch["token_ids_BCT"][world, -1, -1]), int(answers[world]))

    def test_grain_iterator_batches_and_shards_whole_pairs(self):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=29,
            segment_length=16,
            num_segments=2,
        )
        iterator = make_memory_task_iterator(
            config,
            global_batch_size=8,
            num_batches=2,
            flatten=False,
            dp_size=1,
            fsdp_size=1,
            process_index=0,
        )

        first = next(iterator)[BATCH_PRETRAIN_METADATA_KEY]
        second = next(iterator)[BATCH_PRETRAIN_METADATA_KEY]
        np.testing.assert_array_equal(first["pair_index_B"], [0, 0, 1, 1])
        np.testing.assert_array_equal(first["world_B"], [0, 1, 0, 1])
        np.testing.assert_array_equal(second["pair_index_B"], [2, 2, 3, 3])
        np.testing.assert_array_equal(second["world_B"], [0, 1, 0, 1])

    def test_real_statepassing_trainer_consumes_answer_only_batch(self):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=31,
            segment_length=16,
            num_segments=2,
            vocab_size=512,
        )
        iterator = make_memory_task_iterator(
            config,
            global_batch_size=4,
            num_batches=1,
            flatten=False,
            dp_size=1,
            fsdp_size=1,
            process_index=0,
        )
        train_config = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=4,
            seq_len=16,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )

        _, metrics = pretrain_trainer.run_pretrain(
            _tiny_gdn_config(),
            train_config,
            iterator,
            pretrain_mode=pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
            text_attn_backend="xla",
            bptt_chunks=None,
            pass_gdn_state=True,
        )

        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertEqual(metrics["supervised_tokens"], 2.0)

    def test_real_model_evaluator_runs_all_causal_conditions(self):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=37,
            segment_length=16,
            num_segments=2,
            vocab_size=512,
        )
        batch = build_memory_task_batch((0,), config, flatten=False)
        model, model_config = text_api.init_model(
            _tiny_gdn_config(),
            jax.random.key(0),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )

        results = evaluate_memory_batch(model, model_config, batch)

        self.assertEqual(set(results), set(MemoryEvalMode))
        for result in results.values():
            self.assertBetween(result.accuracy, 0.0, 1.0)
            self.assertTrue(np.isfinite(result.answer_nll))
            self.assertEqual(result.predictions_B.shape, (2,))
        np.testing.assert_array_equal(
            results[MemoryEvalMode.RESET].predictions_B,
            results[MemoryEvalMode.ZERO_STATE].predictions_B,
        )
        self.assertAlmostEqual(
            results[MemoryEvalMode.RESET].answer_nll,
            results[MemoryEvalMode.ZERO_STATE].answer_nll,
        )

    def test_counterfactual_swap_uses_the_other_world_in_each_pair(self):
        state = (jnp.asarray([[10], [20], [30], [40]]),)
        swapped = swap_counterfactual_pairs(
            state,
            pair_index_B=np.asarray([7, 7, 9, 9]),
            world_B=np.asarray([0, 1, 0, 1]),
        )
        np.testing.assert_array_equal(np.asarray(swapped[0])[:, 0], [20, 10, 40, 30])

    def test_counterfactual_state_evaluator_actually_consumes_the_donor_state(self):
        # The final segment's tokens up to (but excluding) the answer position are
        # byte-identical between paired worlds, so the only way COUNTERFACTUAL_STATE's
        # hidden state for world w can differ from CORRECT_STATE's hidden state for
        # world w is via the swapped-in donor state. We check this immediately after
        # the swap (position 0 of the final segment) rather than at the answer position
        # many steps later: an untrained GDN gate's forget rate is random and, at this
        # tiny config, decays a fed-in state to a difference of exactly 0.0 in float32
        # within ~3 steps (verified empirically across 12 model-init seeds), so
        # checking at the answer position would pass identically whether or not the
        # swap is even wired up -- not because the wiring is correct, but because the
        # untrained kernel forgets any initial state before it gets that far.
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=43,
            segment_length=16,
            num_segments=2,
            vocab_size=512,
        )
        batch = build_memory_task_batch((0,), config, flatten=False)
        # The task generator places the bind value at position 3 of segment 0 (12
        # steps before the segment ends), which the untrained GDN gate already fully
        # forgets by the time h_end_0 is extracted -- before any swap even happens.
        # Move it to the last position of segment 0 (a permutation of otherwise
        # identical filler, verified elsewhere to be a byte-for-byte swap of two
        # positions) so h_end_0 genuinely differs between worlds at the swap point.
        mutated_tokens = np.array(batch["token_ids_BCT"])
        mutated_tokens[:, 0, [3, -1]] = mutated_tokens[:, 0, [-1, 3]]
        token_ids_BCT = jnp.asarray(mutated_tokens, dtype=jnp.int32)
        attention_mask_BCT = jnp.asarray(batch["attention_mask_BCT"], dtype=jnp.int32)
        metadata = batch[BATCH_PRETRAIN_METADATA_KEY]
        pair_index_B = np.asarray(metadata["pair_index_B"])
        world_B = np.asarray(metadata["world_B"])
        model, model_config = text_api.init_model(
            _tiny_gdn_config(), jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1
        )

        correct_hidden = memory_evals._segmented_final_hidden(
            model,
            model_config,
            token_ids_BCT,
            attention_mask_BCT,
            batch["reset_state_BC"],
            MemoryEvalMode.CORRECT_STATE,
            pair_index_B=pair_index_B,
            world_B=world_B,
        )
        counterfactual_hidden = memory_evals._segmented_final_hidden(
            model,
            model_config,
            token_ids_BCT,
            attention_mask_BCT,
            batch["reset_state_BC"],
            MemoryEvalMode.COUNTERFACTUAL_STATE,
            pair_index_B=pair_index_B,
            world_B=world_B,
        )
        boundary_position = 0

        # Canary: the swap must actually change the outcome relative to each window's
        # own (unswapped) state -- otherwise this test would pass identically whether
        # or not the swap wiring is broken.
        self.assertFalse(
            np.allclose(
                counterfactual_hidden[:, boundary_position],
                correct_hidden[:, boundary_position],
            )
        )
        # Correctness: window w's post-swap hidden state must exactly match what
        # CORRECT_STATE computes for the *other* world at the same position.
        np.testing.assert_allclose(
            np.asarray(counterfactual_hidden[:, boundary_position]),
            np.asarray(correct_hidden[::-1, boundary_position]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_evaluator_rejects_reset_state_that_violates_segment_zero_invariant(self):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=41,
            segment_length=16,
            num_segments=2,
            vocab_size=512,
        )
        batch = build_memory_task_batch((0,), config, flatten=False)
        batch["reset_state_BC"][:, 1] = True
        model, model_config = text_api.init_model(
            _tiny_gdn_config(), jax.random.key(0), tp_size=1, fsdp_size=1, dp_size=1
        )

        with self.assertRaises(ValueError):
            evaluate_memory_batch(model, model_config, batch)

    def test_train_and_eval_splits_generate_disjoint_streams_even_with_same_seed(self):
        train_config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=5,
            segment_length=16,
            num_segments=2,
            split=MemoryTaskSplit.TRAIN,
        )
        eval_config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=5,
            segment_length=16,
            num_segments=2,
            split=MemoryTaskSplit.EVAL,
        )

        train_batch = build_memory_task_batch((3,), train_config, flatten=False)
        eval_batch = build_memory_task_batch((3,), eval_config, flatten=False)

        self.assertFalse(np.array_equal(train_batch["token_ids_BCT"], eval_batch["token_ids_BCT"]))

    def test_eval_contract_validation_rejects_task_or_split_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memory_task_contract.json").write_text(
                json.dumps({"task": {"task": "exact", "split": "train"}})
            )

            _validate_training_contract(root, MemoryTask.EXACT)

            with self.assertRaises(ValueError):
                _validate_training_contract(root, MemoryTask.DOMINANT_RULE)

            (root / "memory_task_contract.json").write_text(
                json.dumps({"task": {"task": "exact", "split": "eval"}})
            )
            with self.assertRaises(ValueError):
                _validate_training_contract(root, MemoryTask.EXACT)


if __name__ == "__main__":
    absltest.main()
