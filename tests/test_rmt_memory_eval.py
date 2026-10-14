"""RMT passing in the synthetic memory evaluator.

Everything runs the real stack on CPU: a tiny Qwen3.5 hybrid (GDN + conv + full attention
+ RMT memory), the real memory-task generator, and the production evaluator/CLIs. The
only independent code is ``_Oracle`` below, which re-derives each evaluation mode from
direct public text-API calls instead of the evaluator's own segment/carry helpers.
"""

import contextlib
import io
import itertools
import json
import logging
import os
import socket
from unittest import mock

from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from omegalax.data.memory_tasks import MemoryTask, MemoryTaskConfig, build_memory_task_batch
from omegalax.data.pretrain_data_set import BATCH_PRETRAIN_METADATA_KEY
from omegalax.evals.memory_tasks import MemoryEvalMode, make_memory_evaluator
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.shard_config import ShardConfig
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.trainers import pretrain as pretrain_trainer
from scripts import build_memory_task_eval_manifest, eval_memory_tasks, train_memory_tasks
from tests.pretrain_real_data_test_utils import test_temp_dir

_NUM_RMT_TOKENS = 2
_NUM_SEGMENTS = 3
_SEGMENT_LENGTH = 16
_CHANNELS = ("gdn", "conv", "rmt")


def _tiny_hybrid_config(num_rmt_tokens: int) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=("linear_attention", "linear_attention", "full_attention"),
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
        num_rmt_tokens=num_rmt_tokens,
    )


def _make_model(num_rmt_tokens: int):
    model, config = text_api.init_model(
        _tiny_hybrid_config(num_rmt_tokens),
        jax.random.key(0),
        tp_size=1,
        fsdp_size=1,
        dp_size=1,
    )
    set_attn_backend(model, text_backend="xla")
    # An untrained GDN gate forgets a carried state within a few tokens and a 0.02-std
    # memory barely moves the text, so a swapped channel would be invisible at the answer.
    # Stand in for training: freeze the gate decay and give the memory unit scale.
    for layer in model.text.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.A_log[...] = jnp.full_like(layer.linear_attn.A_log[...], -10.0)
    if num_rmt_tokens:
        model.rmt_memory[...] = model.rmt_memory[...] * 50
    return model, config


def _memory_batch(exact_sequence_length: int, pair_index: int = 0) -> dict:
    task = MemoryTaskConfig(
        task=MemoryTask.EXACT,
        seed=37,
        segment_length=_SEGMENT_LENGTH,
        num_segments=_NUM_SEGMENTS,
        vocab_size=512,
        exact_sequence_length=exact_sequence_length,
    )
    return build_memory_task_batch((pair_index,), task, flatten=False)


class _Oracle:
    """Re-derives each mode with direct public text-API calls, independent of the evaluator.

    The calls are wrapped in ``nnx.jit`` only for CPU speed; all bundle/mode/generation logic
    below is written out separately from ``omegalax.evals.memory_tasks``.
    """

    def __init__(self, model, config):
        self.model = model

        @nnx.jit
        def prefix_segment(model, tokens_BT, mask_BT, gdn_states, conv_states, rmt_memory):
            _, _, gdn_states, conv_states, rmt_memory = text_api.forward_with_gdn_state(
                model,
                tokens_BT,
                0,
                config,
                attention_mask_BT=mask_BT,
                initial_gdn_states=gdn_states,
                initial_conv_states=conv_states,
                initial_rmt_memory_BMD=rmt_memory,
                return_conv_states=True,
                return_rmt_memory=True,
            )
            return gdn_states, conv_states, rmt_memory

        @nnx.jit
        def final_segment(model, tokens_BT, mask_BT, gdn_states, conv_states, rmt_memory):
            hidden_BTD, *_ = text_api.forward_with_gdn_state(
                model,
                tokens_BT,
                0,
                config,
                attention_mask_BT=mask_BT,
                initial_gdn_states=gdn_states,
                initial_conv_states=conv_states,
                initial_rmt_memory_BMD=rmt_memory,
            )
            return hidden_BTD

        @nnx.jit
        def full_context(model, tokens_BT, mask_BT):
            hidden_BTD, _ = text_api.forward(model, tokens_BT, 0, config, attention_mask_BT=mask_BT)
            return hidden_BTD

        self.prefix_segment = prefix_segment
        self.final_segment = final_segment
        self.full_context = full_context

    def __call__(self, batch, mode, *, swapped=_CHANNELS):
        """Return (predictions, own-answer NLL, donor-answer NLL) of one mode."""
        model = self.model
        tokens_BCT = jnp.asarray(batch["token_ids_BCT"], dtype=jnp.int32)
        mask_BCT = jnp.asarray(batch["attention_mask_BCT"], dtype=jnp.int32)
        metadata = batch[BATCH_PRETRAIN_METADATA_KEY]
        answers_BD = jnp.asarray(metadata["answer_tokens_BD"], dtype=jnp.int32)
        positions_BD = jnp.asarray(metadata["answer_positions_BD"], dtype=jnp.int32)
        batch_size, num_segments, segment_length = tokens_BCT.shape
        rows_B = jnp.arange(batch_size)
        donor_B = np.arange(batch_size) ^ 1  # worlds 0/1 of one pair are adjacent
        donor_answers_BD = answers_BD[donor_B]
        donor_tokens_BCT = tokens_BCT.at[rows_B[:, None], -1, positions_BD].set(donor_answers_BD)

        bundle = (None, None, None)
        if mode not in (MemoryEvalMode.FULL_CONTEXT, MemoryEvalMode.RESET):
            bundle = (None, None, None)
            for segment_idx in range(num_segments - 1):
                bundle = self.prefix_segment(
                    model, tokens_BCT[:, segment_idx], mask_BCT[:, segment_idx], *bundle
                )
            if mode is MemoryEvalMode.ZERO_STATE:
                bundle = jax.tree.map(jnp.zeros_like, bundle)
            elif mode is MemoryEvalMode.COUNTERFACTUAL_STATE:
                bundle = tuple(
                    jax.tree.map(lambda state: state[donor_B], channel)
                    if name in swapped
                    else channel
                    for name, channel in zip(_CHANNELS, bundle, strict=True)
                )

        def hidden_and_positions(candidate_tokens_BCT):
            if mode is MemoryEvalMode.FULL_CONTEXT:
                hidden_BTD = self.full_context(
                    model,
                    candidate_tokens_BCT.reshape(batch_size, -1),
                    mask_BCT.reshape(batch_size, -1),
                )
                return hidden_BTD, positions_BD + (num_segments - 1) * segment_length
            hidden_BTD = self.final_segment(
                model, candidate_tokens_BCT[:, -1], mask_BCT[:, -1], *bundle
            )
            return hidden_BTD, positions_BD

        def teacher_forced_nll(candidate_tokens_BCT, targets_BD):
            hidden_BTD, positions = hidden_and_positions(candidate_tokens_BCT)
            logits_BDV = hidden_BTD[rows_B[:, None], positions - 1] @ model.output_weight()
            target_logits_BD = jnp.take_along_axis(logits_BDV, targets_BD[..., None], axis=-1)[
                ..., 0
            ]
            return np.asarray(jax.nn.logsumexp(logits_BDV, axis=-1) - target_logits_BD)

        working_tokens_BCT = tokens_BCT
        predictions = []
        for answer_idx in range(answers_BD.shape[1]):
            hidden_BTD, positions = hidden_and_positions(working_tokens_BCT)
            prediction_B = jnp.argmax(
                hidden_BTD[rows_B, positions[:, answer_idx] - 1] @ model.output_weight(), axis=-1
            )
            predictions.append(np.asarray(prediction_B))
            working_tokens_BCT = working_tokens_BCT.at[rows_B, -1, positions_BD[:, answer_idx]].set(
                prediction_B
            )
        return (
            np.stack(predictions, axis=1),
            teacher_forced_nll(tokens_BCT, answers_BD),
            teacher_forced_nll(donor_tokens_BCT, donor_answers_BD),
        )


def setUpModule():
    # tokamax parses sys.argv on its first attention call, which fails when other test
    # modules collected in the same process registered required flags.
    flags.FLAGS.mark_as_parsed()


class _CompileCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.names = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("Compiling jit("):
            self.names.append(message.split(")", 1)[0])


class RmtMemoryEvalTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model, cls.config = _make_model(_NUM_RMT_TOKENS)
        cls.oracle = _Oracle(cls.model, cls.config)
        evaluator = make_memory_evaluator(cls.config, pass_gdn_state=True, pass_conv_state=True)
        cls.evaluator = staticmethod(evaluator)
        cls.batch_d1 = _memory_batch(1)
        cls.batch_d7 = _memory_batch(7)
        cls.results_d1 = evaluator(cls.model, cls.batch_d1)
        cls.results_d7 = evaluator(cls.model, cls.batch_d7)

    def _assert_matches_oracle(self, results, batch):
        for mode in MemoryEvalMode:
            with self.subTest(mode=mode.value):
                predictions_BD, nll_BD, donor_nll_BD = self.oracle(batch, mode)
                np.testing.assert_array_equal(results[mode].predictions_BD, predictions_BD)
                np.testing.assert_allclose(
                    results[mode].answer_nll_BD, nll_BD, rtol=2e-5, atol=2e-5
                )
                np.testing.assert_allclose(
                    results[mode].donor_answer_nll_BD, donor_nll_BD, rtol=2e-5, atol=2e-5
                )

    def test_single_token_modes_carry_reset_zero_and_swap_every_channel(self):
        self._assert_matches_oracle(self.results_d1, self.batch_d1)
        nll = {mode: result.answer_nll_B for mode, result in self.results_d1.items()}
        # Canaries: the modes must actually differ, otherwise matching the oracle is vacuous.
        # Zero memory is not the learned initial memory, so ZERO_STATE differs from RESET.
        for first, second in (
            (MemoryEvalMode.CORRECT_STATE, MemoryEvalMode.COUNTERFACTUAL_STATE),
            (MemoryEvalMode.CORRECT_STATE, MemoryEvalMode.RESET),
            (MemoryEvalMode.ZERO_STATE, MemoryEvalMode.RESET),
        ):
            self.assertGreater(np.max(np.abs(nll[first] - nll[second])), 1e-3, (first, second))

    def test_d7_free_generation_carries_every_channel_in_every_mode(self):
        self._assert_matches_oracle(self.results_d7, self.batch_d7)
        self.assertEqual(self.results_d7[MemoryEvalMode.CORRECT_STATE].predictions_BD.shape, (2, 7))

    def test_counterfactual_swaps_the_complete_bundle_not_a_channel_subset(self):
        production_nll_BD = self.results_d1[MemoryEvalMode.COUNTERFACTUAL_STATE].answer_nll_BD
        subsets = itertools.chain.from_iterable(
            itertools.combinations(_CHANNELS, size) for size in (1, 2)
        )
        for swapped in subsets:
            with self.subTest(swapped=swapped):
                _, partial_nll_BD, _ = self.oracle(
                    self.batch_d1, MemoryEvalMode.COUNTERFACTUAL_STATE, swapped=swapped
                )
                self.assertGreater(np.max(np.abs(partial_nll_BD - production_nll_BD)), 1e-3)

    def test_final_segment_answer_tokens_never_influence_earlier_scores(self):
        metadata = self.batch_d7[BATCH_PRETRAIN_METADATA_KEY]
        answers_BD = np.asarray(metadata["answer_tokens_BD"])
        positions_BD = np.asarray(metadata["answer_positions_BD"])
        tokens_BCT = np.array(self.batch_d7["token_ids_BCT"])
        tokens_BCT[np.arange(tokens_BCT.shape[0])[:, None], -1, positions_BD] = (
            answers_BD + 1
        ) % self.config.vocab_size
        perturbed = self.evaluator(self.model, {**self.batch_d7, "token_ids_BCT": tokens_BCT})

        later_nll_shift = 0.0
        for mode in MemoryEvalMode:
            with self.subTest(mode=mode.value):
                baseline = self.results_d7[mode]
                # Free generation overwrites every answer position with its own prediction.
                np.testing.assert_array_equal(
                    perturbed[mode].predictions_BD, baseline.predictions_BD
                )
                # The first answer is predicted before any answer token is visible.
                np.testing.assert_allclose(
                    perturbed[mode].answer_nll_BD[:, 0], baseline.answer_nll_BD[:, 0], atol=1e-6
                )
                shift_BD = perturbed[mode].answer_nll_BD[:, 1:] - baseline.answer_nll_BD[:, 1:]
                later_nll_shift = max(later_nll_shift, float(np.max(np.abs(shift_BD))))
        # Canary: teacher-forced later answers do read the perturbed context, so the
        # unchanged first answer above is not an artifact of a perturbation that did nothing.
        self.assertGreater(later_nll_shift, 1e-4)

    def test_correct_state_matches_the_trainer_eval_step_for_enabled_channel_sets(self):
        zero_token_model, zero_token_config = _make_model(0)
        full_context_nll_BD = self.results_d7[MemoryEvalMode.FULL_CONTEXT].answer_nll_BD
        for model, config, pass_gdn_state, pass_conv_state in (
            (self.model, self.config, True, True),
            (self.model, self.config, False, False),
            (zero_token_model, zero_token_config, True, False),
            (zero_token_model, zero_token_config, True, True),
        ):
            with self.subTest(
                num_rmt_tokens=config.num_rmt_tokens,
                pass_gdn_state=pass_gdn_state,
                pass_conv_state=pass_conv_state,
            ):
                eval_step = pretrain_trainer.make_pretrain_eval_step(
                    pretrain_trainer.PretrainMode.STATEPASSING_BPTT,
                    config,
                    bptt_chunks=_NUM_SEGMENTS,
                    pass_gdn_state=pass_gdn_state,
                    pass_conv_state=pass_conv_state,
                )
                evaluator = make_memory_evaluator(
                    config, pass_gdn_state=pass_gdn_state, pass_conv_state=pass_conv_state
                )
                for batch in (self.batch_d1, self.batch_d7):
                    trainer_batch = {
                        name: jnp.asarray(value)
                        for name, value in batch.items()
                        if name != BATCH_PRETRAIN_METADATA_KEY
                    }
                    metrics = eval_step(model, trainer_batch)
                    results = evaluator(model, batch)
                    answers_BD = np.asarray(batch[BATCH_PRETRAIN_METADATA_KEY]["answer_tokens_BD"])
                    self.assertEqual(float(metrics["supervised_tokens"]), answers_BD.size)
                    self.assertAlmostEqual(
                        float(metrics["nll"]),
                        results[MemoryEvalMode.CORRECT_STATE].answer_nll,
                        delta=2e-5,
                    )
                    if config.num_rmt_tokens and batch is self.batch_d7:
                        # Full context is a direct-access diagnostic on the same weights: no
                        # carry is applied, so the enabled-channel flags must not change it.
                        np.testing.assert_allclose(
                            results[MemoryEvalMode.FULL_CONTEXT].answer_nll_BD,
                            full_context_nll_BD,
                            rtol=0,
                            atol=1e-6,
                        )
                    if not config.num_rmt_tokens:
                        # Without RMT, all-zero GDN/conv states are the reinitialized states.
                        np.testing.assert_allclose(
                            results[MemoryEvalMode.ZERO_STATE].answer_nll_BD,
                            results[MemoryEvalMode.RESET].answer_nll_BD,
                            atol=1e-6,
                        )

    def test_segment_jit_is_compiled_once_per_carry_structure_and_reused_across_batches(self):
        evaluator = make_memory_evaluator(self.config, pass_gdn_state=True, pass_conv_state=True)
        counter = _CompileCounter()
        root_logger = logging.getLogger()
        root_logger.addHandler(counter)
        self.addCleanup(root_logger.removeHandler, counter)
        with jax.log_compiles():
            evaluator(self.model, self.batch_d7)
            first_batch_compiles = counter.names.count("Compiling jit(run_segment")
            evaluator(self.model, _memory_batch(7, pair_index=1))
            second_batch_compiles = counter.names.count("Compiling jit(run_segment")
        # One executable for the initial all-None carry and one for the carried bundle; the
        # prefix segments, every mode and the D7 re-runs all reuse them.
        self.assertEqual(first_batch_compiles, 2)
        self.assertEqual(second_batch_compiles, first_batch_compiles)


class MemoryChannelResolutionTest(absltest.TestCase):
    def test_reset_and_full_context_training_keep_diagnostic_gdn_carry(self):
        model, config = _make_model(0)
        batch = _memory_batch(1)
        historical = make_memory_evaluator(config)(model, batch)
        for condition in ("reset", "full_context"):
            with self.subTest(condition=condition):
                contract = {
                    "condition": condition,
                    "state": {"pass_gdn_state": False, "pass_conv_state": False},
                }
                channels = eval_memory_tasks._resolve_memory_channels(contract, config)
                self.assertTrue(channels["pass_gdn_state"])
                results = make_memory_evaluator(
                    config,
                    pass_gdn_state=channels["pass_gdn_state"],
                    pass_conv_state=channels["pass_conv_state"],
                )(model, batch)
                np.testing.assert_allclose(
                    results[MemoryEvalMode.CORRECT_STATE].answer_nll_B,
                    historical[MemoryEvalMode.CORRECT_STATE].answer_nll_B,
                    atol=1e-6,
                )
                self.assertGreater(
                    np.max(
                        np.abs(
                            results[MemoryEvalMode.CORRECT_STATE].answer_nll_B
                            - results[MemoryEvalMode.RESET].answer_nll_B
                        )
                    ),
                    1e-3,
                )

    def test_conv_flag_comes_from_the_contract_and_rmt_tokens_from_the_model_config(self):
        contract = {
            "state": {"pass_gdn_state": True, "pass_conv_state": False, "num_rmt_tokens": 2}
        }
        channels = eval_memory_tasks._resolve_memory_channels(contract, _tiny_hybrid_config(2))
        self.assertEqual(
            (channels["pass_gdn_state"], channels["pass_conv_state"], channels["num_rmt_tokens"]),
            (True, False, 2),
        )
        self.assertFalse(channels["full_context"]["carry_applied"])
        self.assertIn("learned initial read memory", channels["full_context"]["rmt"])

    def test_contract_without_rmt_key_is_a_zero_token_checkpoint(self):
        contract = {"state": {"pass_gdn_state": True, "pass_conv_state": True}}
        channels = eval_memory_tasks._resolve_memory_channels(contract, _tiny_hybrid_config(0))
        self.assertEqual(channels["num_rmt_tokens"], 0)
        self.assertEqual(channels["full_context"]["rmt"], "none")

    def test_rmt_token_count_must_agree_between_contract_and_model_config(self):
        state = {"pass_gdn_state": True, "pass_conv_state": True}
        for recorded, config_tokens in ((32, 0), (0, 32), (16, 32)):
            with self.subTest(contract=recorded, model_config=config_tokens):
                contract = {"state": {**state, "num_rmt_tokens": recorded}}
                with self.assertRaisesRegex(ValueError, "num_rmt_tokens"):
                    eval_memory_tasks._resolve_memory_channels(
                        contract, _tiny_hybrid_config(config_tokens)
                    )
        with self.assertRaisesRegex(ValueError, "num_rmt_tokens"):
            eval_memory_tasks._resolve_memory_channels({"state": state}, _tiny_hybrid_config(32))

    def test_contract_without_state_flags_is_rejected(self):
        for contract in ({}, {"state": {"pass_gdn_state": True}}):
            with self.subTest(contract=contract):
                with self.assertRaisesRegex(ValueError, "pass_gdn_state"):
                    eval_memory_tasks._resolve_memory_channels(contract, _tiny_hybrid_config(0))


class RmtMemoryEvalCliTest(absltest.TestCase):
    def test_trained_rmt_checkpoint_is_evaluated_with_the_channels_it_was_trained_with(self):
        slurm_environment = mock.patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "test-job", "SLURMD_NODENAME": socket.gethostname()},
        )
        slurm_environment.start()
        self.addCleanup(slurm_environment.stop)
        with test_temp_dir() as tmp:
            tmpdir = os.fspath(tmp)
            save_dir = os.path.join(tmpdir, "run")
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(
                    [
                        "--task=exact",
                        "--condition=state",
                        "--pass_conv_state",
                        f"--num_rmt_tokens={_NUM_RMT_TOKENS}",
                        f"--save_dir={save_dir}",
                        "--model_id=qwen3.5-smoke-dense",
                        "--seed=7",
                        "--data_seed=11",
                        "--exact_sequence_length=7",
                        "--generator_version=2",
                        "--num_steps=1",
                        "--pairs_per_batch=1",
                        "--log_every=0",
                        f"--jax_cache_dir={os.path.join(tmpdir, 'jax_cache')}",
                        "--text_attn_backend=xla",
                        f"--run_plan_sha256={'a' * 64}",
                        "--success_criterion=unit-test criterion",
                    ]
                )
            contract = json.loads(open(os.path.join(save_dir, "memory_task_contract.json")).read())
            self.assertEqual(
                contract["state"],
                {
                    "pass_gdn_state": True,
                    "bptt_chunks": 2,
                    "pass_rope_positions": False,
                    "pass_conv_state": True,
                    "num_rmt_tokens": _NUM_RMT_TOKENS,
                },
            )

            manifest_path = os.path.join(tmpdir, "eval_manifest.json")
            with contextlib.redirect_stdout(io.StringIO()):
                build_memory_task_eval_manifest.main(
                    [
                        f"--checkpoint={save_dir}",
                        f"--output={manifest_path}",
                        "--task=exact",
                        "--exact_sequence_length=7",
                        "--generator_version=2",
                        "--num_pairs=2",
                    ]
                )
            with (
                mock.patch.dict(eval_memory_tasks._CANONICAL_NUM_PAIRS, {MemoryTask.EXACT: 2}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                eval_memory_tasks.main(
                    [
                        f"--checkpoint={save_dir}",
                        "--task=exact",
                        "--exact_sequence_length=7",
                        "--generator_version=2",
                        f"--eval_manifest={manifest_path}",
                        "--num_pairs=2",
                        "--pairs_per_batch=1",
                        "--text_attn_backend=xla",
                        f"--jax_cache_dir={os.path.join(tmpdir, 'eval_jax_cache')}",
                    ]
                )

            eval_dir = os.path.join(
                save_dir, "evals", "memory_exact_d7_pollution_guard_v1", "step_1"
            )
            eval_config = json.loads(open(os.path.join(eval_dir, "eval_config.json")).read())
            self.assertEqual(
                eval_config["memory_channels"],
                {
                    "pass_gdn_state": True,
                    "pass_conv_state": True,
                    "num_rmt_tokens": _NUM_RMT_TOKENS,
                    "full_context": {
                        "carry_applied": False,
                        "rmt": (
                            "learned initial read memory before the concatenated text; "
                            "write block unused"
                        ),
                    },
                },
            )
            rows = [
                json.loads(line)
                for line in open(os.path.join(eval_dir, "raw.jsonl")).read().splitlines()
            ]
            self.assertLen(rows, 20)
            self.assertTrue(all(np.isfinite(row["answer_nll"]) for row in rows))


if __name__ == "__main__":
    absltest.main()
