"""Production-contract tests for the controlled symbolic memory tasks."""

import contextlib
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest import mock

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from omegalax.data import memory_tasks
from omegalax.data.memory_tasks import (
    MemoryTask,
    MemoryTaskConfig,
    MemoryTaskCue,
    MemoryTaskSplit,
    build_memory_task_batch,
    make_memory_task_iterator,
)
from omegalax.data.pretrain_data_set import BATCH_PRETRAIN_METADATA_KEY
from omegalax.evals import memory_pollution, memory_tasks as memory_evals
from omegalax.evals.memory_tasks import (
    MemoryEvalMode,
    evaluate_memory_batch,
    swap_counterfactual_pairs,
)
from omegalax.evals.memory_pollution import (
    MemoryTrainingStream,
    build_pollution_free_manifest,
    load_training_stream,
    pair_digest,
    table_digest,
    validate_pollution_free_manifest,
)
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig
from omegalax.models.shard_config import ShardConfig
from omegalax.text import api as text_api
from omegalax.trainers import pretrain as pretrain_trainer
from omegalax.trainers import checkpoint_utils
from scripts import (
    audit_memory_task_overlap,
    build_memory_task_eval_manifest,
    eval_memory_tasks,
    train_memory_tasks,
)
from scripts.train_memory_tasks import TrainingCondition, training_layout
from scripts.eval_memory_tasks import (
    _canonical_json_sha256,
    _ensure_not_complete,
    _plot,
    _require_canonical_num_pairs,
    _validate_training_contract,
    _write_artifact_manifest,
    _write_mirror_file,
    summarize_rows,
)
from tests.pretrain_real_data_test_utils import test_temp_dir


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


class MemoryTaskTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        slurm_environment = mock.patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "test-job", "SLURMD_NODENAME": socket.gethostname()},
        )
        slurm_environment.start()
        self.addCleanup(slurm_environment.stop)

    @parameterized.named_parameters(
        (
            "manifest_builder",
            build_memory_task_eval_manifest.main,
            [
                "--checkpoint=/missing/checkpoint",
                "--output=/missing/manifest.json",
                "--task=exact",
                "--num_pairs=2",
            ],
        ),
        (
            "overlap_auditor",
            audit_memory_task_overlap.main,
            [
                "--training_contract=/missing/contract.json",
                "--checkpoint_step=1",
                "--eval_manifest=/missing/manifest.json",
            ],
        ),
        (
            "memory_evaluator",
            eval_memory_tasks.main,
            [
                "--checkpoint=/missing/checkpoint",
                "--task=exact",
                "--eval_manifest=/missing/manifest.json",
                "--num_pairs=2",
                "--jax_cache_dir=/missing/cache",
                "--wiki_mirror_dir=/missing/wiki",
            ],
        ),
    )
    def test_heavy_entrypoints_require_slurm(self, entrypoint, argv):
        with mock.patch.dict(
            os.environ,
            {"SLURM_JOB_ID": "stale-allocation", "SLURMD_NODENAME": "not-this-node"},
        ):
            with self.assertRaisesRegex(RuntimeError, "outside a Slurm compute node"):
                entrypoint(argv)

    @parameterized.parameters(
        "build_memory_task_eval_manifest.py",
        "audit_memory_task_overlap.py",
        "eval_memory_tasks.py",
    )
    def test_direct_cli_rejects_before_third_party_imports(self, script_name):
        environment = dict(os.environ)
        environment.pop("SLURM_JOB_ID", None)
        environment.pop("SLURMD_NODENAME", None)
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "-S", str(repo_root / "scripts" / script_name)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("outside a Slurm compute node", completed.stderr)
        self.assertNotIn("No module named", completed.stderr)

    @parameterized.parameters(
        "build_memory_task_eval_manifest.py",
        "audit_memory_task_overlap.py",
        "eval_memory_tasks.py",
    )
    def test_direct_cli_allows_local_help(self, script_name):
        environment = dict(os.environ)
        environment.pop("SLURM_JOB_ID", None)
        environment.pop("SLURMD_NODENAME", None)
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / script_name), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout)

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

    def test_eval_summary_groups_accuracy_and_nll_by_associative_and_rule_bank_fields(self):
        associative_rows = [
            {
                "mode": "correct_state",
                "correct": True,
                "donor_correct": False,
                "answer_nll": 0.0,
                "donor_answer_nll": 1.0,
                "target_slot": 0,
                "target_segment": 0,
                "target_age": 1,
            },
            {
                "mode": "correct_state",
                "correct": False,
                "donor_correct": True,
                "answer_nll": 2.0,
                "donor_answer_nll": 0.0,
                "target_slot": 1,
                "target_segment": 1,
                "target_age": 0,
            },
        ]

        associative_summary = summarize_rows(associative_rows)["correct_state"]

        self.assertEqual(associative_summary["accuracy_by_target_slot"], {"0": 1.0, "1": 0.0})
        self.assertEqual(associative_summary["answer_nll_by_target_slot"], {"0": 0.0, "1": 2.0})
        self.assertEqual(associative_summary["accuracy_by_target_segment"], {"0": 1.0, "1": 0.0})
        self.assertEqual(associative_summary["answer_nll_by_target_segment"], {"0": 0.0, "1": 2.0})
        self.assertEqual(associative_summary["accuracy_by_target_age"], {"0": 0.0, "1": 1.0})
        self.assertEqual(associative_summary["answer_nll_by_target_age"], {"0": 2.0, "1": 0.0})
        self.assertNotIn("accuracy_by_query_namespace", associative_summary)

        rule_bank_rows = [
            {
                "mode": "correct_state",
                "correct": True,
                "donor_correct": False,
                "answer_nll": 0.5,
                "donor_answer_nll": 1.5,
                "query_namespace": 3,
            },
            {
                "mode": "correct_state",
                "correct": False,
                "donor_correct": True,
                "answer_nll": 1.5,
                "donor_answer_nll": 0.5,
                "query_namespace": 5,
            },
        ]

        rule_bank_summary = summarize_rows(rule_bank_rows)["correct_state"]

        self.assertEqual(rule_bank_summary["accuracy_by_query_namespace"], {"3": 1.0, "5": 0.0})
        self.assertEqual(rule_bank_summary["answer_nll_by_query_namespace"], {"3": 0.5, "5": 1.5})
        self.assertNotIn("accuracy_by_target_segment", rule_bank_summary)

    @parameterized.parameters(
        dict(task=MemoryTask.EXACT, num_pairs=2048),
        dict(task=MemoryTask.DOMINANT_RULE, num_pairs=2048),
        dict(task=MemoryTask.RULE_BANK, num_pairs=2048),
        dict(task=MemoryTask.ASSOCIATIVE_BANK, num_pairs=3072),
    )
    def test_canonical_num_pairs_gate_accepts_the_task_specific_count(self, task, num_pairs):
        _require_canonical_num_pairs(task.value, num_pairs)

    @parameterized.parameters(
        dict(task=MemoryTask.EXACT, num_pairs=3072),
        dict(task=MemoryTask.DOMINANT_RULE, num_pairs=3072),
        dict(task=MemoryTask.RULE_BANK, num_pairs=3072),
        dict(task=MemoryTask.ASSOCIATIVE_BANK, num_pairs=2048),
    )
    def test_canonical_num_pairs_gate_rejects_a_mismatched_count(self, task, num_pairs):
        with self.assertRaisesRegex(ValueError, "num_pairs"):
            _require_canonical_num_pairs(task.value, num_pairs)

    def test_pollution_signatures_ignore_filler_and_table_order(self):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=0,
            segment_length=16,
            num_segments=4,
            num_exact_bindings=3,
        )
        pair, _ = memory_tasks._paired_tokens(7, config)
        filler_changed = pair.copy()
        filler_changed[filler_changed >= 256] = 999
        reordered = filler_changed.copy()
        reordered[:, 0, 1:4], reordered[:, 0, 4:7] = (
            reordered[:, 0, 4:7].copy(),
            reordered[:, 0, 1:4].copy(),
        )

        for world in (0, 1):
            self.assertEqual(
                table_digest(pair[world], config), table_digest(reordered[world], config)
            )
        self.assertEqual(pair_digest(pair, config), pair_digest(filler_changed, config))

    def test_pollution_manifest_rejects_seen_tables_and_pairs(self):
        train_config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=0,
            segment_length=16,
            num_segments=2,
            num_exact_bindings=3,
        )
        eval_config = dataclasses.replace(
            train_config,
            seed=1,
            num_segments=4,
            split=MemoryTaskSplit.EVAL,
        )
        stream = MemoryTrainingStream(
            checkpoint_root=Path("/training/run"),
            checkpoint_step=300,
            contract_sha256="a" * 64,
            config=train_config,
            train_pairs=300,
        )

        manifest = build_pollution_free_manifest(
            eval_config,
            num_pairs=8,
            training_streams=(stream,),
        )
        validate_pollution_free_manifest(manifest)

        train_tables = set()
        train_pairs = set()
        for pair_index in range(stream.train_pairs):
            pair, _ = memory_tasks._paired_tokens(pair_index, train_config)
            train_pairs.add(pair_digest(pair, train_config))
            train_tables.update(table_digest(pair[world], train_config) for world in (0, 1))
        for pair_index in manifest["pair_indices"]:
            pair, _ = memory_tasks._paired_tokens(pair_index, eval_config)
            self.assertNotIn(pair_digest(pair, eval_config), train_pairs)
            for world in (0, 1):
                self.assertNotIn(table_digest(pair[world], eval_config), train_tables)
        self.assertEqual(manifest["pollution_guard"]["collision_action"], "reject")

    def test_single_binding_exact_fails_when_no_disjoint_eval_set_exists(self):
        train_config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=0,
            segment_length=16,
            num_segments=2,
        )
        eval_config = dataclasses.replace(
            train_config,
            seed=1,
            split=MemoryTaskSplit.EVAL,
        )
        stream = MemoryTrainingStream(
            checkpoint_root=Path("/training/run"),
            checkpoint_step=300,
            contract_sha256="a" * 64,
            config=train_config,
            train_pairs=10_000,
        )

        with self.assertRaisesRegex(ValueError, "Could not select 32 disjoint eval pairs"):
            build_pollution_free_manifest(
                eval_config,
                num_pairs=32,
                training_streams=(stream,),
            )

    def test_pollution_manifest_rejects_malformed_pair_indices(self):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=0,
            segment_length=16,
            num_segments=2,
            num_exact_bindings=3,
            split=MemoryTaskSplit.EVAL,
        )
        stream = MemoryTrainingStream(
            checkpoint_root=Path("/training/run"),
            checkpoint_step=1,
            contract_sha256="a" * 64,
            config=dataclasses.replace(config, split=MemoryTaskSplit.TRAIN),
            train_pairs=1,
        )
        manifest = build_pollution_free_manifest(
            config,
            num_pairs=2,
            training_streams=(stream,),
        )
        manifest["pair_indices"] = [0, 1.5]
        manifest["sha256"] = _canonical_json_sha256(
            {name: value for name, value in manifest.items() if name != "sha256"}
        )

        with self.assertRaisesRegex(ValueError, "non-negative JSON integers"):
            validate_pollution_free_manifest(manifest)

    def test_known_legacy_v1_training_content_hash_remains_reconstructable(self):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=0,
            segment_length=16,
            num_segments=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = {
                "task": {
                    name: value
                    for name, value in dataclasses.asdict(config).items()
                    if name not in ("bank_value_alphabet", "generator_version")
                },
                "source": {"model_id": "qwen3.5-smoke-dense"},
                "pairs_per_batch": 2,
                "optimizer": {"num_steps": 2, "grad_accum_steps": 1},
                "runtime": {"topology": {"tp_size": 1, "fsdp_size": 1, "dp_size": 1}},
                "revision": {
                    "content_sha256": (
                        "51089c1ecf697b43266ae81844a30b2fbb67b3de0f2eb04d7a5bb92ea0a6d018"
                    )
                },
            }
            (root / "memory_task_contract.json").write_text(json.dumps(contract))

            stream = load_training_stream(root, 2)

        self.assertEqual(stream.config.generator_version, 1)
        self.assertEqual(stream.train_pairs, 4)

    @parameterized.parameters(
        "32f307dc07996ab71d09f6a8aa660238447a817e9be31d27464a97d386478478",
        "fe8d88245437d92c8c096624f2aa112ef46b506fe41e1fc6aa17f0c611c4c03e",
        "e8b1a5abf304cc8c9163b717254fe54568e242b8d12c9ad3db26b50f4c0e8ad2",
        "c382c3dfbd0e9f4113e12320cab0761d3ed4377a961c096e7f0e7ac85b5bdd9f",
    )
    def test_known_previous_v2_training_content_hash_remains_reconstructable(self, content_sha256):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=0,
            segment_length=16,
            num_segments=2,
            generator_version=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = {
                "task": {
                    name: value
                    for name, value in dataclasses.asdict(config).items()
                    if name != "bank_value_alphabet"
                },
                "source": {"model_id": "qwen3.5-smoke-dense"},
                "pairs_per_batch": 2,
                "optimizer": {"num_steps": 2, "grad_accum_steps": 1},
                "runtime": {"topology": {"tp_size": 1, "fsdp_size": 1, "dp_size": 1}},
                "revision": {"content_sha256": content_sha256},
            }
            (root / "memory_task_contract.json").write_text(json.dumps(contract))

            stream = load_training_stream(root, 2)

        self.assertEqual(stream.config.generator_version, 2)
        self.assertEqual(stream.train_pairs, 4)

    def test_unknown_v2_training_content_hash_is_rejected(self):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=0,
            segment_length=16,
            num_segments=2,
            generator_version=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = {
                "task": dataclasses.asdict(config),
                "source": {"model_id": "qwen3.5-smoke-dense"},
                "pairs_per_batch": 2,
                "optimizer": {"num_steps": 2, "grad_accum_steps": 1},
                "runtime": {"topology": {"tp_size": 1, "fsdp_size": 1, "dp_size": 1}},
                "revision": {"content_sha256": "f" * 64},
            }
            (root / "memory_task_contract.json").write_text(json.dumps(contract))

            with self.assertRaisesRegex(ValueError, "training content hash changed"):
                load_training_stream(root, 2)

    @parameterized.parameters(MemoryTask.EXACT, MemoryTask.DOMINANT_RULE)
    def test_generator_v2_keeps_task_semantics_fixed_across_horizons(self, task):
        c2 = MemoryTaskConfig(
            task=task,
            seed=5,
            segment_length=16,
            num_segments=2,
            generator_version=2,
        )
        c32 = dataclasses.replace(c2, num_segments=32)

        for pair_index in range(8):
            pair_c2, _ = memory_tasks._paired_tokens(pair_index, c2)
            pair_c32, _ = memory_tasks._paired_tokens(pair_index, c32)
            self.assertEqual(pair_digest(pair_c2, c2), pair_digest(pair_c32, c32))
            for world in (0, 1):
                self.assertEqual(
                    table_digest(pair_c2[world], c2), table_digest(pair_c32[world], c32)
                )

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

    def test_multi_slot_pair_mutates_only_target_slot_value_between_worlds(self):
        # Canary against the primary implementation risk of the multi-slot
        # extension: independently resampling all K slots per world (instead
        # of copy-then-mutate-one-slot) would make more than one position
        # differ between worlds, breaking the paired-counterfactual causal
        # isolation the evaluator depends on.
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=0,
            segment_length=16,
            num_segments=2,
            num_exact_bindings=3,
            relevance_cue=MemoryTaskCue.MARKED,
        )
        batch = build_memory_task_batch((3,), config, flatten=False)
        tokens = batch["token_ids_BCT"]
        target_slot = int(batch[BATCH_PRETRAIN_METADATA_KEY]["target_slot_B"][0])

        segment0_world0 = tokens[0, 0]
        segment0_world1 = tokens[1, 0]
        differing_positions = np.flatnonzero(segment0_world0 != segment0_world1)

        self.assertEqual(differing_positions.tolist(), [1 + 3 * target_slot + 2])
        np.testing.assert_array_equal(tokens[0, -1, :-1], tokens[1, -1, :-1])
        self.assertNotEqual(int(tokens[0, -1, -1]), int(tokens[1, -1, -1]))
        revealed_key = int(tokens[0, -1, -3])
        self.assertEqual(revealed_key, int(segment0_world0[1 + 3 * target_slot + 1]))

        # The two counterfactual target values and every decoy value must all
        # be pairwise distinct: a decoy colliding with either target value
        # would let a model score "correct" or "donor-correct" by parroting
        # a decoy it attended to, instead of genuinely retrieving the target
        # slot -- contaminating the donor/shortcut metrics the paired-world
        # design exists to keep clean. seed=0/pair_index=3 is a known
        # collision under independent-draw sampling; it must go green once
        # the K+1 values are drawn from a single without-replacement pool.
        target_values = (
            int(segment0_world0[1 + 3 * target_slot + 2]),
            int(segment0_world1[1 + 3 * target_slot + 2]),
        )
        decoy_values = [
            int(segment0_world0[1 + 3 * slot + 2])
            for slot in range(config.num_exact_bindings)
            if slot != target_slot
        ]
        all_values = (*target_values, *decoy_values)
        self.assertEqual(
            len(set(all_values)),
            len(all_values),
            f"expected {len(all_values)} pairwise-distinct values, got {all_values}",
        )

    def test_multi_slot_target_slot_is_not_fixed_across_examples(self):
        # Guards against a regression that silently collapses the surprise
        # condition back into a marked task by always querying the same slot.
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=59,
            segment_length=16,
            num_segments=2,
            num_exact_bindings=3,
            relevance_cue=MemoryTaskCue.SURPRISE,
        )
        batch = build_memory_task_batch(tuple(range(40)), config, flatten=False)
        target_slots = batch[BATCH_PRETRAIN_METADATA_KEY]["target_slot_B"][::2]

        self.assertEqual({int(slot) for slot in target_slots}, {0, 1, 2})

    def test_marked_and_surprise_share_latent_draws_and_differ_only_by_decoy_tag(self):
        marked_config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=61,
            segment_length=16,
            num_segments=2,
            num_exact_bindings=3,
            relevance_cue=MemoryTaskCue.MARKED,
        )
        surprise_config = dataclasses.replace(marked_config, relevance_cue=MemoryTaskCue.SURPRISE)

        marked_batch = build_memory_task_batch((9,), marked_config, flatten=False)
        surprise_batch = build_memory_task_batch((9,), surprise_config, flatten=False)
        marked_tokens = marked_batch["token_ids_BCT"]
        surprise_tokens = surprise_batch["token_ids_BCT"]
        marked_slot = int(marked_batch[BATCH_PRETRAIN_METADATA_KEY]["target_slot_B"][0])
        surprise_slot = int(surprise_batch[BATCH_PRETRAIN_METADATA_KEY]["target_slot_B"][0])

        # Same seed/pair_index must yield the same target slot and the same
        # final query regardless of cue condition -- the two conditions may
        # only differ in how segment 0 tags the (already identical) slots.
        self.assertEqual(marked_slot, surprise_slot)
        np.testing.assert_array_equal(marked_tokens[:, -1], surprise_tokens[:, -1])
        for slot in range(3):
            start = 1 + 3 * slot
            self.assertEqual(
                int(marked_tokens[0, 0, start + 1]), int(surprise_tokens[0, 0, start + 1])
            )
            np.testing.assert_array_equal(
                marked_tokens[:, 0, start + 2], surprise_tokens[:, 0, start + 2]
            )
            marked_tag = int(marked_tokens[0, 0, start])
            surprise_tag = int(surprise_tokens[0, 0, start])
            if slot == marked_slot:
                self.assertEqual(marked_tag, surprise_tag)
            else:
                self.assertNotEqual(marked_tag, surprise_tag)
                self.assertEqual(surprise_tag, int(surprise_tokens[0, 0, 1 + 3 * surprise_slot]))

    def test_surprise_condition_requires_at_least_three_bindings(self):
        with self.assertRaises(ValueError):
            MemoryTaskConfig(
                task=MemoryTask.EXACT,
                seed=1,
                segment_length=16,
                num_segments=2,
                num_exact_bindings=2,
                relevance_cue=MemoryTaskCue.SURPRISE,
            )

    def test_multi_slot_bindings_and_surprise_are_scoped_to_exact_task(self):
        with self.assertRaises(ValueError):
            MemoryTaskConfig(
                task=MemoryTask.DOMINANT_RULE,
                seed=1,
                segment_length=24,
                num_segments=2,
                num_exact_bindings=2,
            )
        with self.assertRaises(ValueError):
            MemoryTaskConfig(
                task=MemoryTask.DOMINANT_RULE,
                seed=1,
                segment_length=24,
                num_segments=2,
                relevance_cue=MemoryTaskCue.SURPRISE,
            )

    def test_dominant_rule_num_rule_examples_is_bounded_by_the_rule_item_pool(self):
        # _RULE_ITEMS holds 64 pairwise-distinct items; the generator draws
        # num_rule_examples + 1 of them without replacement, so 63 is the
        # boundary-valid value and 65 must be rejected here -- at
        # MemoryTaskConfig construction -- rather than inside
        # np.random.Generator.choice during batch generation.
        MemoryTaskConfig(
            task=MemoryTask.DOMINANT_RULE,
            seed=1,
            segment_length=1 + 3 * 63,
            num_segments=2,
            num_rule_examples=63,
        )
        with self.assertRaisesRegex(ValueError, "num_rule_examples"):
            MemoryTaskConfig(
                task=MemoryTask.DOMINANT_RULE,
                seed=1,
                segment_length=1 + 3 * 65,
                num_segments=2,
                num_rule_examples=65,
            )

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

    def test_associative_bank_eval_covers_every_slot_with_shared_prefix_and_single_value_diff(
        self,
    ):
        # associative_bank is only defined for a fixed C=4/L=64 geometry: the
        # generator hardcodes segment 3 as the query segment and slots 0..23
        # across segments 0..2, so a mismatched geometry must fail at config
        # construction rather than corrupt/index-crash inside the generator.
        with self.assertRaises(ValueError):
            MemoryTaskConfig(
                task=MemoryTask.ASSOCIATIVE_BANK,
                seed=11,
                segment_length=32,
                num_segments=2,
                vocab_size=1024,
            )

        config = MemoryTaskConfig(
            task=MemoryTask.ASSOCIATIVE_BANK,
            seed=11,
            segment_length=64,
            num_segments=4,
            vocab_size=1024,
            split=MemoryTaskSplit.EVAL,
        )
        batch = build_memory_task_batch(tuple(range(24)), config, flatten=False)
        tokens = batch["token_ids_BCT"]
        metadata = batch[BATCH_PRETRAIN_METADATA_KEY]
        target_slots = metadata["target_slot_B"][::2]
        target_segments = metadata["target_segment_B"][::2]
        target_ages = metadata["target_age_B"][::2]

        # base_index=pair_index//24 groups pair_index 0..23 into a single bank
        # and query slot=pair_index%24 must therefore sweep every slot once.
        np.testing.assert_array_equal(target_slots, np.arange(24))
        np.testing.assert_array_equal(target_segments, target_slots // 8)
        np.testing.assert_array_equal(target_ages, 3 - target_segments)
        self.assertLess(int(tokens.max()), 1024)

        # Every row in the 24-group shares the same world0 binding prefix
        # (segments 0..2), since only the query slot changes across the group.
        world0_prefix = tokens[0::2, :3]
        for row in range(1, 24):
            np.testing.assert_array_equal(world0_prefix[row], world0_prefix[0])

        # world1 differs from world0 only at the queried slot's value token
        # and the final answer token -- exactly two positions per pair.
        for pair in range(24):
            diff_positions = np.flatnonzero(tokens[2 * pair] != tokens[2 * pair + 1])
            self.assertEqual(len(diff_positions), 2)

        # TRAIN draws a unique bank per pair_index, unlike EVAL's shared
        # base_index grouping.
        train_config = dataclasses.replace(config, split=MemoryTaskSplit.TRAIN)
        train_tokens = build_memory_task_batch((0, 1), train_config, flatten=False)["token_ids_BCT"]
        self.assertFalse(np.array_equal(train_tokens[0, :3], train_tokens[2, :3]))

    def test_associative_bank_default_stream_and_table_digests_are_unchanged(self):
        config = MemoryTaskConfig(
            task=MemoryTask.ASSOCIATIVE_BANK,
            seed=0,
            segment_length=64,
            num_segments=4,
            generator_version=2,
        )
        pair, _ = memory_tasks._paired_tokens(0, config)

        self.assertEqual(
            hashlib.sha256(pair.tobytes()).hexdigest(),
            "c03966cd4d3750dc250334bd93242f0ab235a1609268bdd204d3f226bd469f53",
        )
        self.assertEqual(
            pair_digest(pair, config).hex(),
            "ab59c68a20a60de779dad78ce5bfe9375415e502430c58c3aed0d71d0242c7e4",
        )
        self.assertEqual(
            table_digest(pair[0], config).hex(),
            "830076c20ee40ac1ec5d124b5858fda5bbd7086a8b54f92e92d0519e8b204fb6",
        )
        self.assertEqual(
            table_digest(pair[1], config).hex(),
            "4e2f02e8b190e818c7650d0aa684e485287c3dbaafdc4f4e30fe5143ccd48566",
        )

    @parameterized.parameters(47, 257, 48.0)
    def test_associative_bank_rejects_invalid_value_alphabet(self, bank_value_alphabet):
        with self.assertRaisesRegex(ValueError, "bank_value_alphabet"):
            MemoryTaskConfig(
                task=MemoryTask.ASSOCIATIVE_BANK,
                seed=0,
                segment_length=64,
                num_segments=4,
                bank_value_alphabet=bank_value_alphabet,
            )

    @parameterized.parameters(MemoryTask.EXACT, MemoryTask.DOMINANT_RULE, MemoryTask.RULE_BANK)
    def test_bank_value_alphabet_is_scoped_to_associative_bank(self, task):
        kwargs = {"segment_length": 16, "num_segments": 2}
        if task is MemoryTask.RULE_BANK:
            kwargs = {"segment_length": 64, "num_segments": 5}
        with self.assertRaisesRegex(ValueError, "only configurable"):
            MemoryTaskConfig(
                task=task,
                seed=0,
                bank_value_alphabet=48,
                **kwargs,
            )

    def test_associative_bank_minimum_alphabet_keeps_all_pair_answers_distinct(self):
        config = MemoryTaskConfig(
            task=MemoryTask.ASSOCIATIVE_BANK,
            seed=0,
            segment_length=64,
            num_segments=4,
            bank_value_alphabet=48,
            split=MemoryTaskSplit.EVAL,
            generator_version=2,
        )
        batch = build_memory_task_batch(tuple(range(24)), config, flatten=False)
        answers = np.asarray(batch[BATCH_PRETRAIN_METADATA_KEY]["answer_token_B"])

        self.assertLen(set(answers.tolist()), 48)
        self.assertEqual(set(answers.tolist()), set(range(96, 144)))

    def test_associative_bank_overlap_audit_uses_configured_value_alphabet(self):
        train_config = MemoryTaskConfig(
            task=MemoryTask.ASSOCIATIVE_BANK,
            seed=0,
            segment_length=64,
            num_segments=4,
            bank_value_alphabet=48,
            generator_version=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = {
                "task": dataclasses.asdict(train_config),
                "source": {"model_id": "qwen3.5-smoke-dense"},
                "pairs_per_batch": 1,
                "optimizer": {"num_steps": 1, "grad_accum_steps": 1},
                "runtime": {"topology": {"tp_size": 1, "fsdp_size": 1, "dp_size": 1}},
                "revision": {"content_sha256": memory_pollution._current_training_content_sha256()},
            }
            contract_path = root / "memory_task_contract.json"
            contract_path.write_text(json.dumps(contract))
            stream = MemoryTrainingStream(
                checkpoint_root=root,
                checkpoint_step=1,
                contract_sha256=memory_pollution.canonical_json_sha256(contract),
                config=train_config,
                train_pairs=1,
            )
            manifest = build_pollution_free_manifest(
                dataclasses.replace(train_config, seed=1, split=MemoryTaskSplit.EVAL),
                num_pairs=24,
                training_streams=(stream,),
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                audit_memory_task_overlap.main(
                    [
                        f"--training_contract={contract_path}",
                        "--checkpoint_step=1",
                        f"--eval_manifest={manifest_path}",
                    ]
                )

        audit = json.loads(stdout.getvalue())
        self.assertEqual(audit["symbol_pools"]["values"]["allowed"], 48)

    def test_rule_bank_eval_group_covers_every_namespace_with_shared_world0_evidence(self):
        # rule_bank is only defined for a fixed C=5/L=64 geometry: the
        # generator hardcodes segment 4 as the query segment and 56 demos
        # (14 per segment) spread across segments 0..3, so a mismatched
        # geometry must fail at config construction rather than
        # corrupt/index-crash inside the generator.
        with self.assertRaises(ValueError):
            MemoryTaskConfig(
                task=MemoryTask.RULE_BANK,
                seed=13,
                segment_length=64,
                num_segments=4,
                vocab_size=1024,
            )

        config = MemoryTaskConfig(
            task=MemoryTask.RULE_BANK,
            seed=13,
            segment_length=64,
            num_segments=5,
            vocab_size=1024,
            split=MemoryTaskSplit.EVAL,
        )
        batch = build_memory_task_batch(tuple(range(8)), config, flatten=False)
        tokens = batch["token_ids_BCT"]
        metadata = batch[BATCH_PRETRAIN_METADATA_KEY]

        # RULE_BANK's generic target_slot is the query namespace itself, so a
        # separate target_slot_B would just be a misleading duplicate of
        # query_namespace_B in raw rows/summaries; only query_namespace_B may
        # exist for this task.
        self.assertNotIn("target_slot_B", metadata)
        query_namespaces = metadata["query_namespace_B"][::2]

        # base_index=pair_index//8 groups pair_index 0..7 into a single bank
        # and query_namespace=pair_index%8 must therefore sweep every
        # namespace once.
        np.testing.assert_array_equal(query_namespaces, np.arange(8))
        self.assertLess(int(tokens.max()), 1024)

        # Every row in the 8-group shares the same world0 demo segments
        # (0..3), since only the queried namespace's world1 labels and the
        # final query segment differ across the group.
        world0_demo_segments = tokens[0::2, :4]
        for row in range(1, 8):
            np.testing.assert_array_equal(world0_demo_segments[row], world0_demo_segments[0])

        for pair in range(8):
            demos = tokens[2 * pair, :4, 1:57].reshape(-1, 4)
            world1_demos = tokens[2 * pair + 1, :4, 1:57].reshape(-1, 4)
            namespace_id = 352 + int(query_namespaces[pair])
            own_mask = demos[:, 1] == namespace_id

            # Only the queried namespace's labels may change between worlds;
            # every DEMO tag/namespace/item token is identical in both worlds.
            np.testing.assert_array_equal(demos[:, :3], world1_demos[:, :3])
            np.testing.assert_array_equal(demos[~own_mask], world1_demos[~own_mask])
            self.assertEqual(int(own_mask.sum()), 7)

            # Both worlds keep the symmetric 4/1/1/1 label histogram for the
            # queried namespace, with base/alt dominance swapped between them.
            _, world0_counts = np.unique(demos[own_mask, 3], return_counts=True)
            _, world1_counts = np.unique(world1_demos[own_mask, 3], return_counts=True)
            self.assertEqual(sorted(world0_counts.tolist()), [1, 1, 1, 4])
            self.assertEqual(sorted(world1_counts.tolist()), [1, 1, 1, 4])

            base_label = int(tokens[2 * pair, -1, -1])
            alt_label = int(tokens[2 * pair + 1, -1, -1])
            self.assertNotEqual(base_label, alt_label)
            np.testing.assert_array_equal(tokens[2 * pair, -1, :-1], tokens[2 * pair + 1, -1, :-1])

        # TRAIN draws a unique bank per pair_index, unlike EVAL's shared
        # base_index grouping.
        train_config = dataclasses.replace(config, split=MemoryTaskSplit.TRAIN)
        train_tokens = build_memory_task_batch((0, 1), train_config, flatten=False)["token_ids_BCT"]
        self.assertFalse(np.array_equal(train_tokens[0, :4], train_tokens[2, :4]))

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

    def test_grain_iterator_restores_the_next_pair_cursor_after_grad_accumulation(self):
        # A checkpoint is saved only after an optimizer update. With two
        # microbatches per update, the restored iterator must therefore start
        # at pair 2 rather than replaying either accumulation microbatch.
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=29,
            segment_length=16,
            num_segments=2,
        )
        with test_temp_dir() as tmp:
            iterator = make_memory_task_iterator(
                config,
                global_batch_size=4,
                num_batches=4,
                flatten=False,
                dp_size=1,
                fsdp_size=1,
                process_index=0,
            )
            _ = next(iterator)
            _ = next(iterator)
            registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
            registry.add("train_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
            registry.add(
                "train_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
            )
            checkpoint_utils.register_grain_iterator_handler(registry)
            manager = ocp.CheckpointManager(
                Path(tmp) / "iterator_checkpoint",
                options=ocp.CheckpointManagerOptions(save_interval_steps=1),
                handler_registry=registry,
            )
            manager.save(
                1,
                args=checkpoint_utils.make_grain_save_args(
                    {"step": np.asarray(1, dtype=np.int32)}, iterator
                ),
            )
            manager.wait_until_finished()

            restored = make_memory_task_iterator(
                config,
                global_batch_size=4,
                num_batches=4,
                flatten=False,
                dp_size=1,
                fsdp_size=1,
                process_index=0,
            )
            restored_state = manager.restore(
                1,
                args=checkpoint_utils.make_grain_restore_args(
                    {"step": jax.ShapeDtypeStruct((), jnp.int32)}, restored
                ),
            )
            restored = checkpoint_utils.restored_input_iter(restored_state)
            metadata = next(restored)[BATCH_PRETRAIN_METADATA_KEY]
            np.testing.assert_array_equal(metadata["pair_index_B"], [2, 2])
            manager.close()

    @parameterized.parameters(
        dict(
            condition=TrainingCondition.STATE,
            num_exact_bindings=1,
            relevance_cue=MemoryTaskCue.MARKED,
        ),
        dict(
            condition=TrainingCondition.STATE,
            num_exact_bindings=3,
            relevance_cue=MemoryTaskCue.SURPRISE,
        ),
        dict(
            condition=TrainingCondition.RESET,
            num_exact_bindings=1,
            relevance_cue=MemoryTaskCue.MARKED,
        ),
        dict(
            condition=TrainingCondition.FULL_CONTEXT,
            num_exact_bindings=3,
            relevance_cue=MemoryTaskCue.SURPRISE,
        ),
    )
    def test_real_trainer_consumes_answer_only_batch_across_conditions(
        self, condition, num_exact_bindings, relevance_cue
    ):
        # Layout (batch_size/seq_len/flatten/pretrain_mode/bptt_chunks/
        # pass_gdn_state) comes from training_layout -- the same function
        # scripts/train_memory_tasks.py uses -- rather than being
        # hand-duplicated here, so this exercises the actual Grain iterator
        # (flatten=True for FULL_CONTEXT, PretrainMode.IID_BASELINE) through
        # the real run_pretrain path.
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=31,
            segment_length=16,
            num_segments=2,
            vocab_size=512,
            num_exact_bindings=num_exact_bindings,
            relevance_cue=relevance_cue,
        )
        layout = training_layout(
            condition,
            pairs_per_batch=1,
            num_segments=config.num_segments,
            segment_length=config.segment_length,
        )
        iterator = make_memory_task_iterator(
            config,
            global_batch_size=layout.batch_size,
            num_batches=1,
            flatten=layout.flatten,
            dp_size=1,
            fsdp_size=1,
            process_index=0,
        )
        train_config = pretrain_trainer.TrainConfig(
            seed=0,
            batch_size=layout.batch_size,
            seq_len=layout.seq_len,
            num_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            print_every=0,
        )

        _, metrics = pretrain_trainer.run_pretrain(
            _tiny_gdn_config(),
            train_config,
            iterator,
            pretrain_mode=layout.pretrain_mode,
            log_every=0,
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
            text_attn_backend="xla",
            bptt_chunks=layout.bptt_chunks,
            pass_gdn_state=layout.pass_gdn_state,
        )

        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertEqual(metrics["supervised_tokens"], 2.0)

    def test_contract_accepts_legacy_implicit_model_seed_and_bank_alphabet(self):
        with test_temp_dir() as tmp:
            save_dir = Path(tmp)
            legacy_contract = {"schema_version": 2, "task": {"seed": 7}}
            (save_dir / "memory_task_contract.json").write_text(json.dumps(legacy_contract))

            train_memory_tasks._ensure_contract(
                save_dir,
                {
                    **legacy_contract,
                    "model_seed": 7,
                    "task": {"seed": 7, "bank_value_alphabet": 256},
                },
            )

            self.assertEqual(
                json.loads((save_dir / "memory_task_contract.json").read_text()),
                legacy_contract,
            )

    def test_cli_trains_one_step_with_explicit_adam_hparams_and_device_memory(self):
        # Exercises the real argparse -> Grain iterator -> Qwen3.5 ->
        # run_pretrain -> optimizer -> checkpoint path through
        # scripts.train_memory_tasks.main, rather than calling run_pretrain
        # directly, so a regression in flag wiring or contract persistence
        # would be caught here even if the lower-level trainer tests pass.
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            save_dir = tmpdir / "run"
            argv = [
                "--task=exact",
                "--condition=full_context",
                f"--save_dir={save_dir}",
                "--model_id=qwen3.5-smoke-dense",
                "--seed=7",
                "--data_seed=11",
                "--num_steps=1",
                "--pairs_per_batch=1",
                "--grad_accum_steps=2",
                "--log_every=0",
                f"--jax_cache_dir={tmpdir / 'jax_cache'}",
                "--text_attn_backend=xla",
                f"--run_plan_sha256={'a' * 64}",
                "--success_criterion=unit-test criterion",
                "--adam_beta1=0.8",
                "--adam_beta2=0.9",
                "--adam_eps=1e-6",
            ]

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                train_memory_tasks.main(argv)

            metrics = json.loads(stdout.getvalue().strip().splitlines()[-1])
            contract = json.loads((save_dir / "memory_task_contract.json").read_text())

            self.assertEqual(contract["model_seed"], 7)
            self.assertEqual(contract["task"]["seed"], 11)
            self.assertEqual(contract["optimizer"]["adam_beta1"], 0.8)
            self.assertEqual(contract["optimizer"]["adam_beta2"], 0.9)
            self.assertEqual(contract["optimizer"]["adam_eps"], 1e-6)
            self.assertEqual(
                contract["execution"],
                {"save_every": 100, "keep_latest": 3, "log_every": 0},
            )
            self.assertEqual(
                contract["run_plan"],
                {
                    "sha256": "a" * 64,
                    "success_criterion": "unit-test criterion",
                },
            )
            self.assertEqual(contract["runtime"]["text_attn_backend"], "xla")
            self.assertEqual(
                contract["runtime"]["topology"], {"tp_size": 1, "fsdp_size": 1, "dp_size": 1}
            )
            self.assertEqual(contract["runtime"]["jax_backend"], jax.default_backend())
            self.assertEqual(contract["runtime"]["jax_device_count"], jax.device_count())
            self.assertEqual(contract["runtime"]["deltanet_kernel"], "xla")
            self.assertIn("commit", contract["revision"])
            self.assertIn("dirty", contract["revision"])
            self.assertIn("content_sha256", contract["revision"])
            self.assertTrue(Path(contract["runtime"]["omegalax_file"]).is_file())
            self.assertTrue((save_dir / "000001").is_dir())
            self.assertTrue((save_dir / "config.json").exists())
            self.assertIn("device_memory", metrics)
            self.assertIsInstance(metrics["device_memory"], list)

    def test_training_cli_defaults_to_mosaic_gpu_attention(self):
        action = next(
            action
            for action in train_memory_tasks._parser()._actions
            if action.dest == "text_attn_backend"
        )
        self.assertEqual(action.default, "mosaic_gpu")

    def test_tail_cli_and_eval_run_from_a_real_tiny_checkpoint(self):
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            source_dir = tmpdir / "source"
            source_argv = [
                "--task=exact",
                "--condition=full_context",
                f"--save_dir={source_dir}",
                "--model_id=qwen3.5-smoke-dense",
                "--seed=13",
                "--num_steps=1",
                "--pairs_per_batch=1",
                "--log_every=0",
                f"--jax_cache_dir={tmpdir / 'source_jax_cache'}",
                "--text_attn_backend=xla",
                f"--run_plan_sha256={'b' * 64}",
                "--success_criterion=source checkpoint for tail CLI test",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(source_argv)
            source_contract = json.loads((source_dir / "memory_task_contract.json").read_text())
            self.assertEqual(source_contract["model_seed"], 13)
            self.assertEqual(source_contract["task"]["seed"], 13)

            tail_dir = tmpdir / "tail"
            tail_argv = [
                "--task=exact",
                "--condition=full_context",
                f"--save_dir={tail_dir}",
                f"--init_model_checkpoint={source_dir}",
                "--num_steps=1",
                "--pairs_per_batch=1",
                "--log_every=0",
                f"--jax_cache_dir={tmpdir / 'tail_jax_cache'}",
                "--text_attn_backend=xla",
                f"--run_plan_sha256={'c' * 64}",
                "--success_criterion=tail CLI initialization from real checkpoint",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(tail_argv)

            contract = json.loads((tail_dir / "memory_task_contract.json").read_text())
            self.assertEqual(contract["model_seed"], 0)
            self.assertEqual(contract["task"]["seed"], 0)
            self.assertEqual(contract["source"]["checkpoint_root"], str(source_dir.resolve()))
            self.assertEqual(contract["source"]["checkpoint_step"], 1)
            self.assertEqual(len(contract["source"]["memory_task_contract_sha256"]), 64)
            self.assertTrue((tail_dir / "000002").is_dir())
            manifest_path = tmpdir / "eval_manifest.json"
            with contextlib.redirect_stdout(io.StringIO()):
                build_memory_task_eval_manifest.main(
                    [
                        f"--checkpoint={tail_dir}",
                        f"--output={manifest_path}",
                        "--task=exact",
                        "--num_pairs=2",
                    ]
                )
            pollution_guard = json.loads(manifest_path.read_text())["pollution_guard"]
            self.assertEqual(len(pollution_guard["training_streams"]), 2)
            audit_stdout = io.StringIO()
            with contextlib.redirect_stdout(audit_stdout):
                audit_memory_task_overlap.main(
                    [
                        f"--training_contract={tail_dir / 'memory_task_contract.json'}",
                        "--checkpoint_step=2",
                        f"--eval_manifest={manifest_path}",
                    ]
                )
            audit = json.loads(audit_stdout.getvalue())
            self.assertEqual(len(audit["training_streams"]), 2)
            self.assertEqual(audit["semantic_pair_overlap"], 0)
            self.assertEqual(audit["semantic_table_overlap"], 0)
            mirror_dir = tmpdir / "wiki_mirror"
            eval_argv = [
                f"--checkpoint={tail_dir}",
                "--task=exact",
                f"--eval_manifest={manifest_path}",
                "--num_pairs=2",
                "--pairs_per_batch=1",
                "--text_attn_backend=xla",
                f"--jax_cache_dir={tmpdir / 'eval_jax_cache'}",
            ]
            incomplete_manifest = json.loads(manifest_path.read_text())
            incomplete_manifest["pollution_guard"]["training_streams"] = [
                reference
                for reference in incomplete_manifest["pollution_guard"]["training_streams"]
                if reference["checkpoint_root"] == str(tail_dir.resolve())
            ]
            incomplete_manifest["sha256"] = _canonical_json_sha256(
                {name: value for name, value in incomplete_manifest.items() if name != "sha256"}
            )
            incomplete_manifest_path = tmpdir / "incomplete_eval_manifest.json"
            incomplete_manifest_path.write_text(json.dumps(incomplete_manifest))
            with (
                mock.patch.dict(eval_memory_tasks._CANONICAL_NUM_PAIRS, {MemoryTask.EXACT: 2}),
                self.assertRaisesRegex(ValueError, "complete checkpoint ancestry"),
            ):
                eval_memory_tasks.main(
                    [
                        argument
                        if not argument.startswith("--eval_manifest=")
                        else f"--eval_manifest={incomplete_manifest_path}"
                        for argument in eval_argv
                    ]
                )
            with (
                mock.patch.dict(eval_memory_tasks._CANONICAL_NUM_PAIRS, {MemoryTask.EXACT: 2}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                eval_memory_tasks.main(eval_argv)

            eval_dir = tail_dir / "evals" / "memory_exact_pollution_guard_v1" / "step_2"
            eval_config = json.loads((eval_dir / "eval_config.json").read_text())
            self.assertIn("training_contract_sha256", eval_config)
            self.assertNotIn("eval_role", eval_config)
            self.assertEqual(eval_config["text_attn_backend"], "xla")
            self.assertEqual(
                eval_config["runtime"],
                {
                    "jax_backend": jax.default_backend(),
                    "visible_device_platforms": sorted(
                        {device.platform for device in jax.devices()}
                    ),
                    "visible_device_count": jax.device_count(),
                    "deltanet_kernel": "xla",
                },
            )
            self.assertTrue((eval_dir / "raw.jsonl").is_file())
            self.assertTrue((eval_dir / "summary.json").is_file())
            self.assertTrue((eval_dir / "artifact_manifest.json").is_file())
            self.assertTrue((eval_dir / "plots" / "accuracy.svg").is_file())
            self.assertTrue((eval_dir / "plots" / "answer_nll.svg").is_file())
            self.assertEqual(
                json.loads((eval_dir / "status.json").read_text()), {"status": "complete"}
            )
            self.assertFalse(mirror_dir.exists())

            final_manifest_path = tmpdir / "final_eval_manifest.json"
            with contextlib.redirect_stdout(io.StringIO()):
                build_memory_task_eval_manifest.main(
                    [
                        f"--checkpoint={tail_dir}",
                        f"--exclude_manifest={manifest_path}",
                        "--final_holdout",
                        f"--output={final_manifest_path}",
                        "--task=exact",
                        "--seed=2",
                        "--num_pairs=2",
                    ]
                )
            final_eval_argv = [
                f"--checkpoint={tail_dir}",
                "--task=exact",
                "--seed=2",
                f"--eval_manifest={final_manifest_path}",
                "--eval_role=final",
                "--num_pairs=2",
                "--pairs_per_batch=1",
                "--text_attn_backend=xla",
                f"--jax_cache_dir={tmpdir / 'final_eval_jax_cache'}",
                f"--wiki_mirror_dir={tmpdir / 'final_wiki_mirror'}",
            ]
            final_eval_dir = tail_dir / "evals" / "memory_exact_final_pollution_guard_v1" / "step_2"
            with (
                mock.patch.dict(eval_memory_tasks._CANONICAL_NUM_PAIRS, {MemoryTask.EXACT: 2}),
                self.assertRaisesRegex(ValueError, "requires a final-holdout"),
            ):
                eval_memory_tasks.main(
                    [
                        argument
                        if not argument.startswith("--eval_manifest=")
                        else f"--eval_manifest={manifest_path}"
                        for argument in final_eval_argv
                    ]
                )
            self.assertFalse(final_eval_dir.exists())

            gate_eval_dir = tail_dir / "evals" / "memory_exact_gate_pollution_guard_v1" / "step_2"
            with (
                mock.patch.dict(eval_memory_tasks._CANONICAL_NUM_PAIRS, {MemoryTask.EXACT: 2}),
                self.assertRaisesRegex(ValueError, "rejects a final-holdout"),
            ):
                eval_memory_tasks.main(
                    [
                        *(
                            argument
                            for argument in final_eval_argv
                            if argument != "--eval_role=final"
                        ),
                        "--eval_role=gate",
                    ]
                )
            self.assertFalse(gate_eval_dir.exists())

            gate_eval_argv = [
                argument
                if not argument.startswith("--wiki_mirror_dir=")
                else f"--wiki_mirror_dir={tmpdir / 'gate_wiki_mirror'}"
                for argument in eval_argv
            ]
            with (
                mock.patch.dict(eval_memory_tasks._CANONICAL_NUM_PAIRS, {MemoryTask.EXACT: 2}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                eval_memory_tasks.main([*gate_eval_argv, "--eval_role=gate"])
            gate_eval_config = json.loads((gate_eval_dir / "eval_config.json").read_text())
            self.assertEqual(gate_eval_config["eval_role"], "gate")

            with (
                mock.patch.dict(eval_memory_tasks._CANONICAL_NUM_PAIRS, {MemoryTask.EXACT: 2}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                eval_memory_tasks.main(final_eval_argv)

            final_eval_config = json.loads((final_eval_dir / "eval_config.json").read_text())
            self.assertEqual(final_eval_config["eval_role"], "final")
            self.assertTrue((final_eval_dir / "artifact_manifest.json").is_file())
            final_mirror_dir = tmpdir / "final_wiki_mirror"
            self.assertTrue((final_mirror_dir / "accuracy.svg").is_file())
            self.assertTrue((final_mirror_dir / "answer_nll.svg").is_file())
            self.assertTrue((final_mirror_dir / "README.md").is_file())
            self.assertFalse((final_mirror_dir / "raw.jsonl").exists())

    def test_zero_shot_horizon_override_runs_real_cli_and_preserves_immutable_outputs(self):
        # This is deliberately end-to-end: train_memory_tasks creates the
        # actual contract/checkpoint and eval_memory_tasks restores it before
        # evaluating a longer C=4 horizon.  It protects the flag wiring and
        # the generated artifact layout, not a parallel test-only evaluator.
        with test_temp_dir() as tmp:
            tmpdir = Path(tmp)
            save_dir = tmpdir / "run"
            train_argv = [
                "--task=exact",
                "--condition=state",
                f"--save_dir={save_dir}",
                "--model_id=qwen3.5-smoke-dense",
                "--segment_length=16",
                "--num_segments=2",
                "--num_steps=1",
                "--save_every=1",
                "--pairs_per_batch=1",
                "--log_every=0",
                f"--jax_cache_dir={tmpdir / 'train_jax_cache'}",
                "--text_attn_backend=xla",
                f"--run_plan_sha256={'d' * 64}",
                "--success_criterion=zero-shot horizon CLI contract test",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                train_memory_tasks.main(train_argv)

            manifest_path = tmpdir / "eval_manifest.json"
            with contextlib.redirect_stdout(io.StringIO()):
                build_memory_task_eval_manifest.main(
                    [
                        f"--checkpoint={save_dir}",
                        f"--output={manifest_path}",
                        "--task=exact",
                        "--segment_length=16",
                        "--num_segments=4",
                        "--num_pairs=2",
                    ]
                )

            base_eval_argv = [
                f"--checkpoint={save_dir}",
                "--task=exact",
                "--segment_length=16",
                "--num_segments=4",
                f"--eval_manifest={manifest_path}",
                "--num_pairs=2",
                "--pairs_per_batch=1",
                "--text_attn_backend=xla",
                f"--jax_cache_dir={tmpdir / 'eval_jax_cache'}",
                f"--wiki_mirror_dir={tmpdir / 'wiki_mirror'}",
            ]

            # C/L still fail closed unless the evaluator's explicit flag is
            # present; this must happen before a model restore or output write.
            with self.assertRaisesRegex(ValueError, "mismatched memory-task config"):
                eval_memory_tasks.main(base_eval_argv)
            with self.assertRaisesRegex(ValueError, "mismatched memory-task config"):
                eval_memory_tasks.main(
                    [*base_eval_argv, "--vocab_size=768", "--allow_zero_shot_horizon_override"]
                )

            with (
                mock.patch.dict(eval_memory_tasks._CANONICAL_NUM_PAIRS, {MemoryTask.EXACT: 2}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                eval_memory_tasks.main([*base_eval_argv, "--allow_zero_shot_horizon_override"])

            eval_dir = (
                save_dir
                / "evals"
                / "memory_exact_zero_shot_horizon_c4_l16_pollution_guard_v1"
                / "step_1"
            )
            self.assertTrue(eval_dir.is_dir())
            self.assertFalse((save_dir / "evals" / "memory_exact_pollution_guard_v1").exists())
            config = json.loads((eval_dir / "eval_config.json").read_text())
            contract = json.loads((save_dir / "memory_task_contract.json").read_text())
            self.assertEqual(config["source_training_task"], contract["task"])
            self.assertEqual(config["requested_eval_task"]["segment_length"], 16)
            self.assertEqual(config["requested_eval_task"]["num_segments"], 4)
            self.assertTrue(config["zero_shot_horizon_override"])
            self.assertEqual(
                config["allowed_training_task_differences"],
                ["num_segments", "seed", "segment_length", "split"],
            )

            rows = [json.loads(line) for line in (eval_dir / "raw.jsonl").read_text().splitlines()]
            self.assertEqual({row["mode"] for row in rows}, {mode.value for mode in MemoryEvalMode})
            self.assertTrue((eval_dir / "artifact_manifest.json").is_file())
            self.assertTrue((eval_dir / "plots" / "accuracy.svg").is_file())
            self.assertTrue((eval_dir / "plots" / "answer_nll.svg").is_file())
            self.assertTrue((tmpdir / "wiki_mirror" / "accuracy.svg").is_file())
            self.assertTrue((tmpdir / "wiki_mirror" / "answer_nll.svg").is_file())
            self.assertTrue((tmpdir / "wiki_mirror" / "README.md").is_file())

            # Same source/requested contract resolves to the same isolated
            # horizon path and is immutable once its artifacts are complete.
            with (
                mock.patch.dict(eval_memory_tasks._CANONICAL_NUM_PAIRS, {MemoryTask.EXACT: 2}),
                self.assertRaisesRegex(ValueError, "already complete"),
            ):
                eval_memory_tasks.main([*base_eval_argv, "--allow_zero_shot_horizon_override"])

    @parameterized.parameters(
        dict(num_exact_bindings=1, relevance_cue=MemoryTaskCue.MARKED),
        dict(num_exact_bindings=3, relevance_cue=MemoryTaskCue.SURPRISE),
    )
    def test_real_model_evaluator_runs_all_causal_conditions(
        self, num_exact_bindings, relevance_cue
    ):
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=37,
            segment_length=16,
            num_segments=2,
            vocab_size=512,
            num_exact_bindings=num_exact_bindings,
            relevance_cue=relevance_cue,
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

    @parameterized.named_parameters(
        ("c2_marked_k1", 8, 2, 1, MemoryTaskCue.MARKED, 1e-6),
        ("c4_surprise_k3", 16, 4, 3, MemoryTaskCue.SURPRISE, 2e-5),
    )
    def test_jitted_evaluator_matches_eager_qwen_path_for_every_mode(
        self,
        segment_length,
        num_segments,
        num_exact_bindings,
        relevance_cue,
        nll_tolerance,
    ):
        """The reusable evaluator must preserve the established five-mode semantics."""
        config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=71,
            segment_length=segment_length,
            num_segments=num_segments,
            vocab_size=512,
            num_exact_bindings=num_exact_bindings,
            relevance_cue=relevance_cue,
        )
        model, model_config = text_api.init_model(
            _tiny_gdn_config(),
            jax.random.key(0),
            tp_size=1,
            fsdp_size=1,
            dp_size=1,
        )
        evaluator = memory_evals.make_memory_evaluator(model_config)

        batch = build_memory_task_batch((0,), config, flatten=False)
        eager_results = memory_evals._evaluate_memory_batch_eager(model, model_config, batch)
        jitted_results = evaluator(model, batch)

        self.assertEqual(set(jitted_results), set(MemoryEvalMode))
        for mode in MemoryEvalMode:
            eager = eager_results[mode]
            jitted = jitted_results[mode]
            self.assertEqual(jitted.accuracy, eager.accuracy)
            self.assertEqual(jitted.donor_accuracy, eager.donor_accuracy)
            np.testing.assert_allclose(
                jitted.answer_nll, eager.answer_nll, rtol=nll_tolerance, atol=nll_tolerance
            )
            np.testing.assert_allclose(
                jitted.donor_answer_nll,
                eager.donor_answer_nll,
                rtol=nll_tolerance,
                atol=nll_tolerance,
            )
            np.testing.assert_array_equal(jitted.predictions_B, eager.predictions_B)
            np.testing.assert_allclose(
                jitted.answer_nll_B,
                eager.answer_nll_B,
                rtol=nll_tolerance,
                atol=nll_tolerance,
            )
            np.testing.assert_allclose(
                jitted.donor_answer_nll_B,
                eager.donor_answer_nll_B,
                rtol=nll_tolerance,
                atol=nll_tolerance,
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

    @parameterized.parameters(
        dict(num_exact_bindings=1, relevance_cue=MemoryTaskCue.MARKED),
        dict(num_exact_bindings=3, relevance_cue=MemoryTaskCue.SURPRISE),
    )
    def test_train_and_eval_splits_generate_disjoint_streams_even_with_same_seed(
        self, num_exact_bindings, relevance_cue
    ):
        train_config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=5,
            segment_length=16,
            num_segments=2,
            split=MemoryTaskSplit.TRAIN,
            num_exact_bindings=num_exact_bindings,
            relevance_cue=relevance_cue,
        )
        eval_config = dataclasses.replace(train_config, split=MemoryTaskSplit.EVAL)

        train_batch = build_memory_task_batch((3,), train_config, flatten=False)
        eval_batch = build_memory_task_batch((3,), eval_config, flatten=False)

        self.assertFalse(np.array_equal(train_batch["token_ids_BCT"], eval_batch["token_ids_BCT"]))

    def test_eval_contract_validation_rejects_task_or_split_mismatch(self):
        base_config = MemoryTaskConfig(
            task=MemoryTask.EXACT,
            seed=0,
            segment_length=16,
            num_segments=2,
            split=MemoryTaskSplit.TRAIN,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memory_task_contract.json").write_text(
                json.dumps({"task": dataclasses.asdict(base_config)})
            )

            self.assertEqual(
                _validate_training_contract(root, base_config),
                _canonical_json_sha256({"task": dataclasses.asdict(base_config)}),
            )
            # seed and split are intentionally allowed to differ between the
            # trained contract and the requested eval config.
            _validate_training_contract(
                root, dataclasses.replace(base_config, seed=99, split=MemoryTaskSplit.EVAL)
            )

            # Changing the horizon is deliberately rejected by default: a
            # zero-shot horizon evaluation must be an explicit choice, never
            # a silent relaxation of the task contract.
            with self.assertRaisesRegex(ValueError, "mismatched memory-task config"):
                _validate_training_contract(
                    root,
                    dataclasses.replace(base_config, num_segments=4),
                )

            # The one explicit override admits only the two geometry fields.
            _validate_training_contract(
                root,
                dataclasses.replace(
                    base_config,
                    segment_length=24,
                    num_segments=4,
                    split=MemoryTaskSplit.EVAL,
                ),
                allow_zero_shot_horizon_override=True,
            )
            with self.assertRaisesRegex(ValueError, "requires segment_length and/or num_segments"):
                _validate_training_contract(
                    root,
                    dataclasses.replace(base_config, split=MemoryTaskSplit.EVAL),
                    allow_zero_shot_horizon_override=True,
                )

            with self.assertRaises(ValueError):
                _validate_training_contract(
                    root, dataclasses.replace(base_config, task=MemoryTask.DOMINANT_RULE)
                )
            with self.assertRaisesRegex(ValueError, "mismatched memory-task config"):
                _validate_training_contract(
                    root,
                    dataclasses.replace(base_config, vocab_size=768),
                    allow_zero_shot_horizon_override=True,
                )
            with self.assertRaises(ValueError):
                _validate_training_contract(
                    root,
                    dataclasses.replace(
                        base_config,
                        num_exact_bindings=3,
                        relevance_cue=MemoryTaskCue.SURPRISE,
                    ),
                )

            (root / "memory_task_contract.json").write_text(
                json.dumps(
                    {
                        "task": dataclasses.asdict(
                            dataclasses.replace(base_config, split=MemoryTaskSplit.EVAL)
                        )
                    }
                )
            )
            with self.assertRaises(ValueError):
                _validate_training_contract(root, base_config)

    def test_eval_contract_treats_legacy_bank_alphabet_as_256_only(self):
        train_config = MemoryTaskConfig(
            task=MemoryTask.ASSOCIATIVE_BANK,
            seed=0,
            segment_length=64,
            num_segments=4,
            split=MemoryTaskSplit.TRAIN,
            generator_version=2,
        )
        legacy_task = dataclasses.asdict(train_config)
        del legacy_task["bank_value_alphabet"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contract = {"task": legacy_task}
            (root / "memory_task_contract.json").write_text(json.dumps(contract))

            _validate_training_contract(
                root,
                dataclasses.replace(train_config, split=MemoryTaskSplit.EVAL),
            )
            with self.assertRaisesRegex(ValueError, "mismatched memory-task config"):
                _validate_training_contract(
                    root,
                    dataclasses.replace(
                        train_config,
                        split=MemoryTaskSplit.EVAL,
                        bank_value_alphabet=48,
                    ),
                )

    def test_eval_contract_validation_rejects_missing_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "memory_task_contract.json"):
                _validate_training_contract(
                    Path(tmp),
                    MemoryTaskConfig(
                        task=MemoryTask.EXACT,
                        seed=0,
                        segment_length=16,
                        num_segments=2,
                    ),
                )

    def test_eval_refuses_to_overwrite_a_completed_identical_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            manifest = {"sha256": "manifest"}
            config = {"manifest_sha256": "manifest"}
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            (run_dir / "eval_config.json").write_text(json.dumps(config))
            (run_dir / "status.json").write_text(json.dumps({"status": "complete"}))
            (run_dir / "raw.jsonl").write_text("{}\n")
            (run_dir / "summary.json").write_text("{}\n")
            plots = run_dir / "plots"
            plots.mkdir()
            (plots / "accuracy.svg").write_text("<svg />\n")
            (plots / "answer_nll.svg").write_text("<svg />\n")
            _write_artifact_manifest(run_dir)

            with self.assertRaisesRegex(ValueError, "already complete"):
                _ensure_not_complete(run_dir, manifest, config)
            (run_dir / "raw.jsonl").write_text('{"corrupt": true}\n')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _ensure_not_complete(run_dir, manifest, config)

    def test_eval_svg_output_is_deterministic_without_date_metadata(self):
        summary = {
            mode.value: {
                "accuracy": 0.5,
                "donor_accuracy": 0.5,
                "answer_nll": 1.0,
                "donor_answer_nll": 1.0,
            }
            for mode in MemoryEvalMode
        }
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            _plot(summary, first)
            _plot(summary, second)
            for name in ("accuracy.svg", "answer_nll.svg"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                self.assertNotIn(b"<dc:date>", (first / name).read_bytes())

    def test_wiki_mirror_refuses_to_replace_different_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "accuracy.svg"
            _write_mirror_file(path, b"first")
            _write_mirror_file(path, b"first")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                _write_mirror_file(path, b"second")

    def test_eval_rejects_unsupported_multi_device_topology_before_restore(self):
        with self.assertRaisesRegex(ValueError, "only one device"):
            eval_memory_tasks.main(
                [
                    "--checkpoint=/unused",
                    "--task=exact",
                    "--eval_manifest=/unused",
                    "--num_pairs=2",
                    "--pairs_per_batch=2",
                    "--wiki_mirror_dir=/unused",
                    "--tp_size=2",
                    "--jax_cache_dir=/unused",
                ]
            )


if __name__ == "__main__":
    absltest.main()
