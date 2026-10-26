"""Tests the production grouped teacher KL loss against independent float64 math.

All cases use the real 248320-wide vocabulary with few hidden/target rows. The sharded
case runs in a fresh four-device CPU process with a real tiny Qwen3.5 model. The
boundary-window NLL diagnostic uses the real 8192/256 windows on chains of 33280 tokens.
"""

import os
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import dataclasses
import functools
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import omegalax
from omegalax.data import rollout_teacher
from omegalax.trainers import loss as loss_lib
from omegalax.trainers.loss import chunked_teacher_kl_stats

REPO = Path(__file__).resolve().parents[1]
assert Path(omegalax.__file__).resolve().is_relative_to(REPO), omegalax.__file__

V = 248_320
K = 512
SPECIAL_IDS = np.arange(248_044, 248_077)
_CHILD_ENV = "ROLLOUT_TEACHER_LOSS_SHARDED_CHILD"


def _lse(x):
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def _teacher_row(logits_V):
    """Stored target row as defined by the dataset: actual Top512, all specials, logZs."""
    ids = np.lexsort((np.arange(V), -logits_V))[:K]
    tail = np.ones(V, dtype=bool)
    tail[ids] = False
    tail[SPECIAL_IDS] = False
    return {
        "teacher_top512_ids_BRK": ids.astype(np.int32),
        "teacher_top512_logits_BRK": logits_V[ids].astype(np.float32),
        "teacher_special_logits_BRS": logits_V[SPECIAL_IDS].astype(np.float32),
        "teacher_logz_all_BR": np.float32(_lse(logits_V)),
        "teacher_logz_tail_BR": np.float32(_lse(logits_V[tail])),
    }


def _targets(rows, positions_BR, mask_BR):
    """Stack per-row dicts (None = zero-padded masked row) into the batch contract."""
    zero = {
        "teacher_top512_ids_BRK": np.zeros(K, np.int32),
        "teacher_top512_logits_BRK": np.zeros(K, np.float32),
        "teacher_special_logits_BRS": np.zeros(len(SPECIAL_IDS), np.float32),
        "teacher_logz_all_BR": np.float32(0.0),
        "teacher_logz_tail_BR": np.float32(0.0),
    }
    out = {
        name: np.stack([np.stack([(row or zero)[name] for row in batch]) for batch in rows])
        for name in zero
    }
    out["teacher_positions_BR"] = np.asarray(positions_BR, np.int32)
    out["teacher_mask_BR"] = np.asarray(mask_BR, bool)
    return out


def _reference(hidden_BTD, kernel_DV, targets):
    """Independent float64 grouped KL and its analytic gradients, one valid row at a time.

    Classes are deduplicated as a set union, the student tail is summed directly over the
    complement, and dKL/dz_v = q_v * sum_g p_g - p_g(v) * q_v / q_g(v).
    """
    h = np.asarray(hidden_BTD, np.float64)
    w = np.asarray(kernel_DV, np.float64)
    t = {name: np.asarray(value) for name, value in targets.items()}
    kl, grad_h, grad_w = 0.0, np.zeros_like(h), np.zeros_like(w)
    for b, r in zip(*np.nonzero(t["teacher_mask_BR"])):
        pos = t["teacher_positions_BR"][b, r]
        z = h[b, pos] @ w
        log_q = z - _lse(z)
        q = np.exp(log_q)
        teacher_logit = dict(zip(SPECIAL_IDS.tolist(), t["teacher_special_logits_BRS"][b, r]))
        teacher_logit.update(
            zip(t["teacher_top512_ids_BRK"][b, r].tolist(), t["teacher_top512_logits_BRK"][b, r])
        )
        classes = np.fromiter(teacher_logit, dtype=np.int64)
        logz_all = float(t["teacher_logz_all_BR"][b, r])
        log_p = np.array([teacher_logit[c] for c in classes], np.float64) - logz_all
        log_p_tail = float(t["teacher_logz_tail_BR"][b, r]) - logz_all
        tail = np.ones(V, dtype=bool)
        tail[classes] = False
        q_tail = np.sum(q[tail])
        p, p_tail = np.exp(log_p), np.exp(log_p_tail)
        kl += np.sum(p * (log_p - log_q[classes])) + p_tail * (log_p_tail - np.log(q_tail))
        dz = q * (np.sum(p) + p_tail)
        dz[classes] -= p
        dz[tail] -= p_tail * q[tail] / q_tail
        grad_h[b, pos] += w @ dz
        grad_w += np.outer(h[b, pos], dz)
    return kl, int(t["teacher_mask_BR"].sum()), grad_h, grad_w


@functools.cache
def _kl_and_grad(block_size, logits_out_sharding=None):
    def kl_fn(hidden_BTD, kernel_DV, targets):
        return chunked_teacher_kl_stats(
            hidden_BTD,
            kernel_DV,
            targets,
            block_size=block_size,
            logits_out_sharding=logits_out_sharding,
        )

    return jax.jit(jax.value_and_grad(kl_fn, argnums=(0, 1), has_aux=True))


def _run(block_size, hidden, kernel, targets, logits_out_sharding=None):
    (kl, count), (grad_h, grad_w) = _kl_and_grad(block_size, logits_out_sharding)(
        hidden, kernel, targets
    )
    return float(kl), float(count), np.asarray(grad_h), np.asarray(grad_w)


def _assert_close(actual, desired, tol):
    np.testing.assert_allclose(actual, desired, rtol=tol, atol=tol * np.max(np.abs(desired)))


def _base_case():
    """Two batch elements, C*T = 12 hidden rows, R = 6 target rows with masks and edge cases.

    b0: plain, two specials inside Top512 (one is top-1), padding, tiny teacher tail,
        tiny student tail, padding.
    b1: plain, underflowing teacher tail, hidden position 0, padding, all 33 specials inside
        Top512, last hidden position.
    """
    rng = np.random.default_rng(0)
    B, L, D = 2, 12, 8
    hidden = rng.normal(size=(B, L, D)).astype(np.float32)
    hidden[:, :, 0] = 0.0
    kernel = (0.3 * rng.normal(size=(D, V))).astype(np.float32)
    kernel[0] = 0.0

    def plain():
        return 3.0 * rng.normal(size=V)

    def concentrated(top_shift, special_shift):
        logits = 0.1 * rng.normal(size=V)
        top = rng.choice(np.setdiff1d(np.arange(V), SPECIAL_IDS), K, replace=False)
        logits[top] += top_shift + rng.normal(size=K)
        logits[SPECIAL_IDS] += special_shift + rng.normal(size=len(SPECIAL_IDS))
        return logits

    overlap = plain()
    overlap[248_046], overlap[248_044] = 25.0, 20.0
    all_specials = plain()
    all_specials[SPECIAL_IDS] += 15.0
    student_tiny_tail = _teacher_row(plain())
    # The student puts ~exp(80) on this row's stored classes, so its tail mass is ~1e-32.
    student_classes = np.union1d(student_tiny_tail["teacher_top512_ids_BRK"], SPECIAL_IDS)
    kernel[0, student_classes] = 8.0
    hidden[0, 5, 0] = 10.0
    rows = [
        [
            _teacher_row(plain()),
            _teacher_row(overlap),
            None,
            _teacher_row(concentrated(60.0, 40.0)),
            student_tiny_tail,
            None,
        ],
        [
            _teacher_row(plain()),
            _teacher_row(concentrated(150.0, 130.0)),
            _teacher_row(plain()),
            None,
            _teacher_row(all_specials),
            _teacher_row(plain()),
        ],
    ]
    positions = [[3, 7, 0, 1, 5, 0], [2, 9, 0, 0, 4, 11]]
    mask = [[row is not None for row in batch] for batch in rows]
    return hidden, kernel, _targets(rows, positions, mask)


STRIDE, WINDOW = 8192, 256
BOUNDARY_KEYS = (
    "teacher_boundary_nll_sum",
    "teacher_boundary_nll_tokens",
    "teacher_boundary_windows",
    "teacher_boundary_rollouts",
)


def _window(b, lo=0, hi=WINDOW):
    return list(range(b + lo, b + hi))


def _boundary_case():
    """Four chains of 65 * 512 positions with target rows j at the real 8192/256 windows.

    b0: complete windows at 8192 and 16384, plus targets 8191, 8448, 16383, 16640, so a
        window shifted by one in either direction would look complete as well.
    b1: near misses only: 8192 lacks 8209 (its row exists but is masked), 16384 holds
        [16385, 16641) and 24576 holds [24575, 24831).
    b2: complete window at 24576; 8192 lacks 8447, 16384 has its row 16500 masked, and the
        window at 32768 is cut by the chain end after 128 targets.
    b3: padding; its masked rows would form a complete window at 8192.
    Rows are in position order with masked rows behind the valid ones. The hidden rows that
    predict the targets right before and after each complete window are scaled up, so a
    window or predictor slipped by one position changes the NLL sum by far more than noise.
    """
    rng = np.random.default_rng(2)
    B, L, D, R = 4, 65 * 512, 8, 1024
    chains = [  # (valid target positions, masked target positions)
        (
            list(range(100, 140))
            + [8191]
            + _window(8192)
            + [8448, 16383]
            + _window(16384)
            + [16640, 24575, 24832],
            [],
        ),
        (
            [8191]
            + [j for j in _window(8192) if j != 8209]
            + [8448]
            + _window(16384, 1, 257)
            + _window(24576, -1, 255),
            [8209],
        ),
        (
            [j for j in _window(8192) if j != 8447]
            + [j for j in _window(16384) if j != 16500]
            + _window(24576)
            + _window(32768, 0, 128),
            [16500],
        ),
        ([], _window(8192)),
    ]
    positions = np.zeros((B, R), np.int32)
    mask = np.zeros((B, R), bool)
    for b, (valid, masked) in enumerate(chains):
        rows = np.asarray(valid + masked, np.int32)
        positions[b, : len(rows)] = rows - 1  # hidden position j - 1 predicts target j
        mask[b, : len(valid)] = True
    base = rng.integers(0, V, size=(B, R, 1))
    targets = {
        "teacher_positions_BR": positions,
        "teacher_mask_BR": mask,
        "teacher_top512_ids_BRK": ((base + 481 * np.arange(K)) % V).astype(np.int32),
        "teacher_top512_logits_BRK": -np.sort(rng.exponential(size=(B, R, K)), axis=-1).astype(
            np.float32
        ),
        "teacher_special_logits_BRS": (rng.normal(size=(B, R, len(SPECIAL_IDS))) - 3.0).astype(
            np.float32
        ),
        "teacher_logz_all_BR": (8.0 + rng.random((B, R))).astype(np.float32),
    }
    targets["teacher_logz_tail_BR"] = targets["teacher_logz_all_BR"] - np.float32(1.0)
    hidden = rng.normal(size=(B, L, D)).astype(np.float32)
    for b, boundary in ((0, 8192), (0, 16384), (2, 24576)):
        hidden[b, [boundary - 2, boundary + WINDOW - 1]] *= 30.0
    kernel = (0.3 * rng.normal(size=(D, V))).astype(np.float32)
    token_ids = rng.integers(0, V, size=(B, L)).astype(np.int32)
    return hidden, kernel, targets, token_ids


def _reference_windows(targets, length):
    """Eligible (b, boundary) pairs from the plain set definition of a complete window."""
    windows = []
    for b in range(targets["teacher_mask_BR"].shape[0]):
        valid = set(
            (targets["teacher_positions_BR"][b][targets["teacher_mask_BR"][b]] + 1).tolist()
        )
        for boundary in range(STRIDE, length - WINDOW + 1, STRIDE):
            if valid.issuperset(range(boundary, boundary + WINDOW)):
                windows.append((b, boundary))
    return windows


def _reference_nll(hidden, kernel, token_ids, windows, hidden_shift=-1, window_shift=0):
    """Float64 NLL sum of token j from hidden j + hidden_shift over the given windows."""
    h, w = np.asarray(hidden, np.float64), np.asarray(kernel, np.float64)
    total = 0.0
    for b, boundary in windows:
        j = np.arange(boundary, boundary + WINDOW) + window_shift
        z = h[b, j + hidden_shift] @ w
        m = np.max(z, axis=-1)
        lse = m + np.log(np.sum(np.exp(z - m[:, None]), axis=-1))
        total += float(np.sum(lse - z[np.arange(WINDOW), token_ids[b, j]]))
    return total


@functools.cache
def _boundary_fn(block_size, logits_out_sharding=None):
    def fn(hidden_BTD, kernel_DV, targets, token_ids_BT):
        return chunked_teacher_kl_stats(
            hidden_BTD,
            kernel_DV,
            targets,
            block_size=block_size,
            logits_out_sharding=logits_out_sharding,
            boundary_token_ids_BT=token_ids_BT,
        )

    return jax.jit(fn)


def _boundary_stats(block_size, hidden, kernel, targets, token_ids, logits_out_sharding=None):
    kl, count, stats = _boundary_fn(block_size, logits_out_sharding)(
        hidden, kernel, targets, token_ids
    )
    return float(kl), float(count), {key: float(stats[key]) for key in BOUNDARY_KEYS}


@functools.cache
def _boundary_kl_and_grad(block_size):
    def kl_fn(hidden_BTD, kernel_DV, targets, token_ids_BT):
        kl, count, stats = chunked_teacher_kl_stats(
            hidden_BTD,
            kernel_DV,
            targets,
            block_size=block_size,
            boundary_token_ids_BT=token_ids_BT,
        )
        return kl, (count, stats)

    return jax.jit(jax.value_and_grad(kl_fn, argnums=(0, 1), has_aux=True))


@functools.cache
def _boundary_nll_grad(block_size):
    def nll_fn(hidden_BTD, kernel_DV, targets, token_ids_BT):
        _, _, stats = chunked_teacher_kl_stats(
            hidden_BTD,
            kernel_DV,
            targets,
            block_size=block_size,
            boundary_token_ids_BT=token_ids_BT,
        )
        return stats["teacher_boundary_nll_sum"]

    return jax.jit(jax.grad(nll_fn, argnums=(0, 1)))


class TeacherKLTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hidden, cls.kernel, cls.targets = _base_case()
        cls.reference = _reference(cls.hidden, cls.kernel, cls.targets)

    def test_case_covers_overlap_and_tiny_tails(self):
        t = self.targets
        overlaps = [
            int(np.isin(t["teacher_top512_ids_BRK"][b, r], SPECIAL_IDS).sum())
            for b, r in ((0, 1), (1, 4))
        ]
        self.assertEqual(overlaps, [2, 33])
        log_p_tail = t["teacher_logz_tail_BR"] - t["teacher_logz_all_BR"]
        self.assertLess(log_p_tail[0, 3], -50.0)
        self.assertLess(log_p_tail[1, 1], -110.0)
        self.assertEqual(float(np.exp(log_p_tail[1, 1])), 0.0)
        z = self.hidden[0, 5].astype(np.float64) @ self.kernel.astype(np.float64)
        tail = np.ones(V, dtype=bool)
        tail[t["teacher_top512_ids_BRK"][0, 4]] = False
        tail[SPECIAL_IDS] = False
        # Far below float32 epsilon: 1 - sum(q_classes) cannot resolve this student tail.
        self.assertLess(_lse(z[tail]) - _lse(z), -70.0)

    def test_matches_independent_float64_reference_and_gradients(self):
        kl_ref, count_ref, grad_h_ref, grad_w_ref = self.reference
        kl, count, grad_h, grad_w = _run(4, self.hidden, self.kernel, self.targets)
        self.assertEqual(count, count_ref)
        self.assertTrue(np.isfinite(kl) and np.isfinite(grad_h).all() and np.isfinite(grad_w).all())
        np.testing.assert_allclose(kl, kl_ref, rtol=1e-5)
        _assert_close(grad_h, grad_h_ref, 1e-4)
        _assert_close(grad_w, grad_w_ref, 1e-4)

    def test_equals_dataset_reference_definition_on_full_student_logits(self):
        t = self.targets
        b, r = np.nonzero(t["teacher_mask_BR"])
        hidden_rows = self.hidden[b, t["teacher_positions_BR"][b, r]].astype(np.float64)
        student_logits = (hidden_rows @ self.kernel.astype(np.float64)).astype(np.float32)
        rows = {
            "top512_ids": t["teacher_top512_ids_BRK"][b, r],
            "top512_logits": t["teacher_top512_logits_BRK"][b, r],
            "special_logits": t["teacher_special_logits_BRS"][b, r],
            "logz_all": t["teacher_logz_all_BR"][b, r],
            "logz_tail": t["teacher_logz_tail_BR"][b, r],
        }
        dataset_kl = float(np.sum(rollout_teacher.top512_special_tail_kl(student_logits, rows)))
        kl, count, _, _ = _run(4, self.hidden, self.kernel, self.targets)
        self.assertEqual(count, len(b))
        np.testing.assert_allclose(kl, dataset_kl, rtol=1e-5)

    def test_block_sizes_are_equivalent(self):
        kl_ref, count_ref, grad_h_ref, grad_w_ref = self.reference
        results = {
            block: _run(block, self.hidden, self.kernel, self.targets)
            for block in (1, 2, 4, 5, 6, 7, 256)
        }
        kl_6, _, grad_h_6, grad_w_6 = results[6]
        for block, (kl, count, grad_h, grad_w) in results.items():
            with self.subTest(block_size=block):
                self.assertEqual(count, count_ref)
                np.testing.assert_allclose(kl, kl_ref, rtol=1e-5)
                np.testing.assert_allclose(kl, kl_6, rtol=1e-6)
                _assert_close(grad_h, grad_h_6, 1e-6)
                _assert_close(grad_w, grad_w_6, 1e-6)
                _assert_close(grad_w, grad_w_ref, 1e-4)

    def test_masked_rows_are_neutral_even_with_garbage(self):
        clean = _run(4, self.hidden, self.kernel, self.targets)
        garbage = {name: value.copy() for name, value in self.targets.items()}
        masked = ~garbage["teacher_mask_BR"]
        garbage["teacher_positions_BR"][masked] = [10**6, -3, 12]
        garbage["teacher_top512_ids_BRK"][masked] = V + 5
        garbage["teacher_top512_ids_BRK"][0, 2, :7] = -7
        garbage["teacher_top512_logits_BRK"][masked] = np.nan
        garbage["teacher_special_logits_BRS"][masked] = 1e30
        garbage["teacher_logz_all_BR"][masked] = -1e30
        garbage["teacher_logz_tail_BR"][masked] = np.inf
        dirty = _run(4, self.hidden, self.kernel, garbage)
        for name, a, b in zip(("kl", "count", "grad_h", "grad_w"), clean, dirty):
            with self.subTest(output=name):
                np.testing.assert_array_equal(a, b)

    def test_all_masked_batch_has_zero_loss_and_zero_finite_gradients(self):
        targets = {name: value.copy() for name, value in self.targets.items()}
        targets["teacher_mask_BR"][:] = False
        kl, count, grad_h, grad_w = _run(4, self.hidden, self.kernel, targets)
        self.assertEqual((kl, count), (0.0, 0.0))
        self.assertTrue(np.all(grad_h == 0.0) and np.all(grad_w == 0.0))

    def test_kl_is_near_zero_when_teacher_equals_student(self):
        # Teacher rows built from the student's own float64 logits: grouped KL is ~0.
        z = self.hidden.astype(np.float64) @ self.kernel.astype(np.float64)
        rows = [[_teacher_row(z[b, p]) for p in (3, 7, 1)] for b in range(2)]
        targets = _targets(rows, [[3, 7, 1], [3, 7, 1]], np.ones((2, 3), bool))
        kl, count, _, _ = _run(2, self.hidden, self.kernel, targets)
        self.assertEqual(count, 6.0)
        self.assertLess(abs(kl), 1e-5)

    def test_runs_are_bitwise_deterministic(self):
        first = _run(4, self.hidden, self.kernel, self.targets)
        second = _run(4, self.hidden, self.kernel, self.targets)
        _kl_and_grad.cache_clear()
        recompiled = _run(4, self.hidden, self.kernel, self.targets)
        for a, b, c in zip(first, second, recompiled):
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a, c)

    def test_compiled_temp_memory_grows_with_block_not_target_rows(self):
        def temp_bytes(R, block):
            sds = jax.ShapeDtypeStruct
            targets = {
                "teacher_positions_BR": sds((1, R), jnp.int32),
                "teacher_mask_BR": sds((1, R), jnp.bool_),
                "teacher_top512_ids_BRK": sds((1, R, K), jnp.int32),
                "teacher_top512_logits_BRK": sds((1, R, K), jnp.float32),
                "teacher_special_logits_BRS": sds((1, R, len(SPECIAL_IDS)), jnp.float32),
                "teacher_logz_all_BR": sds((1, R), jnp.float32),
                "teacher_logz_tail_BR": sds((1, R), jnp.float32),
            }
            args = (sds((1, 256, 16), jnp.bfloat16), sds((16, V), jnp.float32), targets)
            compiled = _kl_and_grad(block).lower(*args).compile()
            return compiled.memory_analysis().temp_size_in_bytes

        one_block_logits = 8 * V * 4
        small, large = temp_bytes(64, 8), temp_bytes(256, 8)
        self.assertLess(large - small, one_block_logits / 4)
        self.assertLess(large, 256 * V * 4 / 4)
        self.assertGreater(temp_bytes(256, 32) - large, 2 * one_block_logits)

    def test_rejects_wrong_vocabulary_width_before_gathering(self):
        for width in (1024, V - 1, V + 1):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "248320"):
                chunked_teacher_kl_stats(
                    jnp.asarray(self.hidden),
                    jnp.zeros((self.hidden.shape[-1], width), jnp.float32),
                    {name: jnp.asarray(value) for name, value in self.targets.items()},
                )

    def test_teacher_binding_matches_reader(self):
        self.assertEqual(loss_lib._TEACHER_VOCAB_SIZE, rollout_teacher.VOCAB_SIZE)
        self.assertEqual(loss_lib._TEACHER_TOP_K, rollout_teacher.TOP_K)
        self.assertEqual(loss_lib._TEACHER_SPECIAL_IDS, rollout_teacher.SPECIAL_IDS)

    def test_production_sharding_matches_reference_in_four_device_process(self):
        env = dict(os.environ, JAX_PLATFORMS="cpu", OMEGALAX_DELTANET_KERNEL="xla")
        env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        env["PYTHONPATH"] = str(REPO)
        env[_CHILD_ENV] = "1"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-8000:])
        reports = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual([r["mesh"] for r in reports], [[1, 1, 4], [2, 2, 1]])
        for report in reports:
            self.assertEqual(report["boundary_windows"], 3.0)


class TeacherBoundaryNLLTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hidden, cls.kernel, cls.targets, cls.token_ids = _boundary_case()
        cls.windows = _reference_windows(cls.targets, cls.hidden.shape[1])
        cls.nll_ref = _reference_nll(cls.hidden, cls.kernel, cls.token_ids, cls.windows)
        cls.stats = _boundary_stats(256, cls.hidden, cls.kernel, cls.targets, cls.token_ids)

    def _sub_batch(self, rows):
        return (
            self.hidden[rows],
            self.kernel,
            {name: value[rows] for name, value in self.targets.items()},
            self.token_ids[rows],
        )

    def test_reference_finds_exactly_the_designed_windows(self):
        self.assertEqual(self.windows, [(0, 8192), (0, 16384), (2, 24576)])

    def test_matches_float64_reference_with_hidden_j_minus_1_alignment(self):
        _, count, stats = self.stats
        self.assertEqual(count, float(self.targets["teacher_mask_BR"].sum()))
        self.assertEqual(stats["teacher_boundary_windows"], 3.0)
        self.assertEqual(stats["teacher_boundary_rollouts"], 2.0)
        self.assertEqual(stats["teacher_boundary_nll_tokens"], 3.0 * WINDOW)
        np.testing.assert_allclose(stats["teacher_boundary_nll_sum"], self.nll_ref, rtol=1e-5)
        # The reference resolves a one-position slip of the predictor or of the window.
        for shift in ({"hidden_shift": 0}, {"window_shift": 1}, {"window_shift": -1}):
            with self.subTest(**shift):
                shifted = _reference_nll(
                    self.hidden, self.kernel, self.token_ids, self.windows, **shift
                )
                self.assertGreater(abs(shifted - stats["teacher_boundary_nll_sum"]), 30.0)

    def test_block_size_row_order_and_masked_garbage_are_neutral(self):
        rng = np.random.default_rng(3)
        perm = np.stack(
            [rng.permutation(self.targets["teacher_mask_BR"].shape[1]) for _ in range(4)]
        )
        shuffled = {
            name: np.take_along_axis(value, perm.reshape(perm.shape + (1,) * (value.ndim - 2)), 1)
            for name, value in self.targets.items()
        }
        # Masked rows of b1 and b3 keep the positions that would complete a window.
        masked = ~shuffled["teacher_mask_BR"]
        padding = masked & (shuffled["teacher_positions_BR"] == 0)
        shuffled["teacher_positions_BR"][padding] = rng.choice([10**6, -3], padding.sum())
        shuffled["teacher_top512_ids_BRK"][masked] = V + 5
        shuffled["teacher_top512_logits_BRK"][masked] = np.nan
        for block in (100, 256):
            with self.subTest(block_size=block):
                kl, count, stats = _boundary_stats(
                    block, self.hidden, self.kernel, shuffled, self.token_ids
                )
                np.testing.assert_allclose(kl, self.stats[0], rtol=1e-6)
                self.assertEqual(count, self.stats[1])
                for key in BOUNDARY_KEYS[1:]:
                    self.assertEqual(stats[key], self.stats[2][key])
                np.testing.assert_allclose(
                    stats["teacher_boundary_nll_sum"],
                    self.stats[2]["teacher_boundary_nll_sum"],
                    rtol=1e-6,
                )

    def test_near_misses_padding_and_short_chains_give_zero_sums(self):
        _, _, stats = _boundary_stats(256, *self._sub_batch([1, 3]))
        self.assertEqual(stats, dict.fromkeys(BOUNDARY_KEYS, 0.0))
        # Position 0 is no boundary: targets 1..255 plus one padding row (read as target 1).
        targets = {name: value[:1, :WINDOW].copy() for name, value in self.targets.items()}
        targets["teacher_positions_BR"][0] = np.arange(WINDOW) % (WINDOW - 1)
        targets["teacher_mask_BR"][0] = np.arange(WINDOW) < WINDOW - 1
        _, _, stats = _boundary_stats(
            256, self.hidden[:1, :512], self.kernel, targets, self.token_ids[:1, :512]
        )
        self.assertEqual(stats, dict.fromkeys(BOUNDARY_KEYS, 0.0))
        hidden, kernel, targets = _base_case()
        token_ids = np.ones(hidden.shape[:2], np.int32)
        _, _, stats = _boundary_stats(4, hidden, kernel, targets, token_ids)
        self.assertEqual(stats, dict.fromkeys(BOUNDARY_KEYS, 0.0))

    def test_microbatch_sums_add_up_to_token_weighted_update_mean(self):
        # Update = microbatches {b0, b1}, {b2, b3} and an all-ineligible {b1, b3}.
        parts = [
            _boundary_stats(256, *self._sub_batch(rows))[2] for rows in ([0, 1], [2, 3], [1, 3])
        ]
        totals = {key: sum(part[key] for part in parts) for key in BOUNDARY_KEYS}
        for key in BOUNDARY_KEYS[1:]:
            self.assertEqual(totals[key], self.stats[2][key])
        np.testing.assert_allclose(
            totals["teacher_boundary_nll_sum"], self.stats[2]["teacher_boundary_nll_sum"], rtol=1e-6
        )
        update_nll = totals["teacher_boundary_nll_sum"] / totals["teacher_boundary_nll_tokens"]
        np.testing.assert_allclose(update_nll, self.nll_ref / (3 * WINDOW), rtol=1e-5)
        # Microbatches hold 2 and 1 windows, so a mean of their means is a different number.
        means = [
            p["teacher_boundary_nll_sum"] / p["teacher_boundary_nll_tokens"] for p in parts[:2]
        ]
        self.assertGreater(abs(np.mean(means) - update_nll), 1e-3)

    def test_kl_and_gradients_are_unchanged_and_nll_has_no_gradient(self):
        base = _base_case()
        cases = {
            "boundary_case": (256, self.hidden, self.kernel, self.targets, self.token_ids),
            "base_case": (4, *base, np.ones(base[0].shape[:2], np.int32)),
        }
        for name, (block, hidden, kernel, targets, token_ids) in cases.items():
            with self.subTest(case=name):
                (kl, count), grads = _kl_and_grad(block)(hidden, kernel, targets)
                (kl_b, (count_b, _)), grads_b = _boundary_kl_and_grad(block)(
                    hidden, kernel, targets, token_ids
                )
                np.testing.assert_array_equal(kl_b, kl)
                np.testing.assert_array_equal(count_b, count)
                for grad, grad_b in zip(grads, grads_b):
                    self.assertTrue(np.isfinite(np.asarray(grad)).all())
                    np.testing.assert_array_equal(np.asarray(grad_b), np.asarray(grad))
                for grad in _boundary_nll_grad(block)(hidden, kernel, targets, token_ids):
                    self.assertFalse(np.any(np.asarray(grad)))

    def test_rejects_token_ids_of_another_layout(self):
        with self.assertRaisesRegex(ValueError, "boundary_token_ids_BT"):
            chunked_teacher_kl_stats(
                jnp.asarray(self.hidden),
                jnp.asarray(self.kernel),
                {name: jnp.asarray(value) for name, value in self.targets.items()},
                boundary_token_ids_BT=jnp.asarray(self.token_ids[:, :-1]),
            )


def _sharded_child():
    """Real tiny Qwen3.5 (vocab 248320) on (tp, fsdp, dp) meshes; prints one JSON per mesh."""
    from flax import nnx
    from jax.sharding import PartitionSpec as P

    from omegalax.distributed.mesh import make_mesh, mesh_rules
    from omegalax.models.sharding_runtime import set_attn_backend
    from omegalax.text import api as text_api

    assert jax.device_count() == 4, jax.devices()
    rng = np.random.default_rng(1)
    base_cfg = dataclasses.replace(
        text_api.resolve_config("qwen3.5-smoke-dense"), vocab_size=V, tie_word_embeddings=False
    )
    B, L = 4, 16
    tokens = rng.integers(1, V, size=(B, L)).astype(np.int32)
    overlap = 3.0 * rng.normal(size=V)
    overlap[248_046] = 25.0
    rows = [
        [_teacher_row(overlap), _teacher_row(3.0 * rng.normal(size=V)), None],
        [None, _teacher_row(3.0 * rng.normal(size=V)), _teacher_row(3.0 * rng.normal(size=V))],
        [_teacher_row(3.0 * rng.normal(size=V)), None, None],
        [None, None, None],
    ]
    positions = [[0, 6, 0], [0, 9, 15], [4, 0, 0], [0, 0, 0]]
    targets_np = _targets(rows, positions, [[r is not None for r in batch] for batch in rows])
    _, _, boundary_targets, boundary_ids = _boundary_case()
    boundary_hidden = rng.normal(size=boundary_ids.shape + (base_cfg.hidden_size,)).astype(
        jnp.bfloat16
    )

    for tp, fsdp, dp in ((1, 1, 4), (2, 2, 1)):
        with mesh_rules(make_mesh(tp_size=tp, fsdp_size=fsdp, dp_size=dp)) as mesh:
            model, cfg = text_api.init_model(
                base_cfg, jax.random.key(0), tp_size=tp, fsdp_size=fsdp, dp_size=dp
            )
            set_attn_backend(model, "xla")
            batch = text_api.shard_batch_dict({"token_ids_BT": tokens, **targets_np}, cfg, mesh)
            token_ids = batch.pop("token_ids_BT")
            hidden, _ = nnx.jit(lambda m, x: text_api.forward(m, x, 0, cfg))(model, token_ids)
            kernel = model.output_weight()
            vocab_axis = "tp" if tp > 1 else None
            assert hidden.dtype == jnp.bfloat16 and kernel.dtype == jnp.float32
            assert cfg.shd_cfg.logits_btv[2] == vocab_axis, cfg.shd_cfg.logits_btv
            assert kernel.sharding.spec == P("fsdp" if fsdp > 1 else None, vocab_axis)
            kl, count, grad_h, grad_w = _run(2, hidden, kernel, batch, cfg.shd_cfg.logits_btv)
            if tp == fsdp == 1:
                # Pure DP: rows stay local, so no gather of indices, logits or hidden states.
                hlo = _kl_and_grad(2, cfg.shd_cfg.logits_btv).lower(hidden, kernel, batch)
                assert "all-gather" not in hlo.compile().as_text()
            b_batch = text_api.shard_batch_dict(
                {"token_ids_BT": boundary_ids, **boundary_targets}, cfg, mesh
            )
            b_ids = b_batch.pop("token_ids_BT")
            b_hidden = jax.device_put(
                boundary_hidden, jax.sharding.NamedSharding(mesh, cfg.shd_cfg.act_btd)
            )
            b_kl, b_count, b_stats = _boundary_stats(
                256, b_hidden, kernel, b_batch, b_ids, cfg.shd_cfg.logits_btv
            )
            if tp == fsdp == 1:
                hlo = _boundary_fn(256, cfg.shd_cfg.logits_btv).lower(
                    b_hidden, kernel, b_batch, b_ids
                )
                assert "all-gather" not in hlo.compile().as_text()
            hidden, kernel = np.asarray(hidden), np.asarray(kernel)
        b_unsharded = _boundary_stats(256, boundary_hidden, kernel, boundary_targets, boundary_ids)
        b_windows = _reference_windows(boundary_targets, boundary_ids.shape[1])
        b_nll_ref = _reference_nll(boundary_hidden, kernel, boundary_ids, b_windows)
        assert b_windows == [(0, 8192), (0, 16384), (2, 24576)], b_windows
        for key in BOUNDARY_KEYS[1:]:
            assert b_stats[key] == b_unsharded[2][key], (key, b_stats, b_unsharded)
        assert b_count == b_unsharded[1], (b_count, b_unsharded[1])
        np.testing.assert_allclose(b_kl, b_unsharded[0], rtol=1e-6)
        np.testing.assert_allclose(
            b_stats["teacher_boundary_nll_sum"],
            b_unsharded[2]["teacher_boundary_nll_sum"],
            rtol=1e-6,
        )
        np.testing.assert_allclose(b_stats["teacher_boundary_nll_sum"], b_nll_ref, rtol=1e-5)
        unsharded = _run(2, hidden, kernel, targets_np)
        kl_ref, count_ref, grad_h_ref, grad_w_ref = _reference(hidden, kernel, targets_np)
        assert count == count_ref == 5, (count, count_ref)
        np.testing.assert_allclose(kl, kl_ref, rtol=1e-5)
        _assert_close(grad_w, grad_w_ref, 1e-4)
        # The hidden cotangent has the bf16 model dtype, like the CE path.
        _assert_close(grad_h.astype(np.float64), grad_h_ref, 1e-2)
        np.testing.assert_allclose(kl, unsharded[0], rtol=1e-6)
        _assert_close(grad_w, unsharded[3], 1e-5)
        print(
            json.dumps(
                {
                    "mesh": [tp, fsdp, dp],
                    "kl": kl,
                    "kl_reference": kl_ref,
                    "boundary_windows": b_stats["teacher_boundary_windows"],
                    "boundary_nll_sum": b_stats["teacher_boundary_nll_sum"],
                    "boundary_nll_reference": b_nll_ref,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    if os.environ.get(_CHILD_ENV) == "1":
        _sharded_child()
    else:
        absltest.main()
