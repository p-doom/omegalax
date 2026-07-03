"""Sequence packing tests: bin-packing, segment/position/loss layout, and the
packed==unpacked equivalence that proves no cross-document attention leakage and
correct per-document positions.

Runs on CPU (``xla`` attention). The GPU equivalence (triton backend) + throughput
and the CP path are exercised by the GPU verification (see the branch report)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402

from omegalax.data.packing import (  # noqa: E402
    build_packed_sequences,
    first_fit_pack,
)
from omegalax.distributed.mesh import make_mesh, mesh_rules  # noqa: E402
from omegalax.models.qwen3.config import make_config as make_qwen3_config  # noqa: E402
from omegalax.models.qwen3.model import Qwen3  # noqa: E402
from omegalax.models.shard_config import (  # noqa: E402
    ShardConfig,
    axis_rules_for_mesh,
    shard_config_for_mesh,
)
from omegalax.models.sharding_runtime import init_model_sharded, set_attn_backend  # noqa: E402


class FirstFitPackTest(absltest.TestCase):
    def test_bins_respect_capacity(self):
        lengths = [7, 5, 11, 4, 9]
        bins = first_fit_pack(lengths, max_length=16)
        # Each bin's total length must not exceed capacity.
        for b in bins:
            self.assertLessEqual(sum(lengths[i] for i in b), 16)
        # Every sample assigned exactly once, order preserved within a bin.
        flat = [i for b in bins for i in b]
        self.assertEqual(sorted(flat), list(range(len(lengths))))
        for b in bins:
            self.assertEqual(b, sorted(b))

    def test_first_fit_placement(self):
        # 7 -> bin0(rem 9); 5 -> bin0(rem 4); 11 -> bin1(rem 5); 4 -> bin0(rem 0);
        # 9 -> bin2 (doesn't fit bin1's 5).
        bins = first_fit_pack([7, 5, 11, 4, 9], max_length=16)
        self.assertEqual(bins, [[0, 1, 3], [2], [4]])

    def test_oversized_sample_is_own_bin(self):
        bins = first_fit_pack([20, 3], max_length=16)
        self.assertEqual(bins[0], [0])  # 20 > 16 -> own bin (truncated later)

    def test_single_bin_when_all_fit(self):
        self.assertEqual(first_fit_pack([7, 5, 11], max_length=32), [[0, 1, 2]])


def _doc(tokens, mask):
    return {"token_ids": np.array(tokens, np.int32), "loss_mask": np.array(mask, np.int32)}


class BuildPackedSequencesTest(absltest.TestCase):
    def test_layout_of_packed_row(self):
        docs = [
            _doc([10, 11, 12], [0, 1, 1]),  # doc 1, len 3
            _doc([20, 21], [0, 1]),  # doc 2, len 2
        ]
        out = build_packed_sequences(docs, max_length=8, pad_id=0, bins=[[0, 1]])
        self.assertEqual(out["token_ids_BT"].shape, (1, 8))
        np.testing.assert_array_equal(out["token_ids_BT"][0], [10, 11, 12, 20, 21, 0, 0, 0])
        # segment ids: doc 1 -> 1, doc 2 -> 2, padding -> 0.
        np.testing.assert_array_equal(out["segment_ids_BT"][0], [1, 1, 1, 2, 2, 0, 0, 0])
        # positions RESET per document; padding -> 0.
        np.testing.assert_array_equal(out["position_ids_BT"][0], [0, 1, 2, 0, 1, 0, 0, 0])
        # attention mask marks real tokens.
        np.testing.assert_array_equal(out["attention_mask_BT"][0], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_boundary_loss_masked(self):
        # Each doc's FIRST token must be unsupervised (it is the previous doc's
        # cross-boundary next-token target). Interior mask is preserved.
        docs = [
            _doc([10, 11, 12], [1, 1, 1]),  # note: first token mask forced to 0
            _doc([20, 21], [1, 1]),
        ]
        out = build_packed_sequences(docs, max_length=8, pad_id=0, bins=[[0, 1]])
        loss = out["loss_mask_BT"][0]
        # doc-1 first token (idx 0) and doc-2 first token (idx 3) are zeroed.
        self.assertEqual(loss[0], 0)
        self.assertEqual(loss[3], 0)
        np.testing.assert_array_equal(loss, [0, 1, 1, 0, 1, 0, 0, 0])
        # No supervised target ever crosses a document boundary: for every
        # supervised target position j, token j-1 shares its segment.
        seg = out["segment_ids_BT"][0]
        for j in range(1, len(loss)):
            if loss[j]:
                self.assertEqual(seg[j], seg[j - 1], f"cross-doc target at {j}")

    def test_oversized_doc_truncated(self):
        docs = [_doc(list(range(20)), [1] * 20)]
        out = build_packed_sequences(docs, max_length=8, pad_id=0)
        self.assertEqual(out["token_ids_BT"].shape, (1, 8))
        np.testing.assert_array_equal(out["token_ids_BT"][0], list(range(8)))
        np.testing.assert_array_equal(out["segment_ids_BT"][0], [1] * 8)


class _EquivHarness:
    """Small fp32 Qwen3 dense model on a 1-device mesh with the xla backend."""

    def __init__(self):
        cfg = dataclasses.replace(make_qwen3_config("qwen3-smoke"), dtype=jnp.float32)
        self.mesh = make_mesh(tp_size=1, fsdp_size=1, dp_size=1, cp_size=1)
        cfg = dataclasses.replace(cfg, shd_cfg=shard_config_for_mesh(ShardConfig.default(), self.mesh))
        with mesh_rules(self.mesh):
            self.model = init_model_sharded(
                Qwen3, cfg, jax.random.key(0), self.mesh, axis_rules_for_mesh(self.mesh)
            )
        set_attn_backend(self.model, "xla")
        self.cfg = cfg
        self.V = cfg.vocab_size

    def logits(self, token_ids_BT, segment_ids_BT, position_ids_BT):
        with mesh_rules(self.mesh):
            hidden, _ = self.model(
                jnp.asarray(token_ids_BT),
                jnp.asarray(segment_ids_BT),
                None,
                jnp.array(0, dtype=jnp.int32),
                position_ids_BT=jnp.asarray(position_ids_BT),
            )
            logits = self.model.lm_head(hidden)
        return np.asarray(jax.device_get(logits))


def _token_nll(logits_TV, targets_T):
    """Per-position next-token NLL: NLL[i] predicts targets_T[i] from logits_TV[i]."""
    lse = np.log(np.exp(logits_TV - logits_TV.max(-1, keepdims=True)).sum(-1)) + logits_TV.max(-1)
    tgt = logits_TV[np.arange(len(targets_T)), targets_T]
    return lse - tgt


class PackedEquivalenceTest(absltest.TestCase):
    """The gold test: per-document logits + loss are identical packed vs standalone."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.h = _EquivHarness()
        rng = np.random.RandomState(0)
        cls.lens = [7, 5, 11]
        cls.docs = [rng.randint(1, cls.h.V, size=L).astype(np.int32) for L in cls.lens]

    def test_packed_equals_unpacked_logits_and_loss(self):
        h = self.h
        # ---- Standalone: each doc as its own sequence, own positions. ----
        standalone = []
        for tok in self.docs:
            L = len(tok)
            lg = h.logits(tok[None], np.ones((1, L), np.int32), np.arange(L)[None])[0]
            standalone.append(lg)

        # ---- Packed: 3 docs in one sequence (segment ids + reset positions). ----
        samples = [{"token_ids": t, "loss_mask": np.ones_like(t)} for t in self.docs]
        max_len = 32
        packed = build_packed_sequences(samples, max_len, pad_id=0)
        self.assertEqual(packed["token_ids_BT"].shape[0], 1)  # all fit one bin
        pack_lg = h.logits(
            packed["token_ids_BT"], packed["segment_ids_BT"], packed["position_ids_BT"]
        )[0]

        worst_logit = 0.0
        worst_loss = 0.0
        start = 0
        for tok, sa in zip(self.docs, standalone):
            L = len(tok)
            pk = pack_lg[start : start + L]
            worst_logit = max(worst_logit, float(np.abs(pk - sa).max()))
            # per-token NLL of predicting the actual next token (interior positions).
            if L >= 2:
                nll_sa = _token_nll(sa[:-1], tok[1:])
                nll_pk = _token_nll(pk[:-1], tok[1:])
                worst_loss = max(worst_loss, float(np.abs(nll_sa - nll_pk).max()))
            start += L

        print(f"\n[packed==unpacked] max|logit diff|={worst_logit:.3e} "
              f"max|per-token loss diff|={worst_loss:.3e}")
        self.assertLess(worst_logit, 1e-4, f"logit diff {worst_logit}")
        self.assertLess(worst_loss, 1e-4, f"loss diff {worst_loss}")

    def test_no_cross_document_leakage(self):
        # doc-1 logits must be INVARIANT to doc-2/doc-3 token values.
        h = self.h
        samples = [{"token_ids": t, "loss_mask": np.ones_like(t)} for t in self.docs]
        packed = build_packed_sequences(samples, 32, pad_id=0)
        base = h.logits(
            packed["token_ids_BT"], packed["segment_ids_BT"], packed["position_ids_BT"]
        )[0]

        rng = np.random.RandomState(1)
        docs2 = [self.docs[0]] + [
            rng.randint(1, h.V, size=L).astype(np.int32) for L in self.lens[1:]
        ]
        samples2 = [{"token_ids": t, "loss_mask": np.ones_like(t)} for t in docs2]
        packed2 = build_packed_sequences(samples2, 32, pad_id=0)
        other = h.logits(
            packed2["token_ids_BT"], packed2["segment_ids_BT"], packed2["position_ids_BT"]
        )[0]

        L0 = self.lens[0]
        diff = float(np.abs(base[:L0] - other[:L0]).max())
        print(f"\n[no-leakage] doc-1 logit diff when later docs change: {diff:.3e}")
        self.assertLess(diff, 1e-5, f"doc-1 leaked cross-document info: {diff}")


if __name__ == "__main__":
    absltest.main()
