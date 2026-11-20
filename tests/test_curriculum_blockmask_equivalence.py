"""Curriculum: segment length T with full-BPTT statepassing versus pieces of length P.

Formulation (i) is the production nnx.scan over T-segments. Formulation (ii) runs the same
scan over P-pieces (T divides P) with ``block_pos_BCT = t mod T``, which sets the RoPE
positions and a block-local causal attention mask, while the GDN state runs continuously
through each piece. Without conv state passing the conv restarts at every T-block start, as
at a T-segment start; with conv state passing it runs on, as across T-segments. Both use the
real tiny Qwen3.5 model, the XLA DeltaNet kernel and XLA attention on CPU, and the
production statepassing loss. Negative controls (patched, clearly separated) show which
ingredient is necessary.
"""

import os
import socket

assert os.environ.get("SLURM_JOB_ID")
assert os.environ.get("SLURMD_NODENAME") == socket.gethostname().split(".")[0]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["OMEGALAX_DELTANET_KERNEL"] = "xla"

import json  # noqa: E402
from unittest import mock  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402
from flax import nnx  # noqa: E402

import omegalax  # noqa: E402
from omegalax.distributed.mesh import mesh_rules_for  # noqa: E402
from omegalax.models.qwen3_5 import attention as qwen_attention  # noqa: E402
from omegalax.models.qwen3_5 import deltanet as qwen_deltanet  # noqa: E402
from omegalax.models.qwen3_5.config import Qwen3_5TextConfig  # noqa: E402
from omegalax.models.shard_config import ShardConfig  # noqa: E402
from omegalax.models.sharding_runtime import set_attn_backend  # noqa: E402
from omegalax.text import api as text_api  # noqa: E402
from omegalax.trainers import pretrain as pretrain_trainer  # noqa: E402

print("omegalax.__file__ =", omegalax.__file__, flush=True)

B = 2
PAD_TAIL = 5
MODE = pretrain_trainer.PretrainMode.STATEPASSING_BPTT
COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
_compile_events = []
jax.monitoring.register_event_duration_secs_listener(
    lambda event, duration, **kwargs: (
        _compile_events.append(event) if event == COMPILE_EVENT else None
    )
)


def _cfg(dtype=jnp.float32) -> Qwen3_5TextConfig:
    # Qwen3.5 layout in miniature: 3 GDN layers then 1 full-attention layer, twice.
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        layer_types=(("linear_attention",) * 3 + ("full_attention",)) * 2,
        rope_theta=10_000_000,
        partial_rotary_factor=0.25,
        mrope_section=(1, 1, 0),
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_value_head_dim=8,
        intermediate_size=64,
        shd_cfg=ShardConfig.no_sharding(),
        dtype=dtype,
    )


def _model(cfg):
    model, cfg = text_api.init_model(cfg, jax.random.PRNGKey(0), tp_size=1, fsdp_size=1, dp_size=1)
    set_attn_backend(model, text_backend="xla")
    # Slow GDN decay (|g| ~ 0.01..0.1 per token) so the carried state matters over the
    # whole chain; the default init decays by up to e^-20 per token.
    rng = np.random.RandomState(1)
    for layer in model.text.layers:
        if layer.layer_type != "full_attention":
            H = layer.linear_attn.A_log[...].shape[0]
            layer.linear_attn.A_log[...] = jnp.log(
                jnp.asarray(rng.uniform(0.01, 0.1, H), dtype=jnp.float32)
            )
    return model, cfg


def _chains(N: int, vocab: int, seed: int, reset_pos: int):
    """Row 0: one rollout of N tokens. Row 1: rollouts at 0 and reset_pos, last PAD_TAIL padded."""
    rng = np.random.RandomState(seed)
    tokens = rng.randint(1, vocab, size=(B, N)).astype(np.int32)
    mask = np.ones((B, N), dtype=np.int32)
    mask[1, N - PAD_TAIL :] = 0
    tokens[mask == 0] = 0
    loss = (mask * (rng.rand(B, N) < 0.8)).astype(np.int32)
    starts = np.zeros((B, N), dtype=bool)
    starts[:, 0] = True
    starts[1, reset_pos] = True
    rollout_pos = np.zeros((B, N), dtype=np.int32)
    for b in range(B):
        begin = 0
        for t in range(N):
            if starts[b, t]:
                begin = t
            rollout_pos[b, t] = t - begin
    return tokens, mask, loss, starts, rollout_pos


def _segment_batch(chains, L: int, T: int | None = None, piece_resets: bool = True):
    tokens, mask, loss, starts, rollout_pos = chains
    N = tokens.shape[1]
    C = N // L
    batch = {
        "token_ids_BCT": jnp.asarray(tokens.reshape(B, C, L)),
        "attention_mask_BCT": jnp.asarray(mask.reshape(B, C, L)),
        "loss_mask_BCT": jnp.asarray(loss.reshape(B, C, L)),
        "reset_state_BC": jnp.asarray(starts.reshape(B, C, L)[:, :, 0]),
    }
    if not piece_resets:
        # Mid-piece rollout starts cannot be expressed as a piece-level GDN reset.
        batch["reset_state_BC"] = batch["reset_state_BC"].at[:, 1:].set(False)
    if T is not None:
        batch["block_pos_BCT"] = jnp.asarray((rollout_pos % T).reshape(B, C, L))
    return batch


def _run(model, cfg, batch, pass_conv_state=False, **path_kwargs):
    path_kwargs["pass_conv_state"] = pass_conv_state

    @nnx.jit
    def run(model, batch):
        def loss_fn(inner):
            loss, _ = pretrain_trainer._statepassing_loss_stats(
                inner, batch, cfg, 0, MODE, **path_kwargs
            )
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        hidden, _ = pretrain_trainer._statepassing_hidden_states(
            model, batch, cfg, 0, MODE, **path_kwargs
        )
        logits = hidden @ model.output_weight()
        # Final GDN states through an independent unrolled loop of the same segments/pieces.
        states, conv_states = None, None
        for c in range(batch["token_ids_BCT"].shape[1]):
            kwargs = {}
            if "block_pos_BCT" in batch:
                block_pos_BT = batch["block_pos_BCT"][:, c]
                kwargs = {
                    "position_ids_ZBT": jnp.stack([block_pos_BT] * 3, axis=0),
                    "block_pos_BT": block_pos_BT,
                }
                if not pass_conv_state:
                    kwargs["conv_block_pos_BT"] = block_pos_BT
            if pass_conv_state:
                kwargs["initial_conv_states"] = conv_states
                kwargs["return_conv_states"] = True
            if states is not None:
                states = pretrain_trainer.apply_state_reset(states, batch["reset_state_BC"][:, c])
            if conv_states is not None:
                conv_states = pretrain_trainer.apply_state_reset(
                    conv_states, batch["reset_state_BC"][:, c]
                )
            _, _, states, *rest = text_api.forward_with_gdn_state(
                model,
                batch["token_ids_BCT"][:, c],
                0,
                cfg,
                attention_mask_BT=batch["attention_mask_BCT"][:, c],
                initial_gdn_states=states,
                **kwargs,
            )
            conv_states = rest[0] if pass_conv_state else None
        return loss, grads, logits, states

    return jax.tree.map(np.asarray, run(model, batch))


def _compare(ref, other, real_mask):
    loss_r, grads_r, logits_r, states_r = ref
    loss_o, grads_o, logits_o, states_o = other
    real = real_mask.astype(bool)
    logit_diff = np.abs(logits_r - logits_o)
    grad_rel_max, grad_rel_l2, grad_abs = 0.0, 0.0, 0.0
    for g_r, g_o in zip(jax.tree.leaves(grads_r), jax.tree.leaves(grads_o), strict=True):
        diff = np.abs(g_r - g_o)
        grad_abs = max(grad_abs, float(diff.max()))
        grad_rel_max = max(grad_rel_max, float(diff.max()) / max(float(np.abs(g_r).max()), 1e-30))
        grad_rel_l2 = max(
            grad_rel_l2,
            float(np.linalg.norm(g_r - g_o)) / max(float(np.linalg.norm(g_r)), 1e-30),
        )
    state_abs = max(float(np.abs(a - b).max()) for a, b in zip(states_r, states_o, strict=True))
    state_scale = max(float(np.abs(a).max()) for a in states_r)
    return {
        "loss_abs": abs(float(loss_r) - float(loss_o)),
        "loss_rel": abs(float(loss_r) - float(loss_o)) / abs(float(loss_r)),
        "logits_abs_real": float(logit_diff[real].max()),
        "logits_rel_real": float(logit_diff[real].max()) / float(np.abs(logits_r[real]).max()),
        "logits_abs_all": float(logit_diff.max()),
        "state_abs": state_abs,
        "state_rel": state_abs / state_scale,
        "grad_abs": grad_abs,
        "grad_rel_max": grad_rel_max,
        "grad_rel_l2": grad_rel_l2,
    }


def _fmt(name, r):
    return (
        f"{name:<44} loss_rel={r['loss_rel']:.2e} logits_abs={r['logits_abs_real']:.2e} "
        f"logits_rel={r['logits_rel_real']:.2e} logits_abs_all={r['logits_abs_all']:.2e} "
        f"state_abs={r['state_abs']:.2e} state_rel={r['state_rel']:.2e} "
        f"grad_abs={r['grad_abs']:.2e} grad_rel_max={r['grad_rel_max']:.2e} "
        f"grad_rel_l2={r['grad_rel_l2']:.2e}"
    )


RESULTS = {}


class CurriculumBlockMaskEquivalenceTest(absltest.TestCase):
    def test_pieces_with_block_mask_match_segment_scan(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model, cfg = _model(_cfg())
            for P, N in ((16, 48), (32, 64)):
                chains = _chains(N, cfg.vocab_size, seed=P, reset_pos=P)
                real = chains[1]
                for T in (2, 4, 8, 16, 32):
                    if T > P:
                        continue
                    ref = _run(model, cfg, _segment_batch(chains, T))
                    new = _run(model, cfg, _segment_batch(chains, P, T=T))
                    # fp32 rounding floor: the same T-segments through the unrolled loop, a
                    # second production path with identical math and none of the new code.
                    loop = _run(model, cfg, _segment_batch(chains, T), gdn_layer_limit=6)
                    r = _compare(ref, new, real)
                    f = _compare(ref, loop, real)
                    RESULTS[f"equiv_T{T}_P{P}"] = r
                    RESULTS[f"floor_scan_vs_loop_T{T}_P{P}"] = f
                    print(_fmt(f"equiv T={T:<2} P={P}", r), flush=True)
                    print(_fmt(f"floor scan vs loop T={T:<2} N={N}", f), flush=True)
                    self.assertLess(r["loss_rel"], 1e-5)
                    self.assertLess(r["logits_rel_real"], 1e-4)
                    self.assertLess(r["state_rel"], 1e-4)
                    # T == P is the same math through the new code path and already reaches
                    # ~1e-4 (job 170142); the negative controls sit at ~1e0.
                    self.assertLess(r["grad_rel_l2"], 1e-3)

    def test_pieces_with_conv_passing_match_conv_passing_segment_scan(self):
        # Conv state passing: the conv runs through T-block starts inside a piece and its
        # state crosses piece boundaries in the scan carry, as across T-segments.
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            model, cfg = _model(_cfg())
            P, N = 16, 48
            chains = _chains(N, cfg.vocab_size, seed=P, reset_pos=P)
            real = chains[1]
            for T in (2, 4, 8, 16):
                ref = _run(model, cfg, _segment_batch(chains, T), pass_conv_state=True)
                new = _run(model, cfg, _segment_batch(chains, P, T=T), pass_conv_state=True)
                r = _compare(ref, new, real)
                RESULTS[f"conv_passing_equiv_T{T}_P{P}"] = r
                print(_fmt(f"conv passing equiv T={T:<2} P={P}", r), flush=True)
                self.assertLess(r["loss_rel"], 1e-5)
                self.assertLess(r["logits_rel_real"], 1e-4)
                self.assertLess(r["state_rel"], 1e-4)
                self.assertLess(r["grad_rel_l2"], 1e-3)
                if T < P:
                    # Negative control: the conv reset at T-block starts is a different model.
                    reset = _run(model, cfg, _segment_batch(chains, P, T=T))
                    c = _compare(ref, reset, real)
                    RESULTS[f"ctrl_conv_reset_vs_conv_passing_T{T}_P{P}"] = c
                    print(_fmt(f"ctrl conv reset vs passing T={T:<2}", c), flush=True)
                    self.assertGreater(c["logits_rel_real"], 1e-3)

    def test_negative_controls(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            T, P, N = 4, 16, 48
            model, cfg = _model(_cfg())
            chains = _chains(N, cfg.vocab_size, seed=P, reset_pos=P)
            real = chains[1]
            ref = _run(model, cfg, _segment_batch(chains, T))

            def control(name, batch, patches=()):
                stack = [mock.patch.object(*p) for p in patches]
                for patch in stack:
                    patch.start()
                try:
                    out = _run(model, cfg, batch)
                finally:
                    for patch in reversed(stack):
                        patch.stop()
                r = _compare(ref, out, real)
                RESULTS[name] = r
                print(_fmt(name, r), flush=True)
                return r

            orig_conv = qwen_deltanet._causal_depthwise_conv1d
            orig_attn = qwen_attention.Attention.__call__
            orig_fwd = text_api.forward_with_gdn_state

            def conv_without_block_reset(*args, block_pos_BT=None, **kwargs):
                return orig_conv(*args, **kwargs)

            def attn_without_block_mask(self, *args, block_pos_BT=None, **kwargs):
                return orig_attn(self, *args, **kwargs)

            def fwd_with_piece_positions(*args, position_ids_ZBT=None, **kwargs):
                return orig_fwd(*args, **kwargs)

            P_batch = _segment_batch(chains, P, T=T)
            no_conv = control(
                "ctrl_no_conv_block_reset",
                P_batch,
                [(qwen_deltanet, "_causal_depthwise_conv1d", conv_without_block_reset)],
            )
            no_mask = control(
                "ctrl_no_attention_block_mask",
                P_batch,
                [(qwen_attention.Attention, "__call__", attn_without_block_mask)],
            )
            abs_rope = control(
                "ctrl_rope_positions_t_not_t_mod_T",
                P_batch,
                [(pretrain_trainer.text_api, "forward_with_gdn_state", fwd_with_piece_positions)],
            )
            other_T = control("ref_scan_T8_vs_T4", _segment_batch(chains, 2 * T))
            no_state = _compare(
                ref,
                _run(
                    model,
                    cfg,
                    {
                        **_segment_batch(chains, T),
                        "reset_state_BC": jnp.ones((B, N // T), dtype=jnp.bool_),
                    },
                ),
                real,
            )
            RESULTS["ref_no_gdn_state_passing"] = no_state
            print(_fmt("ref_no_gdn_state_passing", no_state), flush=True)

            # A rollout start that is T- but not P-aligned (here at 24 with P=16) needs a GDN
            # reset in the middle of a piece, which the piece scan does not have.
            mid = _chains(N, cfg.vocab_size, seed=P, reset_pos=24)
            mid_ref = _run(model, cfg, _segment_batch(mid, T))
            mid_new = _run(model, cfg, _segment_batch(mid, P, T=T, piece_resets=False))
            mid_r = _compare(mid_ref, mid_new, mid[1])
            RESULTS["ctrl_mid_piece_rollout_start"] = mid_r
            print(_fmt("ctrl_mid_piece_rollout_start", mid_r), flush=True)

            self.assertGreater(no_conv["logits_rel_real"], 1e-3)
            self.assertGreater(no_mask["logits_rel_real"], 1e-3)
            self.assertGreater(no_state["logits_rel_real"], 1e-3)
            self.assertGreater(mid_r["logits_rel_real"], 1e-3)
            self.assertLess(abs_rope["logits_rel_real"], 1e-4)
            del other_T

    def test_changing_T_does_not_recompile_piece_program(self):
        with mesh_rules_for(tp_size=1, fsdp_size=1, dp_size=1):
            P, N = 16, 48
            model, cfg = _model(_cfg())
            chains = _chains(N, cfg.vocab_size, seed=P, reset_pos=P)

            @nnx.jit
            def step(model, batch):
                def loss_fn(inner):
                    loss, _ = pretrain_trainer._statepassing_loss_stats(inner, batch, cfg, 0, MODE)
                    return loss

                return nnx.value_and_grad(loss_fn)(model)

            piece_compiles, scan_compiles = {}, {}
            for T in (2, 4, 8, 16):
                piece_batch = _segment_batch(chains, P, T=T)
                scan_batch = _segment_batch(chains, T)
                before = len(_compile_events)
                jax.block_until_ready(step(model, piece_batch))
                piece_compiles[T] = len(_compile_events) - before
                before = len(_compile_events)
                jax.block_until_ready(step(model, scan_batch))
                scan_compiles[T] = len(_compile_events) - before
            RESULTS["compiles_piece_program"] = piece_compiles
            RESULTS["compiles_segment_scan"] = scan_compiles
            print(f"backend compiles per T, piece program: {piece_compiles}", flush=True)
            print(f"backend compiles per T, T-segment scan: {scan_compiles}", flush=True)
            self.assertGreater(piece_compiles[2], 0)
            self.assertEqual([piece_compiles[T] for T in (4, 8, 16)], [0, 0, 0])

    @classmethod
    def tearDownClass(cls):
        out = os.environ.get("CURRICULUM_POC_RESULTS")
        if out:
            with open(out, "w") as stream:
                json.dump(RESULTS, stream, indent=2, default=str)


if __name__ == "__main__":
    absltest.main()
