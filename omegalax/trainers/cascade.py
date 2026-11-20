"""Cascade schedule for statepassing: layers outer, segments folded into the batch.

Ported from the validated variant ``v3_cascade`` of the GDN kernel work, where it
was the single largest win (1.21x) but lived only in a runtime monkeypatch harness
and therefore never reached the production path.

Production runs the segments in the outer loop::

    for segment s in 0..C-1:
        for layer l in 0..L-1:
            layer l on segment s          # needs GDN state of (l, s-1)

The only reason for that order is the GDN state. But the state dependency runs
*per layer* (``(l, s)`` needs ``(l, s-1)``), while the hidden-state dependency runs
*per segment* (``(l, s)`` needs ``(l-1, s)``). Swapping the loops satisfies both:
when layer ``l`` starts, all C segment outputs of layer ``l-1`` already exist.

With the loops swapped, every segment is an independent batch row for everything
except the GDN recurrence — attention, MLP, projections, conv and norms never cross
a segment boundary anyway. So the swap needs no model change: reshape ``(B, C, T)``
to ``(B*C, T)`` and do **one** forward. All GEMMs return to IID size instead of
being C times smaller, which is where the speedup comes from.

The GDN recurrence is re-established inside the kernel wrapper: a shard-local batch
of ``B*C`` rows is viewed as ``B`` chains of length ``C*T``, and the chunked delta
rule runs over all ``C*J`` chunks in one sequential pass. That is the same
recurrence the segment loop produces, because segment ``c`` starts from the state
segment ``c-1`` left behind.

Preconditions, all checked in :func:`is_fusible`. If any fails we fall back to the
production path rather than silently computing something else:

* ``bptt_chunks == C`` — otherwise ``_prepare_carried_states_for_segment`` inserts
  ``stop_gradient`` at segment boundaries, which a single fused recurrence cannot
  express.
* ``pass_gdn_state`` and ``SAME_LAYER`` routing with no layer limit.
* ``not pass_rope_positions`` and ``not pass_conv_state`` — each row must restart
  positions at 0 and start from a zero conv state, matching the per-segment forward.

Row order is ``b * C + c``, so a chain's C segments stay contiguous; batch sharding
must keep multiples of C together, which is asserted.

Enabled with ``MFU_CASCADE=1``. Default off, so the production path is unchanged
until the A/B measurement says otherwise.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

# Set by the loss function before the model forward is traced, read by the kernel
# wrapper. Single-threaded tracing, so a module global is safe here.
_SEGMENTS: int | None = None


def enabled() -> bool:
    return os.environ.get("MFU_CASCADE", "0") == "1"


def is_fusible(
    *,
    segments: int,
    resolved_bptt_chunks: int,
    pass_gdn_state: bool,
    gdn_layer_limit: int | None,
    routing_is_same_layer: bool,
    pass_rope_positions: bool,
    pass_conv_state: bool,
) -> bool:
    return (
        segments > 1
        and resolved_bptt_chunks == segments
        and pass_gdn_state
        and gdn_layer_limit is None
        and routing_is_same_layer
        and not pass_rope_positions
        and not pass_conv_state
    )


_INSTALLED = False


def install() -> None:
    """Swap the delta rule for the chain-relinking wrapper. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    from omegalax.models.qwen3_5 import deltanet

    deltanet.chunk_gated_delta_rule = wrap_delta_rule(deltanet.chunk_gated_delta_rule)
    _INSTALLED = True


def wrap_delta_rule(base):
    """Wrap ``chunk_gated_delta_rule`` so it re-links C batch rows into one chain."""

    def cascade_delta_rule(
        q_BTHA,
        k_BTHA,
        v_BTHU,
        g_BTH,
        beta_BTH,
        chunk_size: int = 64,
        initial_state_BHAU=None,
        *,
        return_final_state: bool = False,
    ):
        C = _SEGMENTS
        if C is None or C == 1:
            return base(
                q_BTHA, k_BTHA, v_BTHU, g_BTH, beta_BTH, chunk_size,
                initial_state_BHAU, return_final_state=return_final_state,
            )

        BC, T, H, A = q_BTHA.shape
        U = v_BTHU.shape[-1]
        if BC % C:
            raise ValueError(
                f"cascade: shard-local batch {BC} must be a multiple of C={C}; a "
                "chain's segments must not be split across devices"
            )
        B = BC // C

        def fold(x):
            return x.reshape((B, C * T) + x.shape[2:])

        # The GDN module hands one initial state per *row*; the fused chain wants
        # one per *chain*, i.e. the state entering segment 0. Rows b*C+c for c>0
        # are the zeros the module created for segments that used to receive a
        # carried state, and are subsumed by the chain.
        init_chain = None if initial_state_BHAU is None else initial_state_BHAU[::C]

        out, final = base(
            fold(q_BTHA), fold(k_BTHA), fold(v_BTHU), fold(g_BTH), fold(beta_BTH),
            chunk_size, init_chain, return_final_state=True,
        )
        out_BTHU = out.reshape(BC, T, H, U)
        if return_final_state:
            # Caller contract is one final state per row. The cascade loss does not
            # read them (there is no segment loop left to feed), so replicate the
            # chain's final state across its rows to keep the shape contract.
            return out_BTHU, jnp.repeat(final, C, axis=0)
        return out_BTHU

    return cascade_delta_rule


def loss_stats(
    model,
    batch,
    cfg,
    pad_id,
    pretrain_mode,
    *,
    bptt_chunks,
    pass_gdn_state,
    gdn_layer_limit,
    gdn_state_routing,
    pass_rope_positions,
    pass_conv_state,
):
    """One fused forward over B*C rows, then the usual chunked cross-entropy."""
    global _SEGMENTS
    from omegalax.text import api as text_api
    from omegalax.trainers import pretrain as P

    token_ids_BCT = batch["token_ids_BCT"]
    attention_mask_BCT = batch["attention_mask_BCT"]
    loss_mask_BCT = batch["loss_mask_BCT"]
    reset_state_BC = batch["reset_state_BC"]
    B, C, T = token_ids_BCT.shape

    _SEGMENTS = C
    hidden_BCT_D, aux_loss, _final = text_api.forward_with_gdn_state(
        model,
        token_ids_BCT.reshape(B * C, T),
        pad_id,
        cfg,
        attention_mask_BT=attention_mask_BCT.reshape(B * C, T),
        initial_gdn_states=None,
    )
    D = hidden_BCT_D.shape[-1]
    # The rows axis is FSDP-sharded, and splitting a sharded axis into (B, C) is not
    # a sharding-preserving reshape JAX can infer. Say where the result lives, the
    # same way the model does for its own reshapes.
    hidden_BT_D = jax.lax.reshape(hidden_BCT_D, (B, C * T, D), out_sharding=cfg.shd_cfg.act_btd)

    token_ids_BT = token_ids_BCT.reshape(B, -1)
    masks = P.statepassing_target_masks(loss_mask_BCT, reset_state_BC, bptt_chunks=bptt_chunks)
    mask_names = ("total", "iid_comparable", "boundary") + tuple(f"segment{i}" for i in range(C))
    mask_stack = jnp.stack(
        [masks.total, masks.iid_comparable, masks.boundary, *masks.segments], axis=0
    )
    nll_sums, token_counts = P.chunked_cross_entropy_multi_stats(
        hidden_BT_D,
        P._model_output_weight(model),
        token_ids_BT,
        mask_stack,
        num_tiles=P._NUM_LOSS_TILES,
        logits_out_sharding=cfg.shd_cfg.logits_btv,
    )

    total_nll = P._nll(nll_sums[0], token_counts[0])
    loss = total_nll + aux_loss
    metrics = {
        "loss": loss,
        "nll": total_nll,
        "ppl": P._ppl(total_nll),
        "aux_loss": aux_loss,
        "nll_sum": nll_sums[0],
        "supervised_tokens": token_counts[0],
    }
    for idx, name in enumerate(mask_names[1:], start=1):
        metrics.update(P._metrics_from_nll(name, nll_sums[idx], token_counts[idx]))
    return loss, metrics
