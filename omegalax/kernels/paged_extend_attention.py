"""Causal attention of packed extend tiles over a paged KV cache (Pallas, Mosaic GPU, sm90).

FlashAttention-3 structure, after ``jax.experimental.pallas.ops.gpu.attention_mgpu``: each CTA
takes one query head and a tile of ``TILE`` consecutive tokens of one sequence, split over two
compute warpgroups (wgmma, ping-pong scheduled); a third warpgroup streams the sequence's KV
pages, one 64-token page per step, with TMA.
"""

import functools
import math

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jax import lax

TILE = 128  # Tokens per CTA.
ROWS = 64  # Tokens per compute warpgroup.


@functools.partial(jax.jit, static_argnames=("scale",))
def paged_extend_attention(
    q_THK: jax.Array,
    k_pages_GPSK: jax.Array,
    v_pages_GPSK: jax.Array,
    page_table_NP: jax.Array,
    q_pos_N: jax.Array,
    kv_len_N: jax.Array,
    *,
    scale: float,
) -> jax.Array:
    """Tile ``n`` holds tokens ``[n * TILE, (n + 1) * TILE)`` of one sequence, at positions
    ``q_pos_N[n] + i``; they attend causally to KV positions ``< kv_len_N[n]``, read through
    ``page_table_NP[n]``. Rows past ``kv_len_N[n]`` are padding; an empty tile returns zeros.
    Returns attention outputs shaped like ``q_THK``.
    """
    T, H, K = q_THK.shape
    G, _, S, _ = k_pages_GPSK.shape
    if S != ROWS or T % TILE or K % 64:
        raise ValueError(
            f"Needs {ROWS}-token pages, T % {TILE} == 0 and K % 64 == 0; got {S=}, {T=}, {K=}."
        )
    R = H // G
    N = T // TILE
    max_concurrent_steps = 2
    dtype = q_THK.dtype
    log2e = math.log2(math.e)

    def kernel(q_ref, k_ref, v_ref, pt_ref, qpos_ref, kvlen_ref, out_ref, scoped):
        tile = lax.axis_index("tile")
        q_head = lax.axis_index("heads")
        smem_buffers, buffer_barriers, consumed_barriers, schedule_barrier = scoped
        wg_idx = lax.axis_index("wg")
        qo_smem2, k_smem, v_smem = smem_buffers
        k_barriers, v_barriers, q_barriers = buffer_barriers
        k_consumed_barriers, v_consumed_barriers = consumed_barriers
        q_pos = qpos_ref[tile]
        kv_len = kvlen_ref[tile]
        # KV steps (pages) the tile needs: up to its last token's position, within kv_len.
        block_max_kv_steps = pl.cdiv(jnp.minimum(kv_len, q_pos + TILE), jnp.int32(S))

        def perform_schedule_barrier():
            plgpu.barrier_arrive(schedule_barrier)
            plgpu.barrier_wait(schedule_barrier)

        @pl.when(wg_idx < 2)
        def _compute_wg():
            plgpu.set_max_registers(232, action="increase")
            qo_smem = qo_smem2.at[wg_idx]
            row0 = tile * TILE + wg_idx * ROWS
            q_seq_base = q_pos + wg_idx * ROWS
            # A warpgroup whose rows all lie past kv_len holds only padding: it skips the KV loop.
            kv_steps = jnp.where(
                q_seq_base < kv_len,
                pl.cdiv(jnp.minimum(kv_len, q_seq_base + ROWS), jnp.int32(S)),
                0,
            )
            full_kv_steps = jnp.minimum(
                lax.div(jnp.minimum(q_seq_base, kv_len), jnp.int32(S)), kv_steps
            )

            plgpu.copy_gmem_to_smem(
                q_ref.at[pl.ds(row0, ROWS), q_head], qo_smem, q_barriers.at[wg_idx]
            )
            plgpu.barrier_wait(q_barriers.at[wg_idx])
            m_i = plgpu.layout_cast(
                jnp.full((ROWS,), -jnp.inf, jnp.float32), plgpu.Layout.WGMMA.reduce(1)
            )
            l_i = plgpu.layout_cast(jnp.full((ROWS,), 0, jnp.float32), plgpu.Layout.WGMMA.reduce(1))
            acc = plgpu.layout_cast(jnp.full((ROWS, K), 0, jnp.float32), plgpu.Layout.WGMMA)

            @pl.when(kv_steps > 0)
            def _():
                plgpu.barrier_wait(k_barriers.at[0])

            pl.when(wg_idx == 1)(perform_schedule_barrier)

            def kv_loop(kv_step, carry, masked: bool = False):
                acc, m_i, l_i = carry
                slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))

                def compute_qk(acc_ref):
                    plgpu.wgmma(acc_ref, qo_smem, plgpu.transpose_ref(k_smem.at[slot], (1, 0)))
                    perform_schedule_barrier()
                    return acc_ref[...]

                qk = pl.run_scoped(compute_qk, plgpu.ACC((ROWS, S), jnp.float32))
                plgpu.barrier_arrive(k_consumed_barriers.at[slot])
                if masked:
                    q_ids = plgpu.broadcasted_iota(
                        jnp.int32, (ROWS, S), 0, layout=plgpu.Layout.WGMMA
                    )
                    kv_ids = plgpu.broadcasted_iota(
                        jnp.int32, (ROWS, S), 1, layout=plgpu.Layout.WGMMA
                    )
                    kv_pos = kv_ids + kv_step * S
                    qk = jnp.where((kv_pos <= q_ids + q_seq_base) & (kv_pos < kv_len), qk, -jnp.inf)
                # Base-2 softmax with the scale folded in; m stays in log2 units.
                m_ij = jnp.maximum(m_i, qk.max(axis=1) * (scale * log2e))
                alpha = jnp.exp2(m_i - m_ij)
                m_i = m_ij
                p = jnp.exp2(qk * (scale * log2e) - lax.broadcast_in_dim(m_ij, qk.shape, [0]))
                acc *= lax.broadcast_in_dim(alpha, acc.shape, [0])
                l_i *= alpha
                p16 = p.astype(dtype)

                def end_softmax_barriers():
                    plgpu.barrier_arrive(schedule_barrier)
                    plgpu.barrier_wait(v_barriers.at[slot])
                    plgpu.barrier_wait(schedule_barrier)

                if K <= 128:
                    l_i += p.sum(axis=1)
                    acc, l_i, m_i, p16 = lax.optimization_barrier((acc, l_i, m_i, p16))
                    end_softmax_barriers()
                else:
                    end_softmax_barriers()
                    l_i += p.sum(axis=1)

                def compute_pv(acc_ref):
                    plgpu.wgmma(acc_ref, p16, v_smem.at[slot])
                    wait_step = kv_step + 1
                    wait_slot = lax.rem(wait_step, jnp.array(max_concurrent_steps, kv_step.dtype))

                    @pl.when(wait_step < kv_steps)
                    def _wait():
                        plgpu.barrier_wait(k_barriers.at[wait_slot])

                acc = pl.run_state(compute_pv)(plgpu.ACC.init(acc))
                plgpu.barrier_arrive(v_consumed_barriers.at[slot])
                return acc, m_i, l_i

            def epilogue_kv_loop(kv_step, _):
                # Release the pipeline slots this warpgroup does not need (the other one may).
                slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))
                plgpu.barrier_arrive(k_consumed_barriers.at[slot])
                plgpu.barrier_arrive(v_consumed_barriers.at[slot])
                perform_schedule_barrier()
                perform_schedule_barrier()

            acc, m_i, l_i = lax.fori_loop(0, full_kv_steps, kv_loop, (acc, m_i, l_i))
            acc, m_i, l_i = lax.fori_loop(
                full_kv_steps, kv_steps, functools.partial(kv_loop, masked=True), (acc, m_i, l_i)
            )
            lax.fori_loop(kv_steps, block_max_kv_steps, epilogue_kv_loop, None)
            pl.when(wg_idx == 0)(perform_schedule_barrier)
            # Rows of an empty tile have l == 0; write zeros rather than NaN.
            l_safe = jnp.where(l_i == 0, 1.0, l_i)
            acc /= lax.broadcast_in_dim(l_safe, (ROWS, K), [0])
            qo_smem[...] = acc.astype(dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(qo_smem, out_ref.at[pl.ds(row0, ROWS), q_head])
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg_idx == 2)
        def _memory_wg():
            plgpu.set_max_registers(40, action="decrease")
            kv_head = lax.div(q_head, jnp.array(R, q_head.dtype))
            for i in range(max_concurrent_steps):

                @pl.when(i < block_max_kv_steps)
                def _():
                    page = pt_ref[tile, i]
                    plgpu.copy_gmem_to_smem(k_ref.at[kv_head, page], k_smem.at[i], k_barriers.at[i])
                    plgpu.copy_gmem_to_smem(v_ref.at[kv_head, page], v_smem.at[i], v_barriers.at[i])

            @pl.loop(0, jnp.maximum(block_max_kv_steps - max_concurrent_steps, 0))
            def _kv_loop(kv_step):
                tma_step = kv_step + max_concurrent_steps
                tma_slot = lax.rem(kv_step, jnp.array(max_concurrent_steps, kv_step.dtype))
                page = pt_ref[tile, tma_step]
                plgpu.barrier_wait(k_consumed_barriers.at[tma_slot])
                plgpu.copy_gmem_to_smem(
                    k_ref.at[kv_head, page], k_smem.at[tma_slot], k_barriers.at[tma_slot]
                )
                plgpu.barrier_wait(v_consumed_barriers.at[tma_slot])
                plgpu.copy_gmem_to_smem(
                    v_ref.at[kv_head, page], v_smem.at[tma_slot], v_barriers.at[tma_slot]
                )

    def entry(q_ref, k_ref, v_ref, pt_ref, qpos_ref, kvlen_ref, out_ref):
        tiling = plgpu.TilingTransform((8, 64))
        swizzle = plgpu.SwizzleTransform(128)
        qo_scratch = plgpu.SMEM((2, ROWS, K), dtype, transforms=(tiling, swizzle))
        k_scratch = plgpu.SMEM((max_concurrent_steps, S, K), dtype, transforms=(tiling, swizzle))
        v_scratch = plgpu.SMEM((max_concurrent_steps, S, K), dtype, transforms=(tiling, swizzle))
        pl.run_scoped(
            lambda *args: kernel(q_ref, k_ref, v_ref, pt_ref, qpos_ref, kvlen_ref, out_ref, args),
            [qo_scratch, k_scratch, v_scratch],
            (
                plgpu.Barrier(num_barriers=max_concurrent_steps),
                plgpu.Barrier(num_barriers=max_concurrent_steps),
                plgpu.Barrier(num_barriers=2),
            ),
            (plgpu.Barrier(num_arrivals=2, num_barriers=max_concurrent_steps),) * 2,
            plgpu.Barrier(num_arrivals=2),
            collective_axes="wg",
        )

    return plgpu.kernel(
        entry,
        out_shape=jax.ShapeDtypeStruct(q_THK.shape, dtype),
        grid=(H, N),
        grid_names=("heads", "tile"),
        num_threads=3,
        thread_name="wg",
        compiler_params=plgpu.CompilerParams(approx_math=True),
    )(q_THK, k_pages_GPSK, v_pages_GPSK, page_table_NP, q_pos_N, kv_len_N)
