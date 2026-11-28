"""Token-in/token-out rollout engine over a paged KV cache.

A session is one sequence resident in the cache across turns. A turn appends
tokens (a prompt, or ``[last sampled token] + tool output``), runs one packed
extend that samples the turn's first token, then decodes ``decode_steps`` tokens
per device call until ``max_new_tokens`` or EOS. All device work runs on one
engine thread; public methods enqueue commands and return futures.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import queue
import threading
import time
from concurrent.futures import Future
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from omegalax.engine import steps
from omegalax.models.qwen3_5.cache import LinearExtend
from omegalax.models.qwen3_5.deltanet import CHUNK_SIZE as _DELTANET_CHUNK


@dataclasses.dataclass(frozen=True)
class EngineConfig:
    num_pages: int
    max_context: int
    decode_steps: int
    decode_buckets: tuple[int, ...]
    extend_buckets: tuple[int, ...]
    eos_token_ids: tuple[int, ...]
    pad_id: int
    max_group: int = 16
    max_sessions: int = 256
    page_size: int = 64
    extend_rows: int = 128
    spare_work: int = 132


@dataclasses.dataclass(frozen=True)
class Request:
    """Starts one session per key from ``tokens`` (``session is None``) or continues ``session``.

    ``keys`` is uint32 threefry key data shaped ``(n, 2)``; the token at position ``p`` of a
    session is sampled with ``fold_in(key, p)``, so sampling does not depend on batching or
    eviction. ``max_new_tokens`` and ``temperatures`` hold one entry per key.
    """

    tokens: Sequence[int]
    keys: np.ndarray
    max_new_tokens: Sequence[int]
    temperatures: Sequence[float]
    session: int | None = None
    stop_at_eos: bool = True


@dataclasses.dataclass(frozen=True)
class Completion:
    session: int
    tokens: list[int]
    logprobs: list[float]
    weight_versions: list[int]
    finish_reason: str


@dataclasses.dataclass(eq=False)
class _Turn:
    future: Future
    remaining: int
    temperature: float
    key_data: np.ndarray
    stop_at_eos: bool
    tokens: list[int] = dataclasses.field(default_factory=list)
    logprobs: list[float] = dataclasses.field(default_factory=list)
    versions: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(eq=False)
class _Session:
    id: int
    tokens: list[int]
    page_row: np.ndarray
    slot: int
    pages: list[int] = dataclasses.field(default_factory=list)
    length: int = 0
    turn: _Turn | None = None
    forks: list["_Session"] = dataclasses.field(default_factory=list)
    last_used: float = 0.0
    # Admitted to the current turn: its pages are never evicted and the pages it may still
    # allocate before the turn ends are reserved.
    admitted: bool = False


class OutOfPages(Exception):
    pass


def _inference_dtypes(model: nnx.Module, state: nnx.State) -> list:
    """Per state leaf, the dtype the engine stores it in: every Linear kernel (but the
    lm_head) and embedding table in its module's compute dtype, which the forward pass casts
    them to before every use anyway. The lm_head keeps its storage dtype because the trainer's
    loss multiplies it directly.
    """
    cast = {}
    for path, m in nnx.iter_modules(model):
        if isinstance(m, nnx.Linear) and m is not model.lm_head:
            cast[(*path, "kernel")] = m.dtype
        elif isinstance(m, nnx.Embed):
            cast[(*path, "embedding")] = m.dtype
    return [
        cast.get(tuple(k.key for k in path if isinstance(k, jax.tree_util.DictKey)), x.dtype)
        for path, x in jax.tree_util.tree_leaves_with_path(state)
    ]


class Engine:
    def __init__(self, model: nnx.Module, cfg: EngineConfig):
        self.cfg = cfg
        S = cfg.page_size
        self._max_pages = -(-cfg.max_context // S)
        graphdef, state = nnx.split(model)
        self._mesh = jax.tree.leaves(state)[0].sharding.mesh
        tp = self._mesh.shape["tp"]
        if self._mesh.shape["fsdp"] != 1 or self._mesh.shape["dp"] != 1:
            raise ValueError(
                f"The engine shards over 'tp' only; got mesh {dict(self._mesh.shape)}."
            )
        # The jitted programs take the state as a flat leaf list so their cache keys depend on
        # array types only, not on nnx variable metadata.
        self._abstract = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), state
        )
        dtypes = _inference_dtypes(model, state)
        self._to_inference = jax.jit(lambda leaves: [x.astype(d) for x, d in zip(leaves, dtypes)])
        self._state = jax.tree.leaves(state)
        graphdef = (graphdef, jax.tree.structure(state))
        layers = model.text.layers if isinstance(model, steps.Qwen3_5ForCausalLM) else model.layers
        num_slots = cfg.max_sessions + 2
        # Pools are sharded like the activations they hold: KV and DeltaNet states by head, the
        # conv window by channel (the sharding of in_proj_qkv's output).
        self._pool_specs = []
        for layer in layers:
            if hasattr(layer, "attn"):
                a = layer.attn
                if a.num_kv_heads % tp:
                    raise ValueError(f"{a.num_kv_heads} KV heads do not split over tp={tp}.")
                page = ((a.num_kv_heads, cfg.num_pages, S, a.head_dim), jnp.bfloat16, P("tp"))
                self._pool_specs.append((page, page))
                self._block_q = max(1, min(128, 128 * 128 // a.head_dim) // a.n_rep)
            else:
                d = layer.linear_attn
                if d.num_k_heads % tp:
                    raise ValueError(
                        f"{d.num_k_heads} DeltaNet key heads do not split over tp={tp}."
                    )
                self._conv_k = d.conv_kernel_size
                conv_shape = (num_slots, d.conv_kernel_size - 1, 2 * d.key_dim + d.value_dim)
                state_shape = (num_slots, d.num_v_heads, d.head_k_dim, d.head_v_dim)
                self._pool_specs.append(
                    (
                        (conv_shape, jnp.bfloat16, P(None, None, "tp")),
                        (state_shape, jnp.float32, P(None, "tp")),
                    )
                )
        self._linear = len(self._pool_specs) != sum(hasattr(x, "attn") for x in layers)
        self._align = max(self._block_q, _DELTANET_CHUNK if self._linear else 1)
        self._fork_slack = cfg.max_group * S
        self._free_slots = list(range(num_slots - 1, steps.TRASH_SLOT, -1))
        self._pools = None
        bind = lambda fn: lambda *args, **kwargs: fn(graphdef, *args, **kwargs)
        # XLA's dot merger would concatenate the weights of sibling projections (DeltaNet's four
        # in_proj) on every call, a full extra weight copy per decode step.
        no_merge = {"xla_gpu_dot_merger_threshold_mb": 0}
        self._decode = jax.jit(
            bind(steps.decode),
            static_argnames=("num_steps", "num_work"),
            donate_argnums=(1,),
            compiler_options=no_merge,
        )
        self._extend = jax.jit(
            bind(steps.extend),
            static_argnames=("block_q", "num_work"),
            donate_argnums=(1,),
            compiler_options=no_merge,
        )
        self._score = jax.jit(bind(steps.score), static_argnames=("pad_id",))
        self._eos = np.asarray(cfg.eos_token_ids, np.int32)
        self._free = list(range(cfg.num_pages - 1, 0, -1))
        self._refs = np.zeros(cfg.num_pages, np.int32)
        self._sessions: dict[int, _Session] = {}
        self._ids = itertools.count()
        self._extend_queue: collections.deque[_Session] = collections.deque()
        self._decoding: dict[int, _Session] = {}
        self._version = 0
        self._commands: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._command: tuple = ()
        self._thread: threading.Thread | None = None
        self.stats = collections.Counter()

    def start(self) -> dict[str, float]:
        """Starts the engine thread and returns once it is ready to serve.

        The thread casts the weights, allocates the page pool and compiles every bucket by
        running it on inactive rows; it returns the seconds each bucket took. All device work
        stays on this thread because JAX's trace and compile caches are thread-local. The pool
        is allocated here, not in ``__init__``, so a caller can drop its float32 model first.
        """
        ready = Future()
        self._thread = threading.Thread(target=self._loop, args=(ready,), daemon=True)
        self._thread.start()
        return ready.result()

    def _warmup(self) -> dict[str, float]:
        # The first trace of the trainer forward (tokamax attention) extends JAX's trace context,
        # which is part of every jit cache key, so trace it before anything is compiled. tokamax
        # parses absl flags from sys.argv at that point unless the caller already has.
        self._score.trace(self._state, np.zeros((1, 128), np.int32), pad_id=self.cfg.pad_id)
        self._state = self._to_inference(self._state)
        zeros = lambda shape, dtype, spec: jax.jit(
            lambda: jnp.zeros(shape, dtype), out_shardings=NamedSharding(self._mesh, spec)
        )()
        self._pools = [tuple(zeros(*spec) for spec in specs) for specs in self._pool_specs]
        runs = {
            f"decode_{B}": (self._run_decode_call, self._decode_inputs([], B))
            for B in self.cfg.decode_buckets
        }
        runs |= {
            f"extend_{T}": (self._run_extend_call, self._extend_inputs([], T))
            for T in self.cfg.extend_buckets
        }
        seconds = {}
        for name, (run, inputs) in runs.items():
            t = time.perf_counter()
            run(inputs)
            seconds[name] = time.perf_counter() - t
        self.stats.clear()
        return seconds

    def stop(self) -> None:
        self._submit(("stop",))
        self._thread.join()

    def _submit(self, cmd: tuple) -> None:
        with self._lock:
            if self._error is not None:
                raise RuntimeError("The engine thread failed.") from self._error
            self._commands.put(cmd)

    def generate(self, request: Request) -> list[Future]:
        n = len(request.keys)
        if request.keys.dtype != np.uint32 or request.keys.shape != (n, 2):
            raise ValueError("keys must be uint32 threefry key data shaped (n, 2).")
        if len(request.max_new_tokens) != n or len(request.temperatures) != n:
            raise ValueError("max_new_tokens and temperatures need one entry per key.")
        if not 1 <= n <= (1 if request.session is not None else self.cfg.max_group):
            raise ValueError(f"n={n} must be 1 for a continuation and <= {self.cfg.max_group}.")
        if min(request.max_new_tokens) < 1 or min(request.temperatures) <= 0:
            raise ValueError("max_new_tokens must be >= 1 and temperatures > 0.")
        futures = [Future() for _ in range(n)]
        self._submit(("generate", request, futures))
        return futures

    def release(self, session: int) -> None:
        self._submit(("release", session))

    def abort(self, sessions: Sequence[int]) -> None:
        self._submit(("abort", list(sessions)))

    def update_weights(self, params: dict, version: int) -> Future:
        """Swaps in ``params``, an ``nnx.to_pure_dict`` of the model state (trainer layout)."""
        future = Future()
        self._submit(("weights", params, version, future))
        return future

    def score(self, token_lists: Sequence[Sequence[int]]) -> Future:
        if any(self.cfg.pad_id in t for t in token_lists):
            raise ValueError("score input contains pad_id, which the trainer forward masks out.")
        future = Future()
        self._submit(("score", [list(t) for t in token_lists], future))
        return future

    def _loop(self, ready: Future) -> None:
        try:
            with jax.set_mesh(self._mesh):
                ready.set_result(self._warmup())
                self._serve()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            with self._lock:
                self._error = exc
            pending = [s.turn.future for s in self._sessions.values() if s.turn is not None]
            commands = [self._command]
            while not self._commands.empty():
                commands.append(self._commands.get())
            for cmd in commands:
                for x in cmd:
                    pending += x if isinstance(x, list) else [x] if isinstance(x, Future) else []
            for future in pending:
                if not future.done():
                    future.set_exception(exc)
            raise

    def _serve(self) -> None:
        while True:
            idle = not self._extend_queue and not self._decoding
            if not self._drain(block=idle):
                return
            if self._extend_queue:
                self._step_extend()
            if self._decoding:
                self._step_decode()

    def _drain(self, block: bool) -> bool:
        while True:
            try:
                cmd = self._commands.get(block=block)
            except queue.Empty:
                return True
            block = False
            self._command = cmd
            kind = cmd[0]
            if kind == "stop":
                return False
            if kind == "generate":
                self._admit(cmd[1], cmd[2])
            elif kind == "release":
                self._release(self._sessions[cmd[1]])
            elif kind == "abort":
                for sid in cmd[1]:
                    self._abort(self._sessions[sid])
            elif kind == "weights":
                self._set_weights(cmd[1], cmd[2])
                cmd[3].set_result(cmd[2])
            elif kind == "score":
                cmd[2].set_result(self._run_score(cmd[1]))

    def _set_weights(self, params: dict, version: int) -> None:
        state = jax.tree.map(lambda x: x, self._abstract)
        nnx.replace_by_pure_dict(state, params)
        leaves, abstract = jax.tree.leaves(state), jax.tree.leaves(self._abstract)
        spec = lambda x: (x.shape, x.dtype)
        if list(map(spec, leaves)) != list(map(spec, abstract)):
            raise ValueError("update_weights needs the initial shapes and dtypes.")
        # Params may live on the trainer's mesh; one device_put reshards them onto the engine's.
        leaves = jax.device_put(leaves, [x.sharding for x in abstract])
        self._state = self._to_inference(leaves)
        self._version = version

    def _new_session(self, tokens: list[int]) -> _Session:
        s = _Session(
            next(self._ids), tokens, np.zeros(self._max_pages, np.int32), self._free_slots.pop()
        )
        self._sessions[s.id] = s
        return s

    def _admit(self, req: Request, futures: list[Future]) -> None:
        lead = None if req.session is None else self._sessions[req.session]
        if lead is not None and lead.turn is not None:
            futures[0].set_exception(ValueError(f"Session {lead.id} is still generating."))
            return
        base = 0 if lead is None else len(lead.tokens)
        if base + len(req.tokens) + max(req.max_new_tokens) > self.cfg.max_context:
            futures[0].set_exception(
                ValueError(f"Request exceeds max_context={self.cfg.max_context}.")
            )
            return
        if lead is None and len(futures) > len(self._free_slots):
            futures[0].set_exception(ValueError(f"More than max_sessions={self.cfg.max_sessions}."))
            return
        if lead is None:
            lead = self._new_session(list(req.tokens))
            lead.forks = [self._new_session(list(req.tokens)) for _ in futures[1:]]
        else:
            lead.tokens.extend(req.tokens)
        for s, future, key_data, max_new, temp in zip(
            [lead, *lead.forks], futures, req.keys, req.max_new_tokens, req.temperatures
        ):
            s.turn = _Turn(future, max_new, temp, key_data, req.stop_at_eos)
        self._extend_queue.append(lead)

    def _alloc(self, s: _Session, num_tokens: int) -> None:
        need = -(-num_tokens // self.cfg.page_size) - len(s.pages)
        while need > len(self._free):
            self._evict_one(s, need - len(self._free))
        for _ in range(need):
            page = self._free.pop()
            self._refs[page] = 1
            s.page_row[len(s.pages)] = page
            s.pages.append(page)

    def _drop_pages(self, s: _Session, keep: int) -> None:
        for page in s.pages[keep:]:
            self._refs[page] -= 1
            if self._refs[page] == 0:
                self._free.append(page)
        s.page_row[keep : len(s.pages)] = 0
        s.pages = s.pages[:keep]
        s.length = min(s.length, keep * self.cfg.page_size)

    def _evict_one(self, keep: _Session, pages: int) -> None:
        """Frees ``pages`` tail pages of the least recently used idle session, else of the
        unadmitted session last in the extend queue; it re-extends the dropped tail when admitted.
        DeltaNet state cannot be rewound, so a model with DeltaNet layers drops whole sessions.
        Admission reserves every page an admitted session can need, so one is always found.
        """
        idle = [s for s in self._sessions.values() if s.turn is None and s.pages]
        waiting = [s for s in self._extend_queue if not s.admitted and s is not keep and s.pages]
        if idle:
            victim = min(idle, key=lambda s: s.last_used)
            self.stats["evict_idle"] += 1
        elif waiting:
            victim = waiting[-1]
            self.stats["evict_waiting"] += 1
        else:
            raise OutOfPages("An admitted session outgrew its reserved pages.")
        kept = 0 if self._linear else max(0, len(victim.pages) - pages)
        self.stats["evicted_tokens"] += victim.length - min(
            victim.length, kept * self.cfg.page_size
        )
        self._drop_pages(victim, kept)

    def _future_pages(self, s: _Session) -> int:
        """Pages ``s`` and its not yet extended forks may still allocate before their turn ends;
        a fork will share the lead's full prompt pages."""
        S = self.cfg.page_size
        pages = lambda x, have: max(0, -(-(len(x.tokens) + x.turn.remaining) // S) - have)
        shared = len(s.tokens) // S
        return pages(s, len(s.pages)) + sum(pages(f, shared) for f in s.forks)

    def _admissible(self, s: _Session) -> bool:
        """The one admission rule: ``s`` starts its turn only if its future pages fit in the free
        pages plus those held by unadmitted sessions only, after the reservations of every
        admitted session."""
        held = {p for x in self._sessions.values() if x.admitted or x is s for p in x.pages}
        evictable = {p for x in self._sessions.values() for p in x.pages} - held
        reserved = sum(self._future_pages(x) for x in self._sessions.values() if x.admitted)
        return self._future_pages(s) <= len(self._free) + len(evictable) - reserved

    def _release(self, s: _Session) -> None:
        if s.turn is not None:
            raise ValueError(f"Session {s.id} is still generating; abort it instead.")
        self._drop_pages(s, 0)
        self._free_slots.append(s.slot)
        del self._sessions[s.id]

    def _abort(self, s: _Session) -> None:
        if s.turn is not None:
            if s in self._extend_queue:
                self._extend_queue.remove(s)
            self._decoding.pop(s.id, None)
            for f in [s, *s.forks]:
                self._finish(f, "abort")
        for f in [s, *s.forks]:
            self._release(f)

    def _finish(self, s: _Session, reason: str) -> None:
        t = s.turn
        s.turn = None
        s.admitted = False
        s.last_used = time.monotonic()
        t.future.set_result(Completion(s.id, t.tokens, t.logprobs, t.versions, reason))

    def _record(self, s: _Session, toks, logps) -> None:
        t = s.turn
        s.tokens.extend(toks)
        t.tokens.extend(toks)
        t.logprobs.extend(logps)
        t.versions.extend([self._version] * len(toks))
        t.remaining -= len(toks)
        if t.stop_at_eos and toks[-1] in self._eos:
            self._finish(s, "stop")
        elif t.remaining == 0:
            self._finish(s, "length")
        else:
            self._decoding[s.id] = s

    def _step_extend(self) -> None:
        cap = self.cfg.extend_buckets[-1]
        S, A = self.cfg.page_size, self._align
        items, used, rows, forks = [], 0, 0, 0
        while self._extend_queue:
            s = self._extend_queue[0]
            todo = len(s.tokens) - s.length
            room = (cap - used) // A * A
            take = todo if -(-todo // A) * A <= room else room // max(A, S) * max(A, S)
            final = take == todo
            new_forks = len(s.forks) if final else 0
            new_rows = 1 + new_forks if final else 0
            # At most max_group forks per extend bounds both the forks' KV page-tail writes
            # (_fork_slack) and the DeltaNet state writes (W in _linear_extend).
            if (
                take <= 0
                or rows + new_rows > self.cfg.extend_rows
                or forks + new_forks > self.cfg.max_group
            ):
                break
            if not s.admitted:
                if not self._admissible(s):
                    break
                s.admitted = True
            self._extend_queue.popleft()
            self._alloc(s, s.length + take)
            if final:
                for f in s.forks:
                    self._alloc_fork(s, f)
            items.append((s, s.length, s.length + take, final))
            used += -(-take // A) * A
            rows += new_rows
            forks += new_forks
        if not items:
            if not any(x.admitted for x in self._sessions.values()):
                raise OutOfPages(
                    f"Session {self._extend_queue[0].id} does not fit in the page pool."
                )
            return
        T = next(b for b in self.cfg.extend_buckets if b >= used)
        tok_R, logp_R = self._run_extend_call(self._extend_inputs(items, T))
        self.stats["extend_calls"] += 1
        self.stats["extend_tokens"] += used
        r = 0
        for s, start, end, final in items:
            s.length = end
            if not final:
                self._extend_queue.appendleft(s)
                continue
            for f in [s, *s.forks]:
                f.length = end
                self._record(f, [int(tok_R[r])], [float(logp_R[r])])
                r += 1
            s.forks = []

    def _alloc_fork(self, lead: _Session, fork: _Session) -> None:
        """Shares the lead's full prompt pages; the partial last page gets a private copy."""
        full = len(lead.tokens) // self.cfg.page_size
        for page in lead.pages[:full]:
            self._refs[page] += 1
            fork.page_row[len(fork.pages)] = page
            fork.pages.append(page)
        fork.admitted = True
        self._alloc(fork, len(lead.tokens))

    def _extend_inputs(self, items, T: int):
        S, Q, A = self.cfg.page_size, self._block_q, self._align
        N = T // Q
        R = self.cfg.extend_rows
        W = T + self._fork_slack
        tokens = np.zeros(T, np.int32)
        positions = np.zeros(T, np.int32)
        write_src = np.arange(W, dtype=np.int32) % T
        write_slot = self.cfg.num_pages * S + np.arange(W, dtype=np.int32)
        page_table = np.zeros((N, self._max_pages), np.int32)
        q_pos = np.zeros(N, np.int32)
        kv_len = np.zeros(N, np.int32)
        sample_idx = np.zeros(R, np.int32)
        temp = np.ones(R, np.float32)
        key_data = np.zeros((R, 2), np.uint32)
        o, r, w = 0, 0, T
        for s, start, end, final in items:
            n = end - start
            pos = np.arange(start, end, dtype=np.int32)
            tokens[o : o + n] = s.tokens[start:end]
            positions[o : o + n] = pos
            write_slot[o : o + n] = s.page_row[pos // S] * S + pos % S
            for j in range(-(-n // Q)):
                b = o // Q + j
                page_table[b] = s.page_row
                q_pos[b] = start + j * Q
                kv_len[b] = min(end, start + (j + 1) * Q)
            if final:
                tail = np.arange(end // S * S, end, dtype=np.int32)
                for f in [s, *s.forks]:
                    sample_idx[r] = o + n - 1
                    temp[r] = f.turn.temperature
                    key_data[r] = f.turn.key_data
                    r += 1
                    if f is not s:
                        write_src[w : w + len(tail)] = o + tail - start
                        write_slot[w : w + len(tail)] = f.page_row[tail // S] * S + tail % S
                        w += len(tail)
            o += -(-n // A) * A
        linear = self._linear_extend(items, T) if self._linear else None
        args = (
            tokens,
            positions,
            write_src,
            write_slot,
            page_table,
            q_pos,
            kv_len,
            sample_idx,
            temp,
            key_data,
            linear,
        )
        return args, N

    def _linear_extend(self, items, T: int) -> LinearExtend:
        """DeltaNet metadata for the packed batch; see ``omegalax.models.qwen3_5.cache``."""
        K, A, C = self._conv_k, self._align, _DELTANET_CHUNK
        K1 = K - 1
        W = T // A + self.cfg.max_group
        zero_row = T + steps.ZERO_SLOT * K1
        F = T // A * K1
        valid = np.zeros(T, bool)
        conv_fix = T + np.arange(F, dtype=np.int32)
        conv_src = np.full((F, K), zero_row, np.int32)
        conv_out_src = np.full((W, K1), zero_row, np.int32)
        write_slot = self.cfg.max_sessions + 2 + np.arange(W, dtype=np.int32)
        init_slot = np.zeros(T // A, np.int32)
        init_idx = np.full((1, T // C), -1, np.int32)
        final_chunk = np.zeros(W, np.int32)

        def rows(pos, s, start, o):
            prev = np.where(pos >= 0, T + s.slot * K1 + pos - (start - K1), zero_row)
            return np.where(pos >= start, o + pos - start, prev).astype(np.int32)

        o, w = 0, 0
        for i, (s, start, end, final) in enumerate(items):
            n = end - start
            valid[o : o + n] = True
            head = np.arange(min(n, K1))
            conv_fix[i * K1 : i * K1 + len(head)] = o + head
            conv_src[i * K1 : i * K1 + len(head)] = rows(
                start + head[:, None] - K1 + np.arange(K), s, start, o
            )
            init_idx[0, o // C] = i
            init_slot[i] = steps.ZERO_SLOT if start == 0 else s.slot
            out_rows = rows(end - K1 + np.arange(K1), s, start, o)
            for f in [s, *s.forks] if final else [s]:
                write_slot[w] = f.slot
                final_chunk[w] = (o + n - 1) // C
                conv_out_src[w] = out_rows
                w += 1
            o += -(-n // A) * A
        return LinearExtend(
            valid, conv_fix, conv_src, conv_out_src, write_slot, init_slot, init_idx, final_chunk
        )

    def _run_extend_call(self, inputs):
        args, N = inputs
        t = time.perf_counter()
        self._pools, tok_R, logp_R = self._extend(
            self._state,
            self._pools,
            *args,
            block_q=self._block_q,
            num_work=N + self.cfg.spare_work,
        )
        tok_R, logp_R = np.asarray(tok_R), np.asarray(logp_R)
        self.stats["extend_s"] += time.perf_counter() - t
        return tok_R, logp_R

    def _step_decode(self) -> None:
        K = self.cfg.decode_steps
        # The oldest sessions decode first; the rest wait for a call with room in the largest bucket.
        rows = sorted(self._decoding.values(), key=lambda s: s.id)[: self.cfg.decode_buckets[-1]]
        for s in rows:
            self._alloc(s, s.length + min(K, s.turn.remaining))
        B = next(b for b in self.cfg.decode_buckets if b >= len(rows))
        tok_KB, logp_KB, valid_KB = self._run_decode_call(self._decode_inputs(rows, B))
        self.stats["decode_calls"] += 1
        counts = valid_KB.sum(axis=0)
        self.stats["decode_tokens"] += int(counts.sum())
        self.stats["decode_row_steps"] += K * B
        for b, s in enumerate(rows):
            n = int(counts[b])
            del self._decoding[s.id]
            s.length += n
            self._record(s, tok_KB[:n, b].tolist(), logp_KB[:n, b].tolist())

    def _decode_inputs(self, rows, B: int):
        tok = np.zeros(B, np.int32)
        pos = np.zeros(B, np.int32)
        remaining = np.zeros(B, np.int32)
        temp = np.ones(B, np.float32)
        key_data = np.zeros((B, 2), np.uint32)
        stop = np.zeros(B, bool)
        page_table = np.zeros((B, self._max_pages), np.int32)
        slot = np.full(B, steps.TRASH_SLOT, np.int32)
        for b, s in enumerate(rows):
            tok[b] = s.tokens[-1]
            pos[b] = s.length
            remaining[b] = s.turn.remaining
            temp[b] = s.turn.temperature
            key_data[b] = s.turn.key_data
            stop[b] = s.turn.stop_at_eos
            page_table[b] = s.page_row
            slot[b] = s.slot
        return tok, pos, remaining, temp, key_data, stop, page_table, slot

    def _run_decode_call(self, args):
        B = args[0].shape[0]
        t = time.perf_counter()
        self._pools, tok_KB, logp_KB, valid_KB = self._decode(
            self._state,
            self._pools,
            *args,
            self._eos,
            num_steps=self.cfg.decode_steps,
            num_work=B + self.cfg.spare_work,
        )
        out = np.asarray(tok_KB), np.asarray(logp_KB), np.asarray(valid_KB)
        self.stats["decode_s"] += time.perf_counter() - t
        return out

    def _run_score(self, token_lists: list[list[int]]) -> list[np.ndarray]:
        T = max(len(t) for t in token_lists)
        T = max(128, 1 << (T - 1).bit_length())
        B = 1 << (len(token_lists) - 1).bit_length()
        tokens = np.full((B, T), self.cfg.pad_id, np.int32)
        for b, t in enumerate(token_lists):
            tokens[b, : len(t)] = t
        logp_BT = np.asarray(self._score(self._state, tokens, pad_id=self.cfg.pad_id))
        return [logp_BT[b, : len(t) - 1] for b, t in enumerate(token_lists)]
