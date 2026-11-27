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
import functools
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

from omegalax.engine import steps


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
    pages: list[int] = dataclasses.field(default_factory=list)
    length: int = 0
    turn: _Turn | None = None
    forks: list["_Session"] = dataclasses.field(default_factory=list)
    last_used: float = 0.0


class OutOfPages(Exception):
    pass


def _signature(state: nnx.State):
    leaves, treedef = jax.tree.flatten(state)
    return treedef, [(x.shape, x.dtype, x.sharding) for x in leaves]


def _inference_cast(model: nnx.Module):
    """Stores every Linear kernel (but the lm_head) and embedding table in its module's
    compute dtype, which the forward pass casts them to before every use anyway. The
    lm_head stays in its storage dtype because the trainer's loss multiplies it directly.
    """
    cast = {}
    for path, m in nnx.iter_modules(model):
        if isinstance(m, nnx.Linear) and m is not model.lm_head:
            cast[(*path, "kernel")] = m.dtype
        elif isinstance(m, nnx.Embed):
            cast[(*path, "embedding")] = m.dtype

    def to_inference(state):
        def leaf(path, x):
            dtype = cast.get(tuple(k.key for k in path if isinstance(k, jax.tree_util.DictKey)))
            return x if dtype is None else x.astype(dtype)

        return jax.tree_util.tree_map_with_path(leaf, state)

    return to_inference


class Engine:
    def __init__(self, model: nnx.Module, cfg: EngineConfig):
        self.cfg = cfg
        S = cfg.page_size
        self._max_pages = -(-cfg.max_context // (8 * S)) * 8
        graphdef, state = nnx.split(model)
        self._mesh = jax.tree.leaves(state)[0].sharding.mesh
        self._signature = _signature(state)
        self._to_inference = jax.jit(_inference_cast(model))
        self._state = self._to_inference(state)
        attn = model.layers[0].attn
        self._kv_shape = (len(model.layers), attn.num_kv_heads, cfg.num_pages, S, attn.head_dim)
        self._block_q = max(1, 128 // attn.n_rep)
        self._fork_slack = cfg.max_group * S
        self._kv = None
        self._decode = jax.jit(
            functools.partial(steps.decode, graphdef),
            static_argnames=("num_steps", "num_work"),
            donate_argnums=(1,),
        )
        self._extend = jax.jit(
            functools.partial(steps.extend, graphdef),
            static_argnames=("block_q", "num_work"),
            donate_argnums=(1,),
        )
        self._score = jax.jit(functools.partial(steps.score, graphdef), static_argnames=("pad_id",))
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
        self._thread: threading.Thread | None = None
        self.stats = collections.Counter()

    def warmup(self) -> None:
        """Allocates the page pool and compiles every bucket by running it on inactive rows.

        The pool is allocated here, not in ``__init__``, so a caller can drop its float32
        model between the two.
        """
        L, *shape = self._kv_shape
        self._kv = [(jnp.zeros(shape, jnp.bfloat16), jnp.zeros(shape, jnp.bfloat16)) for _ in range(L)]
        for B in self.cfg.decode_buckets:
            self._run_decode_call(self._decode_inputs([], B))
        for T in self.cfg.extend_buckets:
            self._run_extend_call(self._extend_inputs([], T))
        jax.block_until_ready(self._kv)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

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

    def update_weights(self, state: nnx.State, version: int) -> Future:
        future = Future()
        self._submit(("weights", state, version, future))
        return future

    def score(self, token_lists: Sequence[Sequence[int]]) -> Future:
        future = Future()
        self._submit(("score", [list(t) for t in token_lists], future))
        return future

    def _loop(self) -> None:
        try:
            with jax.set_mesh(self._mesh):
                self._serve()
        except BaseException as exc:
            with self._lock:
                self._error = exc
            pending = [s.turn.future for s in self._sessions.values() if s.turn is not None]
            while not self._commands.empty():
                for x in self._commands.get():
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

    def _set_weights(self, state: nnx.State, version: int) -> None:
        if _signature(state) != self._signature:
            raise ValueError("update_weights needs the same tree, shapes, dtypes and shardings.")
        self._state = self._to_inference(state)
        self._version = version

    def _new_session(self, tokens: list[int]) -> _Session:
        s = _Session(next(self._ids), tokens, np.zeros(self._max_pages, np.int32))
        self._sessions[s.id] = s
        return s

    def _admit(self, req: Request, futures: list[Future]) -> None:
        lead = None if req.session is None else self._sessions[req.session]
        if lead is not None and lead.turn is not None:
            futures[0].set_exception(ValueError(f"Session {lead.id} is still generating."))
            return
        base = 0 if lead is None else len(lead.tokens)
        if base + len(req.tokens) + max(req.max_new_tokens) > self.cfg.max_context:
            futures[0].set_exception(ValueError(f"Request exceeds max_context={self.cfg.max_context}."))
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

    def _alloc(self, s: _Session, num_tokens: int, preempt: bool) -> None:
        need = -(-num_tokens // self.cfg.page_size) - len(s.pages)
        while need > len(self._free):
            self._evict_one(s, preempt)
        for _ in range(need):
            page = self._free.pop()
            self._refs[page] = 1
            s.page_row[len(s.pages)] = page
            s.pages.append(page)

    def _drop_pages(self, s: _Session) -> None:
        for page in s.pages:
            self._refs[page] -= 1
            if self._refs[page] == 0:
                self._free.append(page)
        s.pages = []
        s.page_row[:] = 0
        s.length = 0

    def _evict_one(self, keep: _Session, preempt: bool) -> None:
        """Drops the least recently used idle session, else (if allowed) the newest decoder."""
        idle = [s for s in self._sessions.values() if s.turn is None and s.pages]
        if idle:
            victim = min(idle, key=lambda s: s.last_used)
            self._drop_pages(victim)
            self.stats["evict_idle"] += 1
            return
        running = [s for s in self._decoding.values() if s is not keep] if preempt else []
        if not running:
            raise OutOfPages
        victim = max(running, key=lambda s: s.id)
        del self._decoding[victim.id]
        self._drop_pages(victim)
        self._extend_queue.appendleft(victim)
        self.stats["evict_running"] += 1

    def _release(self, s: _Session) -> None:
        if s.turn is not None:
            raise ValueError(f"Session {s.id} is still generating; abort it instead.")
        self._drop_pages(s)
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
        S, Q = self.cfg.page_size, self._block_q
        items, used, rows, fork_writes = [], 0, 0, 0
        reserve = (len(self._decoding) + 1) * (-(-self.cfg.decode_steps // S) + 1)
        while self._extend_queue:
            s = self._extend_queue[0]
            todo = len(s.tokens) - s.length
            room = (cap - used) // Q * Q
            take = todo if -(-todo // Q) * Q <= room else room // max(Q, S) * max(Q, S)
            final = take == todo
            new_rows = 1 + len(s.forks) if final else 0
            new_fork_writes = len(s.forks) * (len(s.tokens) % S) if final else 0
            if (
                take <= 0
                or rows + new_rows > self.cfg.extend_rows
                or fork_writes + new_fork_writes > self._fork_slack
            ):
                break
            if s.length == 0:
                pages = -(-(take + Q) // self.cfg.page_size) + len(s.forks) + reserve
                evictable = sum(len(x.pages) for x in self._sessions.values() if x.turn is None)
                if pages > len(self._free) + evictable:
                    break
            try:
                self._alloc(s, s.length + take, preempt=False)
                if final:
                    for f in s.forks:
                        self._alloc_fork(s, f)
            except OutOfPages:
                break
            self._extend_queue.popleft()
            items.append((s, s.length, s.length + take, final))
            used += -(-take // Q) * Q
            rows += new_rows
            fork_writes += new_fork_writes
        if not items:
            if not self._decoding:
                raise OutOfPages(f"Session {self._extend_queue[0].id} does not fit in the page pool.")
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
        self._alloc(fork, len(lead.tokens), preempt=False)

    def _extend_inputs(self, items, T: int):
        S, Q = self.cfg.page_size, self._block_q
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
            o += -(-n // Q) * Q
        args = (tokens, positions, write_src, write_slot, page_table, q_pos, kv_len, sample_idx, temp, key_data)
        return args, N

    def _run_extend_call(self, inputs):
        args, N = inputs
        t = time.perf_counter()
        self._kv, tok_R, logp_R = self._extend(
            self._state,
            self._kv,
            *args,
            block_q=self._block_q,
            num_work=N + self.cfg.spare_work,
        )
        tok_R, logp_R = np.asarray(tok_R), np.asarray(logp_R)
        self.stats["extend_s"] += time.perf_counter() - t
        return tok_R, logp_R

    def _step_decode(self) -> None:
        K, S = self.cfg.decode_steps, self.cfg.page_size
        for s in sorted(self._decoding.values(), key=lambda s: s.id):
            if s.id in self._decoding:
                self._alloc(s, s.length + min(K, s.turn.remaining), preempt=True)
        rows = sorted(self._decoding.values(), key=lambda s: s.id)
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
        for b, s in enumerate(rows):
            tok[b] = s.tokens[-1]
            pos[b] = s.length
            remaining[b] = s.turn.remaining
            temp[b] = s.turn.temperature
            key_data[b] = s.turn.key_data
            stop[b] = s.turn.stop_at_eos
            page_table[b] = s.page_row
        return tok, pos, remaining, temp, key_data, stop, page_table

    def _run_decode_call(self, args):
        B = args[0].shape[0]
        t = time.perf_counter()
        self._kv, tok_KB, logp_KB, valid_KB = self._decode(
            self._state,
            self._kv,
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
            if self.cfg.pad_id in t:
                raise ValueError("score input contains pad_id, which the trainer forward masks out.")
            tokens[b, : len(t)] = t
        logp_BT = np.asarray(self._score(self._state, tokens, pad_id=self.cfg.pad_id))
        return [logp_BT[b, : len(t) - 1] for b, t in enumerate(token_lists)]
