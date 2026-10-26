"""Data-only format, reader and KL reference for ``rollout_teacher_v1`` datasets.

Only NumPy and ArrayRecord are imported at module level, so the Torch builder/scorer
can load this file directly. JAX is imported lazily by the KL reference function.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from array_record.python.array_record_module import ArrayRecordReader

FORMAT = "rollout_teacher_v1"
MAGIC = b"OMXRT01\n"
_HEADER_LEN = struct.Struct("<Q")
VOCAB_SIZE = 248320
TOP_K = 512
SPECIAL_IDS = tuple(range(248044, 248077))
STOP_TOKEN_IDS = (248044, 248046)
TARGET_BLOCK_ROWS = 1024
TURN_KINDS = ("complete", "final_truncated")
MANIFEST_FILENAME = "manifest.json"
CATALOG_FILENAME = "catalog.json"
_READER_OPTIONS = "readahead_buffer_size:0,max_parallelism:0"

# Array name -> (little-endian dtype, trailing shape). Order is the on-disk order.
_SCHEMA = {
    "sequence": {
        "token_ids": ("<i4", ()),
        "target_positions": ("<i4", ()),
    },
    "targets": {
        "target_positions": ("<i4", ()),
        "top512_ids": ("<i4", (TOP_K,)),
        "top512_logits": ("<f4", (TOP_K,)),
        "special_logits": ("<f4", (len(SPECIAL_IDS),)),
        "logz_all": ("<f4", ()),
        "logz_tail": ("<f4", ()),
    },
}
_SEQUENCE_KEYS = {"sequence_id", "family", "turns", "aliases", "stop_positions"}
_TURN_KEYS = {"start", "end", "call", "turn_kind", "eligible", "audit_reason"}
_TARGET_KEYS = {"sequence_id", "block_index", "binding"}


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_constant(name: str) -> None:
    raise ValueError(f"Non-finite JSON constant in {FORMAT} header: {name}")


def _validate_sequence(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    tokens = arrays["token_ids"]
    positions = arrays["target_positions"]
    n = tokens.shape[0]
    _require(n >= 1, "sequence.token_ids must not be empty")
    _require(
        tokens.min() >= 0 and tokens.max() < VOCAB_SIZE, f"token_ids must be in [0, {VOCAB_SIZE})"
    )
    _require(not positions.size or positions[-1] < n, "target_positions must be < len(token_ids)")
    _require(
        _SEQUENCE_KEYS <= metadata.keys(), f"sequence metadata requires {sorted(_SEQUENCE_KEYS)}"
    )
    for key in ("sequence_id", "family"):
        _require(
            isinstance(metadata[key], str) and metadata[key],
            f"sequence {key} must be a non-empty string",
        )
    _require(
        isinstance(metadata["aliases"], list) and metadata["aliases"],
        "aliases must be a non-empty list",
    )

    turns = metadata["turns"]
    _require(isinstance(turns, list), "turns must be a list")
    generated = np.zeros(n, dtype=bool)
    target_mask = np.zeros(n, dtype=bool)
    previous_end = 1
    for index, turn in enumerate(turns):
        _require(
            isinstance(turn, dict) and _TURN_KEYS <= turn.keys(),
            f"turn {index} requires {sorted(_TURN_KEYS)}",
        )
        start, end = turn["start"], turn["end"]
        _require(
            type(start) is int and type(end) is int and previous_end <= start < end <= n,
            f"turn {index} range must be ordered, non-empty and inside [1, {n}]",
        )
        _require(
            turn["turn_kind"] in TURN_KINDS, f"turn {index} turn_kind must be one of {TURN_KINDS}"
        )
        _require(type(turn["eligible"]) is bool, f"turn {index} eligible must be a bool")
        reason = turn["audit_reason"]
        _require(
            reason is None or isinstance(reason, str),
            f"turn {index} audit_reason must be str or None",
        )
        _require(turn["eligible"] or reason, f"masked turn {index} requires an audit_reason")
        generated[start:end] = True
        target_mask[start:end] = turn["eligible"]
        previous_end = end
    _require(
        np.array_equal(np.flatnonzero(target_mask), positions),
        "target_positions must equal the union of eligible turn ranges",
    )

    stops = metadata["stop_positions"]
    _require(
        isinstance(stops, list)
        and all(type(p) is int for p in stops)
        and stops == sorted(set(stops)),
        "stop_positions must be sorted unique ints",
    )
    stops = np.asarray(stops, dtype=np.int64)
    _require(
        np.all((stops >= 1) & (stops < n))
        and generated[stops].all()
        and np.isin(tokens[stops], STOP_TOKEN_IDS).all(),
        f"stop_positions must be generated tokens with IDs in {STOP_TOKEN_IDS}",
    )


def _validate_targets(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    m = arrays["target_positions"].shape[0]
    _require(1 <= m <= TARGET_BLOCK_ROWS, f"target block rows must be in [1, {TARGET_BLOCK_ROWS}]")
    _require(
        all(a.shape[0] == m for a in arrays.values()), "target arrays must share the row count"
    )
    ids = arrays["top512_ids"]
    logits = arrays["top512_logits"]
    _require(ids.min() >= 0 and ids.max() < VOCAB_SIZE, f"top512_ids must be in [0, {VOCAB_SIZE})")
    _require(np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0), "top512_ids must be unique per row")
    step = np.diff(logits, axis=1)
    _require(
        np.all((step < 0) | ((step == 0) & (np.diff(ids, axis=1) > 0))),
        "top512 rows must be sorted by logit descending, then ID ascending",
    )
    for name in ("top512_logits", "special_logits", "logz_all", "logz_tail"):
        _require(np.isfinite(arrays[name]).all(), f"{name} must be finite")
    _require(_TARGET_KEYS <= metadata.keys(), f"targets metadata requires {sorted(_TARGET_KEYS)}")
    _require(
        isinstance(metadata["sequence_id"], str) and metadata["sequence_id"],
        "targets sequence_id must be non-empty",
    )
    _require(
        type(metadata["block_index"]) is int and metadata["block_index"] >= 0,
        "block_index must be an int >= 0",
    )
    _require(
        isinstance(metadata["binding"], str) and metadata["binding"],
        "binding must be a non-empty string",
    )


def _validate(kind: str, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    _require(isinstance(kind, str) and kind in _SCHEMA, f"Unknown {FORMAT} record kind: {kind!r}")
    schema = _SCHEMA[kind]
    _require(isinstance(metadata, dict), "metadata must be a dict")
    _require(
        isinstance(arrays, dict) and set(arrays) == set(schema),
        f"{kind} arrays must be exactly {list(schema)}",
    )
    for name, (dtype, trailing) in schema.items():
        array = arrays[name]
        _require(
            isinstance(array, np.ndarray)
            and array.dtype == np.dtype(dtype)
            and array.ndim == 1 + len(trailing)
            and array.shape[1:] == trailing,
            f"{kind}.{name} must be a {dtype} array with shape [*{', ' if trailing else ''}"
            f"{', '.join(map(str, trailing))}]",
        )
    positions = arrays["target_positions"]
    _require(
        not positions.size or (positions[0] >= 1 and np.all(np.diff(positions) > 0)),
        "target_positions must be strictly increasing and >= 1",
    )
    if kind == "sequence":
        _validate_sequence(metadata, arrays)
    else:
        _validate_targets(metadata, arrays)


def encode_record(kind: str, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    """Validate and serialize one record: magic, header length, JSON header, raw arrays."""

    _validate(kind, metadata, arrays)
    specs = []
    offset = 0
    for name in _SCHEMA[kind]:
        specs.append([name, arrays[name].dtype.str, list(arrays[name].shape), offset])
        offset += arrays[name].nbytes
    header = json.dumps(
        {"format": FORMAT, "kind": kind, "metadata": metadata, "arrays": specs},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    # Pad so array bytes start 8-byte aligned within the payload.
    header += b" " * (-(len(MAGIC) + _HEADER_LEN.size + len(header)) % 8)
    return b"".join(
        [
            MAGIC,
            _HEADER_LEN.pack(len(header)),
            header,
            *(arrays[name].tobytes() for name in _SCHEMA[kind]),
        ]
    )


def decode_record(payload: bytes) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    """Strictly parse one record; arrays are read-only views into ``payload``."""

    _require(isinstance(payload, bytes) and payload.startswith(MAGIC), f"Not a {FORMAT} record")
    header_start = len(MAGIC) + _HEADER_LEN.size
    _require(len(payload) >= header_start, "Truncated record header")
    (header_len,) = _HEADER_LEN.unpack_from(payload, len(MAGIC))
    data_start = header_start + header_len
    _require(len(payload) >= data_start, "Truncated record header")
    header = json.loads(payload[header_start:data_start], parse_constant=_reject_constant)
    _require(
        isinstance(header, dict) and set(header) == {"format", "kind", "metadata", "arrays"},
        "Malformed record header",
    )
    _require(header["format"] == FORMAT, f"Expected format {FORMAT}, got {header['format']!r}")
    kind = header["kind"]
    _require(isinstance(kind, str) and kind in _SCHEMA, f"Unknown {FORMAT} record kind: {kind!r}")
    schema = _SCHEMA[kind]
    specs = header["arrays"]
    _require(
        isinstance(specs, list)
        and all(isinstance(spec, list) and len(spec) == 4 for spec in specs)
        and [spec[0] for spec in specs] == list(schema),
        f"{kind} header must list arrays {list(schema)}",
    )
    arrays = {}
    offset = 0
    for name, dtype, shape, spec_offset in specs:
        _require(
            dtype == schema[name][0]
            and spec_offset == offset
            and isinstance(shape, list)
            and all(type(dim) is int and dim >= 0 for dim in shape),
            f"Invalid header entry for {kind}.{name}",
        )
        count = math.prod(shape)
        nbytes = count * np.dtype(dtype).itemsize
        _require(data_start + offset + nbytes <= len(payload), f"Truncated array {kind}.{name}")
        arrays[name] = np.frombuffer(
            payload, dtype=dtype, count=count, offset=data_start + offset
        ).reshape(shape)
        offset += nbytes
    _require(data_start + offset == len(payload), "Trailing bytes after record arrays")
    _validate(kind, header["metadata"], arrays)
    return kind, header["metadata"], arrays


def _read_records(path: Path, indices: list[int]) -> list[bytes]:
    reader = ArrayRecordReader(str(path), _READER_OPTIONS)
    _require(reader.ok(), f"Cannot open ArrayRecord shard: {path}")
    try:
        num_records = reader.num_records()
        _require(all(0 <= i < num_records for i in indices), f"Record index out of range in {path}")
        return reader.read(indices)
    finally:
        reader.close()


def dataset_source_binding(manifest: dict[str, Any]) -> str:
    """Bind the exact source snapshot and prepared index, not only the builder version."""
    fields = {
        "builder": manifest["binding_sha256"],
        "snapshot": manifest["snapshot"]["sha256"],
        "prepared": manifest["prepared"]["sha256"],
    }
    _require(
        all(isinstance(v, str) and v for v in fields.values()), "Invalid dataset source binding"
    )
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class TeacherDataset:
    """Random access to sequences and selected target blocks of one dataset root.

    Only ``manifest.json``, ``catalog.json`` and the indexed ArrayRecord records are read;
    raw rollout JSON is never needed. Shards are opened per call, so the object pickles.
    """

    def __init__(self, root: str | Path, split: str | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = json.loads((self.root / MANIFEST_FILENAME).read_text())
        _require(
            self.manifest.get("format") == FORMAT
            and self.manifest.get("vocab_size") == VOCAB_SIZE
            and self.manifest.get("special_ids") == list(SPECIAL_IDS),
            f"Manifest does not bind {FORMAT}, vocab {VOCAB_SIZE} and the fixed special IDs",
        )
        catalog = json.loads((self.root / CATALOG_FILENAME).read_text())
        _require(catalog.get("format") == FORMAT, f"Catalog format must be {FORMAT}")
        _require(
            catalog.get("complete") is True,
            "Dataset annotation is incomplete; resume it before training",
        )
        _require(
            catalog.get("dataset_binding") == dataset_source_binding(self.manifest)
            and isinstance(catalog.get("dataset_binding"), str),
            "Catalog and source manifest bindings differ",
        )
        self.binding = catalog["binding"]
        self.sequences = [e for e in catalog["sequences"] if split is None or e["split"] == split]
        _require(self.sequences, f"No catalog sequences for split={split!r}")
        self._by_id = {entry["sequence_id"]: entry for entry in self.sequences}
        _require(len(self._by_id) == len(self.sequences), "Duplicate sequence_id in catalog")
        for entry in self.sequences:
            for key in ("path", "targets_path"):
                path = PurePosixPath(entry[key])
                _require(
                    not path.is_absolute() and ".." not in path.parts,
                    f"{key} must stay inside the root",
                )
            row, position = 0, 0
            for block in entry["target_records"]:
                rows = block["row_stop"] - block["row_start"]
                _require(
                    block["row_start"] == row
                    and 1 <= rows <= TARGET_BLOCK_ROWS
                    and position < block["position_min"] <= block["position_max"],
                    f"Non-contiguous target_records for {entry['sequence_id']}",
                )
                row, position = block["row_stop"], block["position_max"]
            _require(
                row == entry["target_count"] and position < entry["token_count"],
                f"target_records do not cover {entry['sequence_id']}",
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        return self.get_sequence(self.sequences[index]["sequence_id"])

    def get_sequence(self, sequence_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        return read_sequence(self.root, self._by_id[sequence_id])

    def get_targets(
        self, sequence_id: str, start: int = 0, stop: int | None = None
    ) -> dict[str, np.ndarray]:
        """Target rows whose absolute token position j is in [start, stop); see ``read_targets``."""

        return read_targets(self.root, self._by_id[sequence_id], self.binding, start, stop)


def read_sequence(
    root: Path, entry: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read the sequence record of one catalog entry (or an index pointer with the same keys)."""

    sequence_id = entry["sequence_id"]
    (payload,) = _read_records(root / entry["path"], [entry["record_index"]])
    kind, metadata, arrays = decode_record(payload)
    _require(
        kind == "sequence"
        and metadata["sequence_id"] == sequence_id
        and arrays["token_ids"].shape[0] == entry["token_count"]
        and arrays["target_positions"].shape[0] == entry["target_count"],
        f"Sequence record does not match catalog entry {sequence_id}",
    )
    return metadata, arrays


def read_targets(
    root: Path, entry: dict[str, Any], binding: str, start: int = 0, stop: int | None = None
) -> dict[str, np.ndarray]:
    """Target rows of one catalog entry whose absolute token position j is in [start, stop).

    Row j is the teacher distribution after tokens 0..j-1, i.e. the prediction made
    from student hidden position j-1. A segment of tokens [a, b) therefore needs
    ``read_targets(root, entry, binding, a + 1, b + 1)``, including the boundary target b.
    """

    sequence_id = entry["sequence_id"]
    stop = entry["token_count"] if stop is None else stop
    _require(0 <= start <= stop, f"Invalid target window [{start}, {stop})")
    selected = [
        (index, block)
        for index, block in enumerate(entry["target_records"])
        if block["position_min"] < stop and block["position_max"] >= start
    ]
    if not selected:
        return {
            name: np.empty((0, *trailing), dtype=dtype)
            for name, (dtype, trailing) in _SCHEMA["targets"].items()
        }
    payloads = _read_records(
        root / entry["targets_path"], [block["record_index"] for _, block in selected]
    )
    parts = []
    for (index, block), payload in zip(selected, payloads):
        kind, metadata, arrays = decode_record(payload)
        positions = arrays["target_positions"]
        _require(
            kind == "targets"
            and metadata["sequence_id"] == sequence_id
            and metadata["block_index"] == index
            and metadata["binding"] == binding
            and positions.shape[0] == block["row_stop"] - block["row_start"]
            and positions[0] == block["position_min"]
            and positions[-1] == block["position_max"],
            f"Target block {index} does not match catalog entry {sequence_id}",
        )
        keep = (positions >= start) & (positions < stop)
        parts.append({name: array[keep] for name, array in arrays.items()})
    return {name: np.concatenate([part[name] for part in parts]) for name in _SCHEMA["targets"]}


def top512_special_tail_kl(student_logits, targets: dict[str, Any], special_ids=SPECIAL_IDS):
    """Per-row forward KL(teacher || student) over unique stored IDs plus one tail group.

    Groups are the row's Top-K IDs, the special IDs not already in that Top-K (counted
    once), and the complement of both. Teacher probabilities are the stored raw logits,
    special logits and ``logz_tail`` minus ``logz_all``. Row i of ``student_logits`` is
    the full-vocabulary student prediction for ``targets["target_positions"][i]``.
    """

    import jax
    import jax.numpy as jnp

    logits = jnp.asarray(student_logits, dtype=jnp.float32)
    ids = jnp.asarray(targets["top512_ids"])
    special = jnp.asarray(special_ids, dtype=jnp.int32)
    if (
        logits.ndim != 2
        or logits.shape[0] != ids.shape[0]
        or int(np.max(special_ids)) >= logits.shape[1]
    ):
        raise ValueError(
            "student_logits must be [rows, vocab] covering the target rows and special IDs"
        )
    rows = jnp.arange(ids.shape[0])[:, None]
    overlap = (ids[:, :, None] == special[None, None, :]).any(axis=1)
    student_tail = jax.nn.logsumexp(
        logits.at[rows, ids].set(-jnp.inf).at[:, special].set(-jnp.inf), axis=-1
    )
    student = jnp.concatenate(
        [logits[rows, ids], logits[:, special], student_tail[:, None]], axis=-1
    ) - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    teacher = (
        jnp.concatenate(
            [
                jnp.asarray(targets["top512_logits"]),
                jnp.asarray(targets["special_logits"]),
                jnp.asarray(targets["logz_tail"])[:, None],
            ],
            axis=-1,
        )
        - jnp.asarray(targets["logz_all"])[:, None]
    )
    keep = jnp.concatenate(
        [jnp.ones(ids.shape, dtype=bool), ~overlap, jnp.ones((ids.shape[0], 1), dtype=bool)],
        axis=-1,
    )
    return jnp.sum(jnp.where(keep, jnp.exp(teacher) * (teacher - student), 0.0), axis=-1)
