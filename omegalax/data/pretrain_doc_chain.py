"""Data-only helpers for ``omegalax_doc_chain_v1`` pretraining corpora."""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grain
import numpy as np

DOC_CHAIN_FORMAT = "omegalax_doc_chain_v1"
DOC_CHAIN_DATASET_VERSION = 1
DOC_CHAIN_BINARY_MAGIC = b"OMXDC01\n"
DOC_CHAIN_BINARY_HEADER = struct.Struct("<QQ")
COMPILED_METADATA_FILENAME = "metadata.json"
ARRAY_RECORD_SUFFIX = ".array_record"
DEFAULT_SEGMENT_LENGTH = 4096
DEFAULT_PAD_ID = 0
BATCH_PRETRAIN_METADATA_KEY = "metadata"


@dataclass(frozen=True)
class DocChainRecord:
    doc_id: str
    token_ids: np.ndarray
    doc_token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentRef:
    source_idx: int
    record_idx: int
    doc_id: str
    doc_token_count: int


@dataclass(frozen=True)
class DocChunkRef:
    source_idx: int
    record_idx: int
    doc_id: str
    chunk_idx: int
    start: int
    end: int
    is_last_chunk: bool


@dataclass(frozen=True)
class DocPairRef:
    source_idx: int
    record_idx: int
    doc_id: str
    pair_idx: int
    start: int
    mid: int
    end: int
    doc_token_count: int
    eos_token_idx: int | None = None


ChunkRef = DocChunkRef


# Die Funktion nimmt einen oder mehrere Dateipfade entgegen, bereinigt sie und löst sie zu absoluten Pfaden auf.
# Sie gibt eine Liste von Path-Objekten zurück.
def _normalize_sources(sources: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(sources, (str, Path)):
        raw_sources = [sources]
    elif isinstance(sources, Sequence):
        raw_sources = list(sources)
    else:
        raise TypeError("No valid path")
    return [Path(path).expanduser().resolve() for path in raw_sources]


# Die Funktion nimmt serialisierte Byteströme, Text oder Dictionaries entgegen und bestimmt deren Format.
# Sie parst diese Daten und gibt das rohe Metadaten-Dictionary zurück.
def _json_payload(payload: bytes | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, bytes):
        if payload.startswith(DOC_CHAIN_BINARY_MAGIC):
            return _binary_payload(payload)
        return json.loads(payload)
    if isinstance(payload, str):
        return json.loads(payload)
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"Unsupported doc-chain payload type: {type(payload).__name__}")


# Die Funktion nimmt ein binäres Byte-Array im OMXDC01\n Format entgegen und liest den Header sowie die Token-Bytes aus.
# Sie gibt ein bereinigtes Python-Dictionary mit den Token-IDs als NumPy-Array zurück.
def _binary_payload(payload: bytes) -> dict[str, Any]:
    pos = len(DOC_CHAIN_BINARY_MAGIC)
    header_end = pos + DOC_CHAIN_BINARY_HEADER.size
    if len(payload) < header_end:
        raise ValueError("truncated doc-chain binary header")

    header_len, token_count = DOC_CHAIN_BINARY_HEADER.unpack(payload[pos:header_end])
    pos = header_end
    header = json.loads(payload[pos : pos + header_len].decode("utf-8"))
    pos += header_len

    expected = int(token_count) * np.dtype(np.int32).itemsize
    token_bytes = payload[pos : pos + expected]
    if len(token_bytes) != expected:
        raise ValueError(
            f"truncated token payload: expected {expected} bytes, got {len(token_bytes)}"
        )

    header["token_ids"] = np.frombuffer(token_bytes, dtype=np.int32).copy()
    return header


# Die Funktion nimmt verschiedene Record-Formate entgegen, parst diese mittels der internen Payload-Hilfsfunktionen und validiert die Pflichtfelder.
# Sie gibt ein gebrauchsfertiges DocChainRecord-Objekt zurück.
def deserialize_doc_chain(payload: DocChainRecord | bytes | str | dict[str, Any]) -> DocChainRecord:
    if isinstance(payload, DocChainRecord):
        return payload
    raw = _json_payload(payload)
    fmt = raw.get("format") or raw.get("data_format") or raw.get("type")
    if fmt is not None and fmt != DOC_CHAIN_FORMAT:
        raise ValueError(f"Unsupported doc-chain format: {fmt}")

    doc_id = raw.get("doc_id", raw.get("id"))
    if doc_id is None:
        raise ValueError("Doc-chain record is missing doc_id")

    raw_tokens = None
    for key in ("token_ids", "tokens", "input_ids"):
        if key in raw:
            raw_tokens = raw[key]
            break
    if raw_tokens is None:
        raise ValueError("Doc-chain record is missing token_ids")

    token_ids = np.asarray(raw_tokens, dtype=np.int32)
    doc_token_count = int(raw.get("doc_token_count", token_ids.shape[0]))
    if doc_token_count != int(token_ids.shape[0]):
        raise ValueError(
            f"doc_token_count={doc_token_count} does not match "
            f"token_ids length={token_ids.shape[0]}"
        )

    metadata = dict(raw.get("metadata") or {})
    core_keys = {
        "format",
        "data_format",
        "type",
        "doc_id",
        "id",
        "token_ids",
        "tokens",
        "input_ids",
        "doc_token_count",
        "metadata",
    }
    for key, value in raw.items():
        if key not in core_keys:
            metadata[key] = value

    return DocChainRecord(
        doc_id=str(doc_id),
        token_ids=token_ids,
        doc_token_count=doc_token_count,
        metadata=metadata,
    )


# Die Funktion nimmt den Pfad eines Datensatz-Ordners entgegen und prüft, ob die dazugehörige metadata.json existiert.
# Sie parst die JSON-Metadaten und gibt sie als Dictionary zurück.
def load_arrayrecord_metadata(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    metadata_path = path / COMPILED_METADATA_FILENAME
    if not metadata_path.exists():
        raise ValueError(f"Compiled ArrayRecord dataset metadata does not exist: {metadata_path}")
    return json.loads(metadata_path.read_text())


# Die Funktion nimmt den Pfad eines Datensatzes entgegen und lädt dessen Metadaten.
# Sie überprüft das Formatflag und gibt das Metadaten-Dictionary zurück.
def load_doc_chain_metadata(path: str | Path) -> dict[str, Any]:
    metadata = load_arrayrecord_metadata(path)
    fmt = metadata.get("format")
    if fmt != DOC_CHAIN_FORMAT:
        raise ValueError(f"Expected {DOC_CHAIN_FORMAT} dataset, got format={fmt}")
    return metadata


# Die Funktion nimmt eine Datei oder einen Ordnerpfad entgegen und löst alle enthaltenen Shard-Pfade auf.
# Sie prüft, ob die Dateien existieren, und gibt eine Liste von Path-Objekten der .array_record-Dateien zurück.
def resolve_arrayrecord_paths(path: str | Path) -> list[Path]:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ARRAY_RECORD_SUFFIX:
            raise ValueError(
                f"Expected an ArrayRecord shard ({ARRAY_RECORD_SUFFIX}) "
                f"or dataset directory, got: {path}"
            )
        return [path]

    metadata = load_arrayrecord_metadata(path)
    shard_paths = [path / rel for rel in metadata["shard_paths"]]
    if not shard_paths:
        raise ValueError(f"No ArrayRecord shards found under: {path}")
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        raise ValueError(f"Missing ArrayRecord shard(s): {missing}")
    return shard_paths


# Die Funktion nimmt den Pfad eines Index-Ordners entgegen und liest alle JSON-Indexeinträge.
# Sie liefert als Iterator nacheinander Tupel aus Record-Index und dem geparsten Dictionary.
def iter_json_arrayrecord_records(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    source = grain.sources.ArrayRecordDataSource([str(p) for p in resolve_arrayrecord_paths(path)])
    for record_idx in range(len(source)):
        yield record_idx, json.loads(source[record_idx])


class DocChainReader:
    # Der Konstruktor nimmt Datenquellen-Pfade entgegen, validiert deren Metadaten und bereitet die Shard-Auflösung vor.
    # Er initialisiert die internen Datenquellen-Objekte.
    def __init__(self, sources: str | Path | Sequence[str | Path]) -> None:
        self.source_paths = _normalize_sources(sources)
        for path in self.source_paths:
            if path.is_dir():
                load_doc_chain_metadata(path)
        self._shard_paths = [resolve_arrayrecord_paths(path) for path in self.source_paths]
        self._sources: list[grain.sources.ArrayRecordDataSource | None] = [
            None for _ in self.source_paths
        ]

    # Die Methode gibt die Anzahl der registrierten Datenquellen (Pfade) als Integer zurück.
    @property
    def num_sources(self) -> int:
        return len(self.source_paths)

    # Die Methode nimmt einen Datenquellen-Index entgegen und lädt die entsprechende ArrayRecordDataSource bei Bedarf in den Cache.
    # Sie gibt das geladene Datenquellen-Objekt zurück.
    def _source(self, source_idx: int) -> grain.sources.ArrayRecordDataSource:
        if source_idx < 0 or source_idx >= len(self.source_paths):
            raise IndexError(f"source_idx out of range: {source_idx}")
        source = self._sources[source_idx]
        if source is None:
            source = grain.sources.ArrayRecordDataSource(
                [str(path) for path in self._shard_paths[source_idx]]
            )
            self._sources[source_idx] = source
        return source

    # Die Methode nimmt einen Datenquellen-Index entgegen und ermittelt die Gesamtzahl der darin enthaltenen Records.
    # Sie gibt diese Anzahl als Integer zurück.
    def num_records(self, source_idx: int) -> int:
        return len(self._source(source_idx))

    # Die Methode nimmt einen Datenquellen- und Record-Index entgegen, liest die rohen Bytes aus dem Shard und deserialisiert diese.
    # Sie gibt das rekonstruierte DocChainRecord zurück.
    def read(self, source_idx: int, record_idx: int) -> DocChainRecord:
        source = self._source(source_idx)
        return deserialize_doc_chain(source[record_idx])

    # Die Methode iteriert über alle Records aller registrierten Datenquellen.
    # Sie liefert nacheinander Tupel aus Datenquellen-Index, Record-Index und dem gelesenen DocChainRecord zurück.
    def iter_records(self) -> Iterator[tuple[int, int, DocChainRecord]]:
        for source_idx in range(self.num_sources):
            source = self._source(source_idx)
            for record_idx in range(len(source)):
                yield source_idx, record_idx, deserialize_doc_chain(source[record_idx])


# Die Funktion nimmt Datenquellen-Pfade entgegen, instanziiert einen Reader und iteriert über das gesamte Dataset.
# Sie liefert nacheinander die Dokumenten-Indizes und DocChainRecord-Objekte als Iterator zurück.
def iter_doc_chain_records(
    sources: str | Path | Sequence[str | Path],
) -> Iterator[tuple[int, int, DocChainRecord]]:
    yield from DocChainReader(sources).iter_records()


# Die Funktion nimmt Datenquellen-Pfade entgegen, liest die Dokumente ein und extrahiert deren grundlegende Kennzahlen.
# Sie gibt einen Iterator von DocumentRef-Objekten zurück.
def iter_document_refs(sources: str | Path | Sequence[str | Path]) -> Iterator[DocumentRef]:
    for source_idx, record_idx, doc in iter_doc_chain_records(sources):
        yield DocumentRef(
            source_idx=source_idx,
            record_idx=record_idx,
            doc_id=doc.doc_id,
            doc_token_count=doc.doc_token_count,
        )


# Die Funktion nimmt ein Dokument sowie die gewünschte Segmentlänge entgegen und berechnet die Start- und Endpunkte der Chunks.
# Sie liefert nacheinander die berechneten DocChunkRef-Objekten als Iterator zurück.
def iter_document_chunks(
    doc: DocChainRecord,
    *,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    source_idx: int = 0,
    record_idx: int = 0,
) -> Iterator[DocChunkRef]:
    if segment_length <= 0:
        raise ValueError("segment_length must be > 0")
    if doc.doc_token_count <= 0:
        return

    num_chunks = (doc.doc_token_count + segment_length - 1) // segment_length
    for chunk_idx in range(num_chunks):
        start = chunk_idx * segment_length
        end = min(start + segment_length, doc.doc_token_count)
        yield DocChunkRef(
            source_idx=source_idx,
            record_idx=record_idx,
            doc_id=doc.doc_id,
            chunk_idx=chunk_idx,
            start=start,
            end=end,
            is_last_chunk=chunk_idx == num_chunks - 1,
        )


# Die Funktion nimmt ein Dokument und eine Segmentlänge entgegen, berechnet die Chunk-Grenzbereiche und sammelt diese in einer Liste.
# Sie gibt eine Liste von DocChunkRef-Objekten zurück.
def chunk_document(
    doc: DocChainRecord,
    *,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    source_idx: int = 0,
    record_idx: int = 0,
) -> list[DocChunkRef]:
    return list(
        iter_document_chunks(
            doc,
            segment_length=segment_length,
            source_idx=source_idx,
            record_idx=record_idx,
        )
    )


# Die Funktion nimmt Datenquellen-Pfade und die Segmentlänge entgegen und zerlegt alle enthaltenen Dokumente sequenziell in Chunks.
# Sie gibt einen Iterator von DocChunkRef-Objekten zurück.
def iter_dataset_chunks(
    sources: str | Path | Sequence[str | Path],
    *,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
) -> Iterator[DocChunkRef]:
    for source_idx, record_idx, doc in iter_doc_chain_records(sources):
        yield from iter_document_chunks(
            doc,
            segment_length=segment_length,
            source_idx=source_idx,
            record_idx=record_idx,
        )


# Die Funktion nimmt ein Dokument, die Segmentlänge und optionale EOS-Angaben entgegen und berechnet zusammenhängende 2-Segment-Paare für das State-Passing.
# Sie gibt nacheinander die berechneten DocPairRef-Objekte als Iterator zurück.
def iter_document_pair_refs(
    doc: DocChainRecord,
    *,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    source_idx: int = 0,
    record_idx: int = 0,
    eos_id: int | None = None,
) -> Iterator[DocPairRef]:
    """Yield retained 2-segment windows for statepassing pretraining."""

    if segment_length <= 0:
        raise ValueError("segment_length must be > 0")
    if doc.doc_token_count <= segment_length:
        return

    pair_length = 2 * segment_length
    num_full_pairs, tail = divmod(doc.doc_token_count, pair_length)
    keep_tail_pair = tail > segment_length
    num_pairs = num_full_pairs + int(keep_tail_pair)
    if num_pairs == 0:
        return

    retained_end = num_full_pairs * pair_length
    if keep_tail_pair:
        retained_end += tail

    eos_token_idx = None
    if (
        eos_id is not None
        and tail > 0
        and tail <= segment_length
        and retained_end > 0
        and int(doc.token_ids[-1]) == int(eos_id)
    ):
        eos_token_idx = retained_end - 1

    for pair_idx in range(num_pairs):
        start = pair_idx * pair_length
        end = min(start + pair_length, doc.doc_token_count)
        pair_eos_idx = (
            eos_token_idx if eos_token_idx is not None and start <= eos_token_idx < end else None
        )
        yield DocPairRef(
            source_idx=source_idx,
            record_idx=record_idx,
            doc_id=doc.doc_id,
            pair_idx=pair_idx,
            start=start,
            mid=start + segment_length,
            end=end,
            doc_token_count=doc.doc_token_count,
            eos_token_idx=pair_eos_idx,
        )


# Die Funktion nimmt Quellpfade, Segmentlänge und optionale EOS-Daten entgegen und berechnet die Paare für alle Dokumente des Datensatzes.
# Sie liefert nacheinander die DocPairRef-Objekte als Iterator zurück.
def iter_dataset_pair_refs(
    sources: str | Path | Sequence[str | Path],
    *,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    eos_id: int | None = None,
) -> Iterator[DocPairRef]:
    for source_idx, record_idx, doc in iter_doc_chain_records(sources):
        yield from iter_document_pair_refs(
            doc,
            segment_length=segment_length,
            source_idx=source_idx,
            record_idx=record_idx,
            eos_id=eos_id,
        )


# Die Funktion nimmt Token-IDs sowie einen Offset-Bereich entgegen, schneidet den Bereich aus und paddet ihn auf die feste Länge.
# Sie gibt ein Dictionary mit den Trainings-Arrays und deren Masken zurück.
def build_chunk_arrays(
    token_ids: Sequence[int] | np.ndarray,
    *,
    start: int = 0,
    end: int | None = None,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    pad_id: int = DEFAULT_PAD_ID,
    eos_id: int | None = None,
    eos_token_idx: int | None = None,
) -> dict[str, np.ndarray]:
    if segment_length <= 0:
        raise ValueError("segment_length must be > 0")

    ids = np.asarray(token_ids, dtype=np.int32)
    end = int(ids.shape[0] if end is None else end)
    start = int(start)
    if start < 0 or end < start or end > ids.shape[0]:
        raise ValueError(f"Invalid chunk range start={start}, end={end}, length={ids.shape[0]}")

    real_length = end - start
    if real_length > segment_length:
        raise ValueError(f"Chunk length {real_length} exceeds segment_length={segment_length}")

    token_ids_T = np.full((segment_length,), int(pad_id), dtype=np.int32)
    attention_mask_T = np.zeros((segment_length,), dtype=np.int32)
    if real_length:
        token_ids_T[:real_length] = ids[start:end]
        if eos_token_idx is not None:
            if eos_id is None:
                raise ValueError("eos_id is required when eos_token_idx is set")
            eos_token_idx = int(eos_token_idx)
            if eos_token_idx < start or eos_token_idx >= end:
                raise ValueError(
                    f"eos_token_idx={eos_token_idx} is outside chunk range [{start}, {end})"
                )
            token_ids_T[eos_token_idx - start] = int(eos_id)
        attention_mask_T[:real_length] = 1

    return {
        "token_ids_T": token_ids_T,
        "attention_mask_T": attention_mask_T,
        "loss_mask_T": attention_mask_T.copy(),
    }


# Die Funktion nimmt Token-IDs und eine Paar-Referenz entgegen, verarbeitet beide Segmente des Paars separat und heftet sie zusammen.
# Sie gibt ein Dictionary mit 2D-Arrays und Zustands-Flags wie reset_state_S zurück.
def build_pair_arrays(
    token_ids: Sequence[int] | np.ndarray,
    pair: DocPairRef,
    *,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    pad_id: int = DEFAULT_PAD_ID,
    eos_id: int | None = None,
) -> dict[str, np.ndarray]:
    token_ids_ST = []
    attention_mask_ST = []
    loss_mask_ST = []
    chunk_idx_S = []
    is_last_chunk_S = []

    for segment_in_pair, start, end, eos_token_idx in _iter_pair_segments(pair):
        arrays = build_chunk_arrays(
            token_ids,
            start=start,
            end=end,
            segment_length=segment_length,
            pad_id=pad_id,
            eos_id=eos_id,
            eos_token_idx=eos_token_idx,
        )
        token_ids_ST.append(arrays["token_ids_T"])
        attention_mask_ST.append(arrays["attention_mask_T"])
        loss_mask_ST.append(arrays["loss_mask_T"])
        chunk_idx_S.append(pair.pair_idx * 2 + segment_in_pair)
        is_last_chunk_S.append(end >= pair.doc_token_count)

    return {
        "token_ids_ST": np.stack(token_ids_ST).astype(np.int32),
        "attention_mask_ST": np.stack(attention_mask_ST).astype(np.int32),
        "loss_mask_ST": np.stack(loss_mask_ST).astype(np.int32),
        "chunk_idx_S": np.asarray(chunk_idx_S, dtype=np.int32),
        "reset_state_S": np.asarray([True, False], dtype=np.bool_),
        "is_last_chunk_S": np.asarray(is_last_chunk_S, dtype=np.bool_),
    }


# Die Funktion nimmt eine Paar-Referenz entgegen, berechnet die genauen Start- und Endpositionen beider Segmente und ermittelt eventuell vorhandene EOS-Indizes.
# Sie liefert nacheinander diese Positionsdaten als Iterator zurück.
def _iter_pair_segments(pair: DocPairRef) -> Iterator[tuple[int, int, int, int | None]]:
    for segment_in_pair, (start, end) in enumerate(
        ((pair.start, min(pair.mid, pair.end)), (pair.mid, pair.end))
    ):
        eos_token_idx = (
            pair.eos_token_idx
            if pair.eos_token_idx is not None and start <= pair.eos_token_idx < end
            else None
        )
        yield segment_in_pair, start, end, eos_token_idx


# Die Funktion nimmt eine Paar-Referenz entgegen und zerlegt sie in zwei einzelne, flache Chunk-Referenzen mit passenden Indices.
# Sie gibt eine Liste von Dictionaries mit diesen Chunk-Daten zurück.
def flatten_pair_ref(pair: DocPairRef) -> list[dict[str, Any]]:
    return [
        {
            "source_idx": pair.source_idx,
            "record_idx": pair.record_idx,
            "doc_id": pair.doc_id,
            "pair_idx": pair.pair_idx,
            "segment_in_pair": segment_in_pair,
            "chunk_idx": pair.pair_idx * 2 + segment_in_pair,
            "start": start,
            "end": end,
            "eos_token_idx": eos_token_idx,
        }
        for segment_in_pair, start, end, eos_token_idx in _iter_pair_segments(pair)
    ]


# Die Funktion nimmt eine Paar-Referenz entgegen und konvertiert sie in ein einfaches Python-Dictionary, das sich leicht JSON-serialisieren lässt.
# Sie gibt dieses Dictionary zurück.
def pair_ref_to_record(pair: DocPairRef) -> dict[str, Any]:
    return {
        "source_idx": pair.source_idx,
        "record_idx": pair.record_idx,
        "doc_id": pair.doc_id,
        "pair_idx": pair.pair_idx,
        "start": pair.start,
        "mid": pair.mid,
        "end": pair.end,
        "doc_token_count": pair.doc_token_count,
        "eos_token_idx": pair.eos_token_idx,
    }


# Die Funktion nimmt ein Batch-Dictionary entgegen und entfernt daraus den Metadaten-Key, der für das Training nicht als Array benötigt wird.
# Sie gibt die extrahierten Metadaten zurück und modifiziert das Batch-Dictionary direkt.
def pop_pretrain_metadata(batch: dict[str, Any]) -> dict[str, Any] | None:
    """Remove non-array pretraining metadata before batch sharding.

    IID and statepassing iterators surface document ids and source record
    positions under ``metadata`` for logging/debugging. The JAX sharding helpers
    expect every top-level batch value to be an array, so callers should pop this
    field before passing a batch to ``shard_batch_dict``.
    """

    raw = batch.pop(BATCH_PRETRAIN_METADATA_KEY, None)
    return dict(raw) if raw is not None else None


# Die Funktion nimmt die Konfiguration der verteilten Umgebung entgegen und ermittelt die effektive Partitionsgröße sowie den lokalen Shard-Index.
# Sie gibt ein Tupel aus Gesamt-Datenparallelismus und GPU-ID zurück.
def resolve_pretrain_dp(
    *,
    dp_size: int,
    fsdp_size: int = 1,
    process_index: int | None = None,
) -> tuple[int, int]:
    """Resolve effective data-shard count and local shard index.

    This mirrors the existing Grain pipeline convention: data loading is
    sharded across the product of data-parallel and FSDP axes.
    """

    if dp_size <= 0:
        raise ValueError("dp_size must be > 0")
    if fsdp_size <= 0:
        raise ValueError("fsdp_size must be > 0")

    effective_dp_size = int(dp_size) * int(fsdp_size)
    if process_index is None:
        import jax

        process_index = int(jax.process_index())
    if process_index < 0:
        raise ValueError("process_index must be >= 0")
    return effective_dp_size, int(process_index) % effective_dp_size
