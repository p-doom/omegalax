"""IID-style chunk indexing and iteration for pretraining documents."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain
from pathlib import Path
from typing import Any

import grain
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

from omegalax.data.pretrain_doc_chain import (
    DEFAULT_PAD_ID,
    DEFAULT_SEGMENT_LENGTH,
    BATCH_PRETRAIN_METADATA_KEY,
    ARRAY_RECORD_SUFFIX,
    COMPILED_METADATA_FILENAME,
    DOC_CHAIN_DATASET_VERSION,
    DocChainReader,
    build_chunk_arrays,
    flatten_pair_ref,
    iter_document_pair_refs,
    load_arrayrecord_metadata,
    resolve_arrayrecord_paths,
    resolve_pretrain_dp,
)

IID_CHUNK_INDEX_FORMAT = "omegalax_pretrain_iid_chunk_index_v1"


# Die Funktion nimmt Quellpfade, einen Ausgabeordner und Chunk-Konfigurationen entgegen.
# Sie liest die Quelldokumente ein, zerlegt sie in Segment-Paare, flacht diese zu einzelnen Chunk-Referenzen also quasi ein inhaltsverzeichnis für sammples ab und speichert sie als ArrayRecord-Dataset ab.
# Sie gibt den Pfad des Ausgabeordners zurück.
def build_iid_chunk_index(
    sources: str | Path | Sequence[str | Path],
    out_dir: str | Path,
    *,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    eos_id: int | None = None,
    records_per_shard: int = 100_000,
    overwrite: bool = False,
) -> Path:
    if segment_length <= 0:
        raise ValueError("segment_length must be > 0")

    reader = DocChainReader(sources)
    dynamic_metadata: dict[str, Any] = {
        "format": IID_CHUNK_INDEX_FORMAT,
        "source_paths": [str(path) for path in reader.source_paths],
        "segment_length": int(segment_length),
        "eos_id": eos_id,
        "num_chunks": 0,
        "num_pairs": 0,
        "num_source_records": 0,
        "source_record_counts": [],
    }

    # Die Hilfsfunktion nimmt keine Argumente entgegen.
    # Sie durchläuft alle Originaldokumente, zählt die verarbeiteten Records und generiert flache Chunk-Referenzen.
    # Sie liefert die einzelnen Chunk-Referenzen nacheinander als Iterator zurück.
    def _iter_index_records() -> Iterator[dict[str, Any]]:
        source_record_counts = [0 for _ in reader.source_paths]
        num_chunks = 0
        num_pairs = 0
        for source_idx, record_idx, doc in reader.iter_records():
            source_record_counts[source_idx] += 1
            for pair in iter_document_pair_refs(
                doc,
                segment_length=segment_length,
                source_idx=source_idx,
                record_idx=record_idx,
                eos_id=eos_id,
            ):
                num_pairs += 1
                for chunk in flatten_pair_ref(pair):
                    num_chunks += 1
                    yield chunk
        dynamic_metadata["num_chunks"] = num_chunks
        dynamic_metadata["num_pairs"] = num_pairs
        dynamic_metadata["num_source_records"] = sum(source_record_counts)
        dynamic_metadata["source_record_counts"] = source_record_counts

    return _write_json_arrayrecord_dataset(
        _iter_index_records(),
        out_dir,
        records_per_shard=records_per_shard,
        overwrite=overwrite,
        metadata=dynamic_metadata,
    )


# Die Funktion nimmt einen Iterator von Datensätzen, einen Ausgabeordner und Shard-Einstellungen entgegen.
# Sie serialisiert die Datensätze nacheinander als JSON in durchnummerierte ArrayRecord-Dateien und schreibt eine zentrale Metadatendatei.
# Sie gibt den Pfad des erstellten Ausgabeordners zurück.
def _write_json_arrayrecord_dataset(
    records: Iterable[dict[str, Any]],
    out_dir: str | Path,
    *,
    records_per_shard: int,
    overwrite: bool,
    metadata: dict[str, Any],
) -> Path:
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be > 0")

    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.exists():
        has_contents = any(out_dir.iterdir())
        if has_contents and not overwrite:
            raise ValueError(f"Refusing to overwrite non-empty output directory: {out_dir}")
        if has_contents and overwrite:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[str] = []
    records_in_shard = 0
    total_records = 0
    shard_idx = 0

    # Die Hilfsfunktion nimmt eine Shard-Indexnummer entgegen.
    # Sie baut den passenden Dateinamen für den Shard zusammen und öffnet einen neuen ArrayRecordWriter.
    # Sie gibt das erstellte Writer-Objekt zurück.
    def _open_writer(next_idx: int) -> ArrayRecordWriter:
        shard_path = out_dir / f"part-{next_idx:05d}{ARRAY_RECORD_SUFFIX}"
        shard_paths.append(shard_path.name)
        return ArrayRecordWriter(str(shard_path), "group_size:1")

    records_iter = iter(records)
    try:
        first_record = next(records_iter)
    except StopIteration:
        raise ValueError("No records were written") from None

    writer = _open_writer(shard_idx)
    try:
        for record in chain((first_record,), records_iter):
            if records_in_shard >= records_per_shard:
                writer.close()
                shard_idx += 1
                writer = _open_writer(shard_idx)
                records_in_shard = 0
            writer.write(json.dumps(record, sort_keys=True).encode("utf-8"))
            records_in_shard += 1
            total_records += 1
    finally:
        writer.close()

    final_metadata = dict(metadata)
    final_metadata.update(
        {
            "version": DOC_CHAIN_DATASET_VERSION,
            "num_records": total_records,
            "num_shards": len(shard_paths),
            "shard_paths": shard_paths,
        }
    )
    (out_dir / COMPILED_METADATA_FILENAME).write_text(
        json.dumps(final_metadata, indent=2) + "\n"
    )
    return out_dir


# Die Funktion nimmt den Pfad des Index-Ordners und die erwartete Segmentlänge entgegen.
# Sie lädt und überprüft die Metadatendatei des Index auf Formatkompatibilität und korrekte Segmentlänge.
# Sie gibt das geladene Metadaten-Dictionary zurück.
def _load_index_metadata(index_path: str | Path, segment_length: int | None) -> dict[str, Any]:
    metadata = load_arrayrecord_metadata(index_path)
    fmt = metadata.get("format")
    if fmt != IID_CHUNK_INDEX_FORMAT:
        raise ValueError(f"Expected {IID_CHUNK_INDEX_FORMAT} dataset, got format={fmt}")
    index_segment_length = int(metadata["segment_length"])
    if segment_length is not None and int(segment_length) != index_segment_length:
        raise ValueError(
            f"segment_length mismatch: index has {index_segment_length}, "
            f"loader got {segment_length}"
        )
    return metadata


# Die Funktion nimmt eine Liste von Indexeinträgen, einen Reader und Formatierungsparameter entgegen.
# Sie lädt die rohen Token-IDs der Einträge über den Reader, schneidet und paddet diese auf die Segmentlänge und stapelt sie zu Batches zusammen.
# Sie gibt ein Dictionary mit den fertigen Batch-Arrays und Metadaten zurück.
def _make_batch(
    entries: Sequence[dict[str, Any]],
    *,
    reader: DocChainReader,
    segment_length: int,
    pad_id: int,
    eos_id: int | None,
) -> dict[str, Any]:
    token_ids = []
    attention_masks = []
    loss_masks = []
    chunk_indices = []
    doc_ids = []
    source_indices = []
    record_indices = []
    pair_indices = []
    segments_in_pair = []
    doc_cache = {}

    for entry in entries:
        source_idx = int(entry["source_idx"])
        record_idx = int(entry["record_idx"])
        doc_key = (source_idx, record_idx)
        doc = doc_cache.get(doc_key)
        if doc is None:
            doc = reader.read(source_idx, record_idx)
            doc_cache[doc_key] = doc
        arrays = build_chunk_arrays(
            doc.token_ids,
            start=int(entry["start"]),
            end=int(entry["end"]),
            segment_length=segment_length,
            pad_id=pad_id,
            eos_id=eos_id,
            eos_token_idx=entry.get("eos_token_idx"),
        )
        token_ids.append(arrays["token_ids_T"])
        attention_masks.append(arrays["attention_mask_T"])
        loss_masks.append(arrays["loss_mask_T"])
        chunk_indices.append(int(entry["chunk_idx"]))
        doc_ids.append(str(entry["doc_id"]))
        source_indices.append(source_idx)
        record_indices.append(record_idx)
        pair_indices.append(int(entry["pair_idx"]))
        segments_in_pair.append(int(entry["segment_in_pair"]))

    return {
        "token_ids_BT": np.stack(token_ids).astype(np.int32),
        "attention_mask_BT": np.stack(attention_masks).astype(np.int32),
        "loss_mask_BT": np.stack(loss_masks).astype(np.int32),
        "chunk_idx_B": np.asarray(chunk_indices, dtype=np.int32),
        BATCH_PRETRAIN_METADATA_KEY: {
            "doc_ids": doc_ids,
            "source_idx_B": np.asarray(source_indices, dtype=np.int32),
            "record_idx_B": np.asarray(record_indices, dtype=np.int32),
            "pair_idx_B": np.asarray(pair_indices, dtype=np.int32),
            "segment_in_pair_B": np.asarray(segments_in_pair, dtype=np.int32),
        },
    }


class IIDPretrainIterator:
    # Der Konstruktor nimmt den Pfad des Index-Ordners, die Batch-Größe und Konfigurationen für Sharding/Shuffling entgegen.
    # Er lädt den Index sowie die Dokumenten-Datenquellen und berechnet die anfängliche Daten-Reihenfolge für den lokalen Shard.
    # Er initialisiert die Iterationsvariablen, gibt aber keinen Wert zurück.
    def __init__(
        self,
        index_path: str | Path,
        *,
        batch_size: int,
        segment_length: int | None = DEFAULT_SEGMENT_LENGTH,
        pad_id: int = DEFAULT_PAD_ID,
        eos_id: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        num_epochs: int | None = None,
        dp_size: int = 1,
        fsdp_size: int = 1,
        dp_index: int | None = None,
        process_index: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        effective_dp_size, resolved_dp_index = resolve_pretrain_dp(
            dp_size=dp_size,
            fsdp_size=fsdp_size,
            process_index=process_index,
        )
        if dp_index is not None:
            resolved_dp_index = int(dp_index)
        if resolved_dp_index < 0 or resolved_dp_index >= effective_dp_size:
            raise ValueError(
                f"dp_index must be in [0, {effective_dp_size}), got {resolved_dp_index}"
            )

        self.index_path = Path(index_path).expanduser().resolve()
        self.metadata = _load_index_metadata(self.index_path, segment_length)
        self.segment_length = int(self.metadata["segment_length"])
        index_eos_id = self.metadata.get("eos_id")
        if eos_id != index_eos_id:
            raise ValueError(
                f"eos_id mismatch: index has {index_eos_id}, loader got {eos_id}"
            )
        self.batch_size = int(batch_size)
        self.pad_id = int(pad_id)
        self.eos_id = eos_id
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_epochs = num_epochs
        self.dp_size = int(effective_dp_size)
        self.dp_index = int(resolved_dp_index)

        self.reader = DocChainReader(self.metadata["source_paths"])
        self.index_source = grain.sources.ArrayRecordDataSource(
            [str(path) for path in resolve_arrayrecord_paths(self.index_path)]
        )
        self.num_records = len(self.index_source)
        if self.num_records == 0:
            raise ValueError(f"IID chunk index has no records: {self.index_path}")

        self.epoch = 0
        self.order: list[int] = []
        self.order_pos = 0
        self._reset_epoch_order()

    # Die Methode nimmt keine weiteren Argumente entgegen.
    # Sie registriert das Iterator-Objekt selbst für die Python-Schleife.
    # Sie gibt das Iterator-Objekt selbst zurück.
    def __iter__(self) -> "IIDPretrainIterator":
        return self

    # Die Methode nimmt keine Argumente entgegen.
    # Sie generiert eine Liste aller Chunk-Indizes, mischt diese optional deterministisch pro Epoche und teilt sie dem aktuellen Shard-Index zu.
    # Sie aktualisiert die interne Reihenfolge und gibt keinen Wert zurück.
    def _reset_epoch_order(self) -> None:
        order = np.arange(self.num_records)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(order)
        self.order = [int(i) for i in order[self.dp_index :: self.dp_size].tolist()]
        self.order_pos = 0
        if not self.order:
            raise ValueError(
                f"No IID chunks assigned to dp_index={self.dp_index} "
                f"with dp_size={self.dp_size}"
            )

    # Die Methode nimmt keine Argumente entgegen.
    # Sie erhöht die Epochen-Nummer um eins und setzt die Reihenfolge der Indizes für die neue Epoche neu auf.
    # Sie gibt True zurück, wenn eine weitere Epoche erlaubt ist, andernfalls False.
    def _advance_epoch(self) -> bool:
        if self.num_epochs is not None and self.epoch + 1 >= self.num_epochs:
            return False
        self.epoch += 1
        self._reset_epoch_order()
        return True

    # Die Methode nimmt keine Argumente entgegen.
    # Sie sammelt die nächsten Indexeinträge entsprechend der aktuellen Epochenreihenfolge auf und erstellt daraus das nächste Batch.
    # Sie gibt das fertig zusammengestellte Batch-Dictionary zurück oder wirft StopIteration am Ende des Datensatzes.
    def __next__(self) -> dict[str, Any]:
        batch_entries: list[dict[str, Any]] = []
        while len(batch_entries) < self.batch_size:
            if self.order_pos >= len(self.order) and not self._advance_epoch():
                raise StopIteration
            index_record_idx = self.order[self.order_pos]
            self.order_pos += 1
            batch_entries.append(json.loads(self.index_source[index_record_idx]))

        return _make_batch(
            batch_entries,
            reader=self.reader,
            segment_length=self.segment_length,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
        )

    # Die Methode nimmt keine Argumente entgegen.
    # Sie sammelt alle zustandsrelevanten Variablen wie Epoche, Index-Reihenfolge und die aktuelle Position.
    # Sie gibt diesen Zustand als serialisierbares Dictionary zurück.
    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "index_path": str(self.index_path),
            "batch_size": self.batch_size,
            "segment_length": self.segment_length,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "num_epochs": self.num_epochs,
            "dp_size": self.dp_size,
            "dp_index": self.dp_index,
            "epoch": self.epoch,
            "order": list(self.order),
            "order_pos": self.order_pos,
        }

    # Die Methode nimmt ein Zustands-Dictionary entgegen.
    # Sie validiert die Kompatibilität des gespeicherten Zustands mit dem aktuellen Iterator und stellt die Iterationsposition wieder her.
    # Sie modifiziert den internen Zustand direkt und gibt keinen Wert zurück.
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 0)) != 1:
            raise ValueError(f"Unsupported IID iterator state version: {state.get('version')}")
        if str(state["index_path"]) != str(self.index_path):
            raise ValueError(
                f"IID iterator state is for {state['index_path']}, not {self.index_path}"
            )
        expected = {
            "batch_size": self.batch_size,
            "segment_length": self.segment_length,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "num_epochs": self.num_epochs,
            "dp_size": self.dp_size,
            "dp_index": self.dp_index,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"IID iterator state mismatch for {key}: "
                    f"state={state.get(key)!r}, iterator={value!r}"
                )
        self.epoch = int(state["epoch"])
        self.order = [int(i) for i in state["order"]]
        self.order_pos = int(state["order_pos"])


# Die Funktion nimmt den Pfad des Index-Ordners sowie Batch- und Sharding-Parameter entgegen.
# Sie instanziiert den IID-Iterator mit den übergebenen Einstellungen.
# Sie gibt das erstellte IIDPretrainIterator-Objekt zurück.
def make_iid_iterator(
    index_path: str | Path,
    *,
    batch_size: int,
    segment_length: int | None = DEFAULT_SEGMENT_LENGTH,
    pad_id: int = DEFAULT_PAD_ID,
    eos_id: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    num_epochs: int | None = None,
    dp_size: int = 1,
    fsdp_size: int = 1,
    dp_index: int | None = None,
    process_index: int | None = None,
) -> IIDPretrainIterator:
    return IIDPretrainIterator(
        index_path,
        batch_size=batch_size,
        segment_length=segment_length,
        pad_id=pad_id,
        eos_id=eos_id,
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
        dp_size=dp_size,
        fsdp_size=fsdp_size,
        dp_index=dp_index,
        process_index=process_index,
    )

