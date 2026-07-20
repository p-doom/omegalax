"""Deterministic full-document manifests for state-usage evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omegalax.data.pretrain_data_set import (
    DOC_CHAIN_FORMAT,
    DataSetReader,
    WindowMetadata,
    build_window_arrays,
)

FULL_DOCUMENT_MANIFEST_FORMAT = "omegalax_full_document_eval_manifest_v1"
FULL_DOCUMENT_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ManifestDocument:
    bucket_idx: int
    bucket_name: str
    record_idx: int
    doc_id: str
    doc_token_count: int
    doc_num_chunks: int
    sample_rank: int
    donor_bucket_idx: int
    donor_record_idx: int
    donor_doc_id: str


@dataclass(frozen=True)
class ManifestLengthCount:
    doc_num_chunks: int
    available: int
    selected: int


@dataclass(frozen=True)
class FullDocumentManifest:
    format: str
    version: int
    dataset_root: str
    split: str
    chunk_length: int
    seed: int
    sample_cap: int
    min_doc_chunks: int
    max_doc_chunks: int
    bucket_names: tuple[str, ...]
    dataset_hash: str
    counts_by_length: tuple[ManifestLengthCount, ...]
    documents: tuple[ManifestDocument, ...]
    manifest_hash: str


@dataclass(frozen=True, eq=False)
class LoadedDocument:
    doc_id: str
    doc_token_count: int
    doc_num_chunks: int
    token_ids_CT: np.ndarray
    attention_mask_CT: np.ndarray
    loss_mask_CT: np.ndarray
    chunk_idx_C: np.ndarray
    reset_state_C: np.ndarray


@dataclass(frozen=True)
class _SourceDocument:
    bucket_idx: int
    bucket_name: str
    record_idx: int
    doc_id: str
    doc_token_count: int
    doc_num_chunks: int


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_hash(payload: dict[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def _ranking_key(doc: _SourceDocument, seed: int) -> tuple[bytes, int, int]:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "seed": int(seed),
                "bucket_name": doc.bucket_name,
                "record_idx": doc.record_idx,
                "doc_id": doc.doc_id,
                "doc_num_chunks": doc.doc_num_chunks,
            }
        )
    ).digest()
    return digest, doc.bucket_idx, doc.record_idx


def _donor_indices(num_documents: int) -> list[int]:
    donor_indices = list(range(num_documents))
    paired_end = num_documents if num_documents % 2 == 0 else num_documents - 3
    for idx in range(0, paired_end, 2):
        donor_indices[idx] = idx + 1
        donor_indices[idx + 1] = idx
    if num_documents % 2:
        donor_indices[-3] = num_documents - 2
        donor_indices[-2] = num_documents - 1
        donor_indices[-1] = num_documents - 3
    return donor_indices


def _manifest_payload(manifest: FullDocumentManifest) -> dict[str, Any]:
    return {
        "format": manifest.format,
        "version": manifest.version,
        "dataset_root": manifest.dataset_root,
        "split": manifest.split,
        "chunk_length": manifest.chunk_length,
        "seed": manifest.seed,
        "sample_cap": manifest.sample_cap,
        "min_doc_chunks": manifest.min_doc_chunks,
        "max_doc_chunks": manifest.max_doc_chunks,
        "bucket_names": list(manifest.bucket_names),
        "dataset_hash": manifest.dataset_hash,
        "counts_by_length": [asdict(count) for count in manifest.counts_by_length],
        "documents": [asdict(document) for document in manifest.documents],
    }


def build_full_document_manifest(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    split: str = "val",
    chunk_length: int = 2048,
    seed: int = 0,
    sample_cap: int = 500,
    min_doc_chunks: int = 2,
    max_doc_chunks: int = 16,
) -> FullDocumentManifest:
    """Build and persist one deterministic, length-stratified document manifest."""

    if chunk_length <= 0:
        raise ValueError("chunk_length must be > 0")
    if sample_cap < 2:
        raise ValueError("sample_cap must be >= 2 so every document has a donor")
    if min_doc_chunks < 2:
        raise ValueError("min_doc_chunks must be >= 2")
    if max_doc_chunks < min_doc_chunks:
        raise ValueError("max_doc_chunks must be >= min_doc_chunks")
    if split != "val":
        raise ValueError("Only split='val' is supported for full-document eval manifests")

    dataset_root = Path(dataset_root).expanduser().resolve()
    reader = DataSetReader(dataset_root, split=split)
    candidates = {
        doc_num_chunks: [] for doc_num_chunks in range(min_doc_chunks, max_doc_chunks + 1)
    }
    dataset_hasher = hashlib.sha256()
    dataset_hasher.update(
        _canonical_json(
            {
                "dataset_format": DOC_CHAIN_FORMAT,
                "split": split,
                "chunk_length": int(chunk_length),
                "bucket_names": reader.bucket_names,
            }
        )
    )

    for bucket_idx, record_idx, doc in reader.iter_records():
        if doc.token_ids.ndim != 1 or doc.doc_token_count != doc.token_ids.shape[0]:
            raise ValueError(
                f"Document length mismatch for bucket_idx={bucket_idx}, "
                f"record_idx={record_idx}, doc_id={doc.doc_id}: "
                f"doc_token_count={doc.doc_token_count}, token_count={doc.token_ids.shape}"
            )
        if doc.doc_token_count <= 0:
            raise ValueError(
                f"Document must contain tokens: bucket_idx={bucket_idx}, "
                f"record_idx={record_idx}, doc_id={doc.doc_id}"
            )

        doc_num_chunks = (doc.doc_token_count + chunk_length - 1) // chunk_length
        source_doc = _SourceDocument(
            bucket_idx=bucket_idx,
            bucket_name=reader.bucket_names[bucket_idx],
            record_idx=record_idx,
            doc_id=doc.doc_id,
            doc_token_count=doc.doc_token_count,
            doc_num_chunks=doc_num_chunks,
        )
        dataset_hasher.update(b"\n")
        dataset_hasher.update(_canonical_json(asdict(source_doc)))
        dataset_hasher.update(b"\ntokens:")
        dataset_hasher.update(doc.token_ids.astype("<i4", copy=False).tobytes())
        if min_doc_chunks <= doc_num_chunks <= max_doc_chunks:
            candidates[doc_num_chunks].append(source_doc)

    documents = []
    counts_by_length = []
    for doc_num_chunks in range(min_doc_chunks, max_doc_chunks + 1):
        ranked = sorted(candidates[doc_num_chunks], key=lambda doc: _ranking_key(doc, seed))
        selected = ranked[:sample_cap]
        if len(selected) < 2:
            raise ValueError(
                f"Exact document length L{doc_num_chunks} has {len(selected)} selected "
                "documents; at least 2 are required for donor assignment"
            )
        counts_by_length.append(
            ManifestLengthCount(
                doc_num_chunks=doc_num_chunks,
                available=len(ranked),
                selected=len(selected),
            )
        )
        donor_indices = _donor_indices(len(selected))
        for sample_rank, (doc, donor_idx) in enumerate(zip(selected, donor_indices, strict=True)):
            donor = selected[donor_idx]
            documents.append(
                ManifestDocument(
                    bucket_idx=doc.bucket_idx,
                    bucket_name=doc.bucket_name,
                    record_idx=doc.record_idx,
                    doc_id=doc.doc_id,
                    doc_token_count=doc.doc_token_count,
                    doc_num_chunks=doc.doc_num_chunks,
                    sample_rank=sample_rank,
                    donor_bucket_idx=donor.bucket_idx,
                    donor_record_idx=donor.record_idx,
                    donor_doc_id=donor.doc_id,
                )
            )

    manifest_without_hash = FullDocumentManifest(
        format=FULL_DOCUMENT_MANIFEST_FORMAT,
        version=FULL_DOCUMENT_MANIFEST_VERSION,
        dataset_root=str(dataset_root),
        split=str(split),
        chunk_length=int(chunk_length),
        seed=int(seed),
        sample_cap=int(sample_cap),
        min_doc_chunks=int(min_doc_chunks),
        max_doc_chunks=int(max_doc_chunks),
        bucket_names=tuple(reader.bucket_names),
        dataset_hash=f"sha256:{dataset_hasher.hexdigest()}",
        counts_by_length=tuple(counts_by_length),
        documents=tuple(documents),
        manifest_hash="",
    )
    payload = _manifest_payload(manifest_without_hash)
    payload["manifest_hash"] = _payload_hash(payload)

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return load_full_document_manifest(output_path)


def load_full_document_manifest(path: str | Path) -> FullDocumentManifest:
    """Load a persisted manifest and verify its content hash."""

    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text())
    manifest_hash = raw.pop("manifest_hash", None)
    expected_hash = _payload_hash(raw)
    if manifest_hash != expected_hash:
        raise ValueError(
            f"Full-document manifest hash mismatch for {path}: "
            f"stored={manifest_hash}, computed={expected_hash}"
        )
    if raw.get("format") != FULL_DOCUMENT_MANIFEST_FORMAT:
        raise ValueError(f"Unsupported full-document manifest format: {raw.get('format')}")
    if raw.get("version") != FULL_DOCUMENT_MANIFEST_VERSION:
        raise ValueError(f"Unsupported full-document manifest version: {raw.get('version')}")

    manifest = FullDocumentManifest(
        format=raw["format"],
        version=int(raw["version"]),
        dataset_root=str(raw["dataset_root"]),
        split=str(raw["split"]),
        chunk_length=int(raw["chunk_length"]),
        seed=int(raw["seed"]),
        sample_cap=int(raw["sample_cap"]),
        min_doc_chunks=int(raw["min_doc_chunks"]),
        max_doc_chunks=int(raw["max_doc_chunks"]),
        bucket_names=tuple(str(name) for name in raw["bucket_names"]),
        dataset_hash=str(raw["dataset_hash"]),
        counts_by_length=tuple(ManifestLengthCount(**count) for count in raw["counts_by_length"]),
        documents=tuple(ManifestDocument(**document) for document in raw["documents"]),
        manifest_hash=manifest_hash,
    )
    _validate_manifest(manifest)
    return manifest


def validate_manifest_resume_compatibility(
    existing: FullDocumentManifest,
    requested: FullDocumentManifest,
) -> None:
    """Validate that ``requested`` can reuse every result from ``existing``."""

    if existing.dataset_hash != requested.dataset_hash:
        raise ValueError(
            "Manifest resume dataset_hash mismatch: "
            f"existing={existing.dataset_hash}, requested={requested.dataset_hash}"
        )
    for field in (
        "format",
        "version",
        "split",
        "chunk_length",
        "seed",
        "min_doc_chunks",
        "max_doc_chunks",
        "bucket_names",
    ):
        if getattr(existing, field) != getattr(requested, field):
            label = (
                "document length range" if field in {"min_doc_chunks", "max_doc_chunks"} else field
            )
            raise ValueError(
                f"Manifest resume {label} mismatch: "
                f"existing={getattr(existing, field)!r}, requested={getattr(requested, field)!r}"
            )

    existing_counts = {count.doc_num_chunks: count for count in existing.counts_by_length}
    requested_counts = {count.doc_num_chunks: count for count in requested.counts_by_length}
    if set(existing_counts) != set(requested_counts):
        raise ValueError("Manifest resume document length structure mismatch")
    for length, old_count in existing_counts.items():
        new_count = requested_counts[length]
        if old_count.available != new_count.available:
            raise ValueError(
                f"Manifest resume available-count mismatch for L{length}: "
                f"existing={old_count.available}, requested={new_count.available}"
            )
        if new_count.selected < old_count.selected:
            raise ValueError(
                f"Manifest resume requested sample is smaller for L{length}: "
                f"existing={old_count.selected}, requested={new_count.selected}"
            )

        old_documents = [
            document for document in existing.documents if document.doc_num_chunks == length
        ]
        new_prefix = [
            document for document in requested.documents if document.doc_num_chunks == length
        ][: len(old_documents)]
        if len(new_prefix) != len(old_documents):
            raise ValueError(f"Manifest resume prefix is incomplete for L{length}")
        for old_document, new_document in zip(old_documents, new_prefix, strict=True):
            old_source = (
                old_document.bucket_idx,
                old_document.bucket_name,
                old_document.record_idx,
                old_document.doc_id,
                old_document.doc_token_count,
                old_document.doc_num_chunks,
                old_document.sample_rank,
            )
            new_source = (
                new_document.bucket_idx,
                new_document.bucket_name,
                new_document.record_idx,
                new_document.doc_id,
                new_document.doc_token_count,
                new_document.doc_num_chunks,
                new_document.sample_rank,
            )
            if old_source != new_source:
                raise ValueError(f"Manifest resume document prefix mismatch for L{length}")
            old_donor = (
                old_document.donor_bucket_idx,
                old_document.donor_record_idx,
                old_document.donor_doc_id,
            )
            new_donor = (
                new_document.donor_bucket_idx,
                new_document.donor_record_idx,
                new_document.donor_doc_id,
            )
            if old_donor != new_donor:
                raise ValueError(
                    f"Manifest resume donor assignment changed for document {old_document.doc_id}"
                )


def _validate_manifest(manifest: FullDocumentManifest) -> None:
    if manifest.split != "val":
        raise ValueError("Only split='val' is supported for full-document eval manifests")
    if manifest.chunk_length <= 0 or manifest.sample_cap < 2:
        raise ValueError("Invalid chunk_length or sample_cap in full-document manifest")
    if manifest.min_doc_chunks < 2 or manifest.max_doc_chunks < manifest.min_doc_chunks:
        raise ValueError("Invalid document length range in full-document manifest")

    documents_by_ref = {
        (document.bucket_idx, document.record_idx): document for document in manifest.documents
    }
    if len(documents_by_ref) != len(manifest.documents):
        raise ValueError("Full-document manifest contains duplicate source references")

    expected_lengths = set(range(manifest.min_doc_chunks, manifest.max_doc_chunks + 1))
    counts_by_length = {count.doc_num_chunks: count for count in manifest.counts_by_length}
    if (
        len(counts_by_length) != len(manifest.counts_by_length)
        or set(counts_by_length) != expected_lengths
        or any(document.doc_num_chunks not in expected_lengths for document in manifest.documents)
    ):
        raise ValueError("Invalid manifest counts by exact document length")

    for doc_num_chunks in range(manifest.min_doc_chunks, manifest.max_doc_chunks + 1):
        documents = [
            document for document in manifest.documents if document.doc_num_chunks == doc_num_chunks
        ]
        count = counts_by_length.get(doc_num_chunks)
        if (
            count is None
            or count.selected != len(documents)
            or count.selected != min(manifest.sample_cap, count.available)
            or count.selected < 2
        ):
            raise ValueError(f"Invalid manifest counts for exact document length L{doc_num_chunks}")
        if [document.sample_rank for document in documents] != list(range(len(documents))):
            raise ValueError(f"Invalid sample ranks for exact document length L{doc_num_chunks}")

        for document, donor_idx in zip(documents, _donor_indices(len(documents)), strict=True):
            if not 0 <= document.bucket_idx < len(manifest.bucket_names):
                raise ValueError(f"Invalid bucket_idx for document {document.doc_id}")
            if manifest.bucket_names[document.bucket_idx] != document.bucket_name:
                raise ValueError(f"Bucket name mismatch for document {document.doc_id}")
            expected_donor = documents[donor_idx]
            if (
                document.donor_bucket_idx != expected_donor.bucket_idx
                or document.donor_record_idx != expected_donor.record_idx
                or document.donor_doc_id != expected_donor.doc_id
            ):
                raise ValueError(f"Invalid donor assignment for document {document.doc_id}")


class FullDocumentLoader:
    """Load complete padded document chains referenced by a manifest."""

    def __init__(self, manifest: FullDocumentManifest) -> None:
        self.manifest = manifest
        self.reader = DataSetReader(manifest.dataset_root, split=manifest.split)
        if tuple(self.reader.bucket_names) != manifest.bucket_names:
            raise ValueError(
                "Manifest bucket names do not match the source dataset: "
                f"manifest={manifest.bucket_names}, source={tuple(self.reader.bucket_names)}"
            )
        self._documents_by_ref = {
            (document.bucket_idx, document.record_idx): document for document in manifest.documents
        }

    def load_document(self, entry: ManifestDocument) -> LoadedDocument:
        stored_entry = self._documents_by_ref.get((entry.bucket_idx, entry.record_idx))
        if stored_entry != entry:
            raise ValueError(f"Document is not part of this manifest: {entry.doc_id}")

        doc = self.reader.read(entry.bucket_idx, entry.record_idx)
        if doc.doc_id != entry.doc_id or doc.doc_token_count != entry.doc_token_count:
            raise ValueError(
                f"Manifest/source mismatch for bucket_idx={entry.bucket_idx}, "
                f"record_idx={entry.record_idx}: expected doc_id={entry.doc_id}, "
                f"doc_token_count={entry.doc_token_count}; got doc_id={doc.doc_id}, "
                f"doc_token_count={doc.doc_token_count}"
            )
        if doc.token_ids.ndim != 1 or doc.token_ids.shape[0] != entry.doc_token_count:
            raise ValueError(f"Source token length mismatch for document {entry.doc_id}")

        doc_num_chunks = (
            entry.doc_token_count + self.manifest.chunk_length - 1
        ) // self.manifest.chunk_length
        if doc_num_chunks != entry.doc_num_chunks:
            raise ValueError(f"Manifest chunk length mismatch for document {entry.doc_id}")
        arrays = build_window_arrays(
            doc.token_ids,
            WindowMetadata(
                bucket_idx=entry.bucket_idx,
                record_idx=entry.record_idx,
                doc_id=entry.doc_id,
                window_idx=0,
                start_chunk=0,
                num_segments=entry.doc_num_chunks,
                doc_token_count=entry.doc_token_count,
                doc_num_chunks=entry.doc_num_chunks,
                eos_token_idx=None,
            ),
            chunk_length=self.manifest.chunk_length,
        )
        return LoadedDocument(
            doc_id=entry.doc_id,
            doc_token_count=entry.doc_token_count,
            doc_num_chunks=entry.doc_num_chunks,
            token_ids_CT=arrays["token_ids_CT"],
            attention_mask_CT=arrays["attention_mask_CT"],
            loss_mask_CT=arrays["loss_mask_CT"],
            chunk_idx_C=arrays["chunk_idx_C"],
            reset_state_C=arrays["reset_state_C"],
        )

    def load_pair(self, entry: ManifestDocument) -> tuple[LoadedDocument, LoadedDocument]:
        donor_entry = self._documents_by_ref.get((entry.donor_bucket_idx, entry.donor_record_idx))
        if donor_entry is None or donor_entry.doc_id != entry.donor_doc_id:
            raise ValueError(f"Invalid donor reference for document {entry.doc_id}")
        return self.load_document(entry), self.load_document(donor_entry)
