"""Full-context evaluation for local Qwen3.5 Hugging Face weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from flax import nnx
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from omegalax.distributed.mesh import ensure_mesh, mesh_rules
from omegalax.evals.aggregation import SummaryPoint, aggregate_metrics
from omegalax.evals.manifest import FullDocumentLoader, load_full_document_manifest
from omegalax.evals.plotting import (
    make_comparison_figure,
    make_heatmap_figure,
    make_single_figure,
    save_common_exact_length_scale,
    save_figure_formats,
)
from omegalax.evals.storage import MetricRow
from omegalax.models.qwen3_5.params import create_qwen3_5_from_safetensors
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.trainers.loss import chunked_cross_entropy_multi_stats


RESULT_VERSION = "full_context_v1"
EXPERIMENT = "gdn"
CONDITION = "true_gdn"
_NUM_LOSS_TILES = 16


@dataclass(frozen=True)
class RelativeNllPoint:
    label: str
    view: str
    doc_num_chunks: int | None
    chunk_position: int
    baseline_nll: float
    candidate_nll: float
    relative_nll_reduction: float
    token_count: float


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        pq.write_table(pa.Table.from_pylist(list(rows)), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_fingerprint(model_path: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(model_path.glob("*.json")) + sorted(model_path.glob("*.safetensors")):
        stat = path.stat()
        digest.update(path.name.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return f"sha256:{digest.hexdigest()}"


def _position_ids(batch_size: int, seq_len: int) -> jax.Array:
    positions_BT = jnp.broadcast_to(
        jnp.arange(seq_len, dtype=jnp.int32)[None, :],
        (batch_size, seq_len),
    )
    return jnp.stack([positions_BT] * 3, axis=0)


def full_context_arrays(document) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten one document while preserving one separate mask per original chunk."""
    token_ids = np.asarray(document.token_ids_CT).reshape(1, -1)
    attention_mask = np.asarray(document.attention_mask_CT).reshape(1, -1)
    num_chunks, chunk_length = document.loss_mask_CT.shape
    loss_masks = np.zeros((num_chunks, 1, num_chunks * chunk_length), dtype=np.int32)
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_length
        # Chunked CE drops mask position 0 after shifting targets. Reproduce that
        # effective mask explicitly so flattening cannot add a boundary target.
        loss_masks[chunk_idx, 0, start + 1 : start + chunk_length] = document.loss_mask_CT[
            chunk_idx, 1:
        ]
    return token_ids, attention_mask, loss_masks


def validate_full_context_arrays(document, loss_masks_NBT: np.ndarray) -> None:
    """Validate target counts and the excluded first target at every chunk boundary."""
    num_chunks, chunk_length = document.loss_mask_CT.shape
    if loss_masks_NBT.shape != (num_chunks, 1, num_chunks * chunk_length):
        raise ValueError(f"Unexpected full-context loss-mask shape: {loss_masks_NBT.shape}")
    for chunk_idx in range(num_chunks):
        boundary = chunk_idx * chunk_length
        if loss_masks_NBT[chunk_idx, 0, boundary] != 0:
            raise ValueError(
                f"Full-context mask includes the cross-boundary target for chunk {chunk_idx + 1}"
            )
        expected = int(document.loss_mask_CT[chunk_idx, 1:].sum())
        actual = int(loss_masks_NBT[chunk_idx, 0, 1:].sum())
        if actual != expected:
            raise ValueError(
                f"Chunk {chunk_idx + 1} target-count mismatch: expected={expected}, actual={actual}"
            )


def make_run_full_context(mesh):
    """Create the single-forward full-document JIT while its mesh is active."""
    with mesh_rules(mesh):

        @nnx.jit(static_argnames=("pad_id",))
        def run_full_context(
            model,
            token_ids_BT: jax.Array,
            attention_mask_BT: jax.Array,
            loss_masks_NBT: jax.Array,
            *,
            pad_id: int,
        ):
            hidden_BTD, _ = text_api.forward(
                model,
                token_ids_BT,
                pad_id=pad_id,
                cfg=model.text.cfg,
                attention_mask_BT=attention_mask_BT,
                position_ids_ZBT=_position_ids(token_ids_BT.shape[0], token_ids_BT.shape[1]),
            )
            return chunked_cross_entropy_multi_stats(
                hidden_BTD,
                model.output_weight(),
                token_ids_BT,
                loss_masks_NBT,
                num_tiles=_NUM_LOSS_TILES,
            )

    return run_full_context


def _selected_documents(manifest, document_cap: int | None, min_chunks: int, max_chunks: int):
    if document_cap is not None and document_cap <= 0:
        raise ValueError("document_cap must be > 0")
    cap = manifest.sample_cap if document_cap is None else document_cap
    if cap > manifest.sample_cap:
        raise ValueError("document_cap exceeds the manifest sample_cap")
    return tuple(
        (shard_id, document)
        for shard_id, document in enumerate(manifest.documents)
        if min_chunks <= document.doc_num_chunks <= max_chunks and document.sample_rank < cap
    )


def _write_plots(points: Sequence[SummaryPoint], output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    for view in ("in_horizon", "beyond_horizon"):
        if any(point.view == view for point in points):
            figure = make_single_figure(
                points,
                experiment=EXPERIMENT,
                metric="nll",
                view=view,
                title="Off-the-shelf Qwen3.5-2B: full context",
            )
            save_figure_formats(figure, plot_dir / f"{view}_nll")
            plt.close(figure)

    scale_path = save_common_exact_length_scale(
        points,
        output_dir / "summary",
        experiment=EXPERIMENT,
        metric="nll",
    )
    y_limits = tuple(json.loads(scale_path.read_text())["y_limits"])
    lengths = sorted(
        {int(point.doc_num_chunks) for point in points if point.doc_num_chunks is not None}
    )
    for length in lengths:
        figure = make_single_figure(
            points,
            experiment=EXPERIMENT,
            metric="nll",
            view="exact_length",
            doc_num_chunks=length,
            title=f"Full context: documents with {length} chunks",
        )
        figure.axes[0].set_ylim(*y_limits)
        save_figure_formats(
            figure,
            plot_dir / f"exact_length_L{length}_nll",
            svg_subdirectory="exact_length_svgs",
        )
        plt.close(figure)

    figure = make_heatmap_figure(
        points,
        experiment=EXPERIMENT,
        metric="nll",
        condition=CONDITION,
        title="Off-the-shelf Qwen3.5-2B: full-context NLL",
    )
    save_figure_formats(figure, plot_dir / "heatmap_nll_full_context")
    plt.close(figure)


def _summary_rows(points: Sequence[SummaryPoint]) -> list[dict[str, Any]]:
    return [asdict(point) for point in points]


def run_full_context_eval(
    *,
    model_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    c_train: int = 4,
    pad_id: int = 248046,
    document_cap: int | None = None,
    min_doc_chunks: int = 2,
    max_doc_chunks: int = 16,
    tp_size: int = 1,
    fsdp_size: int = 1,
    dp_size: int = 1,
) -> Path:
    """Evaluate local safetensors with one causal forward per complete document."""
    if min(c_train, min_doc_chunks, max_doc_chunks, tp_size, fsdp_size, dp_size) <= 0:
        raise ValueError("Chunk and topology sizes must be > 0")
    if min_doc_chunks > max_doc_chunks:
        raise ValueError("min_doc_chunks must not exceed max_doc_chunks")
    if fsdp_size * dp_size != 1:
        raise ValueError("Full-context evaluation currently requires fsdp_size=dp_size=1")

    model_path = Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = load_full_document_manifest(manifest_path)
    selected = _selected_documents(manifest, document_cap, min_doc_chunks, max_doc_chunks)
    if not selected:
        raise ValueError("No manifest documents match the requested selection")

    config = {
        "result_version": RESULT_VERSION,
        "model_path": str(model_path),
        "model_source_fingerprint": _source_fingerprint(model_path),
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "manifest_hash": manifest.manifest_hash,
        "dataset_hash": manifest.dataset_hash,
        "chunk_length": manifest.chunk_length,
        "c_train": c_train,
        "pad_id": pad_id,
        "document_cap": document_cap,
        "min_doc_chunks": min_doc_chunks,
        "max_doc_chunks": max_doc_chunks,
        "tp_size": tp_size,
        "fsdp_size": fsdp_size,
        "dp_size": dp_size,
        "attention_backend": "cudnn",
        "deltanet_kernel": os.environ.get("OMEGALAX_DELTANET_KERNEL", "pallas"),
        "sequence_semantics": "single_forward_continuous_positions",
        "metric_contract": "nll_sum_token_count_v1",
        "selected_shards": [shard_id for shard_id, _ in selected],
    }
    config_path = output_dir / "eval_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text())
        if _canonical_json(existing) != _canonical_json(config):
            raise ValueError(f"Existing evaluation config conflicts at {output_dir}")
    else:
        _atomic_json(config_path, config)
    _atomic_json(output_dir / "status.json", {"complete": False, "completed": 0})
    raw_dir = output_dir / "raw"
    missing = [
        (shard_id, document)
        for shard_id, document in selected
        if not (raw_dir / f"full_context_shard_{shard_id}.parquet").is_file()
    ]
    if missing:
        mesh = ensure_mesh(tp_size=tp_size, fsdp_size=fsdp_size, dp_size=dp_size)
        with mesh_rules(mesh):
            model, cfg = create_qwen3_5_from_safetensors(
                str(model_path),
                tp_size=tp_size,
                fsdp_size=fsdp_size,
                dp_size=dp_size,
                text_only=True,
            )
            set_attn_backend(model, "cudnn")
            loader = FullDocumentLoader(manifest)
            run_document = make_run_full_context(mesh)
            completed = len(selected) - len(missing)
            for shard_id, entry in missing:
                document = loader.load_document(entry)
                tokens, attention, loss_masks = full_context_arrays(document)
                validate_full_context_arrays(document, loss_masks)
                arrays = text_api.shard_batch_dict(
                    {
                        "tokens": tokens,
                        "attention": attention,
                        "loss_masks": loss_masks,
                    },
                    cfg.text_config,
                    mesh,
                )
                nll_sums, token_counts = run_document(
                    model,
                    arrays["tokens"],
                    arrays["attention"],
                    arrays["loss_masks"],
                    pad_id=pad_id,
                )
                nll_sums = np.asarray(jax.device_get(nll_sums))
                token_counts = np.asarray(jax.device_get(token_counts))
                expected_counts = np.asarray(document.loss_mask_CT[:, 1:].sum(axis=1))
                if not np.array_equal(token_counts.astype(np.int64), expected_counts):
                    raise ValueError(
                        f"Runtime token counts differ from source masks for {entry.doc_id}: "
                        f"runtime={token_counts}, expected={expected_counts}"
                    )
                rows = [
                    asdict(
                        MetricRow(
                            experiment=EXPERIMENT,
                            condition=CONDITION,
                            bucket_idx=entry.bucket_idx,
                            record_idx=entry.record_idx,
                            doc_id=entry.doc_id,
                            doc_num_chunks=entry.doc_num_chunks,
                            chunk_position=chunk_idx + 1,
                            nll_sum=float(nll_sums[chunk_idx]),
                            token_count=int(token_counts[chunk_idx]),
                        )
                    )
                    for chunk_idx in range(entry.doc_num_chunks)
                ]
                _atomic_parquet(raw_dir / f"full_context_shard_{shard_id}.parquet", rows)
                completed += 1
                _atomic_json(
                    output_dir / "status.json",
                    {"complete": False, "completed": completed, "total": len(selected)},
                )
            del model
            jax.clear_caches()

    rows = tuple(
        MetricRow(**row)
        for path in sorted(raw_dir.glob("full_context_shard_*.parquet"))
        for row in pq.read_table(path).to_pylist()
    )
    expected_ids = {entry.doc_id for _, entry in selected}
    rows = tuple(row for row in rows if row.doc_id in expected_ids)
    population_counts = {
        count.doc_num_chunks: count.available for count in manifest.counts_by_length
    }
    points = aggregate_metrics(rows, population_counts=population_counts, c_train=c_train)
    _atomic_parquet(output_dir / "raw" / "metrics.parquet", [asdict(row) for row in rows])
    _atomic_parquet(output_dir / "summary" / "metrics.parquet", _summary_rows(points))
    _write_plots(points, output_dir)
    _atomic_json(
        output_dir / "status.json",
        {"complete": True, "completed": len(selected), "total": len(selected)},
    )
    return output_dir


def _read_summary(path: str | Path) -> tuple[SummaryPoint, ...]:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = (
            path / "summary" / "all" / "metrics.parquet",
            path / "summary" / "metrics.parquet",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
    if not path.is_file():
        raise ValueError(f"Summary metrics do not exist: {path}")
    return tuple(SummaryPoint(**row) for row in pq.read_table(path).to_pylist())


def _result_c_train(path: str | Path) -> int:
    path = Path(path).expanduser().resolve()
    candidates = (
        (path / "eval_config.json",)
        if path.is_dir()
        else tuple(parent / "eval_config.json" for parent in path.parents)
    )
    config_path = next(
        (candidate for candidate in candidates if candidate.is_file()), candidates[0]
    )
    if not config_path.is_file():
        raise ValueError(f"Evaluation config does not exist: {config_path}")
    raw = json.loads(config_path.read_text())
    config = raw.get("identity", raw).get("eval_config", raw)
    return int(config["c_train"])


def _write_comparison_plots(
    baseline_result: str | Path,
    baseline: Sequence[SummaryPoint],
    candidates: Sequence[tuple[str, str | Path, Sequence[SummaryPoint]]],
    relative: Sequence[RelativeNllPoint],
    output_dir: Path,
) -> None:
    models = {"Full-context baseline": (baseline, _result_c_train(baseline_result))}
    models.update({label: (points, _result_c_train(path)) for label, path, points in candidates})
    plot_dir = output_dir / "plots"
    for view in ("in_horizon", "beyond_horizon"):
        if all(not any(point.view == view for point in points) for points, _ in models.values()):
            continue
        figure = make_comparison_figure(
            models,
            experiment=EXPERIMENT,
            metric="nll",
            view=view,
            title="Full context versus true-GDN checkpoints",
        )
        save_figure_formats(figure, plot_dir / f"absolute_{view}_nll")
        plt.close(figure)

    lengths = sorted(
        {
            int(point.doc_num_chunks)
            for point in baseline
            if point.view == "exact_length" and point.doc_num_chunks is not None
        }
    )
    for length in lengths:
        figure = make_comparison_figure(
            models,
            experiment=EXPERIMENT,
            metric="nll",
            view="exact_length",
            doc_num_chunks=length,
            title=f"Full context versus true GDN: documents with {length} chunks",
        )
        save_figure_formats(
            figure,
            plot_dir / f"absolute_exact_length_L{length}_nll",
            svg_subdirectory="exact_length_svgs",
        )
        plt.close(figure)

    for view in ("in_horizon", "beyond_horizon"):
        selected = [point for point in relative if point.view == view]
        if not selected:
            continue
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        for label in sorted({point.label for point in selected}):
            series = sorted(
                (point for point in selected if point.label == label),
                key=lambda point: point.chunk_position,
            )
            axis.plot(
                [point.chunk_position for point in series],
                [100 * point.relative_nll_reduction for point in series],
                marker="o",
                label=label,
            )
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.set_xlabel("Chunk Position")
        axis.set_ylabel("NLL reduction vs full context (%)")
        axis.set_title("Positive means lower NLL than the full-context baseline")
        axis.legend()
        figure.tight_layout()
        save_figure_formats(figure, plot_dir / f"relative_{view}_nll_reduction")
        plt.close(figure)

    relative_lengths = sorted(
        {
            int(point.doc_num_chunks)
            for point in relative
            if point.view == "exact_length" and point.doc_num_chunks is not None
        }
    )
    for length in relative_lengths:
        selected = [
            point
            for point in relative
            if point.view == "exact_length" and point.doc_num_chunks == length
        ]
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        for label in sorted({point.label for point in selected}):
            series = sorted(
                (point for point in selected if point.label == label),
                key=lambda point: point.chunk_position,
            )
            axis.plot(
                [point.chunk_position for point in series],
                [100 * point.relative_nll_reduction for point in series],
                marker="o",
                label=label,
            )
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.set_xlabel("Chunk Position")
        axis.set_ylabel("NLL reduction vs full context (%)")
        axis.set_title(
            f"Documents with {length} chunks: positive means lower NLL than full context"
        )
        axis.set_xticks(range(1, length + 1))
        axis.legend()
        figure.tight_layout()
        save_figure_formats(
            figure,
            plot_dir / f"relative_exact_length_L{length}_nll_reduction",
            svg_subdirectory="exact_length_svgs",
        )
        plt.close(figure)

    for label in sorted({point.label for point in relative}):
        selected = [
            point
            for point in relative
            if point.label == label
            and point.view == "exact_length"
            and point.doc_num_chunks is not None
        ]
        if not selected:
            continue
        label_lengths = sorted({int(point.doc_num_chunks) for point in selected})
        max_position = max(point.chunk_position for point in selected)
        matrix = np.full((len(label_lengths), max_position), np.nan)
        rows = {length: idx for idx, length in enumerate(label_lengths)}
        for point in selected:
            matrix[rows[int(point.doc_num_chunks)], point.chunk_position - 1] = (
                100 * point.relative_nll_reduction
            )
        radius = float(np.nanmax(np.abs(matrix)))
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        image = axis.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="RdBu",
            vmin=-radius,
            vmax=radius,
        )
        axis.set_xlabel("Chunk Position")
        axis.set_ylabel("Document Length (chunks)")
        axis.set_xticks(np.arange(max_position), labels=range(1, max_position + 1))
        axis.set_yticks(np.arange(len(label_lengths)), labels=label_lengths)
        axis.set_title(f"{label}: NLL reduction versus full context")
        figure.colorbar(image, ax=axis, label="NLL reduction (%)")
        figure.tight_layout()
        safe_label = "".join(character if character.isalnum() else "_" for character in label)
        save_figure_formats(figure, plot_dir / f"relative_heatmap_{safe_label}")
        plt.close(figure)


def compare_relative_nll(
    *,
    baseline_result: str | Path,
    candidate_results: Sequence[tuple[str, str | Path]],
    output_dir: str | Path,
) -> Path:
    """Compare existing true-GDN summaries against the full-context baseline."""
    baseline = _read_summary(baseline_result)
    baseline_by_key = {
        (point.view, point.doc_num_chunks, point.chunk_position): point
        for point in baseline
        if point.metric == "nll" and point.condition == CONDITION
    }
    relative = []
    candidate_summaries = []
    for label, result in candidate_results:
        candidate = _read_summary(result)
        candidate_summaries.append((label, result, candidate))
        candidate_by_key = {
            (point.view, point.doc_num_chunks, point.chunk_position): point
            for point in candidate
            if point.experiment == EXPERIMENT
            and point.metric == "nll"
            and point.condition == CONDITION
        }
        missing = sorted(set(baseline_by_key) - set(candidate_by_key))
        if missing:
            raise ValueError(f"Candidate {label!r} lacks aligned true_gdn points: {missing[:8]}")
        for key, baseline_point in baseline_by_key.items():
            candidate_point = candidate_by_key[key]
            if not np.isclose(baseline_point.token_count, candidate_point.token_count):
                raise ValueError(
                    f"Candidate {label!r} token count differs for {key}: "
                    f"baseline={baseline_point.token_count}, candidate={candidate_point.token_count}"
                )
            relative.append(
                RelativeNllPoint(
                    label=label,
                    view=key[0],
                    doc_num_chunks=key[1],
                    chunk_position=key[2],
                    baseline_nll=baseline_point.value,
                    candidate_nll=candidate_point.value,
                    relative_nll_reduction=(baseline_point.value - candidate_point.value)
                    / baseline_point.value,
                    token_count=baseline_point.token_count,
                )
            )

    output_dir = Path(output_dir).expanduser().resolve()
    _atomic_json(
        output_dir / "eval_config.json",
        {
            "result_version": "full_context_relative_nll_v1",
            "baseline_result": str(Path(baseline_result).expanduser().resolve()),
            "candidate_results": [
                {"label": label, "result": str(Path(result).expanduser().resolve())}
                for label, result in candidate_results
            ],
            "token_selection": "all",
            "definition": "(full_context_baseline_nll - true_gdn_nll) / full_context_baseline_nll",
        },
    )
    _atomic_parquet(
        output_dir / "summary" / "metrics.parquet",
        [asdict(point) for point in relative],
    )
    _atomic_json(
        output_dir / "summary" / "relative_nll.json",
        {
            "definition": "(full_context_baseline_nll - true_gdn_nll) / full_context_baseline_nll",
            "positive_means": "NLL reduction versus the full-context baseline",
            "points": [asdict(point) for point in relative],
        },
    )
    _write_comparison_plots(
        baseline_result,
        baseline,
        candidate_summaries,
        relative,
        output_dir,
    )
    _atomic_json(output_dir / "status.json", {"complete": True})
    return output_dir
