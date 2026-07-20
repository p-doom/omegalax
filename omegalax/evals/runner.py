"""Checkpoint evaluation execution, resume, plotting, and comparison."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
from matplotlib import pyplot as plt

from omegalax.distributed.mesh import ensure_mesh
from omegalax.evals.aggregation import SummaryPoint, aggregate_metrics
from omegalax.evals.executor import ChainBatch
from omegalax.evals.experiments import run_conv_experiment, run_gdn_experiment
from omegalax.evals.manifest import (
    FullDocumentLoader,
    FullDocumentManifest,
    ManifestDocument,
    load_full_document_manifest,
)
from omegalax.evals.plotting import (
    make_comparison_figure,
    make_heatmap_figure,
    make_single_figure,
    save_common_exact_length_scale,
    save_figure_formats,
)
from omegalax.evals.storage import EvalRunIdentity, EvalRunStore, MetricRow
from omegalax.models.sharding_runtime import set_attn_backend
from omegalax.text import api as text_api
from omegalax.text.chat import StatePassingConfig
from omegalax.text.checkpoint import (
    ResolvedCheckpoint,
    resolve_checkpoint,
    restore_model_params,
)
from omegalax.training_contract import ManualEvalConfig, resolve_eval_config


_CONDITIONS = {
    "gdn": ("true_gdn", "zero_gdn", "shuffled_gdn"),
    "conv": ("true_conv", "zero_conv", "shuffled_conv"),
}
_METRICS = {
    "gdn": ("nll", "gdn_state_gain", "gdn_semantic_gain"),
    "conv": ("nll", "conv_state_gain", "conv_semantic_gain"),
}
_PLOT_TYPES = ("in_horizon", "beyond_horizon", "exact_length", "heatmap")
_METRIC_CONTRACT = "nll_sum_token_count_v1"
_IMMUTABLE_CONFIG_FIELDS = (
    "c_train",
    "pass_gdn_state",
    "gdn_layer_limit",
    "pass_conv_state",
    "pass_rope_positions",
    "pad_id",
    "eos_id",
    "resolution_source",
    "training_contract_hash",
    "tp_size",
    "fsdp_size",
    "dp_size",
    "batch_size",
    "population_counts",
    "metric_contract",
    "deltanet_kernel",
    "attention_backend",
)


@dataclass(frozen=True)
class CheckpointEvalRequest:
    checkpoint: str | Path
    manual_config: ManualEvalConfig | None = None


@dataclass(frozen=True)
class CheckpointEvalSpec:
    checkpoint: str | Path
    c_train: int
    pass_gdn_state: bool
    gdn_layer_limit: int | None
    pass_conv_state: bool
    pass_rope_positions: bool
    pad_id: int
    eos_id: int
    resolution_source: str = "manual_flags"
    training_contract_hash: str | None = None


def resolve_checkpoint_eval_request(request: CheckpointEvalRequest) -> CheckpointEvalSpec:
    try:
        checkpoint = resolve_checkpoint(request.checkpoint)
        checkpoint_root = checkpoint.root
        checkpoint_step = checkpoint.step
    except ValueError:
        run_dir = _result_dir_from_checkpoint_request(request.checkpoint)
        identity = _identity_from_path(run_dir / "eval_config.json")
        checkpoint_root = Path(identity.checkpoint_root)
        checkpoint_step = identity.checkpoint_step
    config = resolve_eval_config(checkpoint_root, checkpoint_step, request.manual_config)
    return CheckpointEvalSpec(
        checkpoint=request.checkpoint,
        c_train=config.c_train,
        pass_gdn_state=config.pass_gdn_state,
        gdn_layer_limit=config.gdn_layer_limit,
        pass_conv_state=config.pass_conv_state,
        pass_rope_positions=config.pass_rope_positions,
        pad_id=config.pad_id,
        eos_id=config.eos_id,
        resolution_source=config.resolution_source,
        training_contract_hash=config.training_contract_hash,
    )


def resolve_checkpoint_eval_requests(
    requests: Iterable[CheckpointEvalRequest | CheckpointEvalSpec],
) -> tuple[CheckpointEvalSpec, ...]:
    return tuple(
        request
        if isinstance(request, CheckpointEvalSpec)
        else resolve_checkpoint_eval_request(request)
        for request in requests
    )


def applicable_experiments(spec: CheckpointEvalSpec) -> tuple[str, ...]:
    experiments = ["gdn"]
    if spec.pass_conv_state:
        experiments.append("conv")
    return tuple(experiments)


def result_dir_for_checkpoint(checkpoint: ResolvedCheckpoint) -> Path:
    return checkpoint.root / "evals" / "state_usage_v1" / f"step_{checkpoint.step}"


def _result_dir_from_checkpoint_request(checkpoint: str | Path) -> Path:
    try:
        return result_dir_for_checkpoint(resolve_checkpoint(checkpoint))
    except ValueError as live_checkpoint_error:
        requested = Path(checkpoint).expanduser().resolve()
        if requested.name.isdigit():
            candidates = (
                requested.parent / "evals" / "state_usage_v1" / f"step_{int(requested.name)}",
            )
        else:
            candidates = tuple(
                sorted(
                    (requested / "evals" / "state_usage_v1").glob("step_*"),
                    key=lambda path: int(path.name.removeprefix("step_")),
                    reverse=True,
                )
            )
        for candidate in candidates:
            if (candidate / "eval_config.json").is_file():
                return candidate
        raise live_checkpoint_error


def _validate_experiments(experiments: Sequence[str] | None) -> tuple[str, ...] | None:
    if experiments is None:
        return None
    experiments = tuple(experiments)
    if not experiments:
        raise ValueError("At least one evaluation experiment is required")
    unknown = [experiment for experiment in experiments if experiment not in _CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown evaluation experiment(s): {unknown}")
    if len(set(experiments)) != len(experiments):
        raise ValueError(f"Duplicate evaluation experiments: {experiments}")
    return experiments


def _validate_plot_types(plot_types: Sequence[str]) -> tuple[str, ...]:
    plot_types = tuple(plot_types)
    unknown = [plot_type for plot_type in plot_types if plot_type not in _PLOT_TYPES]
    if unknown:
        raise ValueError(f"Unknown plot type(s): {unknown}")
    if not plot_types:
        raise ValueError("At least one plot type is required")
    if len(set(plot_types)) != len(plot_types):
        raise ValueError(f"Duplicate plot types: {plot_types}")
    return plot_types


def _code_hash() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    paths = set((repo_root / "omegalax" / "evals").glob("*.py"))
    paths.update((repo_root / "omegalax" / "models" / "qwen3_5").rglob("*.py"))
    paths.update(
        {
            repo_root / "omegalax" / "data" / "pretrain_data_set.py",
            repo_root / "omegalax" / "distributed" / "mesh.py",
            repo_root / "omegalax" / "models" / "params_utils.py",
            repo_root / "omegalax" / "models" / "shard_config.py",
            repo_root / "omegalax" / "models" / "sharding_runtime.py",
            repo_root / "omegalax" / "text" / "api.py",
            repo_root / "omegalax" / "text" / "chat.py",
            repo_root / "omegalax" / "text" / "checkpoint.py",
            repo_root / "omegalax" / "trainers" / "loss.py",
            repo_root / "scripts" / "run_checkpoint_evals.py",
            repo_root / "scripts" / "submit_checkpoint_evals.py",
            repo_root / "uv.lock",
        }
    )
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _runtime_backends() -> tuple[str, str]:
    deltanet_kernel = os.environ.get("OMEGALAX_DELTANET_KERNEL")
    if deltanet_kernel is None:
        deltanet_kernel = "xla" if jax.default_backend() == "cpu" else "pallas"
    attention_backend = "xla" if jax.default_backend() == "cpu" else "cudnn"
    return deltanet_kernel.lower(), attention_backend


def _selected_documents(
    manifest: FullDocumentManifest,
    document_cap: int | None,
) -> tuple[tuple[int, ManifestDocument], ...]:
    if document_cap is not None:
        if document_cap < 2:
            raise ValueError("document_cap must be >= 2")
        if document_cap > manifest.sample_cap:
            raise ValueError(
                f"document_cap={document_cap} exceeds manifest sample_cap={manifest.sample_cap}"
            )
    cap = manifest.sample_cap if document_cap is None else document_cap
    selected = tuple(
        (shard_id, document)
        for shard_id, document in enumerate(manifest.documents)
        if document.sample_rank < cap
    )
    selected_refs = {
        (document.bucket_idx, document.record_idx, document.doc_id) for _, document in selected
    }
    missing_donors = [
        document.doc_id
        for _, document in selected
        if (
            document.donor_bucket_idx,
            document.donor_record_idx,
            document.donor_doc_id,
        )
        not in selected_refs
    ]
    if missing_donors:
        raise ValueError(
            "document_cap prefix is not donor-closed; missing donors for "
            f"documents: {missing_donors[:8]}"
        )
    return selected


def _eval_config(
    spec: CheckpointEvalSpec,
    manifest: FullDocumentManifest,
    *,
    experiments: Sequence[str],
    document_cap: int | None,
    tp_size: int,
    fsdp_size: int,
    dp_size: int,
    batch_size: int,
) -> dict[str, Any]:
    selected = _selected_documents(manifest, document_cap)
    shard_ids = [shard_id for shard_id, _ in selected]
    experiments = tuple(experiment for experiment in _CONDITIONS if experiment in experiments)
    deltanet_kernel, attention_backend = _runtime_backends()
    return {
        "c_train": spec.c_train,
        "pass_gdn_state": spec.pass_gdn_state,
        "gdn_layer_limit": spec.gdn_layer_limit,
        "pass_conv_state": spec.pass_conv_state,
        "pass_rope_positions": spec.pass_rope_positions,
        "pad_id": spec.pad_id,
        "eos_id": spec.eos_id,
        "resolution_source": spec.resolution_source,
        "training_contract_hash": spec.training_contract_hash,
        "tp_size": tp_size,
        "fsdp_size": fsdp_size,
        "dp_size": dp_size,
        "batch_size": batch_size,
        "document_cap": document_cap,
        "effective_document_cap": (manifest.sample_cap if document_cap is None else document_cap),
        "experiments": list(experiments),
        "population_counts": {
            str(count.doc_num_chunks): count.available for count in manifest.counts_by_length
        },
        "metric_contract": _METRIC_CONTRACT,
        "deltanet_kernel": deltanet_kernel,
        "attention_backend": attention_backend,
        "conditions_by_experiment": {
            experiment: list(_CONDITIONS[experiment]) for experiment in experiments
        },
        "expected_shards": {
            experiment: {condition: list(shard_ids) for condition in _CONDITIONS[experiment]}
            for experiment in experiments
        },
    }


def _requested_identity(
    spec: CheckpointEvalSpec,
    checkpoint: ResolvedCheckpoint,
    manifest: FullDocumentManifest,
    *,
    experiments: Sequence[str],
    document_cap: int | None,
    tp_size: int,
    fsdp_size: int,
    dp_size: int,
    batch_size: int,
) -> EvalRunIdentity:
    return EvalRunIdentity(
        dataset_hash=manifest.dataset_hash,
        manifest_hash=manifest.manifest_hash,
        checkpoint_root=str(checkpoint.root.resolve()),
        checkpoint_step=checkpoint.step,
        code_hash=_code_hash(),
        eval_config=_eval_config(
            spec,
            manifest,
            experiments=experiments,
            document_cap=document_cap,
            tp_size=tp_size,
            fsdp_size=fsdp_size,
            dp_size=dp_size,
            batch_size=batch_size,
        ),
    )


def _identity_from_path(config_path: Path) -> EvalRunIdentity:
    if not config_path.is_file():
        raise ValueError(f"Evaluation result has no eval_config.json: {config_path.parent}")
    raw = json.loads(config_path.read_text())
    return EvalRunIdentity(**dict(raw.get("identity", raw)))


def _existing_store(run_dir: str | Path) -> EvalRunStore:
    run_dir = Path(run_dir).expanduser().resolve()
    identity = _identity_from_path(run_dir / "eval_config.json")
    expected_dir = (
        Path(identity.checkpoint_root).expanduser().resolve()
        / "evals"
        / "state_usage_v1"
        / f"step_{identity.checkpoint_step}"
    )
    if run_dir != expected_dir:
        raise ValueError(f"Evaluation run directory conflicts with its stored identity: {run_dir}")
    return EvalRunStore(
        run_dir=run_dir,
        config_path=run_dir / "eval_config.json",
        status_path=run_dir / "status.json",
        identity=identity,
    )


def _merge_identity(
    existing: EvalRunIdentity,
    requested: EvalRunIdentity,
    manifest: FullDocumentManifest,
) -> EvalRunIdentity:
    for name in ("dataset_hash", "manifest_hash", "checkpoint_root", "checkpoint_step"):
        if getattr(existing, name) != getattr(requested, name):
            raise ValueError(
                f"Evaluation identity {name} conflict: "
                f"stored={getattr(existing, name)!r}, requested={getattr(requested, name)!r}"
            )
    if existing.code_hash != requested.code_hash:
        raise ValueError(
            "Evaluation identity code_hash conflict: "
            f"stored={existing.code_hash!r}, requested={requested.code_hash!r}"
        )
    for name in _IMMUTABLE_CONFIG_FIELDS:
        if existing.eval_config.get(name) != requested.eval_config.get(name):
            raise ValueError(
                f"Immutable eval_config conflict for {name}: "
                f"stored={existing.eval_config.get(name)!r}, "
                f"requested={requested.eval_config.get(name)!r}"
            )

    existing_experiments = tuple(existing.eval_config["experiments"])
    requested_experiments = tuple(requested.eval_config["experiments"])
    merged_experiments = tuple(
        experiment
        for experiment in _CONDITIONS
        if experiment in existing_experiments or experiment in requested_experiments
    )
    existing_effective_cap = int(existing.eval_config["effective_document_cap"])
    requested_effective_cap = int(requested.eval_config["effective_document_cap"])
    existing_cap = existing.eval_config.get("document_cap")
    requested_cap = requested.eval_config.get("document_cap")
    if existing_cap is None or requested_cap is None:
        merged_cap = None
    else:
        merged_cap = max(int(existing_cap), int(requested_cap))

    if (
        merged_experiments == existing_experiments
        and requested_effective_cap <= existing_effective_cap
    ):
        return existing

    merged_config = dict(requested.eval_config)
    merged_config.update(
        _eval_config(
            CheckpointEvalSpec(
                requested.checkpoint_root,
                c_train=int(requested.eval_config["c_train"]),
                pass_gdn_state=bool(requested.eval_config["pass_gdn_state"]),
                gdn_layer_limit=requested.eval_config["gdn_layer_limit"],
                pass_conv_state=bool(requested.eval_config["pass_conv_state"]),
                pass_rope_positions=bool(requested.eval_config["pass_rope_positions"]),
                pad_id=int(requested.eval_config["pad_id"]),
                eos_id=int(requested.eval_config["eos_id"]),
                resolution_source=str(requested.eval_config["resolution_source"]),
                training_contract_hash=requested.eval_config["training_contract_hash"],
            ),
            manifest,
            experiments=merged_experiments,
            document_cap=merged_cap,
            tp_size=int(requested.eval_config["tp_size"]),
            fsdp_size=int(requested.eval_config["fsdp_size"]),
            dp_size=int(requested.eval_config["dp_size"]),
            batch_size=int(requested.eval_config["batch_size"]),
        )
    )
    return replace(requested, eval_config=merged_config)


def _open_or_extend_store(
    checkpoint: ResolvedCheckpoint,
    requested: EvalRunIdentity,
    manifest: FullDocumentManifest,
) -> EvalRunStore:
    run_dir = result_dir_for_checkpoint(checkpoint)
    if not (run_dir / "eval_config.json").is_file():
        return EvalRunStore.open(checkpoint.root, checkpoint.step, requested)

    existing = _identity_from_path(run_dir / "eval_config.json")
    merged = _merge_identity(existing, requested, manifest)
    store = EvalRunStore.open(checkpoint.root, checkpoint.step, existing)
    if merged != existing:
        store.extend_identity(merged)
    return store


def _validate_raw_results(
    store: EvalRunStore,
    experiments: Sequence[str],
) -> None:
    config = store.identity.eval_config
    expected_by_experiment = dict(config.get("expected_shards", {}))
    configured_experiments = set(config.get("experiments", ()))
    target_signatures: dict[int, tuple[Any, ...]] = {}
    chunk_one_nll: dict[int, float] = {}
    for experiment in experiments:
        if experiment not in configured_experiments:
            raise ValueError(f"Evaluation component is missing: {experiment}")
        expected_conditions = dict(expected_by_experiment.get(experiment, {}))
        for condition in _CONDITIONS[experiment]:
            if condition not in expected_conditions:
                raise ValueError(
                    f"Evaluation condition is incomplete or missing: {experiment}/{condition}"
                )
            for raw_shard_id in expected_conditions[condition]:
                shard_id = int(raw_shard_id)
                rows = store.read_shard(experiment, condition, shard_id)
                if not rows:
                    raise ValueError(
                        f"Evaluation condition is incomplete; missing shard "
                        f"{experiment}/{condition}/{shard_id}"
                    )
                first = rows[0]
                expected_positions = list(range(1, first.doc_num_chunks + 1))
                positions = sorted(row.chunk_position for row in rows)
                if positions != expected_positions:
                    raise ValueError(
                        f"Evaluation shard has incomplete chunk rows for "
                        f"{experiment}/{condition}/{shard_id}: "
                        f"expected={expected_positions}, actual={positions}"
                    )
                for row in rows:
                    if (
                        row.experiment != experiment
                        or row.condition != condition
                        or row.bucket_idx != first.bucket_idx
                        or row.record_idx != first.record_idx
                        or row.doc_id != first.doc_id
                        or row.doc_num_chunks != first.doc_num_chunks
                    ):
                        raise ValueError(
                            f"Evaluation shard rows conflict for "
                            f"{experiment}/{condition}/{shard_id}"
                        )
                signature = (
                    first.bucket_idx,
                    first.record_idx,
                    first.doc_id,
                    first.doc_num_chunks,
                    tuple(
                        (row.chunk_position, row.token_count)
                        for row in sorted(rows, key=lambda row: row.chunk_position)
                    ),
                )
                previous_signature = target_signatures.setdefault(shard_id, signature)
                if previous_signature != signature:
                    raise ValueError(
                        f"Evaluation target/token_count conflict across conditions for "
                        f"shard {shard_id}: {experiment}/{condition}"
                    )
                first_chunk_nll = next(row.nll_sum for row in rows if row.chunk_position == 1)
                reference_nll = chunk_one_nll.setdefault(shard_id, first_chunk_nll)
                if abs(first_chunk_nll - reference_nll) > 1e-6:
                    raise ValueError(
                        f"Evaluation chunk 1 nll_sum conflict across conditions for "
                        f"shard {shard_id}: {experiment}/{condition}"
                    )


def validate_checkpoint_results(
    run_dir: str | Path,
    *,
    experiments: Sequence[str] | None = None,
) -> EvalRunStore:
    store = _existing_store(run_dir)
    selected = _validate_experiments(experiments)
    if selected is None:
        selected = tuple(store.identity.eval_config["experiments"])
    _validate_raw_results(store, selected)
    return store


def _donor_components(
    documents: Sequence[tuple[int, ManifestDocument]],
) -> tuple[tuple[tuple[int, ManifestDocument], ...], ...]:
    by_ref = {
        (document.bucket_idx, document.record_idx, document.doc_id): (shard_id, document)
        for shard_id, document in documents
    }
    adjacency = {reference: set() for reference in by_ref}
    for reference, (_, document) in by_ref.items():
        donor_ref = (
            document.donor_bucket_idx,
            document.donor_record_idx,
            document.donor_doc_id,
        )
        if donor_ref not in by_ref:
            raise ValueError(f"Selected document set is not donor-closed: {document.doc_id}")
        adjacency[reference].add(donor_ref)
        adjacency[donor_ref].add(reference)

    components = []
    visited = set()
    for reference in by_ref:
        if reference in visited:
            continue
        pending = [reference]
        component_refs = []
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            component_refs.append(current)
            pending.extend(adjacency[current] - visited)
        component = tuple(
            sorted((by_ref[item] for item in component_refs), key=lambda item: item[0])
        )
        components.append(component)
    return tuple(components)


def _evaluation_batches(
    documents: Sequence[tuple[int, ManifestDocument]],
    *,
    batch_size: int,
) -> tuple[tuple[tuple[int, ManifestDocument], ...], ...]:
    batches = []
    current: list[tuple[int, ManifestDocument]] = []
    for component in _donor_components(documents):
        if len(component) > batch_size:
            raise ValueError(
                f"Donor component size {len(component)} exceeds batch_size={batch_size}"
            )
        if current and len(current) + len(component) > batch_size:
            batches.append(tuple(current))
            current = []
        current.extend(component)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _make_chain_batch(
    loader: FullDocumentLoader,
    documents: Sequence[tuple[int, ManifestDocument]],
    *,
    batch_size: int,
    cfg,
    mesh,
) -> tuple[ChainBatch, jax.Array, int]:
    loaded = [(shard_id, entry, loader.load_document(entry)) for shard_id, entry in documents]
    original_count = len(loaded)
    while len(loaded) < batch_size:
        loaded.append(loaded[0])

    donor_indices = []
    local_by_ref = {
        (entry.bucket_idx, entry.record_idx, entry.doc_id): index
        for index, (_, entry, _) in enumerate(loaded[:original_count])
    }
    for index, (_, entry, _) in enumerate(loaded):
        if index >= original_count:
            donor_indices.append(index)
            continue
        donor_ref = (entry.donor_bucket_idx, entry.donor_record_idx, entry.donor_doc_id)
        donor_indices.append(local_by_ref[donor_ref])

    arrays = text_api.shard_batch_dict(
        {
            "token_ids_BCT": np.stack([document.token_ids_CT for _, _, document in loaded]),
            "attention_mask_BCT": np.stack(
                [document.attention_mask_CT for _, _, document in loaded]
            ),
            "loss_mask_BCT": np.stack([document.loss_mask_CT for _, _, document in loaded]),
            "chunk_indices_BC": np.stack([document.chunk_idx_C for _, _, document in loaded]),
            "donor_indices_B": np.asarray(donor_indices, dtype=np.int32),
        },
        cfg,
        mesh,
    )
    return (
        ChainBatch(
            document_ids=tuple(entry.doc_id for _, entry, _ in loaded),
            token_ids_BCT=arrays["token_ids_BCT"],
            attention_mask_BCT=arrays["attention_mask_BCT"],
            loss_mask_BCT=arrays["loss_mask_BCT"],
            chunk_indices_BC=arrays["chunk_indices_BC"],
        ),
        arrays["donor_indices_B"],
        original_count,
    )


def _result_rows(
    experiment: str,
    condition: str,
    result,
    documents: Sequence[tuple[int, ManifestDocument]],
) -> tuple[tuple[int, tuple[MetricRow, ...]], ...]:
    nll_by_segment = tuple(
        np.asarray(jax.device_get(segment.nll_sum_B)) for segment in result.segments
    )
    counts_by_segment = tuple(
        np.asarray(jax.device_get(segment.token_count_B)) for segment in result.segments
    )
    rows_by_shard = []
    for batch_index, (shard_id, document) in enumerate(documents):
        rows = tuple(
            MetricRow(
                experiment=experiment,
                condition=condition,
                bucket_idx=document.bucket_idx,
                record_idx=document.record_idx,
                doc_id=document.doc_id,
                doc_num_chunks=document.doc_num_chunks,
                chunk_position=segment_index + 1,
                nll_sum=float(nll_by_segment[segment_index][batch_index]),
                token_count=int(counts_by_segment[segment_index][batch_index]),
            )
            for segment_index, segment in enumerate(result.segments)
        )
        rows_by_shard.append((shard_id, rows))
    return tuple(rows_by_shard)


def _run_missing_inference(
    store: EvalRunStore,
    spec: CheckpointEvalSpec,
    checkpoint: ResolvedCheckpoint,
    manifest: FullDocumentManifest,
) -> None:
    config = store.identity.eval_config
    experiments = tuple(config["experiments"])
    expected = dict(config["expected_shards"])
    missing = any(
        not store.shard_is_complete(experiment, condition, int(shard_id))
        for experiment in experiments
        for condition, shard_ids in dict(expected[experiment]).items()
        for shard_id in shard_ids
    )
    if not missing:
        return

    model, cfg = text_api.init_model(
        str(checkpoint.config_path),
        jax.random.key(0),
        tp_size=int(config["tp_size"]),
        fsdp_size=int(config["fsdp_size"]),
        dp_size=int(config["dp_size"]),
    )
    model = restore_model_params(model, checkpoint)
    set_attn_backend(model, str(config["attention_backend"]))
    mesh = ensure_mesh(
        tp_size=int(config["tp_size"]),
        fsdp_size=int(config["fsdp_size"]),
        dp_size=int(config["dp_size"]),
    )
    state_config = StatePassingConfig(
        pass_gdn_state=spec.pass_gdn_state,
        gdn_layer_limit=spec.gdn_layer_limit,
        pass_conv_state=spec.pass_conv_state,
        pass_rope_positions=spec.pass_rope_positions,
    )
    selected_ids = {
        int(shard_id)
        for experiment in experiments
        for shard_ids in dict(expected[experiment]).values()
        for shard_id in shard_ids
    }
    selected = tuple(
        (shard_id, document)
        for shard_id, document in enumerate(manifest.documents)
        if shard_id in selected_ids
    )
    loader = FullDocumentLoader(manifest)
    batch_size = int(config["batch_size"])
    by_length: dict[int, list[tuple[int, ManifestDocument]]] = {}
    for item in selected:
        by_length.setdefault(item[1].doc_num_chunks, []).append(item)

    experiment_functions = {
        "gdn": run_gdn_experiment,
        "conv": run_conv_experiment,
    }
    for doc_num_chunks in sorted(by_length):
        for documents in _evaluation_batches(by_length[doc_num_chunks], batch_size=batch_size):
            needed_experiments = [
                experiment
                for experiment in experiments
                if any(
                    not store.shard_is_complete(experiment, condition, shard_id)
                    for condition in _CONDITIONS[experiment]
                    for shard_id, _ in documents
                )
            ]
            if not needed_experiments:
                continue
            chain_batch, donor_indices_B, original_count = _make_chain_batch(
                loader,
                documents,
                batch_size=batch_size,
                cfg=cfg,
                mesh=mesh,
            )
            original_documents = tuple(documents[:original_count])
            for experiment in needed_experiments:
                experiment_result = experiment_functions[experiment](
                    model,
                    cfg,
                    chain_batch,
                    state_config=state_config,
                    donor_indices_B=donor_indices_B,
                    pad_id=spec.pad_id,
                    checkpoint=str(checkpoint.step_path),
                )
                for condition, chain_result in experiment_result.conditions.items():
                    for shard_id, rows in _result_rows(
                        experiment,
                        condition,
                        chain_result,
                        original_documents,
                    ):
                        if not store.shard_is_complete(experiment, condition, shard_id):
                            store.write_shard(experiment, condition, shard_id, rows)


def _configured_rows(store: EvalRunStore) -> tuple[MetricRow, ...]:
    experiments = set(store.identity.eval_config["experiments"])
    return tuple(row for row in store.read_rows() if row.experiment in experiments)


def _population_counts(config: dict[str, Any]) -> dict[int, int]:
    return {int(length): int(count) for length, count in config["population_counts"].items()}


def _clear_plot_formats(directory: Path) -> None:
    if not directory.exists():
        return
    for suffix in ("*.pdf", "*.svg", "*.png"):
        for path in directory.glob(suffix):
            path.unlink()


def _save_figure_to_targets(figure, targets: Sequence[Path]) -> None:
    try:
        for target in targets:
            save_figure_formats(figure, target)
    finally:
        plt.close(figure)


def _single_plot_targets(
    store: EvalRunStore,
    experiment: str,
    stem: str,
    plot_output_roots: Sequence[str | Path],
) -> tuple[Path, ...]:
    checkpoint_name = Path(store.identity.checkpoint_root).name
    step_name = f"step_{store.identity.checkpoint_step}"
    return (
        store.run_dir / "plots" / experiment / stem,
        *(
            Path(root).expanduser().resolve() / checkpoint_name / step_name / experiment / stem
            for root in plot_output_roots
        ),
    )


def _plot_single_checkpoint(
    store: EvalRunStore,
    points: Sequence[SummaryPoint],
    *,
    experiments: Sequence[str],
    plot_types: Sequence[str],
    plot_output_roots: Sequence[str | Path],
) -> None:
    summary_dir = store.run_dir / "summary"
    for experiment in experiments:
        target_dirs = {
            target.parent
            for target in _single_plot_targets(
                store,
                experiment,
                "placeholder",
                plot_output_roots,
            )
        }
        for directory in target_dirs:
            _clear_plot_formats(directory)

        if "exact_length" not in plot_types:
            for path in summary_dir.glob(f"{experiment}_*_exact_length_scale.json"):
                path.unlink()

        for view in ("in_horizon", "beyond_horizon"):
            if view not in plot_types:
                continue
            if not any(point.experiment == experiment and point.view == view for point in points):
                continue
            for metric in _METRICS[experiment]:
                figure = make_single_figure(
                    points,
                    experiment=experiment,
                    metric=metric,
                    view=view,
                )
                _save_figure_to_targets(
                    figure,
                    _single_plot_targets(
                        store,
                        experiment,
                        f"{view}_{metric}",
                        plot_output_roots,
                    ),
                )

        if "exact_length" in plot_types:
            lengths = sorted(
                {
                    int(point.doc_num_chunks)
                    for point in points
                    if point.experiment == experiment
                    and point.view == "exact_length"
                    and point.doc_num_chunks is not None
                }
            )
            for metric in _METRICS[experiment]:
                scale_path = save_common_exact_length_scale(
                    points,
                    summary_dir,
                    experiment=experiment,
                    metric=metric,
                )
                y_limits = tuple(json.loads(scale_path.read_text())["y_limits"])
                for length in lengths:
                    figure = make_single_figure(
                        points,
                        experiment=experiment,
                        metric=metric,
                        view="exact_length",
                        doc_num_chunks=length,
                    )
                    figure.axes[0].set_ylim(*y_limits)
                    _save_figure_to_targets(
                        figure,
                        _single_plot_targets(
                            store,
                            experiment,
                            f"exact_length_L{length}_{metric}",
                            plot_output_roots,
                        ),
                    )

        if "heatmap" in plot_types:
            for condition in _CONDITIONS[experiment]:
                figure = make_heatmap_figure(
                    points,
                    experiment=experiment,
                    metric="nll",
                    condition=condition,
                )
                _save_figure_to_targets(
                    figure,
                    _single_plot_targets(
                        store,
                        experiment,
                        f"heatmap_nll_{condition}",
                        plot_output_roots,
                    ),
                )
            for metric in _METRICS[experiment][1:]:
                figure = make_heatmap_figure(
                    points,
                    experiment=experiment,
                    metric=metric,
                    condition=None,
                )
                _save_figure_to_targets(
                    figure,
                    _single_plot_targets(
                        store,
                        experiment,
                        f"heatmap_{metric}",
                        plot_output_roots,
                    ),
                )


def _derive_checkpoint_outputs(
    store: EvalRunStore,
    *,
    experiments: Sequence[str],
    plot_types: Sequence[str],
    plot_output_roots: Sequence[str | Path],
) -> tuple[SummaryPoint, ...]:
    configured_experiments = tuple(store.identity.eval_config["experiments"])
    _validate_raw_results(store, configured_experiments)
    (store.run_dir / "summary").mkdir(parents=True, exist_ok=True)
    points = aggregate_metrics(
        _configured_rows(store),
        population_counts=_population_counts(store.identity.eval_config),
        c_train=int(store.identity.eval_config["c_train"]),
    )
    store.write_summary(points)
    _plot_single_checkpoint(
        store,
        points,
        experiments=experiments,
        plot_types=plot_types,
        plot_output_roots=plot_output_roots,
    )
    return points


def run_checkpoint_eval(
    spec: CheckpointEvalSpec,
    *,
    manifest_path: str | Path,
    experiments: Sequence[str] | None,
    plot_types: Sequence[str],
    plot_output_roots: Sequence[str | Path] = (),
    document_cap: int | None = None,
    tp_size: int = 1,
    fsdp_size: int = 1,
    dp_size: int = 1,
    batch_size: int = 1,
) -> Path:
    selected_experiments = _validate_experiments(experiments)
    plot_types = _validate_plot_types(plot_types)
    manifest = load_full_document_manifest(manifest_path)
    _selected_documents(manifest, document_cap)
    if spec.c_train <= 0:
        raise ValueError(f"c_train must be > 0, got {spec.c_train}")
    if spec.gdn_layer_limit is not None and spec.gdn_layer_limit < 0:
        raise ValueError("gdn_layer_limit must be non-negative or None")
    if min(tp_size, fsdp_size, dp_size, batch_size) <= 0:
        raise ValueError("Topology sizes and batch_size must be > 0")
    if batch_size % (fsdp_size * dp_size):
        raise ValueError(
            f"batch_size={batch_size} must be divisible by fsdp_size*dp_size={fsdp_size * dp_size}"
        )
    applicable = applicable_experiments(spec)
    if selected_experiments is None:
        selected_experiments = applicable
    unavailable = [
        experiment for experiment in selected_experiments if experiment not in applicable
    ]
    if unavailable:
        raise ValueError(
            f"Experiment(s) are not applicable to this checkpoint configuration: {unavailable}"
        )

    checkpoint = resolve_checkpoint(spec.checkpoint)
    identity = _requested_identity(
        spec,
        checkpoint,
        manifest,
        experiments=selected_experiments,
        document_cap=document_cap,
        tp_size=tp_size,
        fsdp_size=fsdp_size,
        dp_size=dp_size,
        batch_size=batch_size,
    )
    store = _open_or_extend_store(checkpoint, identity, manifest)
    _run_missing_inference(store, spec, checkpoint, manifest)
    _derive_checkpoint_outputs(
        store,
        experiments=selected_experiments,
        plot_types=plot_types,
        plot_output_roots=plot_output_roots,
    )
    store.mark_complete()
    return store.run_dir


def plot_checkpoint_results(
    run_dir: str | Path,
    *,
    experiments: Sequence[str] | None = None,
    plot_types: Sequence[str] = _PLOT_TYPES,
    plot_output_roots: Sequence[str | Path] = (),
) -> Path:
    selected_experiments = _validate_experiments(experiments)
    plot_types = _validate_plot_types(plot_types)
    store = _existing_store(run_dir)
    if selected_experiments is None:
        selected_experiments = tuple(store.identity.eval_config["experiments"])
    _validate_raw_results(store, selected_experiments)
    _derive_checkpoint_outputs(
        store,
        experiments=selected_experiments,
        plot_types=plot_types,
        plot_output_roots=plot_output_roots,
    )
    store.mark_complete()
    return store.run_dir


def _comparison_targets(
    run_dirs: Sequence[Path],
    plot_output_roots: Sequence[str | Path],
    comparison_name: str,
    experiment: str,
    stem: str,
) -> tuple[Path, ...]:
    return (
        *(run_dir / "comparisons" / comparison_name / experiment / stem for run_dir in run_dirs),
        *(
            Path(root).expanduser().resolve() / "comparisons" / comparison_name / experiment / stem
            for root in plot_output_roots
        ),
    )


def compare_checkpoint_results(
    run_dirs: Sequence[str | Path],
    *,
    experiments: Sequence[str],
    plot_types: Sequence[str],
    comparison_name: str,
    plot_output_roots: Sequence[str | Path] = (),
) -> None:
    run_dirs = tuple(Path(run_dir).expanduser().resolve() for run_dir in run_dirs)
    if len(run_dirs) < 2:
        raise ValueError("Checkpoint comparison requires at least two result directories")
    selected_experiments = _validate_experiments(experiments)
    if not selected_experiments:
        raise ValueError("Checkpoint comparison requires at least one experiment")
    plot_types = _validate_plot_types(plot_types)
    if "heatmap" in plot_types:
        raise ValueError("Heatmap plots are not defined for checkpoint comparison")
    if not comparison_name or Path(comparison_name).name != comparison_name:
        raise ValueError(f"Invalid comparison name: {comparison_name!r}")

    stores = tuple(
        validate_checkpoint_results(run_dir, experiments=selected_experiments)
        for run_dir in run_dirs
    )
    baseline = stores[0].identity
    compatibility_fields = (
        ("dataset_hash", baseline.dataset_hash),
        ("manifest_hash", baseline.manifest_hash),
        ("cap", baseline.eval_config["effective_document_cap"]),
        ("metric", baseline.eval_config["metric_contract"]),
    )
    for store in stores[1:]:
        candidate_values = {
            "dataset_hash": store.identity.dataset_hash,
            "manifest_hash": store.identity.manifest_hash,
            "cap": store.identity.eval_config["effective_document_cap"],
            "metric": store.identity.eval_config["metric_contract"],
        }
        for label, expected_value in compatibility_fields:
            if candidate_values[label] != expected_value:
                raise ValueError(
                    f"Checkpoint comparison {label} conflict: "
                    f"expected={expected_value!r}, actual={candidate_values[label]!r}"
                )

    summaries = []
    labels: list[str] = []
    for store in stores:
        base_label = Path(store.identity.checkpoint_root).name
        label = base_label
        if label in labels:
            label = f"{base_label} step {store.identity.checkpoint_step}"
        suffix = 2
        while label in labels:
            label = f"{base_label} step {store.identity.checkpoint_step} ({suffix})"
            suffix += 1
        labels.append(label)
        summaries.append(
            aggregate_metrics(
                _configured_rows(store),
                population_counts=_population_counts(store.identity.eval_config),
                c_train=int(store.identity.eval_config["c_train"]),
            )
        )
    models = {
        label: (summary, int(store.identity.eval_config["c_train"]))
        for label, summary, store in zip(labels, summaries, stores, strict=True)
    }

    for experiment in selected_experiments:
        target_dirs = {
            target.parent
            for target in _comparison_targets(
                run_dirs,
                plot_output_roots,
                comparison_name,
                experiment,
                "placeholder",
            )
        }
        for directory in target_dirs:
            _clear_plot_formats(directory)

        for view in ("in_horizon", "beyond_horizon"):
            if view not in plot_types:
                continue
            for metric in _METRICS[experiment]:
                figure = make_comparison_figure(
                    models,
                    experiment=experiment,
                    metric=metric,
                    view=view,
                )
                _save_figure_to_targets(
                    figure,
                    _comparison_targets(
                        run_dirs,
                        plot_output_roots,
                        comparison_name,
                        experiment,
                        f"{view}_{metric}",
                    ),
                )

        if "exact_length" in plot_types:
            lengths = sorted(
                {
                    int(point.doc_num_chunks)
                    for point in summaries[0]
                    if point.experiment == experiment
                    and point.view == "exact_length"
                    and point.doc_num_chunks is not None
                }
            )
            for length in lengths:
                for metric in _METRICS[experiment]:
                    figure = make_comparison_figure(
                        models,
                        experiment=experiment,
                        metric=metric,
                        view="exact_length",
                        doc_num_chunks=length,
                    )
                    _save_figure_to_targets(
                        figure,
                        _comparison_targets(
                            run_dirs,
                            plot_output_roots,
                            comparison_name,
                            experiment,
                            f"exact_length_L{length}_{metric}",
                        ),
                    )


def run_evals(
    requests: Iterable[CheckpointEvalRequest | CheckpointEvalSpec],
    *,
    mode: str,
    manifest_path: str | Path | None = None,
    experiments: Sequence[str] | None = None,
    plot_types: Sequence[str] = _PLOT_TYPES,
    plot_output_roots: Sequence[str | Path] = (),
    comparison_name: str | None = None,
    document_cap: int | None = None,
    tp_size: int = 1,
    fsdp_size: int = 1,
    dp_size: int = 1,
    batch_size: int = 1,
) -> None:
    specs = resolve_checkpoint_eval_requests(requests)
    if not specs:
        raise ValueError("At least one checkpoint specification is required")
    if mode not in {"all", "subset", "plot", "compare"}:
        raise ValueError(f"Unknown evaluation mode: {mode!r}")
    selected_experiments = _validate_experiments(experiments)
    plot_types = _validate_plot_types(plot_types)

    if mode in {"all", "subset"}:
        if manifest_path is None:
            raise ValueError(f"mode={mode!r} requires manifest_path")
        if mode == "subset" and selected_experiments is None:
            raise ValueError("mode='subset' requires explicit experiments")
        for spec in specs:
            run_checkpoint_eval(
                spec,
                manifest_path=manifest_path,
                experiments=None if mode == "all" else selected_experiments,
                plot_types=plot_types,
                plot_output_roots=plot_output_roots,
                document_cap=document_cap,
                tp_size=tp_size,
                fsdp_size=fsdp_size,
                dp_size=dp_size,
                batch_size=batch_size,
            )
        return

    if mode == "compare":
        if len(specs) < 2:
            raise ValueError("mode='compare' requires at least two checkpoint specifications")
        if not comparison_name or Path(comparison_name).name != comparison_name:
            raise ValueError("mode='compare' requires a valid comparison_name")
        if selected_experiments is None:
            raise ValueError("mode='compare' requires explicit experiments")
        if "heatmap" in plot_types:
            raise ValueError("Heatmap plots are not defined for checkpoint comparison")

    run_dirs = tuple(_result_dir_from_checkpoint_request(spec.checkpoint) for spec in specs)
    if mode == "plot":
        for run_dir in run_dirs:
            plot_checkpoint_results(
                run_dir,
                experiments=selected_experiments,
                plot_types=plot_types,
                plot_output_roots=plot_output_roots,
            )
        return

    compare_checkpoint_results(
        run_dirs,
        experiments=selected_experiments,
        plot_types=plot_types,
        comparison_name=comparison_name,
        plot_output_roots=plot_output_roots,
    )
