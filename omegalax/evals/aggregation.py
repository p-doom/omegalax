"""Component-separated aggregation for state-usage evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:
    from omegalax.evals.storage import MetricRow


_CONDITIONS = {
    "gdn": ("true_gdn", "zero_gdn", "shuffled_gdn"),
    "conv": ("true_conv", "zero_conv", "shuffled_conv"),
}

_GAINS = {
    "gdn": (
        ("gdn_state_gain", "zero_gdn"),
        ("gdn_semantic_gain", "shuffled_gdn"),
    ),
    "conv": (
        ("conv_state_gain", "zero_conv"),
        ("conv_semantic_gain", "shuffled_conv"),
    ),
}


@dataclass(frozen=True)
class SummaryPoint:
    experiment: str
    metric: str
    view: str
    doc_num_chunks: int | None
    chunk_position: int
    condition: str | None
    value: float
    token_count: float


def aggregate_metrics(
    rows: Iterable[MetricRow],
    *,
    population_counts: Mapping[int, int],
    c_train: int,
) -> tuple[SummaryPoint, ...]:
    rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.experiment,
                row.condition,
                row.bucket_idx,
                row.record_idx,
                row.doc_id,
                row.doc_num_chunks,
                row.chunk_position,
                row.nll_sum,
                row.token_count,
            ),
        )
    )
    sampled_documents: dict[int, set[tuple[int, int, str]]] = {}
    for row in rows:
        if row.experiment not in _CONDITIONS or row.condition not in _CONDITIONS[row.experiment]:
            raise ValueError(f"Unsupported evaluation condition: {row.experiment}/{row.condition}")
        sampled_documents.setdefault(row.doc_num_chunks, set()).add(
            (row.bucket_idx, row.record_idx, row.doc_id)
        )

    totals: dict[tuple[str, str, str, int | None, int], list[float]] = {}
    for row in rows:
        population_weight = population_counts[row.doc_num_chunks] / len(
            sampled_documents[row.doc_num_chunks]
        )
        overview = "in_horizon" if row.doc_num_chunks <= c_train else "beyond_horizon"
        for view, doc_num_chunks, weight in (
            ("exact_length", row.doc_num_chunks, 1.0),
            (overview, None, population_weight),
        ):
            key = (
                row.experiment,
                row.condition,
                view,
                doc_num_chunks,
                row.chunk_position,
            )
            total = totals.setdefault(key, [0.0, 0.0])
            total[0] += weight * row.nll_sum
            total[1] += weight * row.token_count

    points = []
    for (
        experiment,
        condition,
        view,
        doc_num_chunks,
        chunk_position,
    ), (nll_sum, token_count) in totals.items():
        points.append(
            SummaryPoint(
                experiment=experiment,
                metric="nll",
                view=view,
                doc_num_chunks=doc_num_chunks,
                chunk_position=chunk_position,
                condition=condition,
                value=nll_sum / token_count,
                token_count=token_count,
            )
        )

    for experiment, gains in _GAINS.items():
        true_condition = f"true_{experiment}"
        true_keys = [key for key in totals if key[0] == experiment and key[1] == true_condition]
        for metric, ablated_condition in gains:
            for true_key in true_keys:
                _, _, view, doc_num_chunks, chunk_position = true_key
                ablated_key = (
                    experiment,
                    ablated_condition,
                    view,
                    doc_num_chunks,
                    chunk_position,
                )
                if ablated_key not in totals:
                    continue
                true_nll_sum, token_count = totals[true_key]
                ablated_nll_sum, _ = totals[ablated_key]
                points.append(
                    SummaryPoint(
                        experiment=experiment,
                        metric=metric,
                        view=view,
                        doc_num_chunks=doc_num_chunks,
                        chunk_position=chunk_position,
                        condition=None,
                        value=(ablated_nll_sum - true_nll_sum) / token_count,
                        token_count=token_count,
                    )
                )

    return tuple(
        sorted(
            points,
            key=lambda point: (
                point.experiment,
                point.metric,
                point.view,
                -1 if point.doc_num_chunks is None else point.doc_num_chunks,
                point.chunk_position,
                "" if point.condition is None else point.condition,
            ),
        )
    )
