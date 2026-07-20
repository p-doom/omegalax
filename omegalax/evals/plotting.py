"""Figures and shared scales for component-separated evaluation summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from omegalax.evals.aggregation import SummaryPoint


_CONDITIONS = {
    "gdn": ("true_gdn", "zero_gdn", "shuffled_gdn"),
    "conv": ("true_conv", "zero_conv", "shuffled_conv"),
}
_CONDITION_STYLES = (
    ("True State", "#0072B2", "o", "-"),
    ("Zero State", "#D55E00", "s", "--"),
    ("Shuffled State", "#009E73", "^", ":"),
)
_MODEL_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#000000")
_MODEL_MARKERS = ("o", "s", "^", "D", "v", "P")


def _validate_metric(experiment: str, metric: str) -> None:
    if experiment not in _CONDITIONS:
        raise ValueError(f"Unsupported experiment: {experiment!r}")
    valid_metrics = {
        "nll",
        f"{experiment}_state_gain",
        f"{experiment}_semantic_gain",
    }
    if metric not in valid_metrics:
        raise ValueError(f"Unsupported metric for {experiment!r}: {metric!r}")


def _metric_label(metric: str) -> str:
    if metric == "nll":
        return "NLL (nats/token)"
    if metric.endswith("_state_gain"):
        return "State Gain (nats/token)"
    return "Semantic Gain (nats/token)"


def _y_limits(values: Sequence[float], metric: str) -> tuple[float, float]:
    if metric != "nll":
        radius = max(abs(float(value)) for value in values)
        radius = radius * 1.05 if radius else 0.025
        return -radius, radius

    lower = min(float(value) for value in values)
    upper = max(float(value) for value in values)
    if upper == lower:
        return lower - 0.025, upper + 0.025
    padding = 0.05 * (upper - lower)
    return lower - padding, upper + padding


def _matching_points(
    points: Sequence[SummaryPoint],
    *,
    experiment: str,
    metric: str,
    view: str,
    doc_num_chunks: int | None,
    condition: str | None,
) -> list[SummaryPoint]:
    return sorted(
        (
            point
            for point in points
            if point.experiment == experiment
            and point.metric == metric
            and point.view == view
            and point.doc_num_chunks == doc_num_chunks
            and point.condition == condition
        ),
        key=lambda point: point.chunk_position,
    )


def _finish_line_figure(
    figure: Figure,
    axis,
    *,
    metric: str,
    title: str | None,
    x_values: Sequence[int],
    y_values: Sequence[float],
    legend_handles: Sequence[Line2D],
) -> None:
    first = min(x_values)
    last = max(x_values)
    axis.set_xlabel("Chunk Position")
    axis.set_ylabel(_metric_label(metric))
    axis.set_xticks(range(first, last + 1))
    axis.set_xlim(first - 0.15, last + 0.15)
    axis.set_ylim(*_y_limits(y_values, metric))
    if title:
        axis.set_title(title)

    figure.subplots_adjust(left=0.12, right=0.68, bottom=0.16, top=0.9 if title else 0.95)
    figure.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.71, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )


def make_single_figure(
    points: Sequence[SummaryPoint],
    *,
    experiment: str,
    metric: str,
    view: str,
    doc_num_chunks: int | None = None,
    title: str | None = None,
) -> Figure:
    """Plot one component metric, separating NLL state conditions."""

    _validate_metric(experiment, metric)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    handles = []
    x_values = []
    y_values = []

    if metric == "nll":
        for condition, (label, color, marker, linestyle) in zip(
            _CONDITIONS[experiment], _CONDITION_STYLES, strict=True
        ):
            selected = _matching_points(
                points,
                experiment=experiment,
                metric=metric,
                view=view,
                doc_num_chunks=doc_num_chunks,
                condition=condition,
            )
            if not selected:
                continue
            (line,) = axis.plot(
                [point.chunk_position for point in selected],
                [point.value for point in selected],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
                label=label,
            )
            handles.append(line)
            x_values.extend(point.chunk_position for point in selected)
            y_values.extend(point.value for point in selected)
    else:
        selected = _matching_points(
            points,
            experiment=experiment,
            metric=metric,
            view=view,
            doc_num_chunks=doc_num_chunks,
            condition=None,
        )
        if selected:
            (line,) = axis.plot(
                [point.chunk_position for point in selected],
                [point.value for point in selected],
                color=_CONDITION_STYLES[0][1],
                marker=_CONDITION_STYLES[0][2],
                linestyle="-",
                linewidth=1.8,
                markersize=5,
                label=_metric_label(metric).split(" (", maxsplit=1)[0],
            )
            handles.append(line)
            x_values.extend(point.chunk_position for point in selected)
            y_values.extend(point.value for point in selected)

    if not handles:
        plt.close(figure)
        raise ValueError(
            f"No summary points for experiment={experiment!r}, metric={metric!r}, "
            f"view={view!r}, doc_num_chunks={doc_num_chunks!r}"
        )
    _finish_line_figure(
        figure,
        axis,
        metric=metric,
        title=title,
        x_values=x_values,
        y_values=y_values,
        legend_handles=handles,
    )
    return figure


def make_comparison_figure(
    models: Mapping[str, tuple[Sequence[SummaryPoint], int]],
    *,
    experiment: str,
    metric: str,
    view: str,
    doc_num_chunks: int | None = None,
    title: str | None = None,
) -> Figure:
    """Compare checkpoints, using only the component's true NLL condition."""

    _validate_metric(experiment, metric)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    handles = []
    x_values = []
    y_values = []
    has_extrapolation = False
    condition = _CONDITIONS[experiment][0] if metric == "nll" else None

    for style_index, label in enumerate(sorted(models)):
        model_points, c_train = models[label]
        selected = _matching_points(
            model_points,
            experiment=experiment,
            metric=metric,
            view=view,
            doc_num_chunks=doc_num_chunks,
            condition=condition,
        )
        if not selected:
            continue
        color = _MODEL_COLORS[style_index % len(_MODEL_COLORS)]
        marker = _MODEL_MARKERS[style_index % len(_MODEL_MARKERS)]
        solid = [point for point in selected if point.chunk_position <= c_train]
        if solid:
            (line,) = axis.plot(
                [point.chunk_position for point in solid],
                [point.value for point in solid],
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=1.8,
                markersize=5,
                label=label,
            )
            handles.append(line)

        if any(point.chunk_position > c_train for point in selected):
            extrapolated = [point for point in selected if point.chunk_position >= c_train]
            axis.plot(
                [point.chunk_position for point in extrapolated],
                [point.value for point in extrapolated],
                color=color,
                marker=marker,
                linestyle="--",
                linewidth=1.8,
                markersize=5,
                label="_nolegend_",
            )
            has_extrapolation = True

        x_values.extend(point.chunk_position for point in selected)
        y_values.extend(point.value for point in selected)

    if not handles:
        plt.close(figure)
        raise ValueError(
            f"No summary points for experiment={experiment!r}, metric={metric!r}, "
            f"view={view!r}, doc_num_chunks={doc_num_chunks!r}"
        )
    if has_extrapolation:
        handles.append(
            Line2D([], [], color="#333333", linestyle="--", label="Extrapolated (dashed)")
        )
    _finish_line_figure(
        figure,
        axis,
        metric=metric,
        title=title,
        x_values=x_values,
        y_values=y_values,
        legend_handles=handles,
    )
    return figure


def make_heatmap_figure(
    points: Sequence[SummaryPoint],
    *,
    experiment: str,
    metric: str,
    condition: str | None = None,
    title: str | None = None,
) -> Figure:
    """Plot exact-length summaries by document length and chunk position."""

    _validate_metric(experiment, metric)
    if metric == "nll" and condition not in _CONDITIONS[experiment]:
        raise ValueError(f"A valid condition is required for NLL heatmaps: {condition!r}")
    if metric != "nll" and condition is not None:
        raise ValueError("condition must be None for gain heatmaps")

    selected = sorted(
        (
            point
            for point in points
            if point.experiment == experiment
            and point.metric == metric
            and point.view == "exact_length"
            and point.condition == condition
            and point.doc_num_chunks is not None
        ),
        key=lambda point: (point.doc_num_chunks, point.chunk_position),
    )
    if not selected:
        raise ValueError(
            f"No exact-length points for experiment={experiment!r}, metric={metric!r}, "
            f"condition={condition!r}"
        )

    lengths = sorted({int(point.doc_num_chunks) for point in selected})
    max_position = max(point.chunk_position for point in selected)
    length_rows = {length: row for row, length in enumerate(lengths)}
    matrix = np.ma.masked_all((len(lengths), max_position), dtype=float)
    for point in selected:
        matrix[length_rows[int(point.doc_num_chunks)], point.chunk_position - 1] = point.value

    values = [point.value for point in selected]
    color_min, color_max = _y_limits(values, metric)
    color_map = plt.get_cmap("viridis" if metric == "nll" else "RdBu_r").with_extremes(
        bad="#EEEEEE"
    )

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    image = axis.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=color_map,
        vmin=color_min,
        vmax=color_max,
    )
    axis.set_xlabel("Chunk Position")
    axis.set_ylabel("Document Length (chunks)")
    axis.set_xticks(
        np.arange(max_position),
        labels=[str(value) for value in range(1, max_position + 1)],
    )
    axis.set_yticks(np.arange(len(lengths)), labels=[str(value) for value in lengths])
    if title:
        axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(_metric_label(metric))
    figure.subplots_adjust(left=0.13, right=0.86, bottom=0.15, top=0.9 if title else 0.95)
    return figure


def save_common_exact_length_scale(
    points: Sequence[SummaryPoint],
    output_dir: str | Path,
    *,
    experiment: str,
    metric: str,
) -> Path:
    """Atomically save a common exact-length color/y scale for one metric."""

    _validate_metric(experiment, metric)
    valid_conditions = _CONDITIONS[experiment] if metric == "nll" else (None,)
    selected = [
        point
        for point in points
        if point.experiment == experiment
        and point.metric == metric
        and point.view == "exact_length"
        and point.condition in valid_conditions
        and point.doc_num_chunks is not None
    ]
    if not selected:
        raise ValueError(f"No exact-length points for experiment={experiment!r}, metric={metric!r}")

    conditions = [
        condition
        for condition in _CONDITIONS[experiment]
        if any(point.condition == condition for point in selected)
    ]
    metadata = {
        "experiment": experiment,
        "metric": metric,
        "conditions": conditions,
        "view": "exact_length",
        "doc_num_chunks": sorted({int(point.doc_num_chunks) for point in selected}),
        "y_limits": list(_y_limits([point.value for point in selected], metric)),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_part = "_all_conditions" if metric == "nll" else ""
    output_path = output_dir / (f"{experiment}_{metric}{condition_part}_exact_length_scale.json")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(metadata, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def save_figure_formats(figure: Figure, output_stem: str | Path) -> tuple[Path, ...]:
    """Save a figure as PDF, SVG, and a 300-DPI PNG."""

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png"))
    for path in paths:
        save_kwargs = {"dpi": 300} if path.suffix == ".png" else {}
        figure.savefig(path, **save_kwargs)
    return paths
