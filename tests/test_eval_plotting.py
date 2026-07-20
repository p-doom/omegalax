"""Tests for component-separated state-usage evaluation figures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

from absl.testing import absltest
import matplotlib

matplotlib.use("Agg")

from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from omegalax.evals.aggregation import SummaryPoint  # noqa: E402
from omegalax.evals.plotting import (  # noqa: E402
    make_comparison_figure,
    make_heatmap_figure,
    make_single_figure,
    save_common_exact_length_scale,
    save_figure_formats,
)
from tests.pretrain_real_data_test_utils import test_temp_dir  # noqa: E402


class EvalPlottingTest(absltest.TestCase):
    def _point(
        self,
        *,
        experiment: str,
        metric: str,
        view: str,
        position: int,
        value: float,
        condition: str | None = None,
        length: int | None = None,
    ) -> SummaryPoint:
        return SummaryPoint(
            experiment=experiment,
            metric=metric,
            view=view,
            doc_num_chunks=length,
            chunk_position=position,
            condition=condition,
            value=value,
            token_count=100,
        )

    def _titles(self, figure) -> set[str]:
        titles = {axis.get_title() for axis in figure.axes if axis.get_title()}
        if figure.get_suptitle():
            titles.add(figure.get_suptitle())
        return titles

    def _assert_small_integer_x(self, axis, first: int, last: int):
        left, right = axis.get_xlim()
        self.assertLess(left, first)
        self.assertGreater(right, last)
        self.assertLessEqual(first - left, 0.25)
        self.assertLessEqual(right - last, 0.25)
        visible_ticks = [tick for tick in axis.get_xticks() if left <= tick <= right]
        self.assertTrue(all(float(tick).is_integer() for tick in visible_ticks))
        self.assertLessEqual(set(range(first, last + 1)), {int(tick) for tick in visible_ticks})

    def _heatmap_matrix(self, axis) -> tuple[np.ma.MaskedArray, object]:
        mappables = [
            artist
            for artist in (*axis.images, *axis.collections)
            if hasattr(artist, "get_array") and artist.get_array() is not None
        ]
        self.assertLen(mappables, 1)
        return np.ma.asarray(mappables[0].get_array()), mappables[0]

    def _legend(self, figure, axis):
        legend = axis.get_legend()
        if legend is not None:
            return legend
        self.assertLen(figure.legends, 1)
        return figure.legends[0]

    def _assert_legend_visible_without_data_overlap(self, figure, axis):
        legend = self._legend(figure, axis)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer)
        figure_box = figure.bbox
        self.assertGreaterEqual(legend_box.x0, figure_box.x0)
        self.assertLessEqual(legend_box.x1, figure_box.x1)
        self.assertGreaterEqual(legend_box.y0, figure_box.y0)
        self.assertLessEqual(legend_box.y1, figure_box.y1)
        for line in axis.lines:
            if not line.get_visible() or len(line.get_xdata()) == 0:
                continue
            data_box = line.get_window_extent(renderer).padded(2.0)
            overlaps = not (
                legend_box.x1 <= data_box.x0
                or legend_box.x0 >= data_box.x1
                or legend_box.y1 <= data_box.y0
                or legend_box.y0 >= data_box.y1
            )
            self.assertFalse(overlaps, line.get_label())

    def _assert_line_axis_labels(self, axis, metric: str):
        self.assertRegex(axis.get_xlabel().lower(), r"chunk|position")
        if metric == "nll":
            self.assertRegex(axis.get_ylabel().lower(), r"nll|negative log|nats")
        else:
            self.assertRegex(
                axis.get_ylabel().lower(),
                r"gain|delta|difference|nats",
            )

    def _assert_tight_absolute_limits(self, limits, values):
        lower, upper = (float(value) for value in limits)
        data_min = min(float(value) for value in values)
        data_max = max(float(value) for value in values)
        data_span = data_max - data_min
        self.assertGreater(data_span, 0.0)
        self.assertLess(lower, data_min)
        self.assertGreater(upper, data_max)
        self.assertLessEqual(data_min - lower, 0.2 * data_span)
        self.assertLessEqual(upper - data_max, 0.2 * data_span)

    def _assert_tight_absolute_y(self, axis, values):
        self._assert_tight_absolute_limits(axis.get_ylim(), values)

    def _assert_tight_delta_y(self, axis, values):
        lower, upper = axis.get_ylim()
        radius = max(abs(float(value)) for value in values)
        self.assertGreater(radius, 0.0)
        self.assertLessEqual(lower, min(float(value) for value in values))
        self.assertGreaterEqual(upper, max(float(value) for value in values))
        self.assertAlmostEqual(-lower, upper)
        self.assertLessEqual(upper, 1.2 * radius)

    def _assert_no_uncertainty_artists(self, axis):
        self.assertFalse(
            any(
                isinstance(collection, (LineCollection, PolyCollection))
                for collection in axis.collections
            )
        )

    def _assert_colorblind_distinguishable(self, colors):
        rgb = np.asarray([mcolors.to_rgb(color) for color in colors], dtype=float)
        self.assertGreaterEqual(len(rgb), 2)
        # Identity plus full-severity Machado protanopia/deuteranopia transforms.
        simulations = (
            np.eye(3),
            np.asarray(
                (
                    (0.152286, 1.052583, -0.204868),
                    (0.114503, 0.786281, 0.099216),
                    (-0.003882, -0.048116, 1.051998),
                )
            ),
            np.asarray(
                (
                    (0.367322, 0.860646, -0.227968),
                    (0.280085, 0.672501, 0.047413),
                    (-0.011820, 0.042940, 0.968881),
                )
            ),
        )
        for simulation in simulations:
            simulated = np.clip(rgb @ simulation.T, 0.0, 1.0)
            distances = [
                np.linalg.norm(simulated[left] - simulated[right])
                for left in range(len(simulated))
                for right in range(left + 1, len(simulated))
            ]
            self.assertGreater(min(distances), 0.12)

    def _assert_labeled_colorbar(self, figure, axis, label_pattern: str):
        self.assertLen(figure.axes, 2)
        colorbar_axis = next(candidate for candidate in figure.axes if candidate is not axis)
        label = colorbar_axis.get_ylabel() or colorbar_axis.get_xlabel()
        self.assertRegex(label.lower(), label_pattern)

    def test_single_figure_filters_component_view_and_length_with_clear_styles(self):
        values = {
            "true_gdn": (2.0, 2.1, 2.2, 2.3),
            "zero_gdn": (2.4, 2.5, 2.6, 2.7),
            "shuffled_gdn": (2.2, 2.3, 2.4, 2.5),
        }
        points = [
            self._point(
                experiment="gdn",
                metric="nll",
                view="exact_length",
                length=4,
                position=position,
                condition=condition,
                value=value,
            )
            for condition, condition_values in reversed(tuple(values.items()))
            for position, value in enumerate(condition_values, start=1)
        ]
        points.extend(
            self._point(
                experiment=experiment,
                metric=metric,
                view=view,
                length=length,
                position=position,
                condition=condition,
                value=sentinel,
            )
            for experiment, metric, view, length, condition, sentinel in (
                ("conv", "nll", "exact_length", 4, "true_conv", 50.0),
                ("gdn", "nll", "in_horizon", None, "true_gdn", 60.0),
                ("gdn", "nll", "exact_length", 5, "true_gdn", 70.0),
                ("gdn", "gdn_state_gain", "exact_length", 4, None, 80.0),
            )
            for position in range(1, 5)
        )

        figure = make_single_figure(
            tuple(points),
            experiment="gdn",
            metric="nll",
            view="exact_length",
            doc_num_chunks=4,
        )
        self.addCleanup(plt.close, figure)

        self.assertLen(figure.axes, 1)
        axis = figure.axes[0]
        lines = [line for line in axis.lines if not line.get_label().startswith("_")]
        self.assertLen(lines, 3)
        self.assertEqual(
            {line.get_label() for line in lines},
            {"True State", "Zero State", "Shuffled State"},
        )
        self.assertTrue(all("gdn" not in line.get_label().lower() for line in lines))
        self.assertEqual(
            {tuple(float(value) for value in line.get_ydata()) for line in lines},
            set(values.values()),
        )
        self.assertLen({line.get_color() for line in lines}, 3)
        self.assertLen({line.get_marker() for line in lines}, 3)
        self.assertLen({line.get_linestyle() for line in lines}, 3)
        self._assert_colorblind_distinguishable(line.get_color() for line in lines)
        self._assert_legend_visible_without_data_overlap(figure, axis)
        self._assert_line_axis_labels(axis, "nll")
        self.assertEmpty(self._titles(figure))
        self._assert_small_integer_x(axis, 1, 4)
        self._assert_tight_absolute_y(
            axis, [value for series in values.values() for value in series]
        )
        self._assert_no_uncertainty_artists(axis)

        alternate_points = tuple(
            self._point(
                experiment="gdn",
                metric="nll",
                view="in_horizon",
                position=position,
                condition=condition,
                value=value,
            )
            for condition, condition_values in values.items()
            for position, value in enumerate(condition_values, start=1)
        )
        titled = make_single_figure(
            alternate_points,
            experiment="gdn",
            metric="nll",
            view="in_horizon",
            title="GDN NLL",
        )
        self.addCleanup(plt.close, titled)
        self.assertIn("GDN NLL", self._titles(titled))
        titled_styles = {
            line.get_label(): (
                mcolors.to_hex(line.get_color()),
                line.get_marker(),
                line.get_linestyle(),
            )
            for line in titled.axes[0].lines
            if not line.get_label().startswith("_")
        }
        self.assertEqual(
            titled_styles,
            {
                line.get_label(): (
                    mcolors.to_hex(line.get_color()),
                    line.get_marker(),
                    line.get_linestyle(),
                )
                for line in lines
            },
        )

    def test_conv_uses_component_true_condition_and_readable_labels(self):
        conditions = {
            "true_conv": (2.0, 2.1, 2.2),
            "zero_conv": (2.4, 2.5, 2.6),
            "shuffled_conv": (2.2, 2.3, 2.4),
        }
        label_patterns = {
            "true_conv": r"true.*state|state.*true",
            "zero_conv": r"zero.*state|state.*zero",
            "shuffled_conv": r"shuffled.*state|state.*shuffled",
        }

        def points_for(offset: float) -> tuple[SummaryPoint, ...]:
            return tuple(
                self._point(
                    experiment="conv",
                    metric="nll",
                    view="exact_length",
                    length=3,
                    position=position,
                    condition=condition,
                    value=value + offset,
                )
                for condition, values in conditions.items()
                for position, value in enumerate(values, start=1)
            )

        points = points_for(0.0)
        figure = make_single_figure(
            points,
            experiment="conv",
            metric="nll",
            view="exact_length",
            doc_num_chunks=3,
        )
        self.addCleanup(plt.close, figure)
        axis = figure.axes[0]
        lines = [line for line in axis.lines if not line.get_label().startswith("_")]
        self.assertLen(lines, len(conditions))
        lines_by_values = {
            tuple(float(value) for value in line.get_ydata()): line for line in lines
        }
        for condition, values in conditions.items():
            line = lines_by_values[values]
            self.assertNotIn("_", line.get_label())
            self.assertRegex(
                line.get_label().lower(),
                label_patterns[condition],
            )
        self._assert_line_axis_labels(axis, "nll")
        self._assert_legend_visible_without_data_overlap(figure, axis)

        comparison = make_comparison_figure(
            {
                "Model B": (points_for(0.2), 6),
                "Model A": (points, 4),
            },
            experiment="conv",
            metric="nll",
            view="exact_length",
            doc_num_chunks=3,
        )
        self.addCleanup(plt.close, comparison)
        comparison_axis = comparison.axes[0]
        model_lines = {
            line.get_label(): line
            for line in comparison_axis.lines
            if line.get_label() in {"Model A", "Model B"}
        }
        self.assertEqual(set(model_lines), {"Model A", "Model B"})
        true_values = conditions["true_conv"]
        self.assertEqual(
            tuple(float(value) for value in model_lines["Model A"].get_ydata()),
            true_values,
        )
        self.assertEqual(
            tuple(float(value) for value in model_lines["Model B"].get_ydata()),
            tuple(value + 0.2 for value in true_values),
        )
        self.assertLen(comparison_axis.lines, 2)
        self._assert_line_axis_labels(comparison_axis, "nll")
        self._assert_legend_visible_without_data_overlap(
            comparison,
            comparison_axis,
        )

    def test_comparison_filters_sentinels_and_uses_each_models_c_train(self):
        target_values = (-0.4, -0.2, 0.0, 0.2, 0.4, 0.3, 0.1, -0.1)

        def checkpoint_points(offset: float) -> tuple[SummaryPoint, ...]:
            points = [
                self._point(
                    experiment="gdn",
                    metric="gdn_state_gain",
                    view="beyond_horizon",
                    position=position,
                    value=value + offset,
                )
                for position, value in enumerate(target_values, start=1)
            ]
            points.extend(
                self._point(
                    experiment=experiment,
                    metric=metric,
                    view=view,
                    position=position,
                    value=50.0,
                )
                for experiment, metric, view in (
                    ("conv", "conv_state_gain", "beyond_horizon"),
                    ("gdn", "gdn_semantic_gain", "beyond_horizon"),
                    ("gdn", "gdn_state_gain", "in_horizon"),
                )
                for position in range(1, 9)
            )
            return tuple(points)

        figure = make_comparison_figure(
            {
                "C4": (checkpoint_points(0.0), 4),
                "C6": (checkpoint_points(0.05), 6),
            },
            experiment="gdn",
            metric="gdn_state_gain",
            view="beyond_horizon",
            title="C4 versus C6",
        )
        self.addCleanup(plt.close, figure)

        self.assertLen(figure.axes, 1)
        axis = figure.axes[0]
        self.assertIn("C4 versus C6", self._titles(figure))
        self._assert_small_integer_x(axis, 1, 8)
        self._assert_tight_delta_y(
            axis,
            (*target_values, *(value + 0.05 for value in target_values)),
        )
        legend = self._legend(figure, axis)
        legend_text = {text.get_text() for text in legend.get_texts()}
        self.assertLessEqual({"C4", "C6"}, legend_text)
        self.assertTrue(
            any("extrapolat" in text.lower() or "dash" in text.lower() for text in legend_text)
        )

        labelled_lines = {
            line.get_label(): line for line in axis.lines if line.get_label() in {"C4", "C6"}
        }
        self.assertEqual(set(labelled_lines), {"C4", "C6"})
        self.assertNotEqual(
            labelled_lines["C4"].get_color(),
            labelled_lines["C6"].get_color(),
        )
        self._assert_colorblind_distinguishable(
            line.get_color() for line in labelled_lines.values()
        )
        self._assert_legend_visible_without_data_overlap(figure, axis)
        self._assert_line_axis_labels(axis, "gdn_state_gain")
        for label, c_train in (("C4", 4), ("C6", 6)):
            solid = labelled_lines[label]
            self.assertEqual(solid.get_linestyle(), "-")
            self.assertEqual(
                [int(value) for value in solid.get_xdata()],
                list(range(1, c_train + 1)),
            )
            dashed = [
                line
                for line in axis.lines
                if line.get_color() == solid.get_color() and line.get_linestyle() == "--"
            ]
            self.assertLen(dashed, 1)
            self.assertEqual(
                [int(value) for value in dashed[0].get_xdata()],
                list(range(c_train, 9)),
            )
            self.assertTrue(
                all(
                    abs(float(value)) < 1.0
                    for value in (*solid.get_ydata(), *dashed[0].get_ydata())
                )
            )

        for line in axis.lines:
            xdata = [float(value) for value in line.get_xdata()]
            if len(xdata) > 1:
                self.assertNotEqual(min(xdata), max(xdata), "comparison must not draw an axvline")
        self._assert_no_uncertainty_artists(axis)

    def test_comparison_nll_uses_only_true_condition_and_filters_exact_length(self):
        true_values = {
            "C4": (2.0, 2.1, 2.2, 2.3),
            "C6": (2.4, 2.5, 2.6, 2.7),
            "C8": (2.8, 2.9, 3.0, 3.1),
        }

        def checkpoint_points(label: str) -> tuple[SummaryPoint, ...]:
            points = [
                self._point(
                    experiment="gdn",
                    metric="nll",
                    view="exact_length",
                    length=4,
                    position=position,
                    condition="true_gdn",
                    value=value,
                )
                for position, value in enumerate(true_values[label], start=1)
            ]
            points.extend(
                self._point(
                    experiment="gdn",
                    metric="nll",
                    view="exact_length",
                    length=4,
                    position=position,
                    condition=condition,
                    value=sentinel,
                )
                for condition, sentinel in (
                    ("zero_gdn", 20.0),
                    ("shuffled_gdn", 30.0),
                )
                for position in range(1, 5)
            )
            points.extend(
                self._point(
                    experiment=experiment,
                    metric="nll",
                    view=view,
                    length=length,
                    position=position,
                    condition=condition,
                    value=sentinel,
                )
                for experiment, view, length, condition, sentinel in (
                    ("gdn", "exact_length", 5, "true_gdn", 40.0),
                    ("gdn", "in_horizon", None, "true_gdn", 50.0),
                    ("conv", "exact_length", 4, "true_conv", 60.0),
                )
                for position in range(1, 5)
            )
            return tuple(points)

        figure = make_comparison_figure(
            {
                "C8": (checkpoint_points("C8"), 8),
                "C6": (checkpoint_points("C6"), 6),
                "C4": (checkpoint_points("C4"), 4),
            },
            experiment="gdn",
            metric="nll",
            view="exact_length",
            doc_num_chunks=4,
        )
        self.addCleanup(plt.close, figure)

        self.assertLen(figure.axes, 1)
        axis = figure.axes[0]
        model_lines = {
            line.get_label(): line for line in axis.lines if line.get_label() in true_values
        }
        self.assertEqual(set(model_lines), set(true_values))
        self.assertLen(axis.lines, 3)
        for label, line in model_lines.items():
            self.assertEqual(
                tuple(float(value) for value in line.get_ydata()),
                true_values[label],
            )
            self.assertEqual(
                tuple(int(value) for value in line.get_xdata()),
                (1, 2, 3, 4),
            )
        self._assert_tight_absolute_y(
            axis,
            [value for values in true_values.values() for value in values],
        )
        self._assert_colorblind_distinguishable(line.get_color() for line in model_lines.values())
        self._assert_legend_visible_without_data_overlap(figure, axis)
        self._assert_line_axis_labels(axis, "nll")
        self.assertEmpty(self._titles(figure))
        self._assert_no_uncertainty_artists(axis)

        gain_points = tuple(
            self._point(
                experiment="gdn",
                metric="gdn_state_gain",
                view="exact_length",
                length=4,
                position=position,
                value=value,
            )
            for position, value in enumerate((-0.2, -0.1, 0.0, 0.1), start=1)
        )
        gain_figure = make_comparison_figure(
            {
                "C4": (gain_points, 4),
                "C8": (gain_points, 8),
                "C6": (gain_points, 6),
            },
            experiment="gdn",
            metric="gdn_state_gain",
            view="exact_length",
            doc_num_chunks=4,
        )
        self.addCleanup(plt.close, gain_figure)
        gain_styles = {
            line.get_label(): (
                mcolors.to_hex(line.get_color()),
                line.get_marker(),
                line.get_linestyle(),
            )
            for line in gain_figure.axes[0].lines
            if line.get_label() in true_values
        }
        self.assertEqual(
            gain_styles,
            {
                label: (
                    mcolors.to_hex(line.get_color()),
                    line.get_marker(),
                    line.get_linestyle(),
                )
                for label, line in model_lines.items()
            },
        )

    def test_heatmap_filters_component_metric_and_absolute_condition(self):
        points = []
        for length in (2, 3, 4):
            for position in range(1, length + 1):
                true_value = 0.1 * length + 0.01 * position
                zero_value = true_value + 1.0
                shuffled_value = true_value + 2.0
                points.extend(
                    (
                        self._point(
                            experiment="gdn",
                            metric="nll",
                            view="exact_length",
                            length=length,
                            position=position,
                            condition="true_gdn",
                            value=true_value,
                        ),
                        self._point(
                            experiment="gdn",
                            metric="nll",
                            view="exact_length",
                            length=length,
                            position=position,
                            condition="zero_gdn",
                            value=zero_value,
                        ),
                        self._point(
                            experiment="gdn",
                            metric="nll",
                            view="exact_length",
                            length=length,
                            position=position,
                            condition="shuffled_gdn",
                            value=shuffled_value,
                        ),
                        self._point(
                            experiment="conv",
                            metric="nll",
                            view="exact_length",
                            length=length,
                            position=position,
                            condition="true_conv",
                            value=50.0,
                        ),
                    )
                )

        with self.assertRaisesRegex(ValueError, "condition"):
            make_heatmap_figure(
                tuple(points),
                experiment="gdn",
                metric="nll",
            )

        true_figure = make_heatmap_figure(
            tuple(points),
            experiment="gdn",
            metric="nll",
            condition="true_gdn",
        )
        zero_figure = make_heatmap_figure(
            tuple(points),
            experiment="gdn",
            metric="nll",
            condition="zero_gdn",
        )
        shuffled_figure = make_heatmap_figure(
            tuple(points),
            experiment="gdn",
            metric="nll",
            condition="shuffled_gdn",
        )
        self.addCleanup(plt.close, true_figure)
        self.addCleanup(plt.close, zero_figure)
        self.addCleanup(plt.close, shuffled_figure)

        def expected_nll_matrix(offset: float) -> np.ma.MaskedArray:
            expected = np.ma.masked_all((3, 4), dtype=float)
            for row, length in enumerate((2, 3, 4)):
                for column, position in enumerate(range(1, length + 1)):
                    expected[row, column] = 0.1 * length + 0.01 * position + offset
            return expected

        for figure, offset in (
            (true_figure, 0.0),
            (zero_figure, 1.0),
            (shuffled_figure, 2.0),
        ):
            axis = figure.axes[0]
            matrix, _ = self._heatmap_matrix(axis)
            expected = expected_nll_matrix(offset)
            self.assertEqual(matrix.shape, (3, 4))
            np.testing.assert_array_equal(
                np.ma.getmaskarray(matrix),
                np.ma.getmaskarray(expected),
            )
            np.testing.assert_allclose(
                matrix.filled(np.nan),
                expected.filled(np.nan),
                equal_nan=True,
            )
            self._assert_labeled_colorbar(figure, axis, r"nll|negative log|nats")

        true_axis = true_figure.axes[0]
        self.assertEmpty(self._titles(true_figure))
        self.assertRegex(true_axis.get_xlabel().lower(), r"chunk|position")
        self.assertRegex(true_axis.get_ylabel().lower(), r"document|length")
        self.assertEqual(
            [label.get_text() for label in true_axis.get_xticklabels() if label.get_visible()],
            ["1", "2", "3", "4"],
        )
        self.assertEqual(
            [label.get_text() for label in true_axis.get_yticklabels() if label.get_visible()],
            ["2", "3", "4"],
        )

        def gain_value(length: int, position: int) -> float:
            if position == 1:
                return 0.0
            return 0.2 * (position - (length + 1) / 2)

        gain_points = [
            self._point(
                experiment="gdn",
                metric="gdn_state_gain",
                view="exact_length",
                length=length,
                position=position,
                value=gain_value(length, position),
            )
            for length in (2, 3, 4)
            for position in range(1, length + 1)
        ]
        gain_points.extend(
            self._point(
                experiment=experiment,
                metric=metric,
                view=view,
                length=4,
                position=position,
                value=50.0,
            )
            for experiment, metric, view in (
                ("conv", "conv_state_gain", "exact_length"),
                ("gdn", "gdn_semantic_gain", "exact_length"),
                ("gdn", "gdn_state_gain", "beyond_horizon"),
            )
            for position in range(1, 5)
        )
        gain_figure = make_heatmap_figure(
            tuple(gain_points),
            experiment="gdn",
            metric="gdn_state_gain",
            condition=None,
        )
        self.addCleanup(plt.close, gain_figure)
        gain_axis = gain_figure.axes[0]
        gain_matrix, gain_mappable = self._heatmap_matrix(gain_axis)
        expected_gain = np.ma.masked_all((3, 4), dtype=float)
        for row, length in enumerate((2, 3, 4)):
            for column, position in enumerate(range(1, length + 1)):
                expected_gain[row, column] = gain_value(length, position)
        self.assertEqual(gain_matrix.shape, (3, 4))
        np.testing.assert_array_equal(
            np.ma.getmaskarray(gain_matrix),
            np.ma.getmaskarray(expected_gain),
        )
        np.testing.assert_allclose(
            gain_matrix.filled(np.nan),
            expected_gain.filled(np.nan),
            equal_nan=True,
        )
        color_min, color_max = gain_mappable.get_clim()
        self.assertAlmostEqual(-color_min, color_max)
        radius = max(abs(float(value)) for value in expected_gain.compressed())
        self.assertGreaterEqual(color_max, radius)
        self.assertLessEqual(color_max, 1.2 * radius)
        self._assert_labeled_colorbar(
            gain_figure,
            gain_axis,
            r"gain|delta|difference|nats",
        )

    def test_nonconstant_single_gain_has_tight_symmetric_labeled_axes(self):
        values = (0.0, -0.25, 0.35, 0.1)
        points = tuple(
            self._point(
                experiment="gdn",
                metric="gdn_state_gain",
                view="in_horizon",
                position=position,
                value=value,
            )
            for position, value in enumerate(values, start=1)
        )
        figure = make_single_figure(
            points,
            experiment="gdn",
            metric="gdn_state_gain",
            view="in_horizon",
        )
        self.addCleanup(plt.close, figure)

        self.assertLen(figure.axes, 1)
        axis = figure.axes[0]
        labelled_lines = [line for line in axis.lines if not line.get_label().startswith("_")]
        self.assertLen(labelled_lines, 1)
        self.assertNotIn("_", labelled_lines[0].get_label())
        self.assertRegex(
            labelled_lines[0].get_label().lower(),
            r"state.*gain|gain.*state",
        )
        legend = self._legend(figure, axis)
        self.assertIn(
            labelled_lines[0].get_label(),
            {text.get_text() for text in legend.get_texts()},
        )
        self._assert_legend_visible_without_data_overlap(figure, axis)
        self._assert_tight_delta_y(axis, values)
        self._assert_small_integer_x(axis, 1, 4)
        self._assert_line_axis_labels(axis, "gdn_state_gain")
        self._assert_no_uncertainty_artists(axis)
        self.assertEmpty(self._titles(figure))

    def test_constant_y_fallback_and_atomic_common_exact_length_scale(self):
        constant_nll_points = tuple(
            self._point(
                experiment="gdn",
                metric="nll",
                view="in_horizon",
                position=position,
                condition=condition,
                value=2.5,
            )
            for condition in ("true_gdn", "zero_gdn", "shuffled_gdn")
            for position in range(1, 4)
        )
        constant_nll_figure = make_single_figure(
            constant_nll_points,
            experiment="gdn",
            metric="nll",
            view="in_horizon",
        )
        self.addCleanup(plt.close, constant_nll_figure)
        nll_lower, nll_upper = constant_nll_figure.axes[0].get_ylim()
        self.assertLess(nll_lower, 2.5)
        self.assertGreater(nll_upper, 2.5)
        self.assertAlmostEqual(2.5 - nll_lower, nll_upper - 2.5)
        self.assertLessEqual(nll_upper - nll_lower, 0.1)

        constant_gain_points = tuple(
            self._point(
                experiment="gdn",
                metric="gdn_state_gain",
                view="in_horizon",
                position=position,
                value=0.0,
            )
            for position in range(1, 4)
        )
        constant_gain_figure = make_single_figure(
            constant_gain_points,
            experiment="gdn",
            metric="gdn_state_gain",
            view="in_horizon",
        )
        self.addCleanup(plt.close, constant_gain_figure)
        gain_lower, gain_upper = constant_gain_figure.axes[0].get_ylim()
        self.assertLess(gain_lower, 0.0)
        self.assertGreater(gain_upper, 0.0)
        self.assertAlmostEqual(-gain_lower, gain_upper)
        self.assertLessEqual(gain_upper - gain_lower, 0.1)

        exact_values = {
            "true_gdn": {
                2: (1.0, 1.1),
                3: (4.0, 4.1, 4.2),
            },
            "zero_gdn": {
                2: (1.2, 1.3),
                3: (4.2, 4.3, 4.4),
            },
            "shuffled_gdn": {
                2: (1.1, 1.2),
                3: (4.1, 4.2, 4.3),
            },
        }
        exact_points = [
            self._point(
                experiment="gdn",
                metric="nll",
                view="exact_length",
                length=length,
                position=position,
                condition=condition,
                value=value,
            )
            for condition, values_by_length in exact_values.items()
            for length, values in values_by_length.items()
            for position, value in enumerate(values, start=1)
        ]
        exact_points.extend(
            self._point(
                experiment="conv",
                metric="nll",
                view="exact_length",
                length=length,
                position=position,
                condition="true_conv",
                value=50.0,
            )
            for length in (2, 3)
            for position in range(1, length + 1)
        )
        exact_points.extend(
            self._point(
                experiment="gdn",
                metric="gdn_state_gain",
                view="exact_length",
                length=length,
                position=position,
                value=60.0,
            )
            for length in (2, 3)
            for position in range(1, length + 1)
        )
        exact_points.extend(
            self._point(
                experiment="gdn",
                metric="nll",
                view="in_horizon",
                position=position,
                condition="true_gdn",
                value=70.0,
            )
            for position in range(1, 4)
        )

        primary_spans = []
        for length in (2, 3):
            figure = make_single_figure(
                tuple(exact_points),
                experiment="gdn",
                metric="nll",
                view="exact_length",
                doc_num_chunks=length,
            )
            self.addCleanup(plt.close, figure)
            length_values = [
                value
                for values_by_length in exact_values.values()
                for value in values_by_length[length]
            ]
            self._assert_tight_absolute_y(figure.axes[0], length_values)
            lower, upper = figure.axes[0].get_ylim()
            primary_spans.append(upper - lower)

        with test_temp_dir() as tmp:
            summary_dir = Path(tmp) / "summary"
            expected_path = summary_dir / "gdn_nll_all_conditions_exact_length_scale.json"
            with (
                mock.patch("os.replace", side_effect=OSError("injected scale publish failure")),
                self.assertRaisesRegex(OSError, "injected scale publish failure"),
            ):
                save_common_exact_length_scale(
                    tuple(exact_points),
                    summary_dir,
                    experiment="gdn",
                    metric="nll",
                )
            self.assertFalse(expected_path.exists())
            self.assertEmpty(tuple(path for path in summary_dir.rglob("*") if path.is_file()))

            with mock.patch("os.replace", wraps=os.replace) as atomic_replace:
                scale_path = save_common_exact_length_scale(
                    tuple(exact_points),
                    summary_dir,
                    experiment="gdn",
                    metric="nll",
                )
            self.assertEqual(scale_path, expected_path)
            scale_replaces = [
                call
                for call in atomic_replace.call_args_list
                if Path(call.args[1]) == expected_path
            ]
            self.assertLen(scale_replaces, 1)
            source, destination = (Path(value) for value in scale_replaces[0].args[:2])
            self.assertEqual(source.parent, destination.parent)
            self.assertNotEqual(source, destination)

            metadata = json.loads(scale_path.read_text())
            self.assertEqual(metadata["experiment"], "gdn")
            self.assertEqual(metadata["metric"], "nll")
            self.assertNotIn("condition", metadata)
            self.assertEqual(
                set(metadata["conditions"]),
                {"true_gdn", "zero_gdn", "shuffled_gdn"},
            )
            self.assertEqual(metadata["view"], "exact_length")
            self.assertEqual(metadata["doc_num_chunks"], [2, 3])
            common_limits = tuple(metadata["y_limits"])
            all_exact_values = [
                value
                for values_by_length in exact_values.values()
                for values in values_by_length.values()
                for value in values
            ]
            self._assert_tight_absolute_limits(common_limits, all_exact_values)
            self.assertTrue(
                all(span < common_limits[1] - common_limits[0] for span in primary_spans)
            )

            gain_scale_points = tuple(
                self._point(
                    experiment="gdn",
                    metric="gdn_state_gain",
                    view="exact_length",
                    length=length,
                    position=position,
                    value=value,
                )
                for length, values in {
                    2: (0.0, -0.2),
                    3: (0.0, 0.1, 0.4),
                }.items()
                for position, value in enumerate(values, start=1)
            )
            gain_scale_path = save_common_exact_length_scale(
                gain_scale_points,
                summary_dir,
                experiment="gdn",
                metric="gdn_state_gain",
            )
            self.assertEqual(
                gain_scale_path,
                summary_dir / "gdn_gdn_state_gain_exact_length_scale.json",
            )
            gain_metadata = json.loads(gain_scale_path.read_text())
            self.assertNotIn("condition", gain_metadata)
            gain_scale_lower, gain_scale_upper = gain_metadata["y_limits"]
            self.assertAlmostEqual(-gain_scale_lower, gain_scale_upper)
            self.assertGreaterEqual(gain_scale_upper, 0.4)
            self.assertLessEqual(gain_scale_upper, 0.48)

    def test_save_figure_formats_writes_pdf_svg_and_300_dpi_png(self):
        points = tuple(
            self._point(
                experiment="gdn",
                metric="gdn_state_gain",
                view="in_horizon",
                position=position,
                value=value,
            )
            for position, value in enumerate((0.1, 0.2, 0.3), start=1)
        )
        figure = make_single_figure(
            points,
            experiment="gdn",
            metric="gdn_state_gain",
            view="in_horizon",
        )
        self.addCleanup(plt.close, figure)

        with test_temp_dir() as tmp:
            output_stem = Path(tmp) / "plots" / "gdn" / "gdn_state_gain"
            paths = save_figure_formats(figure, output_stem)

            self.assertIsInstance(paths, tuple)
            self.assertEqual(
                set(paths),
                {output_stem.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")},
            )
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))
            self.assertTrue(output_stem.with_suffix(".pdf").read_bytes().startswith(b"%PDF"))
            self.assertIn("<svg", output_stem.with_suffix(".svg").read_text())
            with Image.open(output_stem.with_suffix(".png")) as image:
                self.assertAlmostEqual(image.info["dpi"][0], 300.0, delta=1.0)
                self.assertAlmostEqual(image.info["dpi"][1], 300.0, delta=1.0)


if __name__ == "__main__":
    absltest.main()
