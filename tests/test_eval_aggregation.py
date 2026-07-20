"""Tests for component-separated state-usage metric aggregation."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields

from absl.testing import absltest

from omegalax.evals.aggregation import SummaryPoint, aggregate_metrics
from omegalax.evals.storage import MetricRow


_DOCUMENTS = (
    (2, "l2-a", 2, 1.0, 0.5),
    (2, "l2-b", 8, 3.0, 2.0),
    (3, "l3-a", 1, 4.0, 0.25),
    (3, "l3-b", 4, 6.0, 1.0),
    (3, "l3-c", 7, 8.0, 3.0),
    (4, "l4-a", 2, 2.0, 0.75),
    (4, "l4-b", 5, 5.0, 2.5),
    (5, "l5-a", 3, 7.0, 0.5),
    (5, "l5-b", 5, 9.0, 1.5),
    (5, "l5-c", 7, 11.0, 4.0),
)

_CONDITION_OFFSETS = {
    "gdn": {
        "true_gdn": lambda position, length, document_effect: 0.0,
        "zero_gdn": lambda position, length, document_effect: (
            0.0 if position == 1 else document_effect * 0.05 * length * (position - 1)
        ),
        "shuffled_gdn": lambda position, length, document_effect: (
            0.0
            if position == 1
            else document_effect * (0.1 * length + 0.03 * length * (position - 2))
        ),
    },
    "conv": {
        "true_conv": lambda position, length, document_effect: 0.0,
        "zero_conv": lambda position, length, document_effect: (
            0.0 if position == 1 else document_effect * 0.04 * length * (position - 1)
        ),
        "shuffled_conv": lambda position, length, document_effect: (
            0.0
            if position == 1
            else document_effect * (0.12 * length - 0.02 * length * (position - 2))
        ),
    },
}


class EvalAggregationTest(absltest.TestCase):
    def _rows(self) -> tuple[MetricRow, ...]:
        experiment_shift = {"gdn": 0.0, "conv": 10.0}
        rows = []
        for record_idx, (
            length,
            doc_id,
            token_count,
            base_nll,
            document_effect,
        ) in enumerate(_DOCUMENTS):
            for position in range(1, length + 1):
                for experiment, condition_offsets in _CONDITION_OFFSETS.items():
                    true_nll = base_nll + experiment_shift[experiment] + 0.1 * position
                    for condition, offset_fn in condition_offsets.items():
                        rows.append(
                            MetricRow(
                                experiment=experiment,
                                condition=condition,
                                bucket_idx=0,
                                record_idx=record_idx,
                                doc_id=doc_id,
                                doc_num_chunks=length,
                                chunk_position=position,
                                nll_sum=(true_nll + offset_fn(position, length, document_effect))
                                * token_count,
                                token_count=token_count,
                            )
                        )
        return tuple(rows)

    def _point(
        self,
        points: tuple[SummaryPoint, ...],
        *,
        experiment: str,
        metric: str,
        view: str,
        condition: str | None,
        chunk_position: int,
        doc_num_chunks: int | None,
    ) -> SummaryPoint:
        matches = [
            point
            for point in points
            if point.experiment == experiment
            and point.metric == metric
            and point.view == view
            and point.condition == condition
            and point.chunk_position == chunk_position
            and point.doc_num_chunks == doc_num_chunks
        ]
        self.assertLen(matches, 1)
        return matches[0]

    def _expected_gain(
        self,
        experiment: str,
        condition: str,
        position: int,
        lengths: tuple[int, ...],
        population_counts: dict[int, int] | None = None,
    ) -> tuple[float, float]:
        sampled_documents = Counter(length for length, *_ in _DOCUMENTS)
        weighted_nll_sum = 0.0
        weighted_token_count = 0.0
        for length, _doc_id, token_count, _base_nll, document_effect in _DOCUMENTS:
            if length not in lengths or position > length:
                continue
            population_weight = (
                1.0
                if population_counts is None
                else population_counts[length] / sampled_documents[length]
            )
            weighted_nll_sum += (
                population_weight
                * token_count
                * _CONDITION_OFFSETS[experiment][condition](
                    position,
                    length,
                    document_effect,
                )
            )
            weighted_token_count += population_weight * token_count
        return weighted_nll_sum / weighted_token_count, weighted_token_count

    def test_non_integer_population_and_token_weighting_stays_component_separated(self):
        points = aggregate_metrics(
            self._rows(),
            population_counts={2: 3, 3: 5, 4: 5, 5: 4},
            c_train=3,
        )

        self.assertIsInstance(points, tuple)
        expected_conditions = {
            experiment: set(condition_offsets)
            for experiment, condition_offsets in _CONDITION_OFFSETS.items()
        }
        for experiment, conditions in expected_conditions.items():
            self.assertEqual(
                {
                    point.condition
                    for point in points
                    if point.experiment == experiment and point.metric == "nll"
                },
                conditions,
            )

        exact = self._point(
            points,
            experiment="gdn",
            metric="nll",
            view="exact_length",
            condition="true_gdn",
            doc_num_chunks=3,
            chunk_position=2,
        )
        self.assertAlmostEqual(exact.value, (1 * 4.2 + 4 * 6.2 + 7 * 8.2) / 12)
        self.assertEqual(exact.token_count, 12)

        in_horizon = self._point(
            points,
            experiment="gdn",
            metric="nll",
            view="in_horizon",
            condition="true_gdn",
            doc_num_chunks=None,
            chunk_position=2,
        )
        # w_2=3/2 and w_3=5/3, applied to both NLL sums and token counts.
        self.assertAlmostEqual(
            in_horizon.value,
            ((3 / 2) * 28 + (5 / 3) * 86.4) / ((3 / 2) * 10 + (5 / 3) * 12),
        )
        self.assertAlmostEqual(in_horizon.token_count, 35)
        self.assertNotAlmostEqual(in_horizon.value, (28 + 86.4) / (10 + 12))
        self.assertNotAlmostEqual(
            in_horizon.value,
            ((3 / 2) * (28 / 10) + (5 / 3) * (86.4 / 12)) / ((3 / 2) + (5 / 3)),
        )

        beyond_horizon = self._point(
            points,
            experiment="gdn",
            metric="nll",
            view="beyond_horizon",
            condition="true_gdn",
            doc_num_chunks=None,
            chunk_position=2,
        )
        # w_4=5/2 and w_5=4/3 with different sampled token totals.
        self.assertAlmostEqual(
            beyond_horizon.value,
            ((5 / 2) * 30.4 + (4 / 3) * 146) / ((5 / 2) * 7 + (4 / 3) * 15),
        )
        self.assertAlmostEqual(beyond_horizon.token_count, 37.5)
        self.assertNotAlmostEqual(beyond_horizon.value, (30.4 + 146) / (7 + 15))
        self.assertNotAlmostEqual(
            beyond_horizon.value,
            ((5 / 2) * (30.4 / 7) + (4 / 3) * (146 / 15)) / ((5 / 2) + (4 / 3)),
        )

        conv = self._point(
            points,
            experiment="conv",
            metric="nll",
            view="exact_length",
            condition="true_conv",
            doc_num_chunks=3,
            chunk_position=2,
        )
        self.assertAlmostEqual(conv.value, exact.value + 10.0)

    def test_all_component_gains_keep_position_dependent_effects(self):
        points = aggregate_metrics(
            self._rows(),
            population_counts={2: 3, 3: 5, 4: 5, 5: 4},
            c_train=3,
        )
        expected_gains = {
            ("gdn", "gdn_state_gain"): "zero_gdn",
            ("gdn", "gdn_semantic_gain"): "shuffled_gdn",
            ("conv", "conv_state_gain"): "zero_conv",
            ("conv", "conv_semantic_gain"): "shuffled_conv",
        }

        self.assertEqual(
            {point.metric for point in points if point.metric != "nll"},
            {metric for _, metric in expected_gains},
        )
        chunk_one_gains = [
            point for point in points if point.metric != "nll" and point.chunk_position == 1
        ]
        self.assertLen(chunk_one_gains, len(expected_gains) * 6)
        self.assertTrue(all(point.value == 0.0 for point in chunk_one_gains))

        for experiment, conditions in _CONDITION_OFFSETS.items():
            chunk_one_nll = [
                point
                for point in points
                if point.experiment == experiment
                and point.metric == "nll"
                and point.chunk_position == 1
            ]
            grouped_values = {}
            for point in chunk_one_nll:
                key = (point.view, point.doc_num_chunks)
                grouped_values.setdefault(key, {})[point.condition] = point.value
            self.assertLen(grouped_values, 6)
            for condition_values in grouped_values.values():
                self.assertEqual(set(condition_values), set(conditions))
                self.assertLen(set(condition_values.values()), 1)

        population_counts = {2: 3, 3: 5, 4: 5, 5: 4}
        for (experiment, metric), ablated_condition in expected_gains.items():
            for length in (2, 3, 4, 5):
                for position in range(1, length + 1):
                    exact = self._point(
                        points,
                        experiment=experiment,
                        metric=metric,
                        view="exact_length",
                        condition=None,
                        doc_num_chunks=length,
                        chunk_position=position,
                    )
                    expected_value, expected_tokens = self._expected_gain(
                        experiment,
                        ablated_condition,
                        position,
                        (length,),
                    )
                    self.assertAlmostEqual(exact.value, expected_value)
                    self.assertAlmostEqual(exact.token_count, expected_tokens)

            for view, lengths in (
                ("in_horizon", (2, 3)),
                ("beyond_horizon", (4, 5)),
            ):
                for position in range(1, max(lengths) + 1):
                    overview = self._point(
                        points,
                        experiment=experiment,
                        metric=metric,
                        view=view,
                        condition=None,
                        doc_num_chunks=None,
                        chunk_position=position,
                    )
                    expected_value, expected_tokens = self._expected_gain(
                        experiment,
                        ablated_condition,
                        position,
                        lengths,
                        population_counts,
                    )
                    self.assertAlmostEqual(overview.value, expected_value)
                    self.assertAlmostEqual(overview.token_count, expected_tokens)

        exact_l3 = self._point(
            points,
            experiment="gdn",
            metric="gdn_state_gain",
            view="exact_length",
            condition=None,
            doc_num_chunks=3,
            chunk_position=2,
        )
        l3_document_offsets = [
            _CONDITION_OFFSETS["gdn"]["zero_gdn"](2, length, document_effect)
            for length, _doc_id, _tokens, _base_nll, document_effect in _DOCUMENTS
            if length == 3
        ]
        self.assertNotAlmostEqual(
            exact_l3.value,
            sum(l3_document_offsets) / len(l3_document_offsets),
        )
        self.assertEqual(exact_l3.token_count, 12)

        in_horizon = self._point(
            points,
            experiment="gdn",
            metric="gdn_state_gain",
            view="in_horizon",
            condition=None,
            doc_num_chunks=None,
            chunk_position=2,
        )
        raw_token_average, _ = self._expected_gain(
            "gdn",
            "zero_gdn",
            2,
            (2, 3),
        )
        sampled_documents = Counter(length for length, *_ in _DOCUMENTS)
        per_length = {
            length: self._expected_gain("gdn", "zero_gdn", 2, (length,))[0] for length in (2, 3)
        }
        population_only_average = sum(
            population_counts[length] / sampled_documents[length] * per_length[length]
            for length in (2, 3)
        ) / sum(population_counts[length] / sampled_documents[length] for length in (2, 3))
        self.assertNotAlmostEqual(in_horizon.value, raw_token_average)
        self.assertNotAlmostEqual(in_horizon.value, population_only_average)

    def test_every_view_contains_all_reachable_chunk_positions_and_no_ci_fields(self):
        points = aggregate_metrics(
            self._rows(),
            population_counts={2: 3, 3: 5, 4: 5, 5: 4},
            c_train=3,
        )

        gain_metrics = {
            "gdn": ("gdn_state_gain", "gdn_semantic_gain"),
            "conv": ("conv_state_gain", "conv_semantic_gain"),
        }
        expected_nll_keys = []
        expected_gain_keys = []
        for experiment, condition_offsets in _CONDITION_OFFSETS.items():
            for condition in condition_offsets:
                expected_nll_keys.extend(
                    (experiment, condition, "in_horizon", None, position)
                    for position in range(1, 4)
                )
                expected_nll_keys.extend(
                    (experiment, condition, "beyond_horizon", None, position)
                    for position in range(1, 6)
                )
                for length in (2, 3, 4, 5):
                    expected_nll_keys.extend(
                        (experiment, condition, "exact_length", length, position)
                        for position in range(1, length + 1)
                    )
            for metric in gain_metrics[experiment]:
                expected_gain_keys.extend(
                    (experiment, metric, "in_horizon", None, position) for position in range(1, 4)
                )
                expected_gain_keys.extend(
                    (experiment, metric, "beyond_horizon", None, position)
                    for position in range(1, 6)
                )
                for length in (2, 3, 4, 5):
                    expected_gain_keys.extend(
                        (experiment, metric, "exact_length", length, position)
                        for position in range(1, length + 1)
                    )

        actual_nll_keys = Counter(
            (
                point.experiment,
                point.condition,
                point.view,
                point.doc_num_chunks,
                point.chunk_position,
            )
            for point in points
            if point.metric == "nll"
        )
        actual_gain_keys = Counter(
            (
                point.experiment,
                point.metric,
                point.view,
                point.doc_num_chunks,
                point.chunk_position,
            )
            for point in points
            if point.metric != "nll"
        )
        self.assertEqual(actual_nll_keys, Counter(expected_nll_keys))
        self.assertEqual(actual_gain_keys, Counter(expected_gain_keys))

        summary_fields = {field.name.lower() for field in fields(SummaryPoint)}
        self.assertEqual(
            summary_fields,
            {
                "experiment",
                "metric",
                "view",
                "doc_num_chunks",
                "chunk_position",
                "condition",
                "value",
                "token_count",
            },
        )
        self.assertFalse(
            any(
                "confidence" in name
                or name.startswith("ci")
                or name.endswith("_ci")
                or name in {"lower", "upper", "stderr", "standard_error"}
                for name in summary_fields
            )
        )


if __name__ == "__main__":
    absltest.main()
