"""Unit tests for _consensus_build_multicomponent_frequency_consensus."""

from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from pgmuvi.lightcurve import Lightcurve


def _make_minimal_multiband_lightcurve():
    t = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    wl = np.array([1.0, 1.0, 2.0, 2.0], dtype=float)
    x = np.column_stack([t, wl])
    y = np.array([0.0, 0.1, -0.1, 0.0], dtype=float)
    band = np.array(["A", "A", "B", "B"])
    return Lightcurve(x, y, band=band)


class TestConsensusMulticomponentFrequencyConsensus(unittest.TestCase):
    def setUp(self):
        self.lc = _make_minimal_multiband_lightcurve()

    def test_accepted_clusters_populate_consensus_arrays(self):
        clusters = [
            {
                "cluster_id": 10,
                "accepted": True,
                "rejection_reasons": [],
                "member_bands": ["J", "K"],
                "n_member_bands": 2,
                "center_frequency": 1.0,
                "center_period": 1.0,
                "log_center_frequency": 0.0,
                "frequency_scatter": 0.01,
                "log_frequency_scatter": 0.01,
                "members": [
                    {"band_name": "J", "frequency": 1.0, "peak_power": 3.0, "significant": True},
                    {"band_name": "K", "frequency": 1.01, "peak_power": 2.0, "significant": True},
                ],
                "duplicate_band_candidates": [],
            },
            {
                "cluster_id": 11,
                "accepted": True,
                "rejection_reasons": [],
                "member_bands": ["H", "L"],
                "n_member_bands": 2,
                "center_frequency": 2.0,
                "center_period": 0.5,
                "log_center_frequency": math.log(2.0),
                "frequency_scatter": 0.02,
                "log_frequency_scatter": 0.01,
                "members": [
                    {"band_name": "H", "frequency": 2.0, "peak_power": 1.5, "significant": True},
                    {"band_name": "L", "frequency": 2.02, "peak_power": 1.0, "significant": True},
                ],
                "duplicate_band_candidates": [],
            },
        ]
        result = self.lc._consensus_build_multicomponent_frequency_consensus(clusters)
        self.assertEqual(len(result["consensus_frequencies"]), 2)
        self.assertEqual(len(result["consensus_frequency_widths"]), 2)
        self.assertEqual(len(result["consensus_scales"]), 2)
        self.assertTrue(all(value > 0.0 for value in result["consensus_scales"]))

    def test_rejected_clusters_are_preserved_but_excluded_from_consensus_arrays(self):
        rejected = {
            "cluster_id": 99,
            "accepted": False,
            "rejection_reasons": ["insufficient_bands"],
            "member_bands": ["Z"],
            "n_member_bands": 1,
            "center_frequency": 9.0,
            "center_period": 1.0 / 9.0,
            "log_center_frequency": math.log(9.0),
            "frequency_scatter": 0.0,
            "log_frequency_scatter": 0.0,
            "members": [{"band_name": "Z", "frequency": 9.0}],
            "duplicate_band_candidates": [],
        }
        accepted = {
            "cluster_id": 0,
            "accepted": True,
            "rejection_reasons": [],
            "member_bands": ["A", "B"],
            "n_member_bands": 2,
            "center_frequency": 1.0,
            "center_period": 1.0,
            "log_center_frequency": 0.0,
            "frequency_scatter": 0.01,
            "log_frequency_scatter": 0.01,
            "members": [
                {"band_name": "A", "frequency": 1.0, "peak_power": 2.0, "significant": True},
                {"band_name": "B", "frequency": 1.02, "peak_power": 2.0, "significant": True},
            ],
            "duplicate_band_candidates": [],
        }
        result = self.lc._consensus_build_multicomponent_frequency_consensus(
            [rejected, accepted]
        )
        self.assertEqual(len(result["consensus_frequencies"]), 1)
        self.assertEqual(len(result["rejected_clusters"]), 1)
        self.assertEqual(result["rejected_clusters"][0]["cluster_id"], 99)

    def test_cluster_id_ordering_is_preserved_in_consensus_outputs(self):
        clusters = [
            {
                "cluster_id": 1,
                "accepted": True,
                "rejection_reasons": [],
                "member_bands": ["B", "C"],
                "n_member_bands": 2,
                "center_frequency": 1.0,
                "center_period": 1.0,
                "log_center_frequency": 0.0,
                "frequency_scatter": 0.0,
                "log_frequency_scatter": 0.0,
                "members": [
                    {"band_name": "B", "frequency": 1.0, "peak_power": 1.0, "significant": True},
                    {"band_name": "C", "frequency": 1.0, "peak_power": 1.0, "significant": True},
                ],
                "duplicate_band_candidates": [],
            },
            {
                "cluster_id": 0,
                "accepted": True,
                "rejection_reasons": [],
                "member_bands": ["A", "D"],
                "n_member_bands": 2,
                "center_frequency": 4.0,
                "center_period": 0.25,
                "log_center_frequency": math.log(4.0),
                "frequency_scatter": 0.0,
                "log_frequency_scatter": 0.0,
                "members": [
                    {"band_name": "A", "frequency": 4.0, "peak_power": 1.0, "significant": True},
                    {"band_name": "D", "frequency": 4.0, "peak_power": 1.0, "significant": True},
                ],
                "duplicate_band_candidates": [],
            },
        ]
        result = self.lc._consensus_build_multicomponent_frequency_consensus(clusters)
        self.assertEqual(
            [cluster["cluster_id"] for cluster in result["accepted_clusters"]],
            [0, 1],
        )
        self.assertEqual(
            [summary["source_cluster_id"] for summary in result["component_summaries"]],
            [0, 1],
        )
        self.assertEqual(
            [summary["component_index"] for summary in result["component_summaries"]],
            [0, 1],
        )
        self.assertEqual(result["consensus_frequencies"], [4.0, 1.0])
        self.assertEqual(result["consensus_frequency_widths"], [0.2, 0.05])
        self.assertEqual(result["consensus_scales"], [2.0, 2.0])

    def test_weighted_log_frequency_center_matches_expected_value(self):
        clusters = [
            {
                "cluster_id": 0,
                "accepted": True,
                "rejection_reasons": [],
                "member_bands": ["A", "B"],
                "n_member_bands": 2,
                "center_frequency": 2.0,
                "center_period": 0.5,
                "log_center_frequency": math.log(2.0),
                "frequency_scatter": 0.0,
                "log_frequency_scatter": 0.0,
                "members": [
                    {"band_name": "A", "frequency": 2.0, "peak_power": 3.0, "significant": True},
                    {"band_name": "B", "frequency": 8.0, "peak_power": 1.0, "significant": True},
                ],
                "duplicate_band_candidates": [],
            }
        ]
        result = self.lc._consensus_build_multicomponent_frequency_consensus(clusters)
        expected = math.exp((3.0 * math.log(2.0) + 1.0 * math.log(8.0)) / 4.0)
        self.assertAlmostEqual(result["consensus_frequencies"][0], expected, places=12)
        self.assertEqual(
            result["component_summaries"][0]["consensus_method"],
            "weighted_log_frequency_center",
        )

    def test_non_significant_members_are_downweighted(self):
        clusters = [
            {
                "cluster_id": 0,
                "accepted": True,
                "rejection_reasons": [],
                "member_bands": ["A", "B"],
                "n_member_bands": 2,
                "center_frequency": 1.0,
                "center_period": 1.0,
                "log_center_frequency": 0.0,
                "frequency_scatter": 0.0,
                "log_frequency_scatter": 0.0,
                "members": [
                    {"band_name": "A", "frequency": 1.0, "peak_power": 1.0, "significant": True},
                    {"band_name": "B", "frequency": 4.0, "peak_power": 1.0, "significant": False},
                ],
                "duplicate_band_candidates": [],
            }
        ]
        result = self.lc._consensus_build_multicomponent_frequency_consensus(clusters)
        expected = math.exp((1.0 * math.log(1.0) + 0.5 * math.log(4.0)) / 1.5)
        self.assertAlmostEqual(result["consensus_frequencies"][0], expected, places=12)
        self.assertAlmostEqual(result["consensus_scales"][0], 1.5, places=12)

    def test_missing_or_invalid_peak_power_falls_back_to_unit_weight(self):
        clusters = [
            {
                "cluster_id": 0,
                "accepted": True,
                "rejection_reasons": [],
                "member_bands": ["A", "B", "C"],
                "n_member_bands": 3,
                "center_frequency": 2.0,
                "center_period": 0.5,
                "log_center_frequency": math.log(2.0),
                "frequency_scatter": 0.0,
                "log_frequency_scatter": 0.0,
                "members": [
                    {"band_name": "A", "frequency": 1.0, "peak_power": None, "significant": True},
                    {"band_name": "B", "frequency": 2.0, "peak_power": -5.0, "significant": True},
                    {"band_name": "C", "frequency": 4.0, "peak_power": float("nan"), "significant": False},
                ],
                "duplicate_band_candidates": [],
            }
        ]
        result = self.lc._consensus_build_multicomponent_frequency_consensus(clusters)
        expected = math.exp((math.log(1.0) + math.log(2.0) + 0.5 * math.log(4.0)) / 2.5)
        self.assertAlmostEqual(result["consensus_frequencies"][0], expected, places=12)
        self.assertAlmostEqual(result["consensus_scales"][0], 2.5, places=12)

    def test_frequency_width_floor_is_enforced_for_single_member_clusters(self):
        clusters = [
            {
                "cluster_id": 0,
                "accepted": True,
                "rejection_reasons": [],
                "member_bands": ["A"],
                "n_member_bands": 1,
                "center_frequency": 3.0,
                "center_period": 1.0 / 3.0,
                "log_center_frequency": math.log(3.0),
                "frequency_scatter": 0.0,
                "log_frequency_scatter": 0.0,
                "members": [
                    {"band_name": "A", "frequency": 3.0, "peak_power": 2.0, "significant": True}
                ],
                "duplicate_band_candidates": [],
            }
        ]
        result = self.lc._consensus_build_multicomponent_frequency_consensus(
            clusters, min_width_fraction=0.05
        )
        expected_width = 0.05 * result["consensus_frequencies"][0]
        self.assertAlmostEqual(result["consensus_frequency_widths"][0], expected_width, places=12)

    def test_empty_input_returns_empty_arrays_and_diagnostics(self):
        result = self.lc._consensus_build_multicomponent_frequency_consensus([])
        self.assertEqual(result["consensus_frequencies"], [])
        self.assertEqual(result["consensus_frequency_widths"], [])
        self.assertEqual(result["consensus_scales"], [])
        self.assertEqual(result["accepted_clusters"], [])
        self.assertEqual(result["rejected_clusters"], [])
        self.assertEqual(result["component_summaries"], [])

    def test_component_summaries_preserve_cluster_and_member_metadata(self):
        members = [
            {
                "band_name": "A",
                "wavelength": 1.23,
                "frequency": 2.0,
                "period": 0.5,
                "ls_rank": 4,
                "peak_power": 3.0,
                "peak_prominence": 0.7,
                "significant": True,
            },
            {
                "band_name": "B",
                "wavelength": 2.34,
                "frequency": 2.1,
                "period": 1.0 / 2.1,
                "ls_rank": 1,
                "peak_power": 2.0,
                "peak_prominence": 0.2,
                "significant": False,
            },
        ]
        clusters = [
            {
                "cluster_id": 7,
                "accepted": True,
                "rejection_reasons": [],
                "member_bands": ["A", "B"],
                "n_member_bands": 2,
                "center_frequency": 2.05,
                "center_period": 1.0 / 2.05,
                "log_center_frequency": math.log(2.05),
                "frequency_scatter": 0.05,
                "log_frequency_scatter": 0.02,
                "members": members,
                "duplicate_band_candidates": [],
            }
        ]
        result = self.lc._consensus_build_multicomponent_frequency_consensus(clusters)
        summary = result["component_summaries"][0]
        self.assertEqual(summary["source_cluster_id"], 7)
        self.assertEqual(summary["member_bands"], ["A", "B"])
        self.assertEqual(summary["n_member_bands"], 2)
        self.assertEqual(summary["frequency_scatter"], 0.05)
        self.assertEqual(summary["log_frequency_scatter"], 0.02)
        self.assertEqual(summary["members"][0]["wavelength"], 1.23)
        self.assertEqual(summary["members"][1]["ls_rank"], 1)

    def test_helper_does_not_modify_input_clusters_in_place(self):
        clusters = [
            {
                "cluster_id": 0,
                "accepted": True,
                "rejection_reasons": [],
                "member_bands": ["A", "B"],
                "n_member_bands": 2,
                "center_frequency": 1.0,
                "center_period": 1.0,
                "log_center_frequency": 0.0,
                "frequency_scatter": 0.0,
                "log_frequency_scatter": 0.0,
                "members": [
                    {"band_name": "A", "frequency": 1.0, "peak_power": 2.0, "significant": True},
                    {"band_name": "B", "frequency": 2.0, "peak_power": 1.0, "significant": True},
                ],
                "duplicate_band_candidates": [],
            }
        ]
        original = copy.deepcopy(clusters)
        _ = self.lc._consensus_build_multicomponent_frequency_consensus(clusters)
        self.assertEqual(clusters, original)


if __name__ == "__main__":
    unittest.main()
