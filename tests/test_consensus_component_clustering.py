"""Unit tests for _consensus_cluster_component_candidates."""

from __future__ import annotations

import math
import unittest

import numpy as np

from pgmuvi.lightcurve import Lightcurve


def _make_minimal_multiband_lightcurve():
    """Create a minimal multiband lightcurve instance for helper tests.

    Returns
    -------
    Lightcurve
        Two-band synthetic lightcurve with two points per band.
    """
    t = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    wl = np.array([1.0, 1.0, 2.0, 2.0], dtype=float)
    x = np.column_stack([t, wl])
    y = np.array([0.0, 0.1, -0.1, 0.0], dtype=float)
    band = np.array(["A", "A", "B", "B"])
    return Lightcurve(x, y, band=band)


class TestConsensusComponentClustering(unittest.TestCase):
    def setUp(self):
        self.lc = _make_minimal_multiband_lightcurve()

    def test_nearby_cross_band_candidates_cluster_together(self):
        candidates = [
            {
                "band_name": "J",
                "wavelength": 1.2,
                "component_candidates": [
                    {
                        "frequency": 1.00,
                        "period": 1.0,
                        "ls_rank": 5,
                        "peak_power": 2.0,
                        "peak_prominence": 1.0,
                        "significant": True,
                    }
                ],
            },
            {
                "band_name": "K",
                "wavelength": 2.2,
                "component_candidates": [
                    {
                        "frequency": 1.05,
                        "period": 1.0 / 1.05,
                        "ls_rank": 0,
                        "peak_power": 1.0,
                        "peak_prominence": 0.8,
                        "significant": False,
                    }
                ],
            },
            {
                "band_name": "H",
                "wavelength": 1.6,
                "component_candidates": [
                    {
                        "frequency": 3.0,
                        "period": 1.0 / 3.0,
                        "ls_rank": 0,
                        "peak_power": 10.0,
                        "peak_prominence": 3.0,
                        "significant": True,
                    }
                ],
            },
        ]
        clusters = self.lc._consensus_cluster_component_candidates(candidates)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0]["member_bands"], ["J", "K"])
        self.assertEqual(clusters[1]["member_bands"], ["H"])

    def test_cluster_ids_are_assigned_after_frequency_sort(self):
        candidates = [
            {
                "band_name": "A",
                "wavelength": 1.0,
                "component_candidates": [
                    {
                        "frequency": 5.0,
                        "period": 0.2,
                        "ls_rank": 0,
                        "peak_power": 100.0,
                        "peak_prominence": 10.0,
                        "significant": True,
                    }
                ],
            },
            {
                "band_name": "B",
                "wavelength": 2.0,
                "component_candidates": [
                    {
                        "frequency": 5.1,
                        "period": 1.0 / 5.1,
                        "ls_rank": 1,
                        "peak_power": 90.0,
                        "peak_prominence": 9.0,
                        "significant": True,
                    }
                ],
            },
            {
                "band_name": "C",
                "wavelength": 3.0,
                "component_candidates": [
                    {
                        "frequency": 1.0,
                        "period": 1.0,
                        "ls_rank": 5,
                        "peak_power": 1.0,
                        "peak_prominence": 0.1,
                        "significant": True,
                    }
                ],
            },
            {
                "band_name": "D",
                "wavelength": 4.0,
                "component_candidates": [
                    {
                        "frequency": 1.02,
                        "period": 1.0 / 1.02,
                        "ls_rank": 6,
                        "peak_power": 0.9,
                        "peak_prominence": 0.1,
                        "significant": True,
                    }
                ],
            },
        ]
        clusters = self.lc._consensus_cluster_component_candidates(candidates)
        self.assertEqual([c["cluster_id"] for c in clusters], [0, 1])
        self.assertLess(clusters[0]["center_frequency"], clusters[1]["center_frequency"])
        self.assertEqual(clusters[0]["member_bands"], ["C", "D"])
        self.assertEqual(clusters[1]["member_bands"], ["A", "B"])

    def test_ls_rank_does_not_define_component_identity(self):
        candidates = [
            {
                "band_name": "J",
                "wavelength": 1.2,
                "component_candidates": [
                    {
                        "frequency": 10.0,
                        "period": 0.1,
                        "ls_rank": 0,
                        "peak_power": 9.0,
                        "peak_prominence": 1.1,
                        "significant": True,
                    },
                    {
                        "frequency": 1.0,
                        "period": 1.0,
                        "ls_rank": 1,
                        "peak_power": 4.0,
                        "peak_prominence": 1.0,
                        "significant": True,
                    },
                ],
            },
            {
                "band_name": "K",
                "wavelength": 2.2,
                "component_candidates": [
                    {
                        "frequency": 1.03,
                        "period": 1.0 / 1.03,
                        "ls_rank": 0,
                        "peak_power": 5.0,
                        "peak_prominence": 1.2,
                        "significant": True,
                    }
                ],
            },
        ]
        clusters = self.lc._consensus_cluster_component_candidates(candidates)
        low_freq_cluster = min(clusters, key=lambda c: c["center_frequency"])
        ranks = sorted(member["ls_rank"] for member in low_freq_cluster["members"])
        self.assertEqual(ranks, [0, 1])
        freqs = sorted(member["frequency"] for member in low_freq_cluster["members"])
        self.assertAlmostEqual(freqs[0], 1.0, places=12)
        self.assertAlmostEqual(freqs[1], 1.03, places=12)

    def test_same_band_duplicates_are_deduplicated_with_diagnostics(self):
        candidates = [
            {
                "band_name": "J",
                "wavelength": 1.2,
                "component_candidates": [
                    {
                        "frequency": 1.00,
                        "period": 1.0,
                        "ls_rank": 0,
                        "peak_power": 50.0,
                        "peak_prominence": 4.0,
                        "significant": False,
                    },
                    {
                        "frequency": 1.01,
                        "period": 1.0 / 1.01,
                        "ls_rank": 1,
                        "peak_power": 5.0,
                        "peak_prominence": 2.0,
                        "significant": True,
                    },
                ],
            },
            {
                "band_name": "K",
                "wavelength": 2.2,
                "component_candidates": [
                    {
                        "frequency": 1.02,
                        "period": 1.0 / 1.02,
                        "ls_rank": 0,
                        "peak_power": 7.0,
                        "peak_prominence": 2.3,
                        "significant": True,
                    }
                ],
            },
        ]
        clusters = self.lc._consensus_cluster_component_candidates(candidates)
        cluster = clusters[0]
        self.assertEqual(cluster["member_bands"], ["J", "K"])
        j_member = next(m for m in cluster["members"] if m["band_name"] == "J")
        self.assertTrue(j_member["significant"])
        self.assertAlmostEqual(j_member["frequency"], 1.01, places=12)
        self.assertEqual(len(cluster["duplicate_band_candidates"]), 1)
        duplicate = cluster["duplicate_band_candidates"][0]
        self.assertEqual(duplicate["band_name"], "J")
        self.assertAlmostEqual(duplicate["frequency"], 1.00, places=12)

    def test_min_band_support_filter_marks_rejected_clusters(self):
        candidates = [
            {
                "band_name": "A",
                "wavelength": 1.0,
                "component_candidates": [
                    {
                        "frequency": 1.0,
                        "period": 1.0,
                        "ls_rank": 0,
                        "peak_power": 4.0,
                        "peak_prominence": 1.0,
                        "significant": True,
                    }
                ],
            },
            {
                "band_name": "B",
                "wavelength": 2.0,
                "component_candidates": [
                    {
                        "frequency": 3.0,
                        "period": 1.0 / 3.0,
                        "ls_rank": 0,
                        "peak_power": 4.0,
                        "peak_prominence": 1.0,
                        "significant": True,
                    }
                ],
            },
        ]
        clusters = self.lc._consensus_cluster_component_candidates(candidates)
        self.assertEqual(len(clusters), 2)
        self.assertTrue(all(cluster["accepted"] is False for cluster in clusters))
        self.assertTrue(
            all(len(cluster["rejection_reasons"]) > 0 for cluster in clusters),
            "Rejected clusters must include rejection reasons",
        )

    def test_invalid_nonpositive_or_nonfinite_frequencies_are_ignored(self):
        candidates = [
            {
                "band_name": "J",
                "wavelength": 1.2,
                "component_candidates": [
                    {
                        "frequency": -1.0,
                        "period": 1.0,
                        "ls_rank": 0,
                        "peak_power": 1.0,
                        "peak_prominence": 1.0,
                        "significant": True,
                    },
                    {
                        "frequency": 0.0,
                        "period": math.inf,
                        "ls_rank": 1,
                        "peak_power": 1.0,
                        "peak_prominence": 1.0,
                        "significant": False,
                    },
                    {
                        "frequency": float("nan"),
                        "period": float("nan"),
                        "ls_rank": 2,
                        "peak_power": 1.0,
                        "peak_prominence": 1.0,
                        "significant": False,
                    },
                ],
            },
            {
                "band_name": "K",
                "wavelength": 2.2,
                "component_candidates": [
                    {
                        "frequency": 1.0,
                        "period": 1.0,
                        "ls_rank": 0,
                        "peak_power": 2.0,
                        "peak_prominence": 1.0,
                        "significant": True,
                    },
                    {
                        "frequency": float("inf"),
                        "period": 0.0,
                        "ls_rank": 1,
                        "peak_power": 2.0,
                        "peak_prominence": 1.0,
                        "significant": False,
                    },
                ],
            },
        ]
        clusters = self.lc._consensus_cluster_component_candidates(candidates)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["member_bands"], ["K"])
        self.assertEqual(len(clusters[0]["members"]), 1)
        self.assertAlmostEqual(clusters[0]["members"][0]["frequency"], 1.0, places=12)

    def test_member_metadata_is_preserved(self):
        candidates = [
            {
                "band_name": "J",
                "wavelength": 1.23,
                "component_candidates": [
                    {
                        "frequency": 2.0,
                        "period": 0.5,
                        "ls_rank": 3,
                        "peak_power": 7.5,
                        "peak_prominence": 0.33,
                        "significant": False,
                    }
                ],
            },
            {
                "band_name": "K",
                "wavelength": 2.34,
                "component_candidates": [
                    {
                        "frequency": 2.05,
                        "period": 1.0 / 2.05,
                        "ls_rank": 8,
                        "peak_power": 6.0,
                        "peak_prominence": 0.25,
                        "significant": True,
                    }
                ],
            },
        ]
        clusters = self.lc._consensus_cluster_component_candidates(candidates)
        member = next(m for m in clusters[0]["members"] if m["band_name"] == "J")
        self.assertEqual(member["wavelength"], 1.23)
        self.assertEqual(member["frequency"], 2.0)
        self.assertEqual(member["period"], 0.5)
        self.assertEqual(member["ls_rank"], 3)
        self.assertEqual(member["peak_power"], 7.5)
        self.assertEqual(member["peak_prominence"], 0.33)
        self.assertEqual(member["significant"], False)


if __name__ == "__main__":
    unittest.main()
