"""Regression tests for two-band consensus frequency agreement."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from pgmuvi.lightcurve import ConsensusFitError, Lightcurve


def _lightcurve() -> Lightcurve:
    t = torch.linspace(0.0, 3.0, 4, dtype=torch.float64)
    y = torch.ones(4, dtype=torch.float64)
    return Lightcurve(t, y, max_samples=None)


def _band_record(lc: Lightcurve, band: str, frequency: float) -> dict:
    record = lc._consensus_initialize_band_record(band)
    record.update(
        {
            "status": "accepted",
            "dominant_frequency": float(frequency),
            "dominant_period": float(1.0 / frequency),
            "ls_significant": True,
        }
    )
    return record


class TestTwoBandFrequencyAgreementHelper(unittest.TestCase):
    def setUp(self):
        self.lc = _lightcurve()

    def _build(self, f1: float, f2: float, **kwargs):
        records = {
            "g": _band_record(self.lc, "g", f1),
            "r": _band_record(self.lc, "r", f2),
        }
        return self.lc._consensus_build_frequency_consensus(
            records,
            ["g", "r"],
            **kwargs,
        )

    def test_close_pair_produces_supported_consensus(self):
        result = self._build(1.0, 1.05)

        self.assertFalse(result.get("insufficient_inliers", False))
        self.assertAlmostEqual(result["final_consensus_frequency"], 1.025)
        self.assertTrue(result["two_band_pairwise_check_applied"])
        self.assertTrue(result["two_band_frequency_agreement"])
        self.assertAlmostEqual(
            result["two_band_fractional_frequency_difference"], 0.05
        )

    def test_incompatible_pair_does_not_create_midpoint_consensus(self):
        result = self._build(1.0, 1.30)

        self.assertTrue(result["insufficient_inliers"])
        self.assertTrue(np.isnan(result["final_consensus_frequency"]))
        self.assertEqual(result["insufficient_inliers_count"], 0)
        self.assertFalse(result["two_band_frequency_agreement"])
        self.assertAlmostEqual(
            result["two_band_fractional_frequency_difference"], 0.30
        )
        self.assertEqual(result["inlier_bands"], [])

    def test_custom_pairwise_limit_is_respected(self):
        result = self._build(
            1.0,
            1.30,
            two_band_max_fractional_frequency_difference=0.40,
        )

        self.assertFalse(result.get("insufficient_inliers", False))
        self.assertTrue(result["two_band_frequency_agreement"])
        self.assertAlmostEqual(result["final_consensus_frequency"], 1.15)

    def test_invalid_pairwise_limit_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "two_band_max_fractional_frequency_difference",
        ):
            self._build(
                1.0,
                1.05,
                two_band_max_fractional_frequency_difference=0.0,
            )


class TestTwoBandFrequencyAgreementWorkflow(unittest.TestCase):
    def test_standard_consensus_surfaces_pairwise_failure(self):
        lc = _lightcurve()
        records = {
            "g": _band_record(lc, "g", 1.0),
            "r": _band_record(lc, "r", 1.30),
        }
        candidate_diag = {
            "controls": {
                "outlier_sigma": 3.5,
                "consensus_width_factor": 3.0,
            },
            "band_records": records,
            "accepted_bands": ["g", "r"],
            "rejected_bands": [],
            "rejection_reasons": {},
        }

        with mock.patch(
            "pgmuvi.lightcurve.Lightcurve.ndim",
            new_callable=mock.PropertyMock,
            return_value=2,
        ), mock.patch.object(
            lc,
            "_consensus_collect_band_candidates",
            return_value=candidate_diag,
        ):
            with self.assertRaises(ConsensusFitError) as caught:
                lc._consensus_standard_fit(
                    constrain_consensus=False,
                    two_band_max_fractional_frequency_difference=0.10,
                    verbose=False,
                )

        self.assertIn("two accepted bands", str(caught.exception))
        failure = caught.exception.failure_diagnostics
        self.assertEqual(failure["reason"], "insufficient_consensus_inliers")
        self.assertTrue(failure["two_band_pairwise_check_applied"])
        self.assertFalse(failure["two_band_frequency_agreement"])
        self.assertAlmostEqual(
            failure["two_band_fractional_frequency_difference"], 0.30
        )

        diagnostics = lc.consensus_diagnostics
        self.assertFalse(diagnostics["consensus_success"])
        self.assertTrue(diagnostics["two_band_pairwise_check_applied"])
        self.assertFalse(diagnostics["two_band_frequency_agreement"])
        self.assertIsNone(diagnostics["final_consensus_frequency"])
        self.assertEqual(
            diagnostics["controls"][
                "two_band_max_fractional_frequency_difference"
            ],
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
