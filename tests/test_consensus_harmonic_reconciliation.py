import unittest
from unittest import mock

import numpy as np

from pgmuvi.lightcurve import Lightcurve


class _FakeBandLightcurve:
    def acf(self, method="data", normalize=True):
        return {"method": method, "normalize": normalize}


class TestConsensusHarmonicReconciliation(unittest.TestCase):
    def test_lower_acf_frequency_is_promoted_as_fundamental(self):
        result = Lightcurve._consensus_reconcile_ls_acf_harmonic(
            ls_frequency=1.0 / 250.0,
            acf_frequency=1.0 / 500.0,
            comparison_status="harmonic",
            harmonic_order=2,
        )

        self.assertTrue(result["applied"])
        self.assertAlmostEqual(result["frequency"], 1.0 / 500.0)
        self.assertEqual(
            result["selected_from"],
            "acf_fundamental_harmonic_reconciliation",
        )

    def test_higher_acf_frequency_does_not_replace_slower_ls_candidate(self):
        result = Lightcurve._consensus_reconcile_ls_acf_harmonic(
            ls_frequency=1.0 / 500.0,
            acf_frequency=1.0 / 250.0,
            comparison_status="harmonic",
            harmonic_order=2,
        )

        self.assertFalse(result["applied"])
        self.assertAlmostEqual(result["frequency"], 1.0 / 500.0)
        self.assertEqual(result["selected_from"], "ls_primary_peak")

    def test_direct_agreement_retains_ls_candidate(self):
        result = Lightcurve._consensus_reconcile_ls_acf_harmonic(
            ls_frequency=1.0 / 500.0,
            acf_frequency=1.0 / 510.0,
            comparison_status="agreement",
            harmonic_order=1,
        )

        self.assertFalse(result["applied"])
        self.assertAlmostEqual(result["frequency"], 1.0 / 500.0)

    def test_collection_preserves_ls_and_promotes_acf_fundamental(self):
        lightcurve = object.__new__(Lightcurve)
        controls = {
            "min_points_per_band": 5,
            "max_gap_fraction": 0.5,
            "min_duty_cycle": 0.05,
            "outlier_sigma": 3.5,
            "consensus_width_factor": 3.0,
        }
        ls_frequency = 1.0 / 250.0
        acf_frequency = 1.0 / 500.0
        ls_payload = {
            "ls_frequencies": np.asarray([ls_frequency], dtype=float),
            "ls_significant": np.asarray([True], dtype=bool),
            "candidates": [
                {
                    "frequency": ls_frequency,
                    "period": 250.0,
                    "ls_rank": 0,
                    "significant": True,
                }
            ],
            "min_detectable_frequency": 1.0 / 600.0,
            "nyquist_frequency": 1.0,
        }

        with mock.patch.object(
            Lightcurve,
            "_consensus_prepare_band_consensus_inputs",
            return_value={
                "per_band_lc": {"V": _FakeBandLightcurve()},
                "metrics_by_band": {"V": {"baseline": 1200.0}},
                "controls": controls,
            },
        ), mock.patch.object(
            Lightcurve,
            "_consensus_reject_bad_bands",
            return_value=[],
        ), mock.patch.object(
            Lightcurve,
            "_consensus_extract_band_ls_candidates",
            return_value=ls_payload,
        ), mock.patch.object(
            Lightcurve,
            "_consensus_extract_acf_candidate",
            return_value={"frequency": acf_frequency, "period": 500.0},
        ), mock.patch.object(
            Lightcurve,
            "_consensus_debug_checkpoint_from_candidate_state",
            return_value=None,
        ):
            result = lightcurve._consensus_collect_band_candidates(
                use_acf=True,
                gp_validation_requested=False,
            )

        record = result["band_records"]["V"]
        self.assertEqual(result["accepted_bands"], ["V"])
        self.assertAlmostEqual(record["ls_frequency"], ls_frequency)
        self.assertAlmostEqual(record["ls_period"], 250.0)
        self.assertAlmostEqual(record["acf_frequency"], acf_frequency)
        self.assertAlmostEqual(record["dominant_frequency"], acf_frequency)
        self.assertAlmostEqual(record["dominant_period"], 500.0)
        self.assertTrue(record["harmonic_reconciliation_applied"])
        self.assertEqual(
            record["selected_from"],
            "acf_fundamental_harmonic_reconciliation",
        )


if __name__ == "__main__":
    unittest.main()
