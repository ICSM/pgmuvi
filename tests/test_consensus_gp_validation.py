"""Tests for consensus GP-validation helpers."""

import unittest
from types import SimpleNamespace

import numpy as np

from pgmuvi.lightcurve import Lightcurve


class _FakeBandLightcurve:
    """Minimal fake Lightcurve used for per-band GP-validation tests."""

    def __init__(self, dominant_frequency, fit_error=None):
        self._dominant_frequency = dominant_frequency
        self._fit_error = fit_error
        self.fit_calls = []
        self.summary_calls = []

    def fit(self, **kwargs):
        self.fit_calls.append(dict(kwargs))
        if self._fit_error is not None:
            raise self._fit_error

    def get_period_summary(self, **kwargs):
        self.summary_calls.append(dict(kwargs))
        return SimpleNamespace(dominant_frequency=self._dominant_frequency)


class TestConsensusGPValidationHelpers(unittest.TestCase):
    """Unit tests for GP-validation kwarg sanitization and candidate vetting."""

    def setUp(self):
        x = np.linspace(0.0, 10.0, 32)
        y = np.sin(x)
        self.lc = Lightcurve(x, y)

    def test_prepare_gp_validation_fit_kwargs_strips_blocked_keys(self):
        """Blocked consensus keys are removed and fit_strategy is always None."""
        kwargs = Lightcurve._consensus_prepare_gp_validation_fit_kwargs(
            {
                "model": "2D",
                "fit_strategy": "consensus",
                "consensus_frequencies": [0.1],
                "use_gp_validation": True,
                "gp_ls_tolerance_base_factor": 0.2,
                "period_summary_kwargs": {"backend": "gp"},
                "training_iter": 7,
                "lr": 0.05,
            }
        )

        self.assertEqual(kwargs["model"], "2D")
        self.assertEqual(kwargs["training_iter"], 7)
        self.assertEqual(kwargs["lr"], 0.05)
        self.assertIsNone(kwargs["fit_strategy"])
        self.assertNotIn("consensus_frequencies", kwargs)
        self.assertNotIn("use_gp_validation", kwargs)
        self.assertNotIn("gp_ls_tolerance_base_factor", kwargs)
        self.assertNotIn("period_summary_kwargs", kwargs)

    def test_validate_candidates_with_1d_gp_updates_diagnostics(self):
        """Agreement/disagreement/failure are reflected in diagnostics fields."""
        candidate_diag = {
            "controls": {},
            "band_records": {
                "A": {"band": "A", "dominant_frequency": 1.0},
                "B": {"band": "B", "dominant_frequency": 1.0},
                "C": {"band": "C", "dominant_frequency": 1.0},
            },
            "accepted_bands": ["A", "B", "C"],
            "rejected_bands": [],
            "rejection_reasons": {},
        }
        band_lcs = {
            "A": _FakeBandLightcurve(dominant_frequency=1.05),
            "B": _FakeBandLightcurve(dominant_frequency=5.0),
            "C": _FakeBandLightcurve(
                dominant_frequency=1.0, fit_error=RuntimeError("fit failed")
            ),
        }

        def _fake_select_bands(labels):
            return band_lcs[labels[0]]

        self.lc.select_bands = _fake_select_bands

        validated = self.lc._consensus_validate_candidates_with_1d_gp(
            candidate_diag=candidate_diag,
            gp_validation_kwargs={
                "training_iter": 3,
                "fit_strategy": "consensus",
                "period_summary_kwargs": {"backend": "gp"},
            },
            gp_frequency_tolerance_factor=1.0,
            verbose=False,
        )

        self.assertEqual(validated["accepted_bands"], ["A"])
        self.assertCountEqual(validated["rejected_bands"], ["B", "C"])
        self.assertEqual(
            validated["rejection_reasons"]["B"], ["gp_ls_frequency_disagreement"]
        )
        self.assertEqual(validated["rejection_reasons"]["C"], ["gp_validation_failed"])

        records = validated["band_records"]
        self.assertEqual(records["A"]["gp_validation_status"], "agreement")
        self.assertEqual(records["B"]["gp_validation_status"], "disagreement")
        self.assertEqual(records["C"]["gp_validation_status"], "failed")

        self.assertIsInstance(records["A"]["gp_fractional_frequency_difference"], float)
        self.assertIsInstance(records["B"]["gp_fractional_frequency_difference"], float)
        self.assertIsNone(records["C"]["gp_fractional_frequency_difference"])

        self.assertTrue(records["A"]["gp_validation_used"])
        self.assertTrue(records["B"]["gp_validation_used"])
        self.assertTrue(records["C"]["gp_validation_used"])
        self.assertIn("RuntimeError", records["C"]["gp_validation_error"])


if __name__ == "__main__":
    unittest.main()
