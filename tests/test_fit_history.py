from __future__ import annotations

import datetime
import math
import unittest

import numpy as np

from pgmuvi.lightcurve import ConsensusFitError, Lightcurve


def _make_multiband_lightcurve(
    period_by_band,
    *,
    n_pts_by_band=None,
    noise_std=0.01,
    seed=0,
):
    rng = np.random.default_rng(seed)
    period_by_band = dict(period_by_band)
    bands = list(period_by_band)
    n_pts_by_band = dict(n_pts_by_band or {})

    x_blocks = []
    y_blocks = []
    yerr_blocks = []
    band_blocks = []

    for band_idx, band in enumerate(bands):
        n = int(n_pts_by_band.get(band, 60))
        t = np.linspace(0.0, 220.0, n)
        period = float(period_by_band[band])
        signal = np.sin(2.0 * math.pi * t / period)
        noise = rng.normal(0.0, float(noise_std), t.shape)
        y = signal + noise
        yerr = np.full_like(t, float(noise_std))
        x = np.column_stack([t, np.full_like(t, float(band_idx + 1))])
        x_blocks.append(x)
        y_blocks.append(y)
        yerr_blocks.append(yerr)
        band_blocks.append(np.full(t.shape, str(band), dtype=np.str_))

    return Lightcurve(
        np.concatenate(x_blocks, axis=0),
        np.concatenate(y_blocks),
        yerr=np.concatenate(yerr_blocks),
        band=np.concatenate(band_blocks),
    )


def _run_consensus_fit(lc, **extra_kwargs):
    kwargs = {
        "fit_strategy": "consensus",
        "model": "2D",
        "training_iter": 0,
        "use_gp_validation": False,
        "use_mls_init": False,
        "use_best_band_init": False,
        "constrain_consensus": False,
    }
    kwargs.update(extra_kwargs)
    return lc.fit(**kwargs)


class _BrokenValue:
    def __float__(self):
        raise TypeError("cannot coerce")

    def __str__(self):
        raise RuntimeError("cannot stringify")


class TestFitHistory(unittest.TestCase):
    def test_successful_fit_entry_creation(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=100)
        self.assertEqual(lc.get_fit_history(), [])
        _run_consensus_fit(lc)
        history = lc.get_fit_history()
        self.assertEqual(len(history), 1)
        entry = history[-1]
        self.assertTrue(entry["success"])
        self.assertFalse(entry["failed"])
        self.assertEqual(entry["fit_strategy"], "consensus")
        self.assertEqual(entry["model_class"], "TwoDSpectralMixtureGPModel")
        parsed = datetime.datetime.fromisoformat(entry["timestamp_utc"])
        self.assertEqual(parsed.tzinfo, datetime.UTC)

    def test_failed_fit_entry_creation(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=101)
        with self.assertRaises(ConsensusFitError):
            _run_consensus_fit(lc, min_points_per_band=10_000)
        history = lc.get_fit_history()
        self.assertEqual(len(history), 1)
        entry = history[-1]
        self.assertFalse(entry["success"])
        self.assertTrue(entry["failed"])
        self.assertEqual(entry["exception_type"], "ConsensusFitError")
        self.assertIsNotNone(entry["exception_message"])

    def test_history_survives_multiple_fits(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=102)
        _run_consensus_fit(lc)
        with self.assertRaises(ConsensusFitError):
            _run_consensus_fit(lc, min_points_per_band=10_000)
        _run_consensus_fit(lc)
        history = lc.get_fit_history()
        self.assertEqual(len(history), 3)
        self.assertEqual([entry["success"] for entry in history], [True, False, True])

    def test_clear_fit_history(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=103)
        _run_consensus_fit(lc)
        self.assertEqual(len(lc.get_fit_history()), 1)
        lc.clear_fit_history()
        self.assertEqual(lc.get_fit_history(), [])

    def test_get_fit_history_returns_deep_copy(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=104)
        _run_consensus_fit(lc)
        history_copy = lc.get_fit_history()
        history_copy[0]["notes"]["source"] = "mutated"
        history_copy.append({"x": 1})
        history_current = lc.get_fit_history()
        self.assertEqual(len(history_current), 1)
        self.assertEqual(history_current[0]["notes"]["source"], "fit_success")

    def test_get_fit_history_summary_values(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=105)
        _run_consensus_fit(lc)
        with self.assertRaises(ConsensusFitError):
            _run_consensus_fit(lc, min_points_per_band=10_000)
        summary = lc.get_fit_history_summary()
        self.assertEqual(summary["total_attempts"], 2)
        self.assertEqual(summary["successful_fits"], 1)
        self.assertEqual(summary["failed_fits"], 1)
        self.assertAlmostEqual(summary["success_fraction"], 0.5)
        self.assertIsNotNone(summary["last_success_timestamp"])
        self.assertIsNotNone(summary["last_failure_timestamp"])

    def test_append_history_sanitizes_nan_inf(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0}, seed=106)
        lc._append_fit_history(
            success=True,
            failed=False,
            elapsed_seconds=np.nan,
            notes={"nan_val": np.nan, "inf_val": np.inf},
        )
        entry = lc.get_fit_history()[-1]
        self.assertIsNone(entry["elapsed_seconds"])
        self.assertIsNone(entry["notes"]["nan_val"])
        self.assertIsNone(entry["notes"]["inf_val"])

    def test_append_history_never_raises_on_malformed_inputs(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0}, seed=107)
        before_len = len(lc.get_fit_history())
        lc._append_fit_history(notes={"broken": _BrokenValue()})
        after_len = len(lc.get_fit_history())
        self.assertGreaterEqual(after_len, before_len)


if __name__ == "__main__":
    unittest.main()
