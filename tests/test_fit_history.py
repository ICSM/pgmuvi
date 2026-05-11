from __future__ import annotations

import datetime
import math
import unittest

import numpy as np
import re

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

    def test_environment_metadata_is_recorded(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=108)
        _run_consensus_fit(lc)
        entry = lc.get_fit_history()[-1]
        env = entry.get("environment", {})
        self.assertIsInstance(env, dict)
        self.assertIn("python_version", env)
        self.assertIn("pgmuvi_version", env)
        self.assertIn("torch_version", env)
        self.assertIn("gpytorch_version", env)
        self.assertRegex(str(env["python_version"]), r"^\d+\.\d+\.\d+")

    def test_extended_summary_fields_present(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=109)
        _run_consensus_fit(lc, model="2D")
        with self.assertRaises(ConsensusFitError):
            _run_consensus_fit(lc, min_points_per_band=10_000)
        summary = lc.get_fit_history_summary()
        self.assertIn("counts_by_backend", summary)
        self.assertIn("counts_by_model_class", summary)
        self.assertIn("counts_by_parameterization", summary)
        self.assertIn("counts_by_constraint_mode", summary)
        self.assertIn("total_runtime_seconds", summary)
        self.assertIn("mean_runtime_seconds", summary)
        self.assertIn("earliest_timestamp", summary)
        self.assertIn("latest_timestamp", summary)
        self.assertIn("unique_bands_used", summary)
        self.assertIn("frequency_space", summary["counts_by_parameterization"])
        self.assertIn("period_space", summary["counts_by_parameterization"])

    def test_fit_history_to_text_and_filters(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=110)
        _run_consensus_fit(lc)
        with self.assertRaises(ConsensusFitError):
            _run_consensus_fit(lc, min_points_per_band=10_000)

        text_all = lc.fit_history_to_text()
        self.assertIn("Timestamp", text_all)
        self.assertIn("Runtime(s)", text_all)
        self.assertIn("Failure reason", text_all)
        self.assertIsNotNone(re.search(r"^\s*1", text_all, flags=re.MULTILINE))

        text_success = lc.fit_history_to_text(success_only=True)
        self.assertIn("True", text_success)
        self.assertNotIn("False", text_success)

        text_failed = lc.fit_history_to_text(failed_only=True)
        self.assertIn("False", text_failed)
        self.assertIn("Consensus", text_failed)

        text_latest = lc.fit_history_to_text(max_entries=1)
        rows = [ln for ln in text_latest.splitlines() if re.match(r"^\s*\d+", ln)]
        self.assertEqual(len(rows), 1)

        printed = lc.print_fit_history(max_entries=1)
        self.assertEqual(printed, text_latest)

    def test_consensus_nested_fit_records_single_entry(self):
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=111)
        _run_consensus_fit(lc)
        history = lc.get_fit_history()
        self.assertEqual(len(history), 1)

    # ------------------------------------------------------------------
    # Enhanced summary fields (counts_by_fit_strategy, flat scalars, etc.)
    # ------------------------------------------------------------------

    def test_summary_counts_by_fit_strategy(self):
        """counts_by_fit_strategy groups entries by fit_strategy."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=120)
        _run_consensus_fit(lc)
        # second call – still consensus
        _run_consensus_fit(lc)
        summary = lc.get_fit_history_summary()
        self.assertIn("counts_by_fit_strategy", summary)
        strat = summary["counts_by_fit_strategy"]
        self.assertIn("consensus", strat)
        self.assertEqual(strat["consensus"], 2)

    def test_summary_counts_by_backend(self):
        """counts_by_backend must be a dict with at least one entry after a fit."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=121)
        _run_consensus_fit(lc)
        summary = lc.get_fit_history_summary()
        self.assertIn("counts_by_backend", summary)
        self.assertIsInstance(summary["counts_by_backend"], dict)
        total = sum(summary["counts_by_backend"].values())
        self.assertEqual(total, summary["total_attempts"])

    def test_summary_counts_by_model_class(self):
        """counts_by_model_class must reflect the resolved model class."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=122)
        _run_consensus_fit(lc)
        summary = lc.get_fit_history_summary()
        self.assertIn("counts_by_model_class", summary)
        total = sum(summary["counts_by_model_class"].values())
        self.assertEqual(total, summary["total_attempts"])

    def test_summary_runtime_values(self):
        """total/mean runtime must be non-negative floats after a real fit."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=123)
        _run_consensus_fit(lc)
        _run_consensus_fit(lc)
        summary = lc.get_fit_history_summary()
        self.assertGreaterEqual(summary["total_runtime_seconds"], 0.0)
        self.assertGreaterEqual(summary["mean_runtime_seconds"], 0.0)
        self.assertAlmostEqual(
            summary["mean_runtime_seconds"],
            summary["total_runtime_seconds"] / 2,
            places=6,
        )

    def test_summary_timestamp_values(self):
        """earliest_timestamp <= latest_timestamp; both parseable ISO-8601."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=124)
        _run_consensus_fit(lc)
        _run_consensus_fit(lc)
        summary = lc.get_fit_history_summary()
        self.assertIsNotNone(summary["earliest_timestamp"])
        self.assertIsNotNone(summary["latest_timestamp"])
        t0 = datetime.datetime.fromisoformat(summary["earliest_timestamp"])
        t1 = datetime.datetime.fromisoformat(summary["latest_timestamp"])
        self.assertLessEqual(t0, t1)

    def test_summary_unique_bands_alias(self):
        """unique_bands is a list alias for unique_bands_used."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=125)
        _run_consensus_fit(lc)
        summary = lc.get_fit_history_summary()
        self.assertIn("unique_bands", summary)
        self.assertIn("unique_bands_used", summary)
        self.assertEqual(summary["unique_bands"], summary["unique_bands_used"])
        # Should contain all three bands
        for band in ("g", "r", "i"):
            self.assertIn(band, summary["unique_bands"])

    def test_summary_flat_parameterization_counts(self):
        """frequency_space_attempts and period_space_attempts are flat ints."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=126)
        _run_consensus_fit(lc)
        summary = lc.get_fit_history_summary()
        self.assertIn("frequency_space_attempts", summary)
        self.assertIn("period_space_attempts", summary)
        # consensus/spectral-mixture models use frequency space
        self.assertGreater(summary["frequency_space_attempts"], 0)
        self.assertEqual(
            summary["frequency_space_attempts"],
            summary["counts_by_parameterization"]["frequency_space"],
        )
        self.assertEqual(
            summary["period_space_attempts"],
            summary["counts_by_parameterization"]["period_space"],
        )

    def test_summary_flat_constraint_counts(self):
        """constrained_fits and unconstrained_fits must sum correctly."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=127)
        _run_consensus_fit(lc)
        summary = lc.get_fit_history_summary()
        self.assertIn("constrained_fits", summary)
        self.assertIn("unconstrained_fits", summary)
        self.assertEqual(
            summary["constrained_fits"],
            summary["counts_by_constraint_mode"]["constrained"],
        )
        self.assertEqual(
            summary["unconstrained_fits"],
            summary["counts_by_constraint_mode"]["unconstrained"],
        )

    def test_summary_backward_compat_minimal_entries(self):
        """Summary must not raise when history entries lack new fields."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0}, seed=128)
        # Manually inject a minimal legacy-style entry (no bands, no
        # uses_frequency_space, no constrained_fit, no constraint_set)
        lc.fit_history.append(
            {
                "timestamp_utc": "2020-01-01T00:00:00+00:00",
                "success": True,
                "failed": False,
                "elapsed_seconds": 1.5,
                "model_class": "SpectralMixtureGPModel",
                "fit_strategy": "standard",
                "backend": "cpu",
                "constrained": None,
                # New fields deliberately absent
            }
        )
        # Must not raise
        summary = lc.get_fit_history_summary()
        self.assertEqual(summary["total_attempts"], 1)
        self.assertEqual(summary["successful_fits"], 1)
        self.assertEqual(summary["total_runtime_seconds"], 1.5)
        # Parameterisation inferred from model class (spectral → frequency)
        self.assertEqual(summary["frequency_space_attempts"], 1)
        # Legacy entry has constrained=None → counts as unknown
        self.assertEqual(summary["counts_by_constraint_mode"]["unknown"], 1)
        self.assertIn("counts_by_fit_strategy", summary)

    def test_nested_delegated_fit_not_double_counted_explicit(self):
        """Consensus (internally calls fit for each band) adds exactly 1 entry."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=129)
        _run_consensus_fit(lc)
        _run_consensus_fit(lc)
        history = lc.get_fit_history()
        # Two outer fit() calls → exactly two history entries regardless of
        # how many internal band-level fits consensus performs.
        self.assertEqual(len(history), 2)
        summary = lc.get_fit_history_summary()
        self.assertEqual(summary["total_attempts"], 2)

    def test_entry_captures_bands(self):
        """Each history entry should record the bands present at fit time."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=130)
        _run_consensus_fit(lc)
        entry = lc.get_fit_history()[-1]
        self.assertIn("bands", entry)
        bands = entry["bands"]
        self.assertIsNotNone(bands)
        self.assertIsInstance(bands, list)
        for band in ("g", "r", "i"):
            self.assertIn(band, bands)

    def test_entry_captures_uses_frequency_space(self):
        """Consensus/spectral-mixture fits should record uses_frequency_space=True."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=131)
        _run_consensus_fit(lc)
        entry = lc.get_fit_history()[-1]
        self.assertIn("uses_frequency_space", entry)
        self.assertTrue(entry["uses_frequency_space"])
        self.assertFalse(entry.get("uses_period_space"))

    def test_entry_captures_constrained_fit_false(self):
        """constrain_consensus=False should produce constrained_fit=False."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=132)
        _run_consensus_fit(lc, constrain_consensus=False)
        entry = lc.get_fit_history()[-1]
        self.assertIn("constrained_fit", entry)
        self.assertFalse(entry["constrained_fit"])

    def test_summary_fit_strategy_unknown_for_legacy(self):
        """Legacy entries without fit_strategy are grouped under 'unknown'."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0}, seed=133)
        lc.fit_history.append(
            {
                "timestamp_utc": "2020-01-01T00:00:00+00:00",
                "success": True,
                "failed": False,
                "elapsed_seconds": 0.5,
                # fit_strategy deliberately absent
            }
        )
        summary = lc.get_fit_history_summary()
        self.assertEqual(summary["counts_by_fit_strategy"].get("unknown", 0), 1)


if __name__ == "__main__":
    unittest.main()
