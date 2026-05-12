from __future__ import annotations

import datetime
import enum
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import warnings

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
from gpytorch.constraints import Interval
from gpytorch.priors import NormalPrior

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


class _OpaqueConfig:
    pass


class _ConfigEnum(enum.Enum):
    FAST = "fast"


def _walk_values(value):
    """Yield all nested values in a fit-configuration structure."""
    yield value
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_values(nested)


def _contains_non_finite(value):
    """Return True if a structure still contains non-finite numeric values."""
    for nested in _walk_values(value):
        if isinstance(nested, float) and not math.isfinite(nested):
            return True
        if isinstance(nested, np.generic):
            try:
                if not math.isfinite(float(nested)):
                    return True
            except (TypeError, ValueError):
                continue
    return False


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
        # second call - still consensus
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


class TestValidateFitHistoryEntry(unittest.TestCase):
    """Unit tests for the _validate_fit_history_entry static method."""

    def test_elapsed_seconds_nan_becomes_none(self):
        """NaN elapsed_seconds should be coerced to None."""
        entry = {"elapsed_seconds": float("nan")}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertIsNone(entry["elapsed_seconds"])

    def test_elapsed_seconds_inf_becomes_none(self):
        """Inf elapsed_seconds should be coerced to None."""
        entry = {"elapsed_seconds": float("inf")}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertIsNone(entry["elapsed_seconds"])

    def test_elapsed_seconds_int_becomes_float(self):
        """A valid integer elapsed_seconds should become a float."""
        entry = {"elapsed_seconds": 3}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertIsInstance(entry["elapsed_seconds"], float)
        self.assertAlmostEqual(entry["elapsed_seconds"], 3.0)

    def test_elapsed_seconds_string_uncoercible_becomes_none(self):
        """An uncoercible string elapsed_seconds should become None."""
        entry = {"elapsed_seconds": "not-a-number"}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertIsNone(entry["elapsed_seconds"])

    def test_constrained_fit_int_coerced_to_bool(self):
        """Integer 1/0 constrained_fit values should be coerced to bool."""
        for raw, expected in [(1, True), (0, False)]:
            entry = {"constrained_fit": raw}
            Lightcurve._validate_fit_history_entry(entry)
            self.assertIsInstance(entry["constrained_fit"], bool)
            self.assertEqual(entry["constrained_fit"], expected)

    def test_constrained_fit_bool_unchanged(self):
        """Bool constrained_fit values must not be changed."""
        for val in (True, False):
            entry = {"constrained_fit": val}
            Lightcurve._validate_fit_history_entry(entry)
            self.assertIs(entry["constrained_fit"], val)

    def test_constrained_fit_none_unchanged(self):
        """None constrained_fit must remain None."""
        entry = {"constrained_fit": None}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertIsNone(entry["constrained_fit"])

    def test_bands_list_of_strings_unchanged(self):
        """A valid list of strings must be left intact (order preserved)."""
        entry = {"bands": ["g", "r", "i"]}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertEqual(entry["bands"], ["g", "r", "i"])

    def test_bands_non_string_elements_converted(self):
        """Non-string band labels should be converted to strings."""
        entry = {"bands": [1, 2.0, "i"]}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertEqual(entry["bands"], ["1", "2.0", "i"])

    def test_bands_duplicates_removed(self):
        """Duplicate band labels should be deduplicated (insertion order)."""
        entry = {"bands": ["g", "r", "g", "i", "r"]}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertEqual(entry["bands"], ["g", "r", "i"])

    def test_bands_tuple_converted_to_list(self):
        """A tuple of bands should be converted to a list of strings."""
        entry = {"bands": ("g", "r")}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertIsInstance(entry["bands"], list)
        self.assertEqual(entry["bands"], ["g", "r"])

    def test_timestamp_int_coerced_to_string(self):
        """An integer timestamp should be coerced to a string."""
        entry = {"timestamp_utc": 12345}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertIsInstance(entry["timestamp_utc"], str)
        self.assertEqual(entry["timestamp_utc"], "12345")

    def test_timestamp_string_unchanged(self):
        """A string timestamp must not be modified."""
        ts = "2024-01-01T00:00:00+00:00"
        entry = {"timestamp_utc": ts}
        Lightcurve._validate_fit_history_entry(entry)
        self.assertEqual(entry["timestamp_utc"], ts)

    def test_non_dict_entry_returned_unchanged(self):
        """Non-dict inputs must be returned as-is without raising."""
        for val in (None, 42, "string", [1, 2]):
            result = Lightcurve._validate_fit_history_entry(val)
            self.assertEqual(result, val)

    def test_entry_with_no_relevant_keys_unchanged(self):
        """Entries with unrelated keys must not be altered."""
        entry = {"success": True, "failed": False, "notes": {"x": 1}}
        before = dict(entry)
        Lightcurve._validate_fit_history_entry(entry)
        self.assertEqual(entry, before)


class TestFitHistorySummaryRobustness(unittest.TestCase):
    """Tests for get_fit_history_summary robustness and stable keys."""

    def _make_lc(self):
        return _make_multiband_lightcurve(
            {"g": 30.0, "r": 30.0}, seed=200
        )

    def test_empty_history_all_keys_present(self):
        """All expected summary keys are present even with empty history."""
        lc = self._make_lc()
        summary = lc.get_fit_history_summary()
        expected_keys = {
            "total_attempts",
            "successful_fits",
            "failed_fits",
            "success_fraction",
            "last_success_timestamp",
            "last_failure_timestamp",
            "earliest_timestamp",
            "latest_timestamp",
            "total_runtime_seconds",
            "mean_runtime_seconds",
            "counts_by_backend",
            "counts_by_model_class",
            "counts_by_fit_strategy",
            "counts_by_parameterization",
            "counts_by_constraint_mode",
            "unique_bands",
            "unique_bands_used",
            "frequency_space_attempts",
            "period_space_attempts",
            "constrained_fits",
            "unconstrained_fits",
        }
        for key in expected_keys:
            self.assertIn(key, summary, f"Key '{key}' missing from empty summary")

    def test_empty_history_zeros_and_nones(self):
        """Empty history produces zeros for counts and None for mean/timestamps."""
        lc = self._make_lc()
        summary = lc.get_fit_history_summary()
        self.assertEqual(summary["total_attempts"], 0)
        self.assertEqual(summary["successful_fits"], 0)
        self.assertEqual(summary["failed_fits"], 0)
        self.assertEqual(summary["total_runtime_seconds"], 0.0)
        self.assertIsNone(summary["mean_runtime_seconds"])
        self.assertIsNone(summary["earliest_timestamp"])
        self.assertIsNone(summary["latest_timestamp"])

    def test_empty_history_empty_dicts_and_lists(self):
        """Empty history returns empty dicts/lists for categorical fields."""
        lc = self._make_lc()
        lc.band = None  # strip band data to keep test predictable
        summary = lc.get_fit_history_summary()
        self.assertIsInstance(summary["counts_by_backend"], dict)
        self.assertIsInstance(summary["counts_by_model_class"], dict)
        self.assertIsInstance(summary["counts_by_fit_strategy"], dict)
        self.assertIsInstance(summary["unique_bands"], list)

    def test_summary_with_missing_runtime_entry(self):
        """Entries missing elapsed_seconds do not affect total_runtime."""
        lc = self._make_lc()
        lc.fit_history.append(
            {
                "timestamp_utc": "2021-01-01T00:00:00+00:00",
                "success": True,
                "failed": False,
                # elapsed_seconds deliberately absent
            }
        )
        summary = lc.get_fit_history_summary()
        self.assertEqual(summary["total_runtime_seconds"], 0.0)
        self.assertIsNone(summary["mean_runtime_seconds"])

    def test_summary_unique_bands_from_history_entries(self):
        """unique_bands aggregates band labels stored in history entries."""
        lc = self._make_lc()
        lc.fit_history.append(
            {
                "timestamp_utc": "2021-06-01T00:00:00+00:00",
                "success": True,
                "failed": False,
                "bands": ["g", "r", "i"],
            }
        )
        lc.fit_history.append(
            {
                "timestamp_utc": "2021-06-02T00:00:00+00:00",
                "success": True,
                "failed": False,
                "bands": ["g", "z"],
            }
        )
        summary = lc.get_fit_history_summary()
        for band in ("g", "r", "i", "z"):
            self.assertIn(band, summary["unique_bands"])

    def test_summary_unique_bands_deduplication(self):
        """Bands appearing in multiple history entries are deduplicated."""
        lc = self._make_lc()
        lc.fit_history.append(
            {
                "timestamp_utc": "2021-06-01T00:00:00+00:00",
                "success": True,
                "failed": False,
                "bands": ["g", "r", "g", "r"],
            }
        )
        summary = lc.get_fit_history_summary()
        # Sorted deduplicated list: ["g", "r"]
        self.assertEqual(summary["unique_bands"].count("g"), 1)
        self.assertEqual(summary["unique_bands"].count("r"), 1)

    def test_summary_unique_bands_non_string_labels(self):
        """Non-string band labels in history entries are normalized to strings."""
        lc = self._make_lc()
        lc.fit_history.append(
            {
                "timestamp_utc": "2021-06-01T00:00:00+00:00",
                "success": True,
                "failed": False,
                "bands": [1, 2, "g"],
            }
        )
        summary = lc.get_fit_history_summary()
        self.assertIn("1", summary["unique_bands"])
        self.assertIn("2", summary["unique_bands"])
        self.assertIn("g", summary["unique_bands"])


class TestPrintFitHistorySummary(unittest.TestCase):
    """Tests for the print_fit_history_summary() method."""

    def _make_lc_with_fit(self, seed=300):
        lc = _make_multiband_lightcurve(
            {"g": 30.0, "r": 30.0, "i": 30.0}, seed=seed
        )
        _run_consensus_fit(lc)
        return lc

    def test_returns_string(self):
        """print_fit_history_summary must return a non-empty string."""
        lc = self._make_lc_with_fit(seed=300)
        result = lc.print_fit_history_summary(print_summary=False)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_contains_section_headers(self):
        """Output must contain all expected section headings."""
        lc = self._make_lc_with_fit(seed=301)
        text = lc.print_fit_history_summary(print_summary=False)
        for heading in (
            "Fit History Summary",
            "Total fits",
            "Total runtime",
            "Mean runtime",
            "Time span",
            "Fit strategies",
            "Backends",
            "Models",
            "Parameterization",
            "Constraints",
            "Bands encountered",
        ):
            self.assertIn(heading, text, f"Heading '{heading}' missing")

    def test_contains_band_names(self):
        """Band names present at fit time should appear in the output."""
        lc = self._make_lc_with_fit(seed=302)
        text = lc.print_fit_history_summary(print_summary=False)
        for band in ("g", "r", "i"):
            self.assertIn(band, text)

    def test_contains_fit_counts(self):
        """The total-fits count should appear in the output."""
        lc = self._make_lc_with_fit(seed=303)
        text = lc.print_fit_history_summary(print_summary=False)
        self.assertIn("1", text)  # at least the count "1" for total fits

    def test_empty_history_does_not_raise(self):
        """print_fit_history_summary must not raise for empty history."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0}, seed=304)
        try:
            text = lc.print_fit_history_summary(print_summary=False)
        except Exception as exc:
            self.fail(
                f"print_fit_history_summary raised with empty history: {exc}"
            )
        self.assertIsInstance(text, str)

    def test_print_summary_true_prints(self, capsys=None):
        """With print_summary=True the text should be emitted to stdout."""
        import io
        import sys

        lc = self._make_lc_with_fit(seed=305)
        captured = io.StringIO()
        _old_stdout = sys.stdout
        sys.stdout = captured
        try:
            returned = lc.print_fit_history_summary(print_summary=True)
        finally:
            sys.stdout = _old_stdout
        printed = captured.getvalue()
        self.assertIn("Fit History Summary", printed)
        self.assertEqual(returned, printed.rstrip("\n"))

    def test_print_summary_false_does_not_print(self):
        """With print_summary=False nothing should go to stdout."""
        import io
        import sys

        lc = self._make_lc_with_fit(seed=306)
        captured = io.StringIO()
        _old_stdout = sys.stdout
        sys.stdout = captured
        try:
            lc.print_fit_history_summary(print_summary=False)
        finally:
            sys.stdout = _old_stdout
        self.assertEqual(captured.getvalue(), "")

    def test_indent_parameter_respected(self):
        """A custom indent should appear in the formatted output."""
        lc = self._make_lc_with_fit(seed=307)
        text4 = lc.print_fit_history_summary(print_summary=False, indent=4)
        # With indent=4, band lines begin with 4 spaces
        self.assertIn("    g", text4)

    def test_return_value_matches_printed_value(self):
        """Returned string should match what would be printed."""
        lc = self._make_lc_with_fit(seed=308)
        result = lc.print_fit_history_summary(print_summary=False)
        # Calling again with print_summary=True returns the same text
        result2 = lc.print_fit_history_summary(print_summary=False)
        self.assertEqual(result, result2)


class TestFitHistoryProvenance(unittest.TestCase):
    """Tests for git/RNG/environment provenance integration."""

    def test_git_provenance_helper_never_raises_outside_git_repo(self):
        """Git provenance collection must degrade to None outside a git repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                Lightcurve,
                "_fit_history_package_directory",
                return_value=Path(tmpdir),
            ):
                try:
                    provenance = Lightcurve._collect_git_provenance()
                except Exception as exc:
                    self.fail(f"_collect_git_provenance raised unexpectedly: {exc}")
        self.assertIsInstance(provenance, dict)
        self.assertEqual(
            provenance,
            {
                "git_commit_hash": None,
                "git_branch": None,
                "git_dirty_worktree": None,
                "git_remote_url": None,
            },
        )

    def test_git_helper_handles_subprocess_failure_gracefully(self):
        """Subprocess failures must not escape the git provenance helper."""
        with mock.patch(
            "pgmuvi.lightcurve.subprocess.run",
            side_effect=OSError("git unavailable"),
        ):
            provenance = Lightcurve._collect_git_provenance()
        self.assertIsInstance(provenance, dict)
        self.assertTrue(all(value is None for value in provenance.values()))

    def test_rng_provenance_helper_never_mutates_rng_state(self):
        """Collecting RNG provenance must not alter any RNG state."""
        np.random.seed(123)
        random.seed(456)
        torch.manual_seed(789)

        np_before = np.random.get_state()
        py_before = random.getstate()
        torch_before = torch.random.get_rng_state().clone()

        provenance = Lightcurve._collect_rng_provenance()

        np_after = np.random.get_state()
        py_after = random.getstate()
        torch_after = torch.random.get_rng_state()

        self.assertIsInstance(provenance, dict)
        self.assertIn("numpy_rng_state_token", provenance)
        self.assertIn("python_rng_state_token", provenance)
        self.assertIn("torch_initial_seed", provenance)
        self.assertIn("numpy_random_seed", provenance)
        self.assertIn("python_random_seed", provenance)
        self.assertIn("torch_random_seed", provenance)
        self.assertEqual(
            provenance["numpy_rng_state_token"], provenance["numpy_random_seed"]
        )
        self.assertEqual(
            provenance["python_rng_state_token"], provenance["python_random_seed"]
        )
        self.assertEqual(provenance["torch_initial_seed"], provenance["torch_random_seed"])
        self.assertEqual(np_before[0], np_after[0])
        self.assertTrue(np.array_equal(np_before[1], np_after[1]))
        self.assertEqual(np_before[2:], np_after[2:])
        self.assertEqual(py_before, py_after)
        self.assertTrue(torch.equal(torch_before, torch_after))

    def test_environment_provenance_helper_always_returns_dict(self):
        """Unified environment provenance must always return a dict."""
        provenance = Lightcurve._collect_environment_provenance()
        self.assertIsInstance(provenance, dict)
        self.assertIn("python_version", provenance)
        self.assertIn("pgmuvi_version", provenance)
        self.assertIn("torch_version", provenance)
        self.assertIn("gpytorch_version", provenance)
        self.assertIn("git", provenance)
        self.assertIn("rng", provenance)
        self.assertIsInstance(provenance["git"], dict)
        self.assertIsInstance(provenance["rng"], dict)

    def test_fit_history_entry_includes_schema_version(self):
        """Recorded fit-history entries must include schema version 2."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=400)
        _run_consensus_fit(lc)
        entry = lc.get_fit_history()[-1]
        self.assertIn("fit_history_schema_version", entry)
        self.assertEqual(entry["fit_history_schema_version"], 2)

    def test_fit_history_json_serialization_with_nested_provenance(self):
        """Nested provenance data in fit history must remain JSON serializable."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=401)
        _run_consensus_fit(lc)
        payload = json.dumps(lc.get_fit_history())
        self.assertIsInstance(payload, str)
        self.assertIn("fit_history_schema_version", payload)
        self.assertIn("\"git\"", payload)
        self.assertIn("\"rng\"", payload)

    def test_fit_history_entry_sanitizable_with_none_provenance_fields(self):
        """Entries with None-valued provenance fields must remain sanitizable."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0}, seed=402)
        lc._append_fit_history(
            environment={
                "python_version": None,
                "pgmuvi_version": None,
                "torch_version": None,
                "gpytorch_version": None,
                "git": {
                    "git_commit_hash": None,
                    "git_branch": None,
                    "git_dirty_worktree": None,
                    "git_remote_url": None,
                },
                "rng": {
                    "numpy_rng_state_token": None,
                    "numpy_random_seed": None,
                    "torch_initial_seed": None,
                    "torch_random_seed": None,
                    "python_rng_state_token": None,
                    "python_random_seed": None,
                    "torch_deterministic_algorithms": None,
                    "torch_cudnn_deterministic": None,
                    "torch_cudnn_benchmark": None,
                },
            },
        )
        entry = lc.get_fit_history()[-1]
        self.assertIsNone(entry["environment"]["git"]["git_commit_hash"])
        self.assertIsNone(entry["environment"]["rng"]["numpy_rng_state_token"])
        self.assertIsNone(entry["environment"]["rng"]["torch_initial_seed"])
        self.assertIsNone(entry["environment"]["rng"]["python_rng_state_token"])
        self.assertIsNone(entry["environment"]["rng"]["torch_random_seed"])
        json.dumps(entry)

    def test_plot_provenance_location_kwarg_works(self):
        """Plot provenance annotations should respect provenance_location."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0, "i": 30.0}, seed=403)
        _run_consensus_fit(lc)
        fig, ax = lc.plot_period_summary(
            show=False,
            annotate_provenance=True,
            provenance_location="upper right",
        )
        try:
            provenance_texts = [
                text for text in ax.texts if "model:" in text.get_text()
            ]
            self.assertEqual(len(provenance_texts), 1)
            text = provenance_texts[0]
            self.assertEqual(text.get_ha(), "right")
            self.assertEqual(text.get_va(), "top")
            self.assertEqual(text.get_position(), (0.98, 0.98))
        finally:
            plt.close(fig)


class TestFitConfigurationSnapshot(unittest.TestCase):
    """Tests for fit-configuration snapshot provenance."""

    def _make_lc(self, seed=500):
        return _make_multiband_lightcurve(
            {"g": 30.0, "r": 30.0, "i": 30.0}, seed=seed
        )

    # ------------------------------------------------------------------
    # Basic existence and structure
    # ------------------------------------------------------------------

    def test_fit_configuration_present_after_successful_fit(self):
        """A successful fit must record fit_configuration in the entry."""
        lc = self._make_lc(seed=500)
        _run_consensus_fit(lc)
        entry = lc.get_fit_history()[-1]
        self.assertIn("fit_configuration", entry)
        self.assertIsNotNone(entry["fit_configuration"])
        self.assertIsInstance(entry["fit_configuration"], dict)

    def test_fit_configuration_contains_core_fields(self):
        """fit_configuration must include key provenance fields."""
        lc = self._make_lc(seed=501)
        _run_consensus_fit(lc)
        entry = lc.get_fit_history()[-1]
        cfg = entry["fit_configuration"]
        self.assertIn("fit_strategy", cfg)
        self.assertIn("model_class", cfg)
        self.assertIn("backend", cfg)
        self.assertIn("training_iter", cfg)

    def test_fit_configuration_records_strategy(self):
        """fit_configuration.fit_strategy matches the strategy used."""
        lc = self._make_lc(seed=502)
        _run_consensus_fit(lc)
        entry = lc.get_fit_history()[-1]
        cfg = entry["fit_configuration"]
        self.assertEqual(cfg["fit_strategy"], "consensus")

    def test_fit_configuration_records_num_mixtures(self):
        """fit_configuration.num_mixtures reflects the argument passed."""
        result = Lightcurve._collect_fit_configuration_snapshot(
            fit_kwargs={"num_mixtures": 2, "fit_strategy": "consensus"},
            context={},
        )
        self.assertEqual(result["num_mixtures"], 2)

    def test_fit_configuration_records_training_iter_zero(self):
        """training_iter=0 should be stored faithfully."""
        lc = self._make_lc(seed=504)
        _run_consensus_fit(lc, training_iter=0)
        cfg = lc.get_fit_history()[-1]["fit_configuration"]
        self.assertEqual(cfg["training_iter"], 0)

    # ------------------------------------------------------------------
    # JSON serialisation
    # ------------------------------------------------------------------

    def test_fit_configuration_is_json_serializable(self):
        """fit_configuration must remain JSON serialisable after a real fit."""
        lc = self._make_lc(seed=505)
        _run_consensus_fit(lc)
        entry = lc.get_fit_history()[-1]
        payload = json.dumps(entry)
        self.assertIsInstance(payload, str)
        self.assertIn("fit_configuration", payload)

    def test_full_history_json_serializable_with_fit_configuration(self):
        """The full history list including fit_configuration must be JSON-safe."""
        lc = self._make_lc(seed=506)
        _run_consensus_fit(lc)
        with self.assertRaises(Exception):
            _run_consensus_fit(lc, min_points_per_band=10_000)
        payload = json.dumps(lc.get_fit_history())
        self.assertIsInstance(payload, str)

    # ------------------------------------------------------------------
    # Sanitizer: giant arrays
    # ------------------------------------------------------------------

    def test_sanitize_large_array_produces_summary(self):
        """Arrays longer than 20 elements become a summary dict."""
        big_array = np.arange(50, dtype=float)
        result = Lightcurve._sanitize_fit_configuration_value(big_array)
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("__truncated__"))
        self.assertEqual(result.get("type"), "ndarray")
        self.assertIn("shape", result)
        self.assertIn("preview", result)

    def test_sanitize_small_array_stays_list(self):
        """Arrays of ≤20 elements become a plain list."""
        small_array = np.array([1.0, 2.0, 3.0])
        result = Lightcurve._sanitize_fit_configuration_value(small_array)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    # ------------------------------------------------------------------
    # Sanitizer: tensors
    # ------------------------------------------------------------------

    def test_sanitize_tensor_produces_summary_dict(self):
        """Multi-element tensors must become summary dicts, not raw tensors."""
        t = torch.randn(30)
        result = Lightcurve._sanitize_fit_configuration_value(t)
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("__truncated__"))
        self.assertEqual(result.get("type"), "tensor")
        self.assertIn("shape", result)
        self.assertIn("preview", result)

    def test_sanitize_scalar_tensor_becomes_python_scalar(self):
        """A single-element tensor must become a plain Python scalar."""
        t = torch.tensor(3.14)
        result = Lightcurve._sanitize_fit_configuration_value(t)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 3.14, places=4)

    def test_sanitized_tensor_summary_is_json_safe(self):
        """Sanitized tensor summary must be JSON serialisable."""
        t = torch.randn(25)
        result = Lightcurve._sanitize_fit_configuration_value(t)
        payload = json.dumps(result)
        self.assertIsInstance(payload, str)

    # ------------------------------------------------------------------
    # Sanitizer: callables / classes
    # ------------------------------------------------------------------

    def test_sanitize_class_returns_placeholder_dict(self):
        """A class object must become an explicit unsupported-object placeholder."""
        result = Lightcurve._sanitize_fit_configuration_value(Lightcurve)
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("__unserializable__"))
        self.assertEqual(result.get("type"), "type")
        self.assertIn("Lightcurve", result.get("repr", ""))

    def test_sanitize_callable_returns_placeholder_dict(self):
        """A callable must become an explicit unsupported-object placeholder."""
        result = Lightcurve._sanitize_fit_configuration_value(lambda x: x)
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("__unserializable__"))
        self.assertEqual(result.get("type"), "callable")

    # ------------------------------------------------------------------
    # Sanitizer: malformed / unserializable values
    # ------------------------------------------------------------------

    def test_sanitize_broken_value_never_raises(self):
        """A value that raises on both repr and str must not propagate."""
        try:
            result = Lightcurve._sanitize_fit_configuration_value(_BrokenValue())
        except Exception as exc:
            self.fail(
                f"_sanitize_fit_configuration_value raised unexpectedly: {exc}"
            )
        self.assertTrue(result.get("__unserializable__"))

    def test_collect_snapshot_never_raises_with_bad_inputs(self):
        """_collect_fit_configuration_snapshot must not raise for bad inputs."""
        try:
            result = Lightcurve._collect_fit_configuration_snapshot(
                fit_kwargs={"broken": _BrokenValue(), "fit_strategy": "test"},
                context={"model_class": _BrokenValue()},
            )
        except Exception as exc:
            self.fail(
                f"_collect_fit_configuration_snapshot raised unexpectedly: {exc}"
            )
        self.assertIsInstance(result, dict)

    def test_collect_snapshot_missing_values_are_none(self):
        """Fields with no source in fit_kwargs/context must be None."""
        result = Lightcurve._collect_fit_configuration_snapshot(
            fit_kwargs={}, context={}
        )
        self.assertIsNone(result["fit_strategy"])
        self.assertIsNone(result["model_class"])
        self.assertIsNone(result["num_mixtures"])
        self.assertIsNone(result["constraint_set"])

    # ------------------------------------------------------------------
    # get_last_fit_configuration
    # ------------------------------------------------------------------

    def test_get_last_fit_configuration_returns_none_on_empty_history(self):
        """get_last_fit_configuration must return None when history is empty."""
        lc = self._make_lc(seed=510)
        self.assertIsNone(lc.get_last_fit_configuration())

    def test_get_last_fit_configuration_returns_none_when_no_cfg(self):
        """get_last_fit_configuration returns None if latest entry lacks cfg."""
        lc = self._make_lc(seed=511)
        lc.fit_history.append({"success": True, "failed": False})
        self.assertIsNone(lc.get_last_fit_configuration())

    def test_get_last_fit_configuration_returns_dict_after_fit(self):
        """get_last_fit_configuration returns a dict after a real fit."""
        lc = self._make_lc(seed=512)
        _run_consensus_fit(lc)
        cfg = lc.get_last_fit_configuration()
        self.assertIsNotNone(cfg)
        self.assertIsInstance(cfg, dict)

    def test_get_last_fit_configuration_is_deep_copy(self):
        """Mutating the returned dict must not affect the stored entry."""
        lc = self._make_lc(seed=513)
        _run_consensus_fit(lc)
        cfg = lc.get_last_fit_configuration()
        original_strategy = cfg.get("fit_strategy")
        cfg["fit_strategy"] = "mutated"
        cfg_again = lc.get_last_fit_configuration()
        self.assertEqual(cfg_again.get("fit_strategy"), original_strategy)

    def test_get_last_fit_configuration_reflects_latest_entry(self):
        """get_last_fit_configuration reflects the most recent fit."""
        lc = self._make_lc(seed=514)
        _run_consensus_fit(lc)
        # Manually inject a second entry with a distinct fit_configuration
        lc.fit_history.append(
            {
                "success": True,
                "failed": False,
                "fit_configuration": {"fit_strategy": "injected"},
            }
        )
        cfg = lc.get_last_fit_configuration()
        self.assertEqual(cfg.get("fit_strategy"), "injected")

    # ------------------------------------------------------------------
    # fit_configuration_to_text
    # ------------------------------------------------------------------

    def test_fit_configuration_to_text_no_history_returns_message(self):
        """fit_configuration_to_text returns a helpful string with no history."""
        lc = self._make_lc(seed=520)
        text = lc.fit_configuration_to_text()
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_fit_configuration_to_text_after_fit_contains_strategy(self):
        """fit_configuration_to_text must include the fit strategy."""
        lc = self._make_lc(seed=521)
        _run_consensus_fit(lc)
        text = lc.fit_configuration_to_text()
        self.assertIn("consensus", text)

    def test_fit_configuration_to_text_contains_section_header(self):
        """fit_configuration_to_text must include the 'Fit Configuration' header."""
        lc = self._make_lc(seed=522)
        _run_consensus_fit(lc)
        text = lc.fit_configuration_to_text()
        self.assertIn("Fit Configuration", text)
        self.assertIn("-" * 40, text)

    def test_fit_configuration_to_text_accepts_explicit_cfg(self):
        """fit_configuration_to_text must accept an explicit cfg dict."""
        lc = self._make_lc(seed=523)
        cfg = {
            "fit_strategy": "custom_strategy",
            "backend": "cpu",
            "model_class": "TestModel",
            "training_iter": 100,
            "num_mixtures": 4,
        }
        text = lc.fit_configuration_to_text(fit_configuration=cfg)
        self.assertIn("custom_strategy", text)
        self.assertIn("TestModel", text)
        self.assertIn("100", text)

    def test_fit_configuration_to_text_never_raises_on_bad_cfg(self):
        """fit_configuration_to_text must not raise for malformed cfg."""
        lc = self._make_lc(seed=524)
        try:
            text = lc.fit_configuration_to_text(
                fit_configuration={"fit_strategy": _BrokenValue()}
            )
        except Exception as exc:
            self.fail(f"fit_configuration_to_text raised unexpectedly: {exc}")
        self.assertIsInstance(text, str)

    # ------------------------------------------------------------------
    # Partial/failed fit snapshots
    # ------------------------------------------------------------------

    def test_failed_fit_still_records_fit_configuration(self):
        """A fit that fails with an exception must still record fit_configuration."""
        lc = self._make_lc(seed=530)
        with self.assertRaises(Exception):
            _run_consensus_fit(lc, min_points_per_band=10_000)
        history = lc.get_fit_history()
        self.assertEqual(len(history), 1)
        entry = history[-1]
        self.assertTrue(entry["failed"])
        # fit_configuration may be present (captured from context)
        # It may be None or a dict — it must not cause an error
        if entry.get("fit_configuration") is not None:
            self.assertIsInstance(entry["fit_configuration"], dict)

    def test_partial_fit_configuration_snapshot_is_json_safe(self):
        """A partial snapshot built from empty kwargs must be JSON serialisable."""
        result = Lightcurve._collect_fit_configuration_snapshot(
            fit_kwargs={}, context={}
        )
        payload = json.dumps(result)
        self.assertIsInstance(payload, str)

    def test_sanitize_unsupported_object_uses_placeholder(self):
        """Unsupported opaque objects must become explicit placeholders."""
        result = Lightcurve._sanitize_fit_configuration_value(_OpaqueConfig())
        self.assertTrue(result["__unserializable__"])
        self.assertEqual(result["type"], "_OpaqueConfig")
        self.assertEqual(result["module"], __name__)
        self.assertIn("_OpaqueConfig", result["repr"])

    def test_sanitize_large_string_uses_truncation_marker(self):
        """Very long strings must become explicit truncation markers."""
        result = Lightcurve._sanitize_fit_configuration_value("x" * 400)
        self.assertTrue(result["__truncated__"])
        self.assertEqual(result["type"], "str")
        self.assertEqual(result["length"], 400)
        self.assertIn("preview", result)

    def test_sanitize_large_list_uses_truncation_marker(self):
        """Large lists must become explicit truncation markers."""
        result = Lightcurve._sanitize_fit_configuration_value(list(range(100)))
        self.assertTrue(result["__truncated__"])
        self.assertEqual(result["type"], "list")
        self.assertEqual(result["length"], 100)
        self.assertEqual(len(result["preview"]), 20)

    def test_sanitize_nested_structure_limits_recursion_depth(self):
        """Recursion-depth protection must emit an explicit truncation marker."""
        nested = {"level0": {"level1": {"level2": {"level3": {"level4": {}}}}}}
        cursor = nested["level0"]["level1"]["level2"]["level3"]["level4"]
        for idx in range(5, 12):
            cursor[f"level{idx}"] = {}
            cursor = cursor[f"level{idx}"]
        result = Lightcurve._sanitize_fit_configuration_value(nested)
        payload = json.dumps(result)
        self.assertIn("__truncated__", payload)
        self.assertIn("max_depth", payload)

    def test_collect_snapshot_round_trips_with_realistic_objects(self):
        """Realistic scientific config objects must survive JSON round-trips."""
        snapshot = Lightcurve._collect_fit_configuration_snapshot(
            fit_kwargs={
                "fit_strategy": "consensus",
                "training_iter": 0,
                "constraint_set": Interval(0.1, 2.5),
                "prior_set": NormalPrior(0.0, 1.0),
                "optimizer": torch.optim.Adam,
                "kernel_kwargs": {
                    "dtype": torch.float64,
                    "device": torch.device("cpu"),
                    "mode": _ConfigEnum.FAST,
                    "bands": {"g", "r", "i"},
                    "weights": (
                        np.float64(1.5),
                        np.float64(2.5),
                    ),
                    "opaque": _OpaqueConfig(),
                },
                "long_string": "y" * 400,
                "big_list": list(range(50)),
                "non_finite": float("nan"),
            },
            context={
                "model_class": "TwoDSpectralMixtureGPModel",
                "backend": torch.device("cpu"),
            },
        )
        round_trip = json.loads(json.dumps(snapshot, allow_nan=False))
        self.assertEqual(round_trip, snapshot)
        self.assertFalse(_contains_non_finite(snapshot))
        for nested in _walk_values(snapshot):
            self.assertFalse(torch.is_tensor(nested))
            self.assertFalse(isinstance(nested, np.ndarray))
        self.assertEqual(snapshot["constraint_set"]["type"], "Interval")
        self.assertEqual(snapshot["prior_set"]["type"], "NormalPrior")
        self.assertEqual(
            snapshot["user_kwargs"]["kernel_kwargs"]["device"],
            "cpu",
        )
        self.assertEqual(
            snapshot["user_kwargs"]["kernel_kwargs"]["dtype"],
            "torch.float64",
        )
        self.assertTrue(
            snapshot["user_kwargs"]["kernel_kwargs"]["opaque"][
                "__unserializable__"
            ]
        )
        self.assertTrue(snapshot["user_kwargs"]["big_list"]["__truncated__"])
        self.assertTrue(snapshot["user_kwargs"]["long_string"]["__truncated__"])
        self.assertTrue(snapshot["user_kwargs"]["non_finite"]["__non_finite__"])

    def test_real_fit_configuration_round_trip_is_stable(self):
        """A real fit snapshot must survive a JSON round-trip unchanged."""
        lc = self._make_lc(seed=531)
        _run_consensus_fit(lc)
        snapshot = lc.get_last_fit_configuration()
        round_trip = json.loads(json.dumps(snapshot, allow_nan=False))
        self.assertEqual(round_trip, snapshot)
        self.assertFalse(_contains_non_finite(snapshot))

    def test_validate_snapshot_warns_for_placeholders_and_truncation(self):
        """Validation should warn about placeholders, truncations, and depth."""
        snapshot = Lightcurve._sanitize_fit_configuration_value(
            {
                "fit_strategy": "consensus",
                "model_class": "TwoDSpectralMixtureGPModel",
                "training_iter": 0,
                "backend": "cpu",
                "user_kwargs": {
                    "opaque": _OpaqueConfig(),
                    "deep": {
                        "l0": {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": 1}}}}}}
                    },
                    "big_list": list(range(100)),
                },
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Lightcurve._validate_fit_configuration_snapshot(snapshot)
        messages = [str(item.message) for item in caught]
        self.assertTrue(
            any("fit_configuration.user_kwargs.opaque" in msg for msg in messages)
        )
        self.assertTrue(
            any("fit_configuration.user_kwargs.big_list" in msg for msg in messages)
        )
        self.assertTrue(any("deeply nested" in msg for msg in messages))

    def test_validate_snapshot_raises_on_raw_tensor_leak(self):
        """Validation must fail loudly if a raw tensor escapes sanitization."""
        snapshot = {
            "fit_strategy": "consensus",
            "model_class": "TwoDSpectralMixtureGPModel",
            "training_iter": 0,
            "backend": "cpu",
            "user_kwargs": {"bad_tensor": torch.tensor([1.0, 2.0])},
        }
        with self.assertRaises(RuntimeError):
            Lightcurve._validate_fit_configuration_snapshot(snapshot)

    def test_validate_snapshot_raises_on_callable_leak(self):
        """Validation must fail loudly if a raw callable escapes sanitization."""
        snapshot = {
            "fit_strategy": "consensus",
            "model_class": "TwoDSpectralMixtureGPModel",
            "training_iter": 0,
            "backend": "cpu",
            "user_kwargs": {"bad_callable": lambda x: x},
        }
        with self.assertRaises(RuntimeError):
            Lightcurve._validate_fit_configuration_snapshot(snapshot)

    def test_fit_configuration_to_text_marks_placeholders_and_truncation(self):
        """Text summaries should visibly flag placeholders and truncation markers."""
        lc = self._make_lc(seed=532)
        cfg = {
            "fit_strategy": "consensus",
            "model_class": "TwoDSpectralMixtureGPModel",
            "training_iter": 0,
            "backend": "cpu",
            "user_kwargs": {
                "opaque": Lightcurve._sanitize_fit_configuration_value(
                    _OpaqueConfig()
                ),
                "big_list": Lightcurve._sanitize_fit_configuration_value(
                    list(range(100))
                ),
            },
        }
        text = lc.fit_configuration_to_text(fit_configuration=cfg)
        self.assertIn("[UNSERIALIZABLE]", text)
        self.assertIn("[TRUNCATED]", text)

    # ------------------------------------------------------------------
    # JSON export integration
    # ------------------------------------------------------------------

    def test_json_export_includes_fit_configuration(self):
        """Serialised fit history must contain 'fit_configuration' field."""
        lc = self._make_lc(seed=540)
        _run_consensus_fit(lc)
        payload = json.dumps(lc.get_fit_history())
        self.assertIn('"fit_configuration"', payload)

    def test_backward_compat_entry_without_fit_configuration(self):
        """Legacy entries without fit_configuration must not break JSON export."""
        lc = self._make_lc(seed=541)
        lc.fit_history.append(
            {
                "timestamp_utc": "2020-01-01T00:00:00+00:00",
                "success": True,
                "failed": False,
                # fit_configuration deliberately absent
            }
        )
        payload = json.dumps(lc.get_fit_history())
        self.assertIsInstance(payload, str)


if __name__ == "__main__":
    unittest.main()
