"""Regression tests for :class:`~pgmuvi.lightcurve.ConsensusFitError`.

These tests verify that:

A) :class:`ConsensusFitError` is raised instead of a bare ``RuntimeError``
   for scientifically meaningful consensus failures.

B) The exception carries a ``failure_diagnostics`` attribute with structured,
   JSON-serializable diagnostic data.

C) No partially initialized GP model remains attached to the
   :class:`~pgmuvi.lightcurve.Lightcurve` after a consensus failure.

D) The ``failure_diagnostics`` dict can be round-tripped through ``json.dumps``
   / ``json.loads`` without data loss (NaN/Inf → null is allowed).

E) A successful consensus fit behaves identically to before — the exception
   class is not raised for valid inputs.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pgmuvi.lightcurve import ConsensusFitError, FitFailureSummary, Lightcurve


# ---------------------------------------------------------------------------
# Helpers shared with test_consensus_scientific_regression
# ---------------------------------------------------------------------------


def _make_multiband_lightcurve(
    period_by_band,
    *,
    n_pts_by_band=None,
    noise_std=0.01,
    seed=0,
):
    """Build a deterministic synthetic multiband :class:`Lightcurve`."""
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
    """Drive ``lc.fit(fit_strategy='consensus', training_iter=0, ...)``.

    Returns the finalized ``lc.consensus_diagnostics`` dict on success or
    re-raises whatever exception the pipeline emits.
    """
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
    lc.fit(**kwargs)
    return lc.consensus_diagnostics


def _is_json_serializable(obj):
    """Return ``True`` if *obj* can be serialized with ``json.dumps``."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConsensusFitErrorClass(unittest.TestCase):
    """Unit tests for :class:`ConsensusFitError` itself (no Lightcurve needed)."""

    def test_is_runtime_error_subclass(self):
        """ConsensusFitError is a subclass of RuntimeError."""
        self.assertTrue(issubclass(ConsensusFitError, RuntimeError))

    def test_basic_construction(self):
        """ConsensusFitError can be constructed with just a message."""
        exc = ConsensusFitError("some message")
        self.assertEqual(str(exc), "some message")
        self.assertIsInstance(exc.failure_diagnostics, dict)
        self.assertEqual(exc.failure_diagnostics.get("status"), "failed")

    def test_construction_with_failure_diagnostics(self):
        """ConsensusFitError stores failure_diagnostics correctly."""
        fd = {
            "status": "failed",
            "reason": "insufficient_consensus_inliers",
            "n_inlier_bands": 1,
            "required_inliers": 2,
            "n_candidate_bands": 4,
            "candidate_periods": [18.0, 31.0, 47.0, 73.0],
        }
        exc = ConsensusFitError("msg", failure_diagnostics=fd)
        self.assertEqual(
            exc.failure_diagnostics["reason"], "insufficient_consensus_inliers"
        )
        self.assertEqual(exc.failure_diagnostics["n_inlier_bands"], 1)
        self.assertEqual(exc.failure_diagnostics["required_inliers"], 2)

    def test_failure_diagnostics_is_independent_copy(self):
        """Mutating the original dict after construction does not change stored copy."""
        fd = {"status": "failed", "reason": "no_accepted_bands"}
        exc = ConsensusFitError("msg", failure_diagnostics=fd)
        fd["extra"] = "should not appear"
        self.assertNotIn("extra", exc.failure_diagnostics)

    def test_failure_diagnostics_json_serializable(self):
        """failure_diagnostics can be serialized with json.dumps."""
        fd = {
            "status": "failed",
            "reason": "insufficient_consensus_inliers",
            "n_inlier_bands": 1,
            "required_inliers": 2,
            "n_candidate_bands": 4,
            "candidate_periods": [18.0, 31.0, 47.0, 73.0],
        }
        exc = ConsensusFitError("msg", failure_diagnostics=fd)
        self.assertTrue(
            _is_json_serializable(exc.failure_diagnostics),
            "failure_diagnostics must be JSON-serializable",
        )


class TestConsensusFitErrorRaised(unittest.TestCase):
    """Integration tests: ConsensusFitError is raised from lc.fit()."""

    # -----------------------------------------------------------------
    # A) ConsensusFitError is raised instead of bare RuntimeError
    # -----------------------------------------------------------------

    def test_no_accepted_bands_raises_consensus_fit_error(self):
        """All-bands quality failure raises ConsensusFitError, not bare RuntimeError."""
        lc = _make_multiband_lightcurve(
            {"g": 7.0, "r": 30.0, "i": 130.0},
            n_pts_by_band={"g": 3, "r": 4, "i": 3},
            noise_std=0.01,
            seed=42,
        )
        with self.assertRaises(ConsensusFitError):
            _run_consensus_fit(lc, min_points_per_band=20)

    def test_insufficient_inliers_raises_consensus_fit_error(self):
        """Inconsistent-period failure raises ConsensusFitError."""
        lc = _make_multiband_lightcurve(
            {"g": 18.0, "r": 31.0, "i": 47.0, "z": 73.0},
            noise_std=0.01,
            seed=77,
        )
        with self.assertRaises(ConsensusFitError):
            _run_consensus_fit(lc, outlier_sigma=1.5, min_consensus_inliers=4)

    def test_no_accepted_bands_is_also_runtime_error(self):
        """ConsensusFitError is a RuntimeError (backwards-compatible)."""
        lc = _make_multiband_lightcurve(
            {"g": 7.0, "r": 30.0},
            n_pts_by_band={"g": 3, "r": 3},
            noise_std=0.01,
            seed=99,
        )
        with self.assertRaises(RuntimeError):
            _run_consensus_fit(lc, min_points_per_band=20)

    # -----------------------------------------------------------------
    # B) Exception carries structured failure_diagnostics
    # -----------------------------------------------------------------

    def test_no_accepted_bands_diagnostics_structure(self):
        """no_accepted_bands failure_diagnostics has expected keys."""
        lc = _make_multiband_lightcurve(
            {"g": 7.0, "r": 30.0, "i": 130.0},
            n_pts_by_band={"g": 3, "r": 4, "i": 3},
            noise_std=0.01,
            seed=42,
        )
        with self.assertRaises(ConsensusFitError) as cm:
            _run_consensus_fit(lc, min_points_per_band=20)

        fd = cm.exception.failure_diagnostics
        self.assertEqual(fd.get("status"), "failed")
        self.assertEqual(fd.get("reason"), "no_accepted_bands")
        self.assertIn("rejection_reasons", fd)

    def test_insufficient_inliers_diagnostics_structure(self):
        """insufficient_consensus_inliers failure_diagnostics has expected keys."""
        lc = _make_multiband_lightcurve(
            {"g": 18.0, "r": 31.0, "i": 47.0, "z": 73.0},
            noise_std=0.01,
            seed=77,
        )
        with self.assertRaises(ConsensusFitError) as cm:
            _run_consensus_fit(lc, outlier_sigma=1.5, min_consensus_inliers=4)

        fd = cm.exception.failure_diagnostics
        self.assertEqual(fd.get("status"), "failed")
        self.assertEqual(fd.get("reason"), "insufficient_consensus_inliers")
        self.assertIn("n_inlier_bands", fd)
        self.assertIn("required_inliers", fd)
        self.assertIn("n_candidate_bands", fd)
        self.assertIn("candidate_periods", fd)
        self.assertIsInstance(fd["candidate_periods"], list)
        self.assertGreater(fd.get("n_candidate_bands", 0), 0)
        self.assertGreater(fd.get("required_inliers", 0), fd.get("n_inlier_bands", 999))

    # -----------------------------------------------------------------
    # C) No partial GP model after failure
    # -----------------------------------------------------------------

    def test_no_partial_gp_model_after_quality_failure(self):
        """lc.gp_model is not set to a broken object after quality-gate failure."""
        lc = _make_multiband_lightcurve(
            {"g": 10.0, "r": 25.0},
            n_pts_by_band={"g": 3, "r": 3},
            noise_std=0.01,
            seed=5,
        )
        pre_model = getattr(lc, "gp_model", None)
        try:
            _run_consensus_fit(lc, min_points_per_band=20)
        except ConsensusFitError:
            pass
        # After failure the gp_model attribute must not have been set to a
        # new non-None value that wasn't there before.
        post_model = getattr(lc, "gp_model", None)
        if pre_model is None:
            self.assertIsNone(
                post_model,
                "gp_model must not be set after consensus failure when it "
                "was None before.",
            )

    def test_no_partial_gp_model_after_inlier_failure(self):
        """lc.gp_model is not set to a broken object after inlier failure."""
        lc = _make_multiband_lightcurve(
            {"g": 18.0, "r": 31.0, "i": 47.0, "z": 73.0},
            noise_std=0.01,
            seed=77,
        )
        pre_model = getattr(lc, "gp_model", None)
        try:
            _run_consensus_fit(lc, outlier_sigma=1.5, min_consensus_inliers=4)
        except ConsensusFitError:
            pass
        post_model = getattr(lc, "gp_model", None)
        if pre_model is None:
            self.assertIsNone(
                post_model,
                "gp_model must not be set after consensus failure when it "
                "was None before.",
            )

    # -----------------------------------------------------------------
    # D) failure_diagnostics is JSON-serializable
    # -----------------------------------------------------------------

    def test_no_accepted_bands_fd_json_serializable(self):
        """failure_diagnostics from no_accepted_bands is JSON-serializable."""
        lc = _make_multiband_lightcurve(
            {"g": 7.0, "r": 30.0, "i": 130.0},
            n_pts_by_band={"g": 3, "r": 4, "i": 3},
            noise_std=0.01,
            seed=42,
        )
        with self.assertRaises(ConsensusFitError) as cm:
            _run_consensus_fit(lc, min_points_per_band=20)

        fd = cm.exception.failure_diagnostics
        self.assertTrue(
            _is_json_serializable(fd),
            f"failure_diagnostics not JSON-serializable: {fd!r}",
        )
        # Round-trip check
        rt = json.loads(json.dumps(fd))
        self.assertEqual(rt["status"], "failed")
        self.assertEqual(rt["reason"], "no_accepted_bands")

    def test_insufficient_inliers_fd_json_serializable(self):
        """failure_diagnostics from insufficient_inliers is JSON-serializable."""
        lc = _make_multiband_lightcurve(
            {"g": 18.0, "r": 31.0, "i": 47.0, "z": 73.0},
            noise_std=0.01,
            seed=77,
        )
        with self.assertRaises(ConsensusFitError) as cm:
            _run_consensus_fit(lc, outlier_sigma=1.5, min_consensus_inliers=4)

        fd = cm.exception.failure_diagnostics
        self.assertTrue(
            _is_json_serializable(fd),
            f"failure_diagnostics not JSON-serializable: {fd!r}",
        )
        rt = json.loads(json.dumps(fd))
        self.assertEqual(rt["status"], "failed")
        self.assertEqual(rt["reason"], "insufficient_consensus_inliers")

    # -----------------------------------------------------------------
    # E) Successful consensus fit still works
    # -----------------------------------------------------------------

    def test_successful_fit_does_not_raise(self):
        """A clean single-period consensus fit does not raise ConsensusFitError."""
        lc = _make_multiband_lightcurve(
            {"g": 30.0, "r": 30.0, "i": 30.0},
            noise_std=0.01,
            seed=0,
        )
        try:
            diag = _run_consensus_fit(lc)
        except ConsensusFitError as exc:
            self.fail(
                f"ConsensusFitError raised unexpectedly for clean input: {exc}"
            )
        self.assertTrue(diag["consensus_success"])
        self.assertIsNotNone(diag["final_consensus_frequency"])

    def test_majority_cluster_succeeds(self):
        """3 bands near 30 d + 1 discrepant band at 10 d → consensus succeeds."""
        lc = _make_multiband_lightcurve(
            {"g": 26.0, "r": 30.0, "i": 34.0, "z": 10.0},
            noise_std=0.01,
            seed=11,
        )
        try:
            diag = _run_consensus_fit(lc, outlier_sigma=2.0, min_consensus_inliers=2)
        except ConsensusFitError as exc:
            self.fail(
                f"ConsensusFitError raised for majority-cluster input: {exc}"
            )
        self.assertTrue(
            diag["consensus_success"],
            "Majority cluster of 3 bands near 30 d should succeed",
        )
        final_freq = diag.get("final_consensus_frequency")
        self.assertIsNotNone(final_freq)
        # Consensus frequency should be close to 1/30 ≈ 0.0333 day⁻¹
        self.assertAlmostEqual(float(final_freq), 1.0 / 30.0, delta=0.01)


class TestConsensusFailureStateAndUX(unittest.TestCase):
    """Regression tests for failure-state robustness and notebook UX."""

    def _fit_success_then_fail(self):
        lc = _make_multiband_lightcurve(
            {"g": 30.0, "r": 30.0, "i": 30.0},
            noise_std=0.01,
            seed=123,
        )
        _run_consensus_fit(lc)
        _ = lc.get_period_summary()
        with self.assertRaises(ConsensusFitError):
            _run_consensus_fit(lc, min_points_per_band=10_000)
        return lc

    def test_failed_consensus_sets_formal_failure_contract(self):
        """Failed consensus run populates canonical failure-state fields."""
        lc = self._fit_success_then_fail()
        self.assertFalse(lc.is_fitted)
        self.assertTrue(lc.fit_failed)
        self.assertIsNotNone(lc.failure_reason)
        self.assertIsNotNone(lc.failure_diagnostics)
        self.assertTrue(_is_json_serializable(lc.failure_diagnostics))

    def test_period_summary_and_plots_raise_cleanly_after_failure(self):
        """Summary/plot helpers raise clean ConsensusFitError after failure."""
        lc = self._fit_success_then_fail()
        with self.assertRaises(ConsensusFitError) as cm_summary:
            lc.get_period_summary()
        self.assertIn("Cannot generate GP period summary", str(cm_summary.exception))

        with self.assertRaises(ConsensusFitError) as cm_plot:
            lc.plot(show=False)
        self.assertIn("Cannot generate GP fit plot", str(cm_plot.exception))

        with self.assertRaises(ConsensusFitError):
            lc.plot_psd(show=False)
        with self.assertRaises(ConsensusFitError):
            lc.plot_period_summary(show=False)

    def test_fit_failure_summary_serializes_and_is_readable(self):
        """FitFailureSummary to_dict/write_json/to_text are notebook-friendly."""
        summary = FitFailureSummary(
            status="failed",
            reason="mutually_inconsistent_periods",
            message=(
                "Consensus fit failed because only one band remained after "
                "consistency filtering."
            ),
            diagnostics={
                "candidate_periods": np.array([26.0, 30.0, 34.0]),
                "scatter": np.nan,
                "quality_flag": True,
            },
        )
        as_dict = summary.to_dict()
        self.assertEqual(as_dict["status"], "failed")
        self.assertTrue(_is_json_serializable(as_dict))
        self.assertIsNone(as_dict["diagnostics"]["scatter"])
        self.assertIn("Consensus fit failed because", summary.to_text())

        as_dict_with_history = summary.to_dict(
            include_fit_history=True,
            fit_history=[
                {
                    "timestamp_utc": "2026-05-10T18:24:03+00:00",
                    "elapsed_seconds": np.nan,
                }
            ],
        )
        self.assertIn("fit_history", as_dict_with_history)
        self.assertIsNone(as_dict_with_history["fit_history"][0]["elapsed_seconds"])

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "failure_summary.json"
            summary.write_json(
                out_path,
                include_fit_history=True,
                fit_history=[{"timestamp_utc": "2026-05-10T18:24:03+00:00"}],
            )
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["reason"], "mutually_inconsistent_periods")
        self.assertIn("candidate_periods", loaded["diagnostics"])
        self.assertIn("fit_history", loaded)

    def test_reset_fit_state_clears_relevant_attributes(self):
        """_reset_fit_state clears fit, cache, diagnostics, and model handles."""
        lc = _make_multiband_lightcurve({"g": 30.0, "r": 30.0}, seed=9)
        lc.results = {"loss": [1.0]}
        lc._period_summary_cache = {"dummy": 1}
        lc.optimizer = object()
        lc.gp_model = object()
        lc.consensus_diagnostics = {"status": "ok"}
        lc.failure_summary = FitFailureSummary(reason="x", message="y")
        lc.fit_failed = True
        lc.failure_reason = "x"
        lc.failure_diagnostics = {"status": "failed", "reason": "x"}
        lc.model = object()
        lc.likelihood = object()
        lc._model_pars = {"k": 1}

        lc._reset_fit_state(
            clear_failure=True,
            clear_model_state=True,
            clear_consensus=True,
        )

        self.assertFalse(lc.is_fitted)
        self.assertFalse(lc.fit_failed)
        self.assertIsNone(lc.failure_reason)
        self.assertIsNone(lc.failure_diagnostics)
        self.assertIsNone(lc.failure_summary)
        self.assertIsNone(lc.results)
        self.assertIsNone(lc._period_summary_cache)
        self.assertIsNone(lc.optimizer)
        self.assertIsNone(lc.gp_model)
        self.assertIsNone(lc.consensus_diagnostics)
        self.assertIsNone(lc.model)
        self.assertIsNone(lc.likelihood)
        self.assertIsNone(lc._model_pars)


if __name__ == "__main__":
    unittest.main()
