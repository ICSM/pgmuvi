"""Scientific regression tests for consensus-fit frequency behavior."""

from __future__ import annotations

import math
import unittest

import numpy as np

from pgmuvi.lightcurve import Lightcurve


def _make_multiband_lightcurve(
    period_by_band,
    *,
    n_points_by_band=None,
    time_by_band=None,
    phase_by_band=None,
    amplitude_by_band=None,
    noise_std=0.02,
    seed=0,
):
    """Create a deterministic synthetic 2D light curve with explicit band labels."""
    rng = np.random.default_rng(seed)
    period_by_band = dict(period_by_band)
    bands = list(period_by_band)

    n_points_by_band = dict(n_points_by_band or {})
    time_by_band = dict(time_by_band or {})
    phase_by_band = dict(phase_by_band or {})
    amplitude_by_band = dict(amplitude_by_band or {})

    x_blocks = []
    y_blocks = []
    yerr_blocks = []
    band_blocks = []

    for band_index, band in enumerate(bands):
        if band in time_by_band:
            t = np.asarray(time_by_band[band], dtype=float)
        else:
            n_points = int(n_points_by_band.get(band, 60))
            t = np.linspace(0.0, 220.0, n_points, dtype=float)

        period = float(period_by_band[band])
        freq = 1.0 / period
        phase = float(phase_by_band.get(band, 0.0))
        amplitude = float(amplitude_by_band.get(band, 1.0))

        signal = amplitude * np.sin(2.0 * math.pi * freq * t + phase)
        noise = rng.normal(loc=0.0, scale=float(noise_std), size=t.shape)
        y = signal + noise
        yerr = np.full_like(t, float(noise_std), dtype=float)

        x_band = np.column_stack([
            t,
            np.full_like(t, float(band_index + 1), dtype=float),
        ])
        b_band = np.full(t.shape, str(band), dtype=np.str_)

        x_blocks.append(x_band)
        y_blocks.append(y)
        yerr_blocks.append(yerr)
        band_blocks.append(b_band)

    x_all = np.concatenate(x_blocks, axis=0)
    y_all = np.concatenate(y_blocks, axis=0)
    yerr_all = np.concatenate(yerr_blocks, axis=0)
    band_all = np.concatenate(band_blocks, axis=0)

    return Lightcurve(x_all, y_all, yerr=yerr_all, band=band_all)


def _run_consensus_frequency_regression(
    lc,
    *,
    use_acf=False,
    min_points_per_band=None,
    max_gap_fraction=None,
    min_duty_cycle=None,
    outlier_sigma=3.5,
):
    """Run consensus candidate collection+aggregation without GP training."""
    diagnostics = lc._consensus_initialize_result_structure(fit_strategy="consensus")
    diagnostics.update({
        "use_acf_validation": bool(use_acf),
        "use_gp_validation": False,
        "gp_validation_requested": False,
        "gp_validation_performed": False,
        "consensus_generation_method": "auto_consensus",
    })

    candidate_diag = lc._consensus_collect_band_candidates(
        min_points_per_band=min_points_per_band,
        max_gap_fraction=max_gap_fraction,
        min_duty_cycle=min_duty_cycle,
        use_acf=use_acf,
        gp_validation_requested=False,
        verbose=False,
    )

    per_band = dict(candidate_diag.get("band_records", {}))
    accepted = list(candidate_diag.get("accepted_bands", []))
    rejected = list(candidate_diag.get("rejected_bands", []))
    rejection_summary = lc._consensus_build_rejection_summary(
        per_band_diagnostics=per_band,
        rejected_bands=rejected,
        rejection_reasons=dict(candidate_diag.get("rejection_reasons", {})),
    )

    diagnostics.update({
        "accepted_bands": accepted,
        "rejected_bands": rejected,
        "per_band_diagnostics": per_band,
        "rejection_reasons": rejection_summary,
        "controls": dict(candidate_diag.get("controls", {})),
    })

    if not accepted:
        diagnostics.update({
            "consensus_success": False,
            "consensus_frequency": None,
            "consensus_period": None,
            "final_consensus_frequency": None,
            "final_consensus_period": None,
            "trusted_candidate_count": 0,
            "candidate_count": 0,
            "consensus_inlier_bands": [],
            "consensus_outlier_bands": [],
        })
        return lc._consensus_finalize_result_structure(diagnostics)

    consensus_diag = lc._consensus_build_frequency_consensus(
        band_records=per_band,
        accepted_bands=accepted,
        outlier_sigma=float(outlier_sigma),
        dedup_rtol=0.01,
        verbose=False,
    )

    final_frequency = float(consensus_diag["final_consensus_frequency"])
    final_period = float(1.0 / final_frequency)

    diagnostics.update({
        "consensus_success": True,
        "consensus_frequency": final_frequency,
        "consensus_period": final_period,
        "consensus_frequency_scatter": consensus_diag["mad_frequency_scatter"],
        "consensus_inlier_bands": list(consensus_diag["inlier_bands"]),
        "consensus_outlier_bands": list(consensus_diag["outlier_bands"]),
        "trusted_candidate_count": len(consensus_diag["inlier_bands"]),
        "candidate_count": len(consensus_diag["frequencies_all"]),
        "median_frequency": consensus_diag["median_frequency"],
        "mad_frequency_scatter": consensus_diag["mad_frequency_scatter"],
        "final_consensus_frequency": final_frequency,
        "final_consensus_period": final_period,
        "robust_frequency_width": consensus_diag["final_mad_frequency_scatter"],
    })

    return lc._consensus_finalize_result_structure(diagnostics)


def assert_frequency_close(actual, expected, tol):
    """Assert frequency proximity with an absolute tolerance."""
    if actual is None:
        raise AssertionError("actual frequency is None")
    delta = abs(float(actual) - float(expected))
    if delta > float(tol):
        raise AssertionError(
            f"frequency mismatch: actual={actual}, expected={expected}, tol={tol}"
        )


def assert_valid_consensus_diagnostics(lc, diagnostics):
    """Ensure diagnostics are schema-valid via the existing validator/finalizer."""
    validated = lc._consensus_finalize_result_structure(diagnostics)
    if not isinstance(validated, dict):
        raise AssertionError("finalized diagnostics must be a dictionary")


def assert_band_rejected_with_reason(diagnostics, band, reason_fragment):
    """Assert that a band was rejected with a reason containing fragment."""
    per_band = diagnostics.get("per_band_diagnostics", {})
    if str(band) not in per_band:
        raise AssertionError(f"band {band!r} missing from per_band_diagnostics")

    reasons = per_band[str(band)].get("rejection_reasons") or []
    reasons = [str(r) for r in reasons]
    if not any(str(reason_fragment) in r for r in reasons):
        raise AssertionError(
            f"band {band!r} rejection reasons {reasons!r} "
            f"do not include fragment {reason_fragment!r}"
        )


class TestConsensusScientificRegression(unittest.TestCase):
    """Scientific regression cases for consensus frequency aggregation."""

    def test_clean_single_period_multiband_recovery(self):
        injected_period = 32.0
        injected_frequency = 1.0 / injected_period
        lc = _make_multiband_lightcurve(
            {"g": injected_period, "r": injected_period, "i": injected_period},
            noise_std=0.01,
            seed=11,
        )

        diagnostics = _run_consensus_frequency_regression(lc, use_acf=False)

        self.assertTrue(diagnostics["consensus_success"])
        assert_frequency_close(
            diagnostics["final_consensus_frequency"],
            injected_frequency,
            tol=0.01,
        )
        self.assertEqual(set(diagnostics["accepted_bands"]), {"g", "r", "i"})
        self.assertEqual(diagnostics["rejected_bands"], [])
        assert_valid_consensus_diagnostics(lc, diagnostics)

    def test_one_discrepant_band_majority_frequency_wins(self):
        majority_period = 28.0
        discrepant_period = 14.0
        majority_frequency = 1.0 / majority_period

        lc = _make_multiband_lightcurve(
            {
                "g": majority_period,
                "r": majority_period,
                "i": majority_period,
                "z": discrepant_period,
            },
            noise_std=0.01,
            seed=17,
        )

        diagnostics = _run_consensus_frequency_regression(lc, use_acf=False)

        self.assertTrue(diagnostics["consensus_success"])
        assert_frequency_close(
            diagnostics["final_consensus_frequency"],
            majority_frequency,
            tol=0.02,
        )
        self.assertIn("z", diagnostics.get("consensus_outlier_bands", []))
        self.assertIn("z", diagnostics["accepted_bands"])
        assert_valid_consensus_diagnostics(lc, diagnostics)

    def test_low_quality_sampling_band_is_rejected(self):
        period = 26.0
        dense_t = np.linspace(0.0, 220.0, 60)
        sparse_gap_t = np.array([0.0, 2.0, 4.0, 6.0, 180.0, 200.0, 220.0])

        lc = _make_multiband_lightcurve(
            {"g": period, "r": period, "i": period},
            time_by_band={"g": dense_t, "r": dense_t, "i": sparse_gap_t},
            noise_std=0.01,
            seed=23,
        )

        diagnostics = _run_consensus_frequency_regression(
            lc,
            use_acf=False,
            min_points_per_band=20,
            max_gap_fraction=0.25,
            min_duty_cycle=0.10,
        )

        self.assertTrue(diagnostics["consensus_success"])
        self.assertIn("i", diagnostics["rejected_bands"])
        assert_band_rejected_with_reason(diagnostics, "i", "too_few_points")
        assert_valid_consensus_diagnostics(lc, diagnostics)

    def test_harmonic_alias_challenge_majority_not_pulled(self):
        true_period = 30.0
        harmonic_period = true_period / 2.0
        true_frequency = 1.0 / true_period

        lc = _make_multiband_lightcurve(
            {
                "g": true_period,
                "r": true_period,
                "i": true_period,
                "z": harmonic_period,
            },
            noise_std=0.01,
            seed=29,
        )

        diagnostics = _run_consensus_frequency_regression(
            lc,
            use_acf=True,
            outlier_sigma=2.5,
        )

        self.assertTrue(diagnostics["consensus_success"])
        assert_frequency_close(
            diagnostics["final_consensus_frequency"],
            true_frequency,
            tol=0.02,
        )
        self.assertIn("z", diagnostics.get("consensus_outlier_bands", []))
        assert_valid_consensus_diagnostics(lc, diagnostics)

    def test_no_consensus_case_with_sampling_failures(self):
        lc = _make_multiband_lightcurve(
            {"g": 22.0, "r": 31.0, "i": 47.0},
            n_points_by_band={"g": 6, "r": 7, "i": 5},
            noise_std=0.01,
            seed=31,
        )

        diagnostics = _run_consensus_frequency_regression(
            lc,
            use_acf=False,
            min_points_per_band=20,
            max_gap_fraction=0.25,
            min_duty_cycle=0.10,
        )

        self.assertFalse(diagnostics["consensus_success"])
        self.assertIsNone(diagnostics["final_consensus_frequency"])
        self.assertGreater(len(diagnostics["rejected_bands"]), 0)
        self.assertGreater(len(diagnostics.get("rejection_reasons", {})), 0)
        assert_valid_consensus_diagnostics(lc, diagnostics)


if __name__ == "__main__":
    unittest.main()
