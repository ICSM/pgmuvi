"""Scientific regression tests for consensus-fit frequency behavior."""

from __future__ import annotations

import math
import unittest

import numpy as np

from pgmuvi.lightcurve import Lightcurve


def _make_multiband_lightcurve(
    period_by_band,
    *,
<<<<<<< HEAD
    n_pts_by_band=None,
    time_by_band=None,
    noise_std=0.01,
    seed=0,
):
    """Build a deterministic synthetic multiband ``Lightcurve``.

    Parameters
    ----------
    period_by_band : mapping
        ``{band_label: period_days}`` for each photometric band.
    n_pts_by_band : mapping, optional
        ``{band_label: n_points}``.  Defaults to 60 per band.
    time_by_band : mapping, optional
        ``{band_label: array_like}`` of observation times (days).
        Overrides ``n_pts_by_band`` for that band.
    noise_std : float, optional
        Gaussian noise standard deviation (magnitudes).  Default 0.01.
    seed : int, optional
        PRNG seed for reproducibility.  Default 0.

    Returns
    -------
    Lightcurve
        A 2-D multiband ``Lightcurve`` ready for consensus fitting.
    """
    rng = np.random.default_rng(seed)
    period_by_band = dict(period_by_band)
    bands = list(period_by_band)
    n_pts_by_band = dict(n_pts_by_band or {})
    time_by_band = dict(time_by_band or {})

    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    yerr_blocks: list[np.ndarray] = []
    band_blocks: list[np.ndarray] = []

    for band_idx, band in enumerate(bands):
        if band in time_by_band:
            t = np.asarray(time_by_band[band], dtype=float)
        else:
            n = int(n_pts_by_band.get(band, 60))
            t = np.linspace(0.0, 220.0, n)

        period = float(period_by_band[band])
        signal = np.sin(2.0 * math.pi * t / period)
        noise = rng.normal(0.0, float(noise_std), t.shape)
        y = signal + noise
        yerr = np.full_like(t, float(noise_std))

        # 2-D input: column 0 = time, column 1 = wavelength proxy
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


# ---------------------------------------------------------------------------
# Public consensus path runner
# ---------------------------------------------------------------------------


def run_public_consensus_fit(
    lc,
    *,
    use_acf=False,
    outlier_sigma=None,
    min_points_per_band=None,
    max_gap_fraction=None,
    min_duty_cycle=None,
    min_consensus_inliers=None,
):
    """Run consensus via the public ``lc.fit(fit_strategy="consensus", ...)`` path.

    Uses ``training_iter=0`` to skip GP optimisation while exercising the
    full consensus pipeline: sampling quality gating, per-band LS, outlier
    detection, and frequency aggregation.

    Parameters
    ----------
    lc : Lightcurve
        The multiband light curve to fit.
    use_acf : bool, optional
        Enable ACF consistency diagnostics.
    outlier_sigma : float, optional
        Robust sigma threshold for frequency outlier rejection.
    min_points_per_band : int, optional
        Minimum number of points required to accept a band.
    max_gap_fraction : float, optional
        Maximum allowed largest-gap fraction per band.
    min_duty_cycle : float, optional
        Minimum allowed duty-cycle estimate per band.
    min_consensus_inliers : int, optional
        Minimum number of bands that must survive outlier rejection and
        cluster around a common frequency for consensus to succeed.  When
        provided, overrides the default (``2``).

    Returns
    -------
    dict
        The finalized ``lc.consensus_diagnostics`` dict.

    Raises
    ------
<<<<<<< HEAD
    ConsensusFitError
        Raised (as a subclass of ``RuntimeError``) from ``lc.fit`` when all
        bands fail quality gating or when fewer than ``min_consensus_inliers``
        bands form a consistent inlier cluster.
=======
    RuntimeError
        Re-raised from ``lc.fit`` when all bands fail quality gating or
        when fewer than ``min_consensus_inliers`` bands form a consistent
        inlier cluster.
>>>>>>> 4ef607d (fix: add min_consensus_inliers guard for mutually inconsistent periods)
    """
    kwargs: dict = {
        "fit_strategy": "consensus",
        "model": "2D",
        "training_iter": 0,
        "use_gp_validation": False,
        "use_mls_init": False,
        "use_best_band_init": False,
        "constrain_consensus": False,
        "use_acf": use_acf,
    }
    if outlier_sigma is not None:
        kwargs["outlier_sigma"] = float(outlier_sigma)
    if min_points_per_band is not None:
        kwargs["min_points_per_band"] = int(min_points_per_band)
    if max_gap_fraction is not None:
        kwargs["max_gap_fraction"] = float(max_gap_fraction)
    if min_duty_cycle is not None:
        kwargs["min_duty_cycle"] = float(min_duty_cycle)
    if min_consensus_inliers is not None:
        kwargs["min_consensus_inliers"] = int(min_consensus_inliers)

    lc.fit(**kwargs)
    return lc.consensus_diagnostics


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_frequency_close(tc, actual, expected, tol, msg=None):
    """Assert ``|actual - expected| <= tol`` in frequency units (day⁻¹).

    Parameters
    ----------
    tc : unittest.TestCase
        The test case used for failure reporting.
    actual : float or None
        Recovered consensus frequency.
    expected : float
        Injected / true frequency.
    tol : float
        Absolute tolerance (day⁻¹).  Should reflect LS grid resolution and
        sampling effects rather than being artificially tight.
    msg : str, optional
        Custom failure message.
    """
    if actual is None:
        tc.fail(f"actual frequency is None (expected ~{expected:.6g} day⁻¹)")
    delta = abs(float(actual) - float(expected))
    if delta > float(tol):
        tc.fail(
            msg
            or (
                f"frequency mismatch: actual={actual:.6g}, "
                f"expected={expected:.6g}, tol={tol:.6g}, delta={delta:.6g}"
            )
        )


def assert_valid_consensus_diagnostics(tc, lc, diagnostics):
    """Assert diagnostics are schema-valid using the existing validator.

    Calls ``lc._consensus_validate_result_structure`` directly — no schema
    logic is duplicated here.
    """
    try:
        lc._consensus_validate_result_structure(diagnostics)
    except Exception as exc:
        tc.fail(f"consensus diagnostics failed schema validation: {exc}")


def assert_band_excluded_or_outlier(tc, diagnostics, band, msg=None):
    """Assert band is in ``rejected_bands`` or ``consensus_outlier_bands``.

    Fails if the band appears to have been silently accepted as an inlier.
    """
    rejected = diagnostics.get("rejected_bands", [])
    outliers = diagnostics.get("consensus_outlier_bands", [])
    if band not in rejected and band not in outliers:
        tc.fail(
            msg
            or (
                f"band {band!r} was silently treated as an inlier — "
                f"expected it in rejected_bands={rejected!r} or "
                f"consensus_outlier_bands={outliers!r}"
            )
        )


# ---------------------------------------------------------------------------
# Scientific regression test suite
# ---------------------------------------------------------------------------


class TestConsensusScientificRegression(unittest.TestCase):
    """End-to-end scientific regressions for the consensus-fit workflow.

    All tests drive ``lc.fit(fit_strategy="consensus", training_iter=0, ...)``.
    Assertions are placed on ``lc.consensus_diagnostics``, which is written
    by the consensus pipeline before the (skipped) GP optimisation step.

    Frequency-space tolerances are set at 3–5× the LS grid spacing for a
    220-day baseline (≈ 0.0023 day⁻¹ per grid step), giving practical
    tolerances of 0.005–0.010 day⁻¹.
    """

    # ------------------------------------------------------------------
    # Test 1: clean single-period multiband recovery
    # ------------------------------------------------------------------

    def test_clean_single_period_multiband_recovery(self):
        """All bands carry the same injected period; expect full acceptance.

        Injected period : 30.0 days  (frequency 0.03333 day⁻¹)
        Sampling        : 60 pts / band over 220 days (cadence ≈ 3.7 days)
        Tolerance       : 0.005 day⁻¹  (≈ 2× LS grid spacing)

        Expected behaviour
        ------------------
        * ``consensus_success`` is True.
        * All three bands pass quality gating.
        * ``rejected_bands`` is empty.
        * Consensus frequency within 0.005 day⁻¹ of injected frequency.
        * Finalized diagnostics pass schema validation.
        """
        injected_period = 30.0
        injected_freq = 1.0 / injected_period

=======
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
>>>>>>> b7ea677 (Add consensus scientific regression test module)
        lc = _make_multiband_lightcurve(
            {"g": injected_period, "r": injected_period, "i": injected_period},
            noise_std=0.01,
            seed=11,
        )
<<<<<<< HEAD
        diagnostics = run_public_consensus_fit(lc, use_acf=False)

        self.assertTrue(diagnostics["consensus_success"])
        assert_frequency_close(
            self, diagnostics["final_consensus_frequency"], injected_freq, tol=0.005
        )
        self.assertCountEqual(diagnostics["accepted_bands"], ["g", "r", "i"])
        self.assertEqual(diagnostics["rejected_bands"], [])
        assert_valid_consensus_diagnostics(self, lc, diagnostics)

    # ------------------------------------------------------------------
    # Test 2: one discrepant band detected as frequency outlier
    # ------------------------------------------------------------------

    def test_one_discrepant_band_is_detected_as_outlier(self):
        """Majority period cluster + one discrepant band far in frequency space.

        Design note
        -----------
        The consensus outlier filter requires ≥ 3 *deduplicated* frequency
        candidates.  Using three majority bands with *slightly different*
        injected periods (26, 30, 34 days) ensures each band produces a
        distinct LS peak that survives the deduplication step
        (relative spacing ≈ 7–14 %, well above the 1 % dedup threshold).
        The discrepant band is injected at 10 days, giving a frequency
        (≈ 0.100 day⁻¹) that is > 4 robust-sigma from the majority median,
        which triggers sigma-clipping rejection.

        Majority periods  : 26, 30, 34 days
        Discrepant period : 10 days  (frequency ≈ 0.100 day⁻¹)
        Majority median f : ≈ 1/30 = 0.0333 day⁻¹
        Tolerance         : 0.007 day⁻¹  (≈ 3× LS grid spacing)

        Expected behaviour
        ------------------
        * ``consensus_success`` is True.
        * Band 'z' appears in ``rejected_bands`` or ``consensus_outlier_bands``.
        * Final consensus frequency within 0.007 day⁻¹ of 1/30 day⁻¹.
        * Finalized diagnostics pass schema validation.
        """
        majority_median_freq = 1.0 / 30.0

        lc = _make_multiband_lightcurve(
            {"g": 26.0, "r": 30.0, "i": 34.0, "z": 10.0},
            noise_std=0.01,
            seed=42,
        )
        diagnostics = run_public_consensus_fit(lc, use_acf=False, outlier_sigma=2.5)

        self.assertTrue(diagnostics["consensus_success"])
        # Discrepant band must NOT be silently treated as an inlier.
        assert_band_excluded_or_outlier(self, diagnostics, "z")
        # Consensus must track the majority cluster, not the discrepant band.
        assert_frequency_close(
            self,
            diagnostics["final_consensus_frequency"],
            majority_median_freq,
            tol=0.007,
        )
        assert_valid_consensus_diagnostics(self, lc, diagnostics)

    # ------------------------------------------------------------------
    # Test 3: one low-quality sampling band is rejected
    # ------------------------------------------------------------------

    def test_low_quality_sampling_band_is_rejected(self):
        """One band is sparsely sampled with a large gap; must fail quality gating.

        Band 'i' has only 8 points with a > 60 % temporal gap, which
        violates both the minimum-points and maximum-gap-fraction thresholds.
        Bands 'g' and 'r' have dense uniform coverage and are accepted.

        Injected period : 30.0 days for all bands
        Sparse band     : 8 pts, gap fraction > 60 %
        Quality thresholds : min_points_per_band=20, max_gap_fraction=0.25
        Tolerance         : 0.005 day⁻¹

        Expected behaviour
        ------------------
        * Band 'i' appears in ``rejected_bands``.
        * Band 'i' has at least one quality-related rejection reason.
        * ``consensus_success`` is True using bands 'g' and 'r'.
        * Consensus frequency within 0.005 day⁻¹ of injected frequency.
        * Finalized diagnostics pass schema validation.
        """
        injected_period = 30.0
        injected_freq = 1.0 / injected_period
        dense_t = np.linspace(0.0, 220.0, 60)
        # 8 points with a ~145-day gap (gap fraction ≈ 65 %)
        sparse_t = np.array([0.0, 5.0, 10.0, 15.0, 160.0, 180.0, 210.0, 220.0])

        lc = _make_multiband_lightcurve(
            {"g": injected_period, "r": injected_period, "i": injected_period},
            time_by_band={"g": dense_t, "r": dense_t, "i": sparse_t},
            noise_std=0.01,
            seed=23,
        )
        diagnostics = run_public_consensus_fit(
=======

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
>>>>>>> b7ea677 (Add consensus scientific regression test module)
            lc,
            use_acf=False,
            min_points_per_band=20,
            max_gap_fraction=0.25,
<<<<<<< HEAD
=======
            min_duty_cycle=0.10,
>>>>>>> b7ea677 (Add consensus scientific regression test module)
        )

        self.assertTrue(diagnostics["consensus_success"])
        self.assertIn("i", diagnostics["rejected_bands"])
<<<<<<< HEAD
        self.assertNotIn("i", diagnostics["accepted_bands"])

        # Band 'i' must have at least one quality-related rejection reason.
        band_i_rec = diagnostics.get("per_band_diagnostics", {}).get("i", {})
        self.assertGreater(
            len(band_i_rec.get("rejection_reasons", [])),
            0,
            "band 'i' must have at least one rejection reason",
        )

        # Consensus must still recover the injected frequency.
        assert_frequency_close(
            self, diagnostics["final_consensus_frequency"], injected_freq, tol=0.005
        )
        assert_valid_consensus_diagnostics(self, lc, diagnostics)

    # ------------------------------------------------------------------
    # Test 4: harmonic / alias challenge
    # ------------------------------------------------------------------

    def test_harmonic_band_is_detected_as_frequency_outlier(self):
        """Sub-harmonic band is detected as outlier; majority consensus is preserved.

        Design note
        -----------
        As in the discrepant-band test, three majority bands use slightly
        different periods (26, 30, 34 days) to produce ≥ 3 deduplicated
        frequency candidates and enable robust outlier detection.
        Band 'z' is injected at 15 days — the 2:1 sub-harmonic of the 30-day
        majority period — giving a LS peak at ≈ 0.0667 day⁻¹.  The ACF is
        enabled so that harmonic relationships are captured in diagnostics.

        Majority periods  : 26, 30, 34 days
        Sub-harmonic period : 15 days  (2× the 30-day majority; f ≈ 0.0667)
        Majority median f : ≈ 1/30 = 0.0333 day⁻¹
        Tolerance         : 0.007 day⁻¹

        Expected behaviour
        ------------------
        * ``consensus_success`` is True.
        * Band 'z' appears in ``rejected_bands`` or ``consensus_outlier_bands``.
        * Final consensus frequency within 0.007 day⁻¹ of 1/30 day⁻¹.
        * Finalized diagnostics pass schema validation.
        """
        majority_median_freq = 1.0 / 30.0

        lc = _make_multiband_lightcurve(
            {"g": 26.0, "r": 30.0, "i": 34.0, "z": 15.0},
            noise_std=0.01,
            seed=29,
        )
        diagnostics = run_public_consensus_fit(lc, use_acf=True, outlier_sigma=2.5)

        self.assertTrue(diagnostics["consensus_success"])
        # Sub-harmonic band must not be silently treated as an inlier.
        assert_band_excluded_or_outlier(self, diagnostics, "z")
        # Consensus frequency must remain close to the true majority period.
        assert_frequency_close(
            self,
            diagnostics["final_consensus_frequency"],
            majority_median_freq,
            tol=0.007,
        )
        assert_valid_consensus_diagnostics(self, lc, diagnostics)

    # ------------------------------------------------------------------
    # Test 5: no-consensus case via total quality-gating failure
    # ------------------------------------------------------------------

    def test_no_consensus_all_bands_fail_quality_gating(self):
        """All bands fail the sampling quality gate; consensus is impossible.

        Algorithm note
        --------------
        ``consensus_success=False`` is produced here because every band is
        rejected before LS frequency extraction (too few points).

        Injected periods : 7, 30, 130 days  (mutually inconsistent)
        Sample counts    : 6, 7, 5  (all < min_points_per_band=20)
        Quality threshold: min_points_per_band=20

        Expected behaviour
        ------------------
        * ``lc.fit(...)`` raises :class:`ConsensusFitError` (a subclass of
          ``RuntimeError``) with ``reason="no_accepted_bands"``.
        * After the exception, ``lc.consensus_diagnostics["consensus_success"]``
          is False.
        * ``final_consensus_frequency`` is None.
        * ``accepted_bands`` is empty; ``rejected_bands`` is populated.
        * Top-level ``rejection_reasons`` is non-empty.
        * Finalized failure diagnostics pass schema validation.
        """
        lc = _make_multiband_lightcurve(
            {"g": 7.0, "r": 30.0, "i": 130.0},
            n_pts_by_band={"g": 6, "r": 7, "i": 5},
=======
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
>>>>>>> b7ea677 (Add consensus scientific regression test module)
            noise_std=0.01,
            seed=31,
        )

<<<<<<< HEAD
        # lc.fit raises ConsensusFitError (subclass of RuntimeError) when all
        # bands fail quality gating.  lc.consensus_diagnostics is set before
        # the error is raised, so it is accessible after assertRaises.
        with self.assertRaises(ConsensusFitError) as cm:
            run_public_consensus_fit(lc, min_points_per_band=20)

        exc = cm.exception
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(exc.failure_diagnostics.get("reason"), "no_accepted_bands")

        diagnostics = lc.consensus_diagnostics
        self.assertFalse(diagnostics["consensus_success"])
        self.assertIsNone(diagnostics["final_consensus_frequency"])
        self.assertEqual(diagnostics["accepted_bands"], [])
        self.assertGreater(len(diagnostics["rejected_bands"]), 0)
        self.assertGreater(
            len(diagnostics.get("rejection_reasons", {})),
            0,
            "rejection_reasons must be non-empty when all bands are rejected",
        )
        assert_valid_consensus_diagnostics(self, lc, diagnostics)

    # ------------------------------------------------------------------
    # Test 6: no-consensus via genuinely inconsistent periods
    # ------------------------------------------------------------------

    def test_no_consensus_inconsistent_periods(self):
        """Adequately sampled but mutually inconsistent periods → no consensus.

        Algorithm note
        --------------
        Four bands are each given enough points to pass pre-LS quality gating
        (60 pts over 220 days), but their injected periods are mutually
        inconsistent (18, 31, 47, 73 days).  With ``outlier_sigma=1.5`` the
        sigma-clipping removes the highest-frequency outlier (band 'g',
        period 18 days), leaving 3 inliers.  Setting
        ``min_consensus_inliers=4`` requires all 4 bands to cluster — a
        condition that cannot be met here — so the consensus pipeline returns
        ``consensus_success=False`` via the insufficient-inliers guard, rather
        than silently reporting the median of inconsistent frequencies as a
        valid result.

        Theoretical verification
        ~~~~~~~~~~~~~~~~~~~~~~~~
        Frequencies: 1/73≈0.0137, 1/47≈0.0213, 1/31≈0.0323, 1/18≈0.0556
        Median ≈ 0.0268; robust σ ≈ 0.0138.
        Outlier threshold (σ=1.5) ≈ 0.0206.
        Only 1/18 (deviation ≈ 0.0288 > 0.0206) is clipped → 3 inliers.
        3 inliers < min_consensus_inliers=4 → failure.

        Injected periods  : 18, 31, 47, 73 days
        Sample count      : 60 per band
        outlier_sigma     : 1.5
        min_consensus_inliers : 4

        Expected behaviour
        ------------------
<<<<<<< HEAD
        * ``lc.fit(...)`` raises :class:`ConsensusFitError` (a subclass of
          ``RuntimeError``) with ``reason="insufficient_consensus_inliers"``.
        * The exception's ``failure_diagnostics`` contains structured data:
          ``n_inlier_bands``, ``required_inliers``, ``n_candidate_bands``,
          ``candidate_periods``.
=======
        * ``lc.fit(...)`` raises ``RuntimeError``.
>>>>>>> 4ef607d (fix: add min_consensus_inliers guard for mutually inconsistent periods)
        * ``lc.consensus_diagnostics["consensus_success"]`` is ``False``.
        * ``final_consensus_frequency`` is ``None``.
        * ``final_consensus_period`` is ``None``.
        * ``consensus_outlier_bands`` is non-empty (the sigma-clipped band).
        * ``accepted_bands`` is non-empty (all 4 bands passed quality gating).
        * ``trusted_candidate_count`` < ``min_consensus_inliers``.
        * Finalized failure diagnostics pass schema validation.
        """
        lc = _make_multiband_lightcurve(
            {"g": 18.0, "r": 31.0, "i": 47.0, "z": 73.0},
            noise_std=0.01,
            seed=77,
        )

<<<<<<< HEAD
        # lc.fit raises ConsensusFitError (subclass of RuntimeError) via the
        # insufficient-inliers guard.  lc.consensus_diagnostics is set before
        # the error, so it is accessible after the assertRaises block.
        with self.assertRaises(ConsensusFitError) as cm:
=======
        # lc.fit raises RuntimeError via the insufficient-inliers guard.
        # lc.consensus_diagnostics is set before the error, so it is
        # accessible after the assertRaises block.
        with self.assertRaises(RuntimeError):
>>>>>>> 4ef607d (fix: add min_consensus_inliers guard for mutually inconsistent periods)
            run_public_consensus_fit(
                lc,
                outlier_sigma=1.5,
                min_consensus_inliers=4,
            )

<<<<<<< HEAD
        exc = cm.exception
        self.assertIsInstance(exc, RuntimeError)
        fd = exc.failure_diagnostics
        self.assertEqual(fd.get("status"), "failed")
        self.assertEqual(fd.get("reason"), "insufficient_consensus_inliers")
        self.assertIn("n_inlier_bands", fd)
        self.assertIn("required_inliers", fd)
        self.assertIn("n_candidate_bands", fd)
        self.assertIn("candidate_periods", fd)
        self.assertIsInstance(fd["candidate_periods"], list)

=======
>>>>>>> 4ef607d (fix: add min_consensus_inliers guard for mutually inconsistent periods)
        diagnostics = lc.consensus_diagnostics

        # Core invariants
        self.assertFalse(diagnostics["consensus_success"])
        self.assertIsNone(diagnostics["final_consensus_frequency"])
        self.assertIsNone(diagnostics["final_consensus_period"])

        # Bands passed quality gating but not the inlier-cluster requirement.
        self.assertGreater(
            len(diagnostics["accepted_bands"]),
            0,
            "accepted_bands must be non-empty: bands passed quality gating",
        )

        # Outlier rejection must have run (consensus_outlier_bands populated).
        self.assertGreater(
            len(diagnostics.get("consensus_outlier_bands", [])),
            0,
            "consensus_outlier_bands must be non-empty: outlier rejection ran",
        )

        # trusted_candidate_count must be below the required threshold.
        trusted = diagnostics.get("trusted_candidate_count")
        self.assertIsNotNone(
            trusted,
            "trusted_candidate_count must be set in failure diagnostics",
        )
        self.assertLess(
            trusted,
            4,
            "trusted_candidate_count must be < min_consensus_inliers (4)",
        )

        assert_valid_consensus_diagnostics(self, lc, diagnostics)
=======
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
>>>>>>> b7ea677 (Add consensus scientific regression test module)


if __name__ == "__main__":
    unittest.main()
