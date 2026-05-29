"""Unit tests for _consensus_collect_band_component_candidates.

This module tests the per-band multi-component LS candidate extraction
helper introduced for the future ``fit_strategy='consensus_multicomp'``
path.

Scope
-----
* Multi-component extraction returns multiple candidates per band when
  the signal genuinely contains multiple periods.
* Every candidate dict is populated with the required metadata keys:
  ``frequency``, ``period``, ``ls_rank``, ``peak_power``,
  ``peak_prominence``, ``significant``.
* LS rank ordering is preserved — candidates appear in descending
  peak-power order (``ls_rank=0`` first).
* The existing ``fit_strategy='consensus'`` public path is NOT affected
  by this new helper; single-component consensus behaviour remains
  identical.

Out of scope (deferred to future PRs)
--------------------------------------
* Cross-band component matching.
* Multi-component consensus aggregation.
* Component IDs or canonical frequency ordering.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_multiband_lc(
    period_by_band,
    *,
    n_pts=60,
    noise_std=0.02,
    seed=0,
    time_span=220.0,
):
    """Build a synthetic multiband Lightcurve.

    Parameters
    ----------
    period_by_band : mapping
        ``{band_label: period_days}`` — one injected sinusoidal period per
        band.
    n_pts : int, optional
        Number of observations per band (default 60).
    noise_std : float, optional
        Gaussian noise standard deviation (default 0.02).
    seed : int, optional
        PRNG seed for reproducibility (default 0).
    time_span : float, optional
        Total time baseline in days (default 220.0).
    """
    rng = np.random.default_rng(seed)
    x_blocks, y_blocks, yerr_blocks, band_blocks = [], [], [], []

    for band_idx, (band, period) in enumerate(period_by_band.items()):
        t = np.linspace(0.0, time_span, n_pts)
        signal = np.sin(2.0 * math.pi * t / float(period))
        noise = rng.normal(0.0, noise_std, t.shape)
        y = signal + noise
        yerr = np.full_like(t, noise_std)
        x = np.column_stack([t, np.full_like(t, float(band_idx + 1))])
        x_blocks.append(x)
        y_blocks.append(y)
        yerr_blocks.append(yerr)
        # Use np.array([...]*n) rather than np.full(..., dtype=np.str_) to
        # avoid NumPy's <U1 truncation when band labels are > 1 character.
        band_blocks.append(np.array([str(band)] * n_pts))

    return Lightcurve(
        np.concatenate(x_blocks, axis=0),
        np.concatenate(y_blocks),
        yerr=np.concatenate(yerr_blocks),
        band=np.concatenate(band_blocks),
    )


def _make_multiperiod_band_lc(
    periods,
    *,
    n_pts=60,
    noise_std=0.01,
    seed=42,
    time_span=220.0,
):
    """Build a two-band Lightcurve where one band carries multiple periods.

    Constructs a two-band light curve where ``band0`` carries the signal
    composed of all entries in ``periods`` so that its LS periodogram
    contains multiple significant peaks, while ``band1`` carries only the
    first period as a control.

    Parameters
    ----------
    periods : sequence of float
        Injected periods (days).  At least two values are expected.
        Both must be less than ``time_span / 2`` (detectable) and greater
        than ``2 * time_span / n_pts`` (above Nyquist).
    n_pts : int, optional
        Observations per band (default 60).
    noise_std : float, optional
        Noise standard deviation (default 0.01).
    seed : int, optional
        PRNG seed (default 42).
    time_span : float, optional
        Time baseline in days (default 220.0).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, time_span, n_pts)

    # band0: sum of all injected sinusoids (multi-component signal)
    signal = sum(np.sin(2.0 * math.pi * t / float(p)) for p in periods)
    y0 = signal + rng.normal(0.0, noise_std, t.shape)

    # band1: single dominant period (single-component control)
    y1 = np.sin(2.0 * math.pi * t / float(periods[0])) + rng.normal(
        0.0, noise_std, t.shape
    )

    x0 = np.column_stack([t, np.ones_like(t)])
    x1 = np.column_stack([t, np.full_like(t, 2.0)])

    # Use np.array([...]*n) to avoid <U1 truncation from dtype=np.str_.
    return Lightcurve(
        np.concatenate([x0, x1], axis=0),
        np.concatenate([y0, y1]),
        yerr=np.concatenate(
            [np.full_like(t, noise_std), np.full_like(t, noise_std)]
        ),
        band=np.concatenate(
            [
                np.array(["band0"] * n_pts),
                np.array(["band1"] * n_pts),
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBandComponentCandidatesStructure(unittest.TestCase):
    """Tests for the structural / metadata requirements of the new helper."""

    def setUp(self):
        # Simple multiband lightcurve: two bands, one period each.
        self.lc = _make_multiband_lc(
            {"J": 30.0, "K": 45.0},
            n_pts=80,
            time_span=400.0,
        )

    def test_returns_list(self):
        """Return value must be a list."""
        result = self.lc._consensus_collect_band_component_candidates()
        self.assertIsInstance(result, list)

    def test_returns_one_entry_per_accepted_band(self):
        """One dict per accepted (quality-passing) band."""
        result = self.lc._consensus_collect_band_component_candidates()
        # Both bands have adequate sampling so both should be accepted.
        self.assertEqual(len(result), 2)

    def test_entry_keys(self):
        """Each band entry must contain band_name, wavelength, component_candidates."""
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            with self.subTest(band=entry.get("band_name")):
                self.assertIn("band_name", entry)
                self.assertIn("wavelength", entry)
                self.assertIn("component_candidates", entry)

    def test_band_name_is_string(self):
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            self.assertIsInstance(entry["band_name"], str)

    def test_wavelength_is_float_or_none(self):
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            wl = entry["wavelength"]
            self.assertTrue(
                wl is None or isinstance(wl, float),
                f"wavelength should be float or None, got {type(wl)}",
            )

    def test_component_candidates_is_list(self):
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            self.assertIsInstance(entry["component_candidates"], list)

    def test_candidate_metadata_keys(self):
        """Every candidate must contain the six required metadata keys."""
        required_keys = {
            "frequency",
            "period",
            "ls_rank",
            "peak_power",
            "peak_prominence",
            "significant",
        }
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            for cand in entry["component_candidates"]:
                with self.subTest(band=entry["band_name"], rank=cand.get("ls_rank")):
                    self.assertTrue(
                        required_keys.issubset(cand.keys()),
                        f"Missing keys: {required_keys - cand.keys()}",
                    )

    def test_frequency_positive_finite(self):
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            for cand in entry["component_candidates"]:
                with self.subTest(band=entry["band_name"], rank=cand["ls_rank"]):
                    self.assertGreater(cand["frequency"], 0.0)
                    self.assertTrue(
                        math.isfinite(cand["frequency"]),
                        "frequency must be finite",
                    )

    def test_period_equals_inverse_frequency(self):
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            for cand in entry["component_candidates"]:
                with self.subTest(band=entry["band_name"], rank=cand["ls_rank"]):
                    self.assertAlmostEqual(
                        cand["period"],
                        1.0 / cand["frequency"],
                        places=10,
                    )

    def test_peak_power_non_negative(self):
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            for cand in entry["component_candidates"]:
                with self.subTest(band=entry["band_name"], rank=cand["ls_rank"]):
                    self.assertGreaterEqual(cand["peak_power"], 0.0)

    def test_significant_is_bool(self):
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            for cand in entry["component_candidates"]:
                with self.subTest(band=entry["band_name"], rank=cand["ls_rank"]):
                    self.assertIsInstance(cand["significant"], bool)

    def test_ls_rank_is_int(self):
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            for cand in entry["component_candidates"]:
                with self.subTest(band=entry["band_name"], rank=cand["ls_rank"]):
                    self.assertIsInstance(cand["ls_rank"], int)


class TestBandComponentCandidatesRankOrdering(unittest.TestCase):
    """Tests that LS rank ordering is correctly preserved.

    ``ls_rank`` stores the raw LS rank of each candidate in the original
    periodogram output (0 = highest power peak *before* any plausibility
    filtering).  Plausibility filtering may skip intermediate ranks (e.g.
    when several alias peaks dominate at above-Nyquist frequencies), so
    ``ls_rank`` values in the output list:

    * are non-negative integers,
    * are strictly increasing across the list (earlier entry → lower rank
      → higher raw LS power),
    * may start above 0 (if the first plausible peak is not rank 0),
    * may have gaps (skipped ranks correspond to implausible alias peaks).
    """

    def setUp(self):
        self.lc = _make_multiband_lc(
            {"J": 30.0, "K": 45.0},
        )

    def test_ls_rank_non_negative(self):
        """Every ls_rank must be a non-negative integer."""
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            for cand in entry["component_candidates"]:
                with self.subTest(band=entry["band_name"]):
                    self.assertGreaterEqual(cand["ls_rank"], 0)

    def test_ls_rank_strictly_increasing(self):
        """ls_rank values must be strictly increasing within each band.

        Candidates are emitted in raw-LS-rank order (lowest rank = highest
        power first).  Gaps are allowed (implausible peaks are skipped).
        """
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            ranks = [c["ls_rank"] for c in entry["component_candidates"]]
            for i in range(len(ranks) - 1):
                with self.subTest(band=entry["band_name"], i=i):
                    self.assertLess(
                        ranks[i],
                        ranks[i + 1],
                        "ls_rank must be strictly increasing (candidates in raw LS order)",
                    )

    def test_peak_power_non_increasing(self):
        """Candidates must be ordered by non-increasing peak_power.

        This validates that the LS extraction order (by descending power)
        is preserved in the output.
        """
        result = self.lc._consensus_collect_band_component_candidates()
        for entry in result:
            powers = [c["peak_power"] for c in entry["component_candidates"]]
            for i in range(len(powers) - 1):
                with self.subTest(band=entry["band_name"], i=i):
                    self.assertGreaterEqual(
                        powers[i],
                        powers[i + 1],
                        "Candidates must be in non-increasing peak_power order",
                    )


class TestBandComponentCandidatesMultipleComponents(unittest.TestCase):
    """Tests that multiple components are extracted when present."""

    def setUp(self):
        # Band 0 has two injected periods → LS should yield ≥ 2 peaks.
        self.lc = _make_multiperiod_band_lc(
            periods=[30.0, 55.0],
            n_pts=120,
            time_span=500.0,
        )

    def test_at_least_one_candidate_per_band(self):
        """Every accepted band yields at least one candidate."""
        result = self.lc._consensus_collect_band_component_candidates(
            max_components_per_band=3
        )
        self.assertGreater(len(result), 0)
        for entry in result:
            self.assertGreater(
                len(entry["component_candidates"]),
                0,
                f"Band {entry['band_name']} has no candidates",
            )

    def test_multi_period_band_returns_multiple_candidates(self):
        """The multi-period band (band0) should return more than one candidate."""
        result = self.lc._consensus_collect_band_component_candidates(
            max_components_per_band=3
        )
        band0_entries = [e for e in result if e["band_name"] == "band0"]
        self.assertTrue(
            len(band0_entries) > 0,
            "band0 should be accepted",
        )
        n_cands = len(band0_entries[0]["component_candidates"])
        self.assertGreater(
            n_cands,
            1,
            f"Expected > 1 candidate for multi-period band0, got {n_cands}",
        )

    def test_max_components_per_band_is_respected(self):
        """Number of candidates must not exceed max_components_per_band."""
        for limit in (1, 2, 3, 5):
            with self.subTest(max_components_per_band=limit):
                result = self.lc._consensus_collect_band_component_candidates(
                    max_components_per_band=limit
                )
                for entry in result:
                    self.assertLessEqual(
                        len(entry["component_candidates"]),
                        limit,
                        f"Band {entry['band_name']} exceeded the component limit",
                    )

    def test_peak_prominence_populated(self):
        """peak_prominence must be a finite float for at least one candidate."""
        result = self.lc._consensus_collect_band_component_candidates(
            max_components_per_band=3
        )
        found_finite_prom = False
        for entry in result:
            for cand in entry["component_candidates"]:
                if math.isfinite(cand["peak_prominence"]):
                    found_finite_prom = True
        self.assertTrue(
            found_finite_prom,
            "Expected at least one finite peak_prominence value",
        )


class TestBandComponentCandidatesDefaultMax(unittest.TestCase):
    """Tests that the default max_components_per_band=3 is used."""

    def test_default_max_is_three(self):
        lc = _make_multiband_lc(
            {"J": 30.0, "K": 45.0},
            n_pts=80,
            time_span=400.0,
        )
        result = lc._consensus_collect_band_component_candidates()
        for entry in result:
            self.assertLessEqual(
                len(entry["component_candidates"]),
                3,
                f"Default max should be 3, got {len(entry['component_candidates'])} "
                f"for band {entry['band_name']}",
            )


class TestBandComponentCandidateMetadataLookup(unittest.TestCase):
    """Focused tests for LS peak metadata lookup safety."""

    def test_off_grid_ls_frequency_sets_metadata_to_nan(self):
        class _MockBand:
            def fit_LS(self, *, num_peaks, return_full=False):
                self.assert_equal_num_peaks = num_peaks
                ls_freqs = torch.tensor([1.75], dtype=torch.float64)
                ls_sig = torch.tensor([True], dtype=torch.bool)
                if return_full:
                    freq_grid = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
                    power_grid = torch.tensor([10.0, 9.0, 8.0], dtype=torch.float64)
                    return ls_freqs, ls_sig, freq_grid, power_grid
                return ls_freqs, ls_sig

        result = Lightcurve._consensus_extract_band_ls_candidates(
            _MockBand(),
            metrics={
                "baseline": 100.0,
                "longest_detectable_period": 100.0,
                "nyquist_frequency": 10.0,
            },
            num_requested_peaks=1,
            max_candidates=1,
            include_peak_metadata=True,
        )

        self.assertEqual(len(result["candidates"]), 1)
        candidate = result["candidates"][0]
        self.assertAlmostEqual(candidate["frequency"], 1.75, places=12)
        self.assertTrue(math.isnan(candidate["peak_power"]))
        self.assertTrue(math.isnan(candidate["peak_prominence"]))


class TestBandWavelengthMapHelper(unittest.TestCase):
    """Focused tests for centralized band-to-wavelength mapping."""

    def test_returns_expected_wavelengths_for_2d_multiband(self):
        lc = _make_multiband_lc(
            {"J": 30.0, "K": 45.0},
            n_pts=20,
            time_span=220.0,
        )
        mapping = lc._consensus_build_band_wavelength_map(["J", "K"])
        self.assertAlmostEqual(mapping["J"], 1.0, places=12)
        self.assertAlmostEqual(mapping["K"], 2.0, places=12)

    def test_returns_none_when_no_wavelength_column_or_no_rows(self):
        lc = _make_multiband_lc(
            {"J": 30.0, "K": 45.0},
            n_pts=20,
            time_span=220.0,
        )
        lc._xdata_raw = lc._xdata_raw[:, :1]
        mapping = lc._consensus_build_band_wavelength_map(["J", "K", "missing"])
        self.assertIsNone(mapping["J"])
        self.assertIsNone(mapping["K"])
        self.assertIsNone(mapping["missing"])


class TestBandComponentCandidatesQualityGating(unittest.TestCase):
    """Tests that poorly-sampled bands are excluded from results."""

    def test_sparse_band_excluded(self):
        """A band with only 2 points must be excluded by the quality gate."""
        rng = np.random.default_rng(0)
        t_good = np.linspace(0.0, 220.0, 60)
        y_good = np.sin(2.0 * math.pi * t_good / 30.0)

        # One good band, one nearly-empty band.
        t_sparse = np.array([0.0, 110.0])
        y_sparse = rng.normal(0.0, 0.1, 2)

        x = np.concatenate(
            [
                np.column_stack([t_good, np.ones_like(t_good)]),
                np.column_stack([t_sparse, np.full_like(t_sparse, 2.0)]),
            ],
            axis=0,
        )
        y = np.concatenate([y_good, y_sparse])
        # Use np.array([...]*n) to avoid <U1 truncation from dtype=np.str_.
        band = np.concatenate(
            [
                np.array(["good"] * len(t_good)),
                np.array(["sparse"] * len(t_sparse)),
            ]
        )

        lc = Lightcurve(x, y, band=band)
        result = lc._consensus_collect_band_component_candidates()
        band_names = [e["band_name"] for e in result]
        self.assertNotIn("sparse", band_names, "Sparse band should be quality-gated out")
        self.assertIn("good", band_names, "Good band should be accepted")


class TestExistingConsensusPathUnchanged(unittest.TestCase):
    """Verify that the existing single-component consensus path is unaffected.

    The new ``_consensus_collect_band_component_candidates`` helper must NOT
    be called by ``fit_strategy='consensus'`` and must NOT modify the
    behaviour of ``_consensus_collect_band_candidates``.
    """

    def setUp(self):
        self.lc = _make_multiband_lc(
            {"J": 30.0, "K": 30.0},
            n_pts=80,
            time_span=400.0,
        )

    def test_collect_band_candidates_still_returns_one_dominant_per_band(self):
        """_consensus_collect_band_candidates must yield one dominant freq per band."""
        result = self.lc._consensus_collect_band_candidates()
        band_records = result["band_records"]
        accepted = result["accepted_bands"]
        self.assertGreater(len(accepted), 0)
        for band in accepted:
            record = band_records[band]
            # Each accepted band record must have exactly one dominant_frequency.
            self.assertIsNotNone(
                record["dominant_frequency"],
                f"Band {band} should have a dominant_frequency",
            )
            # The old helper must NOT have a component_candidates key.
            self.assertNotIn(
                "component_candidates",
                record,
                "Old helper must not populate component_candidates",
            )

    def test_single_path_matches_multicomp_primary_selection(self):
        """Single-candidate path must select the same primary LS peak."""
        single_result = self.lc._consensus_collect_band_candidates()
        multicomp_result = {
            entry["band_name"]: entry
            for entry in self.lc._consensus_collect_band_component_candidates(
                max_components_per_band=10
            )
        }

        for band in single_result["accepted_bands"]:
            with self.subTest(band=band):
                record = single_result["band_records"][band]
                candidates = multicomp_result[band]["component_candidates"]
                chosen = next(
                    (candidate for candidate in candidates if candidate["significant"]),
                    candidates[0],
                )
                self.assertAlmostEqual(
                    record["dominant_frequency"],
                    chosen["frequency"],
                    places=10,
                )
                self.assertAlmostEqual(
                    record["dominant_period"],
                    chosen["period"],
                    places=10,
                )
                self.assertEqual(record["ls_significant"], chosen["significant"])

    def test_public_consensus_fit_succeeds(self):
        """Full public consensus fit must complete without error."""
        self.lc.fit(
            fit_strategy="consensus",
            model="2D",
            training_iter=0,
            use_gp_validation=False,
            constrain_consensus=False,
        )
        diag = self.lc.consensus_diagnostics
        self.assertTrue(diag.get("consensus_success", False))

    def test_public_consensus_diagnostics_unchanged(self):
        """Consensus diagnostics must still contain per_band_dominant_periods."""
        self.lc.fit(
            fit_strategy="consensus",
            model="2D",
            training_iter=0,
            use_gp_validation=False,
            constrain_consensus=False,
        )
        diag = self.lc.consensus_diagnostics
        self.assertIn("per_band_dominant_periods", diag)
        self.assertIn("per_band_dominant_frequencies", diag)
        self.assertIn("final_consensus_frequency", diag)

    def test_new_helper_does_not_modify_lc_state(self):
        """Calling the new helper must not mutate consensus_diagnostics."""
        # Establish baseline diagnostics via public fit.
        self.lc.fit(
            fit_strategy="consensus",
            model="2D",
            training_iter=0,
            use_gp_validation=False,
            constrain_consensus=False,
        )
        freq_before = self.lc.consensus_diagnostics.get(
            "final_consensus_frequency"
        )

        # Call the new helper.
        self.lc._consensus_collect_band_component_candidates()

        # Diagnostics must be unmodified.
        self.assertEqual(
            self.lc.consensus_diagnostics.get("final_consensus_frequency"),
            freq_before,
        )


if __name__ == "__main__":
    unittest.main()
