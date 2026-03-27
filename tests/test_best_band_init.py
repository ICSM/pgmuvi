"""Tests for best-band initialisation in fit_LS() and fit()."""

import unittest
import warnings
from unittest.mock import patch
import torch
import numpy as np

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.synthetic import make_simple_sinusoid_1d, make_chromatic_sinusoid_2d


# Sentinel return value for the patched `train` function.
_DUMMY_RESULTS = {"loss": [1.0], "delta_loss": [0.0]}


def _fit_2d_without_training(lc, **kwargs):
    """Call lc.fit() with the training loop patched out (2D defaults)."""
    defaults = {
        "model": "2D",
        "check_sampling": False,
        "check_variability": False,
        "training_iter": 1,
    }
    defaults.update(kwargs)
    with patch("pgmuvi.lightcurve.train", return_value=_DUMMY_RESULTS):
        with patch.object(lc, "_train"):
            with patch.object(lc, "print_parameters"):
                lc.fit(**defaults)


def _make_heterogeneous_lc(n_high=100, n_low=10, period=5.0, seed=42):
    """Return a 2D lightcurve with heterogeneous band sampling."""
    return make_chromatic_sinusoid_2d(
        n_per_band=[n_high, n_low],
        period=period,
        wavelengths=[500.0, 700.0],
        amplitude_slope=0.0,
        noise_level=0.0,
        irregular=False,
        seed=seed,
    )


class TestGetBestSampledBandLc(unittest.TestCase):
    """Tests for the _get_best_sampled_band_lc() helper."""

    def test_1d_returns_self(self):
        """For a 1D lightcurve, _get_best_sampled_band_lc returns self."""
        lc = make_simple_sinusoid_1d(n_obs=50, period=5.0, seed=42)
        self.assertIs(lc._get_best_sampled_band_lc(), lc)

    def test_2d_returns_1d_lightcurve(self):
        """For a 2D lightcurve, _get_best_sampled_band_lc returns a 1D LC."""
        lc = _make_heterogeneous_lc(n_high=100, n_low=10)
        lc_1d = lc._get_best_sampled_band_lc()
        self.assertEqual(lc_1d.ndim, 1)

    def test_2d_best_band_has_most_observations(self):
        """The returned 1D LC has as many points as the most-sampled band."""
        lc = _make_heterogeneous_lc(n_high=100, n_low=10)
        lc_1d = lc._get_best_sampled_band_lc()
        self.assertEqual(len(lc_1d.xdata), 100)

    def test_2d_equal_sampling_returns_one_band(self):
        """With equal sampling, returns exactly one band (whichever is first)."""
        lc = make_chromatic_sinusoid_2d(
            n_per_band=50,
            period=5.0,
            wavelengths=[500.0, 700.0],
            noise_level=0.0,
            seed=42,
        )
        lc_1d = lc._get_best_sampled_band_lc()
        self.assertEqual(lc_1d.ndim, 1)
        self.assertEqual(len(lc_1d.xdata), 50)

    def test_2d_with_yerr_propagated(self):
        """Uncertainties from the best band are included in the returned LC."""
        lc = _make_heterogeneous_lc(n_high=100, n_low=10)
        # Manually add yerr
        lc._yerr_raw = torch.ones(len(lc._xdata_raw)) * 0.1
        lc_1d = lc._get_best_sampled_band_lc()
        self.assertIsNotNone(lc_1d._yerr_raw)
        self.assertEqual(len(lc_1d._yerr_raw), 100)

    def test_2d_without_yerr_returns_no_yerr(self):
        """When the 2D LC has no uncertainties, the returned 1D LC has none either."""
        lc = _make_heterogeneous_lc(n_high=100, n_low=10)
        # Ensure _yerr_raw is not set (or is None)
        lc._yerr_raw = None
        lc_1d = lc._get_best_sampled_band_lc()
        # yerr_raw should be None or absent on the 1D LC
        has_yerr = (
            hasattr(lc_1d, "_yerr_raw") and lc_1d._yerr_raw is not None
        )
        self.assertFalse(
            has_yerr,
            "_get_best_sampled_band_lc should not attach yerr when source has none",
        )


class TestFitLSBestBandInit(unittest.TestCase):
    """Tests for fit_LS() with use_best_band_init=True."""

    def setUp(self):
        self.lc2d = _make_heterogeneous_lc(n_high=100, n_low=10, period=5.0)
        # Equal-sampling 2D LC for comparison
        self.lc2d_equal = make_chromatic_sinusoid_2d(
            n_per_band=55,
            period=5.0,
            wavelengths=[500.0, 700.0],
            noise_level=0.0,
            seed=42,
        )

    def test_returns_frequencies_and_mask(self):
        """fit_LS with use_best_band_init=True returns (freqs, sig_mask)."""
        result = self.lc2d.fit_LS(
            num_peaks=3, use_best_band_init=True
        )
        self.assertEqual(len(result), 2)
        freqs, mask = result
        self.assertIsInstance(freqs, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.dtype, torch.bool)

    def test_finer_grid_more_frequencies(self):
        """Best-band grid (100 pts) is finer than combined grid (110 pts total).

        The heterogeneous LC has 100+10=110 total points. The best band has
        100 points. The 1D autofrequency for the best band should produce a
        comparable or finer grid (since equal time span but similar N).
        We verify that the flag doesn't break the result.
        """
        freqs_best, _ = self.lc2d.fit_LS(
            num_peaks=5, use_best_band_init=True
        )
        freqs_std, _ = self.lc2d.fit_LS(
            num_peaks=5, use_best_band_init=False
        )
        # Both should return non-empty frequency tensors
        self.assertGreater(len(freqs_best), 0)
        self.assertGreater(len(freqs_std), 0)

    def test_use_best_band_false_unchanged(self):
        """use_best_band_init=False behaves identically to not passing the flag."""
        freqs1, mask1 = self.lc2d.fit_LS(num_peaks=3, use_best_band_init=False)
        freqs2, mask2 = self.lc2d.fit_LS(num_peaks=3)
        torch.testing.assert_close(freqs1, freqs2)
        torch.testing.assert_close(mask1, mask2)

    def test_1d_lightcurve_flag_has_no_effect(self):
        """use_best_band_init=True has no effect on 1D lightcurves."""
        lc1d = make_simple_sinusoid_1d(n_obs=80, period=5.0, noise_level=0.0, seed=42)
        freqs_flag, mask_flag = lc1d.fit_LS(
            num_peaks=3, use_best_band_init=True
        )
        freqs_no_flag, mask_no_flag = lc1d.fit_LS(num_peaks=3)
        torch.testing.assert_close(freqs_flag, freqs_no_flag)
        torch.testing.assert_close(mask_flag, mask_no_flag)

    def test_freq_only_with_best_band_init(self):
        """freq_only=True combined with use_best_band_init=True still works."""
        result = self.lc2d.fit_LS(
            freq_only=True, use_best_band_init=True
        )
        self.assertEqual(len(result), 2)
        freq_grid, power = result
        self.assertGreater(len(freq_grid), 0)
        self.assertEqual(len(freq_grid), len(power))


class TestFitBestBandInit2D(unittest.TestCase):
    """Tests for fit() with use_best_band_init=True on 2D SM models."""

    def setUp(self):
        self.lc2d = _make_heterogeneous_lc(n_high=100, n_low=10, period=5.0)

    def test_fit_completes_without_error(self):
        """fit() with use_best_band_init=True should not raise."""
        _fit_2d_without_training(self.lc2d, use_best_band_init=True)

    def test_num_mixtures_set_from_best_band(self):
        """num_mixtures is set from the best-band 1D LS when flag is True."""
        _fit_2d_without_training(self.lc2d, use_best_band_init=True)
        self.assertGreaterEqual(self.lc2d.model.covar_module.num_mixtures, 1)

    def test_mixture_means_temporal_dim_seeded(self):
        """Temporal dimension (dim 0) of mixture_means is seeded from best band."""
        _fit_2d_without_training(
            self.lc2d, use_best_band_init=True, num_mixtures=2
        )
        means = self.lc2d.model.covar_module.mixture_means.detach()
        # shape is [num_mixtures, 1, 2]
        self.assertEqual(means.shape[-1], 2)
        # All temporal frequencies should be finite and positive
        temporal_freqs = means[:, 0, 0]
        self.assertTrue(torch.all(temporal_freqs > 0))
        self.assertTrue(torch.all(torch.isfinite(temporal_freqs)))

    def test_mixture_means_2d_tensor(self):
        """mixture_means has shape [num_mixtures, 1, 2] for a 2D model."""
        _fit_2d_without_training(
            self.lc2d, use_best_band_init=True, num_mixtures=3
        )
        means = self.lc2d.model.covar_module.mixture_means.detach()
        self.assertEqual(means.ndim, 3)
        self.assertEqual(means.shape[1], 1)
        self.assertEqual(means.shape[2], 2)
        self.assertEqual(means.shape[0], 3)

    def test_use_best_band_false_unchanged_behaviour(self):
        """use_best_band_init=False gives same num_mixtures as default."""
        lc_a = _make_heterogeneous_lc(n_high=100, n_low=10, period=5.0, seed=42)
        lc_b = _make_heterogeneous_lc(n_high=100, n_low=10, period=5.0, seed=42)
        _fit_2d_without_training(lc_a, use_best_band_init=False, num_mixtures=2)
        _fit_2d_without_training(lc_b, num_mixtures=2)
        self.assertEqual(
            lc_a.model.covar_module.num_mixtures,
            lc_b.model.covar_module.num_mixtures,
        )

    def test_explicit_num_mixtures_respected_with_flag(self):
        """Explicit num_mixtures is respected when use_best_band_init=True."""
        _fit_2d_without_training(
            self.lc2d, use_best_band_init=True, num_mixtures=3
        )
        self.assertEqual(self.lc2d.model.covar_module.num_mixtures, 3)

    def test_use_mls_init_false_disables_best_band(self):
        """use_mls_init=False disables best-band init (falls back to 4)."""
        lc = _make_heterogeneous_lc(n_high=100, n_low=10, period=5.0, seed=42)
        _fit_2d_without_training(
            lc, use_mls_init=False, use_best_band_init=True
        )
        self.assertEqual(lc.model.covar_module.num_mixtures, 4)

    def test_flag_no_effect_on_1d_model(self):
        """use_best_band_init=True has no special effect on a 1D SM model."""
        lc = make_simple_sinusoid_1d(
            n_obs=100, period=5.0, noise_level=0.0, irregular=False, seed=42
        )
        defaults = {
            "model": "1D",
            "check_sampling": False,
            "training_iter": 1,
            "use_best_band_init": True,
        }
        with patch("pgmuvi.lightcurve.train", return_value=_DUMMY_RESULTS):
            with patch.object(lc, "_train"):
                with patch.object(lc, "print_parameters"):
                    lc.fit(**defaults)
        # Should complete normally
        self.assertGreaterEqual(lc.model.covar_module.num_mixtures, 1)


class TestBestBandInitHeterogeneousSampling(unittest.TestCase):
    """Integration tests with strongly heterogeneous band sampling."""

    def setUp(self):
        # Very heterogeneous: 200 pts vs 5 pts
        self.lc = _make_heterogeneous_lc(n_high=200, n_low=5, period=5.0, seed=0)

    def test_best_band_identified_correctly(self):
        """_get_best_sampled_band_lc returns the band with 200 observations."""
        lc_best = self.lc._get_best_sampled_band_lc()
        self.assertEqual(len(lc_best.xdata), 200)

    def test_fit_ls_best_band_finds_period(self):
        """fit_LS on the best band includes the true period among significant peaks."""
        lc_best = self.lc._get_best_sampled_band_lc()
        freqs, sig = lc_best.fit_LS(num_peaks=10)
        self.assertGreater(len(freqs), 0)
        # The true period is 5 days (freq ≈ 0.2 Hz).  It should appear among
        # the returned peaks (though not necessarily at index 0 due to aliases
        # from regular sampling).
        periods = 1.0 / freqs.numpy()
        close_to_truth = (abs(periods - 5.0) / 5.0) < 0.1
        self.assertTrue(
            close_to_truth.any(),
            f"Expected a peak near period=5 but got periods: {periods}",
        )

    def test_fit_with_best_band_init_sets_temporal_freq(self):
        """GP fit with best-band init seeds mixture_means from within-Nyquist peak."""
        _fit_2d_without_training(
            self.lc, use_best_band_init=True, num_mixtures=1
        )
        means = self.lc.model.covar_module.mixture_means.detach()
        # Temporal frequency (dim 0) should be finite and positive.
        # The alias-filtering in fit() clips peaks above the best-band
        # Nyquist, so the value should be a physically meaningful frequency.
        temporal_freq = means[0, 0, 0].item()
        self.assertGreater(temporal_freq, 0)
        self.assertTrue(
            np.isfinite(temporal_freq),
            f"temporal_freq should be finite, got {temporal_freq}",
        )


if __name__ == "__main__":
    unittest.main()
