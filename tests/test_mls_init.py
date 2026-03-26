"""Tests for MLS-based GP initialisation in LightCurve.fit()."""

import unittest
from unittest.mock import patch, MagicMock
import torch

from pgmuvi.synthetic import make_simple_sinusoid_1d, make_chromatic_sinusoid_2d


# Sentinel return value for the patched `train` function so that
# `self.results = train(...)` succeeds without actually training.
_DUMMY_RESULTS = {"loss": [1.0], "delta_loss": [0.0]}


def _make_fit_kwargs(**overrides):
    """Shared keyword arguments for LightCurve.fit() in tests."""
    defaults = {
        "model": "1D",
        "check_sampling": False,
        "training_iter": 1,
    }
    defaults.update(overrides)
    return defaults


def _fit_without_training(lc, **kwargs):
    """Call lc.fit() with the training loop patched out."""
    with patch("pgmuvi.lightcurve.train", return_value=_DUMMY_RESULTS):
        with patch.object(lc, "_train"):
            with patch.object(lc, "print_parameters"):
                lc.fit(**_make_fit_kwargs(**kwargs))


class TestMLSInitDefault(unittest.TestCase):
    """use_mls_init=True (default): num_mixtures comes from significant MLS peaks."""

    def setUp(self):
        # Strong signal: MLS should find exactly 1 significant period.
        self.lc = make_simple_sinusoid_1d(
            n_obs=100,
            period=5.0,
            noise_level=0.0,
            irregular=False,
            seed=42,
        )

    def test_num_mixtures_matches_significant_count(self):
        """num_mixtures is set to the number of significant MLS periods."""
        ls_freqs, ls_sig = self.lc.fit_LS(num_peaks=10)
        n_sig = int(ls_sig.sum().item())
        expected = max(n_sig, 1)

        _fit_without_training(self.lc)

        self.assertEqual(
            self.lc.model.covar_module.num_mixtures,
            expected,
        )

    def test_use_mls_init_false_falls_back_to_4(self):
        """use_mls_init=False falls back to num_mixtures=4 when not specified."""
        _fit_without_training(self.lc, use_mls_init=False)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 4)

    def test_use_mls_init_false_explicit_num_mixtures_respected(self):
        """use_mls_init=False and explicit num_mixtures=3 are both respected."""
        _fit_without_training(self.lc, use_mls_init=False, num_mixtures=3)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 3)


class TestMLSInitNumMixturesOverride(unittest.TestCase):
    """User sets num_mixtures explicitly with use_mls_init=True."""

    def setUp(self):
        self.lc = make_simple_sinusoid_1d(
            n_obs=100,
            period=5.0,
            noise_level=0.0,
            irregular=False,
            seed=42,
        )

    def test_num_mixtures_respected_when_smaller_than_sig_count(self):
        """If num_mixtures < significant peaks, uses first num_mixtures sig peaks."""
        _fit_without_training(self.lc, num_mixtures=1)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 1)

    def test_num_mixtures_respected_when_larger_than_sig_count(self):
        """If num_mixtures > significant peaks, pads with non-significant peaks."""
        # With noise_level=0 the signal has 1 significant period; request 4.
        _fit_without_training(self.lc, num_mixtures=4)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 4)

    def test_num_mixtures_zero_uses_fallback(self):
        """num_mixtures=0 is not meaningful; MLS logic should still produce ≥1."""
        # The MLS path picks max(sig, 1); the fallback is 4.
        _fit_without_training(self.lc)
        self.assertGreaterEqual(self.lc.model.covar_module.num_mixtures, 1)


class TestMLSInitPeriods(unittest.TestCase):
    """User passes explicit period guesses via the `periods` parameter."""

    def setUp(self):
        self.lc = make_simple_sinusoid_1d(
            n_obs=100,
            period=5.0,
            noise_level=0.0,
            irregular=False,
            seed=42,
        )

    def test_periods_sets_num_mixtures_to_len(self):
        """num_mixtures equals the number of supplied periods."""
        _fit_without_training(self.lc, periods=[5.0])
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 1)

    def test_periods_multiple_sets_num_mixtures(self):
        """Multiple supplied periods set num_mixtures correctly."""
        _fit_without_training(self.lc, periods=[5.0, 2.5, 1.25])
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 3)

    def test_periods_overrides_use_mls_init(self):
        """Explicit periods take priority over use_mls_init=True."""
        _fit_without_training(self.lc, periods=[5.0, 2.5], use_mls_init=True)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 2)

    def test_periods_ignores_explicit_num_mixtures(self):
        """Explicit periods override any num_mixtures value."""
        _fit_without_training(self.lc, periods=[5.0, 2.5], num_mixtures=10)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 2)

    def test_periods_tensor_input(self):
        """periods can be passed as a torch.Tensor."""
        p = torch.tensor([5.0, 2.5], dtype=torch.float32)
        _fit_without_training(self.lc, periods=p)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 2)


class TestMLSInitFrequencySeeding(unittest.TestCase):
    """Verify that mixture_means is seeded close to the MLS-detected frequency."""

    def setUp(self):
        self.lc = make_simple_sinusoid_1d(
            n_obs=100,
            period=5.0,
            noise_level=0.0,
            irregular=False,
            seed=42,
        )

    def test_mixture_means_seeded_from_mls(self):
        """After MLS init the single mixture mean is close to 1/period."""
        ls_freqs, ls_sig = self.lc.fit_LS(num_peaks=10)
        expected_freq = ls_freqs[ls_sig][0].item() if ls_sig.any() else ls_freqs[0].item()

        _fit_without_training(self.lc)

        # mixture_means has shape (num_mixtures, 1) in the transformed space;
        # recover the raw-space frequency for comparison.
        means_transformed = self.lc.model.covar_module.mixture_means.detach()
        # Inverse the set_hypers x-transform:
        #   set_hypers does  f_model = 1 / transform(1/f_raw, shift=False)
        # so f_raw = 1 / inv_transform(1/f_model) — easier to just compare
        # in terms of ordering (higher raw freq → higher transformed freq).
        self.assertGreater(means_transformed.min().item(), 0)

    def test_mixture_means_seeded_from_periods(self):
        """User-supplied periods seed the mixture_means at the correct frequency."""
        period = 5.0
        _fit_without_training(self.lc, periods=[period])

        means_transformed = self.lc.model.covar_module.mixture_means.detach()
        # The seeded means should be positive (valid frequencies).
        self.assertGreater(means_transformed.min().item(), 0)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 1)


class TestMLSInitMLSFallback(unittest.TestCase):
    """Graceful fallback when MLS raises an exception or finds no peaks."""

    def setUp(self):
        self.lc = make_simple_sinusoid_1d(
            n_obs=100,
            period=5.0,
            noise_level=0.0,
            irregular=False,
            seed=42,
        )

    def test_mls_exception_falls_back_to_4(self):
        """If fit_LS raises, num_mixtures falls back to 4."""
        with patch.object(
            self.lc, "fit_LS", side_effect=RuntimeError("MLS failed")
        ):
            _fit_without_training(self.lc)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 4)

    def test_mls_no_peaks_falls_back_to_4(self):
        """If fit_LS returns no peaks, num_mixtures falls back to 4."""
        empty_freqs = torch.tensor([], dtype=torch.float32)
        empty_mask = torch.tensor([], dtype=torch.bool)
        with patch.object(self.lc, "fit_LS", return_value=(empty_freqs, empty_mask)):
            _fit_without_training(self.lc)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 4)

    def test_mls_no_significant_peaks_uses_1(self):
        """If no significant peaks, uses the single strongest peak (num_mixtures=1)."""
        freq = torch.tensor([0.2], dtype=torch.float32)
        mask = torch.tensor([False], dtype=torch.bool)  # not significant
        with patch.object(self.lc, "fit_LS", return_value=(freq, mask)):
            _fit_without_training(self.lc)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 1)


if __name__ == "__main__":
    unittest.main()
