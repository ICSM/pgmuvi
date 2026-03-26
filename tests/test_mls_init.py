"""Tests for MLS-based GP initialisation in LightCurve.fit()."""

import unittest
import warnings
from unittest.mock import patch
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

    def test_num_mixtures_zero_raises(self):
        """num_mixtures=0 is invalid and raises ValueError."""
        with self.assertRaises(ValueError):
            _fit_without_training(self.lc, num_mixtures=0)

    def test_num_mixtures_negative_raises(self):
        """num_mixtures < 0 is invalid and raises ValueError."""
        with self.assertRaises(ValueError):
            _fit_without_training(self.lc, num_mixtures=-1)

    def test_padding_warns_when_too_few_peaks(self):
        """A RuntimeWarning is emitted when MLS peaks must be padded."""
        # Mock fit_LS to return only 2 peaks, but request 5 mixtures
        # so padding is definitely needed (5 > 2).
        two_freqs = torch.tensor([0.2, 0.4], dtype=torch.float32)
        two_sig = torch.tensor([True, False], dtype=torch.bool)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch.object(self.lc, "fit_LS", return_value=(two_freqs, two_sig)):
                _fit_without_training(self.lc, num_mixtures=5)
        # Check that a padding warning was raised.
        self.assertTrue(
            any(
                issubclass(w.category, RuntimeWarning)
                and "Padding" in str(w.message)
                for w in caught
            ),
            "Expected a padding RuntimeWarning when too few MLS peaks were found.",
        )

    def test_padding_preserves_num_mixtures(self):
        """Padding keeps num_mixtures at the user-requested value."""
        _fit_without_training(self.lc, num_mixtures=10)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 10)


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

    def test_periods_empty_raises(self):
        """Empty periods sequence raises ValueError."""
        with self.assertRaises(ValueError):
            _fit_without_training(self.lc, periods=[])

    def test_periods_zero_raises(self):
        """A period of zero raises ValueError."""
        with self.assertRaises(ValueError):
            _fit_without_training(self.lc, periods=[0.0])

    def test_periods_negative_raises(self):
        """A negative period raises ValueError."""
        with self.assertRaises(ValueError):
            _fit_without_training(self.lc, periods=[-5.0])


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
        """After MLS init the mixture means are seeded from the MLS-detected frequencies."""
        # Use a small noise level so only the fundamental is significant.
        lc = make_simple_sinusoid_1d(
            n_obs=100,
            period=5.0,
            noise_level=0.1,
            noise_type="gaussian",
            irregular=False,
            seed=42,
        )
        ls_freqs, ls_sig = lc.fit_LS(num_peaks=10)
        n_sig = int(ls_sig.sum().item())
        expected_n_mixtures = max(n_sig, 1)

        with patch("pgmuvi.lightcurve.train", return_value=_DUMMY_RESULTS):
            with patch.object(lc, "_train"):
                with patch.object(lc, "print_parameters"):
                    lc.fit(**_make_fit_kwargs())

        means = lc.model.covar_module.mixture_means.detach()
        # Seeded mixture means must be positive (valid frequencies).
        self.assertGreater(means.min().item(), 0)
        # Number of components matches the MLS count.
        self.assertEqual(lc.model.covar_module.num_mixtures, expected_n_mixtures)
        # The seeded mixture means in transformed space should all be
        # monotonically related to the raw MLS frequencies.  Verify that
        # the mixture means are ordered the same way as the seeded frequencies.
        if n_sig > 1 and lc.xtransform is not None:
            sig_sorted = ls_freqs[ls_sig].sort().values
            means_sorted = means.squeeze(-1).sort().values
            # Higher raw freq → higher transformed freq (monotone transform).
            self.assertTrue(
                (means_sorted[1:] > means_sorted[:-1]).all(),
                "Mixture means should be strictly ordered.",
            )

    def test_mixture_means_seeded_from_periods(self):
        """User-supplied periods seed the mixture_means at the correct frequency."""
        period = 5.0
        _fit_without_training(self.lc, periods=[period])

        means = self.lc.model.covar_module.mixture_means.detach()
        # The seeded means should be positive (valid frequencies).
        self.assertGreater(means.min().item(), 0)
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
        """If fit_LS raises, num_mixtures falls back to 4 with a warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch.object(
                self.lc, "fit_LS", side_effect=RuntimeError("MLS failed")
            ):
                _fit_without_training(self.lc)
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 4)
        # A RuntimeWarning should have been emitted.
        self.assertTrue(
            any(issubclass(w.category, RuntimeWarning) for w in caught),
            "Expected a RuntimeWarning when MLS fails.",
        )

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


class TestMLSInitConstraintFiltering(unittest.TestCase):
    """MLS frequencies with periods longer than the data span are filtered."""

    def setUp(self):
        self.lc = make_simple_sinusoid_1d(
            n_obs=100,
            period=5.0,
            noise_level=0.1,
            noise_type="gaussian",
            irregular=False,
            seed=42,
        )

    def test_out_of_range_frequencies_are_filtered(self):
        """Frequencies below f_min (period > data span) trigger a warning."""
        # Return one valid frequency and one unphysically low frequency.
        valid_freq = torch.tensor([0.2], dtype=torch.float32)  # period=5, valid
        # A frequency below 1/t_span is unphysical (period > data span).
        # t_span=20, so f_min=0.05; use 0.001 (period=1000 >> 20).
        invalid_freq = torch.tensor([0.001], dtype=torch.float32)
        all_freqs = torch.cat([valid_freq, invalid_freq])
        all_mask = torch.tensor([True, True], dtype=torch.bool)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch.object(self.lc, "fit_LS", return_value=(all_freqs, all_mask)):
                _fit_without_training(self.lc)
        # A filtering warning should have been emitted.
        self.assertTrue(
            any(
                issubclass(w.category, RuntimeWarning)
                and "longer than the data span" in str(w.message)
                for w in caught
            ),
            "Expected a RuntimeWarning about out-of-range MLS frequencies.",
        )
        # Only the valid frequency should remain → 1 mixture component.
        self.assertEqual(self.lc.model.covar_module.num_mixtures, 1)


class TestMLSInit2D(unittest.TestCase):
    """MLS initialisation with 2D (multiband) light curves."""

    def setUp(self):
        self.lc2d = make_chromatic_sinusoid_2d(
            n_per_band=50,
            period=5.0,
            wavelengths=[500.0, 700.0],
            amplitude_slope=0.0,
            noise_level=0.0,
            irregular=False,
            seed=42,
        )

    def _fit_2d_without_training(self, **kwargs):
        defaults = {
            "model": "2D",
            "check_sampling": False,
            "check_variability": False,
            "training_iter": 1,
        }
        defaults.update(kwargs)
        with patch("pgmuvi.lightcurve.train", return_value=_DUMMY_RESULTS):
            with patch.object(self.lc2d, "_train"):
                with patch.object(self.lc2d, "print_parameters"):
                    self.lc2d.fit(**defaults)

    def test_2d_mls_sets_num_mixtures(self):
        """MLS init sets num_mixtures from significant peaks for 2D SM model."""
        self._fit_2d_without_training()
        self.assertGreaterEqual(self.lc2d.model.covar_module.num_mixtures, 1)

    def test_2d_explicit_num_mixtures_respected(self):
        """Explicit num_mixtures is respected for 2D SM models."""
        self._fit_2d_without_training(num_mixtures=3)
        self.assertEqual(self.lc2d.model.covar_module.num_mixtures, 3)

    def test_2d_use_mls_init_false_fallback(self):
        """use_mls_init=False falls back to num_mixtures=4 for 2D models."""
        self._fit_2d_without_training(use_mls_init=False)
        self.assertEqual(self.lc2d.model.covar_module.num_mixtures, 4)


if __name__ == "__main__":
    unittest.main()
