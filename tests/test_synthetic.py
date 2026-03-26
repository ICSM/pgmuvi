"""Tests for the pgmuvi.synthetic module."""

import math
import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.synthetic import (
    make_chromatic_sinusoid_2d,
    make_multi_sinusoid_1d,
    make_multi_sinusoid_chromatic_2d,
    make_simple_sinusoid_1d,
)
from pgmuvi.synthetic import _resolve_n_per_band  # internal helper
from pgmuvi.synthetic import _DEFAULT_TSPAN_FACTOR  # internal constant


class TestResolveNPerBand(unittest.TestCase):
    """Unit tests for the _resolve_n_per_band helper."""

    def _rng(self, seed=0):
        return np.random.default_rng(seed)

    def test_int_broadcast(self):
        result = _resolve_n_per_band(30, 4, self._rng())
        self.assertEqual(result, [30, 30, 30, 30])

    def test_list_passthrough(self):
        result = _resolve_n_per_band([10, 20, 30], 3, self._rng())
        self.assertEqual(result, [10, 20, 30])

    def test_list_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            _resolve_n_per_band([10, 20], 3, self._rng())

    def test_tuple_range_values_in_bounds(self):
        lo, hi = 5, 15
        result = _resolve_n_per_band((lo, hi), 5, self._rng())
        self.assertEqual(len(result), 5)
        for v in result:
            self.assertGreaterEqual(v, lo)
            self.assertLessEqual(v, hi)

    def test_tuple_reproducible(self):
        r1 = _resolve_n_per_band((10, 40), 4, self._rng(7))
        r2 = _resolve_n_per_band((10, 40), 4, self._rng(7))
        self.assertEqual(r1, r2)

    def test_tuple_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            _resolve_n_per_band((30, 10), 3, self._rng())

    def test_tuple_zero_min_raises(self):
        with self.assertRaises(ValueError):
            _resolve_n_per_band((0, 10), 3, self._rng())

    def test_tuple_negative_min_raises(self):
        with self.assertRaises(ValueError):
            _resolve_n_per_band((-5, 10), 3, self._rng())

    def test_tuple_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            _resolve_n_per_band((10, 20, 30), 3, self._rng())


class TestMakeSimpleSinusoid1D(unittest.TestCase):
    """Tests for make_simple_sinusoid_1d."""

    def test_returns_lightcurve(self):
        lc = make_simple_sinusoid_1d(seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_ndim_is_1(self):
        lc = make_simple_sinusoid_1d(seed=0)
        self.assertEqual(lc.ndim, 1)

    def test_output_shape(self):
        lc = make_simple_sinusoid_1d(n_obs=50, seed=0)
        self.assertEqual(lc.xdata.shape, torch.Size([50]))
        self.assertEqual(lc.ydata.shape, torch.Size([50]))

    def test_reproducible_with_seed(self):
        lc1 = make_simple_sinusoid_1d(seed=42)
        lc2 = make_simple_sinusoid_1d(seed=42)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_different_seeds_differ(self):
        lc1 = make_simple_sinusoid_1d(seed=0)
        lc2 = make_simple_sinusoid_1d(seed=1)
        self.assertFalse(torch.allclose(lc1.ydata, lc2.ydata))

    def test_noise_free(self):
        """Without noise the signal is a pure sinusoid."""
        lc = make_simple_sinusoid_1d(
            n_obs=100,
            period=5.0,
            amplitude=1.0,
            phase=0.0,
            noise_level=0.0,
            t_span=20.0,
            irregular=False,
            seed=0,
        )
        t = lc.xdata.numpy()
        expected = np.sin(2 * math.pi * t / 5.0).astype(np.float32)
        np.testing.assert_allclose(lc.ydata.numpy(), expected, atol=1e-5)

    def test_irregular_sampling(self):
        lc_reg = make_simple_sinusoid_1d(n_obs=50, irregular=False, seed=0)
        lc_irr = make_simple_sinusoid_1d(n_obs=50, irregular=True, seed=0)
        # Regular sampling should be exactly equally spaced
        diffs = torch.diff(lc_reg.xdata)
        self.assertAlmostEqual(diffs.std().item(), 0.0, places=4)
        # Irregular sampling should NOT be exactly equally spaced (with high probability)
        diffs_irr = torch.diff(lc_irr.xdata)
        self.assertGreater(diffs_irr.std().item(), 1e-4)

    def test_t_span_and_t_min(self):
        lc = make_simple_sinusoid_1d(t_min=100.0, t_span=50.0, seed=0)
        self.assertGreaterEqual(lc.xdata.min().item(), 100.0)
        self.assertLessEqual(lc.xdata.max().item(), 150.0)


class TestMakeMultiSinusoid1D(unittest.TestCase):
    """Tests for make_multi_sinusoid_1d."""

    def test_returns_lightcurve(self):
        lc = make_multi_sinusoid_1d(seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_ndim_is_1(self):
        lc = make_multi_sinusoid_1d(seed=0)
        self.assertEqual(lc.ndim, 1)

    def test_output_shape(self):
        lc = make_multi_sinusoid_1d(n_obs=60, seed=0)
        self.assertEqual(lc.xdata.shape, torch.Size([60]))
        self.assertEqual(lc.ydata.shape, torch.Size([60]))

    def test_reproducible_with_seed(self):
        lc1 = make_multi_sinusoid_1d(seed=7)
        lc2 = make_multi_sinusoid_1d(seed=7)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_custom_components(self):
        components = [
            {"period": 2.0, "amplitude": 1.0, "phase": 0.0},
            {"period": 4.0, "amplitude": 0.5, "phase": math.pi / 2},
        ]
        lc = make_multi_sinusoid_1d(components=components, noise_level=0.0, seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_noise_free_sum_of_sinusoids(self):
        """Without noise the signal equals the sum of component sinusoids."""
        components = [
            {"period": 5.0, "amplitude": 1.0, "phase": 0.0},
            {"period": 3.0, "amplitude": 0.5, "phase": math.pi / 3},
        ]
        lc = make_multi_sinusoid_1d(
            n_obs=100,
            components=components,
            noise_level=0.0,
            t_span=20.0,
            irregular=False,
            seed=0,
        )
        t = lc.xdata.numpy()
        expected = (
            np.sin(2 * math.pi * t / 5.0)
            + 0.5 * np.sin(2 * math.pi * t / 3.0 + math.pi / 3)
        ).astype(np.float32)
        np.testing.assert_allclose(lc.ydata.numpy(), expected, atol=1e-5)


class TestMakeChromaticSinusoid2D(unittest.TestCase):
    """Tests for make_chromatic_sinusoid_2d."""

    def test_returns_lightcurve(self):
        lc = make_chromatic_sinusoid_2d(seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_ndim_is_2(self):
        lc = make_chromatic_sinusoid_2d(seed=0)
        self.assertEqual(lc.ndim, 2)

    def test_output_shape_default(self):
        """Default: 3 bands x 50 obs = 150 rows, 2 columns."""
        lc = make_chromatic_sinusoid_2d(n_per_band=50, seed=0)
        self.assertEqual(lc.xdata.shape, torch.Size([150, 2]))
        self.assertEqual(lc.ydata.shape, torch.Size([150]))

    def test_output_shape_per_band_list(self):
        lc = make_chromatic_sinusoid_2d(
            n_per_band=[10, 20, 30],
            wavelengths=[450.0, 600.0, 750.0],
            seed=0,
        )
        self.assertEqual(lc.xdata.shape[0], 60)

    def test_n_per_band_range_tuple(self):
        """tuple (min, max) gives each band a random count in [min, max]."""
        lo, hi = 20, 40
        lc = make_chromatic_sinusoid_2d(
            n_per_band=(lo, hi),
            wavelengths=[450.0, 600.0, 750.0],
            seed=7,
        )
        # Total observations must be within possible bounds
        n_total = lc.xdata.shape[0]
        self.assertGreaterEqual(n_total, 3 * lo)
        self.assertLessEqual(n_total, 3 * hi)

    def test_n_per_band_range_tuple_reproducible(self):
        """Same seed produces identical band counts and data."""
        lc1 = make_chromatic_sinusoid_2d(
            n_per_band=(10, 30), wavelengths=[450.0, 600.0, 750.0], seed=5
        )
        lc2 = make_chromatic_sinusoid_2d(
            n_per_band=(10, 30), wavelengths=[450.0, 600.0, 750.0], seed=5
        )
        self.assertEqual(lc1.xdata.shape, lc2.xdata.shape)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_n_per_band_range_tuple_bands_may_differ(self):
        """With a wide range, different seeds produce different band counts."""
        lc1 = make_chromatic_sinusoid_2d(
            n_per_band=(10, 50), wavelengths=[450.0, 600.0, 750.0], seed=1
        )
        lc2 = make_chromatic_sinusoid_2d(
            n_per_band=(10, 50), wavelengths=[450.0, 600.0, 750.0], seed=2
        )
        # Both results must be within the valid range
        for lc in (lc1, lc2):
            self.assertGreaterEqual(lc.xdata.shape[0], 3 * 10)
            self.assertLessEqual(lc.xdata.shape[0], 3 * 50)
        # Different seeds should produce different totals (with high probability
        # for a range of 40 across 3 bands)
        self.assertNotEqual(lc1.xdata.shape[0], lc2.xdata.shape[0])

    def test_n_per_band_invalid_tuple_raises(self):
        """Tuple with min > max should raise ValueError."""
        with self.assertRaises(ValueError):
            make_chromatic_sinusoid_2d(
                n_per_band=(30, 10), wavelengths=[450.0, 600.0, 750.0]
            )

    def test_n_per_band_tuple_zero_min_raises(self):
        """Tuple with min < 1 should raise ValueError."""
        with self.assertRaises(ValueError):
            make_chromatic_sinusoid_2d(
                n_per_band=(0, 10), wavelengths=[450.0, 600.0, 750.0]
            )

    def test_mismatched_n_per_band_raises(self):
        with self.assertRaises(ValueError):
            make_chromatic_sinusoid_2d(
                n_per_band=[10, 20],
                wavelengths=[450.0, 600.0, 750.0],
            )

    def test_linear_amplitude_law(self):
        lc = make_chromatic_sinusoid_2d(
            amplitude_law="linear", amplitude_slope=0.3, seed=0
        )
        self.assertIsInstance(lc, Lightcurve)

    def test_extinction_amplitude_law(self):
        lc = make_chromatic_sinusoid_2d(
            wavelengths=[0.8, 1.2, 2.2],
            amplitude_law="extinction",
            tau=2.0,
            alpha=1.7,
            seed=0,
        )
        self.assertIsInstance(lc, Lightcurve)

    def test_unknown_amplitude_law_raises(self):
        with self.assertRaises(ValueError):
            make_chromatic_sinusoid_2d(amplitude_law="unknown")

    def test_linear_phase_law(self):
        lc = make_chromatic_sinusoid_2d(phase_law="linear", phase_slope=0.1, seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_unknown_phase_law_raises(self):
        with self.assertRaises(ValueError):
            make_chromatic_sinusoid_2d(phase_law="unknown")

    def test_reproducible_with_seed(self):
        lc1 = make_chromatic_sinusoid_2d(seed=3)
        lc2 = make_chromatic_sinusoid_2d(seed=3)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_wavelength_column_values(self):
        """Second column of xdata must only contain the specified wavelengths."""
        wls = [450.0, 600.0, 750.0]
        lc = make_chromatic_sinusoid_2d(wavelengths=wls, n_per_band=20, seed=0)
        wl_col = lc.xdata[:, 1].numpy()
        unique_wls = set(np.round(wl_col, 3))
        self.assertEqual(unique_wls, set(wls))

    def test_amplitude_varies_with_wavelength(self):
        """With linear law, higher slope => larger amplitude spread across bands."""
        lc_flat = make_chromatic_sinusoid_2d(
            amplitude_slope=0.0, noise_level=0.0, seed=0
        )
        lc_steep = make_chromatic_sinusoid_2d(
            amplitude_slope=1.0, noise_level=0.0, seed=0
        )
        # The steep-slope curve should have a larger range of y values
        self.assertGreater(
            lc_steep.ydata.abs().max().item(),
            lc_flat.ydata.abs().max().item(),
        )


class TestMakeMultiSinusoidChromatic2D(unittest.TestCase):
    """Tests for make_multi_sinusoid_chromatic_2d."""

    def test_returns_lightcurve(self):
        lc = make_multi_sinusoid_chromatic_2d(seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_ndim_is_2(self):
        lc = make_multi_sinusoid_chromatic_2d(seed=0)
        self.assertEqual(lc.ndim, 2)

    def test_output_shape_default(self):
        """Default: 3 bands x 50 obs = 150 rows."""
        lc = make_multi_sinusoid_chromatic_2d(n_per_band=50, seed=0)
        self.assertEqual(lc.xdata.shape[0], 150)

    def test_custom_components(self):
        components = [
            {"period": 5.0, "amplitude_fraction": 0.4, "phase": 0.0},
            {"period": 2.5, "amplitude_fraction": 0.1, "phase": math.pi / 2},
            {"period": 10.0, "amplitude_fraction": 0.05, "phase": math.pi},
        ]
        lc = make_multi_sinusoid_chromatic_2d(components=components, seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_reproducible_with_seed(self):
        lc1 = make_multi_sinusoid_chromatic_2d(seed=99)
        lc2 = make_multi_sinusoid_chromatic_2d(seed=99)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_extinction_law(self):
        lc = make_multi_sinusoid_chromatic_2d(
            wavelengths=[0.8, 1.2, 2.2],
            amplitude_law="extinction",
            tau=2.0,
            alpha=1.7,
            overall_amplitude=5.0,
            offset=0.2,
            seed=0,
        )
        self.assertIsInstance(lc, Lightcurve)
        self.assertEqual(lc.ndim, 2)

    def test_linear_amplitude_law(self):
        lc = make_multi_sinusoid_chromatic_2d(
            amplitude_law="linear",
            amplitude_slope=0.3,
            wl_ref=600.0,
            seed=0,
        )
        self.assertIsInstance(lc, Lightcurve)

    def test_unknown_amplitude_law_raises(self):
        with self.assertRaises(ValueError):
            make_multi_sinusoid_chromatic_2d(amplitude_law="bad")

    def test_mismatched_n_per_band_raises(self):
        with self.assertRaises(ValueError):
            make_multi_sinusoid_chromatic_2d(
                n_per_band=[10, 20],
                wavelengths=[0.8, 1.2, 2.2],
            )

    def test_n_per_band_range_tuple(self):
        """tuple (min, max) gives each band a random count in [min, max]."""
        lo, hi = 15, 35
        lc = make_multi_sinusoid_chromatic_2d(
            n_per_band=(lo, hi),
            wavelengths=[0.8, 1.2, 2.2],
            seed=7,
        )
        n_total = lc.xdata.shape[0]
        self.assertGreaterEqual(n_total, 3 * lo)
        self.assertLessEqual(n_total, 3 * hi)

    def test_n_per_band_range_tuple_reproducible(self):
        """Same seed gives identical results."""
        lc1 = make_multi_sinusoid_chromatic_2d(
            n_per_band=(10, 40), wavelengths=[0.8, 1.2, 2.2], seed=3
        )
        lc2 = make_multi_sinusoid_chromatic_2d(
            n_per_band=(10, 40), wavelengths=[0.8, 1.2, 2.2], seed=3
        )
        self.assertEqual(lc1.xdata.shape, lc2.xdata.shape)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_n_per_band_invalid_tuple_raises(self):
        """Tuple with min > max should raise ValueError."""
        with self.assertRaises(ValueError):
            make_multi_sinusoid_chromatic_2d(
                n_per_band=(40, 10), wavelengths=[0.8, 1.2, 2.2]
            )

    def test_default_components_are_lpv_like(self):
        """Default components should have LPV-like periods (400 and 200 days)."""
        lc = make_multi_sinusoid_chromatic_2d(
            n_per_band=10, noise_level=0.0, seed=0
        )
        # With period 400 and t_span 2.3 * 400 = 920, we should see variation
        self.assertGreater(lc.ydata.abs().max().item(), 0.0)


class TestNoiseType(unittest.TestCase):
    """Tests for the noise_type parameter across all generators."""

    def test_simple_sinusoid_poisson_returns_lightcurve(self):
        lc = make_simple_sinusoid_1d(noise_type="poisson", noise_level=0.1, seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_multi_sinusoid_poisson_returns_lightcurve(self):
        lc = make_multi_sinusoid_1d(noise_type="poisson", noise_level=0.1, seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_chromatic_2d_poisson_returns_lightcurve(self):
        lc = make_chromatic_sinusoid_2d(noise_type="poisson", noise_level=0.1, seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_multi_chromatic_2d_poisson_returns_lightcurve(self):
        lc = make_multi_sinusoid_chromatic_2d(
            noise_type="poisson", noise_level=0.1, seed=0
        )
        self.assertIsInstance(lc, Lightcurve)

    def test_poisson_reproducible_with_seed(self):
        """Same seed and noise_type='poisson' gives identical results."""
        lc1 = make_simple_sinusoid_1d(noise_type="poisson", noise_level=0.1, seed=5)
        lc2 = make_simple_sinusoid_1d(noise_type="poisson", noise_level=0.1, seed=5)
        self.assertTrue(torch.allclose(lc1.ydata, lc2.ydata))

    def test_gaussian_and_poisson_same_shape(self):
        """Both noise types produce the same number of observations."""
        lc_g = make_simple_sinusoid_1d(noise_type="gaussian", seed=0)
        lc_p = make_simple_sinusoid_1d(noise_type="poisson", seed=0)
        self.assertEqual(lc_g.ydata.shape, lc_p.ydata.shape)

    def test_poisson_noise_differs_from_gaussian(self):
        """Poisson and Gaussian noise should produce different y values."""
        lc_g = make_simple_sinusoid_1d(
            noise_type="gaussian", noise_level=0.3, seed=42
        )
        lc_p = make_simple_sinusoid_1d(
            noise_type="poisson", noise_level=0.3, seed=42
        )
        self.assertFalse(torch.allclose(lc_g.ydata, lc_p.ydata))

    def test_unknown_noise_type_raises(self):
        with self.assertRaises(ValueError):
            make_simple_sinusoid_1d(noise_type="bad_type")

    def test_unknown_noise_type_raises_2d(self):
        with self.assertRaises(ValueError):
            make_chromatic_sinusoid_2d(noise_type="bad_type")

    def test_noise_free_poisson(self):
        """noise_level=0 with poisson type gives the same as noise_level=0 gaussian."""
        lc_g = make_simple_sinusoid_1d(
            noise_level=0.0, noise_type="gaussian", seed=0
        )
        lc_p = make_simple_sinusoid_1d(
            noise_level=0.0, noise_type="poisson", seed=0
        )
        self.assertTrue(torch.allclose(lc_g.ydata, lc_p.ydata))

    def test_noise_type_none_simple_1d_returns_lightcurve(self):
        """noise_type=None produces a Lightcurve with no noise."""
        lc = make_simple_sinusoid_1d(noise_type=None, noise_level=0.5, seed=0)
        self.assertIsInstance(lc, Lightcurve)

    def test_noise_type_none_simple_1d_no_yerr(self):
        """noise_type=None leaves yerr unset."""
        lc = make_simple_sinusoid_1d(noise_type=None, noise_level=0.5, seed=0)
        self.assertIsNone(getattr(lc, "yerr", None))

    def test_noise_type_none_simple_1d_same_as_noise_free(self):
        """noise_type=None with any noise_level gives the noiseless signal."""
        lc_none = make_simple_sinusoid_1d(
            noise_type=None, noise_level=1.0, irregular=False, seed=0
        )
        lc_zero = make_simple_sinusoid_1d(
            noise_type="gaussian", noise_level=0.0, irregular=False, seed=0
        )
        self.assertTrue(torch.allclose(lc_none.ydata, lc_zero.ydata))

    def test_noise_type_none_multi_1d(self):
        """noise_type=None works for make_multi_sinusoid_1d."""
        lc = make_multi_sinusoid_1d(noise_type=None, noise_level=0.5, seed=0)
        self.assertIsNone(getattr(lc, "yerr", None))

    def test_noise_type_none_chromatic_2d(self):
        """noise_type=None works for make_chromatic_sinusoid_2d."""
        lc = make_chromatic_sinusoid_2d(noise_type=None, noise_level=0.5, seed=0)
        self.assertIsNone(getattr(lc, "yerr", None))

    def test_noise_type_none_multi_chromatic_2d(self):
        """noise_type=None works for make_multi_sinusoid_chromatic_2d."""
        lc = make_multi_sinusoid_chromatic_2d(
            noise_type=None, noise_level=0.5, seed=0
        )
        self.assertIsNone(getattr(lc, "yerr", None))

    def test_default_noise_type_is_poisson(self):
        """Default noise_type is 'poisson': yerr values vary (not constant)."""
        lc = make_simple_sinusoid_1d(noise_level=0.1, seed=0)
        self.assertIsNotNone(lc.yerr)
        # Poisson yerr is non-constant; gaussian yerr would be all-equal
        self.assertGreater(lc.yerr.std().item(), 0.0)


class TestMultiSinusoid1DValidation(unittest.TestCase):
    """Tests for input validation in make_multi_sinusoid_1d."""

    def test_component_missing_period_raises(self):
        """Component dict without 'period' raises ValueError."""
        bad = [{"amplitude": 1.0, "phase": 0.0}]
        with self.assertRaises(ValueError):
            make_multi_sinusoid_1d(components=bad, seed=0)

    def test_component_missing_amplitude_raises(self):
        """Component dict without 'amplitude' raises ValueError."""
        bad = [{"period": 5.0, "phase": 0.0}]
        with self.assertRaises(ValueError):
            make_multi_sinusoid_1d(components=bad, seed=0)

    def test_component_missing_phase_raises(self):
        """Component dict without 'phase' raises ValueError."""
        bad = [{"period": 5.0, "amplitude": 1.0}]
        with self.assertRaises(ValueError):
            make_multi_sinusoid_1d(components=bad, seed=0)

    def test_component_missing_multiple_keys_raises(self):
        """Component dict missing multiple required keys reports all of them."""
        bad = [{"period": 5.0}]  # missing 'amplitude' and 'phase'
        with self.assertRaises(ValueError) as ctx:
            make_multi_sinusoid_1d(components=bad, seed=0)
        msg = str(ctx.exception)
        self.assertIn("amplitude", msg)
        self.assertIn("phase", msg)

    def test_unknown_noise_type_raises_multi_1d(self):
        """Invalid noise_type raises ValueError."""
        with self.assertRaises(ValueError):
            make_multi_sinusoid_1d(noise_type="bad_type", seed=0)

    def test_unknown_noise_type_raises_multi_chromatic_2d(self):
        """Invalid noise_type raises ValueError."""
        with self.assertRaises(ValueError):
            make_multi_sinusoid_chromatic_2d(noise_type="bad_type", seed=0)


class TestYerrPopulated(unittest.TestCase):
    """Tests that yerr is populated on returned Lightcurve objects."""

    def test_simple_sinusoid_gaussian_yerr_set(self):
        """make_simple_sinusoid_1d with gaussian noise populates yerr."""
        lc = make_simple_sinusoid_1d(noise_level=0.1, noise_type="gaussian", seed=0)
        self.assertIsNotNone(lc.yerr)
        self.assertEqual(lc.yerr.shape, lc.ydata.shape)

    def test_simple_sinusoid_gaussian_yerr_constant(self):
        """Gaussian noise gives constant uncertainties equal to noise_level."""
        noise_level = 0.2
        lc = make_simple_sinusoid_1d(
            noise_level=noise_level, noise_type="gaussian", seed=0
        )
        np.testing.assert_allclose(
            lc.yerr.numpy(), noise_level, rtol=1e-5
        )

    def test_simple_sinusoid_poisson_yerr_set(self):
        """make_simple_sinusoid_1d with poisson noise populates yerr."""
        lc = make_simple_sinusoid_1d(noise_level=0.1, noise_type="poisson", seed=0)
        self.assertIsNotNone(lc.yerr)
        self.assertEqual(lc.yerr.shape, lc.ydata.shape)

    def test_simple_sinusoid_poisson_yerr_positive(self):
        """Poisson uncertainties must all be positive."""
        lc = make_simple_sinusoid_1d(noise_level=0.1, noise_type="poisson", seed=0)
        self.assertTrue((lc.yerr > 0).all())

    def test_simple_sinusoid_no_noise_yerr_none(self):
        """With noise_level=0, yerr should not be set."""
        lc = make_simple_sinusoid_1d(noise_level=0.0, seed=0)
        self.assertIsNone(getattr(lc, "yerr", None))

    def test_multi_sinusoid_1d_gaussian_yerr_set(self):
        """make_multi_sinusoid_1d with gaussian noise populates yerr."""
        lc = make_multi_sinusoid_1d(noise_level=0.1, noise_type="gaussian", seed=0)
        self.assertIsNotNone(lc.yerr)
        self.assertEqual(lc.yerr.shape, lc.ydata.shape)

    def test_multi_sinusoid_1d_no_noise_yerr_none(self):
        """With noise_level=0, yerr should not be set."""
        lc = make_multi_sinusoid_1d(noise_level=0.0, seed=0)
        self.assertIsNone(getattr(lc, "yerr", None))

    def test_chromatic_2d_gaussian_yerr_set(self):
        """make_chromatic_sinusoid_2d with gaussian noise populates yerr."""
        lc = make_chromatic_sinusoid_2d(noise_level=0.1, noise_type="gaussian", seed=0)
        self.assertIsNotNone(lc.yerr)
        self.assertEqual(lc.yerr.shape, lc.ydata.shape)

    def test_chromatic_2d_no_noise_yerr_none(self):
        """With noise_level=0, yerr should not be set."""
        lc = make_chromatic_sinusoid_2d(noise_level=0.0, seed=0)
        self.assertIsNone(getattr(lc, "yerr", None))

    def test_multi_chromatic_2d_gaussian_yerr_set(self):
        """make_multi_sinusoid_chromatic_2d with gaussian noise populates yerr."""
        lc = make_multi_sinusoid_chromatic_2d(
            noise_level=0.1, noise_type="gaussian", seed=0
        )
        self.assertIsNotNone(lc.yerr)
        self.assertEqual(lc.yerr.shape, lc.ydata.shape)

    def test_multi_chromatic_2d_poisson_yerr_set(self):
        """make_multi_sinusoid_chromatic_2d with poisson noise populates yerr."""
        lc = make_multi_sinusoid_chromatic_2d(
            noise_level=0.1, noise_type="poisson", seed=0
        )
        self.assertIsNotNone(lc.yerr)
        self.assertEqual(lc.yerr.shape, lc.ydata.shape)

    def test_multi_chromatic_2d_no_noise_yerr_none(self):
        """With noise_level=0, yerr should not be set."""
        lc = make_multi_sinusoid_chromatic_2d(noise_level=0.0, seed=0)
        self.assertIsNone(getattr(lc, "yerr", None))


class TestDefaultTSpan(unittest.TestCase):
    """Tests that t_span=None computes the span from _DEFAULT_TSPAN_FACTOR * period."""

    def test_simple_sinusoid_1d_default_tspan(self):
        """make_simple_sinusoid_1d: default t_span == _DEFAULT_TSPAN_FACTOR * period."""
        period = 50.0
        lc = make_simple_sinusoid_1d(
            period=period, n_obs=100, noise_level=0.0, irregular=False, seed=0
        )
        expected_span = _DEFAULT_TSPAN_FACTOR * period
        actual_span = lc.xdata.max().item() - lc.xdata.min().item()
        self.assertAlmostEqual(actual_span, expected_span, places=3)

    def test_simple_sinusoid_1d_explicit_tspan_unchanged(self):
        """Explicit t_span is used as-is (not scaled by factor)."""
        lc = make_simple_sinusoid_1d(
            period=50.0, t_span=200.0, n_obs=100, noise_level=0.0,
            irregular=False, seed=0,
        )
        actual_span = lc.xdata.max().item() - lc.xdata.min().item()
        self.assertAlmostEqual(actual_span, 200.0, places=3)

    def test_multi_sinusoid_1d_default_tspan_uses_max_period(self):
        """make_multi_sinusoid_1d: default t_span == factor * max(component periods)."""
        components = [
            {"period": 5.0, "amplitude": 1.0, "phase": 0.0},
            {"period": 12.0, "amplitude": 0.5, "phase": 0.0},
            {"period": 3.0, "amplitude": 0.3, "phase": 0.0},
        ]
        lc = make_multi_sinusoid_1d(
            components=components, n_obs=100, noise_level=0.0,
            irregular=False, seed=0,
        )
        expected_span = _DEFAULT_TSPAN_FACTOR * 12.0
        actual_span = lc.xdata.max().item() - lc.xdata.min().item()
        self.assertAlmostEqual(actual_span, expected_span, places=3)

    def test_chromatic_sinusoid_2d_default_tspan(self):
        """make_chromatic_sinusoid_2d: default t_span == factor * period."""
        period = 200.0
        lc = make_chromatic_sinusoid_2d(
            period=period, n_per_band=50, noise_level=0.0,
            irregular=False, seed=0,
        )
        expected_span = _DEFAULT_TSPAN_FACTOR * period
        # xdata[:, 0] is time; check span per band
        times = lc.xdata[:, 0]
        actual_span = times.max().item() - times.min().item()
        self.assertAlmostEqual(actual_span, expected_span, places=3)

    def test_multi_sinusoid_chromatic_2d_default_tspan_uses_max_period(self):
        """make_multi_sinusoid_chromatic_2d: default t_span uses max period."""
        components = [
            {"period": 400.0, "amplitude_fraction": 0.4, "phase": 0.0},
            {"period": 200.0, "amplitude_fraction": 0.1, "phase": 0.0},
        ]
        lc = make_multi_sinusoid_chromatic_2d(
            components=components, n_per_band=50, noise_level=0.0,
            irregular=False, seed=0,
        )
        expected_span = _DEFAULT_TSPAN_FACTOR * 400.0
        times = lc.xdata[:, 0]
        actual_span = times.max().item() - times.min().item()
        self.assertAlmostEqual(actual_span, expected_span, places=3)


class TestEmptyComponentsValidation(unittest.TestCase):
    """Tests that passing an empty components list raises a clear ValueError."""

    def test_multi_sinusoid_1d_empty_components_raises(self):
        """make_multi_sinusoid_1d: components=[] raises ValueError."""
        with self.assertRaises(ValueError):
            make_multi_sinusoid_1d(components=[], seed=0)

    def test_multi_sinusoid_chromatic_2d_empty_components_raises(self):
        """make_multi_sinusoid_chromatic_2d: components=[] raises ValueError."""
        with self.assertRaises(ValueError):
            make_multi_sinusoid_chromatic_2d(components=[], seed=0)


if __name__ == "__main__":
    unittest.main()
