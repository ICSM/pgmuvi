"""Tests for the extinction amplitude helper."""
import unittest

import numpy as np

from pgmuvi.synthetic import make_chromatic_sinusoid_2d


class TestExtinctionAmplitude(unittest.TestCase):
    """Tests that the extinction amplitude law decreases with wavelength."""

    def test_amplitude_decreases_with_wavelength(self):
        """Extinction amplitude should decrease as wavelength increases."""
        from pgmuvi.synthetic import _extinction_amplitude
        wls = np.array([0.8, 1.2, 2.2])
        amps = _extinction_amplitude(wls, overall_amplitude=5.0, tau=2.0, alpha=1.7, offset=0.0)
        # Each successive amplitude must be strictly smaller
        self.assertGreater(amps[0], amps[1])
        self.assertGreater(amps[1], amps[2])

    def test_make_chromatic_sinusoid_2d_extinction_decreases(self):
        """make_chromatic_sinusoid_2d with extinction law should yield lower amplitude at longer wavelength."""
        wls = [0.8, 1.2, 2.2]
        lc = make_chromatic_sinusoid_2d(
            n_per_band=50,
            period=400.0,
            wavelengths=wls,
            amplitude_law="extinction",
            tau=2.0,
            alpha=1.7,
            noise_level=0.0,
            seed=0,
        )
        # Compute per-band signal range as a proxy for amplitude
        band_ranges = []
        for wl in wls:
            mask = lc.xdata[:, 1] == wl
            band_ranges.append(float(lc.ydata[mask].max() - lc.ydata[mask].min()))
        self.assertGreater(band_ranges[0], band_ranges[1])
        self.assertGreater(band_ranges[1], band_ranges[2])
