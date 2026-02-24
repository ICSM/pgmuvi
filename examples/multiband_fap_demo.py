#!/usr/bin/env python3
"""
Demonstration of multiband false-alarm probability (FAP) computation.

This script demonstrates the new FAP computation capability for multiband
Lomb-Scargle periodograms in pgmuvi. It shows how to:
1. Create multiband lightcurves
2. Compute periods with statistical significance testing
3. Compare different FAP methods
4. Interpret results

Author: pgmuvi contributors
Date: 2024
"""

import numpy as np
import torch
from pgmuvi.lightcurve import Lightcurve
from pgmuvi.multiband_ls_significance import MultibandLSWithSignificance


def create_multiband_lightcurve(signal_strength=2.0, noise_level=0.1, n_samples=100):
    """
    Create a synthetic multiband lightcurve with a periodic signal.

    Parameters
    ----------
    signal_strength : float
        Amplitude of the periodic signal
    noise_level : float
        Standard deviation of Gaussian noise
    n_samples : int
        Number of data points

    Returns
    -------
    lc : Lightcurve
        2D multiband lightcurve object
    true_freq : float
        True frequency of the injected signal
    """
    # Create time array
    time = np.linspace(0, 20, n_samples)

    # Create two bands (e.g., different wavelengths)
    bands = np.concatenate([
        np.ones(n_samples // 2) * 0.5,    # Band 1
        np.ones(n_samples - n_samples // 2) * 1.5  # Band 2
    ])

    # Shuffle to mix observations from different bands
    indices = np.random.permutation(n_samples)
    time = time[indices]
    bands = bands[indices]

    # Create periodic signal with some noise
    true_freq = 0.5  # Hz
    y = signal_strength * np.sin(2 * np.pi * true_freq * time)
    y += noise_level * np.random.randn(n_samples)

    # Convert to torch tensors in 2D format
    # Format: (n_samples, 2) where [:, 0] is time, [:, 1] is band
    xdata = torch.stack([
        torch.as_tensor(time, dtype=torch.float32),
        torch.as_tensor(bands, dtype=torch.float32)
    ], dim=1)
    ydata = torch.as_tensor(y, dtype=torch.float32)

    return Lightcurve(xdata, ydata), true_freq


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("MULTIBAND FALSE-ALARM PROBABILITY (FAP) DEMONSTRATION")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Example 1: Strong signal
    print("\n1. STRONG PERIODIC SIGNAL")
    print("-" * 80)
    lc_strong, true_freq = create_multiband_lightcurve(
        signal_strength=2.0,
        noise_level=0.1,
        n_samples=100
    )

    # Find significant periods
    freq, significant = lc_strong.fit_LS(num_peaks=3, single_threshold=0.05)

    print(f"True frequency: {true_freq:.4f} Hz")
    print(f"Found {len(freq)} peaks:")
    for i, (f, sig) in enumerate(zip(freq, significant)):
        status = "SIGNIFICANT" if sig else "not significant"
        print(f"  {i+1}. Frequency: {f.item():.4f} Hz - {status}")

    # Example 2: Weak signal
    print("\n2. WEAK PERIODIC SIGNAL")
    print("-" * 80)
    lc_weak, true_freq = create_multiband_lightcurve(
        signal_strength=0.5,
        noise_level=0.5,
        n_samples=100
    )

    freq, significant = lc_weak.fit_LS(num_peaks=3, single_threshold=0.05)
    print(f"True frequency: {true_freq:.4f} Hz")
    print(f"Found {len(freq)} peaks:")
    for i, (f, sig) in enumerate(zip(freq, significant)):
        status = "SIGNIFICANT" if sig else "not significant"
        print(f"  {i+1}. Frequency: {f.item():.4f} Hz - {status}")

    # Example 3: Compare FAP methods
    print("\n3. COMPARING FAP METHODS")
    print("-" * 80)
    lc_test, _ = create_multiband_lightcurve(signal_strength=2.0, noise_level=0.1)

    # Get data for direct FAP computation
    t = lc_test.xdata[:, 0].numpy()
    bands = lc_test.xdata[:, 1].numpy()
    y = lc_test.ydata.numpy()

    # Create MultibandLSWithSignificance object
    ls = MultibandLSWithSignificance(t, y, bands)

    # Compute periodogram
    freq_grid = ls.autofrequency()
    power = ls.power(freq_grid)
    max_power = power.max()

    print(f"Maximum power in periodogram: {max_power:.4f}")
    print("\nFAP estimates using different methods:")

    # Method 1: Bootstrap (default, most robust)
    fap_bootstrap = ls.false_alarm_probability(
        max_power,
        method='bootstrap',
        n_samples=100  # Use 100 for speed, increase for accuracy
    )
    print(f"  Bootstrap (n=100):    FAP = {fap_bootstrap:.6f}")

    # Method 2: Analytical (fastest)
    fap_analytical = ls.false_alarm_probability(
        max_power,
        method='analytical'
    )
    print(f"  Analytical:           FAP = {fap_analytical:.6f}")

    # Method 3: Calibrated (conservative)
    fap_calibrated = ls.false_alarm_probability(
        max_power,
        method='calibrated'
    )
    print(f"  Calibrated:           FAP = {fap_calibrated:.6f}")

    # Interpretation guide
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
FAP (False Alarm Probability) interpretation:
  - FAP < 0.001: Very strong evidence for periodicity (< 0.1% chance of being noise)
  - FAP < 0.01:  Strong evidence (< 1% chance)
  - FAP < 0.05:  Moderate evidence (< 5% chance) - commonly used threshold
  - FAP > 0.05:  Weak or no evidence (likely noise)

Method comparison:
  - **bootstrap**: Most robust, accounts for data structure. Recommended default.
    Slower but more accurate. Use n_samples=100-1000.

  - **analytical**: Fastest, uses analytical approximation. Good for quick checks
    or when computational resources are limited.

  - **calibrated**: Conservative estimate using single-band approach. Useful as
    sanity check or when bootstrap is too slow.

  - **phase_scramble**: Preserves temporal correlations. Use when data has
    significant autocorrelation structure.

Tips:
  - Use single_threshold=0.05 (5% FDR) as default
  - For publication-quality results, use bootstrap with n_samples=1000
  - Check multiple methods for consistency
  - Consider multiple testing correction (automatically applied via FDR)
""")


if __name__ == "__main__":
    main()
