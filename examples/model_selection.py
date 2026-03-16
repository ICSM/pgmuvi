"""Example: Auto model selection for astronomical light curves.

This script demonstrates ``Lightcurve.auto_select_model()``, which
automatically recommends the best GP kernel based on the data
characteristics (periodicity strength, wavelength consistency).

Usage::

    python examples/model_selection.py
"""

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_lc(signal_type: str):
    """Create a Lightcurve for a given signal type."""
    if signal_type == "strong_periodic":
        return make_simple_sinusoid_1d(
            n_obs=80, period=5.0, amplitude=1.0, noise_level=0.05,
            t_span=20.0, irregular=True, seed=123,
        )
    elif signal_type == "moderate_periodic":
        return make_simple_sinusoid_1d(
            n_obs=80, period=5.0, amplitude=0.5, noise_level=0.4,
            t_span=20.0, irregular=True, seed=123,
        )
    elif signal_type == "noise":
        torch.manual_seed(123)
        t = torch.sort(torch.rand(80) * 20.0)[0].float()
        y = torch.randn(80).float() * 0.3
        return Lightcurve(t, y)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")


def _make_lc_2d(is_achromatic: bool):
    """Create a 2D multiband Lightcurve."""
    if is_achromatic:
        # Same period in all bands
        return make_chromatic_sinusoid_2d(
            n_per_band=30,
            period=5.0,
            wavelengths=[500.0, 700.0],
            amplitude_law="linear",
            amplitude_slope=0.0,
            noise_level=0.0,
            t_span=20.0,
            irregular=True,
            seed=123,
        )
    else:
        # Different periods per band - build manually since this is a
        # special case not covered by the standard chromatic generator
        torch.manual_seed(123)
        np.random.seed(123)
        n = 60
        t = torch.sort(torch.rand(n) * 20.0)[0].float()
        wl = torch.cat([
            torch.ones(n // 2, dtype=torch.float32) * 500.0,
            torch.ones(n - n // 2, dtype=torch.float32) * 700.0,
        ])
        x = torch.stack([t, wl], dim=1)
        y = torch.zeros(n, dtype=torch.float32)
        y[: n // 2] = torch.sin(2 * np.pi * t[: n // 2] / 4.0).float()
        y[n // 2 :] = torch.sin(2 * np.pi * t[n // 2 :] / 8.0).float()
        return Lightcurve(x, y)


# ---------------------------------------------------------------------------
# Run auto-selection on several scenarios
# ---------------------------------------------------------------------------
scenarios = [
    ("Strong periodic signal", _make_lc("strong_periodic")),
    ("Moderate periodic signal", _make_lc("moderate_periodic")),
    ("Pure noise (no periodicity)", _make_lc("noise")),
    ("2D achromatic variability", _make_lc_2d(is_achromatic=True)),
    ("2D chromatic variability", _make_lc_2d(is_achromatic=False)),
]

print("=" * 70)
print("AUTO MODEL SELECTION DEMO")
print("=" * 70)

for label, lc in scenarios:
    print(f"\nScenario: {label}")
    model_str, diag = lc.auto_select_model(verbose=False)
    print(f"  Recommended model : {model_str}")
    print(f"  Reason            : {diag['reason']}")
    if "max_ls_power" in diag:
        print(f"  Max LS power      : {diag['max_ls_power']:.3f}")

print("\n" + "=" * 70)
print("Verify: use the recommended model string directly with set_model()")
print("=" * 70)
lc_demo = _make_lc("strong_periodic")
model_str, _ = lc_demo.auto_select_model(verbose=False)
lc_demo.set_model(model_str)
print(f"  set_model('{model_str}') => {type(lc_demo.model).__name__}")
print("Done.")
