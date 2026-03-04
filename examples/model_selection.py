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

np.random.seed(123)
torch.manual_seed(123)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_lc(signal_type: str):
    """Create a Lightcurve for a given signal type."""
    n = 80
    t = torch.sort(torch.rand(n) * 20.0)[0].float()

    if signal_type == "strong_periodic":
        y = (torch.sin(2 * np.pi * t / 5.0) + 0.05 * torch.randn(n)).float()
    elif signal_type == "moderate_periodic":
        y = (
            0.5 * torch.sin(2 * np.pi * t / 5.0) + 0.4 * torch.randn(n)
        ).float()
    elif signal_type == "noise":
        y = torch.randn(n).float() * 0.3
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    return Lightcurve(t, y)


def _make_lc_2d(is_achromatic: bool):
    """Create a 2D multiband Lightcurve."""
    n = 60
    t = torch.sort(torch.rand(n) * 20.0)[0].float()
    wl = torch.cat([
        torch.ones(n // 2, dtype=torch.float32) * 500.0,
        torch.ones(n - n // 2, dtype=torch.float32) * 700.0,
    ])
    x = torch.stack([t, wl], dim=1)

    if is_achromatic:
        # Same period in all bands
        y = torch.sin(2 * np.pi * t / 5.0).float()
    else:
        # Different 'periods' per band
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
