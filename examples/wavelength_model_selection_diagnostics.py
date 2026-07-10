"""Example: wavelength-dependence diagnostics for 2D light curves.

This script demonstrates the diagnostics workflow for deciding which
multiwavelength GP model families are plausible.  The pre-fit diagnostic stage
is cheap and does not run a GP fit.  The optional comparison stage runs the
recommended candidate models through the normal ``Lightcurve.fit`` pathway and
summarizes residual/predictive diagnostics.

Usage::

    python examples/wavelength_model_selection_diagnostics.py
"""

from __future__ import annotations

import numpy as np

from pgmuvi.synthetic import make_chromatic_sinusoid_2d


# ---------------------------------------------------------------------------
# Synthetic 2D light curve with a shared period and chromatic amplitude.
# Replace this with Lightcurve.from_csv(...) for real data.
# ---------------------------------------------------------------------------
lc = make_chromatic_sinusoid_2d(
    n_per_band=40,
    period=50.0,
    wavelengths=[1.2, 2.2, 3.4, 4.6],
    amplitude_law="linear",
    amplitude_slope=0.4,
    noise_level=0.05,
    t_span=250.0,
    irregular=True,
    seed=123,
)

# ---------------------------------------------------------------------------
# Stage 1: cheap pre-fit diagnostics.
# ---------------------------------------------------------------------------
# In a real analysis, the period/frequency should usually come from LS, ACF,
# consensus diagnostics, or previous domain knowledge.  Supplying it here lets
# the report measure period-locked amplitude and phase/lag by wavelength.
diag = lc.diagnose_wavelength_dependence(period=50.0)

print("\n" + "=" * 78)
print("PRE-FIT WAVELENGTH DIAGNOSTICS")
print("=" * 78)
print(lc.format_wavelength_diagnostics_report(diag, max_band_rows=10))

print("\nRecommended candidates:")
for rec in diag["recommended_candidate_models"]:
    print(f"  - {rec.get('name')}: model={rec.get('model')} reason={rec.get('reason')}")

# Create science-facing diagnostic plots.  The figures are returned in a
# dictionary so scripts can save selected panels without displaying them.
figs = lc.plot_wavelength_diagnostics(diag, show=False)
print(f"\nCreated diagnostic figures: {sorted(figs)}")

# ---------------------------------------------------------------------------
# Stage 2: optional candidate comparison.
# ---------------------------------------------------------------------------
# Keep this small for an example.  For real sources, increase training_iter and
# miniter, and consider fit_strategy="consensus" for the baseline 2D model.
comparison = lc.compare_wavelength_models(
    diagnostic_report=diag,
    base_fit_kwargs={
        "training_iter": 5,
        "miniter": 0,
        "learn_additional_noise": True,
        "verbose": False,
    },
    residual_diagnostic_kwargs={"period": 50.0},
    stop_on_error=False,
)

print("\n" + "=" * 78)
print("CANDIDATE MODEL COMPARISON")
print("=" * 78)
print(lc.format_wavelength_diagnostics_report(diag, comparison_report=comparison))

comparison_figs = lc.plot_wavelength_model_comparison(comparison, show=False)
print(f"\nCreated comparison figures: {sorted(comparison_figs)}")

if comparison.get("best_candidate"):
    best = comparison["best_candidate"]
    score = best.get("score")
    score_txt = "nan" if score is None or not np.isfinite(score) else f"{score:.3g}"
    print(f"\nLowest predictive-score candidate: {best.get('name')} ({score_txt})")
