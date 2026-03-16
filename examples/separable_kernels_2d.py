"""Example: Separable 2D kernels for multiwavelength light curves.

This script demonstrates how to use ``AchromaticGPModel`` and
``WavelengthDependentGPModel`` for 2D (time x wavelength) data, and
shows how to use smart initialisation with ``initialize_separable_from_data``.

Usage::

    python examples/separable_kernels_2d.py
"""

import gpytorch

from pgmuvi.models import (
    AchromaticGPModel,
    WavelengthDependentGPModel,
    SeparableGPModel,
)
from pgmuvi.initialization import initialize_separable_from_data
from pgmuvi.synthetic import make_chromatic_sinusoid_2d

# ---------------------------------------------------------------------------
# 1. Generate synthetic multiwavelength data
# ---------------------------------------------------------------------------
true_period = 7.0
wavelengths = [450.0, 600.0, 750.0]  # nm

lc = make_chromatic_sinusoid_2d(
    n_per_band=50,
    period=true_period,
    amplitude=1.0,
    wavelengths=wavelengths,
    amplitude_law="linear",
    amplitude_slope=0.3 / 150.0,
    wl_ref=600.0,
    noise_level=0.1,
    t_span=25.0,
    irregular=True,
    seed=0,
)
x = lc.xdata
y = lc.ydata
t = x[:, 0]
wl = x[:, 1]

n_total = len(y)
print("=" * 60)
print("Synthetic multiwavelength data")
print(f"  n_total     = {n_total}")
print(f"  true period = {true_period}")
print(f"  bands       = {wavelengths}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2. Auto-initialise from data
# ---------------------------------------------------------------------------
init_params = initialize_separable_from_data(x, y)
print("\nAuto-initialized separable parameters:")
for k, v in init_params.items():
    if k != "periods_per_band":
        print(f"  {k}: {v}")
print(f"  periods per band: {[f'{p:.2f}' for p in init_params['periods_per_band']]}")
print(f"  is_achromatic: {init_params['is_achromatic']}")

# ---------------------------------------------------------------------------
# 3. Achromatic GP (wavelength-independent variability)
# ---------------------------------------------------------------------------
print("\n--- Achromatic GP (all wavelengths correlated) ---")
lik_a = gpytorch.likelihoods.GaussianLikelihood()
model_a = AchromaticGPModel(x, y, lik_a, time_kernel_type="matern")
n_params_a = sum(p.numel() for p in model_a.parameters())
print(f"  Number of trainable parameters: {n_params_a}")

# ---------------------------------------------------------------------------
# 4. Wavelength-Dependent GP (smooth wavelength correlation)
# ---------------------------------------------------------------------------
print("\n--- Wavelength-Dependent GP ---")
lik_wd = gpytorch.likelihoods.GaussianLikelihood()
model_wd = WavelengthDependentGPModel(
    x, y, lik_wd,
    time_kernel_type="matern",
    wavelength_lengthscale=init_params["wavelength_lengthscale"],
)
n_params_wd = sum(p.numel() for p in model_wd.parameters())
print(f"  Number of trainable parameters: {n_params_wd}")

# ---------------------------------------------------------------------------
# 5. Separable GP with quasi-periodic time kernel
# ---------------------------------------------------------------------------
print("\n--- Separable GP with quasi-periodic time kernel ---")
lik_s = gpytorch.likelihoods.GaussianLikelihood()
from pgmuvi.kernels import QuasiPeriodicKernel  # noqa: E402
from gpytorch.kernels import ScaleKernel, RBFKernel  # noqa: E402

qp = QuasiPeriodicKernel()
qp.period = init_params["period"]
time_kernel = ScaleKernel(qp)
wavelength_kernel = ScaleKernel(RBFKernel())

model_s = SeparableGPModel(
    x, y, lik_s,
    time_kernel=time_kernel,
    wavelength_kernel=wavelength_kernel,
)
n_params_s = sum(p.numel() for p in model_s.parameters())
print(f"  Number of trainable parameters: {n_params_s}")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Parameter count comparison (2D models)")
print("=" * 60)
print(f"  Achromatic GP                  : {n_params_a:3d} parameters")
print(f"  Wavelength-Dependent GP        : {n_params_wd:3d} parameters")
print(f"  Separable GP (QuasiPeriodic)   : {n_params_s:3d} parameters")
print()
print("(For reference, TwoDSpectralMixture with 4 mixtures = ~26 parameters)")
print("=" * 60)
