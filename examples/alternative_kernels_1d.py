"""Example: Alternative 1D kernels for periodic astronomical time series.

This script demonstrates how to use the quasi-periodic, Matérn, and
periodic+stochastic GP models as simpler alternatives to the default
Spectral Mixture kernel.

Usage::

    python examples/alternative_kernels_1d.py
"""

import numpy as np
import torch
import gpytorch

from pgmuvi.models import (
    QuasiPeriodicGPModel,
    MaternGPModel,
    PeriodicPlusStochasticGPModel,
)
from pgmuvi.initialization import initialize_quasi_periodic_from_data

# ---------------------------------------------------------------------------
# 1. Generate synthetic pulsating-star data
# ---------------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

n_obs = 80
true_period = 5.0

t = torch.sort(torch.rand(n_obs) * 20.0)[0].float()
y_clean = torch.sin(2 * np.pi * t / true_period)
noise_level = 0.15
y_noisy = (y_clean + noise_level * torch.randn(n_obs)).float()
yerr = noise_level * torch.ones(n_obs)

print("=" * 60)
print("Synthetic pulsating-star data")
print(f"  n_obs      = {n_obs}")
print(f"  true period = {true_period}")
print(f"  noise level = {noise_level}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2. Initialise hyperparameters from the periodogram
# ---------------------------------------------------------------------------
init_params = initialize_quasi_periodic_from_data(t, y_noisy, yerr=yerr)
print("\nAuto-initialized hyperparameters from Lomb-Scargle:")
for k, v in init_params.items():
    print(f"  {k}: {v:.4f}")

# ---------------------------------------------------------------------------
# 3. Quasi-Periodic GP
# ---------------------------------------------------------------------------
print("\n--- Quasi-Periodic GP ---")
lik_qp = gpytorch.likelihoods.GaussianLikelihood()
model_qp = QuasiPeriodicGPModel(
    t, y_noisy, lik_qp, period=init_params["period"]
)
n_params_qp = sum(p.numel() for p in model_qp.parameters())
print(f"  Number of trainable parameters: {n_params_qp}")
print(f"  Initial period: {float(model_qp.covar_module.base_kernel.period):.4f}")

# ---------------------------------------------------------------------------
# 4. Matérn GP (stochastic)
# ---------------------------------------------------------------------------
print("\n--- Matern GP (nu=1.5) ---")
lik_m = gpytorch.likelihoods.GaussianLikelihood()
model_m = MaternGPModel(t, y_noisy, lik_m, nu=1.5)
n_params_m = sum(p.numel() for p in model_m.parameters())
print(f"  Number of trainable parameters: {n_params_m}")

# ---------------------------------------------------------------------------
# 5. Periodic + Stochastic GP
# ---------------------------------------------------------------------------
print("\n--- Periodic + Stochastic GP ---")
lik_ps = gpytorch.likelihoods.GaussianLikelihood()
model_ps = PeriodicPlusStochasticGPModel(
    t, y_noisy, lik_ps, period=init_params["period"]
)
n_params_ps = sum(p.numel() for p in model_ps.parameters())
print(f"  Number of trainable parameters: {n_params_ps}")

# ---------------------------------------------------------------------------
# 6. Comparison summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Parameter count comparison")
print("=" * 60)
print(f"  QuasiPeriodic GP    : {n_params_qp:3d} parameters")
print(f"  Matern GP           : {n_params_m:3d} parameters")
print(f"  Periodic+Stochastic : {n_params_ps:3d} parameters")
print()
print("(For reference, SpectralMixture with 4 mixtures = ~14 parameters in 1D)")
print("=" * 60)
