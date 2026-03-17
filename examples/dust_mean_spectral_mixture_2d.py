"""Example: Dust-mean GP with spectral-mixture time kernel for AGB stars.

This script demonstrates how to use a 2D ``Lightcurve`` object to fit
multiwavelength photometry of a dust-obscured AGB star.  The model combines:

* **Mean function** - DustMean:
  ``m(t, wl) = A * exp(-tau * wl**(-alpha)) + offset``
  This captures the strong wavelength dependence introduced by circumstellar
  dust, which attenuates optical flux far more than infrared flux.

* **Covariance (time)** - ``SpectralMixtureKernel``:
  Learns the power-spectral-density of the stochastic variability, allowing
  flexible non-parametric period and amplitude recovery.

* **Covariance (wavelength)** - ``ScaleKernel(RBFKernel())``:
  Captures smooth correlation in residual flux across nearby wavelengths.

The complete model is accessed via the ``'2DWavelengthDependent'`` shortcut
with ``mean_module='dust'`` and ``time_kernel_type='sm'``, or equivalently
via the ``'2DDustMean'`` shortcut with ``time_kernel_type='sm'``.

Usage::

    python examples/dust_mean_spectral_mixture_2d.py
"""

import math

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so the script runs headless
import matplotlib.pyplot as plt
import numpy as np

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.synthetic import make_multi_sinusoid_chromatic_2d

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
seed = 0

print("=" * 70)
print("Dust-mean GP with Spectral-Mixture time kernel (2D Lightcurve)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Generate synthetic AGB-star photometry
#
# Physical setup:
#   - Mira-type pulsator with a ~400-day period
#   - Observed in three wavelength bands: optical (I, ~0.8 um),
#     near-IR (J, ~1.2 um) and near-IR (K, ~2.2 um)
#   - Circumstellar dust shell introduces wavelength-dependent attenuation
#     following  flux ~ exp(-tau * wavelength^(-alpha))
#
# The dust-law parameters used to generate the data are:
#   amplitude = 5.0 (arbitrary flux units)
#   tau       = 2.0 (optical depth scale)
#   alpha     = 1.7 (typical ISM-like extinction law index)
#   offset    = 0.2 (background)
# ---------------------------------------------------------------------------

print("\n1. Generating synthetic AGB-star multiwavelength light curve...")

TRUE_PERIOD = 400.0          # days  (Mira-like pulsation)
TRUE_AMPLITUDE = 5.0         # arbitrary flux units
TRUE_TAU = 2.0               # dust optical depth
TRUE_ALPHA = 1.7             # extinction power-law index
TRUE_OFFSET = 0.2            # background offset
NOISE_FRAC = 0.05            # 5 % fractional noise per band

# Three photometric bands: I (0.8 µm), J (1.2 µm), K (2.2 µm)
BANDS = {"I (0.8 µm)": 0.8, "J (1.2 µm)": 1.2, "K (2.2 µm)": 2.2}
N_PER_BAND = 60              # observations per band

# Use make_multi_sinusoid_chromatic_2d with dust extinction amplitude law.
# The two sinusoidal components mimic the fundamental pulsation period and
# its first harmonic.
lc = make_multi_sinusoid_chromatic_2d(
    n_per_band=N_PER_BAND,
    t_span=3 * TRUE_PERIOD,
    components=[
        {"period": TRUE_PERIOD, "amplitude_fraction": 0.4, "phase": 0.0},
        {"period": TRUE_PERIOD / 2, "amplitude_fraction": 0.1,
         "phase": math.pi / 2},
    ],
    wavelengths=list(BANDS.values()),
    amplitude_law="extinction",
    overall_amplitude=TRUE_AMPLITUDE,
    tau=TRUE_TAU,
    alpha=TRUE_ALPHA,
    offset=TRUE_OFFSET,
    phase_law="none",
    noise_level=NOISE_FRAC * TRUE_AMPLITUDE * 0.5,
    irregular=True,
    seed=seed,
)
xdata = lc.xdata
ydata = lc.ydata

# Report band statistics (approximate, since offset now included in y)
band_masks = {}
offset_idx = 0
for band_name, wl_micron in BANDS.items():
    mean_flux = TRUE_AMPLITUDE * np.exp(-TRUE_TAU * wl_micron ** -TRUE_ALPHA)
    mean_flux += TRUE_OFFSET
    print(
        f"   {band_name:16s}: mean flux ~ {mean_flux:.3f},  "
        f"variability amplitude ~ {0.4 * mean_flux:.3f}"
    )
    band_masks[band_name] = slice(offset_idx, offset_idx + N_PER_BAND)
    offset_idx += N_PER_BAND

t_np = xdata[:, 0].numpy()
wl_np = xdata[:, 1].numpy()
y_np = ydata.numpy()

print(f"\n   Total observations  : {len(ydata)}")
print(f"   Time span           : {t_np.min():.0f} - {t_np.max():.0f} days")
print(f"   Wavelength range    : {wl_np.min():.1f} - {wl_np.max():.1f} µm")

# ---------------------------------------------------------------------------
# 2. Create a 2D Lightcurve object
# ---------------------------------------------------------------------------

print("\n2. Creating 2D Lightcurve object...")

lc = Lightcurve(xdata, ydata)
print(f"   Data dimensionality  : {lc.ndim}D")
print(f"   Transformed x shape  : {lc._xdata_transformed.shape}")
print(f"   X transform type     : {type(lc.xtransform).__name__}")

# ---------------------------------------------------------------------------
# 3. Fit the model
#
# We use the '2DWavelengthDependent' model shortcut with:
#   mean_module='dust'      → DustMean  (circumstellar dust law)
#   time_kernel_type='sm'   → SpectralMixtureKernel for temporal variability
#   wavelength_kernel_type  → default 'rbf' (smooth wavelength correlation)
#   num_mixtures=4          → 4 spectral-mixture components
#
# Equivalently, '2DDustMean' with time_kernel_type='sm' gives the same model.
# ---------------------------------------------------------------------------

print("\n3. Fitting dust-mean GP with spectral-mixture time kernel...")
print("   Model      : WavelengthDependentGPModel")
print("   Mean       : DustMean  (m(t,λ) = A·exp(-τ·λ^-α) + offset)")
print("   Time cov   : SpectralMixtureKernel (num_mixtures=4)")
print("   Wl cov     : ScaleKernel(RBFKernel())")

results = lc.fit(
    model="2DWavelengthDependent",
    mean_module="dust",
    time_kernel_type="sm",
    wavelength_kernel_type="rbf",
    num_mixtures=4,
    training_iter=300,
    miniter=100,
    lr=0.05,
    stop=1e-5,
)

print("   Training complete!")
print(f"   Final loss : {float(results['loss'][-1]):.4f}")

# ---------------------------------------------------------------------------
# 4. Inspect fitted dust-mean parameters
#
# Note: the Lightcurve applies a MinMax transform to (time, wavelength) before
# fitting, so the dust-mean parameters are fitted in the normalized wavelength
# coordinate (0 → λ_min = 0.8 µm, 1 → λ_max = 2.2 µm).  The amplitude,
# tau, and alpha printed below are therefore *effective* values in that
# normalized space, not physical µm values.
# ---------------------------------------------------------------------------

print("\n4. Fitted DustMean parameters (in normalized wavelength space):")

mean_mod = lc.model.mean_module
amplitude = mean_mod.log_amplitude.exp().item()
tau = mean_mod.log_tau.exp().item()
alpha = mean_mod.log_alpha.exp().item()
offset_val = mean_mod.offset.item()

print(f"   amplitude = {amplitude:.4f}  (raw parameter in normalized space)")
print(f"   tau       = {tau:.4f}  (raw parameter in normalized space)")
print(f"   alpha     = {alpha:.4f}  (raw parameter in normalized space)")
print(f"   offset    = {offset_val:.4f}  (raw parameter in normalized space)")

# ---------------------------------------------------------------------------
# 5. Visualize
# ---------------------------------------------------------------------------

print("\n5. Creating plots...")

# Use a 2×3 grid: top row = training loss + all-band overview,
# bottom row = one panel per wavelength band.
fig = plt.figure(figsize=(15, 9))
fig.suptitle(
    "Dust-mean GP with Spectral-Mixture time kernel\n"
    r"(mean: $A\,e^{-\tau\lambda^{-\alpha}}$ + offset,  "
    "cov: SpectralMixture × RBF)",
    fontsize=12,
)

ax_loss = fig.add_subplot(2, 3, 1)
ax_data = fig.add_subplot(2, 3, 2)
# leave grid position (2,3,3) empty to balance the layout
band_axes = [fig.add_subplot(2, 3, 4 + i) for i in range(len(BANDS))]

# ---- Panel 1: Training loss ----
losses = [float(v) for v in results["loss"]]
ax_loss.plot(losses, "b-", lw=1.5)
ax_loss.set_xlabel("Iteration")
ax_loss.set_ylabel("Negative log-likelihood")
ax_loss.set_title("Training loss")
ax_loss.grid(True, alpha=0.3)

# ---- Panel 2: Raw data coloured by band ----
colors = {
    "I (0.8 µm)": "firebrick",
    "J (1.2 µm)": "goldenrod",
    "K (2.2 µm)": "steelblue",
}
for band_name, sl in band_masks.items():
    ax_data.scatter(
        t_np[sl], y_np[sl],
        c=colors[band_name], s=15, alpha=0.6,
        label=band_name,
    )
ax_data.set_xlabel("Time (days)")
ax_data.set_ylabel("Flux (arbitrary units)")
ax_data.set_title("Observed multiwavelength photometry")
ax_data.legend(fontsize=8)
ax_data.grid(True, alpha=0.3)

# ---- Panels 3-5: GP predictions per band ----
lc.model.eval()
lc.likelihood.eval()

t_pred = torch.linspace(float(t_np.min()), float(t_np.max()), 400, dtype=torch.float32)

for ax, (band_name, wl_micron) in zip(band_axes, BANDS.items()):
    # Build prediction inputs in the original (untransformed) space and
    # apply the same MinMax transform that Lightcurve applied to training data.
    wl_array = torch.full_like(t_pred, wl_micron)
    x_raw = torch.stack([t_pred, wl_array], dim=1)
    x_pred = lc.xtransform.transform(x_raw)

    with torch.no_grad():
        pred = lc.likelihood(lc.model(x_pred))
        mean = pred.mean.numpy()
        lo = (pred.mean - 2 * pred.stddev).numpy()
        hi = (pred.mean + 2 * pred.stddev).numpy()

    sl = band_masks[band_name]
    ax.scatter(
        t_np[sl], y_np[sl],
        c=colors[band_name], s=15, alpha=0.5,
        label="Observations",
    )
    ax.plot(t_pred.numpy(), mean, color=colors[band_name], lw=1.8, label="GP mean")
    ax.fill_between(
        t_pred.numpy(), lo, hi,
        color=colors[band_name], alpha=0.2, label="95 % CI",
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux (arbitrary units)")
    ax.set_title(f"GP fit – {band_name}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
outfile = "dust_mean_spectral_mixture_2d.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"   Plot saved to '{outfile}'")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("✓  2D Lightcurve fitted with DustMean + SpectralMixture time kernel")
print(f"✓  Fitted amplitude = {amplitude:.4f}  (in normalized wavelength space)")
print(f"✓  Fitted tau       = {tau:.4f}  (in normalized wavelength space)")
print(f"✓  Fitted alpha     = {alpha:.4f}  (in normalized wavelength space)")
print()
print("Key points:")
print("  • mean_module='dust' → DustMean captures the dust-attenuation SED")
print("  • time_kernel_type='sm' → SpectralMixture learns temporal variability")
print("  • The same model can be selected via '2DDustMean' + time_kernel_type='sm'")
print("  • num_mixtures controls the flexibility of the PSD approximation")
print("=" * 70)
