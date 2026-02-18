"""
Example: 2D Multiwavelength Light Curve Fitting

This example demonstrates how to use pgmuvi to fit Gaussian Process models
to multiwavelength (2D) astronomical light curve data. The package can now
handle data with two independent variables:
  - Dimension 0: Time
  - Dimension 1: Wavelength/Band identifier

This is particularly useful for modeling variable sources observed in multiple
wavelength bands with different sampling patterns.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pgmuvi.lightcurve import Lightcurve

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("2D Multiwavelength Light Curve Example")
print("=" * 70)

# ============================================================================
# Step 1: Generate Synthetic 2D Multiwavelength Data
# ============================================================================

print("\n1. Generating synthetic 2D multiwavelength data...")

# Parameters for synthetic data
n_samples_band1 = 150  # Blue band - dense sampling
n_samples_band2 = 100  # Red band - sparser sampling

# Time arrays for two wavelength bands (in days)
time_band1 = np.sort(np.random.uniform(0, 100, n_samples_band1))
time_band2 = np.sort(np.random.uniform(0, 100, n_samples_band2))

# Wavelength identifiers (normalized to [0, 1] range)
# These represent actual wavelength bands:
#   0.45 corresponds to blue band (~450nm)
#   0.65 corresponds to red band (~650nm)
wavelength_band1 = np.ones(n_samples_band1) * 0.45  # Blue band
wavelength_band2 = np.ones(n_samples_band2) * 0.65  # Red band

# Combine data from both bands
time_all = np.concatenate([time_band1, time_band2])
wavelength_all = np.concatenate([wavelength_band1, wavelength_band2])

# Stack into (n_samples, 2) format for 2D data
# Column 0: time, Column 1: wavelength
xdata_2d = torch.tensor(np.column_stack([time_all, wavelength_all]),
                        dtype=torch.float32)

print(f"   - Total samples: {len(xdata_2d)}")
print(f"   - Blue band: {n_samples_band1} observations")
print(f"   - Red band: {n_samples_band2} observations")
print(f"   - Time span: {time_all.min():.1f} to {time_all.max():.1f} days")

# ============================================================================
# Step 2: Generate Signal with Known Properties
# ============================================================================

print("\n2. Creating variable source signal...")

# Known periodic signal parameters
true_period = 12.5  # days
true_freq = 1.0 / true_period

# Base achromatic signal (same in all bands)
base_signal = np.sin(2 * np.pi * time_all / true_period)

# Wavelength-dependent amplitude modulation
# Blue band has higher amplitude variation
amplitude_factor = 1.0 + 0.5 * (wavelength_all - 0.45) / 0.2

# Combine signal with realistic noise
signal = amplitude_factor * base_signal
noise_level = 0.15
noise = np.random.randn(len(signal)) * noise_level

ydata_2d = torch.tensor(signal + noise, dtype=torch.float32)

print(f"   - True period: {true_period:.2f} days")
print(f"   - True frequency: {true_freq:.4f} day⁻¹")
print(f"   - Noise level: {noise_level:.2f}")
print("   - Wavelength-dependent amplitude: Yes")

# ============================================================================
# Step 3: Create Lightcurve Object
# ============================================================================

print("\n3. Creating Lightcurve object with 2D data...")

lightcurve = Lightcurve(xdata_2d, ydata_2d)

print(f"   - Data dimensionality: {lightcurve.ndim}D")
print(f"   - Transformed data shape: {lightcurve._xdata_transformed.shape}")
print(f"   - Transform applied: {type(lightcurve.xtransform).__name__}")

# ============================================================================
# Step 4: Set Up and Fit 2D Model
# ============================================================================

print("\n4. Setting up 2D Spectral Mixture GP model...")

# Fit the model
# Use '2D' model for multiwavelength data
# num_mixtures controls model flexibility (more = more complex)
results = lightcurve.fit(
    model='2D',  # Critical: Use 2D model for multiwavelength data
    likelihood=None,  # Uses default Gaussian likelihood
    num_mixtures=3,  # Number of spectral mixture components
    training_iter=100,  # Training iterations
    miniter=50,  # Minimum iterations before early stopping
    lr=0.1,  # Learning rate
    stop=1e-4  # Early stopping threshold
)

print("   - Model type: 2D Spectral Mixture GP")
print("   - Number of mixture components: 3")
print("   - Training completed!")

# ============================================================================
# Step 5: Check Fitted Parameters
# ============================================================================

print("\n5. Inspecting fitted parameters...")

# Get the fitted mixture means (frequencies)
mixture_means = lightcurve.model.covar_module.mixture_means.detach()

print(f"   - Mixture means shape: {mixture_means.shape}")
print("   - (num_mixtures, batch, ard_num_dims)")

# Extract frequencies for time dimension (dimension 0)
time_frequencies = mixture_means[:, 0, 0].numpy()

print("\n   Fitted frequencies (time dimension):")
for i, freq in enumerate(time_frequencies):
    period = 1.0 / freq if freq > 0 else float('inf')
    print(f"      Component {i+1}: f={freq:.4f} day⁻¹, P={period:.2f} days")

# Check if we recovered the true period
closest_period = min([1/f for f in time_frequencies if f > 0],
                     key=lambda x: abs(x - true_period))
print(
    f"\n   Closest to true period ({true_period:.2f} days): "
    f"{closest_period:.2f} days"
)
rel_error = abs(closest_period - true_period) / true_period * 100
print(f"   Relative error: {rel_error:.1f}%")

# ============================================================================
# Step 6: Visualize Results
# ============================================================================

print("\n6. Creating visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('2D Multiwavelength Light Curve Fitting Results', fontsize=16)

# ---- Plot 1: Training Loss ----
ax = axes[0, 0]
losses = results['loss']
ax.plot([float(loss) for loss in losses], 'b-', alpha=0.7)
ax.set_xlabel('Iteration')
ax.set_ylabel('Negative Log Likelihood')
ax.set_title('Training Loss')
ax.grid(True, alpha=0.3)

# ---- Plot 2: Data by Wavelength Band ----
ax = axes[0, 1]

# Separate data by band
blue_mask = wavelength_all < 0.5
red_mask = wavelength_all >= 0.5

time_blue = time_all[blue_mask]
time_red = time_all[red_mask]
ydata_blue = ydata_2d.numpy()[blue_mask]
ydata_red = ydata_2d.numpy()[red_mask]

ax.scatter(time_blue, ydata_blue, c='blue', alpha=0.5, s=20,
           label='Blue band (450nm)')
ax.scatter(time_red, ydata_red, c='red', alpha=0.5, s=20,
           label='Red band (650nm)')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flux (arbitrary units)')
ax.set_title('Observed Data by Wavelength Band')
ax.legend()
ax.grid(True, alpha=0.3)

# ---- Plot 3: Fitted Model (Blue Band) ----
ax = axes[1, 0]

# Make predictions for blue band
lightcurve.model.eval()
lightcurve.likelihood.eval()

test_time = torch.linspace(0, 100, 500, dtype=torch.float32)
test_wavelength_blue = torch.ones(500, dtype=torch.float32) * 0.45
test_x_blue = torch.stack([test_time, test_wavelength_blue], dim=1)

with torch.no_grad():
    predictions_blue = lightcurve.likelihood(lightcurve.model(test_x_blue))
    mean_blue = predictions_blue.mean.numpy()
    std_blue = predictions_blue.stddev.numpy()

ax.scatter(time_blue, ydata_blue, c='blue', alpha=0.5, s=20,
           label='Observations')
ax.plot(test_time.numpy(), mean_blue, 'b-', linewidth=2, label='GP Mean')
ax.fill_between(test_time.numpy(),
                mean_blue - 2*std_blue,
                mean_blue + 2*std_blue,
                alpha=0.3, color='blue', label='95% CI')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flux (arbitrary units)')
ax.set_title('Fitted Model - Blue Band (450nm)')
ax.legend()
ax.grid(True, alpha=0.3)

# ---- Plot 4: Fitted Model (Red Band) ----
ax = axes[1, 1]

# Make predictions for red band
test_wavelength_red = torch.ones(500, dtype=torch.float32) * 0.65
test_x_red = torch.stack([test_time, test_wavelength_red], dim=1)

with torch.no_grad():
    predictions_red = lightcurve.likelihood(lightcurve.model(test_x_red))
    mean_red = predictions_red.mean.numpy()
    std_red = predictions_red.stddev.numpy()

ax.scatter(time_red, ydata_red, c='red', alpha=0.5, s=20,
           label='Observations')
ax.plot(test_time.numpy(), mean_red, 'r-', linewidth=2, label='GP Mean')
ax.fill_between(test_time.numpy(),
                mean_red - 2*std_red,
                mean_red + 2*std_red,
                alpha=0.3, color='red', label='95% CI')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Flux (arbitrary units)')
ax.set_title('Fitted Model - Red Band (650nm)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('2d_multiwavelength_example_results.png', dpi=150, bbox_inches='tight')
print("   - Figure saved as '2d_multiwavelength_example_results.png'")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("✓ Successfully fitted 2D GP model to multiwavelength data")
print(f"✓ Model captured periodic signal with period ≈ {closest_period:.2f} days")
print("✓ Default constraints were set automatically for 2D data")
print("✓ Model handles different sampling patterns in each band")
print("✓ Wavelength-dependent behavior captured by ARD structure")
print("\nKey features of 2D multiwavelength fitting:")
print("  • Automatic constraint setup for 2D parameters")
print("  • ARD (Automatic Relevance Determination) for time & wavelength")
print("  • Handles irregular sampling patterns per wavelength")
print("  • Captures correlations across wavelengths")
print("=" * 70)

plt.show()
