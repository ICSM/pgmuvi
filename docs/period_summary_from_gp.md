# Converting fitted GP kernel parameters into literature-comparable periods

## Purpose

The Gaussian-process fits in PGMUVI can return kernel hyperparameters that encode periodic or quasi-periodic structure in the light curve. These raw fitted parameters are not always directly comparable with published periods from the literature.

This document defines how to convert the raw fitted kernel parameters into physically interpretable summary quantities that can be compared with published periods, and how to attach a practical uncertainty estimate when full posterior sampling is not yet available.

The main point is simple:

* kernel parameters are not always the final astrophysical answer
* the quantity to compare with the literature is the dominant period implied by the fitted model

---

## What published periods usually represent

Published periods for variable stars are typically obtained from methods such as:

* Lomb–Scargle periodograms
* Fourier decomposition
* phase dispersion minimization
* direct phase folding

In practice, these methods try to identify the dominant variability timescale of the source.

Therefore, the period summary extracted from a GP fit should also represent the dominant variability timescale, rather than the internal decomposition parameters of the kernel.

---

## Relevant kernel classes in PGMUVI

PGMUVI currently supports several GP model classes. The period-summary procedure is different depending on the kernel family.

### 1. Spectral-mixture models

Examples:

* SpectralMixtureGPModel
* TwoDSpectralMixtureGPModel

These models encode periodic structure through:

* mixture_means
* mixture_scales
* mixture_weights

In these models, the periodicity is parameterized in frequency space. Each component is a basis function in the spectral density. Therefore, the individual component periods returned by get_periods() are not automatically the published-comparable periods.

---

### 2. Period-parameterized models

Examples:

* QuasiPeriodicGPModel
* LinearMeanQuasiPeriodicGPModel

These models typically contain a parameter such as period_length. In those cases, the fitted period parameter itself is the directly interpretable periodic quantity.

---

### 3. Non-periodic models

Example:

* MaternGPModel

These models do not contain a periodic parameter. No literature-comparable period should be reported from such a fit.

---

## Why get_periods() is not enough for spectral-mixture models

The existing method Lightcurve.get_periods() returns, for each spectral-mixture component:

* a component period
* a component weight
* a component scale

These are derived from mixture_means, mixture_scales, and mixture_weights.

A spectral-mixture GP represents the power spectrum as a sum of basis components. A real periodic or quasi-periodic signal can be approximated by several such components. Therefore:

* one astrophysical period can be represented by multiple component periods
* the individual returned periods can differ from the literature value even when the overall fit is excellent

---

## Correct summary quantity for spectral-mixture fits

For spectral-mixture kernels, the quantity that should be compared with the literature is:

P_dom = 1 / nu_peak

where nu_peak is the location of the dominant peak of the total positive-frequency PSD implied by the fitted kernel.

This is the GP analogue of the dominant period identified by a periodogram.

---

## Step 1: extract raw component parameters

### 1D spectral-mixture models

Use:

* mixture_means[i]
* mixture_scales[i]
* mixture_weights[i]

If no transform is applied:

P_i = 1 / mu_i
nu_i = mu_i
S_P_i = 1 / (2*pi*sigma_i)
S_nu_i = sigma_i

If xtransform is present:

P_i_raw = xtransform.inverse(1 / mu_i, shift=False)
nu_i_raw = 1 / P_i_raw
S_P_i_raw = xtransform.inverse(1 / (2*pi*sigma_i), shift=False)
S_nu_i_raw = 1 / (2*pi*S_P_i_raw)

---

### 2D spectral-mixture models

Use only the time dimension:

* mixture_means[i, 0]
* mixture_scales[i, 0]
* mixture_weights[i]

---

## Step 2: construct the total PSD

Define:

PSD(nu) = sum over i of
w_i * exp( -0.5 * ((nu - nu_i)/S_nu_i)^2 )

Only the shape matters; normalization is irrelevant.

---

## Step 3: identify the dominant PSD peak

1. Evaluate PSD on a frequency grid
2. Find local maxima
3. Select the highest peak

nu_peak = argmax PSD(nu)

Then:

P_dom = 1 / nu_peak

---

## Step 4: assign an uncertainty without MCMC

### Peak-width interval

Let PSD_max = PSD(nu_peak)

Find nu_L and nu_R such that:

PSD(nu) = 0.5 * PSD_max

Convert to period:

P_lo = 1 / nu_R
P_hi = 1 / nu_L

This is a coherence-based interval, not a posterior uncertainty.

---

### Quality factor

Define:

FWHM_nu = nu_R - nu_L
Q = nu_peak / FWHM_nu

Interpretation:

* large Q → well-defined periodicity
* small Q → weak or quasi-periodic

---

### Multipeak diagnostic

Count peaks satisfying:

PSD_peak >= f_thresh * PSD_max

Return:

* number of significant peaks
* their corresponding periods

---

## Optional empirical uncertainty

If repeated fits are feasible:

1. perturb or resample the data
2. refit
3. recompute P_dom
4. summarize the distribution

Label this as an empirical or bootstrap interval.

---

## Interpretation

### Clean periodic source

Single sharp peak → reliable period comparable to literature.

---

### Broad peak

Preferred timescale exists but is weakly coherent.

---

### Multiple peaks

Multiple candidate periods → do not report a single value without qualification.

---

## Comparison with literature

Compare only:

P_dom

Do NOT compare individual component periods.

---

## Recommended output fields

* component_periods
* component_weights
* component_period_scales
* component_frequencies
* component_frequency_scales
* freq_grid
* psd
* dominant_frequency
* dominant_period
* period_interval_fwhm_like
* q_factor
* peak_fraction
* n_significant_peaks
* significant_periods
* method
* notes

---

## Plotting recommendation

Plot:

* PSD vs frequency
* vertical line at nu_peak
* shaded region for half-maximum interval
* markers for other peaks
* annotation with period, interval, Q, number of peaks

---

## Current limitation

Without MCMC:

* period estimate is robust
* uncertainty is approximate (peak-width or bootstrap)

---

## Bottom line

* get_periods() returns kernel basis parameters
* dominant PSD peak defines the literature-comparable period
* uncertainty must currently be approximate
* diagnostics are essential to judge reliability
