# Bayesian Generalized Lomb–Scargle in PGMUVI: method, assumptions, and scope

## Purpose

This document explains the Torch-based BGLS implementation in `pgmuvi/bgls.py` and, just as importantly, the assumptions under which it should and should not be interpreted as statistically reliable.

The implementation is designed to be **insulated** from the rest of PGMUVI.  It can be used directly on arrays, or it can be called from a `Lightcurve` object via a thin wrapper.

## Literature basis

The implementation draws on four distinct pieces of literature:

1. **Classical and generalized Lomb–Scargle**
   - Lomb (1976)
   - Scargle (1982)
   - Zechmeister & Kürster (2009)

2. **Bayesian GLS / BGLS**
   - Mortier et al. (2015)

3. **Shared-period multiband logic**
   - VanderPlas & Ivezić (2015)

4. **Composite likelihood and Bayesian adjustment**
   - Lindsay (1988)
   - Varin, Reid & Firth (2011)
   - Ribatet, Cooley & Davison (2012)

These references are included in `docs/bgls_references.bib`.

## Single-band model

For a single band, at each trial frequency \(f\), the code fits

\[
y(t_i) = a \cos(2\pi f t_i) + b \sin(2\pi f t_i) + c + \epsilon_i,
\]

with independent Gaussian errors

\[
\epsilon_i \sim \mathcal{N}(0, \sigma_i^2).
\]

This is the weighted floating-mean sinusoid model of Zechmeister & Kürster (2009).  In matrix form,

\[
\mathbf{y} = X_f \boldsymbol{\beta} + \boldsymbol{\epsilon},
\]

where

\[
X_f =
\begin{bmatrix}
\cos(2\pi f t_1) & \sin(2\pi f t_1) & 1 \\
\vdots & \vdots & \vdots \\
\cos(2\pi f t_n) & \sin(2\pi f t_n) & 1
\end{bmatrix},
\qquad
\boldsymbol{\beta} = (a,b,c)^T.
\]

With \(W = \mathrm{diag}(\sigma_i^{-2})\), the code computes the Gaussian linear-model marginal likelihood obtained by analytically integrating over \(a\), \(b\), and \(c\) under flat priors.  Up to a frequency-independent constant,

\[
\log p(\mathbf{y}\mid f)
=
\frac{1}{2}\mathbf{b}_f^T G_f^{-1}\mathbf{b}_f
-
\frac{1}{2}\log |G_f|,
\]

with

\[
G_f = X_f^T W X_f,
\qquad
\mathbf{b}_f = X_f^T W \mathbf{y}.
\]

This is the computational core of `bgls_single_band()`.

## What the single-band posterior means

The code then applies a prior over frequency and normalizes:

\[
p(f \mid \mathbf{y}) \propto p(\mathbf{y}\mid f)\,p(f).
\]

The default prior in the implementation is **log-uniform** in frequency, because the comparison is usually made across decades in period or frequency.  The implementation also supports a frequency-uniform prior and a truncated normal prior in frequency space.

A sharp posterior peak means the sinusoid-plus-offset model strongly prefers that frequency relative to other trial frequencies.  It does **not** prove that the source is physically sinusoidal.

## Extension to multiband data

Suppose the light curve has bands or wavelengths indexed by \(k=1,\dots,K\).  The module computes a separate BGLS curve for each band:

\[
p(D_k \mid f),
\]

where each band has its own marginalized coefficients \(a_k\), \(b_k\), and \(c_k\), but the trial frequency \(f\) is shared across bands.

This mirrors the **shared-period** logic that underlies multiband periodograms such as VanderPlas & Ivezić (2015): the periodicity is common, while the band-specific response may differ.

## Why the multiband combination is not a full Bayesian multiband model

The implementation does **not** fit a single joint model with explicit cross-band covariance.  Instead, it combines the per-band BGLS curves using a **tempered composite likelihood**,

\[
\log p(f \mid D_{1:K})
=
\log p(f)
+
T \sum_{k=1}^{K} w_k \log p(D_k \mid f)
+
C,
\]

where:

- \(w_k\) are per-band weights,
- \(T\) is a global temperature,
- \(C\) is a normalization constant.

This approximation is motivated by the composite-likelihood literature:
Lindsay (1988) introduced the formal composite-likelihood framework,
Varin et al. (2011) review its statistical properties,
and Ribatet et al. (2012) discuss Bayesian adjustment when a composite likelihood is inserted into Bayes' formula.

This matters because a naive product across bands can be **overconfident** if the bands are not conditionally independent.

## Why tempering is needed

If you simply multiply all per-band likelihoods, then any duplicated or strongly correlated evidence across bands can make the posterior artificially sharp.  In real multiband time series, this can happen because:

- observations in different bands share cadence structure,
- reduction systematics may be correlated,
- the intrinsic variability is often coupled across wavelength.

The global temperature \(T\) and the band weights \(w_k\) are there to reduce this overconfidence.

## Weight options implemented

### Option A: equal weights

All bands receive the same weight:

\[
w_k = 1.
\]

This is the simplest and cheapest option.  It is appropriate when the bands are reasonably similar in information content and no fast adaptive weighting is desired.

### Option B: fast information-based weights (**default**)

The code uses a cheap proxy for information content based on:

- inverse-variance weight sum, and
- a heuristic effective sample size based on temporal clustering.

This option is used by default because it is faster than option C and still downweights low-information bands.

### Option C: curvature-based weights

The code estimates how sharply peaked each band's log-posterior is near its dominant peak and uses this as an information proxy.

This is usually more adaptive than option B, but it is slower because it depends on the full per-band BGLS curves.

## Calibration options implemented

### Leave-one-band-out (**default**)

This is the default because it is the fastest calibration that still reacts to cross-band redundancy.

The code compares the curvature of the full composite posterior with the median curvature of the composites formed by leaving one band out at a time, then sets the global temperature \(T\) to reduce the discrepancy.

### Block bootstrap

For each band, the code applies a circular moving-block bootstrap in time, recomputes the per-band BGLS curves, recombines them, and looks at the spread of the resulting peak frequencies.

This is slower but better reflects temporal dependence within bands.

### Nested band-level/time-level resampling

This is the most expensive option in the module.  It resamples both:
- whole bands (with replacement), and
- time blocks within each selected band.

It is the most direct stress test of both cross-band redundancy and within-band dependence.

## Shared frequency grid

The implementation uses a **shared logarithmic frequency grid**.

This is deliberate.  The multiband composite is only meaningful if every band is evaluated on the same grid.  The default builder estimates the range from:

- the total baseline, and
- the median positive cadence,

then constructs the grid in log space.

## Assumptions that must be stated explicitly

The implementation assumes:

1. **Single-sinusoid model in each band.**
   If the source is strongly non-sinusoidal, the method may still find the correct period, but the posterior shape is conditional on a misspecified signal model.

2. **Gaussian independent errors within each band.**
   This is required for the analytic marginalization implemented in the code.

3. **Shared frequency across bands.**
   This is appropriate when the physical periodicity is common across wavelength.

4. **No explicit cross-band covariance model.**
   The multiband combination is a tempered composite approximation, not a full hierarchical Bayesian model.

5. **Common frequency grid.**
   All per-band curves must be evaluated on the same trial frequencies before combination.

## What this implementation is good for

It is well suited for:

- comparing dominant frequencies/periods across bands,
- constructing a combined posterior over a shared frequency,
- comparing multiband BGLS peak structure with LS or multiband LS results,
- generating a fast Bayesian frequency summary to compare with GP-based PSD structure.

## What this implementation is not

It is **not**:

- a Gaussian-process period model,
- a quasi-periodic stochastic model,
- a multiharmonic joint multiband Bayesian light-curve model,
- a full treatment of cross-band covariance.

Those distinctions need to be kept explicit in any scientific write-up.

## Practical interpretation guidance

When comparing this implementation to LS or GP outputs:

- compare **peak locations**,
- compare **relative peak rankings**,
- compare **alias structure**,
- compare **posterior concentration / peak width**,

but do **not** equate the BGLS posterior ordinate directly with LS power or GP PSD amplitude.

They are different statistical objects.

## Summary

The Torch implementation in `pgmuvi/bgls.py` is:

- exact for the single-band weighted sinusoid-plus-offset Gaussian model,
- approximate in the multiband combination step,
- intentionally modular and insulated from the rest of PGMUVI,
- designed for speed and direct integration into the existing workflow.

For a publication, the key sentence should be:

> We computed per-band Bayesian generalized Lomb–Scargle periodograms and combined them using a tempered composite likelihood on a shared logarithmic frequency grid, with band-specific sinusoid parameters marginalized independently and a shared trial frequency across bands.

