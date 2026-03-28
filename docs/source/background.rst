Background
==========

This page provides background on the mathematical and scientific concepts underlying ``pgmuvi``.
It is intended as a companion to the tutorials and how-to guides, and assumes familiarity
with basic statistics and time-series analysis but does not require deep knowledge of
Gaussian processes or Bayesian inference.

.. contents:: On this page
   :local:
   :depth: 2

Gaussian Processes
------------------

A Gaussian process (GP) is a probability distribution over functions.
Any finite collection of function values drawn from a GP follows a multivariate
Gaussian distribution.  A GP is fully specified by:

* a **mean function** :math:`m(\mathbf{x})`, which describes the expected value of the
  function at any input :math:`\mathbf{x}`, and
* a **covariance (kernel) function** :math:`k(\mathbf{x}, \mathbf{x}')`, which encodes
  how strongly two function values at inputs :math:`\mathbf{x}` and :math:`\mathbf{x}'`
  are correlated.

In ``pgmuvi``, the inputs :math:`\mathbf{x}` are observation times (and, optionally,
wavelengths for multiband data), and the function values are source fluxes or magnitudes.
The GP prior is updated with observed data to produce a posterior distribution, which
provides both predictions and rigorous uncertainty estimates.

Further reading:

* Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*
  (`free PDF <http://www.gaussianprocess.org/gpml/>`_)
* `GPyTorch documentation <https://docs.gpytorch.ai/>`_ (the GP library that
  ``pgmuvi`` is built on)

Spectral Mixture Kernels
------------------------

The choice of kernel function determines what kind of variability patterns the GP can
represent.  ``pgmuvi`` uses **Spectral Mixture Kernels** (SMK; Wilson & Adams 2013),
which are a highly flexible family of kernels capable of capturing quasi-periodic
signals, coloured noise, and multiple simultaneous periodicities.

A Spectral Mixture Kernel represents the power spectral density (PSD) of the GP as
a mixture of Gaussians in the frequency domain:

.. math::

   S(\omega) = \sum_{q=1}^{Q} w_q \,
       \mathcal{N}(\omega \,|\, \mu_q, \sigma_q^2)

where :math:`Q` is the number of mixture components (``num_mixtures``), and each
component has weight :math:`w_q`, centre frequency :math:`\mu_q` (which corresponds
to a period :math:`1/\mu_q`), and bandwidth :math:`\sigma_q` (which controls how
quickly coherence is lost over time).

This parameterisation means that ``pgmuvi`` directly infers the *shape* of the PSD
rather than assuming a rigid functional form (e.g., a pure sinusoid or a power law).

Reference:

* Wilson, A. G. & Adams, R. P. (2013), *Gaussian Process Kernels for Pattern
  Discovery and Extrapolation*, ICML. `arXiv:1302.4245 <https://arxiv.org/abs/1302.4245>`_

Power Spectral Density
----------------------

The **power spectral density** (PSD) describes how the variance of a time series is
distributed across frequencies.  A peak in the PSD at frequency :math:`\nu` indicates
quasi-periodic variability with period :math:`P = 1/\nu`.  Broad, low-frequency power
is characteristic of red noise (stochastic variability that is correlated on long
timescales), while narrow peaks indicate coherent oscillations.

Because ``pgmuvi`` infers the PSD non-parametrically via a spectral mixture kernel,
it can simultaneously represent:

* one or more quasi-periodic signals,
* aperiodic stochastic variability (red noise),
* white noise (through the GP likelihood),
* long-term trends (through the mean function).

Multiwavelength Variability Inference
--------------------------------------

Many astrophysical sources are observed simultaneously at multiple wavelengths or in
multiple photometric bands.  ``pgmuvi`` supports **2D Gaussian processes** where the
inputs :math:`\mathbf{x} = (t, \lambda)` include both time and wavelength.  The
kernel function is then a product of a temporal kernel and a wavelength kernel,
allowing the GP to learn:

* the temporal variability structure (periodicities, noise),
* how variability amplitude and coherence vary with wavelength.

This is particularly useful when different bands are observed at different cadences or
have different noise levels — the GP naturally handles irregular and asynchronous
sampling.

Bayesian Parameter Inference
-----------------------------

``pgmuvi`` supports two modes of parameter estimation:

1. **Optimisation (MAP estimation):** The hyperparameters are optimised to maximise
   the (marginal) log likelihood.  This is fast and suitable for exploratory analysis
   or when the posterior is well-behaved.  Use :meth:`~pgmuvi.lightcurve.Lightcurve.fit`.

2. **MCMC sampling:** Full posterior samples are drawn using Hamiltonian Monte Carlo
   via Pyro/NumPyro.  This provides calibrated uncertainties and reveals correlations
   between parameters.  Use :meth:`~pgmuvi.lightcurve.Lightcurve.mcmc`.

For most use cases, MAP estimation followed by MCMC on parameters of interest is a
practical workflow.

Initialisation with Lomb–Scargle
---------------------------------

GP optimisation can be sensitive to the starting point.  ``pgmuvi`` uses the
**Lomb–Scargle periodogram** (and its multiband extension) to initialise the
spectral mixture kernel frequencies before fitting.  This substantially improves
convergence and reduces the risk of settling in a poor local optimum.

The initialisation step is exposed via
:meth:`~pgmuvi.lightcurve.Lightcurve.init_hypers_from_LombScargle` and is also
invoked automatically by :meth:`~pgmuvi.lightcurve.Lightcurve.fit_LS`.
