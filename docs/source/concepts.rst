Key Concepts
============

This page explains the key concepts and terminology used throughout ``pgmuvi``.
It is a companion to the :doc:`background` page and the tutorials.

.. contents:: On this page
   :local:
   :depth: 2

The ``Lightcurve`` Object
--------------------------

The central object in ``pgmuvi`` is the :class:`~pgmuvi.lightcurve.Lightcurve` class.
It stores:

* the **observation times** (``xdata``),
* the **flux or magnitude measurements** (``ydata``),
* the **measurement uncertainties** (``yerr``),
* the **GP model** (kernel + likelihood),
* the **priors** and **constraints** on model parameters,
* and the **fitted or sampled parameters** after inference.

A typical workflow is::

    import pgmuvi
    lc = pgmuvi.lightcurve.Lightcurve(times, fluxes, errors)
    lc.fit_LS()              # optional: inspect candidate periods
    lc.fit(model="1D")       # initialise and optimise the GP model
    lc.plot()                # visualise the predictive distribution

Hyperparameters
---------------

GP hyperparameters control the shape of the kernel function and therefore the kind of
variability the model can represent.  For the default spectral mixture kernel the key
hyperparameters are:

``mixture_means`` (frequencies)
    The centre frequencies of the spectral mixture components, in units of
    1 / (time unit used for ``xdata``).  A peak at frequency :math:`\mu` corresponds
    to a quasi-period :math:`P = 1/\mu`.

``mixture_scales`` (bandwidths)
    The bandwidth (standard deviation) of each spectral component in the frequency
    domain.  Smaller bandwidth means a more coherent (longer-lived) oscillation;
    larger bandwidth means rapid decorrelation.

``mixture_weights`` (amplitudes)
    The relative weight of each spectral component, controlling the amplitude of
    variability associated with each frequency.

``noise`` / ``likelihood.noise``
    The white-noise variance added to the diagonal of the covariance matrix.
    Represents measurement noise and any rapid variability unresolved by the
    observations.

Priors and Constraints
-----------------------

``pgmuvi`` supports **priors** (Bayesian probability distributions placed on
hyperparameters) and **constraints** (hard bounds on the allowed parameter space).

Priors are used during MCMC sampling to regularise the posterior distribution.
Constraints are applied during both optimisation and MCMC to prevent the optimiser
from exploring physically unreasonable regions (e.g., negative noise variance or
periods longer than the observational baseline).

See :doc:`howto/priors_constraints` for practical guidance, and the
:mod:`pgmuvi.priors` and :mod:`pgmuvi.constraints` API reference for full details.

Data Transformations
---------------------

Raw observational data often span many orders of magnitude or have units that are
inconvenient for numerical optimisation.  ``pgmuvi`` provides built-in data
transformations:

* **MinMax:** rescales data to the range [0, 1].
* **ZScore:** standardises data to zero mean and unit variance.
* **RobustZScore:** standardises using the median and interquartile range, making it
  robust to outliers.

Transformations can be applied to the time axis (``xtransform``) or the flux/magnitude
axis (``ytransform``).  All predictions are automatically inverse-transformed back to
the original units when plotting or reporting results.

1D vs 2D Models
----------------

``pgmuvi`` supports two modes of operation:

* **1D (single-band):** The GP input is the time axis alone.  This is appropriate
  when data from a single photometric band or wavelength range are available.

* **2D (multiband):** The GP input is a two-column array ``[time, wavelength]``.
  The kernel is a product of a temporal kernel (spectral mixture) and a wavelength
  kernel, allowing the model to capture how variability changes across wavelengths.
  See :doc:`howto/multiband` for details.

Model Selection
----------------

``pgmuvi`` provides several GP models with different kernel structures:

* **Spectral Mixture (default):** Flexible, non-parametric PSD representation.
* **Spectral Mixture + RBF:** Adds a smooth long-term trend.
* **Spectral Mixture + Flicker:** Adds a :math:`1/f` noise component.
* **Periodic** and **Quasi-Periodic** kernels: Useful when a strict periodic signal
  is expected.

The :meth:`~pgmuvi.lightcurve.Lightcurve.auto_select_model` method can recommend a
model based on the data's characteristics (periodicity strength, inter-band
consistency).  See :doc:`howto/model_selection` for details.

Sampling Metrics and Data Quality
-----------------------------------

Before fitting, it is good practice to assess whether the observations can actually
constrain the variability properties of interest:

* **Nyquist period:** the shortest period resolvable given the sampling cadence.
* **Detectable period range:** periods that are both above the Nyquist limit and
  shorter than the observational baseline.

The methods :meth:`~pgmuvi.lightcurve.Lightcurve.compute_sampling_metrics` and
:meth:`~pgmuvi.lightcurve.Lightcurve.assess_sampling_quality` provide quantitative
summaries and plain-language recommendations.  For multiband data,
:meth:`~pgmuvi.lightcurve.Lightcurve.assess_sampling_quality_per_band` gives
per-band assessments.

Variability Detection
----------------------

Before investing computational effort in GP fitting, you may want to check whether
the source is actually variable.  ``pgmuvi`` provides three complementary variability
statistics:

* **Weighted chi-square** test against a constant flux.
* **:math:`F_\mathrm{var}`** (fractional variability / excess variance).
* **Stetson K** index, a robust measure of variability.

Use :meth:`~pgmuvi.lightcurve.Lightcurve.check_variability` for single-band data and
:meth:`~pgmuvi.lightcurve.Lightcurve.check_variability_per_band` for multiband data.

Synthetic Data
---------------

``pgmuvi`` can generate synthetic light curves from a GP model with known
hyperparameters, which is useful for:

* testing the fitting pipeline,
* assessing parameter recovery,
* creating simulated observations for survey planning.

See the :mod:`pgmuvi.synthetic` module and the synthetic data tutorial notebook for
usage examples.
