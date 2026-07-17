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

Priors regularise the posterior distribution; during MAP optimisation they contribute
log-prior terms to the objective.  Their full role in MCMC sampling is planned for a
future release once ``mcmc()`` is available.

Constraints are applied during MAP optimisation to prevent the optimiser from
exploring physically unreasonable regions (e.g., negative noise variance or periods
longer than the observational baseline).  They will also apply to the planned future
MCMC workflow.

See :doc:`howto/priors_constraints` for practical guidance, and the
:mod:`pgmuvi.priors` and :mod:`pgmuvi.constraints` API reference for full details.

Data Transformations
---------------------

Raw observational data often span many orders of magnitude or use large absolute
time stamps that are inconvenient for numerical optimisation. ``pgmuvi`` provides
built-in data transformations:

* **TimeCenter:** subtracts a reference time from the time coordinate. This is the
  default x-axis behaviour when no explicit ``xtransform`` is supplied
  (``center_time="auto"``). It improves numerical conditioning while preserving
  period and frequency interpretation.
* **Shift:** subtracts a user-specified or automatically determined offset.
* **MinMax:** rescales data to the range [0, 1].
* **ZScore:** standardises data to zero mean and unit variance.
* **RobustZScore:** standardises using the median and median absolute deviation (MAD),
  making it robust to outliers.

Transformations can be applied to the time axis (``xtransform``) or the flux/magnitude
axis (``ytransform``). All predictions are automatically inverse-transformed back to
the original units when plotting or reporting results. Pass ``center_time=False`` to
turn off the default time-centering behaviour.

When a ``ytransform`` is active, dependent-variable uncertainties are transformed as
scales rather than as absolute values.  MinMax uncertainties are divided by the fitted
range, z-score uncertainties by the fitted standard deviation, and robust-z-score
uncertainties by the fitted MAD.  Location shifts such as the fitted minimum, mean,
median, or explicit offset are never subtracted from ``yerr``.
If ``set_likelihood(variance=True)`` is used because the stored values are already
variances, PGMUVI applies the square of the fitted transform scale.

1D vs 2D Models
----------------

``pgmuvi`` supports two modes of operation:

* **1D (single-band):** The GP input is the time axis alone.  This is appropriate
  when data from a single photometric band or wavelength range are available.

* **2D (multiband):** The GP input is a two-column array ``[time, wavelength]``.
  Two families of 2D kernels are available:

  * The default ``model="2D"`` uses a **non-separable** 2D spectral-mixture kernel
    that treats time and wavelength jointly.
  * The separable model family (``"2DSeparable"``, ``"2DAchromatic"``,
    ``"2DWavelengthDependent"``, etc.) uses a **product kernel** — a temporal
    spectral-mixture kernel multiplied by a wavelength kernel — which is a natural
    choice when the variability structure is assumed to be separable across the two
    dimensions.

  See :doc:`howto/multiband` for details on choosing between these families.

Model Choice and Advisory Workflows
-----------------------------------

``pgmuvi`` provides several GP model families with different kernel structures:

* **Spectral Mixture:** flexible, non-parametric PSD representation.
* **Matérn/RBF-style kernels:** smoother aperiodic variability models.
* **Periodic and quasi-periodic kernels:** useful when a coherent or slowly
  decorrelating periodic signal is expected.
* **2D spectral-mixture models:** non-separable models over time and wavelength.
* **Separable multiwavelength models:** product kernels combining a temporal kernel
  with a wavelength kernel and optional wavelength-dependent mean structure.

Older helper methods such as :meth:`~pgmuvi.lightcurve.Lightcurve.auto_select_model`
can recommend a model identifier from coarse data characteristics. Newer wavelength
workflows are advisory rather than automatic: they evaluate and rank plausible
model/kernel configurations but do not install a winning model. See
:doc:`howto/model_selection` for the legacy recommendation API; current
wavelength-advisory documentation is being added separately.

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

The :mod:`pgmuvi.synthetic` helpers generate analytic sinusoidal light curves
with known periods, amplitudes, phases, wavelength trends, and optional
observational noise.  Each helper returns a
:class:`~pgmuvi.lightcurve.Lightcurve`, which is useful for:

* testing validation and fitting pipelines;
* measuring recovery of explicitly injected signals;
* exploring cadence, baseline, and noise effects; and
* creating controlled one-dimensional or multiwavelength examples.

These helpers do not sample a Gaussian-process prior or posterior.  GP sampling
requires an explicitly constructed model and kernel and is a separate advanced
workflow.  See :doc:`notebooks/tutorial_synthetic` for the maintained analytic
generator walkthrough and :mod:`pgmuvi.synthetic` for the full API.

For random latent realizations from explicitly configured PGMUVI kernels, use
:doc:`howto/gp_prior_sampling` and
:doc:`notebooks/pgmuvi_mock_data_from_gp`.  That GP-prior workflow uses the
current parameter schema/application layer and does not start a fit.
