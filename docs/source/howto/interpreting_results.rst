Interpreting Results
=====================

This guide explains how to interpret the output of ``pgmuvi`` fits, including the
fitted hyperparameters, diagnostic plots, and summary statistics.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

After fitting a ``pgmuvi`` model, you have access to:

* fitted (or sampled) hyperparameter values,
* predictive distributions (GP mean and variance at arbitrary times),
* inferred power spectral density (PSD),
* diagnostic and summary outputs.

Fitted Hyperparameters
-----------------------

After calling :meth:`~pgmuvi.lightcurve.Lightcurve.fit`, the model
hyperparameters are updated to their optimised (MAP) values.  Retrieve the
current parameter values as a dictionary after ``fit()``::

    params = lc.get_parameters()
    print(params)

Note that :meth:`~pgmuvi.lightcurve.Lightcurve.fit_LS` computes
Lomb–Scargle peak frequencies for initialisation and does not update the model
hyperparameters to MAP values.

.. note::

   Full posterior sampling via :meth:`~pgmuvi.lightcurve.Lightcurve.mcmc` is
   not yet available; the method currently raises ``NotImplementedError``.
   MCMC support is planned for a future release.

The key parameters are:

* ``mixture_means`` — optimised frequencies (in units of ``1 /`` the time
  unit of ``xdata``).
  Convert to periods by taking the reciprocal: ``periods = 1.0 / mixture_means``.
* ``mixture_scales`` — bandwidths (coherence timescales).
* ``mixture_weights`` — relative amplitudes.
* ``likelihood.noise`` — inferred white-noise variance.

For convenience::

    lc.print_periods()         # prints inferred periods in human-readable form
    lc.print_parameters()      # prints all hyperparameter values
    lc.summary()               # combined human-readable summary

Predictive Distribution
------------------------

Once fitted, the GP predictive mean and credible interval can be visualised
directly using :meth:`~pgmuvi.lightcurve.Lightcurve.plot`::

    fig = lc.plot()

:meth:`~pgmuvi.lightcurve.Lightcurve.plot` shows the data alongside the GP
predictive mean and shaded credible region when the model has been fitted.
When the model has not yet been fitted, only the raw data are shown.

Visualisation
--------------

:meth:`~pgmuvi.lightcurve.Lightcurve.plot`
    Raw data if the model is unfitted; after MAP fitting, also plots the GP
    predictive mean and credible interval.  Returns a
    ``matplotlib.pyplot.Figure`` (or a list of ``Figure`` objects for 2D
    multiband data).

:meth:`~pgmuvi.lightcurve.Lightcurve.plot_results`
    Training-history diagnostics (parameter values vs.\ optimisation
    iteration).  Use this to check convergence behaviour rather than
    predictive fit quality.  Returns ``None``.

:meth:`~pgmuvi.lightcurve.Lightcurve.plot_psd`
    Inferred power spectral density.  Peaks correspond to inferred periods.
    Broad, low-frequency power indicates stochastic variability.
    Returns ``(fig, ax)`` when ``show=False``.

:meth:`~pgmuvi.lightcurve.Lightcurve.plot` and
:meth:`~pgmuvi.lightcurve.Lightcurve.plot_psd` accept standard Matplotlib
keyword arguments.

MCMC Results (Planned)
-----------------------

.. note::

   Full MCMC support is planned for a future release of ``pgmuvi``.
   :meth:`~pgmuvi.lightcurve.Lightcurve.mcmc`,
   :meth:`~pgmuvi.lightcurve.Lightcurve.plot_corner`,
   :meth:`~pgmuvi.lightcurve.Lightcurve.plot_trace`, and
   :meth:`~pgmuvi.lightcurve.Lightcurve.print_results` currently raise
   ``NotImplementedError``.

When MCMC becomes available, the full posterior distribution over
hyperparameters will be accessible.  Planned visualisation methods include:

:meth:`~pgmuvi.lightcurve.Lightcurve.plot_corner`
    Corner plot (parameter covariance matrix).  Reveals correlations between
    periods, amplitudes, and noise.

:meth:`~pgmuvi.lightcurve.Lightcurve.plot_trace`
    MCMC trace plots.  Use these to check convergence — well-mixed chains should
    look like "fuzzy caterpillars".

:meth:`~pgmuvi.lightcurve.Lightcurve.print_results`
    Tabulated posterior summary (mean, median, credible intervals, effective sample
    size, :math:`\hat{R}` diagnostic).

A :math:`\hat{R}` value close to 1.0 (< 1.01 is a common criterion) indicates
that the chains have converged to the same distribution.

Common Warning Signs
---------------------

Overfitting
    The GP predictive mean passes through every data point and the uncertainty band
    is very narrow everywhere.  This suggests the noise variance is too small, or
    that ``num_mixtures`` is too large.  Try constraining the noise or reducing
    ``num_mixtures``.

Poor convergence (MAP)
    The loss does not decrease or oscillates.  Common causes:

    * Poor initialisation — try using Lomb–Scargle initialisation via ``fit_LS()``.
    * Constraints that are too tight — check that the true period lies within your
      constraint bounds.
    * Learning rate too large — reduce ``lr`` in the call to
      :meth:`~pgmuvi.lightcurve.Lightcurve.fit`.

Poor MCMC mixing *(planned feature)*
    High :math:`\hat{R}` or very low effective sample size.  This guidance will
    apply once MCMC support is available.  Common causes:

    * The chains are stuck in different modes — run multiple chains from different
      starting points.
    * Strong parameter correlations — use tighter priors or a different
      parameterisation.
    * Insufficient warmup — increase ``warmup_steps``.

Spurious periods
    The PSD shows peaks at aliases (multiples or sub-multiples of the observing
    cadence), or at the observational baseline.  Always compare the inferred periods
    against the Nyquist period and baseline from
    :meth:`~pgmuvi.lightcurve.Lightcurve.compute_sampling_metrics`.

Period Uncertainty
-------------------

When using MAP optimisation, no formal uncertainty is reported on the period.  A
practical proxy is the inferred bandwidth ``mixture_scales``: a wider bandwidth
(larger :math:`\sigma_q`) corresponds to a less coherent signal and a less
precisely determined period.

.. note::

   Rigorous posterior period uncertainties via MCMC are planned for a future
   release of ``pgmuvi``.  Once available,
   :meth:`~pgmuvi.lightcurve.Lightcurve.mcmc` will provide credible intervals
   on ``1 / mixture_means``.
