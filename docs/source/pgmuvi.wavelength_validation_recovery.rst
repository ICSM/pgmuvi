Synthetic wavelength-recovery validation
========================================

.. automodule:: pgmuvi.wavelength_validation_recovery
   :members:
   :undoc-members:
   :show-inheritance:

Scientific scope
----------------

This module executes the D1 synthetic-recovery layer using the truth-preserving
cases from :mod:`pgmuvi.wavelength_validation_synthetic`.  A run can use the
maintained :class:`pgmuvi.lightcurve.Lightcurve` fitter or injected construction
and fit hooks for controlled validation environments.

The per-run metrics currently cover:

* physical/model wavelength-coordinate round-trip error;
* recovery of the fundamental period;
* recovery of the physical wavelength covariance lengthscale;
* prediction-space recovery of the wavelength-dependent mean at observed bands;
* spectral-mixture ARD value recovery for both ``mixture_means`` and
  ``mixture_scales`` by component and temporal/wavelength dimension;
* spectral-mixture ARD boundary-hit counts and maintained dimension labels.

Missing fitted quantities remain explicit unavailable metrics.  They are never
replaced with zero and are excluded from numeric aggregate summaries.

Default per-run gates
---------------------

The frozen defaults are deliberately broad initial D1 gates:

* coordinate round-trip maximum absolute error: ``1e-10``;
* period relative error: ``0.15``;
* wavelength-lengthscale multiplicative error: ``4``;
* mean-law normalized RMSE: ``0.15`` for moderate/strong dependence and
  ``0.25`` for weak/negligible dependence.

The mean-law error is evaluated at the observed bands and normalized by the
larger of the generating mean span and median absolute generating mean.  This
makes the metric less sensitive to correlated parameterizations that produce
the same wavelength trend.

Aggregate D1 gates are evaluated only for runs whose fitted model matches the
scenario's generating model.  The initial gates require median/p90 period
relative errors no greater than ``0.05``/``0.15`` and median/p90 wavelength
lengthscale factor errors no greater than ``2``/``4``.  Other candidate fits
remain in completion, failure, and metric summaries but do not define recovery
of the generating parameterization.

Execution and failure semantics
-------------------------------

``run_synthetic_wavelength_recovery`` distinguishes setup, optimization, and
recovery-diagnostic failures.  ``run_synthetic_wavelength_recovery_matrix``
continues across failed runs by default and retains every run in the aggregate.
Use ``stop_on_error=True`` only when deliberate fail-fast behavior is required.

The maintained LPV defaults are:

* ``fit_strategy='consensus'``, ``time_kernel_type='quasi_periodic'``, and
  ``use_acf=True`` for ``2DWavelengthDependent``, ``2DDustMean``,
  ``2DPowerLawMean``, and ``2DSeparable`` so nominal D1 runs can reconcile a
  dominant LS harmonic with a longer ACF-supported fundamental;
* independent two-dimensional spectral-mixture initialization for ``2D`` with
  one requested component unless overridden by the validation driver;
* ``learn_additional_noise=True`` and linear-flux fitting.

Model-selection boundary
------------------------

Recovery reports and aggregates are advisory validation artifacts.  They do
not rank candidates as scientific evidence, install a winner, set
``selected_model``, or apply automatic model selection.  Aggregate completion
fractions and pass rates describe validation behavior only.

Full recovery matrices
----------------------

Ordinary package tests should use deterministic injected runners or very small
smoke fits.  The planned multi-seed recovery matrix is an explicit validation
run and should not be added to every routine unit-test or documentation build.
