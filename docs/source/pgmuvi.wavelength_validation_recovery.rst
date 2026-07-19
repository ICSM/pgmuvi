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
* an informational RBF correlation-matrix RMSE at the observed wavelengths,
  including the shortest-to-longest-band correlation;
* an informational comparison between the empirical shared-grid latent-process
  wavelength correlation and the declared generating correlation, so an
  unrepresentative finite realization is distinguishable from fit failure;
* prediction-space recovery of the wavelength-dependent mean at observed bands;
* an informational centered mean-shape RMSE that removes a constant offset before
  comparing wavelength trends;
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

The primary mean-law error is evaluated at the observed bands and normalized by
the larger of the generating mean span and median absolute generating mean.  A
second informational metric removes the median from both predictions before
comparison so an offset mismatch can be distinguished from a wavelength-shape
mismatch.  The RBF correlation-matrix metric likewise separates the scientific
correlation pattern from the raw lengthscale factor error.

Aggregate D1 gates are evaluated only for runs whose fitted model matches the
scenario's generating model.  A single-realization matrix retains the original
median/p90 gates: period relative errors no greater than ``0.05``/``0.15`` and
wavelength-lengthscale factor errors no greater than ``2``/``4``.

When scenario identifiers are repeated across seeds, the aggregate switches to
a population contract.  It requires at least 95 percent completion overall and
within every scenario, a wavelength-lengthscale median no greater than ``2``, an
overall per-run pass fraction of at least ``0.80``, a median no greater than
``4`` within every identifiable scenario, and a minimum scenario pass fraction
of ``0.60``.  The wavelength-lengthscale p90 remains reported.  Values above
``4`` set an explicit tail-instability warning rather than being hidden or used
to invalidate otherwise representative population recovery.

A fitted lengthscale is still reported for fundamental-plus-harmonic cases, but
it is informational and excluded from the strict lengthscale gate when the
maintained separable runner uses one quasi-periodic temporal component.  In that
setting the temporal covariance is deliberately misspecified and can bias the
wavelength scale.  Other candidate fits remain in completion, failure, and
metric summaries but do not define recovery of the generating parameterization.

Execution and failure semantics
-------------------------------

``run_synthetic_wavelength_recovery`` distinguishes setup, optimization, and
recovery-diagnostic failures.  ``run_synthetic_wavelength_recovery_matrix``
continues across failed runs by default and retains every run in the aggregate.
Use ``stop_on_error=True`` only when deliberate fail-fast behavior is required.
The recovery artifact preserves both the compact parameter-workflow summary and
the detailed application report, including estimated values, proposed bounds,
effective registered constraints, and constraint actions.

Canonical nominal D1 scenarios use 72 observations per band on one shared
irregular time grid spanning 3.2 periods.  This prevents finite-sampling phase
differences from being misidentified as wavelength-dependent structure and
provides enough within-cycle information for population-level wavelength-scale
recovery.  Independent, 36-point, uneven, and longer-but-sparser band sampling
are implemented as controlled scenarios in
:mod:`pgmuvi.wavelength_validation_robustness`; D2 aggregates do not reuse the
frozen D1 gates.

The maintained LPV defaults are:

* ``fit_strategy='consensus'``, ``time_kernel_type='quasi_periodic'``, and
  ``use_acf=True`` for ``2DWavelengthDependent``, ``2DDustMean``,
  ``2DPowerLawMean``, and ``2DSeparable`` so nominal D1 runs can reconcile a
  dominant LS harmonic with a longer ACF-supported fundamental;
* independent two-dimensional spectral-mixture initialization for ``2D`` with
  one requested component unless overridden by the validation driver;
* ``verbose`` is consumed as a fit control before model construction, so it is
  never forwarded to GP constructors as an unsupported keyword;
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
