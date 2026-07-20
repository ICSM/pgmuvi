Synthetic wavelength robustness and failure boundaries
======================================================

.. automodule:: pgmuvi.wavelength_validation_robustness
   :members:
   :undoc-members:
   :show-inheritance:

Scientific scope
----------------

This module implements the D2 scenario, execution, expected-failure, and
aggregation framework.  It reuses the maintained D1 fit and recovery-extraction
path so robustness experiments exercise the same wavelength initialization,
constraints, consensus defaults, and ARD diagnostics as nominal recovery.
It does not reuse the D1 population gates as D2 acceptance criteria.

The canonical controlled perturbations cover:

* independent versus shared per-band time grids;
* 36-point sparse sampling and longer-but-sparser baselines;
* uneven per-band observation counts;
* missing blue-edge, red-edge, and interior bands;
* large wavelength gaps;
* heteroscedastic and high-noise measurements;
* weak and strong wavelength dependence;
* sparse independent sampling for the full ``2D`` spectral-mixture ARD model;
  and
* an explicit insufficient-per-band-sampling failure boundary.

Every case preserves the generating truth and identifies its robustness axis,
severity, and paired reference scenario.  The default matrix fits each case only
with its generating model.  An explicit ``models`` argument requests a
case-by-model cross product; it is never interpreted as automatic selection.

Failure semantics
-----------------

A scenario may carry a
:class:`pgmuvi.wavelength_validation.WavelengthValidationFailureExpectation`.
The D2 runner compares the actual failure code, execution stage, status
dimensions, and acceptable exception types with that contract.  A matched
expected failure remains a failed and comparison-ineligible attempt; it is not
relabeled as a successful fit.  Failures in ordinary robustness scenarios are
recorded separately as unexpected failures.

Aggregation boundary
--------------------

D2 aggregates report completion, recovery metrics, expected-failure matches,
unexpected failures, and summaries by perturbation axis, severity, and model.
The scenario layer deliberately leaves empirical boundary calibration to
:mod:`pgmuvi.wavelength_validation_robustness_calibration`.  That module adds
truth-matched reference populations, paired base seeds, multi-seed execution,
constraint-boundary summaries, and advisory empirical classes without
weakening or reusing the frozen D1 aggregate gates.

Maintained documentation notebook
---------------------------------

The public :doc:`notebooks/tutorial_wavelength_constraints` notebook now
demonstrates wavelength-kernel scale derivation, coordinate transformation,
constraint application, fitted boundary diagnostics, the distinction between
wavelength mean and covariance constraints, and the tested sparse/noisy
failure boundaries.  It uses deterministic reduced examples and does not rerun
the full D2 population.

Model-selection boundary
------------------------

Robustness reports are advisory validation artifacts.  They do not rank models,
set ``selected_model``, install a winner, or apply automatic model selection.
