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
The initial framework deliberately marks empirical failure boundaries as not yet
calibrated.  The multi-seed validation tranche will establish those boundaries
without weakening the frozen D1 recovery gates.

Documentation notebook plan
---------------------------

After the empirical D2 boundaries are established, add a maintained Jupyter
notebook that demonstrates wavelength-kernel scale derivation, coordinate
transformation, constraint application, fitted boundary diagnostics, the
distinction between wavelength mean and covariance constraints, and the tested
sparse/noisy failure boundaries.  The notebook must be registered in the public
Tutorials toctree and in ``notebook_status.rst``.

Model-selection boundary
------------------------

Robustness reports are advisory validation artifacts.  They do not rank models,
set ``selected_model``, install a winner, or apply automatic model selection.
