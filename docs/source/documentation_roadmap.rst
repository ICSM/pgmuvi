Documentation expansion roadmap
===============================

.. note::

   **Documentation status:** the PR97--PR109 expansion sequence is complete.
   Ongoing documentation work is now implementation-dependent and is tracked in
   :doc:`future_work`.

This page records the completed documentation-expansion project. It is separate
from the earlier documentation build cleanup and wavelength-advisory synchronization work that made the public docs warning-clean.

Goal
----

The documentation should let a new user understand the package without knowing
PGMUVI's development history. Each major workflow should be documented in small,
self-contained chunks that can be read independently, tested locally, and linked
from examples or notebooks.

Documentation layers
--------------------

Each substantial user-facing workflow should have the following layers where
appropriate:

Concept background
    A short reStructuredText page explaining the scientific and statistical
    concepts behind the workflow, its assumptions, and outputs that require
    cautious interpretation.

Runnable Python script
    A small ``.py`` example in ``examples/`` that can be run from a checkout.
    Scripts should avoid hidden notebook state and should write outputs to an
    explicit directory when they produce files.

Notebook tutorial
    A notebook for interactive exploration once the workflow is stable. Public
    notebooks should execute with current package defaults and should not
    duplicate dead or stale examples.

API reference
    A concise API page or section for the objects used by the workflow. Large
    legacy modules may use manual synopsis pages rather than full autodoc when
    expanded docstrings are too noisy.

TBD markers
    Explicit searchable markers for implementation-dependent future work. The
    marker registry and audit command are documented in :doc:`future_work` and
    :doc:`docs_maintenance`.

Completed workflow coverage
---------------------------

Core package orientation
    **Completed in PR97.** The overview and first-workflow material explains
    what PGMUVI is for, the main workflow paths, model-family roles, and what a
    new user should try first.

Lightcurve creation and validation
    **Completed in PR98, with a cross-reference repair in PR99.** The loading
    guide and runnable example cover CSV expectations, array shapes, band and
    wavelength handling, finite-value filtering, uncertainty requirements,
    sampling checks, subsampling, magnitude conversion, and common warnings.

Single-source fitting
    **Completed in PR99.** The direct and consensus fitting guide covers 1-D and
    2-D baselines, quasi-periodic and spectral-mixture handoff, separable models,
    learned additional noise, time centering, numerical-stability settings, and
    structured success or failure artifacts.

Wavelength-dependent model guidance
    **Completed in PR100.** The model-family guidance and advisory pages explain
    the roles of ``2D``, ``2DSeparable``, ``2DWavelengthDependent``,
    ``2DDustMean``, and ``2DPowerLawMean`` and distinguish advisory ranking from
    validated model selection.

Batch advisory workflows
    **Completed in PR101.** The batch walkthrough documents source-list formats,
    per-source folders, aggregate reports, continuation and exit semantics, and
    ordered failure triage.

Result interpretation
    **Completed in PR102.** The interpretation guide and JSON-report helper cover
    period provenance, PSD peaks, wavelength trends, fit-quality summaries,
    spectral-mixture ARD diagnostics, and failure/fallback reports.

Runnable examples
    **Completed across PR97--PR102 and PR107.** Major workflows now have small
    scripts that are syntax-checked and, where practical, exercised without
    starting expensive GP training.

Notebook refresh
    **Completed in PR103--PR108.** Maintained notebooks are in the public
    tutorial toctree, stale workflow notebooks were replaced, and the
    unavailable MCMC notebook was deleted rather than retained as a
    nonfunctional example.

Future-work registry
    **Completed in PR109.** Implementation-dependent documentation boundaries
    are centralized in :doc:`future_work` and checked by
    ``scripts/audit_docs_tbd_markers.py``.

Completed PR sequence
---------------------

The documentation-expansion sequence was:

1. PR97: core orientation and first-choice workflow guidance.
2. PR98: light-curve input and validation documentation.
3. PR99: single-source and consensus-fitting workflows.
4. PR100: wavelength-dependent model-family guidance.
5. PR101: batch advisory walkthrough and failure triage.
6. PR102: result interpretation and report inspection.
7. PR103--PR108: public notebook refresh and dead-notebook removal.
8. PR109: roadmap closeout and explicit future-work registry.

Completion standard
-------------------

A documentation area is considered complete only when it has:

- a current reStructuredText guide;
- a runnable script or documented reason why no script is appropriate;
- a notebook or an explicit implementation-dependent future-work marker;
- links from the appropriate index page;
- tests that protect the guide from losing key workflow terms; and
- a clean ``make html-strict`` build.

Completion of this roadmap does not mean the software has no missing features.
It means current workflows are documented and known implementation boundaries
are explicit rather than hidden in stale examples or vague promises.
