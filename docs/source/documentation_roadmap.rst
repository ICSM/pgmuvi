Documentation expansion roadmap
===============================

This page tracks the broader documentation-expansion project. It is separate
from the documentation build cleanup and wavelength-advisory synchronization
work that made the current docs warning-clean.

Goal
----

The documentation should let a new user understand the package without knowing
PGMUVI's development history. Each major workflow should be documented in small,
self-contained chunks that can be read independently, tested locally, and linked
from examples or notebooks.

Documentation layers
--------------------

Each substantial user-facing workflow should eventually have all of the
following layers:

Concept background
    A short reStructuredText page explaining the scientific and statistical
    concepts behind the workflow. These pages should explain what the workflow
    is for, what assumptions it makes, and what outputs should be interpreted
    cautiously.

Runnable Python script
    A small ``.py`` example in ``examples/`` that can be run from a checkout.
    Scripts should avoid hidden notebook state and should write outputs to an
    explicit directory when they produce files.

Notebook tutorial
    A notebook for interactive exploration once the workflow is stable. Public
    notebooks should be executable with current package defaults and should not
    duplicate dead or stale examples.

API reference
    A concise API page or section for the objects used by the workflow. Large
    legacy modules may use manual synopsis pages rather than full autodoc when
    the expanded docstrings are too noisy.

TBD markers
    Explicit ``TBD`` notes where documentation describes future work or where
    the implementation is intentionally incomplete. TBD notes should be easy to
    search for and should name the missing functionality.

Workflow coverage plan
----------------------

Core package orientation
    Explain what PGMUVI is for, what kinds of astronomical time-series problems
    it targets, how wavelength enters the models, and what a user should try
    first.

Lightcurve creation and validation
    Document CSV expectations, required columns, band and wavelength handling,
    finite-value filtering, positive flux/error filtering, sampling diagnostics,
    and common warnings.

Single-source fitting
    Cover 1D fitting, 2D baseline fitting, consensus fitting, quasi-periodic
    kernels, spectral-mixture kernels, separable wavelength models, learned
    additional noise, and numerical-stability options.

Wavelength-dependent model guidance
    Explain the advisory workflow, the difference between descriptive
    diagnostics and model/kernel-config fitting, and when models such as
    ``2DWavelengthDependent``, ``2DDustMean``, ``2DPowerLawMean``, and
    ``2DSeparable`` should be considered.

Batch advisory workflows
    Document input layout, per-source output folders, batch summaries, CSV
    reports, Markdown reports, failure artifacts, and failure fallback
    diagnostics.

Result interpretation
    Explain periods, wavelength trends, robust amplitudes, fit-quality scores,
    constrained spectral-mixture ARD diagnostics, and failure/fallback fields.

Notebook refresh
    Keep only maintained notebooks in the public tutorial toctree. Quarantined
    notebooks should stay listed in :doc:`notebook_status` until they are
    refreshed, replaced, or deleted.

Near-term documentation PR sequence
-----------------------------------

The recommended order is:

1. Core package orientation and first-choice workflow guide.
2. Lightcurve input and validation guide with a runnable example script.
3. Single-source consensus fitting guide and runnable example updates.
4. Wavelength-dependent model guidance expansion beyond the advisory reference.
5. Batch advisory workflow walkthrough using a small synthetic or toy dataset.
6. Interpretation guide for reports, fit quality, ARD diagnostics, and failures.
7. Public notebook refresh, one notebook at a time.

Completion standard
-------------------

A documentation area is considered complete only when it has:

- a current reStructuredText guide;
- a runnable script or documented reason why no script is appropriate;
- a notebook or an explicit ``TBD`` marker for future notebook work;
- links from the appropriate index page;
- tests that protect the guide from losing key workflow terms; and
- a clean ``make html-strict`` build.
