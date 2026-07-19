pgmuvi.wavelength_validation
=============================

The validation primitives in this module provide the versioned, JSON-safe
contracts used by the D1--D3 wavelength-model validation programme.  They
record synthetic truth, scenario configuration, classified failures,
per-run recovery metrics, environment provenance, and failure-aware aggregate
summaries.

These records do not run fits, rank models, or modify the wavelength-advisory
workflow.  Every scenario, run, and aggregate remains advisory-only, and the
aggregate contract rejects automatic model-selection claims.  Per-model
summaries may describe recovery, stability, status counts, and boundary
pressure, but they are not formal evidence that one model is selected.

Unknown fields are retained under ``extra_fields`` so later validation stages
can extend the schema without discarding information.  Non-finite numerical
values are normalized to ``None`` during serialization.  Synthetic scenarios
must carry an explicit truth record, while observed D3 scenarios may omit truth
and instead rely on input checksums and provenance.  D1 execution is provided
by :mod:`pgmuvi.wavelength_validation_recovery`; the controlled D2 robustness
and failure-boundary layer is provided by
:mod:`pgmuvi.wavelength_validation_robustness`.

.. automodule:: pgmuvi.wavelength_validation
   :members:
   :undoc-members:
   :show-inheritance:
