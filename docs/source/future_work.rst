Documentation status and future work
====================================

This page is the searchable registry for implementation-dependent documentation
work. It is not a release schedule and does not imply that every listed feature
will be implemented. Each item records a current public boundary that should
remain visible until package behavior changes and tests exist for the new
workflow.

Marker policy
-------------

A future-work item uses a short kebab-case identifier following the literal
``TBD`` prefix. The same identifier must appear both in this registry and on at
least one relevant user-facing page. Run the registry audit from the repository
root with::

   python3 scripts/audit_docs_tbd_markers.py

The audit fails when a marker is malformed, is used outside this registry
without being registered, or is registered without a user-facing reference.

Validation and model-selection boundaries
-----------------------------------------

**TBD[held-out-validation]**
    Add held-out, posterior-predictive, or evidence-style validation before the
    single-source wavelength advisory workflow is described as validated model
    comparison. Until then, its ranking remains training-residual advisory
    evidence only.

**TBD[batch-validation]**
    Add production-scale validation and comparison policies for the batch
    advisory workflow. Existing batch outputs are auditable and failure-aware,
    but they do not establish an automatic survey-wide selection rule.

**TBD[automatic-model-selection]**
    Define, validate, and expose a policy that can select and install a final
    model. The current advisory contract intentionally leaves
    ``selected_model`` as ``None``.

Wavelength-derived fitting constraints and validation
-----------------------------------------------------

Completed wavelength-derived constraint tranche (PR121--PR125)
    The shared wavelength-sampling context is implemented, and its data-derived
    covariance lengthscale is now transformed into the GP input coordinate and
    applied to the separable wavelength kernels in
    ``2DWavelengthDependent``, ``2DDustMean``, ``2DPowerLawMean``, and
    ``2DSeparable``.  Existing constraints are intersected, constraint-before-
    value ordering is preserved, and raw/model-coordinate plus effective-bound
    provenance is recorded.

    Data-derived wavelength-mean initialization and enforceable constraints are
    now implemented independently for ``2DWavelengthDependent``,
    ``2DDustMean``, and ``2DPowerLawMean``.  Dust and power-law means reconstruct
    physical positive wavelength under affine input transforms, while the
    quadratic mean uses the GP input coordinate.  Parameter-level provenance
    records the robust-band fit, proposed interval, effective interval, and
    coordinate basis.

    The full ``2D`` spectral-mixture baseline now gives temporal and wavelength ARD
    dimensions genuinely independent broadcast tensor intervals
    for both ``mixture_means`` and ``mixture_scales``.  ARD index 0
    receives temporal-frequency values and bounds, while index 1 receives
    independently wavelength-derived values and bounds.  Source-type period
    limits alter only index 0, and raw/model-coordinate provenance is retained.

    Component- and dimension-specific fitted diagnostics are now implemented
    for both ``mixture_means`` and ``mixture_scales``.  They report lower and
    upper boundary pressure, distance to each bound, raw/model-coordinate
    values, affected component and ARD dimension, and whether
    ``num_mixtures=1`` was explicitly fixed or merely obtained in the fitted
    kernel.  These records remain descriptive and do not convert constraint
    pressure into model-selection evidence.

**TBD[wavelength-constraint-validation]**
    D1 nominal synthetic recovery and the core D2 robustness/failure-boundary
    scenario, runner, and aggregation framework are implemented.  Complete the
    multi-seed D2 calibration for sparse and uneven band coverage, missing
    bands, large wavelength gaps, weak and strong wavelength dependence,
    supported non-monotonic cases, seed stability, constraint-ceiling
    diagnostics, and consensus quasi-periodic time kernels.  Then add the
    maintained wavelength-constraint Jupyter notebook covering derivation,
    application, diagnostics, mean-versus-covariance constraints, and tested
    failure boundaries before proceeding to representative real LPV sources.
    Successful optimization alone is not sufficient validation.

Wavelength-model extensions
---------------------------

**TBD[multi-periodic-wavelength-models]**
    Add a documented and validated multi-periodic workflow beyond the current
    limited multi-component diagnostics and consensus infrastructure.

**TBD[non-monotonic-wavelength-kernels]**
    Add wavelength kernels or basis functions that can represent physically
    meaningful non-monotonic structure without forcing a simple smooth
    monotonic trend.

**TBD[physical-wavelength-kernels]**
    Add covariance kernels tied to physical wavelength-dependence models rather
    than relying only on generic RBF, Matérn, or spectral-mixture structure.

**TBD[wavelength-dependent-lags]**
    Add models that explicitly encode deterministic wavelength-dependent phase
    shifts or time delays. Current separable wavelength models do not provide
    that mechanism.

Documentation tutorial extensions
---------------------------------

**TBD[wavelength-constraint-notebook]**
    Add a maintained Jupyter notebook after the D2 multi-seed robustness
    calibration is complete.  The notebook must demonstrate wavelength-scale
    derivation, coordinate transformation, constraint application, fitted
    boundary diagnostics, the distinction between wavelength-dependent mean
    and covariance constraints, and the empirically tested D2 failure
    boundaries.

**TBD[batch-notebook]**
    Add a maintained notebook for the batch wavelength-advisory workflow.
    The current command-line walkthrough and deterministic preparation script
    remain the supported executable documentation until that notebook exists.

**TBD[result-interpretation-notebook]**
    Add a maintained notebook for interactive interpretation of period,
    wavelength-trend, ARD, fit-quality, and failure/fallback reports. The
    current interpretation guide and JSON-report helper remain authoritative.

Other package boundaries
------------------------

**TBD[mcmc-implementation]**
    Implement and test the public MCMC workflow before adding a new MCMC
    tutorial. ``Lightcurve.mcmc`` currently raises ``NotImplementedError``.

**TBD[native-magnitude-input]**
    Add native magnitude-domain input and uncertainty handling. Current
    documentation requires users to convert magnitudes to relative flux first.

**TBD[multidimensional-psd-plotting]**
    Add a defined visualization contract for PSDs from 2-D models. The current
    ``plot_psd`` helper does not support multidimensional models.

Closing or changing an item
---------------------------

When implementation work resolves one of these boundaries:

1. update the relevant user guide and API reference;
2. add or refresh a runnable script and notebook when appropriate;
3. add regression tests for the new public behavior;
4. remove the marker from the owning page and this registry in the same PR;
5. run ``python3 scripts/audit_docs_tbd_markers.py``; and
6. run ``cd docs && make clean && make html-strict``.
