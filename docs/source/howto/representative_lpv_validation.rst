Representative observed-LPV validation
==========================================

The D3 workflow applies the maintained wavelength-advisory machinery to a
representative observed LPV and collects the evidence needed to inspect the
fit without claiming known physical truth.

The public example source is
``examples/data/10131+3049.csv``.  It is loaded and fitted in linear flux.

Data-coordinate contract
------------------------

An observational channel identifies an instrument, filter, or data stream.
A physical wavelength is the numeric wavelength coordinate used by the GP.

For this source, multiple observational channels may share one physical
wavelength.  D3 preserves those channel identities and does not average,
merge, calibrate, correct, or reassign their measurements or wavelength
coordinates.

Instrument-specific calibration is not implemented.  That work remains
``TBD[instrument-channel-calibration]``.  No calibration correction is
silently approximated.

Maintained candidate set
------------------------

The representative LPV candidate set is:

* ``2DWavelengthDependent``
* ``2DDustMean``
* ``2DPowerLawMean``
* ``2DSeparable``
* ``2D`` as the structurally different joint spectral-mixture baseline

The preferred LPV configuration for the separable families uses
``fit_strategy="consensus"``,
``time_kernel_type="quasi_periodic"``, and
``learn_additional_noise=True``.

Run one representative-source report
------------------------------------

Load the public source and execute the maintained D3 wrapper::

    from pgmuvi.lightcurve import Lightcurve
    from pgmuvi.wavelength_validation_real_lpv import (
        run_representative_lpv_validation,
    )

    lightcurve = Lightcurve.from_csv(
        "examples/data/10131+3049.csv",
        check_sampling=False,
        max_samples=None,
        max_samples_per_band=None,
    )

    report = run_representative_lpv_validation(
        lightcurve,
        source_id="10131+3049",
        description="Representative public LPV validation source.",
        workflow_kwargs={
            "base_fit_kwargs": {
                "training_iter": 500,
                "miniter": 100,
            },
        },
    )

    payload = report.to_dict()

The wrapper reuses isolated advisory candidate fits.  It does not mutate the
input light curve's retained fit state.

Run an ordered source manifest
------------------------------

``RepresentativeLPVSourceSpecification`` defines one manifest row.  Every row
records:

* a unique ``source_id``;
* the source path;
* a description;
* the source's representative sample role;
* the reason it was selected;
* a nonnegative seed; and
* optional JSON-safe metadata.

Manifest order is preserved.  Duplicate source identifiers are rejected.
The nonnegative seed is applied across source loading and advisory execution.
Python, NumPy, and PyTorch random-number-generator states are restored after
each source, so the batch does not leave the caller's RNG streams advanced.

``run_representative_lpv_validation_batch`` executes the same advisory
single-source wrapper for each manifest entry::

    from pgmuvi.wavelength_validation_real_lpv import (
        RepresentativeLPVSourceSpecification,
        run_representative_lpv_validation_batch,
    )

    manifest = [
        RepresentativeLPVSourceSpecification(
            source_id="10131+3049",
            source_path="examples/data/10131+3049.csv",
            description="Representative public LPV validation source.",
            sample_role="public_validation_source",
            selection_reason=(
                "Public multiwavelength LPV with broad wavelength coverage."
            ),
            seed=0,
        ),
    ]

    batch_report = run_representative_lpv_validation_batch(
        manifest,
        output_root="validation_outputs/d3_real_lpv",
    )

    batch_payload = batch_report.to_dict()

A source-loading failure or advisory-workflow failure is recorded for that
source and does not abort later manifest entries.  Each source result records a
deterministic intended directory of the form
``validation_outputs/d3_real_lpv/sources/<source_id>`` and a corresponding
``report.json`` path.

The protocol runner does not create those directories or write report files.
It reports ``writes_outputs=False``.  The batch envelope retains the exact
workflow configuration and compact runtime provenance, including Python,
NumPy, PyTorch, GPyTorch, and package-version information.

After the batch has completed,
``export_representative_lpv_batch_report`` may serialize the already-computed
payload::

    from pgmuvi.wavelength_validation_real_lpv import (
        export_representative_lpv_batch_report,
    )

    export_manifest = export_representative_lpv_batch_report(
        batch_report,
        "validation_outputs/d3_real_lpv",
    )

The export layer writes strict JSON, a compact source-summary CSV, one
``source_result.json`` per source, and either ``report.json`` or
``failure.json`` for each source.  Filesystem path components are sanitized.
Exporting does not rerun fits or alter any scientific conclusion.

Command-line runner
-------------------

The maintained command-line entry point is
``scripts/run_representative_lpv_validation.py``.  The committed public
manifest and workflow configuration can first be validated without fitting::

    PYTHONPATH=. python3 scripts/run_representative_lpv_validation.py \
        --manifest examples/validation/d3_representative_lpv_manifest.json \
        --workflow-config examples/validation/d3_representative_lpv_workflow.json \
        --validate-manifest-only

The validation-only mode loads and validates the manifest, resolves relative
source paths against the manifest location, and writes no outputs.

A later D3 execution may use the same files without
``--validate-manifest-only``::

    PYTHONPATH=. python3 scripts/run_representative_lpv_validation.py \
        --manifest examples/validation/d3_representative_lpv_manifest.json \
        --workflow-config examples/validation/d3_representative_lpv_workflow.json \
        --output-root validation_outputs/d3_real_lpv

Source failures remain recorded rather than aborting later sources.  The
optional ``--fail-on-source-failure`` flag returns status 2 after all sources
have been attempted and the reports have been emitted.

The batch protocol also performs no automatic model selection, does not install
a model, and does not automatically apply wavelength-derived constraints or
initialization.  Those values remain evidence recorded by the underlying
candidate fits.

Evidence recorded
-----------------

The D3 report records:

* period and consensus evidence by model;
* training-space fit quality;
* residual wavelength structure;
* wavelength-derived constraint and boundary diagnostics;
* warnings; and
* structured failures.

The observed source has no truth record, so the report creates no
truth-recovery metric or truth-recovery gate.

Interpretation boundary
-----------------------

A top-ranked model is descriptive evidence from the available comparative
fit-quality diagnostics.  The workflow does not install or select that model.
It performs no automatic model selection.

The report therefore always retains:

* ``advisory_only=True``;
* ``automatic_model_selection_applied=False``; and
* ``selected_model=None``.

A successful optimization is not, by itself, validation.  Period evidence,
residual structure, constraint pressure, warning state, and failed attempts
must be reviewed together before drawing a scientific conclusion.
