Batch wavelength advisory workflow
==================================

.. note::

   **Documentation status:** current through PR94.

   **Pipeline status:** the batch workflow runs the single-source advisory
   workflow over many sources, exports per-source products, and writes batch
   summaries.  It remains advisory-only and does not apply automatic model
   selection.

   **TBD[batch-validation]:** update this page after held-out validation,
   production comparison modes, or automatic model-selection policies are
   implemented.

Overview
--------

The batch workflow is the survey-scale wrapper around
:doc:`wavelength_advisory`.  It is useful when you want to compare the same set
of LPV-relevant model/kernel configs across many multiwavelength light curves.

The batch helper writes three levels of output:

* one row per source,
* one row per source and model/kernel config, and
* one row per distinct model/kernel config aggregated across the batch.

It also records data-hygiene metadata, per-source output directories, failure
artifacts, spectral-mixture ARD scale-ceiling diagnostics, and fallback failure
summaries.  These fields are intended to make a large real-source batch
auditable without reconstructing state from console logs.

The high-level command-line entry point is:

.. code-block:: bash

   PYTHONPATH=. python3 examples/run_wavelength_advisory_batch.py \
     10131+3049=10131+3049.csv \
     --output-dir wavelength_batch_outputs \
     --batch-prefix example_batch \
     --training-iter 500 \
     --miniter 100 \
     --max-samples 1000 \
     --max-samples-per-band 100

By default, the example script applies a strict post-ingestion data hygiene
filter before advisory fits: rows with non-positive flux values or
non-positive flux-error values are dropped after ``Lightcurve.from_csv`` has
loaded the source.  Use ``--allow-nonpositive-flux`` or
``--allow-nonpositive-flux-error`` only when you intentionally want to inspect
unfiltered input behavior.  Per-source JSON outputs include the underlying
period-independent robust distribution diagnostics, including
``raw_half_amplitude_q02_5_q97_5`` and the central-90%
``raw_half_amplitude_q05_q95`` amplitude proxy.

The same script also accepts a source-list file:

.. code-block:: bash

   cat > sources.txt <<'EOF'
   10131+3049=10131+3049.csv
   07454-7112=07454-7112.csv
   EOF

   PYTHONPATH=. python3 examples/run_wavelength_advisory_batch.py \
     --source-list sources.txt \
     --output-dir wavelength_batch_outputs \
     --batch-prefix example_batch \
     --training-iter 500 \
     --miniter 100

Python API
----------

The equivalent Python entry point is:

.. code-block:: python

   from pgmuvi.lightcurve import Lightcurve as LC

   report = LC.run_period_independent_wavelength_advisory_workflow_batch(
       [
           {"source_id": "10131+3049", "csv_path": "10131+3049.csv"},
           {"source_id": "07454-7112", "csv_path": "07454-7112.csv"},
       ],
       from_csv_kwargs={
           "check_sampling": True,
           "max_samples": 1000,
           "max_samples_per_band": 100,
           "verbose": True,
       },
       positive_data_filter_kwargs={
           "require_positive_flux": True,
           "require_positive_flux_error": True,
       },
       workflow_kwargs={
           "include_2d_baseline": True,
           "base_fit_kwargs": {
               "training_iter": 500,
               "miniter": 100,
               "fit_strategy": "consensus",
               "learn_additional_noise": True,
               "verbose": False,
           },
           "make_text_report": True,
           "make_plots": True,
       },
       output_dir="wavelength_batch_outputs",
       export=True,
       export_kwargs={"close_figures": True},
       batch_prefix="example_batch",
   )


Input hygiene and per-source outputs
------------------------------------

The batch entry points can apply a strict post-ingestion filter after
``Lightcurve.from_csv``.  With the default example-script settings, rows with
non-positive fluxes or non-positive flux errors are dropped before advisory
fits run.  Source rows record ``n_rows_before_positive_filter``,
``n_rows_after_positive_filter``, and ``n_rows_dropped_positive_filter`` so the
filtering decision remains visible in exported tables.

Each source also has explicit output-location metadata.  ``source_output_dir``
and ``source_output_prefix`` identify where that object's JSON, text report,
comparison report, plots, or failure artifacts were written.  This applies to
both successful sources and failed sources.

Batch contract fields
---------------------

The top-level report repeats the advisory contract so that downstream scripts do
not have to infer intent from filenames:

.. list-table:: Batch contract fields
   :header-rows: 1

   * - Field
     - Expected current value
     - Meaning
   * - ``kind``
     - ``period_independent_wavelength_advisory_workflow_batch``
     - Identifies the report type.
   * - ``advisory_only``
     - ``True``
     - The workflow reports/ranks only.
   * - ``runs_fits``
     - ``True``
     - Per-source workflows may run GP fits.
   * - ``applies_to_fit``
     - ``True``
     - The batch evaluates real fit calls for each config.
   * - ``model_kernel_config_state_isolated``
     - ``True``
     - Model/kernel configs are evaluated without installing a winner on the caller's main object.
   * - ``mutates_input_lightcurve``
     - ``False``
     - Input ``Lightcurve`` state is not overwritten by the top-ranked config.
   * - ``automatic_model_selection_applied``
     - ``False``
     - No automatic selection policy has run.
   * - ``selected_model``
     - ``None``
     - No final model has been selected.

Output files
------------

A successful batch run can write the following output bundle:

.. list-table:: Batch output artifacts
   :header-rows: 1

   * - Artifact
     - Path pattern
     - Contents
   * - JSON manifest
     - ``<prefix>_summary.json``
     - Full batch report, including source rows, output paths, aggregate metadata, and exported file list.
   * - Source summary CSV
     - ``<prefix>_summary.csv``
     - One row per source.
   * - Long-form model/kernel-config CSV
     - ``<prefix>_model_kernel_configs.csv``
     - One row per source and evaluated model/kernel config.
   * - Aggregate model/kernel-config CSV
     - ``<prefix>_model_kernel_config_summary.csv``
     - One row per distinct model/kernel config across the batch.
   * - Markdown report
     - ``<prefix>_report.md``
     - Human-readable summary of the batch contract, output files, aggregate table, and source table.
   * - Per-source export directories
     - ``<source_id_sanitized>/``
     - Single-source JSON, text, comparison text, and metric plots.

Source summary CSV
------------------

The one-row-per-source CSV is the right table for answering: "which sources ran,
which failed, and which model was top-ranked for each source?"

Important columns include:

.. list-table:: Source summary columns
   :header-rows: 1

   * - Column
     - Meaning
   * - ``source_index``
     - Position of the source in the input batch.
   * - ``source_id``
     - User-facing source identifier.
   * - ``status``
     - ``passed`` or failure status.
   * - ``top_ranked_model``
     - Advisory top-ranked model for this source, if available.
   * - ``top_ranked_fit_quality_score``
     - Fit-quality score associated with the top-ranked model.
   * - ``score_kind``
     - Scoring method, currently training-residual fit quality for PR61-style reports.
   * - ``n_model_kernel_configs``
     - Number of model/kernel configs evaluated for the source.
   * - ``n_successful_model_kernel_configs``
     - Number of configs that completed successfully.
   * - ``n_failed_model_kernel_configs``
     - Number of configs that failed and were recorded.
   * - ``n_rows_before_positive_filter`` / ``n_rows_after_positive_filter``
     - Row counts before and after optional strict positive flux/error filtering.
   * - ``n_rows_dropped_positive_filter``
     - Number of rows removed by the optional strict positive flux/error filter.
   * - ``exception_type`` / ``exception_message``
     - Failure diagnostics for source-level failures.
   * - ``source_output_dir`` / ``source_output_prefix``
     - Per-source export directory and filename prefix used for successful workflow artifacts or failure artifacts.

Long-form model/kernel-config CSV
---------------------------------

The long-form table is the right table for answering: "how did each model/kernel
config perform for each source?"

Important columns include:

.. list-table:: Long-form model/kernel-config columns
   :header-rows: 1

   * - Column
     - Meaning
   * - ``source_id``
     - Source identifier.
   * - ``model_kernel_config_id``
     - Stable row identifier such as ``rank1_2DWavelengthDependent``.
   * - ``model_kernel_config_rank``
     - Original advisory/evaluation order.
   * - ``quality_rank``
     - Rank after training-residual fit-quality scoring.
   * - ``is_top_ranked``
     - Whether this row is the top-ranked config for the source.
   * - ``model``
     - Model name, e.g. ``2DDustMean``.
   * - ``fit_strategy``
     - Fitting strategy, usually ``consensus`` for this workflow.
   * - ``time_kernel_type``
     - Time-kernel handoff, e.g. ``quasi_periodic`` for separable LPV configs.
   * - ``wavelength_kernel_type``
     - Wavelength kernel where applicable.
   * - ``learn_additional_noise``
     - Whether learned additional likelihood noise was requested.
   * - ``training_iter`` / ``miniter``
     - Training controls used for the fit.
   * - ``fit_success`` / ``fit_failed``
     - Per-config fit status.
   * - ``fit_quality_score``
     - Advisory residual-quality score.
   * - ``consensus_period`` / ``consensus_frequency``
     - Consensus period/frequency handed to or inferred by the config.
   * - ``consensus_time_kernel_constraint_mode``
     - How the consensus period/frequency was represented in the time kernel.
   * - ``training_nrmse_by_target_scale``
     - Training normalized RMSE diagnostic.
   * - ``training_median_abs_standardized_residual``
     - Median absolute standardized training residual.
   * - ``training_outlier_fraction_3sigma``
     - Fraction of large standardized residuals.
   * - ``training_reduced_chi2``
     - Reduced-chi-square-like training residual diagnostic.
   * - ``exception_type`` / ``exception_message``
     - Per-config failure diagnostics.
   * - ``failure_stage`` / ``failure_stage_reason``
     - Broad failure classification and short reason for failed configs.
   * - ``is_consensus_failure`` / ``is_numerical_failure`` / ``is_input_validation_failure``
     - Boolean triage flags for common failure modes.
   * - ``n_constrained_sm_ard_components``
     - Number of full-``2D`` spectral-mixture ARD scale components near the scale ceiling.
   * - ``n_constrained_sm_time_components`` / ``n_constrained_sm_wavelength_components``
     - Whether constrained ARD scales are associated with time-frequency,
       wavelength-frequency, or both.
   * - ``constrained_sm_ard_components``
     - Machine-readable details for constrained ARD scale components.

Aggregate model/kernel-config CSV
---------------------------------

The aggregate table is the right table for answering: "across this batch, which
configs succeeded most often and ranked first most often?"

Important columns include:

.. list-table:: Aggregate model/kernel-config columns
   :header-rows: 1

   * - Column
     - Meaning
   * - ``model``
     - Model name represented by the aggregate row.
   * - ``fit_strategy`` / ``time_kernel_type`` / ``wavelength_kernel_type``
     - Config-defining fit/kernel choices.
   * - ``n_sources_evaluated``
     - Number of sources for which this config was evaluated.
   * - ``n_successful_sources``
     - Number of sources for which this config completed successfully.
   * - ``n_failed_sources``
     - Number of sources for which this config failed.
   * - ``n_top_ranked_sources``
     - Number of sources for which this config ranked first.
   * - ``success_fraction``
     - ``n_successful_sources / n_sources_evaluated``.
   * - ``top_ranked_fraction``
     - ``n_top_ranked_sources / n_sources_evaluated``.
   * - ``mean_fit_quality_score`` / ``median_fit_quality_score``
     - Fit-quality score summaries across sources.
   * - ``best_fit_quality_score`` / ``worst_fit_quality_score``
     - Score range across sources.
   * - ``median_training_nrmse_by_target_scale``
     - Batch median of the training normalized RMSE diagnostic.

Reading the Markdown report
---------------------------

The Markdown report is intended for quick human inspection.  It contains:

* the batch contract,
* source totals,
* output paths,
* the aggregate model/kernel-config summary table, and
* the per-source summary table.

The Markdown report should be the first file to open after a batch run.  Use the
CSV files for sorting, filtering, and analysis.

Interpreting a one-source smoke run
-----------------------------------

For a one-source smoke run with the default LPV-relevant set and the optional
``2D`` baseline, it is normal to see four model/kernel configs:

.. code-block:: text

   n_model_kernel_configs = 4
   n_successful_model_kernel_configs = 4
   n_failed_model_kernel_configs = 0

This means four model/kernel fit configurations were evaluated for that source.
It does not mean four wavelength bands, four periods, or four physical
components.

Failure handling
----------------

Batch runs are designed to keep going when individual sources or individual
configs fail.  Failures are recorded in the source summary and long-form tables
using ``exception_type`` and ``exception_message`` fields.  Per-config rows also
record ``failure_stage``, ``failure_stage_reason``, ``is_consensus_failure``,
``is_numerical_failure``, and ``is_input_validation_failure``.  Inspect these
fields before interpreting aggregate success fractions.

When ``output_dir`` is supplied, failed sources receive compact per-source
failure artifacts: a structured JSON file and a readable text report containing
the exception type, exception message, and traceback.  Their paths are recorded
as ``export_json_path`` and ``export_text_report_path``.

Spectral-mixture ARD scale diagnostics
--------------------------------------

For advisory runs that evaluate the full ``2D`` spectral-mixture baseline, the
long-form model/kernel-config CSV flags fitted spectral-mixture ARD scales near
the consensus scale ceiling.  The most useful summary fields are
``n_constrained_sm_ard_components``, ``n_constrained_sm_time_components``, and
``n_constrained_sm_wavelength_components``; the detailed component list is in
``constrained_sm_ard_components``.  These fields help identify whether the
fit is saturating in time-frequency, wavelength-frequency, or both.
Advisory failure fallback reporting
-----------------------------------

If every model/kernel config for a source fails, the per-source workflow still
exports diagnostic fallback metadata.  ``fallback_diagnostics_available`` marks
that a fallback report is present, and ``fallback_report`` summarizes failure
stages, exception types, failed models, consensus-failure models, and suggested
next inspection steps.  Use this information to triage a source before deciding
whether to rerun with different consensus settings, different model/kernel
configs, or stricter input filtering.

Recommended workflow
--------------------

1. Run a one-source smoke test with small ``training_iter`` and ``miniter``.
2. Confirm that the JSON, Markdown, source CSV, long-form CSV, aggregate CSV,
   and per-source exports are written.
3. Run a small multi-source batch.
4. Inspect the Markdown report and aggregate CSV.
5. Inspect the long-form CSV for sources where the top-ranked model is unstable
   or where configs failed.
6. Increase training controls for production-style runs.
7. Treat all rankings as advisory until stronger validation metrics are added.
