Batch wavelength advisory workflow
==================================

.. note::

   **Documentation status:** current through PR101.

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


Toy batch walkthrough
---------------------

``examples/wavelength_advisory_batch_walkthrough.py`` creates two deterministic
three-band toy light curves, a source-list file, and a machine-readable
walkthrough manifest.  The synthetic inputs exercise file layout, orchestration,
exports, continuation, and failure reporting.  They do not validate scientific
ranking, period recovery, or the physical suitability of any model family.

Prepare the workspace without starting GP training:

.. code-block:: bash

   PYTHONPATH=. python3 examples/wavelength_advisory_batch_walkthrough.py \
     --workspace wavelength_batch_walkthrough \
     --overwrite

The final line should be:

.. code-block:: text

   PREPARE ONLY: batch fitting was not started.

Inspect the generated inputs and command manifest:

.. code-block:: bash

   cat wavelength_batch_walkthrough/sources.txt
   python3 -m json.tool \
     wavelength_batch_walkthrough/walkthrough_manifest.json | less

Run the deliberately small mechanical smoke test:

.. code-block:: bash

   PYTHONPATH=. python3 examples/wavelength_advisory_batch_walkthrough.py \
     --workspace wavelength_batch_walkthrough \
     --overwrite \
     --run-batch \
     --training-iter 20 \
     --miniter 5 \
     --model-kernel-config-limit 1

The walkthrough disables sampling rejection and plot creation because its goal
is to test plumbing with tiny synthetic files.  Do not copy those two choices
blindly into a scientific run.  Increase the training controls, restore sampling
checks, enable plots where useful, and evaluate the full relevant model set on
real data.

Input source-list formats
-------------------------

The maintained batch CLI accepts four input layouts:

``Direct command-line tokens``
   Supply ``path/to/source.csv`` or ``source_id=csv_path`` as positional
   arguments.  A path-only token uses the filename stem as ``source_id``.

``Newline-delimited text``
   Each non-comment line may contain a path, ``source_id=csv_path``, or
   ``source_id,csv_path``.  The toy walkthrough writes this format.

``Headered CSV``
   Include ``csv_path`` (or ``path``, ``filename``, or ``file``) and optionally
   ``source_id`` or ``id`` columns.

``JSON list``
   Use a top-level list whose entries are either source strings or objects with
   ``source_id`` and ``csv_path`` fields.

Relative paths are resolved from the directory in which the batch command is
run.  Use absolute paths in long-lived manifests when the working directory may
change.

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


Output directory anatomy
------------------------

A successful toy run with ``batch_prefix="toy_batch"`` has this approximate
layout.  Figure files are omitted because the walkthrough uses ``--no-plots``.

.. code-block:: text

   wavelength_batch_walkthrough/
   ├── inputs/
   │   ├── toy_lpv_a.csv
   │   └── toy_lpv_b.csv
   ├── sources.txt
   ├── walkthrough_manifest.json
   └── outputs/
       ├── toy_batch_summary.json
       ├── toy_batch_summary.csv
       ├── toy_batch_model_kernel_configs.csv
       ├── toy_batch_model_kernel_config_summary.csv
       ├── toy_batch_report.md
       ├── toy_lpv_a/
       │   ├── toy_lpv_a_wavelength_advisory.json
       │   ├── toy_lpv_a_wavelength_advisory.txt
       │   └── toy_lpv_a_wavelength_advisory_comparison.txt
       └── toy_lpv_b/
           └── ...

A source that fails before producing a workflow receives
``<source_prefix>_failure.json`` and ``<source_prefix>_failure.txt`` in its own
source directory.  The batch-level summary files are still written unless the
run is aborted with ``--stop-on-error``.

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
   * - ``fit_quality_ranking_status``
     - ``available``, ``single_valid_candidate``, or ``unavailable``.
   * - ``fit_quality_ranking_available``
     - True only when at least two candidates have valid fit-quality diagnostics.
   * - ``only_valid_model``
     - Sole candidate with valid diagnostics when no comparative ranking exists.
   * - ``top_ranked_model``
     - Advisory top-ranked model only when comparative ranking is available.
   * - ``top_ranked_fit_quality_score``
     - Fit-quality score associated with a valid comparative top rank.
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
     - Current rank-dependent evaluation label such as
       ``rank1_2DWavelengthDependent``.  It is useful within one report, but it
       is not a stable configuration identity across candidate filtering,
       reordering, or truncation.
   * - ``model_kernel_config_rank``
     - Original advisory/evaluation order.
   * - ``quality_rank``
     - Rank among candidates with valid training-residual fit-quality diagnostics;
       unavailable candidates have no quality rank.
   * - ``is_top_ranked``
     - True only for the first-ranked row when a comparative ranking is available.
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
   * - ``training_standardization_sigma_source``
     - Whether standardized residuals used observed predictive variance or the
       transformed-uncertainty fallback. Observed predictive variance already
       includes likelihood noise and is not combined with ``yerr`` again.
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
   * - ``n_sources_with_fit_quality``
     - Number of source/config rows with valid fit-quality diagnostics.
   * - ``n_sources_with_comparative_ranking``
     - Number of sources for which at least two candidates had valid diagnostics.
   * - ``n_top_ranked_sources``
     - Number of valid comparative rankings in which this config ranked first.
   * - ``success_fraction``
     - ``n_successful_sources / n_sources_evaluated``.
   * - ``top_ranked_fraction``
     - ``n_top_ranked_sources / n_sources_with_comparative_ranking``; unavailable
       when no comparative ranking exists.
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


Exit status and continuation policy
-----------------------------------

By default, the batch helper records a source-level exception and continues to
the next source.  The CLI then returns **status 1** when one or more source rows
failed, even though batch summaries and failure artifacts may have been written.
This is the safest default for CI and scripted production runs.

``--allow-source-failures``
   Keep the same record-and-continue behavior but return **status 0** after the
   batch.  Use this only when a downstream step intentionally consumes partial
   batches and checks ``n_failed`` itself.

``--stop-on-error``
   Re-raise the first source-level exception immediately.  Later sources are not
   attempted, and the batch-level output bundle may be incomplete.

A nonzero CLI status is therefore not evidence that no artifacts were produced.
Inspect the batch manifest and source directories before rerunning.

Triage failures in this order
-----------------------------

``source-level failure``
   Start with the source-summary row, ``exception_type``, ``exception_message``,
   positive-filter row counts, and the per-source ``*_failure.json`` or text
   report.  This category includes CSV loading, input validation, and exceptions
   that prevent the single-source workflow from returning.

``per-config failure``
   If the source row passed but ``n_failed_model_kernel_configs`` is nonzero,
   inspect the long-form CSV.  Use ``failure_stage``, ``failure_stage_reason``,
   ``is_consensus_failure``, ``is_numerical_failure``, and
   ``is_input_validation_failure`` before comparing fit-quality scores.

``non-comparative fallback``
   When comparative fit-quality ranking is unavailable, read
   ``fallback_diagnostics_available`` and ``fallback_report`` in the per-source
   workflow JSON.  This covers all-failed, completed-but-unscored, and
   single-valid-candidate runs.  The fallback suggests the next diagnostic
   inspection; it does not repair the fit or select a replacement model.

``aggregate interpretation``
   Only after the first three checks should you interpret ``success_fraction``,
   ``top_ranked_fraction``, or median fit-quality fields across the batch.
   Aggregate rankings are advisory and can be distorted by systematic failure
   patterns.

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

If comparative fit-quality ranking is unavailable for a source, the per-source
workflow still exports diagnostic fallback metadata.
``fallback_diagnostics_available`` marks that a fallback report is present, and
``fallback_report`` records ``fit_quality_ranking_status`` plus failure stages,
exception types, failed models, consensus-failure models, and suggested next
inspection steps.  Use this information to triage all-failed, unscored, and
single-valid-candidate sources before rerunning with different consensus
settings, model/kernel configs, or input filtering.



.. note::

   **TBD[batch-notebook]:** add an interactive public notebook only after the
   batch output schema and ranking-validation policy are stable.  Until then,
   the maintained ``.py`` walkthrough is the reproducible entry point.

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
