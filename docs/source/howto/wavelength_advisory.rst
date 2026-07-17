Wavelength advisory workflow
============================

.. note::

   **Documentation status:** current through PR94.

   **Pipeline status:** the period-independent wavelength advisory workflow is
   implemented as an advisory ranking and reporting workflow.  It does **not**
   automatically select, install, or endorse a final science model.

   **TBD[held-out-validation]:** update this page after held-out validation,
   posterior-predictive scoring, or evidence-style scoring is implemented.


Before running the advisory workflow, use :doc:`wavelength_models` to
understand the distinct mean and covariance assumptions of each model
family and the current try-first guidance.

Runnable tutorial
-----------------

The maintained :doc:`../notebooks/tutorial_wavelength_advisory` notebook gives
a deterministic no-fitting walkthrough of the period-independent diagnostics,
parameter plan, and model/kernel-config preparation stages.  It replaces the
older automatic-model-selection notebook and keeps ``RUN_FITS = False`` by
default.

Current workflow map
--------------------

The current advisory pipeline has four distinct evidence layers that should be
read together:

.. list-table:: Advisory output layers
   :header-rows: 1

   * - Layer
     - What it tells you
   * - Period-independent diagnostics
     - Per-band flux, scatter, and robust-amplitude behavior before any period
       or GP fit is used.
   * - Model/kernel-config runs
     - Which LPV-relevant configurations completed, failed, or produced useful
       training-residual diagnostics.
   * - Advisory ranking
     - A triage ordering of successful configs based on training residuals, not
       a formal model-selection decision.
   * - Failure fallback diagnostics
     - A compact summary of why model/kernel fits failed when no advisory
       ranking can be trusted.

The current high-level helper returns the model/kernel-config report, run
report, training-quality report, and fallback report in one dictionary.  It
does **not** preserve the complete period-independent diagnostic report or the
parameter-plan report as first-class nested sections.  Run those lower-level
helpers separately when their full provenance is required.  Batch runs export
the retained high-level information per source and per model/kernel config.

Purpose
-------

Multiwavelength light curves can vary in several different ways.  A source may
share the same temporal variability across bands while changing amplitude with
wavelength, may require a physically motivated wavelength-dependent mean, or may
be better described by the full non-separable 2-D spectral-mixture baseline.
The advisory workflow is designed to help inspect these possibilities without
pretending that a short training-residual score is final model selection.

The workflow answers questions such as:

* Do the per-band flux distributions show wavelength-dependent structure before
  any period or GP fit is used?
* Do robust central-90% and central-95% amplitude proxies change with
  wavelength?
* Which LPV-relevant model families should be evaluated explicitly?
* Which model/kernel configurations completed successfully?
* Did the full ``2D`` spectral-mixture baseline push ARD scales toward the
  consensus scale ceiling in time-frequency, wavelength-frequency, or both?
* If advisory fits failed, were the failures mostly consensus, numerical, or
  input-validation failures?
* What should be inspected next before making a scientific choice?

It does **not** answer, by itself:

* Which model is scientifically correct?
* Which model has the best predictive performance on held-out data?
* Which model should be used automatically for all later inference?

Terminology
-----------

``model/kernel config``
   One advisory fitting configuration: a model name plus the fitting options
   needed to evaluate it, including the time-kernel choice, wavelength-kernel
   choice where applicable, fit strategy, learned-noise setting, and training
   controls.  For example, ``2DDustMean`` with
   ``time_kernel_type="quasi_periodic"`` and ``fit_strategy="consensus"`` is
   one model/kernel config.

``fit_quality_ranking_status``
   Whether a comparative training-residual ranking is ``available``,
   ``single_valid_candidate``, or ``unavailable``.  At least two candidates
   with valid fit-quality diagnostics are required for ``available``.

``top_ranked_model``
   The model name with the best score when
   ``fit_quality_ranking_status="available"``.  It is ``None`` when every
   candidate is unscored or only one candidate has valid diagnostics.

``only_valid_model``
   The sole candidate with valid fit-quality diagnostics when
   ``fit_quality_ranking_status="single_valid_candidate"``.  This is not a
   comparative winner.

``selected_model``
   Reserved for a future workflow that actually applies a selection decision.
   In the current advisory workflow this remains ``None``.

``advisory_only``
   A contract field used by reports and exports.  ``True`` means the helper may
   diagnose, rank, score, format, plot, or export, but does not mutate the input
   light curve by installing a winning fit.

Known implementation-dependent extensions
-----------------------------------------

The current advisory workflow is intentionally bounded. The following markers
record implementation work rather than missing explanatory prose:

* **TBD[automatic-model-selection]:** define and validate a policy that can
  populate ``selected_model`` and install a final model.
* **TBD[multi-periodic-wavelength-models]:** add a validated multi-periodic fitting and
  comparison workflow beyond the current limited multi-component diagnostics.
* **TBD[non-monotonic-wavelength-kernels]:** add wavelength structures that can
  represent physically meaningful non-monotonic behavior.
* **TBD[physical-wavelength-kernels]:** add covariance kernels tied
  directly to physical wavelength-dependence models.
* **TBD[wavelength-dependent-lags]:** add models that explicitly encode
  deterministic wavelength-dependent phase shifts or time delays.

See :doc:`../future_work` for completion criteria and the full registry.

Recommended single-source shortcut
----------------------------------

For most users, the high-level method is the right entry point:

.. code-block:: python

   workflow = lc.run_period_independent_wavelength_advisory_workflow(
       include_2d_baseline=True,
       base_fit_kwargs={
           "training_iter": 500,
           "miniter": 100,
           "fit_strategy": "consensus",
           "learn_additional_noise": True,
           "verbose": True,
       },
       make_text_report=True,
       make_plots=True,
   )

   print(workflow["text_report"])
   print(workflow["fit_quality_ranking_status"])
   print(workflow["top_ranked_model"])
   print(workflow["only_valid_model"])
   print(workflow["quality_report"]["selected_model"])

The last line should print ``None`` for the current workflow.  A top-ranked
model is reported only when at least two candidates have valid fit-quality
diagnostics.  A single valid candidate is reported through
``only_valid_model`` instead of being promoted to a comparative winner.

The high-level workflow is equivalent to the staged process below.

Stage 0: period-independent wavelength structure
------------------------------------------------

Use this before fitting any temporal GP model:

.. code-block:: python

   diag = lc.diagnose_period_independent_wavelength_structure()

This diagnostic uses per-band flux distributions only.  It deliberately does not
use Lomb--Scargle peaks, ACF peaks, consensus frequencies, phase folding, GP
fits, or fitted residuals.  The returned dictionary records that the diagnostic
is period-independent and pre-fit.

Useful fields include:

.. list-table:: Selected period-independent diagnostic fields
   :header-rows: 1

   * - Field
     - Meaning
   * - ``kind``
     - Report type, normally ``period_independent_wavelength_diagnostics``.
   * - ``stage``
     - Diagnostic stage, normally ``prefit_period_independent``.
   * - ``is_period_independent``
     - ``True`` because no period or frequency is used.
   * - ``uses_temporal_consensus``
     - ``False`` for this diagnostic.
   * - ``n_bands`` / ``n_usable_bands``
     - Number of wavelength bands considered and retained.
   * - ``median_flux_monotonicity_class``
     - Descriptive monotonicity class for the per-band median flux.
   * - ``raw_half_amplitude_q02_5_q97_5_monotonicity_class``
     - Descriptive monotonicity class for the wider central-95% per-band amplitude proxy.
   * - ``raw_half_amplitude_q05_q95_monotonicity_class``
     - Descriptive monotonicity class for the central-90% robust per-band amplitude proxy.
   * - ``robust_scatter_monotonicity_class``
     - Descriptive monotonicity class for robust scatter.

.. warning::

   The monotonicity labels are descriptive diagnostics, not formal hypothesis
   tests.  They should guide inspection and later model/kernel-config choices,
   not serve as p-values.


Robust amplitude diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each usable wavelength band, the period-independent report records both a
central-90% and a central-95% robust amplitude proxy.  The central-90% fields
use ``q05`` and ``q95`` flux levels.  The wider central-95% fields use
``q02_5`` and ``q97_5`` flux levels and are useful when the extrema are noisy
but the central distribution is still informative.

Useful central-95% fields include
``raw_peak_to_peak_q02_5_q97_5``,
``raw_half_amplitude_q02_5_q97_5``,
``fractional_half_amplitude_q02_5_q97_5``,
``noise_corrected_half_amplitude_q02_5_q97_5``,
``raw_half_amplitude_q02_5_q97_5_ratio_max_to_min``,
``raw_half_amplitude_q02_5_q97_5_monotonicity_class``, and
``raw_half_amplitude_q02_5_q97_5_loglog_slope``.

Stage 1: advisory parameter/model plan
--------------------------------------

Build a non-mutating plan from the period-independent diagnostics:

.. code-block:: python

   plan = lc.build_period_independent_wavelength_parameter_plan(
       diagnostic_report=diag,
   )

The plan is advisory.  It does not set hyperparameters, register constraints,
run consensus fitting, or mutate the ``Lightcurve``.  It ranks LPV-relevant
follow-up model families but does not exclude alternatives automatically.

Typical LPV-relevant models are:

.. list-table:: LPV-relevant model families
   :header-rows: 1

   * - Model
     - Role in the advisory workflow
   * - ``2DWavelengthDependent``
     - Flexible separable wavelength-dependent covariance family.
   * - ``2DDustMean``
     - Separable covariance family with a dust-extinction-inspired wavelength mean.
   * - ``2DPowerLawMean``
     - Separable covariance family with a simple power-law wavelength mean.
   * - ``2D``
     - Full non-separable 2-D spectral-mixture baseline when requested.

Stage 2: build model/kernel configs
-----------------------------------

Convert the plan into explicit model/kernel configs:

.. code-block:: python

   config_report = lc.build_period_independent_wavelength_model_kernel_configs(
       parameter_plan=plan,
       include_2d_baseline=True,
       base_fit_kwargs={
           "training_iter": 500,
           "miniter": 100,
           "fit_strategy": "consensus",
           "learn_additional_noise": True,
       },
   )

The report lists the exact fit calls that will be evaluated later.  For the
separable LPV-relevant models the advisory workflow normally hands the consensus
period to a quasi-periodic time kernel.  The ``2D`` baseline uses the stabilized
2-D spectral-mixture consensus path.

Important fields include:

.. list-table:: Selected model/kernel-config planning fields
   :header-rows: 1

   * - Field
     - Meaning
   * - ``kind``
     - ``period_independent_wavelength_model_kernel_configs``.
   * - ``model_kernel_configs``
     - List of advisory model/kernel configs to evaluate.
   * - ``n_model_kernel_configs``
     - Number of configs generated.
   * - ``applies_to_fit``
     - ``False`` for the planning report itself.
   * - ``advisory_only``
     - ``True``.
   * - ``automatic_model_selection_applied``
     - ``False``.
   * - ``selected_model``
     - ``None``.

Stage 3: run model/kernel configs safely
----------------------------------------

Run the explicit configs in isolated ``Lightcurve`` copies:

.. code-block:: python

   run_report = lc.run_period_independent_wavelength_model_kernel_configs(
       model_kernel_config_report=config_report,
   )

This stage can run real GP fits.  It is still non-selecting.  The main input
``Lightcurve`` state is not overwritten by the best fit.

Useful fields include:

.. list-table:: Selected run-report fields
   :header-rows: 1

   * - Field
     - Meaning
   * - ``runs_fits``
     - ``True`` for this stage.
   * - ``model_kernel_config_state_isolated``
     - ``True`` when configs were run on isolated copies.
   * - ``model_kernel_config_results``
     - One row per attempted config.
   * - ``n_model_kernel_configs``
     - Number of configs attempted.
   * - ``n_successful_model_kernel_configs``
     - Number of configs whose fit completed successfully.
   * - ``n_failed_model_kernel_configs``
     - Number of configs that failed and were recorded.

Stage 4: score completed configs
--------------------------------

Use training-residual diagnostics to rank completed fits:

.. code-block:: python

   quality_report = lc.score_period_independent_wavelength_model_kernel_config_quality(
       run_report=run_report,
   )

This is a diagnostic ranking, not a final evidence calculation.  Current metrics
include training residual summaries such as normalized RMSE, median absolute
standardized residual, reduced chi-square-like summaries, and outlier fraction.
The standardized residuals use the observed predictive variance returned by
``likelihood(model(x_train))``.  Because that variance already contains the
likelihood noise, the stored measurement uncertainties are not added to it a second
time.  Transformed ``yerr`` is used only as a fallback when predictive variance is
unavailable.

.. warning::

   These scores are computed from training residual diagnostics.  They are useful
   for triage and debugging, but they are not cross-validation, not posterior
   predictive checking, and not a formal marginal-likelihood model comparison.

Stage 5: format and plot
------------------------

The comparison report can be formatted as text and plotted:

.. code-block:: python

   text = lc.format_period_independent_wavelength_model_kernel_config_comparison_report(
       quality_report,
   )
   figures = lc.plot_period_independent_wavelength_model_kernel_config_comparison(
       quality_report,
   )

Each plot is returned as a separate matplotlib figure.  The plotting helper does
not run fits, rescore configs, or select a model.

Stage 6: export a single-source workflow
----------------------------------------

A completed workflow dictionary can be exported:

.. code-block:: python

   workflow = lc.run_period_independent_wavelength_advisory_workflow(
       include_2d_baseline=True,
       base_fit_kwargs={"training_iter": 500, "miniter": 100},
       make_text_report=True,
       make_plots=True,
   )

   manifest = lc.export_period_independent_wavelength_advisory_workflow(
       workflow=workflow,
       output_dir="wavelength_advisory_export",
       prefix="source_wavelength_advisory",
       close_figures=True,
   )

The export helper writes files only.  It does not run fits unless the
``Lightcurve`` convenience method is called without a precomputed ``workflow``
and supplied with explicit ``run_workflow_kwargs``.

Interpreting the ranking
------------------------

A common output pattern is that the top two LPV-relevant models have similar
training-residual scores.  Do not treat small score differences as a final
scientific decision.  Inspect:

* the text report,
* the residual-metric plots,
* whether the top-ranked model is stable under longer training,
* whether repeated seeds/subsampling choices change the ranking,
* whether the inferred mean/wavelength behavior is physically plausible, and
* whether the simpler model is adequate for the science question.

Recommended production practice
-------------------------------

For a real source, start with a short smoke run only to verify that the pipeline
works.  Then increase ``training_iter`` and ``miniter`` for production-style
comparison.  Keep the result advisory until stronger validation metrics are
implemented.


Advisory failure fallback reporting
-----------------------------------

The advisory workflow remains non-selecting whenever a comparative fit-quality
ranking is unavailable.  This includes all-failed runs, runs whose completed
fits have no usable residual diagnostics, and runs with only one valid
candidate.  In those cases the workflow includes
``fallback_diagnostics_available=True`` and a ``fallback_report`` with a compact
triage summary.  The fallback records ``fit_quality_ranking_status`` together
with failure stages, exception types, failed models, consensus-failure models,
numerical-failure models, and recommended next inspection steps.

Use these fields to decide whether to inspect the input data, restore missing
diagnostics, relax consensus requirements, increase numerical safeguards, or
rerun only a subset of model/kernel configs.  They are not a substitute for a
valid comparison, and ``only_valid_model`` is not a comparative winner.
