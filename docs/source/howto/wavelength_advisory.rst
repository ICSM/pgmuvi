Wavelength advisory workflow
============================

.. note::

   **Documentation status:** current through PR71.

   **Pipeline status:** the period-independent wavelength advisory workflow is
   implemented as an advisory ranking and reporting workflow.  It does **not**
   automatically select, install, or endorse a final science model.

   **TBD[held-out-validation]:** update this page after held-out validation,
   posterior-predictive scoring, or evidence-style scoring is implemented.

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
* Which LPV-relevant model families should be evaluated explicitly?
* Which model/kernel configurations completed successfully?
* Which completed fits have better training-residual diagnostics?
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

``top_ranked_model``
   The model name with the best score in an advisory report.  This is a summary
   field for inspection.  It is not an automatically selected model.

``selected_model``
   Reserved for a future workflow that actually applies a selection decision.
   In the current advisory workflow this remains ``None``.

``advisory_only``
   A contract field used by reports and exports.  ``True`` means the helper may
   diagnose, rank, score, format, plot, or export, but does not mutate the input
   light curve by installing a winning fit.

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
   print(workflow["quality_report"]["top_ranked_model"])
   print(workflow["quality_report"]["selected_model"])

The last line should print ``None`` for the current workflow.  A top-ranked
model is reported for inspection, but no model is selected or installed.

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
   * - ``raw_half_amplitude_q05_q95_monotonicity_class``
     - Descriptive monotonicity class for a robust per-band amplitude proxy.
   * - ``robust_scatter_monotonicity_class``
     - Descriptive monotonicity class for robust scatter.

.. warning::

   The monotonicity labels are descriptive diagnostics, not formal hypothesis
   tests.  They should guide inspection and later model/kernel-config choices,
   not serve as p-values.

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
