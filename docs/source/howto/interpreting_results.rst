Interpreting PGMUVI results
===========================

.. note::

   **Documentation status:** current through PR174.

   This guide explains how to read period summaries, wavelength-trend
   diagnostics, training-residual fit-quality scores, spectral-mixture ARD
   registered-boundary diagnostics, and failure/fallback reports.  These outputs are
   diagnostics.  They do not turn the current advisory workflow into automatic
   model selection.

.. contents:: On this page
   :local:
   :depth: 2

Read outputs in this order
--------------------------

For a completed single-source analysis, inspect outputs in the following order:

1. confirm whether the requested fit actually passed;
2. inspect the period-summary method and kernel metadata;
3. compare the primary period peak with other significant peaks;
4. inspect wavelength-dependent descriptive trends;
5. read training-residual fit-quality metrics and ARD boundary diagnostics;
6. inspect consensus, numerical, or input-validation failures; and
7. keep ``selected_model=None`` unless a separate, explicit scientific
   selection procedure has been performed.

For batch advisory work, start with the batch summary, then inspect each
source-level JSON report, and finally inspect the long-form model/kernel-config
CSV for individual failures and boundary hits.

The maintained :doc:`../notebooks/tutorial_single_source_analysis` notebook
applies this order interactively to the bundled real source.  It records
per-observational-channel period evidence, consensus acceptance and rejection,
wavelength-constraint provenance, actual optimizer history, predictions,
residual and phase diagnostics, fixed and learned noise components, warnings,
failures, and a JSON-safe report.

Period summaries
----------------

After a fitted periodic or spectral-mixture model, request a kernel-aware
period summary:

.. code-block:: python

   summary = lc.get_period_summary()
   print(summary.to_text())

   primary = summary.get_primary_peak()
   if primary is not None:
       print(primary.period, primary.prominence, primary.area_fraction)

The summary records ``model_name``, ``method``, ``backend``, ``kernel_family``,
``time_kernel_family``, and ``has_stochastic_background``.  Read these fields
before interpreting any period because the meaning of the remaining fields
changes with the kernel family.

Primary peak versus largest-area feature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``primary_peak_rank`` identifies the physically ranked primary candidate.  The
ranking prioritizes prominence, then coherence, integrated area, and peak
height.  ``largest_area_peak_rank`` identifies the feature with the largest
integrated PSD area.  These ranks can differ.

When ``primary_peak_rank != largest_area_peak_rank``, report both features.  Do
not silently replace the primary candidate with the broadest integrated-power
feature.  The convenience fields ``dominant_period`` and
``dominant_frequency`` follow the primary peak, while
``largest_area_period``, ``largest_area_frequency``, and
``largest_area_fraction`` describe the largest-area feature.

Intervals and coherence
~~~~~~~~~~~~~~~~~~~~~~~

``period_interval`` and the backward-compatible
``period_interval_fwhm_like`` contain the interval identified by
``interval_definition``.  They are peak-width summaries, not posterior
credible intervals.  MAP optimisation does not provide formal posterior period
uncertainty.

``q_factor`` is a coherence proxy derived from the primary peak's frequency and
frequency width.  A larger finite value indicates a narrower, more coherent
feature.  It is not a detection significance and should not be compared across
reports without checking that the same summary method and frequency grid were
used.

Multiple peaks and multi-component consensus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``peaks``, ``n_peaks_detected``, ``n_significant_peaks``, and
``significant_periods`` to inspect alternative periodic features.  Harmonics,
aliases, window-function features, and genuinely multi-periodic variability can
all produce more than one peak.  Compare every candidate with the observing
baseline, sampling diagnostics, Lomb--Scargle results, and astrophysical
expectations.

For ``consensus_multicomp`` outputs, inspect ``component_summaries`` rather than
only a single dominant period.  Compare ``consensus_period``,
``initialized_mixture_period``, ``fitted_mixture_period``, member bands,
``fitted_period_drift_flag``, and ``fitted_frequency_drift_flag``.  A
component-identity warning or a large drift flag means the final fit may no
longer represent the component handed off by consensus.

Two-band consensus requires direct agreement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When exactly two bands contribute dominant frequencies, MAD clipping cannot
identify which value is an outlier.  Inspect
``two_band_fractional_frequency_difference``,
``two_band_max_fractional_frequency_difference``, and
``two_band_frequency_agreement``.  If the pair exceeds the configured limit,
consensus fails and ``final_consensus_frequency`` remains unavailable; PGMUVI
does not report the arithmetic midpoint as a shared period.

Kernel-component diagnostics are not final periods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``component_diagnostics`` contains fitted spectral-mixture component periods,
frequencies, weights, and scales.  These are internal kernel diagnostics, not
independent physical periods and not substitutes for the analyzed peaks in
``peaks``.  Cite the analyzed peak result, while using component diagnostics to
understand how the kernel represented the PSD.

Writing period outputs
~~~~~~~~~~~~~~~~~~~~~~

Write text, JSON, and figure products together:

.. code-block:: python

   summary = lc.write_period_summary_outputs(
       text_file="results/period_summary.txt",
       json_file="results/period_summary.json",
       png_file="results/period_summary.png",
       include_fit_history=True,
   )

By default the JSON omits the full PSD arrays to keep the file compact.  Set
``include_psd_in_json=True`` only when downstream analysis needs the complete
frequency grid and PSD.

Predictive plots and fitted parameters
--------------------------------------

``lc.plot()`` shows raw data before fitting and adds the GP predictive mean and
credible region after fitting.  ``lc.plot_psd()`` shows the inferred PSD for
compatible kernels.  Use ``show=False`` when a figure object is needed for
saving or additional annotation.

Use ``lc.get_parameters()`` and ``lc.print_parameters()`` to inspect fitted MAP
parameters.  ``lc.print_periods()`` remains a compact convenience display, but
``get_period_summary()`` is the structured interpretation interface.

``Lightcurve.plot_results()`` is not a maintained training-history interface in
the current release.  Use ``lc.get_fit_history_summary()`` and
``lc.export_fit_history_json(...)`` instead.

Interpreting wavelength trends
------------------------------

The period-independent wavelength report summarizes each usable band before a
temporal GP fit.  Read the per-band ``band_table`` together with the cross-band
``summary``.

Median and robust amplitude fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``median_flux`` describes the central flux level in each band.  Robust
half-amplitude proxies are half of a central quantile range:

* ``raw_half_amplitude_q05_q95`` uses the central 90 percent of values;
* ``raw_half_amplitude_q02_5_q97_5`` uses the central 95 percent of values;
* ``fractional_half_amplitude_*`` divides the raw half-amplitude by the
  absolute median flux; and
* ``noise_corrected_half_amplitude_*`` subtracts an approximate measurement-
  noise contribution in quadrature when positive uncertainties are available.

The wider central-95-percent proxy is less dependent on individual extrema than
raw peak-to-peak amplitude, but it is more sensitive to sparse tails than the
central-90-percent proxy.  Inspect both when band sampling differs.

Monotonicity and slopes are descriptive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fields such as
``raw_half_amplitude_q02_5_q97_5_monotonicity_class`` and
``median_flux_monotonicity_class`` are tolerance-rule descriptions, not
hypothesis tests.  Log--log slope fields summarize the observed coordinate
trend but do not prove a physical power law.  Do not use either field as a hard
model veto or a hard parameter constraint.

A trend is only interpretable when at least two bands are usable and the
wavelength coordinate is physically meaningful.  Integer band codes are not
physical wavelengths.

Fixed-frequency phase and lag diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every band must use one common ``reference_time`` before phases or lags are
compared.  When the caller does not provide one, the diagnostic uses the
midpoint of the full multiband time range and records
``fixed_frequency_reference_time_source="global_time_midpoint"``.

Lag values are periodic.  ``lag_linear_span`` preserves the naive signed
maximum-minus-minimum range for inspection, while ``lag_span`` uses the
``minimum_circular_arc`` method.  The circular value prevents phases lying on
opposite sides of the period boundary from being misclassified as almost one
full cycle apart.  A lag trend remains descriptive and conditional on the
supplied period or frequency.

Training-residual fit-quality scores
------------------------------------

The advisory workflow's ``fit_quality_score`` is a higher-is-better triage score
computed from fitted training-coordinate residuals.  It combines:

* ``training_nrmse_by_target_scale``;
* ``training_median_abs_standardized_residual``;
* ``training_outlier_fraction_3sigma``; and
* ``training_reduced_chi2``.

The score is useful for identifying obviously poor completed fits and ordering
follow-up inspection.  It is not cross-validation, not held-out predictive
performance, not AIC or BIC, and not marginal likelihood or model evidence.
Small score differences do not establish a scientifically preferred model.

Standardized residual metrics use the observed predictive standard deviation from
``likelihood(model(x_train))``.  That predictive variance already includes the
likelihood noise, so the reported measurement uncertainty is not added again.
``standardization_sigma_source`` records whether observed predictive variance was
available or whether transformed ``yerr`` had to be used as a fallback.

The complete single-source notebook also reports channel-level standardized
residual RMS.  Its screening reference is one, values above two are promoted to
a scientific-review warning, and values above three are labelled severe.  These
thresholds expose local mismatch between training residuals and predictive
uncertainty; they do not automatically reject a model or establish that another
model family is preferred.  Inspect the affected observational channels,
sampling, noise assumptions, and residual structure before interpreting the
fit.

A very low score should trigger inspection of the underlying metrics and fit
state.  ``fit_quality_available=False`` means the candidate is unscored; it is
not assigned a very poor sentinel score and cannot become top ranked.  A high
score can still represent overfitting because all metrics are evaluated at
training coordinates.

Read ``fit_quality_ranking_status`` before reading any rank:

``available``
   At least two candidates have valid fit-quality diagnostics.  Only in this
   state may ``top_ranked_model`` and ``top_ranked_fit_quality_score`` be
   populated.

``single_valid_candidate``
   Exactly one candidate has valid diagnostics.  It is exposed as
   ``only_valid_model`` but is not a comparative winner.

``unavailable``
   No candidate has valid diagnostics.  ``top_ranked_model`` remains ``None``.

In every state, ``automatic_model_selection_applied=False`` and
``selected_model=None`` remain the scientific contract.

Retained-state marginal-likelihood fields
------------------------------------------

The advisory result also records exact-GP marginal-likelihood diagnostics at the
currently retained parameter state.  These fields are descriptive and are not
used by ``fit_quality_score``.

``training_log_marginal_likelihood``
   Data log marginal likelihood **per observation**, evaluated in training mode
   from ``likelihood(model(x_train)).log_prob(y_train)``.  It excludes registered
   priors and additional objective terms.

``training_log_marginal_likelihood_total``
   The corresponding total data log marginal likelihood.

``training_map_objective`` and ``training_map_objective_total``
   The exact ``ExactMarginalLogLikelihood`` objective at the retained parameter
   state, reported per observation and in total.  Registered priors are included
   when present.

``training_registered_log_prior`` and ``training_registered_log_prior_total``
   The registered-prior contribution, separated from the data likelihood.
   ``training_registered_prior_count`` and
   ``training_map_objective_includes_registered_priors`` record whether such
   terms were present.

``training_additional_objective_terms``
   Any remaining non-prior terms added by the exact MLL objective, separated
   from both the data likelihood and registered priors.

``training_marginal_likelihood_evaluation_mode`` is ``train`` and
``training_marginal_likelihood_parameter_state`` is
``retained_current_state``.  The evaluator temporarily switches the model and
likelihood to training mode, computes all terms from the current parameters, and
then restores their original modes.  It does not reuse the last loss recorded
before an optimizer step.

These are full-data training quantities, not held-out predictive scores and not
Bayesian model probabilities.  Compare them only when candidates use identical
analysis data, target transforms, likelihood policy, and fitting assumptions.

Spectral-mixture ARD boundary diagnostics
-----------------------------------------

For the full ``2D`` spectral-mixture baseline, ``sm_ard_diagnostics`` reports
fitted values and registered bounds separately for ``mixture_means`` and
``mixture_scales``.  The last-axis coordinate names are:

``temporal_frequency``
   ARD index 0, associated with the time coordinate.

``wavelength_frequency``
   ARD index 1, associated with the wavelength coordinate.

Each component/dimension row records the model-coordinate value, lower and
upper bounds, absolute and normalized distance to each bound, interval
position, and the unconstrained GPyTorch raw parameter.  When PR124 parameter
workflow provenance is retained, the same fitted value and effective bounds are
also transformed back to the raw input coordinate.

``sm_ard_boundary_hits`` lists lower- and upper-bound pressure for both
parameters.  ``sm_ard_boundary_hit_counts_by_parameter``,
``sm_ard_boundary_hit_counts_by_dimension``,
``sm_ard_boundary_component_counts_by_dimension``, and
``sm_ard_boundary_hit_counts_by_side`` provide complementary summaries.
``sm_ard_boundary_pressure_scope`` is ``none``, ``temporal_only``,
``wavelength_only``, or ``both``.

``sm_num_mixtures_is_one`` describes the fitted kernel.
``sm_num_mixtures_fixed_at_one`` is true only when ``num_mixtures=1`` was
explicitly supplied in the fit configuration.  A one-component fit cannot
demonstrate component-to-component stability, so this distinction must be
carried into interpretation.

The older ``constrained_sm_ard_components`` fields remain compatibility aliases
for *upper-bound hits in ``mixture_scales`` only*.  New analyses should use the
full boundary fields above.

A boundary hit means the optimum is constraint-sensitive under the current
initialization, sampling, and optimizer path.  It does not prove that a
physical dependence is absent, infinitely broad, or scientifically preferred.
A long wavelength correlation scale and optimizer pressure can produce similar
boundary behavior; distinguish them through repeated seeds, synthetic recovery,
wavelength coverage, learned noise, and residual diagnostics.

Interpret the coordinate and parameter explicitly:

* temporal ``mixture_means`` pressure concerns fitted temporal frequencies;
* wavelength ``mixture_means`` pressure concerns wavelength-frequency
  structure;
* temporal ``mixture_scales`` pressure concerns temporal spectral width or
  coherence; and
* wavelength ``mixture_scales`` pressure concerns wavelength-coordinate
  spectral width.

Failure and fallback diagnostics
--------------------------------

A failed fit is not a low-quality successful fit.  Read ``status``,
``fit_success``, ``fit_failed``, ``exception_type``, ``exception_message``, and
``failure_stage`` before comparing scores.

For maintained advisory outcomes, also inspect the canonical status dimensions:

``attempt_disposition``
   Whether the configured attempt was attempted, skipped, or not attempted.

``execution_stage``
   The furthest stage reached, such as ``precondition``, ``consensus``,
   ``optimization``, ``diagnostics``, or ``completed``.

``technical_outcome``
   One of ``failed``, ``initialized_only``, ``completed``,
   ``completed_with_warnings``, or ``completed_with_recovery``.  A zero-iteration
   fit is initialization-only even though the legacy status remains ``passed``.

``diagnostic_validity`` / ``scientific_usability``
   Whether diagnostics are valid, partial, unavailable, or invalid, and whether
   the attempt is usable, limited, or unusable for scientific interpretation.

``comparison_eligibility``
   Whether the attempt can participate in the current fit-quality comparison.
   Initialization-only and diagnostically unavailable attempts are
   comparison-ineligible rather than assigned a very poor score.

``failure_code`` / ``failure_substage``
   Stable machine-readable failure identity and optional more specific stage.
   Structured ``failure_diagnostics`` and ``failure_summary`` are retained when
   supplied by a consensus failure, and general outer fit exceptions now also
   populate the canonical ``Lightcurve`` failure state.

The advisory failure classifier uses descriptive stages:

``input_validation``
   The light curve or requested configuration did not satisfy an input
   contract.

``consensus``
   A usable cross-band period handoff was not obtained or could not be applied.

``numerical_stability``
   The fit encountered PSD, Cholesky, positive-definiteness, or related
   numerical failures.

``fit_execution``
   The model/kernel fit raised another execution-time exception.

When comparative fit-quality ranking is unavailable,
``fallback_diagnostics_available=True`` and ``fallback_report.available=True``.
This includes all-failed runs, completed-but-unscored runs, and runs with only
one valid candidate.  Inspect ``fit_quality_ranking_status``,
``failure_stage_counts``, ``exception_type_counts``, failed model lists, and
``recommended_next_steps``.  The fallback report is diagnostic-only:
``automatic_model_selection_applied=False`` and ``selected_model=None``.

Do not turn an all-failed or single-survivor advisory run into a winner.  Fix or
narrow the workflow, then rerun the relevant configurations.

Mean and covariance hypothesis fields
-------------------------------------

Read ``model_hypothesis`` before attributing a score difference to one physical
mechanism.  The fields ``mean_structure`` and ``covariance_structure`` are
orthogonal descriptions of the complete model/kernel configuration.
``mean_wavelength_dependent`` and ``covariance_wavelength_dependent`` make that
separation explicit.

``2DDustMean`` and ``2DPowerLawMean`` are not mean-only alternatives: both also
contain smooth wavelength covariance.  ``2DWavelengthDependent`` likewise
changes the mean and covariance relative to the constant-mean separable model.
``2D`` uses a joint non-separable spectral-mixture covariance and should be read
as a baseline with a different parameterization, not as a nested special case.

The ``comparison_cautions`` list is part of the report precisely because
current training-residual rankings cannot identify which changed mechanism
caused a score difference.  Use the taxonomy for interpretation and grouping,
not as a substitute score or automatic selection policy.

Typed evidence roles
--------------------

:class:`pgmuvi.wavelength_results.WavelengthEvidenceKind` distinguishes
observed facts, derived statistics, heuristic interpretations, formal
comparison results, workflow warnings, and future-work limitations.  This
classification records what a result item claims.  Using it does not turn a
heuristic score into formal model evidence.  In particular, current
training-residual quality scores remain heuristic even when wrapped in a typed
result.

The typed attempt adapter also retains the orthogonal canonical status from
:mod:`pgmuvi.wavelength_status`, any structured failure record, captured
warnings, diagnostic groups, score fields, and the complete legacy payload.
Use ``to_dict()`` for the canonical versioned representation and
``to_legacy_dict()`` when an existing downstream consumer still expects the
flat dictionary contract.

Batch interpretation
--------------------

For a batch report, read:

1. ``n_sources``, ``n_succeeded``, and ``n_failed``;
2. each row in ``source_results`` for source-level ingestion or workflow
   failures;
3. ``model_kernel_config_results`` for individual config outcomes, training-
   residual metrics, and ARD boundary fields; and
4. ``model_kernel_config_summary`` for aggregate completion and score summaries.

Aggregate score statistics do not correct for heterogeneous sampling, source
brightness, uncertainty calibration, or different numbers of usable bands.
Compare sources scientifically only after checking those differences.

Runnable report interpreter
---------------------------

The maintained no-training example reads period-summary, single-source advisory,
or batch advisory JSON and prints a compact interpretation:

.. code-block:: console

   python3 examples/interpret_pgmuvi_outputs.py results/report.json

To save its normalized interpretation dictionary:

.. code-block:: console

   python3 examples/interpret_pgmuvi_outputs.py \
       results/report.json \
       --json-output results/report_interpretation.json

The script does not import ``pgmuvi``, run a fit, rescore candidates, or select a
model.  It only interprets fields already present in the supplied JSON.

Common warning signs
--------------------

Overfitting
   The predictive mean follows nearly every point, uncertainty bands are very
   narrow, learned noise is close to its lower bound, or training residuals look
   excellent while held-out behavior is unknown.

Poor MAP optimization
   Loss is non-finite, strongly oscillatory, or fails to improve; fitted
   parameters remain on bounds; repeated starts disagree substantially; or fit
   history reports an exception.

Spurious or aliased periods
   Peaks coincide with cadence aliases, seasonal gaps, harmonics, or the
   observing baseline.  Compare with sampling metrics and independent period
   diagnostics.

Boundary-limited ARD parameters
   One or more entries appear in ``sm_ard_boundary_hits``.  Treat each named
   parameter, component, dimension, and bound side as constraint-sensitive
   rather than as a well-measured interior optimum.

All advisory configs failed
   ``fallback_report.available`` is true.  No fit-based ranking is available,
   regardless of any pre-fit advisory ordering.

MCMC status and future work
---------------------------

Full MCMC fitting and posterior plotting are not available in the current
release.  ``Lightcurve.mcmc()``, ``plot_corner()``, and ``plot_trace()`` remain
future functionality.  Period intervals in current MAP summaries are therefore
peak-width diagnostics, not posterior credible intervals.

See also
--------

* :doc:`consensus_fitting`
* :doc:`wavelength_models`
* :doc:`wavelength_advisory`
* :doc:`wavelength_advisory_batch`
* :doc:`preprocessing`

Advisory conclusions are scoped, not selections
------------------------------------------------

For the period-independent wavelength workflow, inspect
``advisory_conclusions`` together with ``unresolved_ambiguities``.  A conclusion
about a ``complete_configuration`` applies to the full GP configuration.  It
must not be reinterpreted as isolated evidence for its wavelength mean or its
wavelength covariance.

The disposition ``remains_plausible`` means only that the available diagnostics
do not exclude the hypothesis.  ``weakened`` records explicit adverse advisory
evidence without claiming formal rejection.  ``technically_unevaluable`` means
that technical execution did not yield usable scientific evidence.
``scientifically_ambiguous`` means that the required same-mean or
same-covariance contrast is absent.  ``incomparable`` means that the result is
not eligible for the current like-for-like comparison.

The current scalar ranking remains a training-residual heuristic, not a formal
model-comparison statistic.  Consequently, neither ``top_ranked_model`` nor a
``remains_plausible`` conclusion performs automatic model selection.  Use the
ambiguity records to identify whether stronger interpretation requires another
successful candidate, an isolated mean/covariance contrast, held-out prediction,
seed-stability checks, or resolution of a recorded technical failure.
