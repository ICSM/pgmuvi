Interpreting PGMUVI results
===========================

.. note::

   **Documentation status:** current through PR102.

   This guide explains how to read period summaries, wavelength-trend
   diagnostics, training-residual fit-quality scores, spectral-mixture ARD
   scale-ceiling diagnostics, and failure/fallback reports.  These outputs are
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

A very low score or ``fit_quality_available=False`` should trigger inspection
of the underlying metrics and fit state.  A high score can still represent
overfitting because all metrics are evaluated at training coordinates.

``top_ranked_model`` is therefore an advisory display field.  In the current
workflow, ``automatic_model_selection_applied=False`` and
``selected_model=None`` remain the scientific contract.

Spectral-mixture ARD scale-ceiling diagnostics
----------------------------------------------

For the full ``2D`` spectral-mixture baseline, ARD diagnostics report fitted
scale values by component and coordinate dimension.  Dimension names are:

``time_frequency``
   The spectral-mixture scale associated with the time coordinate.

``wavelength_frequency``
   The spectral-mixture scale associated with the wavelength coordinate.

``constrained_sm_ard_components`` lists component/dimension pairs whose fitted
scale lies within the configured tolerance of the consensus upper bound.
``constrained_sm_ard_dimension_counts`` and
``n_constrained_sm_ard_components`` summarize those hits.

A ceiling hit means the optimum is boundary-limited under the current
constraint and initialization.  It does not prove that the corresponding
physical dependence is absent, infinitely broad, or scientifically preferred.
Check the recorded upper bound, ``fraction_of_upper``, learned noise, time
centering, training stability, and whether repeated fits reproduce the hit.

Interpret the coordinate explicitly:

* time-frequency ceiling hits concern temporal spectral width/coherence;
* wavelength-frequency ceiling hits concern the wavelength-coordinate spectral
  width; and
* hits in both dimensions may indicate that the common ARD constraint is too
  restrictive for that component or that the fit is weakly identified.

Failure and fallback diagnostics
--------------------------------

A failed fit is not a low-quality successful fit.  Read ``status``,
``fit_success``, ``fit_failed``, ``exception_type``, ``exception_message``, and
``failure_stage`` before comparing scores.

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

When all attempted model/kernel configs fail,
``fallback_diagnostics_available=True`` and ``fallback_report.available=True``.
Inspect ``failure_stage_counts``, ``exception_type_counts``, failed model lists,
and ``recommended_next_steps``.  The fallback report is diagnostic-only:
``automatic_model_selection_applied=False`` and ``selected_model=None``.

Do not turn an all-failed advisory run into a winner by choosing the least severe
exception.  Fix or narrow the workflow, then rerun the relevant configurations.

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

Boundary-limited ARD scales
   One or more entries appear in ``constrained_sm_ard_components``.  Treat the
   corresponding scale as constraint-sensitive rather than as a well-measured
   interior optimum.

All advisory configs failed
   ``fallback_report.available`` is true.  No fit-based ranking is available,
   regardless of any pre-fit advisory ordering.

MCMC status and future work
---------------------------

Full MCMC fitting and posterior plotting are not available in the current
release.  ``Lightcurve.mcmc()``, ``plot_corner()``, and ``plot_trace()`` remain
future functionality.  Period intervals in current MAP summaries are therefore
peak-width diagnostics, not posterior credible intervals.

.. admonition:: TBD: interpretation notebook

   ``TBD[result-interpretation-notebook]``: add a maintained notebook that reads
   real exported period and advisory reports, reproduces the interpretation
   sequence above, and compares training-residual diagnostics with future
   held-out validation outputs.

See also
--------

* :doc:`consensus_fitting`
* :doc:`wavelength_models`
* :doc:`wavelength_advisory`
* :doc:`wavelength_advisory_batch`
* :doc:`preprocessing`
