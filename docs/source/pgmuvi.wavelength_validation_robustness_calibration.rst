Synthetic wavelength robustness calibration
===========================================

Twenty-seed empirical calibration
---------------------------------

PR133 executed the canonical D2 robustness matrix for base seeds 0 through 19:
420 canonical runs spanning seven truth-matched references, thirteen
scientifically valid perturbations, and one explicit invalid-sampling
scenario.  The run completed with 393 completed fits and 27 recorded
failures.

Twenty failures came from
``d2-insufficient-per-band-sampling``.  Every one matched the declared
expected-failure contract.  The remaining seven failures were structured
consensus-stage rejections rather than optimizer crashes: six occurred for
base seed 7 and one for base seed 10.  Their preserved diagnostics report
either ``insufficient_consensus_inliers`` or ``no_accepted_bands``.

The empirical advisory classifications were:

* seven reference populations;
* ten robust perturbations;
* one degraded perturbation, ``d2-large-wavelength-gap``;
* two failure boundaries, ``d2-uneven-band-counts`` and
  ``d2-longer-sparse-baseline``; and
* one reproducible expected-failure boundary,
  ``d2-insufficient-per-band-sampling``.

The longer sparse baseline is primarily a period-recovery boundary: only
4 of 20 runs passed the configured period criterion.  Uneven per-band
counts are primarily a wavelength-lengthscale recovery boundary: only 8
of 19 completed runs passed that criterion.  A large wavelength gap is
measurably degraded but remains below the configured failure-boundary
threshold.

For the joint ``2D`` spectral-mixture reference and sparse perturbation,
both temporal and wavelength ARD parameters were near their configured
bounds in every run while recovery remained robust.  Boundary pressure is
therefore reported separately and must not be interpreted as recovery
failure or as evidence for achromatic variability.

These results are advisory evidence only.  They do not perform automatic
model selection.  Generated run records and reports remain local under
``validation_outputs/`` and are intentionally excluded from version
control.  The maintained
:doc:`notebooks/tutorial_wavelength_constraints` notebook applies the maintained
constraint mechanism in bounded real-data fits.  It reports provenance, initial
and fitted parameter positions, predictive light curves, uncertainty intervals,
and residuals without depending on the generated validation-output directory.

.. automodule:: pgmuvi.wavelength_validation_robustness_calibration
   :members:
   :undoc-members:
   :show-inheritance:

Truth-matched reference populations
-----------------------------------

Each non-reference D2 perturbation is paired with a dense reference that has the
same generating model, wavelength-mean family, covariance family, period, and
other truth-defining parameters except for the controlled robustness axis.
Sampling, noise, wavelength coverage, or dependence strength may differ only
when that difference is the declared perturbation.  The population runner uses
the same base seed for each reference/perturbation pair so sampling, latent
process, and measurement-noise random streams remain paired as far as their
changed shapes permit.

The canonical set therefore contains separate references for the maintained
``2DSeparable``, ``2DWavelengthDependent``, ``2DDustMean``,
``2DPowerLawMean``, and joint ``2D`` spectral-mixture ARD configurations.
Comparing every perturbation with one separable reference would confound model
family and mean-law differences with robustness, and is rejected by the
reference-pair validator.

D2 calibration policy
---------------------

The multi-seed runner fits every case with its generating model.  It preserves
technical failures, classified expected failures, recovery metrics,
parameter-workflow provenance, and ARD diagnostics.  D2 classes are based on
paired changes in completion and per-run recovery retention relative to the
truth-matched reference population.  They do not reuse the D1 aggregate gates.

The advisory empirical classes are:

* ``reference`` for nominal truth-matched populations;
* ``robust`` when completion and recovery retention remain inside the D2
  robust-degradation envelope;
* ``degraded`` when the perturbation worsens results but remains short of the
  configured failure boundary;
* ``failure_boundary`` when technical failure or paired recovery degradation
  crosses the D2 boundary policy;
* ``expected_failure_boundary`` when a declared invalid case reproducibly
  matches its typed expected-failure contract; and
* ``inconclusive`` when seed coverage, reference evidence, or common metrics
  are insufficient.

Constraint-boundary diagnostics
-------------------------------

For separable wavelength kernels, the calibration layer combines the fitted
physical wavelength lengthscale with the effective model-coordinate constraint
and its raw/model conversion provenance.  For joint ``2D`` spectral-mixture
kernels, it retains component-, parameter-, and ARD-dimension-specific distances
for ``mixture_means`` and ``mixture_scales``.  Temporal-frequency and
wavelength-frequency pressure are summarized separately.

Near-bound and at-bound frequency are warnings, not automatic recovery
failures.  A fit can recover generating truth while lying near a broad lower or
upper constraint, and a poor recovery can occur without saturation.  Reports
therefore keep boundary pressure separate from the empirical robustness class.

Reproducibility and scope
-------------------------

``run_synthetic_wavelength_robustness_population`` accepts explicit base seeds
and temporarily seeds Python, NumPy, and Torch optimization randomness without
leaving global RNG state changed.  When ``output_dir`` is supplied, every run is
written atomically under ``runs/`` and later invocations can resume from those
records instead of repeating completed fits.  The final ``report.json`` records
base and optimizer seeds, reference validation, scenario summaries, boundary
records, persistence counts, and the D2 policy thresholds.

All outputs remain advisory.  They do not compare candidate models to choose a
winner, set ``selected_model``, or install a fitted model.
