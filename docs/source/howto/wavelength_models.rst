Choosing wavelength-dependent model families
=============================================

.. note::

   **Documentation scope:** this page explains the current model-family roles
   and recommended comparison strategy.  It does not add model-selection
   functionality or declare any model scientifically preferred.

   For the fitting mechanics, see :doc:`consensus_fitting`.  For the current
   non-selecting comparison workflow, see :doc:`wavelength_advisory`.

.. contents:: On this page
   :local:
   :depth: 2

The central distinction: mean versus covariance
-----------------------------------------------

A multiwavelength GP has two conceptually different jobs:

``mean function``
   Describes the average flux level as a function of wavelength.  The
   wavelength-dependent mean can be constant, polynomial, dust-inspired, or a
   power law depending on the model family.

``covariance function``
   Describes correlated departures from the mean.  It controls the temporal
   variability, the wavelength correlation scale, and whether time and
   wavelength are modeled jointly or as a product of separate kernels.

A model may reproduce the mean spectral shape while fitting the variability
poorly, or fit the variability while using an implausible mean.  Inspect both.
Do not treat a wavelength-dependent mean as proof that the same physical law
controls the variability amplitude.

Input-coordinate requirements
-----------------------------

All models on this page require two-dimensional input with time in column 0 and
numeric wavelength in column 1.  See :doc:`loading_data` and :doc:`multiband`
for construction and validation details.

Use physical, strictly positive wavelengths when fitting ``2DDustMean`` or
``2DPowerLawMean``.  Integer band codes do not carry meaningful distances for
these parametric wavelength means.  They are also a poor default for smooth
RBF, Matérn, or rational-quadratic wavelength kernels because the inferred
length scale then depends on an arbitrary coding scheme.

The fitted wavelength-kernel length scale and mean parameters operate in the
coordinate system seen by the GP.  If a wavelength transform is active, do not
interpret those parameters as raw-micron physical quantities without accounting
for that transform.

Model-family map
----------------

.. list-table:: Current LPV-relevant multiwavelength model families
   :header-rows: 1
   :widths: 16 22 27 35

   * - Model
     - Mean structure
     - Covariance structure
     - Recommended role
   * - ``2D``
     - Constant mean.
     - Non-separable two-dimensional spectral-mixture kernel over joint time
       and wavelength frequency space.
     - Broad comparison baseline when the separability assumption may be too
       restrictive.  It has more difficult-to-interpret joint ARD behavior and
       should not be given ``time_kernel_type`` or ``wavelength_kernel_type``
       selectors.
   * - ``2DSeparable``
     - Constant mean by default.
     - Product kernel,
       ``k_time(t,t') * k_wavelength(lambda,lambda')``; defaults to a Matérn
       time kernel and RBF wavelength kernel.
     - Generic direct-fit control when a product covariance is appropriate and
       the per-band mean has already been normalized or does not need a
       parametric wavelength law.  Its convenience interface does not use the
       string kernel selectors documented for ``2DWavelengthDependent``.
   * - ``2DWavelengthDependent``
     - Quadratic-in-wavelength mean by default; linear, constant, dust, power-law,
       or a supplied GPyTorch mean can also be requested.
     - Separable time and smooth wavelength kernels.  The wavelength kernel can
       be RBF, Matérn-3/2, or rational quadratic.
     - Flexible first separable model when the wavelength trend is flat,
       uncertain, or not well described by a monotonic parametric mean.  The
       default quadratic mean can represent one broad turning point, not an
       arbitrary non-monotonic spectrum.
   * - ``2DDustMean``
     - ``amplitude * exp(-tau * lambda**(-alpha)) + offset`` with positive
       amplitude, ``tau``, and ``alpha`` parameterized in log space.
     - Same configurable separable covariance family as
       ``2DWavelengthDependent``.
     - LPV comparison model when the mean flux rises toward longer wavelength
       in a way plausibly associated with short-wavelength attenuation.
       Treat it as dust-inspired unless the data units, wavelength coordinate,
       transforms, and physical assumptions justify stronger interpretation.
   * - ``2DPowerLawMean``
     - ``offset + weight * lambda**exponent``; the signed weight and exponent
       permit increasing or decreasing monotonic trends.
     - Same configurable separable covariance family as
       ``2DWavelengthDependent``.
     - Simple parametric comparison for approximately power-law wavelength
       behavior.  It is less restrictive in trend direction than
       ``2DDustMean`` but cannot reproduce arbitrary curvature.

``2DSeparable`` is not part of the default period-independent advisory ranking.
Use it manually as a direct product-kernel control when that comparison is
scientifically useful.  The default advisory candidates are
``2DWavelengthDependent``, ``2DDustMean``, and ``2DPowerLawMean``, with the
optional ``2D`` baseline.

What should I try first?
------------------------

Unknown multiwavelength behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with ``2D`` as a broad joint baseline and
``2DWavelengthDependent`` as the flexible separable comparison.  If the two
fits differ materially, inspect whether the discrepancy is in the mean,
wavelength covariance, temporal structure, or numerical stability before
adding more models.

LPV with mean flux increasing toward the infrared
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare ``2DDustMean``, ``2DPowerLawMean``, and
``2DWavelengthDependent``.  Retain ``2D`` as a baseline when computationally
practical.  A rising median-flux trend is a reason to test ``2DDustMean``; it is
not evidence that dust attenuation has been uniquely identified.

Monotonic decrease with wavelength
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try ``2DPowerLawMean`` first, then ``2DWavelengthDependent``.  The current
advisory logic gives ``2DDustMean`` lower priority because its positive
amplitude, optical depth, and index naturally produce a mean that rises with
wavelength over the usual positive wavelength domain.

Flat, weak, or poorly constrained wavelength trend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``2DWavelengthDependent`` as the flexible separable baseline.  Compare a
constant-mean configuration or ``2DSeparable`` when the mean has already been
normalized per band and the scientific question concerns covariance only.

Non-monotonic wavelength structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``2DWavelengthDependent`` before forcing a monotonic dust or power-law mean.
The current quadratic default is only modestly flexible.  Strongly structured
or multi-turn wavelength behavior is a future-kernel problem, not a reason to
over-interpret a poor quadratic fit.

Choosing the temporal kernel for separable LPV models
-----------------------------------------------------

For ``2DWavelengthDependent``, ``2DDustMean``, and ``2DPowerLawMean``:

``time_kernel_type="quasi_periodic"``
   Preferred current starting point for an LPV-like source with one dominant
   shared period and evolving coherence.  With ``fit_strategy="consensus"``,
   the consensus frequency is converted to the time kernel's
   ``period_length`` parameter.

``time_kernel_type="spectral_mixture"``
   Use when the temporal PSD needs a more flexible representation or when the
   data support more than one spectral component.  Establish that need from
   diagnostics rather than increasing ``num_mixtures`` automatically.

``time_kernel_type="matern"`` or ``"rbf"``
   Valid for direct aperiodic fits.  These kernels expose neither a
   spectral-mixture frequency target nor a periodic ``period_length`` target,
   so they are not the normal consensus handoff path.

A representative LPV fit is:

.. code-block:: python

   result = lc.fit(
       model="2DDustMean",
       time_kernel_type="quasi_periodic",
       wavelength_kernel_type="rbf",
       fit_strategy="consensus",
       training_iter=500,
       miniter=100,
       learn_additional_noise=True,
       constraint_set="LPV",
       verbose=True,
   )

For the non-separable baseline, omit the separable-kernel selectors:

.. code-block:: python

   baseline = lc.fit(
       model="2D",
       fit_strategy="consensus",
       training_iter=500,
       miniter=100,
       learn_additional_noise=True,
       constraint_set="LPV",
       verbose=True,
   )

Do not compare several fitted families by repeatedly overwriting one
``Lightcurve`` and inspecting only the final state.  Use independent light-curve
instances or the advisory runner, which isolates model/kernel-config fits and
records success and failure metadata for each configuration.

How the current advisory plan uses wavelength trends
----------------------------------------------------

The period-independent advisory plan currently applies transparent heuristics
to raw per-band summaries:

.. list-table:: Current advisory ordering heuristics
   :header-rows: 1

   * - Diagnostic pattern
     - First advisory candidate
     - Comparisons retained
   * - Non-monotonic median or robust-amplitude trend
     - ``2DWavelengthDependent``
     - ``2DDustMean`` and ``2DPowerLawMean``
   * - Positive median-flux log-log slope
     - ``2DDustMean``
     - ``2DPowerLawMean`` and ``2DWavelengthDependent``
   * - Negative median-flux log-log slope
     - ``2DPowerLawMean``
     - ``2DWavelengthDependent`` and ``2DDustMean``
   * - Flat or unavailable trend
     - ``2DWavelengthDependent``
     - ``2DDustMean`` and ``2DPowerLawMean``

These are triage rules, not statistical model selection.  They create an
ordered list of models worth evaluating and preserve the alternatives rather
than excluding them.

What the advisory workflow does not decide
------------------------------------------

The current advisory workflow:

* keeps ``advisory_only=True`` and ``selected_model=None``;
* does not install a top-ranked fit on the input light curve;
* keeps advisory planning metadata separate from fitting decisions; when a
  separable candidate is actually fitted, the parameter workflow may apply the
  data-derived wavelength-covariance lengthscale and bounds, but this does not
  make the advisory ranking automatic model selection;
* ranks successful configurations using descriptive training-space residual
  diagnostics, not held-out predictive performance, Bayes factors, or evidence;
* cannot turn a failed fit into evidence that the model family is scientifically
  wrong; and
* cannot establish that fitted dust or power-law mean parameters are uniquely
  physical.

Read :doc:`wavelength_advisory` for the staged single-source workflow and
:doc:`wavelength_advisory_batch` for output folders, summaries, reports, and
failure triage across many sources.

Data-derived wavelength covariance initialization
-------------------------------------------------

For ``2DWavelengthDependent``, ``2DDustMean``, ``2DPowerLawMean``, and
``2DSeparable``, the parameter workflow now initializes the separable
wavelength-kernel lengthscale from the usable wavelength sampling.  The raw
recommendation combines the median adjacent-band spacing and total wavelength
span, with bounds informed by the minimum spacing, largest gap, and full span.

The raw value and bounds are converted using only the scale part of the active
input transform before they are registered on the GP kernel.  A pure time or
coordinate-origin shift leaves the wavelength lengthscale unchanged; affine
rescaling changes it into the model coordinate.  Existing kernel constraints
are intersected rather than overwritten, and the parameter-workflow result
records both the proposed and effective bounds under
``wavelength_estimate_provenance``.

This is an initialization and optimization-domain improvement, not evidence
that a source has resolved wavelength dependence.  It does not apply to the
full non-separable ``2D`` spectral-mixture kernel, whose temporal and wavelength
ARD entries still share tensor-wide constraints.

Data-derived wavelength mean initialization
-------------------------------------------

The three wavelength-dependent mean families now use robust per-band median
fluxes to construct initial values and finite optimization intervals.  The
parameter workflow records these under
``wavelength_mean_estimate_provenance``.

``2DWavelengthDependent``
   Fits the quadratic bias and two coefficients in the transformed wavelength
   coordinate actually supplied to the GP.  The coefficient intervals scale
   with the robust target range and transformed wavelength span.

``2DDustMean``
   Fits a coarse positive dust-attenuation profile using physical wavelength
   and the transformed training target.  Positive amplitude, optical depth,
   and extinction-index estimates are converted to the model's explicit
   ``log_*`` parameters only when applied.

``2DPowerLawMean``
   Fits ``offset + weight * wavelength**exponent`` using physical wavelength
   and the transformed training target.  The signed weight is retained and the
   exponent is constrained to a finite physical search domain.

For dust and power-law means, an affine wavelength transform is inverted inside
the mean module.  MinMax, Z-score, robust Z-score, and pure coordinate shifts
therefore do not turn physical wavelength into a zero, negative, or otherwise
misinterpreted base.  A custom non-affine wavelength transform is rejected for
these physical means.  The intervals are registered on raw GPyTorch parameters
and remain active throughout optimization.

Interpreting model-specific diagnostics
---------------------------------------

For ``2D``
   Inspect spectral-mixture ARD scale-ceiling diagnostics separately for the
   ``time_frequency`` and ``wavelength_frequency`` dimensions.  Saturation is a
   warning about constraint pressure or identifiability, not proof of a useful
   wavelength dependence.

For separable models
   Inspect the time-kernel parameters, wavelength-kernel length scale, mean
   parameters, learned additional noise, residuals by band, and whether the
   same result is recovered under longer training or repeated initialization.

For every family
   Inspect failed and skipped configurations alongside successful ones.  A
   consensus failure may reflect incoherent sampling or period diagnostics; a
   numerical failure may reflect optimization or constraints; an input failure
   may indicate invalid fluxes, errors, wavelengths, or too few usable bands.

Known limitations and future work
---------------------------------

The following are explicit future-work items rather than capabilities implied by
this guide:

* **TBD[multi-periodic-wavelength-models]:** validated support and guidance for
  sources with multiple shared temporal components.
* **TBD[non-monotonic-wavelength-kernels]:** wavelength kernels or mean families
  with physically interpretable non-monotonic structure beyond the current
  quadratic default.
* **TBD[automatic-model-selection]:** held-out or evidence-based automatic model
  selection.  The current advisory ranking must remain non-selecting.
* **TBD[physical-wavelength-kernels]:** physically motivated
  wavelength-dependence kernels whose parameters can be interpreted under
  documented coordinate and transform conventions.
* **TBD[wavelength-dependent-lags]:** models that explicitly encode deterministic
  phase or time delays as a function of wavelength.
