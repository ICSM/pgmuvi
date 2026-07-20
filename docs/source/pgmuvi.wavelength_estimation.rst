pgmuvi.wavelength_estimation
============================

This module builds the shared wavelength-estimation context used by the
parameter workflow.  It records:

* usable and excluded observational channels;
* total wavelength span and adjacent wavelength spacings;
* the largest wavelength gap and spacing irregularity;
* robust per-observational-channel median, scatter, and quantile amplitudes;
* fractional and uncertainty-corrected amplitude summaries;
* monotonicity indicators for median flux, amplitude, and scatter; and
* the number of observational channels and distinct physical wavelengths; and
* an auditable wavelength-length-scale value and interval.

Observational channels are identified independently of their numeric physical
wavelengths.  Multiple channels may share one wavelength, so channel-level
summaries and distinct physical wavelengths are reported separately.

**TBD[instrument-channel-calibration]:** No instrument-channel calibration is
performed.  When two or more usable observational channels share a physical
wavelength, PGMUVI preserves their channel-level diagnostics but excludes that
uncalibrated wavelength from cross-wavelength trend summaries and
wavelength-mean fits.  It does not silently combine flux summaries, infer
offsets or scales, or assign artificial wavelength differences.

The diagnostics retain the length-scale recommendation in the **raw
wavelength coordinate** and also record the corresponding value and interval in
the coordinate supplied to the GP.  The scale-only transformation is used, so
origin shifts do not alter a lengthscale while MinMax, Z-score, and robust
Z-score transforms rescale it consistently with the model input.

For separable wavelength kernels, ``GuessStrategy.WAVELENGTH_RANGE`` and
``ConstraintStrategy.WAVELENGTH_RANGE`` now apply the model-coordinate value and
bounds through the parameter workflow.  Existing registered bounds are
intersected rather than replaced.  Application provenance records the raw and
model-coordinate recommendations, the transform, the proposed interval, and
the effective registered interval.

This applies to the wavelength covariance in ``2DWavelengthDependent``,
``2DDustMean``, ``2DPowerLawMean``, and ``2DSeparable``.  The
full non-separable ``2D`` baseline consumes the same sampling context through the
dimension-aware spectral-mixture ARD layer documented in
:mod:`pgmuvi.spectral_mixture_ard`.

The same module now builds model-ready wavelength-mean recommendations from
robust per-observational-channel median fluxes.  ``GuessStrategy.WAVELENGTH_MEAN`` and
``ConstraintStrategy.WAVELENGTH_MEAN`` initialize and constrain:

* the quadratic bias and coefficients in ``2DWavelengthDependent``;
* the offset, amplitude, optical depth, and extinction index in
  ``2DDustMean``; and
* the offset, signed amplitude, and exponent in ``2DPowerLawMean``.

The quadratic coefficients are fitted in the wavelength coordinate seen by the
GP.  Dust and power-law means instead evaluate on reconstructed physical,
strictly positive wavelength while their flux parameters remain in the actual
training-target coordinate.  Affine input transforms are supported and recorded;
a non-affine wavelength transform is rejected rather than silently changing the
physical interpretation.  Every affected mean parameter uses a registered
GPyTorch raw-parameter interval, so the reported bounds remain active during
optimization.

For three or more non-flat physical-wavelength evidence points, the power-law recommendation now retains the
full exponent-profile fit and converts its same-sign, near-optimal support into
finite data-derived intervals for offset, signed amplitude, and exponent.  This
prevents the covariance from absorbing a clearly resolved static wavelength
trend by driving the power-law amplitude toward zero or the exponent toward a
flat solution.  The profile criterion, supported parameter ranges, padded
active intervals, and effective registered constraints remain available in
parameter-workflow provenance.

With only two usable physical-wavelength evidence points, the power-law exponent is not identifiable jointly
with its offset and amplitude.  The estimator therefore keeps the documented
``-2`` default and fits only the linear coefficients instead of selecting an
arbitrary grid endpoint.  An exactly flat wavelength-median trend is not promoted to
a dust-shape estimate, and neither under-identified case receives a fabricated
profile interval.

No logarithmic flux fitting is introduced by this workflow.  Explicit
``log_*`` mean parameters receive positive physical estimates and are converted
to their stored log parameterization only at application time.

.. automodule:: pgmuvi.wavelength_estimation
    :members:
    :undoc-members:
    :show-inheritance:
