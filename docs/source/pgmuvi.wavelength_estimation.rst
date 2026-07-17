pgmuvi.wavelength_estimation
============================

This module builds the shared wavelength-estimation context used by the
parameter workflow.  It records:

* usable and excluded bands;
* total wavelength span and adjacent wavelength spacings;
* the largest wavelength gap and spacing irregularity;
* robust per-band median, scatter, and multiple quantile amplitudes;
* fractional and uncertainty-corrected amplitude summaries;
* monotonicity indicators for median flux, amplitude, and scatter; and
* an auditable wavelength-length-scale value and interval.

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
``2DDustMean``, ``2DPowerLawMean``, and ``2DSeparable``.  It does not change the
full non-separable ``2D`` spectral-mixture ARD parameterization, and it does not
yet initialize wavelength-dependent mean parameters.

No logarithmic flux transformation is used when constructing these summaries.

.. automodule:: pgmuvi.wavelength_estimation
    :members:
    :undoc-members:
    :show-inheritance:
