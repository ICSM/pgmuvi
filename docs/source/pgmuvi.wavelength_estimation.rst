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

The length-scale recommendation is expressed in the **raw wavelength
coordinate**.  It is not transformed into the model-input coordinate and is
not applied to any GP model by this package.  The
``GuessStrategy.WAVELENGTH_RANGE`` and
``ConstraintStrategy.WAVELENGTH_RANGE`` builders only make the recommendation
available to later model-specific work.

No logarithmic flux transformation is used when constructing these summaries.

.. automodule:: pgmuvi.wavelength_estimation
    :members:
    :undoc-members:
    :show-inheritance:
