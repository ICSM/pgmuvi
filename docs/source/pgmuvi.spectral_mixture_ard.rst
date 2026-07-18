pgmuvi.spectral_mixture_ard
=================================

The full non-separable ``2D`` spectral-mixture kernel stores both ARD
coordinates in tensors whose last axis is ordered as:

* index 0: temporal frequency;
* index 1: wavelength frequency.

PGMUVI now derives separate initial values and finite bounds for both
``mixture_means`` and ``mixture_scales``.  The registered GPyTorch intervals
have shape ``(1, 1, 2)`` and broadcast across mixture components, so the two
coordinates no longer share a scalar bound.

Temporal bounds are derived from the model-coordinate time span and cadence.
Wavelength bounds are derived independently from wavelength span, adjacent-band
spacing, gap structure, and the wavelength-lengthscale recommendation.  Raw and
model-coordinate values are retained in
``spectral_mixture_ard_provenance``.

This parameterization improves initialization and constraint integrity.  It is
not evidence that wavelength dependence is resolved, and a wavelength value at
or near a bound must be interpreted through the separate saturation
diagnostics planned under ``TBD[wavelength-derived-constraints]``.

.. automodule:: pgmuvi.spectral_mixture_ard
   :members:
   :undoc-members:
   :show-inheritance:
