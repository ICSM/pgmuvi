pgmuvi.wavelength_validation_synthetic
=======================================

This module builds deterministic, truth-preserving synthetic cases for the D1
wavelength-model recovery programme.  It does not run optimization or rank models.
Each case records the observed arrays, noiseless wavelength mean,
latent process, uncertainties, purpose-specific random seeds, coordinate
transform, generating parameters, and parameter ownership.

The canonical suite covers ``2DWavelengthDependent``, ``2DDustMean``,
``2DPowerLawMean``, ``2DSeparable``, and ``2D``.  It includes a supported
quadratic turning point and a known fundamental-plus-harmonic temporal case.
All generated observations remain in strictly positive linear flux.

Use :func:`canonical_synthetic_wavelength_validation_cases` for the nominal D1
scenario set, or :func:`make_synthetic_wavelength_validation_case` for an
explicit configuration.  A case creates a
:class:`~pgmuvi.lightcurve.Lightcurve` only when
:meth:`~pgmuvi.wavelength_validation_synthetic.SyntheticWavelengthValidationCase.to_lightcurve`
is called.

.. automodule:: pgmuvi.wavelength_validation_synthetic
   :members:
   :undoc-members:
   :show-inheritance:

Nominal sampling
----------------

Canonical D1 cases use 72 observations per band on a shared reproducible
irregular time grid spanning 3.2 generating periods.  The denser nominal design
is required because six-band wavelength-lengthscale recovery is driven more by
within-cycle sampling density than by extending a sparse baseline.  Independent,
36-point, uneven, and longer-but-sparser designs belong to the later D2
robustness matrix.
