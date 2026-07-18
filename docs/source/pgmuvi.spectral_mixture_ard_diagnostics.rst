pgmuvi.spectral_mixture_ard_diagnostics
=======================================

The fitted full-``2D`` spectral-mixture diagnostic reads both
``mixture_means`` and ``mixture_scales`` by component and ARD dimension.  It
reports the registered lower and upper bounds, distance to each bound,
normalized interval position, raw GPyTorch parameter values, and fitted values
in both model-input and raw-input coordinates when PR124 provenance is
available.

The coordinate order is:

* index 0: temporal frequency;
* index 1: wavelength frequency.

``num_mixtures_is_one`` records the fitted kernel shape.
``num_mixtures_fixed_at_one`` is true only when the caller explicitly supplied
``num_mixtures=1``.  This distinction matters because a single component cannot
show component-to-component stability.

A ``boundary_hit`` means a fitted value is near or at a registered constraint
under the recorded tolerance.  It is not automatic evidence for or against a
physical wavelength dependence, and the diagnostic does not select a model.

.. automodule:: pgmuvi.spectral_mixture_ard_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:
