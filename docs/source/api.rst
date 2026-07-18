API reference
=============

This reference lists the public modules shipped by ``pgmuvi``.  The
pages below are generated from module docstrings and public members,
but they are grouped by workflow area so the reference remains
navigable as the package grows.

.. automodule:: pgmuvi
    :members:
    :undoc-members:
    :show-inheritance:

Core light-curve and GP model modules
-------------------------------------

.. toctree::
   :maxdepth: 1

   pgmuvi.lightcurve
   pgmuvi.gps
   pgmuvi.models

Fitting, kernels, constraints, and training
-------------------------------------------

.. toctree::
   :maxdepth: 1

   pgmuvi.trainers
   pgmuvi.priors
   pgmuvi.constraints
   pgmuvi.constraint_utils
   pgmuvi.kernels
   pgmuvi.initialization
   pgmuvi.synthetic

Parameter workflow modules
--------------------------

.. toctree::
   :maxdepth: 1

   pgmuvi.parameter_specs
   pgmuvi.parameter_context
   pgmuvi.parameter_estimates
   pgmuvi.parameter_builders
   pgmuvi.parameter_application
   pgmuvi.parameter_workflow

Data, diagnostics, and validation
---------------------------------

.. toctree::
   :maxdepth: 1

   pgmuvi.dtypes
   pgmuvi.preprocess
   pgmuvi.multiband_ls_significance
   pgmuvi.wavelength_conclusions
   pgmuvi.wavelength_diagnostics
   pgmuvi.wavelength_estimation
   pgmuvi.spectral_mixture_ard
   pgmuvi.spectral_mixture_ard_diagnostics
   pgmuvi.wavelength_hypotheses
   pgmuvi.wavelength_results
   pgmuvi.wavelength_status
   pgmuvi.wavelength_validation
   pgmuvi.wavelength_validation_synthetic
   pgmuvi.wavelength_validation_recovery
   pgmuvi.upload_validation
