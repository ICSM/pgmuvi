Generate mock light curves from a GP prior
=============================================

This guide describes the advanced workflow for drawing reproducible mock light
curves from an explicitly configured PGMUVI Gaussian-process prior.  It is
separate from both analytic mock generators and fitted posterior predictions.

Choose the right workflow
-------------------------

Use the three workflows for different purposes:

* :mod:`pgmuvi.synthetic` creates analytic sinusoidal signals with explicitly
  injected periods, phases, amplitudes, and wavelength laws.  Use it when the
  deterministic signal itself should be known.
* GP-prior sampling draws a random latent realization whose covariance follows
  a selected kernel and user-specified hyperparameters.  No observed data are
  conditioned on and no optimizer is run.
* Posterior predictive sampling is performed only after fitting a model to
  observations.  It represents uncertainty conditional on those observations
  and is not interchangeable with a prior draw.

For an executable walkthrough, use
:doc:`../notebooks/pgmuvi_mock_data_from_gp`.  The equivalent standalone script
is ``examples/gp_prior_sampling.py``.

1. Use the package dtype and physical time units
------------------------------------------------

PGMUVI uses :data:`pgmuvi.dtypes.DEFAULT_DTYPE`, currently ``torch.float64``.
Create the time grid in the same dtype and record its units explicitly.  The
examples use days throughout.

.. code-block:: python

   import numpy as np
   import torch

   from pgmuvi.dtypes import DEFAULT_DTYPE

   seed = 20260715
   rng = np.random.default_rng(seed)
   time = torch.as_tensor(
       np.sort(rng.uniform(0.0, 720.0, 72)),
       dtype=DEFAULT_DTYPE,
   )

2. Instantiate the model class directly
----------------------------------------

The direct constructors inherit from GPyTorch ``ExactGP`` and therefore require
placeholder training inputs and values.  In this workflow the placeholder values
only satisfy the constructor; the latent prior is requested directly from the
model's ``forward`` method and no fit is started.

.. code-block:: python

   import gpytorch

   from pgmuvi.gps import QuasiPeriodicGPModel

   placeholder = torch.sin(2.0 * torch.pi * time / 180.0)
   likelihood = gpytorch.likelihoods.GaussianLikelihood()
   model = QuasiPeriodicGPModel(
       time,
       placeholder,
       likelihood,
       period=180.0,
   )

The maintained examples also cover :class:`pgmuvi.gps.MaternGPModel` and
:class:`pgmuvi.gps.SpectralMixtureGPModel`.

3. Apply physical hyperparameters through the parameter layer
--------------------------------------------------------------

Do not write GPyTorch ``raw_*`` parameters directly.  Read the model's
``parameter_schema()``, construct :class:`pgmuvi.parameter_estimates.ParameterEstimate`
objects in physical space, and use
:func:`pgmuvi.parameter_workflow.apply_parameter_estimates`.

.. code-block:: python

   from pgmuvi.parameter_estimates import ParameterEstimate
   from pgmuvi.parameter_estimates import ParameterEstimateCollection
   from pgmuvi.parameter_workflow import apply_parameter_estimates

   schema = model.parameter_schema()
   values = {
       "covar_module.outputscale": 1.4,
       "covar_module.base_kernel.kernels.0.period_length": 180.0,
       "covar_module.base_kernel.kernels.1.lengthscale": 650.0,
   }
   estimates = ParameterEstimateCollection(
       [
           ParameterEstimate(
               spec=schema[name],
               value=value,
               value_source="user_specified_gp_prior",
           )
           for name, value in values.items()
       ]
   )
   application_report = apply_parameter_estimates(model, estimates)

Check that every requested value reports ``"value": True`` before sampling.
The application layer preserves parameter transformations and shape checks and
avoids bypassing the package's constraint machinery.

Kernel-specific quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~

``QuasiPeriodicGPModel``
   Record the period, long-term coherence lengthscale, and output variance.

``MaternGPModel``
   Record ``nu``, the time-domain lengthscale, and output variance.  This
   represents stochastic/red-noise variability rather than a periodic signal.

``SpectralMixtureGPModel``
   ``mixture_means`` are frequencies, not periods.  Convert each injected period
   with ``frequency = 1 / period``.  ``mixture_scales`` are frequency-space
   widths, while ``mixture_weights`` are positive variance contributions.

4. Draw the latent prior
------------------------

Use a fixed Torch seed, switch the model to evaluation mode, and call
``model.forward(time).sample()`` inside ``torch.no_grad()``.

.. code-block:: python

   model.eval()
   torch.manual_seed(seed + 101)
   with torch.no_grad():
       latent_flux = model.forward(time).sample()

This is a latent-function draw.  The Gaussian likelihood is not being used to
condition on data or to add measurement noise.

5. Add observational noise and create a Lightcurve
--------------------------------------------------

Add measurement noise explicitly and retain the latent realization separately
when recovery tests need access to the injected truth.

.. code-block:: python

   from pgmuvi.lightcurve import Lightcurve

   noise_sigma = 0.12
   torch.manual_seed(seed + 202)
   uncertainty = torch.full_like(latent_flux, noise_sigma)
   observed_flux = latent_flux + noise_sigma * torch.randn_like(latent_flux)

   lightcurve = Lightcurve(
       time,
       observed_flux,
       yerr=uncertainty,
       max_samples=None,
       check_sampling=False,
       name="quasi_periodic_gp_prior_mock",
   )

The constructor applies the normal default time-centering transform for future
fitting.  The raw time values remain available in ``lightcurve.xdata``.

6. Record enough information to reproduce the mock
---------------------------------------------------

At minimum record:

* model and kernel family;
* all physical hyperparameter values;
* observation-time array and time units;
* Torch and NumPy seeds;
* observational-noise model and scale;
* dtype;
* latent realization if exact point-by-point comparison is required; and
* package revision used to generate the sample.

A GP-prior realization is random.  It will not necessarily display an obvious
period in every finite, irregularly sampled draw even when a periodic component
is present.  Recovery should therefore be assessed across multiple seeds and
sampling patterns rather than from one favorable realization.

Prior sampling is not fitting
-----------------------------

This workflow does not call :meth:`pgmuvi.lightcurve.Lightcurve.fit`, does not
estimate hyperparameters, and does not provide posterior uncertainty.  To fit a
mock or observed light curve, continue with :doc:`consensus_fitting`.  To
interpret fitted period, peak, fit-quality, ARD, and failure reports, use
:doc:`interpreting_results`.

Posterior predictive samples require a successfully fitted model and its
likelihood.  They should be documented and validated separately from this
no-training prior-sampling workflow.
