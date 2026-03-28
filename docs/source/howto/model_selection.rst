Choosing a Model
================

This guide explains how to choose an appropriate GP model for your data using
``pgmuvi``.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

``pgmuvi`` provides several GP model variants.  The right model depends on the
character of the variability you expect and on the quality of your data.
Choosing a model that is too simple will miss real structure; choosing one that is
too complex risks over-fitting and slow convergence.

Available Models
-----------------

The ``model`` parameter of the :class:`~pgmuvi.lightcurve.Lightcurve` constructor
(or :meth:`~pgmuvi.lightcurve.Lightcurve.set_model`) selects the kernel structure:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model identifier
     - Description
   * - ``'spectral_mixture'``
     - Default.  Pure spectral mixture kernel; highly flexible.
   * - ``'spectral_mixture_rbf'``
     - Spectral mixture + RBF (squared-exponential) component.  Useful when a
       smooth long-term trend is expected alongside quasi-periodic variability.
   * - ``'spectral_mixture_flicker'``
     - Spectral mixture + :math:`1/f` noise.  Suitable for AGN and other sources
       with red-noise dominated PSDs.
   * - ``'periodic'``
     - Exact periodic (cosine) kernel.  Use only when the signal is strictly
       periodic.
   * - ``'quasi_periodic'``
     - Periodic × RBF: the standard quasi-periodic kernel.  Captures a single
       periodic signal that decays in coherence over time.

Number of Mixture Components
-----------------------------

The ``num_mixtures`` parameter controls how many spectral components are included.
As a rule of thumb:

* Start with ``num_mixtures=4`` (the default) for exploratory analysis.
* Increase if the data show evidence for multiple distinct periodicities.
* Decrease if the optimisation struggles to converge (fewer parameters = simpler
  optimisation landscape).

The Bayesian Information Criterion (BIC) or Leave-One-Out cross-validation can be
used for formal model comparison, but visual inspection of the PSD and residuals is
often sufficient.

Automatic Model Selection
--------------------------

``pgmuvi`` provides an automated model selection method based on data
characteristics::

    recommended = lc.auto_select_model()
    print(recommended)

Internally, :meth:`~pgmuvi.lightcurve.Lightcurve.auto_select_model` evaluates:

* the strength and consistency of the Lomb–Scargle periodogram peak(s),
* inter-band agreement in peak frequency (for multiband data),
* noise level relative to variability amplitude.

It returns the identifier of the recommended model, which can then be passed to
:meth:`~pgmuvi.lightcurve.Lightcurve.set_model`.

Manual Model Selection
-----------------------

If you have domain knowledge about the expected variability type, you can set the
model directly::

    lc.set_model('spectral_mixture_flicker', num_mixtures=3)
    lc.fit_LS()

Alternative Kernel Configurations
-----------------------------------

Advanced users can construct custom kernel combinations by working directly with
GPyTorch kernels and passing them to :meth:`~pgmuvi.lightcurve.Lightcurve.set_model`.
See the ``alternative_kernels_1d.py`` example script for an illustration.

.. seealso::

   :mod:`pgmuvi.gps` — GP model classes.

   :mod:`pgmuvi.kernels` — Custom kernel definitions.

Model Selection Tutorial
-------------------------

A notebook tutorial on model selection is provided in the User Guide:

.. toctree::
   :maxdepth: 1

   ../notebooks/tutorial_model_selection
