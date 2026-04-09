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

The kernel structure is selected with
:meth:`~pgmuvi.lightcurve.Lightcurve.set_model` or by passing ``model=`` to
:meth:`~pgmuvi.lightcurve.Lightcurve.fit`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model identifier
     - Description
   * - ``'1D'``
     - Default.  Pure spectral mixture kernel for single-band data;
       highly flexible.
   * - ``'2D'``
     - Spectral mixture kernel for 2D (multiband) data with a time and
       wavelength/band dimension.
   * - ``'1DLinear'``
     - Spectral mixture with a linear mean function; useful when a
       long-term linear trend is expected.
   * - ``'1DQuasiPeriodic'``
     - Quasi-periodic (Periodic × RBF) kernel.  Captures a single
       periodic signal that decays in coherence over time.
   * - ``'1DPeriodicStochastic'``
     - Periodic kernel plus a stochastic (noise) component.
   * - ``'1DMatern'``
     - Matérn kernel.  Suitable for smooth but aperiodic variability.
   * - ``'2DSeparable'``
     - Separable 2D kernel (time × wavelength) for multiband data.
   * - ``'2DAchromatic'``
     - 2D model with achromatic (wavelength-independent) variability.
   * - ``'2DWavelengthDependent'``
     - 2D model where variability amplitude depends on wavelength.

For a complete list see :meth:`~pgmuvi.lightcurve.Lightcurve.set_model`.

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

    recommended, diagnostics = lc.auto_select_model()
    print(f"Recommended model: {recommended}")
    if diagnostics.get("reason"):
        print(f"Reason: {diagnostics['reason']}")

Internally, :meth:`~pgmuvi.lightcurve.Lightcurve.auto_select_model` evaluates:

* the strength and consistency of the Lomb–Scargle periodogram peak(s),
* inter-band agreement in peak frequency (for multiband data),
* noise level relative to variability amplitude.

It returns a tuple of ``(model_identifier, diagnostics_dict)``.  The identifier
can be passed directly to :meth:`~pgmuvi.lightcurve.Lightcurve.set_model`::

    lc.set_model(recommended)

Manual Model Selection
-----------------------

If you have domain knowledge about the expected variability type, you can set the
model directly::

    lc.set_model('1DQuasiPeriodic')
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
