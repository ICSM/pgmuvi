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
   * - ``'1DLinear'``
     - Spectral mixture with a linear mean function; useful when a
       long-term linear trend is expected.
   * - ``'1DSKI'``
     - Spectral mixture with SKI (Structured Kernel Interpolation)
       approximation; faster for large datasets.
   * - ``'1DLinearSKI'``
     - Spectral mixture + linear mean + SKI approximation.
   * - ``'1DQuasiPeriodic'``
     - Quasi-periodic (Periodic × RBF) kernel.  Captures a single
       periodic signal that decays in coherence over time.
   * - ``'1DLinearQuasiPeriodic'``
     - Quasi-periodic kernel with a linear mean function.
   * - ``'1DPeriodicStochastic'``
     - Periodic kernel plus a stochastic (noise) component.
   * - ``'1DMatern'``
     - Matérn kernel.  Suitable for smooth but aperiodic variability.
   * - ``'2D'``
     - 2-D spectral mixture kernel over (time, wavelength); does **not**
       factorise into separate time and wavelength components.
   * - ``'2DLinear'``
     - 2-D spectral mixture with a linear mean function.
   * - ``'2DSKI'``
     - 2-D spectral mixture with SKI approximation.
   * - ``'2DLinearSKI'``
     - 2-D spectral mixture + linear mean + SKI approximation.
   * - ``'2DPowerLaw'``
     - 2-D spectral mixture with a power-law mean function.
   * - ``'2DPowerLawSKI'``
     - 2-D spectral mixture + power-law mean + SKI approximation.
   * - ``'2DDust'``
     - 2-D spectral mixture with a dust-extinction mean function.
   * - ``'2DDustSKI'``
     - 2-D spectral mixture + dust mean + SKI approximation.
   * - ``'2DSeparable'``
     - Explicit product (separable) kernel: time kernel × wavelength kernel.
   * - ``'2DAchromatic'``
     - Separable 2D model with achromatic (wavelength-independent)
       variability amplitude.
   * - ``'2DWavelengthDependent'``
     - Separable 2D model where variability amplitude depends on wavelength.
   * - ``'2DDustMean'``
     - Separable 2D model with a dust-extinction mean function.
   * - ``'2DPowerLawMean'``
     - Separable 2D model with a power-law mean function.

For full parameter details see :meth:`~pgmuvi.lightcurve.Lightcurve.set_model`.

Number of Mixture Components
-----------------------------

The ``num_mixtures`` parameter controls how many spectral components are included.
As a rule of thumb:

* Start with ``num_mixtures=1`` or ``num_mixtures=2`` for initial exploratory
  analysis to reduce the risk of immediately over-fitting.
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
    lc.fit()

Alternative Kernel Configurations
-----------------------------------

Advanced users can construct a custom GP model class (a GPyTorch
:class:`~gpytorch.models.ExactGP` subclass configured with the desired kernel)
and pass the instantiated model object to
:meth:`~pgmuvi.lightcurve.Lightcurve.set_model`.
Note that :meth:`~pgmuvi.lightcurve.Lightcurve.set_model` accepts either a
model identifier string (e.g., ``'1D'``, ``'1DQuasiPeriodic'``) or a GP model
instance; it does not accept a bare kernel object.
See the ``alternative_kernels_1d.py`` example script for a full illustration.

.. seealso::

   :mod:`pgmuvi.gps` — GP model classes.

   :mod:`pgmuvi.kernels` — Custom kernel definitions.

Model Selection Tutorial
-------------------------

A notebook tutorial on model selection is provided in the User Guide:

.. toctree::
   :maxdepth: 1

   ../notebooks/tutorial_model_selection
