Legacy wavelength-candidate diagnostics
=======================================

.. warning::

   This page documents the older candidate-based wavelength diagnostic and
   comparison workflow built around ``diagnose_wavelength_dependence`` and
   ``compare_wavelength_models``.  It is kept for users who need that API, but
   it is **not** the current recommended wavelength workflow.

   For the current workflow, use the period-independent wavelength advisory
   pages: :doc:`wavelength_advisory` for one source and
   :doc:`wavelength_advisory_batch` for batches.  The current workflow reports
   advisory ``model/kernel config`` rankings and keeps ``selected_model=None``;
   it does not automatically install a winning model.

.. contents:: On this page
   :local:
   :depth: 2

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
     - Pure spectral mixture kernel for single-band data; highly flexible.
       ``fit()`` does not choose this automatically; pass ``model='1D'`` explicitly.
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

Legacy Automatic Recommendation
-------------------------------

``pgmuvi`` provides a legacy convenience recommender based on data
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


Legacy Wavelength-Dependence Diagnostics for 2D Light Curves
----------------------------------------------------------------

For multiwavelength light curves, choosing a model is not only a question of
period recovery.  You often need to inspect whether the variability is shared
across bands, whether its amplitude changes smoothly with wavelength, and
whether the data show possible wavelength-dependent phase or lag structure.

.. note::

   This section documents the older staged diagnostic/comparison API. The newer
   period-independent wavelength advisory workflow introduced after PR56 uses
   model/kernel configurations and batch reports rather than automatic model
   selection. A dedicated guide for that current workflow is planned.

Use :meth:`~pgmuvi.lightcurve.Lightcurve.diagnose_wavelength_dependence` before
running a model grid.  This pre-fit stage is cheap: it builds a band-by-band
report with sampling, variability, flux, and robust-amplitude diagnostics.  If a
known period or frequency is supplied, it also measures period-locked amplitude,
phase, and time lag per wavelength::

    diag = lc.diagnose_wavelength_dependence(period=350.0)
    print(lc.format_wavelength_diagnostics_report(diag))

The report contains:

* ``band_table`` — one row per wavelength/band,
* ``summary`` — counts of usable/rejected bands,
* ``amplitude_phase_summary`` — fixed-frequency amplitude and lag summaries,
* ``classification`` — conservative diagnostic labels,
* ``recommended_candidate_models`` — candidate model families and reasons.

The recommendations are intentionally conservative.  They do not claim that a
model is correct; they identify plausible next fits.  Typical outcomes are:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Diagnostic pattern
     - Candidate family
   * - Same period, same amplitude and phase across bands
     - ``model="2DAchromatic"``
   * - Same period, smooth amplitude change with wavelength
     - ``model="2DWavelengthDependent"``
   * - Power-law-like wavelength trend
     - ``model="2DPowerLawMean"`` or ``model="2DPowerLaw"``
   * - Multiple shared components
     - ``fit_strategy="consensus_multicomp"`` with an appropriate final model
   * - Strong wavelength-dependent lag
     - Treat as a warning: current separable wavelength models do not explicitly
       encode deterministic wavelength-dependent delays.

After inspecting the pre-fit diagnostics, you can run the recommended follow-up
fits through the normal fitting pathway with
:meth:`~pgmuvi.lightcurve.Lightcurve.compare_wavelength_models`::

    comparison = lc.compare_wavelength_models(
        diagnostic_report=diag,
        base_fit_kwargs={
            "training_iter": 500,
            "miniter": 100,
            "learn_additional_noise": True,
            "verbose": True,
        },
        residual_diagnostic_kwargs={"period": 350.0},
    )

The comparison report records successful, failed, and skipped follow-up fits; fit
history summaries; learned-noise summaries; residual/predictive scores; and
quality flags.  A lower predictive score is useful evidence, but it is not a
standalone scientific decision.  Always inspect residuals, coverage, rejected
bands, and whether the score improvement is large enough to justify the more
complex model.

The reporting and plotting helpers consume existing reports only.  They do not
recompute diagnostics, run fits, or mutate the light curve::

    print(lc.format_wavelength_diagnostics_report(
        diag,
        comparison_report=comparison,
    ))
    diagnostic_figs = lc.plot_wavelength_diagnostics(diag, show=False)
    comparison_figs = lc.plot_wavelength_model_comparison(
        comparison,
        show=False,
    )

See ``examples/wavelength_model_selection_diagnostics.py`` for a complete
script using a synthetic 2D light curve.

For a deterministic smoke validation of the full public workflow, run::

    PYTHONPATH=. python3 examples/validate_wavelength_model_selection_workflow.py

This writes JSON, Markdown, and plot artifacts to
``wavelength_diagnostics_validation/``.  To also exercise optional follow-up GP fitting
and comparison scoring, run::

    PYTHONPATH=. python3 examples/validate_wavelength_model_selection_workflow.py \
        --run-model-comparison --training-iter 10

The validation script is intentionally an example/integration check rather than
a unit test; it is meant to confirm that the staged diagnostics, reporting,
plotting, and optional comparison APIs work together on a known chromatic
synthetic source.

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

Tutorial status
---------------

The old model-selection tutorial notebook was an unfinished skeleton and is no
longer listed in the public tutorial toctree. Updated wavelength-advisory and
consensus-fitting tutorials are planned.
