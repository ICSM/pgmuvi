PGMUVI package orientation
==========================

This page gives the high-level orientation that should come before choosing a
model or running a fit. It is intentionally short enough to read before the
workflow-specific how-to guides.

What PGMUVI is for
------------------

``pgmuvi`` models astronomical light curves with Gaussian processes. Its main
use case is a source observed at one or more wavelengths or photometric bands,
where the user wants to infer variability properties such as characteristic
periods, wavelength-dependent amplitudes, and whether different bands share a
common variability structure.

The package is built on GPyTorch and exposes high-level ``Lightcurve`` methods
so users can start with CSV files and move gradually toward more specialized
models. The package can be used outside astronomy, but the current examples and
model families are written for astronomical time-series workflows.

The three questions to ask first
--------------------------------

Before fitting, decide which of these questions is closest to the task:

1. **Is there variability in a single time series?**

   Start with a one-dimensional ``Lightcurve`` and the basic fitting or
   period-summary tools. This is the simplest workflow and is useful for a
   single band or a source whose wavelength dependence is not part of the
   question.

2. **Do multiple bands share a coherent time-domain variability pattern?**

   Start with a two-dimensional ``Lightcurve`` and a consensus fit using
   ``model="2D"``. For many long-period-variable use cases, this is the safest
   baseline because it lets the time-domain structure be learned from the data
   while retaining per-band information.

3. **Does the variability change systematically with wavelength?**

   Start with the wavelength advisory workflow. It combines descriptive
   period-independent wavelength diagnostics with advisory model/kernel-config
   fits. It does not select or install a model automatically; it tells you which
   wavelength-dependent model families deserve closer inspection.

Important model families
------------------------

``1D``
    A single time-series GP. Use this when each band is being analyzed
    independently or when wavelength is not part of the question.

``2D``
    The baseline multiwavelength model. Use this as the first multiband
    reference fit, especially with ``fit_strategy="consensus"`` for unevenly
    sampled multi-band light curves.

``2DWavelengthDependent``
    A separable-style wavelength-dependent model with configurable time and
    wavelength kernels. Consider this when the data suggest a smooth wavelength
    trend but not a specific physical mean-function shape.

``2DDustMean`` and ``2DPowerLawMean``
    Wavelength-dependent model families with more structured mean behavior.
    These are useful candidates for long-period-variable applications when the
    mean level and variability amplitude have physically interpretable
    wavelength trends.

``2DSeparable``
    A model family for explicitly separable time and wavelength covariance
    structure. Use it when a shared time-domain process modulated by wavelength
    is the scientific hypothesis being tested.

First-choice workflow
---------------------

For a new multiwavelength source, the recommended order is:

1. Load the data with ``Lightcurve.from_csv`` and inspect warnings about finite
   values, positive uncertainties, bands, wavelengths, and sampling.
2. Run period-independent wavelength diagnostics to summarize per-band mean,
   scatter, and robust amplitude trends.
3. Run the advisory workflow if wavelength dependence is scientifically
   relevant.
4. Fit the conservative baseline ``model="2D"`` with
   ``fit_strategy="consensus"`` and ``learn_additional_noise=True``.
5. Compare LPV-relevant wavelength-dependent model families such as
   ``2DDustMean``, ``2DPowerLawMean``, and ``2DWavelengthDependent`` when the
   diagnostics support doing so.
6. Inspect the report, fit-quality score, failure/fallback diagnostics, and any
   constrained spectral-mixture ARD scale diagnostics before drawing physical
   conclusions.

What PGMUVI does not decide automatically
-----------------------------------------

The package can generate diagnostics and advisory comparisons, but it does not
currently perform final automatic scientific model selection. The user remains
responsible for checking sampling quality, period plausibility, wavelength-trend
interpretation, fit residuals, and whether a model family is physically
appropriate for the source.

TBD
---

The package still needs fuller documentation and implementation support for
multi-periodic model guidance, physically motivated non-monotonic wavelength
kernels, and a formal model-selection framework. The current advisory workflow
should therefore be treated as a triage and interpretation aid, not as an
automated decision engine.

Next pages
----------

- :doc:`howto/first_workflow` gives the first practical workflow and points to a
  runnable example script.
- :doc:`howto/loading_data` and :doc:`howto/preprocessing` cover input handling
  and validation details.
- :doc:`howto/wavelength_advisory` covers wavelength-dependent diagnostics and
  advisory model/kernel-config fitting.
- :doc:`documentation_roadmap` tracks the broader documentation expansion plan.
