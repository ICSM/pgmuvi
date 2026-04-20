Frequently Asked Questions
==========================

.. contents:: On this page
   :local:
   :depth: 2

Installation
------------

**Q: pip install pgmuvi fails with a PyTorch error.**

``pgmuvi`` depends on ``torch`` and ``gpytorch``.  On some platforms (e.g., certain
Linux systems without a GPU), pip may install a CPU-only PyTorch build.  If you
encounter errors, install PyTorch first following the instructions at
https://pytorch.org/get-started/locally/ and then install ``pgmuvi``.

**Q: The docs say Python >=3.10 is required.  Will it work on 3.9?**

No.  ``pgmuvi`` uses Python 3.10+ syntax and typing features such as
PEP 604 union types (for example, ``X | None``).  If you need to use an
older Python version, please open an issue on the GitHub repository.

Data and Input
--------------

**Q: What time unit should I use?**

Any consistent unit is fine.  ``pgmuvi`` does not assume a specific time unit;
it only needs the ``xdata`` and ``ydata`` to be in consistent units.  The inferred
periods will be in the same unit as your input times.  Common choices are days or
Modified Julian Date (MJD).

**Q: Can I use pgmuvi with data in magnitudes rather than flux?**

Not yet natively.  The ``magnitudes`` option of
:class:`~pgmuvi.lightcurve.Lightcurve` is not currently implemented.
For now, convert magnitudes and their uncertainties to (relative) fluxes
before constructing the :class:`~pgmuvi.lightcurve.Lightcurve`.  A standard
conversion is :math:`f \propto 10^{-0.4\,m}`; see
:ref:`working-with-magnitudes` for a code example.

**Q: My data have irregular gaps.  Is that a problem?**

No — handling irregular sampling is one of the main strengths of GP-based methods.
Lomb–Scargle initialisation also handles irregular data naturally.

**Q: How many observations do I need?**

There is no hard minimum, but as a rough guide:

* At least 20–30 observations per band for reliable GP fitting.
* At least 10 observations per estimated period to resolve the periodicity.
* Bands with fewer than ~10 observations should be treated with caution.

Use :meth:`~pgmuvi.lightcurve.Lightcurve.assess_sampling_quality` for a
data-driven assessment.

Fitting and Convergence
------------------------

**Q: fit_LS() is very slow.  How can I speed it up?**

* Subsample dense datasets (see :doc:`howto/preprocessing`).
* Reduce ``num_mixtures`` (fewer parameters → faster optimisation).
* Move the light curve to a GPU before fitting: ``lc = lc.cuda()``.

**Q: The fit converges to a period that makes no physical sense.**

Check that the physical period lies within the detectable range (see
:meth:`~pgmuvi.lightcurve.Lightcurve.compute_sampling_metrics`) and that your
constraints allow it.  Also try running with ``num_mixtures=1`` first to find the
dominant period, then increase.

**Q: Should I use fit() or fit_LS()?**

:meth:`~pgmuvi.lightcurve.Lightcurve.fit_LS` runs a Lomb–Scargle periodogram
to identify candidate periods and can be used as a diagnostic tool.
:meth:`~pgmuvi.lightcurve.Lightcurve.fit` performs the actual GP MAP
optimisation.  For the most common workflow, call :meth:`~pgmuvi.lightcurve.Lightcurve.fit`
with ``use_mls_init=True`` (the default) so that MLS-based frequency seeding is
applied automatically before fitting.

**Q: fit() finds too many / too few mixture components.**

:meth:`~pgmuvi.lightcurve.Lightcurve.fit_LS` does not take a
``num_mixtures`` argument.  To change the number of mixture components, pass
``num_mixtures`` to :meth:`~pgmuvi.lightcurve.Lightcurve.fit` or to
:meth:`~pgmuvi.lightcurve.Lightcurve.set_model`::

    lc.fit(model="1D", num_mixtures=2)
    # or equivalently
    lc.set_model("1D", num_mixtures=2)
    lc.fit()

**Q: MCMC is very slow.**

MCMC is inherently more expensive than MAP estimation.  Suggestions:

* Run MAP first with :meth:`~pgmuvi.lightcurve.Lightcurve.fit` and use the MAP
  solution as the starting point for
  :meth:`~pgmuvi.lightcurve.Lightcurve.mcmc`.
* Reduce ``num_samples`` and ``warmup_steps`` for a quick initial run to check
  convergence behaviour.
* Use a GPU if available.

Periods and Interpretation
---------------------------

**Q: The inferred period is twice (or half) the expected period.**

This is a classical harmonic aliasing issue.  The periodogram may detect harmonics
rather than the fundamental frequency.  Try restricting the frequency range via a
constraint (see :doc:`howto/priors_constraints`) to encourage the model to find
the fundamental.

**Q: I see multiple peaks in the PSD.  Which one is the true period?**

The highest-weighted component typically corresponds to the dominant variability
timescale.  However, inspect the PSD shape carefully:

* Peaks near the Nyquist frequency or at the observing cadence may be sampling
  aliases.
* Peaks at integer multiples of the dominant period are harmonics.
* Very broad peaks (large ``mixture_scales``) indicate stochastic / quasi-periodic
  variability, not a precise period.

**Q: How do I get period uncertainties from a MAP fit?**

Run :meth:`~pgmuvi.lightcurve.Lightcurve.mcmc` to obtain posterior samples.
Report the posterior credible interval on ``1 / mixture_means`` as the period
uncertainty.

Multiband Data
--------------

**Q: Do I have to use the same set of times for each band?**

No.  The observations for each band can be at completely different times.  Simply
stack all observations into a single array with a wavelength/band label in the
second column of ``xdata`` (see :doc:`howto/multiband`).

**Q: Which band is used for Lomb–Scargle initialisation in 2D?**

By default, the multiband Lomb–Scargle periodogram is used.  If one band has
substantially more observations than others, pass ``use_best_band_init=True`` to
:meth:`~pgmuvi.lightcurve.Lightcurve.fit_LS` to first run 1D Lomb–Scargle on the
most-sampled band for frequency seeding.

Contributing and Support
------------------------

**Q: I found a bug.  How do I report it?**

Please open an issue on the `GitHub repository <https://github.com/ICSM/pgmuvi/issues>`_.
Include a minimal reproducible example and the full error traceback.

**Q: I want to add a new feature.  Where do I start?**

See the `contributing guidelines <https://github.com/ICSM/pgmuvi/blob/main/CONTRIBUTING.md>`_
on GitHub.  New contributors are very welcome!
