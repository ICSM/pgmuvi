Consensus multiband fitting
===========================

.. note::

   **Documentation status:** current through PR71/PR73.

   **Pipeline status:** consensus fitting is the recommended GP fitting path
   for coherent multiwavelength variability when the bands share an underlying
   period or variability time-scale.  The wavelength-advisory workflow builds
   on this machinery, but consensus fitting can also be used directly.

Purpose
-------

Many evolved-star and LPV light curves contain measurements in several bands
with different cadences, noise levels, and temporal coverage.  A direct 2-D GP
fit can be numerically fragile if the temporal kernel is initialized without
using the per-band period information.  The consensus strategy separates the
problem into two stages:

1. inspect each usable band for plausible temporal periods or frequencies, and
2. fit the requested GP model using a period or frequency supported by multiple
   bands.

The usual direct call is:

.. code-block:: python

   result = lc.fit(
       model="2D",
       fit_strategy="consensus",
       training_iter=500,
       miniter=100,
       learn_additional_noise=True,
       verbose=True,
   )

This call is intentionally explicit.  The model is still a GP model, but the
``fit_strategy="consensus"`` setting asks ``pgmuvi`` to use multiband period
agreement to initialize or constrain the temporal part of the fit.

When to use consensus fitting
-----------------------------

Use consensus fitting when the scientific assumption is that the bands share a
coherent temporal process.  That is usually the right starting point for LPV
sources whose optical/infrared bands trace the same underlying stellar
variability with wavelength-dependent amplitudes or means.

Consensus fitting is especially useful when:

* the bands have uneven or different sampling,
* the raw Lomb--Scargle peaks differ between some bands,
* one or more bands are too sparse to trust individually,
* the full 2-D spectral-mixture model needs a data-derived starting period, or
* an LPV separable model needs a consensus period handed to a quasi-periodic
  time kernel.

Do not use the result blindly when the accepted bands do not support a coherent
period.  A consensus failure is often a data-quality or source-physics signal,
not necessarily a package crash.

Minimal workflow
----------------

A typical CSV-based workflow is:

.. code-block:: python

   from pgmuvi.lightcurve import Lightcurve

   lc = Lightcurve.from_csv(
       "source.csv",
       check_sampling=True,
       max_samples=1000,
       max_samples_per_band=100,
       verbose=True,
   )

   result = lc.fit(
       model="2D",
       fit_strategy="consensus",
       training_iter=500,
       miniter=100,
       learn_additional_noise=True,
       verbose=True,
   )

   summary = lc.get_period_summary()
   print(summary.dominant_period)

   primary = summary.get_primary_peak()
   if primary is not None:
       print(primary.period, primary.area_fraction)

   # For serialisable output, use the structured dictionary.
   summary_dict = summary.as_dict()

``check_sampling`` is configured when the ``Lightcurve`` is constructed or
loaded, not when ``fit`` is called.  Sampling checks may filter bands before the
fit stage.

Consensus splitting also requires one band label per observation row.
``Lightcurve.from_csv`` preserves these labels when the input file has a band
column.  If constructing a 2-D ``Lightcurve`` manually from arrays, pass the
``band=`` argument explicitly; otherwise ``fit_strategy="consensus"`` cannot
split the data into per-band light curves.

Important fit options
---------------------

.. list-table:: Consensus fit options
   :header-rows: 1

   * - Option
     - Meaning
   * - ``model``
     - GP model family to fit, for example ``"2D"``,
       ``"2DWavelengthDependent"``, ``"2DDustMean"``, or
       ``"2DPowerLawMean"``.
   * - ``fit_strategy="consensus"``
     - Enables the multiband consensus period/frequency stage before the final
       GP fit.
   * - ``training_iter`` / ``miniter``
     - Maximum and minimum optimization iterations.  Short values are useful
       for smoke tests; production runs should use longer training.
   * - ``learn_additional_noise``
     - Adds a learned noise term on top of supplied uncertainties.  This is
       useful when the quoted errors understate the scatter.
   * - ``time_kernel_type``
     - For separable LPV-relevant models, the advisory workflow typically uses
       ``"quasi_periodic"`` so the consensus period becomes a time-kernel
       period length.
   * - ``verbose``
     - Prints accepted/rejected bands, dominant periods, and failure reasons.

Model families
--------------

The consensus strategy can be used with the full 2-D baseline and with the
LPV-relevant separable model families.

.. list-table:: Common consensus model choices
   :header-rows: 1

   * - Model
     - Typical role
   * - ``2D``
     - Full non-separable 2-D spectral-mixture baseline.  This is the main
       robust baseline for shared multiwavelength variability.
   * - ``2DWavelengthDependent``
     - Flexible separable wavelength-dependent covariance model.  Use with a
       quasi-periodic time kernel when comparing wavelength structure.
   * - ``2DDustMean``
     - Separable covariance model with a dust-inspired wavelength mean.  Useful
       as a physically motivated LPV comparison model.
   * - ``2DPowerLawMean``
     - Separable covariance model with a simpler power-law wavelength mean.

Example separable consensus call:

.. code-block:: python

   result = lc.fit(
       model="2DDustMean",
       fit_strategy="consensus",
       time_kernel_type="quasi_periodic",
       training_iter=500,
       miniter=100,
       learn_additional_noise=True,
       verbose=True,
   )

The current wavelength advisory workflow automates this comparison over several
model/kernel configs.  Use the direct fit calls here when you already know which
model family you want to inspect.

Accepted and rejected bands
---------------------------

The consensus stage tries to identify bands that support a common period or
frequency.  Verbose output and fit diagnostics can distinguish:

* accepted bands that support the consensus period,
* rejected bands that fail sampling or variability checks,
* bands whose dominant periods are inconsistent with the consensus, and
* sources for which too few bands support a coherent shared period.

A source can fail consensus even when individual bands have plausible periods.
That usually means the multiband evidence for a shared temporal process is weak
or inconsistent.

Failure contract
----------------

Consensus failures should be interpreted carefully.  In particular,
``ConsensusFitError`` means the consensus stage could not construct a reliable
shared-period fit under the requested criteria.  This is commonly a data-quality
or model-support issue:

* too few usable bands after sampling/variability filtering,
* mutually inconsistent dominant periods,
* insufficient overlap in accepted frequency clusters, or
* a requested model/kernel configuration that does not support the needed
  consensus handoff.

For batch work, record the failure and continue to the next source.  Do not
silently convert a consensus failure into a successful science model.

Inspection after fitting
------------------------

After a successful fit, inspect both the fit result and the period diagnostics.
``get_period_summary()`` returns a ``PeriodSummaryResult`` object rather
than a preformatted report string.  Access scalar fields such as
``dominant_period`` and ``dominant_frequency`` directly, use
``get_primary_peak()`` for the rank-1 peak, and call ``as_dict()`` when
you need a serialisable summary for JSON/text reporting.

Useful follow-up hooks include:

.. code-block:: python

   period_summary = lc.get_period_summary()
   print(period_summary.dominant_period)

   primary = period_summary.get_primary_peak()
   if primary is not None:
       print(primary.period, primary.area_fraction)

   period_summary_dict = period_summary.as_dict()

   # If fit-history export is enabled in your workflow:
   lc.export_fit_history_json("fit_history.json")

   # Plotting remains model/data dependent; use small smoke runs first.
   figs = lc.plot(show=False)

The period summary and fit-history tools are documented more fully in the
results/reporting documentation planned for a later PR.

Relationship to the advisory workflow
-------------------------------------

Consensus fitting and the wavelength advisory workflow are related but not the
same thing.

``fit_strategy="consensus"``
   Executes one requested GP fit using consensus period/frequency information.

``run_period_independent_wavelength_advisory_workflow``
   Builds, runs, scores, and reports several model/kernel configs.  Many of
   those configs use consensus fitting internally, but the workflow remains
   advisory-only and does not select or install a final model.

Use consensus fitting directly for a focused model run.  Use the advisory
workflow when comparing several LPV-relevant model/kernel configs.

Troubleshooting checklist
-------------------------

If consensus fitting fails or gives implausible results, check:

1. Did ``Lightcurve.from_csv`` keep enough bands after sampling filters?
2. Do the accepted bands have plausible and mutually consistent periods?
3. Is ``learn_additional_noise=True`` needed because the quoted errors are too
   optimistic?
4. Is the training run only a smoke test?  Increase ``training_iter`` and
   ``miniter`` before making scientific judgments.
5. Is the source actually coherent across wavelength, or should it be treated as
   a failed/advisory case rather than forced into a shared-period model?
6. For separable LPV models, is the intended ``time_kernel_type`` supplied?

See also
--------

* :doc:`wavelength_advisory`
* :doc:`wavelength_advisory_batch`
* :doc:`preprocessing`
* :doc:`interpreting_results`
