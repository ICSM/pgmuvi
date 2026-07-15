First PGMUVI workflow
=====================

This guide is the first practical workflow for a new source. It assumes you have
read :doc:`../overview` and want a concrete sequence of operations without yet
committing to a specialized model family.

The goal is not to obtain a final publication fit in one command. The goal is to
load the data safely, run low-cost diagnostics, fit a conservative baseline, and
then decide whether wavelength-dependent models deserve closer inspection.

1. Prepare the CSV
------------------

A useful multiwavelength CSV should contain at least:

- a time column such as ``time``, ``jd``, or ``mjd``;
- a flux or magnitude column such as ``flux`` or ``mag``;
- an uncertainty column such as ``flux_error``, ``flux_err``, ``err``, or
  ``mag_err``;
- a numeric wavelength column such as ``wavelength`` when the source has more
  than one band; and
- optionally, a string band column such as ``band`` or ``filter``.

Rows with non-finite values should be removed before fitting. For real-source
batch work, the advisory batch workflow can also drop non-positive fluxes and
non-positive flux uncertainties after loading.

2. Load and inspect the light curve
-----------------------------------

Use explicit column names when the CSV is not obvious:

.. code-block:: python

   from pgmuvi.lightcurve import Lightcurve

   lc = Lightcurve.from_csv(
       "source.csv",
       xcol="time",
       ycol="flux",
       yerrcol="flux_error",
       wavelcol="wavelength",
       check_sampling=True,
       max_samples=1000,
       max_samples_per_band=150,
       verbose=True,
   )

Do not ignore warnings at this stage. Sampling warnings, missing-band warnings,
non-finite-row warnings, and uncertainty warnings often explain later fitting
failures.

3. Run descriptive wavelength diagnostics
-----------------------------------------

For multiwavelength data, first ask whether the per-band mean, scatter, and
robust amplitude vary with wavelength. The wavelength advisory workflow exposes
these diagnostics and keeps them separate from model/kernel-config fitting.

The descriptive diagnostics are not a formal model-selection test. They are a
triage step that helps decide which fits are worth trying.

4. Fit a conservative baseline
------------------------------

For a new multi-band long-period-variable source, start with a consensus 2-D
baseline:

.. code-block:: python

   result = lc.fit(
       model="2D",
       fit_strategy="consensus",
       training_iter=500,
       miniter=100,
       learn_additional_noise=True,
       verbose=True,
   )

Use lower ``training_iter`` and ``miniter`` values for smoke tests. Use longer
runs only after the data load, sampling diagnostics, and initialization look
reasonable.

5. Compare wavelength-dependent families only when justified
------------------------------------------------------------

When the diagnostics suggest wavelength dependence, compare LPV-relevant
families such as:

.. code-block:: python

   for model in ["2DDustMean", "2DPowerLawMean", "2DWavelengthDependent"]:
       result = lc.fit(
           model=model,
           fit_strategy="consensus",
           time_kernel_type="quasi_periodic",
           training_iter=500,
           miniter=100,
           learn_additional_noise=True,
           verbose=True,
       )

The right comparison is not just whether a run completes. Inspect the period,
fit residuals, wavelength trends, failure/fallback fields, and constrained
spectral-mixture ARD diagnostics.

6. Use the runnable orientation script
--------------------------------------

The repository includes a small script that creates a toy multi-band CSV and
prints the first recommended commands:

.. code-block:: bash

   PYTHONPATH=. python3 examples/first_pgmuvi_workflow.py --output-dir first_workflow_demo

The script is intentionally lightweight. It creates a reproducible synthetic CSV
and prints the next commands rather than launching an expensive fit by default.
Use it as a template for the order of operations, not as a scientific benchmark.

7. Decide the next workflow
---------------------------

After the baseline fit and advisory diagnostics, choose one of these paths:

- Continue with :doc:`consensus_fitting` when the main issue is robust period
  recovery from uneven multi-band sampling.
- Continue with :doc:`wavelength_advisory` when wavelength-dependent amplitude,
  mean, or covariance behavior is central to the science question.
- Continue with :doc:`wavelength_advisory_batch` when many sources need the same
  advisory outputs, per-source reports, and failure artifacts.
- Continue with :doc:`interpreting_results` when the main task is understanding
  fit summaries, residuals, periods, and diagnostic fields.

TBD
---

This guide still needs a full notebook version once the public notebook refresh
reaches the first-workflow material. The current runnable script is the tested
entry point.
