Data Preprocessing
==================

This guide covers the preprocessing tools available in ``pgmuvi`` for assessing
data quality, filtering observations, and subsampling large datasets.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

Raw observational datasets often contain:

* observations from poorly sampled or non-variable sources,
* extremely dense sampling that is computationally expensive for GP fitting,
* bands with insufficient coverage to constrain variability.

``pgmuvi`` provides tools to address each of these issues before fitting.

Checking Variability
---------------------

The first question to ask is whether your source is actually variable.  ``pgmuvi``
provides three complementary test statistics:

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Statistic
     - Description
   * - Weighted χ²
     - Tests against a constant-flux (null) model.
   * - F\ :sub:`var`
     - Fractional excess variance; measures variability amplitude relative to noise.
   * - Stetson K
     - A robust, outlier-resistant index of correlated variability.

For single-band data::

    result = lc.check_variability()
    print(result)

For multiband data::

    results = lc.check_variability_per_band()
    for band, r in results.items():
        print(band, r)

To retain only bands that pass a variability criterion::

    lc = lc.filter_variable_bands(fvar_min=0.1)

Sampling Quality Metrics
-------------------------

Even if a source is variable, the observations may not resolve the variability
timescales of interest.  ``pgmuvi`` computes several metrics to assess this:

.. list-table::
   :header-rows: 1
   :widths: 30 55

   * - Key
     - Description
   * - ``n_points``
     - Number of finite observations.
   * - ``baseline``
     - Total time span (``max(t) − min(t)``).
   * - ``max_gap``
     - Largest gap between consecutive observations.
   * - ``max_gap_fraction``
     - ``max_gap / baseline``; large values reduce sensitivity to long-period variability.
   * - ``median_cadence``
     - Median time between consecutive observations.
   * - ``mean_cadence``
     - Mean time between consecutive observations.
   * - ``cadence_std``
     - Standard deviation of the cadence distribution.
   * - ``nyquist_period``
     - ``2 × effective_cadence``; shortest reliably detectable period.
   * - ``nyquist_frequency``
     - ``1 / (2 × effective_cadence)``; corresponding Nyquist frequency.
   * - ``longest_detectable_period``
     - ``baseline / 2`` (heuristic upper limit on detectable periods).
   * - ``duty_cycle``
     - Fraction of the baseline with observations (``n × cadence / baseline``).
   * - ``sampling_uniformity``
     - ``1 − std(cadence) / mean(cadence)``; 1 = perfectly uniform, 0 = highly irregular.

Retrieve numeric metrics::

    metrics = lc.compute_sampling_metrics()
    print(metrics)

Or get a plain-language assessment with recommendations::

    lc.assess_sampling_quality()

For multiband data::

    lc.assess_sampling_quality_per_band()

To retain only well-sampled bands::

    lc = lc.filter_well_sampled_bands(min_points=20)

Subsampling Dense Datasets
---------------------------

GP inference scales as :math:`\mathcal{O}(N^3)` in the number of observations.
For very densely sampled light curves, subsampling can make fitting computationally
feasible without significantly affecting the results, provided the subsampled dataset
still satisfies the Nyquist criterion for the periods of interest.

``pgmuvi`` provides a gap-preserving random subsampling function in the preprocessing
subpackage::

    from pgmuvi.preprocess import subsample_lightcurve

    times  = lc.xdata.cpu().numpy()
    fluxes = lc.ydata.cpu().numpy()
    errors = lc.yerr.cpu().numpy()

    idx = subsample_lightcurve(times, max_samples=500)

    lc_sub = pgmuvi.lightcurve.Lightcurve(
        times[idx], fluxes[idx], errors[idx]
    )

``subsample_lightcurve`` takes only the 1-D time array and returns an index array.
It preserves the overall time coverage (gaps are retained in proportion) so that
long-timescale variability remains detectable after subsampling.

.. seealso::

   :mod:`pgmuvi.preprocess` — full API reference for the preprocessing subpackage.

Quality Filtering
------------------

.. note::

   This section will describe how to apply quality flags or sigma-clipping to remove
   outliers before fitting.  The relevant utilities are located in
   :mod:`pgmuvi.preprocess.quality`.

Preprocessing Tutorial
-----------------------

Use :doc:`../notebooks/tutorial_preprocessing` for a deterministic, executable
walkthrough of mixed-band CSV validation, sampling and variability diagnostics,
reproducible gap-aware subsampling, and per-band filtering.  The notebook does
not start GP training and keeps the diagnostic decisions separate from model
selection.

The notebook complements this reference page and the
:mod:`pgmuvi.preprocess` API documentation; it does not replace the need to
record the thresholds used for scientific filtering.
