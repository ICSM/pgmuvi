Loading and validating light curves
===================================

This guide explains the input contract for :class:`pgmuvi.lightcurve.Lightcurve`,
how CSV columns are interpreted, which rows are removed automatically, and which
quality decisions remain the user's responsibility.

.. contents:: On this page
   :local:
   :depth: 2

The validation sequence
-----------------------

A reliable input workflow has five distinct stages:

1. identify the time, measurement, uncertainty, physical-wavelength, and
   observational-channel label columns;
2. construct a 1-D or 2-D :class:`~pgmuvi.lightcurve.Lightcurve`;
3. inspect rows removed because of non-finite values;
4. decide whether non-positive fluxes or uncertainties are scientifically
   acceptable for the intended workflow; and
5. inspect sampling, variability, and any constructor-time subsampling before
   fitting.

Do not treat these stages as interchangeable.  In particular, successful CSV
loading does not prove that the uncertainties are positive, the sampling is
adequate, or the source is detectably variable.

Array shapes
------------

For a single-band light curve, provide three parallel one-dimensional arrays::

    import numpy as np
    from pgmuvi.lightcurve import Lightcurve

    times = np.asarray([...], dtype=float)
    fluxes = np.asarray([...], dtype=float)
    errors = np.asarray([...], dtype=float)

    lc = Lightcurve(times, fluxes, errors)

The arrays must describe the same observations and therefore have the same
length.  The uncertainty array is optional at construction time, but measured
uncertainties are strongly recommended for scientific fitting and variability
diagnostics.

For a multiband light curve, ``xdata`` must have shape ``(N, 2)``:

* column 0 contains time;
* column 1 contains a numeric physical wavelength coordinate; and
* ``ydata`` and ``yerr`` remain one-dimensional arrays of length ``N``.

For example::

    xdata = np.column_stack([times, wavelengths_um])
    lc = Lightcurve(xdata, fluxes, errors, band=band_labels)

An observational-channel label identifies the instrument, detector, filter, or
data stream used for an observation.  Observational-channel labels are metadata
for reporting, grouping, and diagnostics; they do not replace the numeric physical
wavelength coordinate used by the GP.  In particular, multiple observational
channels may share one physical wavelength.  Lightcurve.band is retained as the
legacy attribute, while
:attr:`~pgmuvi.lightcurve.Lightcurve.observational_channel_labels` exposes the
preferred terminology.

CSV input contract
------------------

:meth:`~pgmuvi.lightcurve.Lightcurve.from_csv` requires a header row.  Column
matching is case-insensitive.  When column names are not passed explicitly, the
loader searches the following aliases in order:

.. list-table:: Auto-detected CSV columns
   :header-rows: 1
   :widths: 20 70

   * - Role
     - Recognised names
   * - Time
     - ``x``, ``time``, ``t``, ``jd``, ``mjd``, ``date``, ``hjd``, ``bjd``,
       ``epoch``
   * - Measurement
     - ``y``, ``magnitude``, ``mag``, ``flux``, ``value``, ``data``
   * - Uncertainty
     - ``yerr``, ``uncertainty``, ``error``, ``err``, ``unc``, ``sigma``,
       ``e_magnitude``, ``e_mag``, ``e_flux``, ``flux_error``, ``mag_error``,
       ``magnitude_error``, ``value_error``, ``data_error``, ``y_error``
   * - Numeric wavelength
     - ``wavelength``, ``wave``, ``wl``, ``lambda``, ``freq``, ``frequency``,
       ``channel``
   * - Observational-channel label
     - ``band``, ``filter``, ``filtername``, ``filter_name``

A minimal single-band file is therefore::

    time,flux,flux_error
    59000.0,1.02,0.03
    59005.0,0.98,0.03

Load it with::

    lc = Lightcurve.from_csv("single_band.csv")

Use explicit names when the file uses project-specific headers::

    lc = Lightcurve.from_csv(
        "source.csv",
        xcol="observation_epoch",
        ycol="relative_flux",
        yerrcol="relative_flux_error",
    )

A multiwavelength file should include numeric physical wavelengths and may also
include human-readable observational-channel labels::

    mjd,wavelength_um,band,flux,flux_error
    59000.0,0.55,V,1.02,0.03
    59001.0,0.80,I,0.91,0.04

Load it with::

    lc = Lightcurve.from_csv(
        "multiband.csv",
        xcol="mjd",
        wavelcol="wavelength_um",
        ycol="flux",
        yerrcol="flux_error",
    )

You can instead pass ``xcol=["mjd", "wavelength_um"]``.  A numeric wavelength
column with more than one distinct value produces 2-D ``xdata``.  A single
numeric wavelength value produces a 1-D light curve.

.. warning::

   Supplying string labels as the wavelength coordinate maps the labels to
   arbitrary numeric indices.  That may be useful for categorical bookkeeping,
   but it is not a physical wavelength scale.  Use actual numeric wavelengths
   for ``2DWavelengthDependent``, ``2DDustMean``, ``2DPowerLawMean``, and other
   workflows whose interpretation depends on wavelength.

Duplicate physical wavelengths during fitting
---------------------------------------------

A numeric physical wavelength may be represented by more than one
observational channel, for example two detector/data-stream identifiers from
the same survey.  Ordinary GP inputs cannot silently treat those channels as
interchangeable.  Before any model or likelihood is constructed,
:meth:`~pgmuvi.lightcurve.Lightcurve.fit` therefore resolves each duplicated
physical-wavelength group.

The default policy retains the first observational channel encountered in the
original aligned input row order captured before constructor sampling,
subsampling, or transformation.  It emits a :class:`UserWarning` that names
the selected channel, every ignored channel, and the corresponding row counts::

    lc.fit(model="2DWavelengthDependent")

Select a different channel explicitly when exactly one physical wavelength is
duplicated::

    lc.fit(
        model="2DWavelengthDependent",
        duplicate_wavelength_policy="select",
        duplicate_wavelength_selection="KELT/OSN_Johnson.Cousins_R3_1",
    )

For several duplicated physical wavelengths, provide one channel per physical
wavelength::

    lc.fit(
        model="2DWavelengthDependent",
        duplicate_wavelength_policy="select",
        duplicate_wavelength_selection={
            0.6561154962791801: "KELT/OSN_Johnson.Cousins_R3_1",
            1.25: "SURVEY/FILTER_CHANNEL",
        },
    )

The ``"all"`` policy is reserved for a scientifically validated
instrument-channel calibration strategy.  It currently raises
:class:`NotImplementedError` before fitting rather than pooling uncalibrated
channels.  Resolution is recorded in ``lc.duplicate_wavelength_channel_resolution``
and in fit-history provenance.  Direct :meth:`~pgmuvi.lightcurve.Lightcurve.fit`
calls preserve the package's established in-place fit contract: the fitted
object stores the observations actually used by its model.

To preserve one loaded source while fitting alternative channels, create
independent resolved copies first::

    source = Lightcurve.from_csv(
        "examples/data/10131+3049.csv",
        max_samples=None,
        max_samples_per_band=None,
    )
    r3_0 = source.copy_with_duplicate_wavelength_channels()
    r3_1 = source.copy_with_duplicate_wavelength_channels(
        policy="select",
        selection="KELT/OSN_Johnson.Cousins_R3_1",
    )

    r3_0.fit(model="2DWavelengthDependent")
    r3_1.fit(model="2DWavelengthDependent")

``source`` remains unchanged; each returned copy owns independent data and
transform objects and records its own channel-resolution provenance.

When constructor-time ``max_samples_per_band`` is enabled, per-row
observational-channel labels are used as the grouping key.  Thus two channels
at one physical wavelength are subsampled independently before the fit policy
selects one.  Numeric physical wavelength is used only when channel labels are
unavailable.

Observational-channel labels and mixed-channel input
------------------------------------------------------

A recognised string label column is handled independently of the numeric
physical-wavelength column.  For 2-D data, observational-channel labels are
stored row by row in ``lc.band`` and exposed through
``lc.observational_channel_labels``.  For 1-D data, a single distinct non-empty
label is stored as the single-channel label.  The legacy ``band`` name remains
part of the public input and serialization contract.

If a nominally 1-D file contains several distinct observational-channel labels
but no usable numeric physical-wavelength coordinate,
:meth:`~pgmuvi.lightcurve.Lightcurve.from_csv` leaves ``lc.band`` unset and warns
that the mixed-channel input was not promoted to 2-D.  Fix the file by supplying
a numeric physical-wavelength column rather than ignoring the warning.

Finite-value filtering
----------------------

For standard light curves, rows containing non-finite values in time,
measurement, or uncertainty columns are removed before fitting:

* CSV loading removes rows containing missing numeric values and empty required
  strings;
* the constructor subsequently removes remaining ``NaN`` or infinite values;
* a :class:`UserWarning` reports the number of removed rows;
* a :class:`ValueError` is raised if no valid rows remain; and
* an additional warning is emitted when fewer than ten rows remain after
  filtering.

Always read these warnings.  A fit that proceeds after substantial row removal
may no longer represent the intended time baseline or band coverage.

Positive fluxes and uncertainties
---------------------------------

The base :class:`~pgmuvi.lightcurve.Lightcurve` constructor does **not** impose a
universal positive-flux rule.  Negative or zero measurements can be legitimate
for background-subtracted linear-flux data.  Do not remove them automatically
without considering the measurement definition and the downstream model.

Non-positive uncertainty values are different: zero or negative standard
uncertainties are not scientifically meaningful.  Correct or remove those rows
before using uncertainty-aware fits or variability diagnostics.  Some
preprocessing diagnostics explicitly reject non-positive uncertainties even
though the constructor itself does not.

The wavelength-advisory batch workflow applies stricter defaults: it drops rows
with non-positive fluxes or flux uncertainties unless its command-line override
flags are used.  That policy belongs to the advisory workflow and should not be
mistaken for a universal constructor rule.  See
:doc:`wavelength_advisory_batch`.

The runnable validation example supplied with PGMUVI can report or explicitly
drop these rows::

    python examples/validate_lightcurve_input.py source.csv \
        --drop-nonpositive-errors

Add ``--drop-nonpositive-flux`` only when positive flux is required by the
scientific workflow.

.. _working-with-magnitudes:

Working with magnitudes
-----------------------

.. note::

   **TBD[native-magnitude-input]:** Native magnitude-domain input is not
   currently available. Convert magnitudes and their uncertainties to
   relative flux before constructing the light curve.

Native magnitude support is not currently implemented.  Convert magnitudes and
their uncertainties to linear relative flux before constructing the light curve.
For an arbitrary reference magnitude :math:`m_0`, one convenient convention is:

.. math::

   f = 10^{-0.4\,(m-m_0)}.

Propagate a small magnitude uncertainty :math:`\sigma_m` to flux space with:

.. math::

   \sigma_f = \frac{\ln 10}{2.5}\,f\,\sigma_m.

For example::

    import numpy as np

    reference_magnitude = np.nanmedian(magnitude)
    flux = 10.0 ** (-0.4 * (magnitude - reference_magnitude))
    flux_error = (np.log(10.0) / 2.5) * flux * magnitude_error

The reference magnitude changes only the overall flux normalisation.  Preserve
the sign convention: smaller magnitudes must map to larger fluxes.  Do not pass
magnitude uncertainties unchanged as flux uncertainties.

Time units and automatic centering
----------------------------------

Time values are assumed to be in days unless ``time_units=`` is supplied.  Any
unit accepted by :mod:`astropy.units` can be converted to days during
construction::

    lc = Lightcurve(times_in_hours, fluxes, errors, time_units="hr")

When no explicit ``xtransform`` is supplied, PGMUVI centers the time coordinate
by default before GP training.  The raw time values remain available through
``lc.xdata`` and reported results remain in the original physical time units.
Use ``center_time=False`` only when you have a specific reason to disable this
numerical-stability default.

Sampling checks and band removal
--------------------------------

Construction does not check sampling unless ``check_sampling=True`` is passed::

    lc = Lightcurve.from_csv(
        "source.csv",
        check_sampling=True,
        sampling_kwargs={
            "min_points": 20,
            "max_gap_fraction": 0.3,
        },
    )

The outcome differs by dimensionality:

* for a 1-D light curve, poor sampling raises :class:`ValueError`;
* for a 2-D light curve, each wavelength is assessed independently;
* failing 2-D bands are removed with warnings; and
* a :class:`ValueError` is raised only when no wavelength band passes.

To inspect without constructor-time rejection or removal, construct with
``check_sampling=False`` and then run::

    metrics = lc.compute_sampling_metrics()
    passes, diagnostics = lc.assess_sampling_quality(verbose=False)

For multiband data use::

    metrics_by_band = lc.compute_sampling_metrics_per_band()
    diagnostics_by_band = lc.assess_sampling_quality_per_band(verbose=False)

See :doc:`preprocessing` for the metric definitions and variability checks.

Default subsampling behaviour
-----------------------------

The constructor's default ``max_samples=1000`` has different meanings for 1-D
and 2-D light curves:

* **1-D:** more than 1000 observations triggers permanent gap-preserving random
  subsampling;
* **2-D:** ``max_samples`` only emits a total-size advisory warning;
* **2-D:** actual per-band subsampling is controlled by
  ``max_samples_per_band`` and is disabled by default.

Make the decision explicit in reproducible work::

    lc = Lightcurve.from_csv(
        "source.csv",
        max_samples=1000,
        max_samples_per_band=100,
        subsample_seed=12345,
    )

Set ``max_samples=None`` to disable 1-D automatic subsampling or the 2-D total
size warning.  Set ``max_samples_per_band=None`` to disable 2-D per-band
subsampling.

Variability checks
------------------

``check_variability=True`` is a constructor gate for 1-D light curves only.  It
raises :class:`ValueError` when the source does not pass the configured
variability tests::

    lc = Lightcurve.from_csv(
        "single_band.csv",
        check_variability=True,
        variability_kwargs={"fvar_min": 0.1},
    )

Pooling multiband measurements into a single variability test can be
misleading, so constructor-time variability checking is rejected for 2-D input.
Use ``check_variability_per_band()`` or ``filter_variable_bands()`` instead.

Common warnings and what they mean
----------------------------------

.. list-table:: Input and validation warnings
   :header-rows: 1
   :widths: 38 62

   * - Warning fragment
     - Required interpretation
   * - ``Dropped ... row(s)``
     - Non-finite or missing input was removed.  Verify the retained baseline,
       wavelength coverage, and row count.
   * - ``Fewer than 10 elements remain``
     - Construction succeeded, but the retained dataset is too small for a
       routine fit to be trusted.
   * - ``multiple distinct ... labels for 1-D input``
     - Mixed bands were found without a numeric wavelength coordinate.  Repair
       the input rather than treating it as a valid single-band series.
   * - ``Skipping band ... due to poor temporal sampling``
     - ``check_sampling=True`` removed that wavelength from a 2-D light curve.
   * - ``Retaining ... wavelength bands``
     - Only a subset of the original bands survived sampling validation.
   * - ``exceeds max_samples``
     - A 1-D series may have been subsampled, or a 2-D series exceeded the
       advisory compute budget.  Read the full warning to distinguish them.
   * - ``median_cadence is zero``
     - Duplicate timestamps made the median cadence unusable; selected period
       limits were computed from positive gaps instead.

Runnable validation example
---------------------------

The example script loads the file without early subsampling, optionally removes
non-positive rows, reconstructs the validated light curve with the requested
sampling and subsampling settings, and prints a JSON summary::

    python examples/validate_lightcurve_input.py source.csv \
        --check-sampling \
        --max-samples 1000 \
        --max-samples-per-band 100 \
        --subsample-seed 12345 \
        --drop-nonpositive-errors

Run ``python examples/validate_lightcurve_input.py --help`` for explicit column
options and all validation switches.

Other input formats
-------------------

:meth:`~pgmuvi.lightcurve.Lightcurve.from_table` accepts an in-memory
:class:`astropy.table.Table` or an Astropy-readable file.  Raw arrays remain the
most explicit route for formats whose column conventions do not match the CSV
loader.

Loaded data can be exported with::

    table = lc.to_table()
    lc.write_votable("lightcurve_output.xml")

Notebook tutorial
-----------------

Use :doc:`../notebooks/tutorial_preprocessing` for an executable companion to
this guide.  It demonstrates non-finite-row handling in a mixed-band CSV,
separate sampling and variability diagnostics, reproducible subsampling, and
per-band filtering under the current dtype and time-centering defaults.

The command-line validator above remains the better interface for validating
many external files or recording machine-readable input summaries.
