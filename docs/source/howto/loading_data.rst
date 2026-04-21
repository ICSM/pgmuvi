Loading and Preparing Data
==========================

This guide shows how to load observational data into ``pgmuvi`` and prepare it for
fitting.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

``pgmuvi`` expects data as three parallel arrays:

* **times** — observation epochs (any consistent time unit, e.g., days, MJD).
* **fluxes** — flux or magnitude measurements.
* **errors** — 1-σ uncertainties on the measurements.

All three arrays must have the same length.  For multiband data, each array has one
row per observation across all bands (see :doc:`multiband`).

Creating a Lightcurve
----------------------

Pass the arrays directly to the constructor::

    import pgmuvi
    import numpy as np

    times  = np.array([...])   # shape (N,)
    fluxes = np.array([...])   # shape (N,)
    errors = np.array([...])   # shape (N,)

    lc = pgmuvi.lightcurve.Lightcurve(times, fluxes, errors)

The data are stored internally as PyTorch tensors.  You can retrieve them as NumPy
arrays via ``lc.xdata.cpu().numpy()``, etc.

Loading from a File
--------------------

**From a CSV file**

:meth:`~pgmuvi.lightcurve.Lightcurve.from_csv` reads a CSV file directly.
Column names are matched case-insensitively using common aliases, so in most
cases no extra arguments are required::

    import pgmuvi

    lc = pgmuvi.lightcurve.Lightcurve.from_csv("my_lightcurve.csv")

For multiband CSV files that include a numeric wavelength column, pass the
column name explicitly or let the method auto-detect it::

    # Explicit wavelength column
    lc = pgmuvi.lightcurve.Lightcurve.from_csv(
        "multiband.csv", wavelcol="wavelength_um"
    )

    # Or specify time and wavelength together
    lc = pgmuvi.lightcurve.Lightcurve.from_csv(
        "multiband.csv", xcol=["mjd", "wavelength_um"]
    )

If the CSV contains a **string band-identifier column** (e.g. ``band`` or
``filter`` with values like ``"V"``, ``"R"``), that column may be automatically
stored in :attr:`~pgmuvi.lightcurve.Lightcurve.band` for labelling purposes.
For **2-D (multiband) lightcurves** this happens automatically.  For **1-D
lightcurves**, auto-population only occurs when the band-ID column contains
exactly one distinct non-empty label (matching the 1-D constructor contract); if
multiple distinct labels are present, ``band`` is left unset and a warning is
emitted.
Note that these string labels are for human readability only — the GP model
requires a numeric wavelength in column 1 of ``xdata`` (see
:doc:`multiband`).

**From an Astropy-compatible format**

:meth:`~pgmuvi.lightcurve.Lightcurve.from_table` builds a light curve from an
:class:`astropy.table.Table` instance or any file format that Astropy can read
(FITS, VOTable, many ASCII dialects)::

    import pgmuvi

    lc = pgmuvi.lightcurve.Lightcurve.from_table("my_lightcurve.vot")

Example from an in-memory table::

    from astropy.table import Table
    import pgmuvi

    t = Table.read("my_lightcurve.fits")
    lc = pgmuvi.lightcurve.Lightcurve.from_table(t)

**From raw arrays**

For any other format, read the data manually and pass arrays directly::

    import numpy as np
    import pgmuvi

    data = np.loadtxt("my_lightcurve.csv", delimiter=",")
    lc = pgmuvi.lightcurve.Lightcurve(data[:, 0], data[:, 1], data[:, 2])

Adding More Observations
--------------------------

**Merging a new band into an existing multiband lightcurve**

:meth:`~pgmuvi.lightcurve.Lightcurve.merge` appends a new band to an
existing 2-D light curve.  The calling object must already be 2-D; 1-D
inputs are promoted automatically when a wavelength is supplied::

    # lc2d is an existing 2-D lightcurve; lc_new is a new single-band lc
    merged = lc2d.merge(lc_new, wavelength=0.80)   # 0.80 μm

You can also merge directly from a CSV path::

    merged = lc2d.merge("new_band.csv", wavelength=0.80)

**Combining multiple lightcurves into one multiband object**

:meth:`~pgmuvi.lightcurve.Lightcurve.concat` is a class method that builds a
2-D light curve from a list of single-band (or already-multiband) objects.
Every input must carry band information (either set at construction time via
``band=`` or via :meth:`~pgmuvi.lightcurve.Lightcurve.from_csv`)::

    combined = pgmuvi.lightcurve.Lightcurve.concat([lc_V, lc_R, lc_I])

Both methods accept ``on_conflict="skip"`` to silently drop duplicate bands
rather than raising an error.

**Concatenating arrays before construction**

For simple cases where band information is not needed, concatenate the NumPy
arrays before constructing the :class:`~pgmuvi.lightcurve.Lightcurve`::

    import numpy as np
    import pgmuvi

    all_times  = np.concatenate([times,  new_times])
    all_fluxes = np.concatenate([fluxes, new_fluxes])
    all_errors = np.concatenate([errors, new_errors])

    lc = pgmuvi.lightcurve.Lightcurve(all_times, all_fluxes, all_errors)

.. note::

   For 2D / multiband data, ``xdata`` must have shape ``(N, 2)`` with column 0
   being time and column 1 being a numeric wavelength.  See :doc:`multiband`.

Data Transformations
---------------------

GP optimisation can be sensitive to the scale of the input data.  ``pgmuvi``
provides built-in transformations to rescale the time and flux axes:

.. list-table::
   :header-rows: 1
   :widths: 20 50

   * - Transform
     - Description
   * - ``'minmax'``
     - Rescale to [0, 1] using min and max.
   * - ``'zscore'``
     - Standardise to zero mean, unit variance.
   * - ``'robust_score'``
     - Standardise using median and IQR (robust to outliers).

Apply a transformation at construction time via the ``xtransform`` and ``ytransform``
keyword arguments::

    lc = pgmuvi.lightcurve.Lightcurve(
        times, fluxes, errors,
        xtransform="minmax",
        ytransform="zscore",
    )

The GP is trained in the transformed space, but all results and plots are
automatically inverse-transformed back to the original units.

.. _working-with-magnitudes:

Working with Magnitudes
------------------------

Native magnitude support is planned for a future release but is not currently
available.  If your data are in magnitudes, convert them to (relative) flux
before constructing the :class:`~pgmuvi.lightcurve.Lightcurve`.  A common
choice is:

.. math::

   f \propto 10^{-0.4\,m}

In code::

    import numpy as np
    import pgmuvi

    # mags and mag_errors are your input magnitudes and uncertainties
    fluxes = 10 ** (-0.4 * mags)
    errors = fluxes * np.log(10) * 0.4 * mag_errors

    lc = pgmuvi.lightcurve.Lightcurve(times, fluxes, errors)

Only relative variations matter for most ``pgmuvi`` analyses, so the overall
flux normalisation is arbitrary.

Checking Data Quality
----------------------

Before fitting, assess whether the observations are sufficient to detect the
variability timescales you are interested in::

    lc.assess_sampling_quality()

See :doc:`preprocessing` for more detail on sampling quality metrics and filtering.

Exporting Data
---------------

The loaded data can be exported to an Astropy table or a VO Table file::

    table = lc.to_table()
    lc.write_votable("lightcurve_output.xml")
