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

``pgmuvi`` does not currently provide a dedicated file reader, so you can use standard
Python libraries such as ``numpy``, ``astropy``, or ``pandas`` to read your data and
then pass the arrays to the constructor.

Example using ``astropy``::

    from astropy.table import Table
    import pgmuvi

    t = Table.read("my_lightcurve.fits")
    lc = pgmuvi.lightcurve.Lightcurve(
        t["time"].data,
        t["flux"].data,
        t["flux_err"].data,
    )

Example using ``numpy``::

    import numpy as np
    import pgmuvi

    data = np.loadtxt("my_lightcurve.csv", delimiter=",")
    lc = pgmuvi.lightcurve.Lightcurve(data[:, 0], data[:, 1], data[:, 2])

Adding More Observations
--------------------------

To combine observations from multiple files or epochs, concatenate the arrays
before constructing the ``Lightcurve``::

    import numpy as np
    import pgmuvi

    all_times  = np.concatenate([times,  new_times])
    all_fluxes = np.concatenate([fluxes, new_fluxes])
    all_errors = np.concatenate([errors, new_errors])

    lc = pgmuvi.lightcurve.Lightcurve(all_times, all_fluxes, all_errors)

.. note::

   For 2D / multiband data, ``xdata`` must have shape ``(N, 2)`` with column 0
   being time and column 1 being a wavelength or band index.  See :doc:`multiband`.

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

Or call :meth:`~pgmuvi.lightcurve.Lightcurve.transform_x` and
:meth:`~pgmuvi.lightcurve.Lightcurve.transform_y` after construction.  All results
and plots are automatically inverse-transformed back to the original units.

Working with Magnitudes
------------------------

By default, ``pgmuvi`` treats ``ydata`` as flux (higher values = brighter).
If your data are in magnitudes (higher values = fainter), set ``magnitudes=True``::

    lc = pgmuvi.lightcurve.Lightcurve(times, mags, mag_errors, magnitudes=True)

The sign convention is handled internally.

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
