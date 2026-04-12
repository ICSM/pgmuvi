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

``pgmuvi`` provides a convenience constructor
:meth:`~pgmuvi.lightcurve.Lightcurve.from_table` that can build a light curve
from an :class:`astropy.table.Table` instance or directly from a filename.
Under the hood it calls :func:`astropy.table.Table.read`, so any format
supported by Astropy (FITS, VOTable, many ASCII dialects) is accepted.

Example loading from a filename::

    import pgmuvi

    lc = pgmuvi.lightcurve.Lightcurve.from_table("my_lightcurve.vot")

Example from an in-memory table::

    from astropy.table import Table
    import pgmuvi

    t = Table.read("my_lightcurve.fits")
    lc = pgmuvi.lightcurve.Lightcurve.from_table(t)

For formats that require column names different from the defaults, or to read
plain-text files, you can also read the data manually and pass arrays to the
constructor::

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
