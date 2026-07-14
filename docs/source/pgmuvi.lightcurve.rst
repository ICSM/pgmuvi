pgmuvi.lightcurve
==================

.. automodule:: pgmuvi.lightcurve

The :mod:`pgmuvi.lightcurve` module is the main user-facing interface
for loading light curves, running period diagnostics, and fitting GP
models.  Its implementation currently contains a large legacy API
surface, so this reference page uses a manual public synopsis instead
of expanding every class and helper with autodoc.

.. currentmodule:: pgmuvi.lightcurve

Core class
----------

.. py:class:: Lightcurve(xdata, ydata, yerr=None, **kwargs)

   Container for one-dimensional or two-dimensional light-curve data.
   The class stores input coordinates, flux or magnitude values,
   uncertainties, optional per-row band labels, preprocessing metadata,
   diagnostic summaries, and fit results.

   Common constructor options include ``band`` for per-row band labels,
   ``time_units`` for converting input time coordinates to days,
   ``center_time`` and ``time_center_method`` for time-coordinate
   centering, and ``check_sampling`` / ``check_variability`` for
   optional pre-fit diagnostics.

Loading helpers
---------------

.. py:method:: Lightcurve.from_csv(filename, **kwargs)

   Load a light curve from a CSV file using flexible column-name
   detection.

.. py:method:: Lightcurve.from_table(table, **kwargs)

   Load a light curve from an Astropy table or table-like object.

.. py:method:: Lightcurve.from_pandas(dataframe, **kwargs)

   Load a light curve from a pandas ``DataFrame``.

Fitting and diagnostics
-----------------------

.. py:method:: Lightcurve.fit(*args, **kwargs)

   Fit a GP model to the light curve.  The accepted keyword arguments
   depend on the requested model family and fit strategy.

.. py:method:: Lightcurve.periodogram(*args, **kwargs)

   Run period-search diagnostics for the light curve.

.. py:method:: Lightcurve.plot(*args, **kwargs)

   Plot the light curve and, when available, fitted model results.

Structured failures
-------------------

.. py:exception:: ConsensusFitError(message, *, failure_diagnostics=None)

   Exception raised when the consensus-fit pipeline rejects a dataset
   for structured data-quality reasons rather than because of an
   unrelated software or optimiser failure.

.. py:class:: FitFailureSummary(status="failed", reason=None, message="", diagnostics=None)

   JSON-safe summary object for failed fit attempts.

Result containers
-----------------

.. py:class:: ACFResult

   Result container for autocorrelation-function diagnostics.

.. py:class:: ComponentDiagnosticsResult

   Result container for per-component diagnostic summaries.

Implementation helpers
----------------------

``InputHelpers`` is an implementation mixin used by ``Lightcurve`` for
file and table loading helpers.  It is intentionally not expanded in
this public API page.

Notes
-----

The detailed legacy docstrings for ``Lightcurve`` and related support
classes are intentionally not expanded here because autodoc currently
renders them with fragile reStructuredText warnings.  This page keeps
the public API discoverable while leaving full docstring curation for a
later, larger documentation-structure pass.
