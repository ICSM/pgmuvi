Multiwavelength (2D) Analysis
==============================

This guide explains how to fit GP models to light curves observed through
multiple observational channels or across a range of physical wavelengths.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

When a source is observed in multiple observational channels, ``pgmuvi`` can
fit a **2D Gaussian process** whose inputs are ``(time, wavelength)``.  The model
captures:

* the temporal variability structure shared across observational channels;
* the wavelength dependence of variability amplitude and coherence; and
* the measurement-noise information supplied with each observation.

This is more informative than fitting each observational channel independently
because the model borrows statistical strength across physical wavelengths,
particularly when some channels have sparse coverage.

Data Format for 2D Models
--------------------------

For 2D / multiband fitting, ``xdata`` must have shape ``(N, 2)`` where:

* column 0 is the **observation time** (same unit across all observations); and
* column 1 is the **numeric physical wavelength coordinate**.

Use an effective wavelength in a documented physical unit for wavelength-dependent
models.  An observational channel is a separate string identifier for an
instrument/filter/data-stream combination.  The distinction matters:
observational-channel labels do not replace the numeric physical wavelength
coordinate, and multiple observational
channels may legitimately share one physical wavelength.

  .. note::

     When loading from a CSV file via :meth:`~pgmuvi.lightcurve.Lightcurve.from_csv`,
     string band labels are mapped to integer codes only if the wavelength input
     column itself is string-typed, for example with ``wavelcol="band"`` or
     ``xcol=["time", "band"]``. In that case, the encoded values are used in
     ``xdata[:, 1]`` and the original labels (e.g. ``"V"``, ``"I"``) are stored in
     :attr:`~pgmuvi.lightcurve.Lightcurve.band`. If the CSV already provides a
     numeric wavelength column, that numeric column is used directly in
     ``xdata[:, 1]`` and any separate string band-ID column is only stored in
     :attr:`~pgmuvi.lightcurve.Lightcurve.band` without remapping. The
     :class:`~pgmuvi.lightcurve.Lightcurve` constructor itself still requires
     numeric ``xdata``; passing string arrays directly will raise an error.

Human-readable observational-channel labels can be stored separately in the
legacy :attr:`~pgmuvi.lightcurve.Lightcurve.band` attribute and read through
:attr:`~pgmuvi.lightcurve.Lightcurve.observational_channel_labels`.  The labels
are used for reporting, grouping, and channel-level diagnostics, but the GP
receives the numeric physical wavelength coordinate.

All observations are stacked into a single array::

    import numpy as np
    import pgmuvi

    # Two observational channels at two physical wavelengths
    times_all  = np.concatenate([times_b0,  times_b1])
    fluxes_all = np.concatenate([fluxes_b0, fluxes_b1])
    errors_all = np.concatenate([errors_b0, errors_b1])
    wavelengths = np.concatenate([
        np.full_like(times_b0, fill_value=0.55),   # band 0: 0.55 μm
        np.full_like(times_b1, fill_value=2.20),   # band 1: 2.20 μm
    ])

    xdata = np.column_stack([times_all, wavelengths])
    lc = pgmuvi.lightcurve.Lightcurve(xdata, fluxes_all, errors_all)

``pgmuvi`` detects that this is 2D input and sets up the light-curve object
accordingly, but you must still choose a 2D model explicitly when fitting
(for example, with ``fit(model="2D")`` or ``set_model("2D"); fit()``).

Fitting a 2D Model
-------------------

The fitting workflow is the same as in 1D::

    lc.fit(model="2D")

When several observational channels share one physical wavelength, the default
fit policy retains the first channel encountered in the original aligned input
row order captured before constructor subsampling and warns which channels were
ignored.  This makes the bundled
``examples/data/10131+3049.csv`` source fit-able without silently pooling its
two KELT data streams.  To choose the other channel explicitly::

    lc.fit(
        model="2D",
        duplicate_wavelength_policy="select",
        duplicate_wavelength_selection="KELT/OSN_Johnson.Cousins_R3_1",
    )

The explicit ``"all"`` policy is unavailable until a scientifically validated
instrument-channel calibration strategy exists; requesting it raises
an explicit unsupported-policy exception before model construction.

For side-by-side fits of the two KELT channels without modifying the loaded
source, use
:meth:`~pgmuvi.lightcurve.Lightcurve.copy_with_duplicate_wavelength_channels`
to create one independent fit target per choice.

For heterogeneous channel sampling (for example, one observational channel has
far more observations than the others), consider using the legacy best-band
initialisation option::

    lc.fit(model="2D", use_best_band_init=True)

This uses a 1-D Lomb–Scargle calculation on the most densely sampled
observational channel instead of the multiwavelength periodogram to seed the
spectral-mixture kernel frequencies.  This can improve frequency initialisation
when one channel has far more observations than the others.

Assessing sampling quality per observational channel
------------------------------------------------------

Because each observational channel may have a different cadence, assess data
quality per channel before fitting.  The existing method name is retained for
backward compatibility::

    lc.assess_sampling_quality_per_band()

You can then filter out observational channels that have insufficient coverage
using the existing compatibility method::

    lc = lc.filter_well_sampled_bands(min_points=20)

.. warning::

   **TBD[instrument-channel-calibration]:** The low-level API in
   :mod:`pgmuvi.instrument_channel_calibration` can construct deterministic
   one-to-one exact-time or nearest-within-caller-tolerance pairs, preserve
   their provenance, fit and apply an affine mapping, record explicit
   dataset-level plans across shared-wavelength groups, and execute only the
   choices recorded in those plans. Execution returns copied arrays and
   completed provenance; it does not choose the reference channel, pairing
   method, tolerance, or calibration family; interpolate measurements; mutate
   input arrays; merge observational channels; propagate fitted-coefficient
   uncertainty; or integrate calibration automatically into a light-curve fit.

Visualising 2D Results
-----------------------

Use :meth:`~pgmuvi.lightcurve.Lightcurve.plot` to visualise multiband fits. For
2D data, it shows each band's observations together with the GP predictive mean and
credible interval.

.. note::

   **TBD[multidimensional-psd-plotting]:**
   :meth:`~pgmuvi.lightcurve.Lightcurve.plot_psd` does not currently support
   2-D models; a multidimensional PSD visualization contract is not yet
   implemented.

Kernel Choices for 2D Models
------------------------------

The choice of kernel depends on which 2D model family you are using:

* **``model="2D"`` (non-separable):** Uses a 2D spectral-mixture kernel that
  treats time and wavelength jointly.  There is no separate "wavelength kernel"
  to configure; the model learns the joint covariance structure directly.

* **Separable model family** (``"2DSeparable"``, ``"2DAchromatic"``,
  ``"2DWavelengthDependent"``, etc.): Uses an explicit product kernel where the
  temporal spectral-mixture kernel and the wavelength kernel are independent
  factors.  Common choices for the wavelength kernel include:

  * **RBF (Squared-Exponential):** Smooth, continuous wavelength dependence.
  * **Matérn ν = 3/2:** Slightly rougher wavelength dependence, more robust to
    outliers.

  See the ``separable_kernels_2d.py`` and ``dust_mean_spectral_mixture_2d.py``
  example scripts for illustrations of different kernel configurations.

Separable vs Non-Separable Models
-----------------------------------

``pgmuvi`` supports more than one kind of 2D kernel.  The explicit
``2DSeparable`` model family uses **product (separable) kernels**, where the time
and wavelength covariance are modeled as separate factors.  In these models, the
separability assumption means that the temporal correlation structure is the same at
every wavelength (up to a scaling factor).  This is a reasonable approximation for
many sources but may break down when:

* the period changes significantly with wavelength (e.g., in accretion disk reverberation),
* the variability mechanism differs qualitatively between bands.

By contrast, ``model="2D"`` uses a 2D spectral-mixture kernel rather than an
explicit ``k_time * k_wavelength`` ProductKernel construction, so the separability
assumption above should only be interpreted as applying to the ``2DSeparable``
family.

In such cases, consider fitting each band independently and comparing the inferred
periods, or contact the ``pgmuvi`` developers to discuss extensions.

2D tutorial
-----------

The maintained :doc:`../notebooks/pgmuvi_tutorial_2d` notebook builds a
deterministic three-band light curve and performs a required
``2DWavelengthDependent`` consensus fit.  It verifies successful training and
period recovery, then calls ``Lightcurve.plot()`` to render fitted predictions
before recording LPV-relevant follow-up configurations.  Detailed controls remain documented in
:doc:`consensus_fitting`.
