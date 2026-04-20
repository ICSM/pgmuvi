Multiwavelength (2D) Analysis
==============================

This guide explains how to fit GP models to light curves observed simultaneously in
multiple photometric bands or across a range of wavelengths.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

When a source is observed in multiple bands, ``pgmuvi`` can fit a **2D Gaussian
process** whose inputs are ``(time, wavelength)``.  The model captures:

* the temporal variability structure (periods, noise) shared across bands,
* the wavelength dependence of variability amplitude and coherence,
* band-specific noise levels.

This is more informative than fitting each band independently because the model
borrows statistical strength across bands, particularly when some bands have sparse
coverage.

Data Format for 2D Models
--------------------------

For 2D / multiband fitting, ``xdata`` must have shape ``(N, 2)`` where:

* column 0 is the **observation time** (same unit across all observations),
* column 1 is the **wavelength or band index** (a numeric label, e.g., effective
  wavelength in microns, or an integer band code 0, 1, 2, …).

All bands are stacked into a single array::

    import numpy as np
    import pgmuvi

    # Two bands: times_b0/b1, fluxes_b0/b1, errors_b0/b1
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

For heterogeneous band sampling (e.g., one band has far more observations than
others), consider using the best-band initialisation option::

    lc.fit(model="2D", use_best_band_init=True)

This runs the Lomb–Scargle periodogram on the most densely sampled band before
computing the multiband periodogram, which can substantially improve frequency
initialisation.

Assessing Sampling Quality per Band
-------------------------------------

Because each band may have a different cadence, it is important to assess data quality
per band before fitting::

    lc.assess_sampling_quality_per_band()

You can then filter out bands that have insufficient coverage::

    lc = lc.filter_well_sampled_bands(min_points=20)

Visualising 2D Results
-----------------------

Use :meth:`~pgmuvi.lightcurve.Lightcurve.plot` to visualise multiband fits. For
2D data, it shows each band's observations together with the GP predictive mean and
credible interval.

The :meth:`~pgmuvi.lightcurve.Lightcurve.plot_psd` method does not currently
support 2D models; plotting PSDs in more than one dimension is not implemented.

Kernel Choices for 2D Models
------------------------------

The temporal kernel (first dimension) is a spectral mixture kernel, as in 1D.
The wavelength kernel (second dimension) can be configured separately.  Common choices
include:

* **RBF (Squared-Exponential):** Smooth, continuous wavelength dependence.
* **Matérn ν = 3/2:** Slightly rougher wavelength dependence, more robust to outliers.

See the ``separable_kernels_2d.py`` and ``dust_mean_spectral_mixture_2d.py`` example
scripts for illustrations of different kernel configurations.

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

2D Tutorial
-----------

A dedicated notebook tutorial for 2D multiwavelength analysis is provided in the
User Guide:

.. toctree::
   :maxdepth: 1

   ../notebooks/pgmuvi_tutorial_2d
