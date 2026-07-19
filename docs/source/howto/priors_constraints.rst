Setting Priors and Constraints
==============================

This guide explains how to use priors and constraints on GP hyperparameters in
``pgmuvi``, why they matter, and when to adjust them.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

GP hyperparameters can take a wide range of values, and without any guidance the
optimiser or sampler may explore physically unreasonable regions.  ``pgmuvi``
provides two mechanisms to constrain the parameter space:

* **Constraints** — hard bounds on parameter values (applied during
  optimisation; MCMC support is planned for a future release).
* **Priors** — probabilistic regularisation, most relevant during MCMC
  (planned) but also useful in MAP optimisation with regularisation.

Default Priors and Constraints
--------------------------------

``pgmuvi`` applies sensible default priors and constraints when you call
:meth:`~pgmuvi.lightcurve.Lightcurve.fit`.  You can also apply those defaults
explicitly after setting a model with
:meth:`~pgmuvi.lightcurve.Lightcurve.set_model`.  Call
:meth:`~pgmuvi.lightcurve.Lightcurve.set_default_priors` and
:meth:`~pgmuvi.lightcurve.Lightcurve.set_default_constraints` to restore the
defaults after manually modifying them (a model must be set first)::

    lc.set_model("1D", num_mixtures=2)
    lc.set_default_priors()
    lc.set_default_constraints()

To inspect the currently active priors and constraints::

    print(lc.get_priors())
    print(lc.get_constraints())

Setting a Period Prior
-----------------------

The most commonly adjusted prior is the prior on the mixture means (frequencies),
which corresponds to placing a prior on the variability period.  Use
:meth:`~pgmuvi.lightcurve.Lightcurve.set_period_prior` to set a log-normal or
normal prior on the period::

    # Log-normal prior: log-mean 5.0 (median ≈ 148 days), log-standard-deviation 0.5
    lc.set_period_prior(
        prior_type="lognormal",
        mu=5.0,
        sigma=0.5,
    )

    # Inspect the resulting prior
    print(lc.get_period_prior())

For a log-normal prior, ``mu`` and ``sigma`` are defined in log-period space
(natural units).  A ``sigma`` of 1.0 corresponds to a broad prior spanning
roughly an order of magnitude; reduce it to tighten the prior around a known
period.

Setting Constraints
--------------------

Constraints are hard bounds applied as GPyTorch ``Interval`` constraints.
:meth:`~pgmuvi.lightcurve.Lightcurve.set_constraint` accepts a dictionary mapping
parameter names to GPyTorch constraint objects.  The most common use case is
restricting the period search range::

    from gpytorch.constraints import Interval

    # Restrict mixture_means (frequencies) to [1/1000, 1/10] per day
    lc.set_constraint({
        "mixture_means": Interval(
            lower_bound=1.0 / 1000.0,   # lower frequency = longer period
            upper_bound=1.0 / 10.0,     # upper frequency = shorter period
        ),
    })

For the noise variance::

    from gpytorch.constraints import Interval

    lc.set_constraint({
        "noise": Interval(lower_bound=1e-6, upper_bound=1.0),
    })

.. note::

   For **separable 2D models** (``"2DSeparable"``, ``"2DAchromatic"``,
   ``"2DWavelengthDependent"``, etc.), constraints on ``mixture_means`` apply to
   the temporal spectral-mixture component; the wavelength kernel parameters are
   separate and can be constrained independently.  When the schema-driven
   parameter workflow is enabled, non-achromatic separable models receive a
   data-derived wavelength-lengthscale interval.  That interval is converted to
   the GP input coordinate and intersected with any existing registered bounds,
   so an existing tighter user interval is not loosened or replaced.

   The wavelength-dependent mean parameters in
   ``"2DWavelengthDependent"``, ``"2DDustMean"``, and
   ``"2DPowerLawMean"`` also use registered GPyTorch raw-parameter
   constraints.  Their data-derived intervals are therefore enforced during
   optimization rather than merely reported for plain ``nn.Parameter``
   objects.  Dust and power-law wavelength formulas are evaluated in the
   reconstructed physical wavelength coordinate when an affine input transform
   is active.

   For the full non-separable ``"2D"`` spectral-mixture baseline, GPyTorch
   tensor-valued intervals are registered with shape ``(1, 1, 2)``.  They
   broadcast across components while keeping ARD index 0 (temporal frequency)
   independent from ARD index 1 (wavelength frequency).  The same separation is
   applied to ``mixture_scales``.  Source-type period limits modify only the
   temporal entry.

   Consensus fitting also respects this separation.  Consensus frequency and
   scale intervals intersect only ARD index 0, while the wavelength-frequency
   bounds at ARD index 1 remain unchanged.  The multi-component workflow still
   uses one broad temporal interval across accepted components; it does not
   impose that temporal interval on the wavelength coordinate.

Using Pre-defined Constraint Sets
-----------------------------------

For common astrophysical source classes, ``pgmuvi`` ships pre-defined constraint
sets (e.g. ``"LPV"`` for Long Period Variables).  Retrieve one with
:func:`~pgmuvi.constraints.get_constraint_set` and inspect or modify the bounds
before applying them::

    from pgmuvi.constraints import get_constraint_set

    # Get the predefined LPV constraint set
    cs = get_constraint_set("LPV")
    print(cs)

You can also use the helper functions in :mod:`pgmuvi.constraints` to build
individual constraints that are appropriate for your data span::

    from pgmuvi.constraints import period_constraint, lengthscale_constraint

    data_span = float(lc.xdata.max() - lc.xdata.min())

    lc.set_constraint({
        "mixture_means": period_constraint(data_span=data_span),
        "mixture_scales": lengthscale_constraint(span=data_span),
    })

See :mod:`pgmuvi.constraints` for the full API.

Maintained wavelength-constraint tutorial
-----------------------------------------

The public :doc:`../notebooks/tutorial_wavelength_constraints` notebook shows
how wavelength sampling produces raw and model-coordinate intervals, how mean
and covariance constraints differ, how registration-before-value ordering is
preserved, and how near-bound and at-bound diagnostics should be interpreted.
Its quasi-periodic fitting example is the tested LPV default hypothesis rather
than a universal requirement for every source.


Common Pitfalls
----------------

* **Period outside the constraint bounds:** If the true period lies outside the
  constrained frequency range, the fit will never find it.  Check that the
  detectable period range (from
  :meth:`~pgmuvi.lightcurve.Lightcurve.compute_sampling_metrics`) is within your
  constraints.

* **Overly tight priors:** A prior that is too narrow can prevent convergence even
  when the true period is nearby.  During exploratory analysis, use wide priors and
  tighten them once you have an initial estimate.

* **Conflicting constraints and priors:** Ensure that the prior has non-negligible
  mass within the constrained region.  A prior mode outside the constraint interval
  will bias the estimate towards the constraint boundary.
