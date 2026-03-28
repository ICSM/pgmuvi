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

* **Constraints** — hard bounds on parameter values (applied during both
  optimisation and MCMC).
* **Priors** — probabilistic regularisation, most relevant during MCMC but also
  useful in MAP optimisation with regularisation.

Default Priors and Constraints
--------------------------------

``pgmuvi`` sets sensible defaults automatically when a
:class:`~pgmuvi.lightcurve.Lightcurve` is created.  Call
:meth:`~pgmuvi.lightcurve.Lightcurve.set_default_priors` and
:meth:`~pgmuvi.lightcurve.Lightcurve.set_default_constraints` to restore the
defaults after manually modifying them::

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
uniform prior on the period::

    # Log-normal prior: mode 100 days, width factor ~2
    lc.set_period_prior(period=100.0, period_sigma=0.5)

    # Inspect the resulting prior
    print(lc.get_period_prior())

The ``period_sigma`` argument controls the width of the log-normal prior in
log-frequency space.

Setting Constraints
--------------------

Constraints are hard bounds applied as GPyTorch ``Interval`` constraints.
The most common use case is restricting the period search range::

    # Restrict periods to [10, 1000] days
    lc.set_constraint(
        parameter="mixture_means",
        lower=1.0 / 1000.0,   # lower frequency = longer period
        upper=1.0 / 10.0,     # upper frequency = shorter period
    )

For the noise variance::

    lc.set_constraint(
        parameter="likelihood.noise",
        lower=1e-6,
        upper=1.0,
    )

.. note::

   For 2D (multiband) models, constraints on ``mixture_means`` apply to the
   temporal dimension.  Wavelength-dimension constraints can be set separately.

Setting Arbitrary Priors
-------------------------

You can attach any GPyTorch-compatible prior distribution using
:meth:`~pgmuvi.lightcurve.Lightcurve.set_prior`::

    import torch
    from gpytorch.priors import LogNormalPrior

    lc.set_prior(
        parameter="covar_module.mixture_weights",
        prior=LogNormalPrior(loc=torch.tensor(0.0), scale=torch.tensor(1.0)),
    )

Refer to the `GPyTorch priors documentation
<https://docs.gpytorch.ai/en/stable/priors.html>`_ for available prior classes.

pgmuvi also defines additional prior classes in :mod:`pgmuvi.priors`, including
period-aware priors and priors adapted for multiband models.

Using the ConstraintSet Helper
--------------------------------

For complex multiparameter scenarios, ``pgmuvi`` provides the
:class:`~pgmuvi.constraints.ConstraintSet` helper class that allows you to define
a collection of constraints and apply them all at once::

    from pgmuvi.constraints import ConstraintSet

    cs = ConstraintSet(
        mixture_means=(1e-3, 0.5),
        noise=(1e-6, 0.1),
    )
    cs.apply(lc)

See :mod:`pgmuvi.constraints` for the full API.

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
