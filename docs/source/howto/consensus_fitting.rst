Single-source GP fitting and consensus workflow
================================================

.. note::

   **Documentation status:** current through PR99.

   **Scope:** this guide covers one source at a time.  It explains the baseline
   1-D and 2-D fits, the single-component consensus workflow, the LPV-relevant
   separable model families, learned additional noise, and the numerical
   defaults that matter before interpreting a result.

What this workflow is for
-------------------------

A PGMUVI fit combines three decisions that should not be confused:

1. whether the data are one-dimensional or multiwavelength;
2. which covariance and mean-function family represents the source; and
3. whether the model is fitted directly or initialized through multiband
   period consensus.

``fit_strategy="consensus"`` is not a model name.  It is a fitting strategy for
2-D light curves whose bands are expected to share a coherent temporal signal.
The requested GP model is still supplied separately through ``model=``.

Decision path
-------------

Use the following order for a new source.

Single band
   Start with the 1-D spectral-mixture model:

   .. code-block:: python

      result = lc.fit(
          model="1D",
          training_iter=500,
          miniter=100,
          learn_additional_noise=True,
          verbose=True,
      )

   A 1-D light curve cannot use consensus fitting because there are no separate
   bands from which to construct a multiband consensus.

Multiwavelength baseline
   Start with a direct ``2D`` fit when you need a general non-separable
   time--wavelength spectral-mixture baseline:

   .. code-block:: python

      result = lc.fit(
          model="2D",
          training_iter=500,
          miniter=100,
          learn_additional_noise=True,
          verbose=True,
      )

Coherent period shared across bands
   Add ``fit_strategy="consensus"`` when the accepted bands should support a
   common temporal period or frequency:

   .. code-block:: python

      result = lc.fit(
          model="2D",
          fit_strategy="consensus",
          training_iter=500,
          miniter=100,
          learn_additional_noise=True,
          verbose=True,
      )

LPV wavelength structure
   After the ``2D`` baseline, compare an LPV-relevant separable model when the
   scientific question concerns a wavelength-dependent mean or covariance.
   For a coherent LPV period, use a quasi-periodic time kernel:

   .. code-block:: python

      result = lc.fit(
          model="2DDustMean",
          fit_strategy="consensus",
          time_kernel_type="quasi_periodic",
          wavelength_kernel_type="rbf",
          training_iter=500,
          miniter=100,
          learn_additional_noise=True,
          verbose=True,
      )

   The corresponding comparison families are ``2DWavelengthDependent`` and
   ``2DPowerLawMean``.  These fits are comparisons, not automatic model
   selection.

Input requirements
------------------

Consensus fitting requires all of the following:

* a 2-D ``Lightcurve`` whose first input column is time and second column is
  wavelength;
* one band label per observation row through ``band=`` or a recognized CSV band
  column;
* enough usable points in at least two bands after construction-time sampling
  and variability checks; and
* a model whose temporal kernel exposes either spectral-mixture frequency
  parameters or a quasi-periodic ``period_length`` parameter.

Load and validate the source before fitting:

.. code-block:: python

   from pgmuvi.lightcurve import Lightcurve

   lc = Lightcurve.from_csv(
       "source.csv",
       check_sampling=True,
       max_samples=1000,
       max_samples_per_band=100,
       verbose=True,
   )

``check_sampling`` belongs to ``Lightcurve.from_csv`` or the constructor, not to
``fit``.  See :doc:`loading_data` for finite-row filtering, positive-error
requirements, band handling, and subsampling behavior.

Model and kernel roles
----------------------

.. list-table:: Single-source model choices
   :header-rows: 1

   * - Model
     - Recommended role
     - Consensus behavior
   * - ``1D``
     - Single-band spectral-mixture baseline.
     - Not applicable.
   * - ``2D``
     - General non-separable 2-D spectral-mixture baseline in joint time and
       wavelength frequency space.
     - Consensus initializes and can constrain the temporal spectral-mixture
       frequency and scale parameters.
   * - ``2DSeparable``
     - Generic direct product-kernel baseline with a Matérn time kernel and RBF
       wavelength kernel by default.
     - The default model has no spectral-mixture or ``period_length`` handoff;
       do not use it as the standard consensus path.
   * - ``2DWavelengthDependent``
     - Flexible separable covariance with a smooth wavelength kernel and a
       configurable wavelength-dependent mean.
     - Use ``time_kernel_type="quasi_periodic"`` for period-length handoff or
       ``"spectral_mixture"`` for frequency-space handoff.
   * - ``2DDustMean``
     - Separable covariance with a dust-inspired wavelength mean.
     - Same supported consensus time-kernel choices as
       ``2DWavelengthDependent``.
   * - ``2DPowerLawMean``
     - Separable covariance with a power-law wavelength mean.
     - Same supported consensus time-kernel choices as
       ``2DWavelengthDependent``.

The generic ``2DSeparable`` family is useful as a direct product-kernel control,
but its convenience interface does not accept ``time_kernel_type`` or
``wavelength_kernel_type`` selectors.  Use ``2DWavelengthDependent`` when you
need those string-configurable kernels.

For a model-by-model explanation of the wavelength means, covariance
assumptions, coordinate requirements, and try-first ordering, see
:doc:`wavelength_models`.

Choosing the temporal kernel
----------------------------

``2D`` spectral mixture
   Use the built-in full 2-D spectral-mixture kernel for the broadest baseline.
   Consensus is handed to the temporal dimension of its mixture parameters.

``time_kernel_type="quasi_periodic"``
   Use this for an LPV-like signal with one dominant shared period whose waveform
   can evolve.  Consensus is converted from frequency to the kernel's
   ``period_length`` parameter.

``time_kernel_type="spectral_mixture"``
   Use this when the separable model needs a more flexible temporal PSD or more
   than one spectral component.  ``num_mixtures`` controls the number of time
   components for these separable models.

``time_kernel_type="matern"`` or ``"rbf"``
   These kernels are valid for direct fits, but they expose neither a
   spectral-mixture frequency target nor a ``period_length`` target.  They are
   therefore not the normal choice for ``fit_strategy="consensus"``.

The current flagship LPV path is a single-component consensus handed to a
quasi-periodic time kernel.  Do not increase ``num_mixtures`` merely because the
option exists; first establish that the data support additional temporal
components.

Direct fit versus consensus fit
-------------------------------

A direct fit optimizes the requested model from its normal initialization and
parameter-workflow estimates.  A consensus fit first performs per-band
frequency analysis, rejects unsuitable bands or outlying frequencies, builds a
shared temporal estimate, and then runs the final GP fit.

Consensus is especially useful when:

* cadence and temporal coverage differ strongly between bands;
* individual Lomb--Scargle peaks disagree in a minority of bands;
* a 2-D spectral-mixture model needs a data-supported temporal initialization;
  or
* an LPV separable model needs a shared period handed to a quasi-periodic time
  kernel.

Consensus should not be forced when the accepted bands genuinely support
incompatible periods.  That outcome can be a source-physics or data-quality
result rather than a software defect.

Learned additional noise and uncertainty units
----------------------------------------------

With per-point uncertainties, the default automatic likelihood treats stored
``yerr`` values as standard deviations because ``variance=False`` by default.
The values are squared before being supplied to the fixed-noise likelihood.

Set ``learn_additional_noise=True`` when the quoted uncertainties may not account
for all scatter:

.. code-block:: python

   result = lc.fit(
       model="2D",
       fit_strategy="consensus",
       variance=False,
       learn_additional_noise=True,
       training_iter=500,
       miniter=100,
   )

This adds one learned homoscedastic variance term on top of the supplied
per-observation variances.  It does not replace the reported errors.  Do not set
``variance=True`` unless the stored uncertainty column already contains
variances rather than standard deviations.  With a ``ytransform``, standard
deviations use the fitted scale once and variances use its square; location
shifts are never applied to either quantity.

Initialization, constraints, and numerical stability
-----------------------------------------------------

PGMUVI currently uses ``torch.float64`` as its package default dtype.  Time
centering is also enabled automatically when no explicit input transform is
supplied; this corresponds to ``center_time="auto"`` at light-curve
construction.  These defaults reduce numerical problems caused by large
absolute time stamps.

For normal use:

* retain automatic time centering rather than disabling it;
* use finite, strictly positive uncertainty values;
* keep the construction-time sample limits appropriate for available memory;
* treat ``training_iter=50`` and ``miniter=10`` as smoke-test values only;
* start scientific fits with longer runs such as ``training_iter=500`` and
  ``miniter=100``, then inspect convergence rather than assuming those numbers
  are universally sufficient; and
* avoid changing optimizer, learning rate, conjugate-gradient limits, or
  transforms until the default run has been diagnosed.

The schema-driven parameter workflow is enabled by default through
``use_parameter_workflow=True``.  After fitting, inspect
``get_parameter_workflow_summary()`` rather than assuming every proposed value
or constraint was applied.

For LPVs, ``constraint_set="LPV"`` applies the package's LPV constraint set,
including a minimum period of 100 in the native time units.  Use it only when
those units and that scientific restriction are appropriate:

.. code-block:: python

   result = lc.fit(
       model="2DDustMean",
       fit_strategy="consensus",
       time_kernel_type="quasi_periodic",
       constraint_set="LPV",
       use_parameter_workflow=True,
       training_iter=500,
       miniter=100,
   )

Consensus controls
------------------

The standard consensus path accepts controls through ``fit`` keyword arguments,
including:

``min_points_per_band``
   Minimum number of observations required for a band to participate.

``max_gap_fraction`` and ``min_duty_cycle``
   Sampling-quality gates used when deciding which bands are acceptable.

``outlier_sigma``
   Controls rejection of per-band frequencies that are inconsistent with the
   robust consensus.

``use_acf``
   Enables the optional ACF contribution to the consensus diagnostics.

``constrain_consensus`` and ``consensus_width_factor``
   Control whether and how tightly the final temporal parameters are constrained
   around the consensus estimate.

Do not tune several consensus controls simultaneously without recording the
configuration and checking which bands were accepted or rejected.

Single-component and multi-component strategies
------------------------------------------------

``fit_strategy="consensus"`` is the recommended first consensus run.  It uses
one dominant frequency per acceptable band and constructs one robust shared
component.

``fit_strategy="consensus_multicomp"`` is an implemented staged workflow for
extracting, clustering, and aggregating multiple frequency components before a
final 2-D spectral-mixture fit.  It is not the default LPV workflow.  The current
implementation still applies one broad global frequency interval to accepted
components rather than component-specific constraints.

.. admonition:: TBD: multi-periodic support

   Component-specific constraints, model-family support beyond the final 2-D
   spectral-mixture fit, and a complete scientific validation standard for
   multi-periodic sources remain future work.

Running the example
-------------------

The maintained example is ``examples/consensus_multiband_fit.py``.  It can use a
CSV source or a deterministic synthetic multi-band light curve.

Inspect the resolved configuration without training:

.. code-block:: bash

   PYTHONPATH=. python3 examples/consensus_multiband_fit.py \
       --model 2D \
       --fit-strategy consensus \
       --dry-run

Run a short synthetic smoke test:

.. code-block:: bash

   PYTHONPATH=. python3 examples/consensus_multiband_fit.py \
       --model 2D \
       --fit-strategy consensus \
       --training-iter 50 \
       --miniter 10 \
       --output-dir consensus_fit_output \
       --verbose

Run the LPV dust-mean configuration on a CSV file:

.. code-block:: bash

   PYTHONPATH=. python3 examples/consensus_multiband_fit.py source.csv \
       --model 2DDustMean \
       --fit-strategy consensus \
       --time-kernel-type quasi_periodic \
       --wavelength-kernel-type rbf \
       --training-iter 500 \
       --miniter 100 \
       --learn-additional-noise \
       --output-dir source_consensus_fit \
       --verbose

When ``--output-dir`` is supplied, the script writes
``run_configuration.json`` and, when available, ``period_summary.json``,
``consensus_diagnostics.json``, and ``fit_history.json``.  A rejected consensus
writes ``failure.json`` and returns exit status 2.  An unrelated exception also
writes ``failure.json`` but returns exit status 1.

Inspecting a successful fit
---------------------------

``get_period_summary()`` returns a ``PeriodSummaryResult`` object rather than a
preformatted report string:

.. code-block:: python

   summary = lc.get_period_summary()
   print(summary.dominant_period)
   print(summary.dominant_frequency)

   primary = summary.get_primary_peak()
   if primary is not None:
       print(primary.period, primary.area_fraction)

   summary_dict = summary.as_dict()

Also inspect:

.. code-block:: python

   print(lc.consensus_diagnostics)
   print(lc.get_parameter_workflow_summary())
   print(lc.get_fit_history_summary())
   lc.export_fit_history_json("fit_history.json")

The consensus diagnostics identify accepted and rejected bands, per-band
periods or frequencies, the consensus estimate, and the parameter target used
for the final handoff.

Failure contract
----------------

``ConsensusFitError`` represents a rejected consensus workflow, not every
possible optimizer or linear-algebra failure.  The exception carries
``failure_diagnostics`` and, when available, ``failure_summary``.  The
``Lightcurve`` also records ``fit_failed``, ``failure_reason``,
``failure_diagnostics``, and ``consensus_diagnostics``.

Common causes include:

* too few usable bands after sampling or variability filtering;
* inconsistent dominant periods across bands;
* no accepted frequency cluster;
* a model/time-kernel combination with no consensus-compatible parameter; or
* final parameters that do not satisfy the registered consensus constraint.

For batch work, record this failure and continue.  Do not silently relabel a
rejected consensus as a successful model fit.

Notebook tutorial
-----------------

The maintained :doc:`../notebooks/pgmuvi_tutorial_2d` notebook provides an
interactive baseline ``2D`` consensus workflow with deterministic data, an
explicit no-training default, structured failure handling, and LPV-relevant
follow-up configurations.  The runnable script remains the preferred reference
for command-line execution and artifact export.

Relationship to wavelength advisory
-----------------------------------

``fit_strategy="consensus"`` executes one requested model configuration.
``run_period_independent_wavelength_advisory_workflow`` constructs, runs, and
scores several model/kernel configurations.  The advisory workflow may use
consensus internally, but it does not automatically select or install a final
model.

Use this guide for a focused fit.  Use :doc:`wavelength_advisory` when comparing
several LPV-relevant wavelength models.

See also
--------

* :doc:`loading_data`
* :doc:`preprocessing`
* :doc:`multiband`
* :doc:`wavelength_advisory`
* :doc:`wavelength_advisory_batch`
* :doc:`interpreting_results`
