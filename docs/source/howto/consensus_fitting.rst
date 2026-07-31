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
     - Use ``time_kernel_type="quasi_periodic"`` for period-length handoff or
       ``"spectral_mixture"`` for frequency-space handoff.
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

The generic ``2DSeparable`` family is useful as a direct product-kernel control.
Its convenience interface accepts the same ``time_kernel_type`` and
``wavelength_kernel_type`` selectors used by the other separable families while
retaining a constant mean function.

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
The values are squared before being supplied as fixed per-observation variances,
and PGMUVI now learns one additional homoscedastic variance by default.  The
additional term starts near 10% of the median supplied variance so it begins as
a perturbation rather than replacing the reported uncertainties.

The explicit ``learn_additional_noise=True`` spelling remains available and can
make the statistical choice visible in a saved workflow:

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
per-observation variances.  It does not replace the reported errors.  Pass
``learn_additional_noise=False`` or ``likelihood="fixed"`` to recover the legacy
fixed-noise-only behaviour explicitly.  Do not set ``variance=True`` unless the
stored uncertainty column already contains variances rather than standard
deviations.  With a ``ytransform``, standard deviations use the fitted scale
once and variances use its square; location shifts are never applied to either
quantity.

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
``use_parameter_workflow=True``.  For the separable LPV models, this includes a
data-derived wavelength-kernel lengthscale and interval converted into the
actual GP input coordinate.  This wavelength-covariance estimate is independent
of the consensus period handoff, so it applies with quasi-periodic and
spectral-mixture time kernels alike.  After fitting, inspect
``get_parameter_workflow_report()`` and the wavelength parameter's
``wavelength_estimate_provenance`` rather than assuming every proposed value or
constraint was applied.

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

``two_band_max_fractional_frequency_difference``
   Applies only when exactly two bands provide valid dominant frequencies.
   Because two values cannot support MAD-based outlier rejection, their
   symmetric fractional frequency difference must not exceed this limit
   (default ``0.10``).  Incompatible pairs fail explicitly rather than being
   averaged into a midpoint period supported by neither band.

``use_acf``
   Enables ACF validation and conservative LS--ACF harmonic reconciliation.
   Direct agreement retains the Lomb--Scargle peak.  When ACF identifies a
   longer period related to the LS period by an integer harmonic, consensus
   promotes the lower ACF frequency as the candidate fundamental and records
   ``selected_from='acf_fundamental_harmonic_reconciliation'``.  A shorter-period
   ACF harmonic does not replace the slower LS candidate.  The original LS
   frequency and period remain available in the per-band diagnostics.

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
implementation still applies one broad global **temporal-frequency** interval
across accepted components rather than component-specific temporal intervals.
For the full ``2D`` spectral-mixture baseline, that consensus interval updates
only ARD index 0; the independently derived wavelength-frequency bounds at ARD
index 1 are preserved.

.. admonition:: TBD: multi-periodic support

   component-specific constraints, model-family support beyond the final 2-D
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

Transform isolation during per-band diagnostics
-----------------------------------------------

Consensus splitting constructs temporary one-dimensional light curves from the
raw time, flux, and uncertainty arrays.  These diagnostic objects deliberately
do not inherit the parent light curve's fitted coordinate or flux transforms.
A two-dimensional affine coordinate transform stores separate state for time
and wavelength and cannot be applied safely to a one-dimensional time vector.
Lomb--Scargle, ACF, and sampling diagnostics therefore remain in the original
physical time units, while the parent multiband light curve keeps its configured
transform for the final GP fit.

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
executed end-to-end ``2DWavelengthDependent`` consensus workflow with
deterministic data.  Its default path performs a required fit, verifies fit
history and recovered period, and calls ``Lightcurve.plot()`` to render fitted
predictions before presenting explicit follow-up configurations.  The runnable script remains the preferred reference
for command-line execution and artifact export.

For a real-source workflow, use
:doc:`../notebooks/tutorial_single_source_analysis`.  It computes independent
Lomb--Scargle and data-ACF evidence for every observational channel, runs
consensus before the wavelength-parameter workflow and optimizer, keeps both
shared-wavelength KELT channels in the consensus scope, and records the explicit
single-channel selection used only for exact-GP training.

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


User-controlled LS and GP components; curve-only ACF
-----------------------------------------------------

The single-source workflow exposes two independent controls:

``LS_NUM_COMPONENTS``
   Number of Lomb--Scargle candidates retained per observational channel.

``GP_NUM_COMPONENTS``
   Number of fitted temporal GP components. A value of one selects the
   quasi-periodic ``consensus`` pathway. Values above one select the
   spectral-mixture ``consensus_multicomp`` pathway and set
   ``num_mixtures`` exactly to the requested count.

The ACF does not identify peaks or components in this comparison workflow. It
is plotted only as a lag--correlation curve. LS candidates and fitted-GP period
features are overlaid as references so the reader can inspect whether those
periods correspond to recurrence structure in the data.

``Lightcurve.plot_period_summary()`` accepts ``x_axis="period"`` or
``x_axis="frequency"``. ``log_x`` controls the selected coordinate scale.
For spectral-mixture fits, ``show_components=True`` draws every individual
fitted mixture-component PSD curve together with the summed PSD. For an
explicit-period quasi-periodic fit no PSD exists; the plot therefore shows a
period marker and coherence-proxy interval with no quantitative y-axis.

For a two-dimensional fit the temporal GP is shared across observational
channels. ``plot_period_diagnostic_comparison()`` consequently creates one
shared GP period-summary figure, rather than repeating the same figure once per
channel.

Lomb--Scargle plotting axis contract
------------------------------------

``Lightcurve.plot_lomb_scargle_periodogram()`` is the maintained plotting path
for full Lomb--Scargle results.  It does not recompute the periodogram.  The
default display contract is:

* period on the x-axis;
* logarithmic x scaling; and
* linear Lomb--Scargle power on the y-axis.

A frequency-axis display remains available as an explicit override for
specialized diagnostics.  The method accepts an existing axes, so several
periodograms can be compared without reproducing conversion, sorting, or scale
logic in notebooks.

All supplied LS candidates and labeled reference periods are retained.  This is
important for multi-component signals: the plotting API never silently reduces
a result to the strongest peak.  Candidate ranks are visual annotations only and
are not treated as physical component identities.

A prior full-grid call can be plotted directly:

.. code-block:: python

   lc.fit_LS(freq_only=True)
   fig, ax = lc.plot_lomb_scargle_periodogram(show=False)

Explicit arrays and multiple reference components are also supported:

.. code-block:: python

   fig, ax = lc.plot_lomb_scargle_periodogram(
       frequency,
       power,
       reference_periods={
           "Primary candidate": primary_period,
           "Secondary candidate": secondary_period,
       },
       show=False,
   )

Parameter-workflow control under consensus fits
-----------------------------------------------

``use_parameter_workflow`` is preserved when ``fit_strategy`` is
``"consensus"`` or ``"consensus_multicomp"``.  This makes the following a
valid controlled comparison::

    enabled = lightcurve_a.fit(
        model="2DWavelengthDependent",
        fit_strategy="consensus",
        constraint_set="LPV",
        use_parameter_workflow=True,
        ...
    )

    disabled = lightcurve_b.fit(
        model="2DWavelengthDependent",
        fit_strategy="consensus",
        constraint_set="LPV",
        use_parameter_workflow=False,
        ...
    )

The two objects should contain identical retained observations and should be
fit with the same seed and optimizer controls.  Only the schema-driven
parameter workflow changes.

When the two independent light curves are reloaded from the same CSV,
each load must replay the same random seed, PyTorch default dtype, working
directory, and sampling options.  Compare the retained tensors and stable
sampling-contract fields directly; do not use equality of an entire diagnostic
dictionary as a substitute for confirming identical retained data.

For numerical parameter tables, ``Lightcurve.get_parameters(raw=False)``
returns model-relative names with ``raw_`` removed.  Resolve the corresponding
constraint on ``lightcurve.model`` using the registered raw parameter name;
the wrapper object does not own submodules such as ``mean_module`` or
``covar_module``.

Interpreting registered parameter constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A registered constraint, a valid initialization, and a fitted value inside the
interval are software and numerical checks.  They do not by themselves show
that the parameter is statistically well identified or that the model is
scientifically adequate.  Report the fitted fractional position within the
interval and distinguish interior, near-bound, and at-bound results.  A useful
technical screen labels a row satisfactory only when the constraint is
registered, initialization and the fitted value are inside it, and the fitted
value is not close to either bound.

Use
:func:`pgmuvi.single_source_analysis.summarize_single_source_constraint_diagnostics`
to produce this interpretation.  Its result records each parameter's model
role, coordinate system, value and interval provenance, technical status,
boundary status, and an explicit identification caveat.  Near-bound and
at-bound parameters require sensitivity checks; an interior result still does
not prove identification.

Notebook tables that must survive PDF export should use Markdown or a plain
tabular representation rather than displaying a raw
``IPython.display.HTML`` object.

This is **not** a comparison between a constrained and an unconstrained GP.
The disabled fit still receives the selected source-type defaults, explicit
user constraints, and consensus-derived temporal initialization or
constraints.  It merely skips automatic schema-driven parameter values and
workflow-derived constraints.  Use
:meth:`pgmuvi.lightcurve.Lightcurve.get_parameter_workflow_report` after each
fit to verify that the enabled fit reports applied entries and the disabled
fit reports ``available=False``.
