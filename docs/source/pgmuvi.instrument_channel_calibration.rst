Instrument-channel calibration
==============================

.. automodule:: pgmuvi.instrument_channel_calibration
   :members:
   :undoc-members:
   :show-inheritance:

Assessment and explicit paired calibration
------------------------------------------

The assessment API preserves the distinction between an observational channel
and its physical wavelength coordinate.  It deterministically reports groups
in which multiple observational channels share one physical wavelength, but it
does not alter wavelengths, merge channels, average their measurements, or
apply a correction.

The :class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelPairing`
record provides a strict JSON-safe provenance contract for explicit pairs.  It
records source-row indices, paired times, time units, construction provenance,
the caller-supplied tolerance when applicable, unmatched counts, reuse policy,
interpolation provenance, and derived time-separation summaries.

:func:`~pgmuvi.instrument_channel_calibration.construct_instrument_channel_pairing`
implements two caller-selected methods:

* ``exact_timestamp`` pairs only numerically identical time coordinates;
* ``nearest_within_tolerance`` requires a finite positive tolerance supplied by
  the caller, maximizes the number of chronological one-to-one pairs, and then
  minimizes their summed absolute time separation.

The inputs must already share one time coordinate and unit.  PGMUVI does not
infer or convert time systems in this callable.  It does not determine whether
the selected method or tolerance is scientifically appropriate.

Both construction methods currently use a dynamic-programming grid with
``O(n_reference * n_channel)`` time and memory cost.  Callers with large
channels should restrict the input observations to the time range relevant to
the calibration before constructing pairs.

Dataset-level orchestration contract
------------------------------------

The
:func:`~pgmuvi.instrument_channel_calibration.define_instrument_channel_calibration_plan`
callable validates an immutable, JSON-safe plan covering every
shared-wavelength group reported by an explicit
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelCalibrationAssessment`.

For each group, the caller must select the reference channel.  Every other
observational channel receives an explicit ``planned``, ``skipped``, or
``unavailable`` disposition.  Planned entries require a caller-selected pairing
method, time unit, calibration family, and any applicable non-zero tolerance.
Skipped and unavailable entries instead require a reason.

Optional
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelPairing` and
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelCalibration`
records can be attached to preserve pairing and fitted-coefficient provenance.
The plan itself does not construct pairs, fit or apply mappings, merge channels,
alter wavelengths, mutate a light curve, or integrate calibration into fitting.

Coefficient-uncertainty provenance contract
-------------------------------------------

The version-2
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelCalibration`
record requires an explicit
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelCalibrationCoefficientUncertainty`
record.  Available uncertainty stores the complete symmetric 2-by-2 covariance
matrix in the fixed coefficient order ``offset, scale``.  Offset and scale
standard errors are derived from the covariance diagonal rather than stored as
independent values.  The record also preserves an explicit uncertainty source,
estimation method, and optional degrees of freedom and residual variance.

The affine fitter estimates covariance for its final fixed-weight least-squares
subproblem, conditional on the final MAD-clipped inlier set, when either no
measurement errors or only reference-channel errors are supplied.  With no
supplied errors it uses the ordinary-least-squares normal-matrix inverse scaled
by the residual variance
``sum(residual**2) / (n_inliers - 2)``.  Reference-channel errors use a
known-variance weighted normal-matrix inverse without residual-variance
rescaling.

Coefficient covariance is unavailable when channel-axis errors are supplied.
Those errors produce effective residual variances that depend on the fitted
scale, while the current iterative weighting procedure does not expose a
full-objective Hessian or estimating-equation covariance for that dependence.
PGMUVI therefore records an explicit unavailable reason rather than presenting
a frozen-weight normal-matrix inverse as uncertainty for the complete
estimator.

Covariance also remains unavailable, with a deterministic reason, when the
final weighted design has zero residual degrees of freedom, invalid numerical
inputs, a failed singular-value decomposition, or numerical rank deficiency.
Available covariance is local and conditional: it does not include uncertainty
from pair construction, MAD-clipping selection, calibration-family choice, or
caller choices about observational-channel pairing.  Its fixed coefficient
order is ``offset, scale``.  Application still propagates only measurement
uncertainty through the fitted scale; it does not propagate the estimated
coefficient covariance into calibrated predictions.

Executing an explicit plan
--------------------------

The
:func:`~pgmuvi.instrument_channel_calibration.execute_instrument_channel_calibration_plan`
callable executes only the choices recorded in an
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelCalibrationPlan`.
For each ``planned`` entry it reuses attached provenance when present, or
constructs pairs using the recorded method and tolerance, fits the recorded
``affine`` family, and applies the mapping to all rows of that target channel at
the shared wavelength. ``skipped`` and ``unavailable`` entries remain unchanged.

Execution returns an immutable
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelCalibrationExecution`
containing copied calibrated arrays, the exact source-row indices transformed,
and a completed plan with pairing and fitted calibration provenance.
It does not mutate input arrays, merge channels, alter wavelengths, or choose
scientific configuration automatically.  It does not propagate
fitted-coefficient uncertainty or integrate calibration into
``Lightcurve.fit()``.

The fitting API accepts **caller-supplied paired** flux measurements for one
observational channel and an explicitly named reference channel.  It fits the
affine mapping

.. math::

   f_{\mathrm{reference}}
   =
   b + a f_{\mathrm{channel}},

where ``b`` is the stored offset and ``a`` is the strictly positive stored
scale.  Optional measurement uncertainties contribute to weighted least
squares, and iterative MAD clipping can reject discrepant pairs.

Pair construction is explicit rather than implicit: the caller selects
the reference channel, method, and any non-zero tolerance.  The construction
callable does not interpolate, reuse an observation, reconcile time systems,
select a tolerance, or select a reference channel.  Non-exact pairs are
described as nearest-within-tolerance pairs rather than as simultaneous
measurements.

PGMUVI also does not select between additive, multiplicative, affine, or other
calibration families.  The pairing configuration and the decision to use this
explicit affine model remain the caller's scientific responsibility.

Applying a fitted calibration maps fluxes onto the reference-channel scale.
Flux uncertainties are multiplied by the absolute fitted scale.  The current
application function does not propagate uncertainty in the fitted offset or
scale themselves.

Predictive-uncertainty propagation
----------------------------------

``apply_instrument_channel_calibration_with_predictive_uncertainty`` applies
the same affine mapping as ``apply_instrument_channel_calibration`` and returns
an ``InstrumentChannelCalibrationPredictiveUncertainty`` record alongside the
calibrated flux.  The existing application callable and its return signature
remain unchanged.

For input :math:`x_i`, the coefficient Jacobian is :math:`J_i=[1, x_i]` in
fixed coefficient order ``offset, scale``.  With coefficient covariance
:math:`C`, shared coefficient covariance is
:math:`J_i C J_j^T`.  Marginal coefficient variance is split into offset
variance, scale variance, and the offset-scale covariance cross-term.
Independent input measurement errors contribute
:math:`a^2 \sigma_i^2` only on the diagonal.  Omitting input errors produces
an explicit zero measurement-variance contribution, and the propagation
assumes measurement errors are independent of fitted coefficients.

``marginal_variance`` records flattened row-major marginal variance components
and derived standard deviation.  ``full_covariance`` additionally records
correlations between predictions that share fitted coefficients.  Scalar input
uses ``input_shape=()``; array results preserve their original shape.

Unavailable coefficient covariance, including current fits with
scale-dependent channel-axis errors, produces an explicit ``unavailable``
result with no numerical predictive uncertainty.  There is no silent fallback
to measurement-only uncertainty.  Predictive propagation is implemented only
in the dedicated application callable; current dataset orchestration continues
to record that propagation was not requested or performed.  The stable future
orchestration dispositions remain ``not_requested``, ``available``,
``skipped``, and ``unavailable``.

``TBD[instrument-channel-calibration]`` remains open
----------------------------------------------------

The low-level paired affine fit and application primitives do not complete the
instrument-channel calibration work.  The marker remains open because PGMUVI
still lacks:

* scientifically validated instrument-specific rules for choosing
  pairing methods and tolerances;
* coefficient-uncertainty estimation for scale-dependent channel-axis errors;
* integration into the normal light-curve fitting workflow; and
* validation across representative instruments, filters, and LPV sources.

No automatic calibration, channel merging, wavelength reassignment, or model
selection is performed.
