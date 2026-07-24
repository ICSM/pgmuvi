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

Instrument-specific pairing and tolerance resolution
----------------------------------------------------

The immutable
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelPairingRule`
record provides JSON-safe instrument-specific pairing guidance.  It records
explicit reference and target instrument identities, explicit
observational-channel identities, physical wavelength, pairing method, time
unit, maximum separation, tolerance provenance, scientific-validation status,
and an evidence reference when validation is claimed.  Physical wavelength is
contextual metadata and is never used in place of observational-channel
identity.

:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelPairingRuleRequest`
records the exact instrument, observational-channel, and physical-wavelength
identity requested by the caller.  The
:func:`~pgmuvi.instrument_channel_calibration.resolve_instrument_channel_pairing_rule`
callable performs deterministic exact-identity lookup over an
explicitly supplied rule catalogue.  It rejects duplicate identifiers,
duplicate explicit
identities, absent exact matches, and matching rules that are not
scientifically validated.  A successful lookup returns the single
evidence-backed rule without fallback.

The immutable
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelPairingRuleCatalogue`
defines the strict JSON-safe contract for a caller-supplied collection of
rules.  It records an explicit catalogue identifier, catalogue version,
provenance reference, catalogue-level validation status, optional validation
evidence, and a non-empty immutable tuple of member rules.  Construction
rejects duplicate rule identifiers and duplicate exact identities.  A
scientifically validated catalogue must carry evidence and cannot silently
upgrade a member rule that is not itself scientifically validated.

The catalogue can be passed explicitly to the existing resolver because it
iterates only over its recorded member rules.  Strict serialization and
deserialization preserve catalogue and rule provenance.  The automatic discovery
and automatic activation mechanisms remain unavailable unless and until the
explicit fail-closed callables are implemented.

The discovery and activation contract adds immutable
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelPairingRuleCatalogueDiscoveryRequest`
and
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelPairingRuleCatalogueActivationRequest`
records.  Discovery requires one explicit source reference and the exact
expected catalogue identifier, catalogue version, and schema version.  Ambient
environment lookup, working-directory scans, package-resource fallback, and
silent activation are prohibited.

The side-effect-free
:func:`~pgmuvi.instrument_channel_calibration.assess_instrument_channel_pairing_rule_catalogue_compatibility`
callable checks exact identity, version, schema, catalogue validation, and
member-rule validation.  It never activates the catalogue and never falls back
to another source or version.  The public discovery and activation callables
fail closed with :exc:`NotImplementedError` until explicit-source loading and
activation are implemented.

The explicit resolver and the discovery/activation contract do not select a
reference observational channel, pairing method, or time tolerance
automatically.  They do not perform approximate physical-wavelength matching,
infer tolerance from cadence, provide populated built-in catalogue content, or
integrate a catalogue into calibration planning or normal light-curve fitting.

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

When channel-axis errors are supplied, the fitter activates the dedicated
scale-dependent full objective after selecting its final MAD-clipped inlier set.
The reported offset, scale, and coefficient covariance then come from the same
converged optimum, and covariance is the inverse observed Hessian of that
objective.  A frozen-weight normal-matrix inverse is not substituted for this
estimator.

Scale-dependent channel-axis uncertainty contract
--------------------------------------------------

:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelCalibrationScaleDependentUncertaintyEstimate`
defines the immutable JSON-safe result of the dedicated
:func:`~pgmuvi.instrument_channel_calibration.estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty`
low-level callable.  The estimator is implemented with analytic objective
derivatives and a log-scale parameterization and is activated by the affine
fitter whenever channel-axis errors participate in the final fit.

For final-inlier paired values, the estimator minimizes the Gaussian
negative log likelihood

.. math::

   \frac{1}{2}\sum_i\left[
   \log(v_i) + \frac{r_i^2}{v_i}
   \right],

where :math:`r_i=y_i-(b+a x_i)` and
:math:`v_i=\sigma_{y,i}^2+a^2\sigma_{x,i}^2`.  Coefficient order remains
``offset, scale`` and the scale domain is strictly positive.  Available
coefficient covariance is defined as the inverse observed Hessian of this full
objective at a converged optimum, not a frozen-weight normal-matrix inverse.

The result records the optimum, objective value, optimizer and convergence
state, gradient method and norm, Hessian method and eigenvalues, final-inlier
conditioning, and deterministic unavailable reason.  It does not include
uncertainty from pair construction, MAD-clipping selection, calibration-family
choice, or caller choices about observational-channel pairing.

Covariance also remains unavailable, with a deterministic reason, when the
final weighted design has zero residual degrees of freedom, invalid numerical
inputs, a failed singular-value decomposition, or numerical rank deficiency.
Available covariance is local and conditional: it does not include uncertainty
from pair construction, MAD-clipping selection, calibration-family choice, or
caller choices about observational-channel pairing.  Its fixed coefficient
order is ``offset, scale``.  Application still propagates only measurement
uncertainty through the fitted scale; it does not propagate the estimated
coefficient covariance into calibrated predictions.

Scale-dependent fitter and orchestration integration contract
---------------------------------------------------------------

:func:`~pgmuvi.instrument_channel_calibration.select_instrument_channel_calibration_uncertainty_estimator`
defines deterministic routing after the fitter has validated optional error
arrays.  No errors and reference-axis errors alone select the existing
``fixed_weight_normal_matrix`` path.  Any observational-channel-axis errors
select ``scale_dependent_full_objective``, whether or not reference-axis errors
are also present.  Neither error axis is silently reinterpreted, frozen, or
ignored by this selection contract.

Each fitter-produced
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelCalibration`
now carries an immutable
:class:`~pgmuvi.instrument_channel_calibration.InstrumentChannelCalibrationFitProvenance`
record.  Pair positions refer to the original caller-supplied paired arrays and
identify both the finite-value subset and exact final MAD-clipped inlier set.
The record also identifies participating error axes, selected uncertainty
estimator, integration status, reported point-estimate source and objective,
and whether the point estimate and covariance share one objective optimum.
The nested ``coefficient_uncertainty`` field remains the sole source of
coefficient covariance in fixed order ``offset, scale``.

Scale-dependent activation is now implemented in the affine fitter.  For
successful fits with observational-channel-axis errors, provenance reports
``active`` and identifies the full-objective optimum as the common source of the
reported offset, scale, and covariance.  The covariance remains conditioned on
the exact final MAD-clipped inlier set.

``defined_not_activated`` remains a stable contract value for records produced
before activation or by external callers.  If full-objective optimization is
attempted but unavailable, the fitter retains its iterative scale-frozen affine
point estimate, records ``attempted_unavailable_fallback``, and leaves
coefficient covariance explicitly unavailable rather than pairing fallback
coefficients with covariance from a different optimum.

Dataset orchestration already preserves complete per-observational-channel
calibration records, so this provenance remains separate for channels that
share one physical wavelength.  Orchestration does not aggregate coefficient
covariance by physical wavelength and does not yet request predictive
propagation.

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

Unavailable coefficient covariance, including an attempted scale-dependent
optimization that falls back to the iterative affine estimate, produces an
explicit ``unavailable`` result with no numerical predictive uncertainty.
There is no silent fallback to measurement-only uncertainty.  Dataset
orchestration performs propagation only for an explicit request and otherwise
retains the ordinary execution-v1 behavior.  The stable orchestration
dispositions remain ``not_requested``, ``available``, ``skipped``, and
``unavailable``.

Dataset predictive-orchestration contract
-----------------------------------------

Dataset propagation is an explicit opt-in represented by
``InstrumentChannelCalibrationPredictiveUncertaintyRequest``.  Absence of a
request means fitted-coefficient propagation was not requested; coefficient
covariance availability alone never activates it.  A request selects
``marginal_variance`` or ``full_covariance`` and the execution callable
executes predictive propagation after completing the caller-authored
calibration plan.  Ordinary execution without a request keeps its existing
return type and serialization, omits the ``predictive_uncertainty`` key, and
continues to serialize ``fitted_coefficient_uncertainty_propagated`` as false.

The immutable orchestration result contract partitions original dataset source
rows among reference rows, non-reference observational-channel results, and
unaffected rows.  Reference-channel rows are explicitly ``not_applicable`` for
fitted calibration-coefficient uncertainty.  ``available`` channel results
carry the existing
``InstrumentChannelCalibrationPredictiveUncertainty`` record; ``unavailable``
results carry the existing explicit non-numeric unavailable record, with no
measurement-only fallback.  ``skipped`` and ``not_requested`` results carry no
numerical predictive values.  Missing results are represented structurally,
not with NaN-filled dataset arrays.

Marginal variances and derived predictive standard deviations remain
per-observational-channel blocks aligned to explicit original
``source_row_indices``.  Full covariance is also channel-local: its row and
column order follows that channel's source-row order, and no global dense
dataset covariance is emitted.  Rows calibrated with one shared pair of
``offset, scale`` coefficients may therefore be correlated within their block.
Distinct observational channels retain separate blocks, coefficients,
covariance, and dispositions even when they share one physical wavelength;
physical wavelength is never a covariance identity.  Cross-channel covariance
is not emitted and is not silently asserted to be zero.

The existing ``calibrated_flux_error`` array remains the ordinary transformed
measurement-error array.  Combined predictive standard deviation is exposed
only through available predictive blocks, so callers cannot confuse it with
the compatibility field.  A manually constructed calibration may participate
when it carries valid available coefficient covariance; fit provenance is not
required merely because the coefficients were supplied manually.

``TBD[instrument-channel-calibration]`` remains open
----------------------------------------------------

The low-level paired affine fit and application primitives do not complete the
instrument-channel calibration work.  The marker remains open because PGMUVI
still lacks:

* populated scientifically validated pairing-rule catalogue content
  plus automatic catalogue discovery and activation;
* integration into the normal light-curve fitting workflow; and
* validation across representative instruments, filters, and LPV sources.

No automatic calibration, channel merging, wavelength reassignment, or model
selection is performed.
