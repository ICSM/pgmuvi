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
record provides a strict JSON-safe provenance contract for pairs already
selected by the caller.  It records row indices, paired times, time units,
reuse permissions, interpolation provenance, and derived time-separation
summaries.  It validates structural consistency only: it does not determine
whether a time separation, interpolation rule, or reuse policy is
scientifically appropriate.

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

PGMUVI does not construct time pairs from asynchronous light curves.  The
pairing record does not perform nearest-neighbour matching, interpolation,
cadence reconciliation, or reference-channel selection.  PGMUVI also does not
select between additive, multiplicative, affine, or other calibration
families.  Pair construction and the decision to use this explicit affine
model remain the caller's scientific responsibility.

Applying a fitted calibration maps fluxes onto the reference-channel scale.
Flux uncertainties are multiplied by the absolute fitted scale.  The current
application function does not propagate uncertainty in the fitted offset or
scale themselves.

``TBD[instrument-channel-calibration]`` remains open
----------------------------------------------------

The low-level paired affine fit and application primitives do not complete the
instrument-channel calibration work.  The marker remains open because PGMUVI
still lacks:

* scientifically validated rules and implementations for constructing
  calibration pairs;
* dataset-level orchestration across shared-wavelength channel groups;
* propagation of fitted-coefficient uncertainty;
* integration into the normal light-curve fitting workflow; and
* validation across representative instruments, filters, and LPV sources.

No automatic calibration, channel merging, wavelength reassignment, or model
selection is performed.
