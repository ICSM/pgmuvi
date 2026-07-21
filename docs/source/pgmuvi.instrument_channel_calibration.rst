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

``TBD[instrument-channel-calibration]`` remains open
----------------------------------------------------

The low-level paired affine fit and application primitives do not complete the
instrument-channel calibration work.  The marker remains open because PGMUVI
still lacks:

* scientifically validated instrument-specific rules for choosing
  pairing methods and tolerances;
* dataset-level orchestration across shared-wavelength channel groups;
* propagation of fitted-coefficient uncertainty;
* integration into the normal light-curve fitting workflow; and
* validation across representative instruments, filters, and LPV sources.

No automatic calibration, channel merging, wavelength reassignment, or model
selection is performed.
