Instrument-channel calibration contract
=======================================

.. automodule:: pgmuvi.instrument_channel_calibration
   :members:
   :undoc-members:
   :show-inheritance:

``TBD[instrument-channel-calibration]`` remains open
----------------------------------------------------

This module does **not** implement an instrumental calibration model.  It
provides a deterministic assessment of whether multiple observational channels
share one physical wavelength and therefore require a future calibration
model.

The assessment preserves observational-channel identities, reports the shared
physical-wavelength groups, records that no correction was applied, and is
JSON-safe.  It does not inspect flux differences as evidence for an offset or
scale and does not alter wavelengths or fluxes.

The public callables
:func:`~pgmuvi.instrument_channel_calibration.fit_instrument_channel_calibration`
and
:func:`~pgmuvi.instrument_channel_calibration.apply_instrument_channel_calibration`
raise :class:`NotImplementedError`.  This explicit failure is part of the
contract: PGMUVI must not silently infer, approximate, or apply an
instrument-channel correction before a scientifically validated implementation
exists.
