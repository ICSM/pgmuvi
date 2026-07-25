Instrument-channel calibration validation
=========================================

.. automodule:: pgmuvi.instrument_channel_calibration_validation
   :members:
   :undoc-members:
   :show-inheritance:

Prospective protocol boundary
-----------------------------

This module defines a **prospectively frozen** validation protocol and a
fail-closed result-assessment contract for one exact candidate
observational-channel pairing rule.  It does not execute the protocol, fit a
calibration, construct a scientifically validated pairing rule, or populate a
pairing-rule catalogue.

The committed protocol at
``examples/validation/kelt_r3_pairing_validation_protocol_v1.json`` binds the
KELT ``R3_0`` reference channel and ``R3_1`` target channel at the shared
physical wavelength to one fixed nearest-within-tolerance configuration.  Its
reference-channel choice, affine-fit configuration, temporal holdout method,
normalization, quantitative gates, and acceptance rationale are fixed before
the maintained execution step.

Each source result contains exact **fold-level** evidence rather than only
caller-supplied aggregates.  Every temporal-fold record preserves training and
holdout pair counts, the held-out reference-flux q05--q95 amplitude, median
reference error, normalized RMSE, signed normalized median bias, maximum time
separation, fitted offset and scale, and explicit failure reasons.  Source-level
median, worst-fold, bias, separation, and scale summaries are derived from
those records and are checked during strict deserialization.

A fold is scientifically usable only when its held-out q05--q95 amplitude spans
at least five median reference error bars and it retains the pre-registered
minimum number of holdout pairs.  Insufficient dynamic range, incomplete folds,
or protocol-configuration violations are ``inconclusive`` rather than evidence
that the calibration itself failed.  Only structurally complete evidence that
exceeds a pre-registered quantitative gate is ``failed``.

Source independence is defined by astrophysical source identity.  Full,
sampled, filtered, or otherwise derived datasets from the same source remain
one source: **derived datasets do not count as independent** astrophysical
sources.  The public ``10131+3049`` CSV is the anchor dataset, and its path and
SHA-256 digest are part of the strict protocol.

The exploratory calculations performed before this contract are diagnostic
only.  **prior exploratory runs** are explicitly ineligible as validation
evidence, cannot satisfy any acceptance gate, and cannot justify changing the
frozen tolerance or thresholds after execution begins.

A result record embeds no scientific-validation claim.  The deterministic
assessor returns ``passed``, ``failed``, or ``inconclusive``.  Missing protocol
identity, a protocol-digest mismatch, a missing anchor dataset, incomplete
source or fold evidence, duplicate primary datasets, or too few independent
astrophysical sources all fail closed as ``inconclusive``.

``TBD[instrument-channel-calibration]`` remains open.  A following PR must
execute the committed protocol on the anchor source and additional independent
astrophysical sources.  This contract
does not populate a pairing-rule catalogue and makes no claim that the KELT
candidate rule is scientifically validated.

Maintained execution
--------------------

The numerical execution path is implemented separately in
:mod:`pgmuvi.instrument_channel_calibration_validation_execution`.
It verifies that every dataset reference resolves within the repository and
that the exact dataset SHA-256 matches the frozen protocol before pairing or
fitting begins.  Fold construction, fitting, held-out metrics, and assessment
remain deterministic for fixed code and data.

Executing a protocol is not catalogue population.  The maintained runner
records evidence and the assessor derives ``passed``, ``failed``, or
``inconclusive``.  It does not activate calibration in ordinary fitting
workflows, and ``TBD[instrument-channel-calibration]`` remains open.
