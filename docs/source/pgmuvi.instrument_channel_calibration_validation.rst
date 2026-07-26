Instrument-channel calibration validation
=========================================

.. automodule:: pgmuvi.instrument_channel_calibration_validation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pgmuvi.instrument_channel_calibration_multisource_validation
   :members:
   :undoc-members:
   :show-inheritance:

Prospective protocol boundary
-----------------------------

This module defines a **prospectively frozen** validation protocol and a
fail-closed result-assessment contract for one exact candidate
observational-channel pairing rule.  It does not execute the protocol, fit a
calibration, or construct a scientifically validated pairing rule.  This module
does not populate a pairing-rule catalogue.

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

``TBD[instrument-channel-calibration]`` remains open.  The committed
anchor execution is retained as a **public reproducibility smoke test** of
pairing, fold construction, fitting, and assessment against the one real
dataset distributed with PGMUVI.  Its ``inconclusive`` disposition records
``source_results_inconclusive`` and
``insufficient_independent_astrophysical_sources``.  The candidate rule is
not claimed as scientifically validated by this public smoke test.  These reasons
describe the limits of the bundled public evidence; they are not the final
scientific decision procedure for the candidate rule.

Maintainer-private multi-source decision contract
-------------------------------------------------

The final decision is governed by
``examples/validation/kelt_r3_maintainer_multisource_validation_protocol_v1.json``.
It binds the same exact KELT ``R3_0``/``R3_1`` protocol-approved candidate
observational-channel pair, but requires five eligible independent
astrophysical sources selected from maintainer-owned data before calibration
outcomes are inspected.

Eligibility requires both exact observational channels at the frozen physical
wavelength, finite strictly positive flux and uncertainty, at least 100
deterministic matched pairs, five informative temporal folds with at least 20
pairs each, and a q05--q95 reference-flux amplitude spanning at least five
median reference error bars.  Derived copies of a source are excluded.

Eligible source identifiers are sorted, permuted with a recorded random seed,
and the first five are selected without replacement.  The scientific test is
source-balanced leave-one-source-out validation: four sources train one common
affine rule and the unseen fifth source is evaluated without refitting.  This
is repeated until every selected source has been held out once.  Equal matched
pair counts from each training source prevent one densely sampled light curve
from dominating the fit.

The public package defines the data-agnostic contract and redacted evidence
schema.  A **private Parquet** ingestion and selection runner is maintainer
infrastructure, is not distributed, and does not create public package Parquet
support.  Raw source identifiers, private light curves, and the detailed
source-level report remain private.  A public redacted summary may contain
only protocol and input digests, the selection seed, hashed selected-source
identities, aggregate held-out metrics, coefficient-stability summaries, and
the final disposition.

Maintained execution
--------------------

The numerical execution path is implemented separately in
:mod:`pgmuvi.instrument_channel_calibration_validation_execution`.
It verifies that every dataset reference resolves within the repository and
that each exact dataset SHA-256 matches its immutable identity before pairing
or fitting begins.  The anchor-only runner remains available, and a strict
protocol-bound dataset manifest can supply additional independent primary
sources without altering the frozen protocol.  Fold construction, fitting,
held-out metrics, and assessment remain deterministic for fixed code and data.

Executing a protocol is not catalogue population.  The maintained runner
records evidence and the assessor derives ``passed``, ``failed``, or
``inconclusive``.  It does not activate calibration in ordinary fitting
workflows, and ``TBD[instrument-channel-calibration]`` remains open.
