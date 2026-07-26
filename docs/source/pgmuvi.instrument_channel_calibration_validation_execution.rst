Instrument-channel calibration validation execution
===================================================

.. automodule:: pgmuvi.instrument_channel_calibration_validation_execution
   :members:
   :undoc-members:
   :show-inheritance:

Scientific boundary
-------------------

The maintained executor reads the prospectively frozen protocol and its
repository-contained anchor dataset, verifies the dataset digest, constructs
deterministic one-to-one pairs, creates the frozen contiguous temporal folds,
fits only the training portion of each fold, evaluates untouched held-out data,
and delegates the final disposition to the validation assessor.

Dataset references must resolve inside the repository.  The executor does not
search sibling directories, populate the scientifically validated pairing-rule
catalogue, or activate instrument-channel calibration in normal fitting.

Additional independent sources
------------------------------

The anchor-only API remains available.  For additional independent sources,
the maintained runner accepts a strict dataset manifest through
``--dataset-manifest``.  The manifest is a repository-relative JSON record
bound to the exact protocol identity
and canonical SHA-256 digest.  It contains only additional primary datasets;
the executor always evaluates the frozen anchor itself and then evaluates each
manifest dataset under the unchanged pairing, fitting, fold, normalization,
and acceptance configuration.

Each manifest dataset must provide a distinct astrophysical-source identity,
a unique dataset identity, a repository-relative CSV reference, and its exact
SHA-256 digest.  Derived, filtered, copied, resampled, or otherwise transformed
versions of the anchor are rejected as additional independent evidence.
Every CSV must contain ``time``, ``flux``, ``flux_error``, ``wavelength``, and
``band`` columns.  Eligible flux and uncertainty values must be finite and
strictly positive; the two exact KELT observational-channel identities must
remain distinct at the frozen physical wavelength.

A manifest has this shape; bracketed values are placeholders, not distributed
datasets or validation evidence:

.. code-block:: json

   {
     "schema_version": "pgmuvi-instrument-channel-calibration-validation-dataset-manifest-v1",
     "protocol_id": "kelt-osn-r3-pairing-validation-v1",
     "protocol_version": "1.0",
     "protocol_sha256": "[canonical frozen protocol SHA-256]",
     "datasets": [
       {
         "schema_version": "pgmuvi-instrument-channel-calibration-validation-dataset-v1",
         "dataset_id": "[independent-dataset-id]",
         "astrophysical_source_id": "[independent-source-id]",
         "dataset_reference": "examples/data/[independent-source].csv",
         "dataset_sha256": "[dataset SHA-256]",
         "derivation_parent_dataset_id": null,
         "is_derived": false
       }
     ]
   }

When ``--dataset-manifest`` is used, explicit non-default
``--result-output`` and ``--report-output`` paths are required so the committed
representative artifacts cannot be overwritten accidentally.  The resulting
assessment combines the anchor and all additional source results.  This
repository-bound manifest surface remains useful for a public reproducibility
smoke test and synthetic contract coverage.  It is not the final
maintainer-private multi-source decision workflow and does not read the private
Parquet catalogue.

Maintainer-private execution boundary
-------------------------------------

The next implementation layer will accept already prepared per-source channel
arrays in memory and execute the frozen maintainer-private multi-source
contract.  It will not own private catalogue discovery or Parquet ingestion.
The non-distributed private runner will identify sources containing the exact
protocol-approved candidate observational-channel pair, apply all pre-fit
eligibility gates, record the sorted eligible set and random seed, select the
first five entries from the seeded permutation, and invoke the public
data-agnostic engine.

The engine will perform source-balanced leave-one-source-out fitting and
held-out temporal-fold evaluation.  It will return detailed in-memory evidence
for the private report and a redacted public summary containing no raw source
identifiers or private paths.  Catalogue population remains a later explicit
step and is permitted only after a passing redacted decision.

Maintained execution status
---------------------------

The frozen KELT ``R3_0``/``R3_1`` protocol has been executed against the
repository-contained anchor dataset.  The maintained artifacts are:

* ``examples/validation/kelt_r3_pairing_validation_protocol_v1.json``;
* ``examples/validation/kelt_r3_pairing_validation_result_v1.json``; and
* ``examples/validation/kelt_r3_pairing_validation_report_v1.json``.

The committed report disposition is ``inconclusive``.  Its reasons are
``source_results_inconclusive`` and
``insufficient_independent_astrophysical_sources``.  The anchor contains one
low-dynamic-range fold.  This remains an accurate assessment of the **bundled
public evidence** and a useful public reproducibility smoke test.  It does not
preclude a final decision from the separate frozen maintainer-private
five-source protocol.

There is **no catalogue population** from the current execution.  It has not
established a scientifically validated pairing rule or authorized calibration
in ordinary light-curve fitting.  The private runner and public
leave-one-source-out engine remain to be implemented and executed.
