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

Maintained execution status
---------------------------

The frozen KELT ``R3_0``/``R3_1`` protocol has been executed against the
repository-contained anchor dataset.  The maintained artifacts are:

* ``examples/validation/kelt_r3_pairing_validation_protocol_v1.json``;
* ``examples/validation/kelt_r3_pairing_validation_result_v1.json``; and
* ``examples/validation/kelt_r3_pairing_validation_report_v1.json``.

The committed report disposition is ``inconclusive``.  Its reasons are
``source_results_inconclusive`` and
``insufficient_independent_astrophysical_sources``.  The execution performed
no catalogue population and does not establish a scientifically validated
pairing rule or authorize calibration in ordinary light-curve fitting.
