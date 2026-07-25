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
