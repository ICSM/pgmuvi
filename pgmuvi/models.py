"""Alternative GP model classes for pgmuvi.

.. deprecated::
    This module is a compatibility shim. The models have been moved to
    :mod:`pgmuvi.gps` where they belong alongside the other GP model classes.
    Import them from there directly::

        from pgmuvi.gps import (
            QuasiPeriodicGPModel,
            MaternGPModel,
            PeriodicPlusStochasticGPModel,
            SeparableGPModel,
            AchromaticGPModel,
            WavelengthDependentGPModel,
            LinearMeanQuasiPeriodicGPModel,
        )
"""

import warnings

warnings.warn(
    "pgmuvi.models is deprecated and will be removed in a future release; "
    "import the model classes from pgmuvi.gps instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .gps import (  # noqa: F401, E402
    AchromaticGPModel,
    LinearMeanQuasiPeriodicGPModel,
    MaternGPModel,
    PeriodicPlusStochasticGPModel,
    QuasiPeriodicGPModel,
    SeparableGPModel,
    WavelengthDependentGPModel,
)
