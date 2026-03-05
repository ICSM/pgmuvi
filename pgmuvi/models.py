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

from .gps import (  # noqa: F401
    AchromaticGPModel,
    LinearMeanQuasiPeriodicGPModel,
    MaternGPModel,
    PeriodicPlusStochasticGPModel,
    QuasiPeriodicGPModel,
    SeparableGPModel,
    WavelengthDependentGPModel,
)
