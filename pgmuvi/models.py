"""Compatibility imports for GP model classes.

This module is a deprecated compatibility shim.  The model classes have
moved to :mod:`pgmuvi.gps`, where they live alongside the other GP model
classes.  New code should import them from :mod:`pgmuvi.gps` directly::

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
