"""
pgmuvi: a package for infering multiwavelength variaiblity of astronomical
sources using Gaussian processes in python

"""

# __version__ = "0.0.1"
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pgmuvi")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = [
    "constraints",
    "gps",
    "initialization",
    "lightcurve",
    "priors",
    "spectral_mixture_ard",
    "spectral_mixture_ard_diagnostics",
    "synthetic",
    "trainers",
    "wavelength_conclusions",
    "wavelength_diagnostics",
    "wavelength_estimation",
    "wavelength_hypotheses",
    "wavelength_results",
    "wavelength_status",
    "wavelength_validation",
    "wavelength_validation_real_lpv",
    "wavelength_validation_recovery",
    "wavelength_validation_robustness",
    "wavelength_validation_robustness_calibration",
    "wavelength_validation_synthetic",
]
