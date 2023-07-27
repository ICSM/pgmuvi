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

__all__=["gps", "trainers", "lightcurve"]
