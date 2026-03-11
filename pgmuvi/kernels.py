"""Kernel helpers for pgmuvi.

This module previously contained custom GPyTorch kernel subclasses
(``QuasiPeriodicKernel``, ``SeparableKernel``, ``AchromaticKernel``).
These have been replaced by compositions of standard GPyTorch kernels
in :mod:`pgmuvi.gps`, which requires no custom ``forward()`` code and
directly benefits from all GPyTorch optimisations.

The helper factory functions ``make_quasi_periodic_kernel``,
``make_matern_kernel``, and ``make_rbf_kernel`` are retained for
convenience.
"""

from gpytorch.kernels import (
    MaternKernel,
    PeriodicKernel,
    ProductKernel,
    RBFKernel,
    ScaleKernel,
)

__all__ = [
    "make_matern_kernel",
    "make_quasi_periodic_kernel",
    "make_rbf_kernel",
]


def make_quasi_periodic_kernel(
    period=1.0, lengthscale=0.5, decay=10.0, outputscale=1.0
):
    """Create a scaled quasi-periodic kernel.

    Returns ``ScaleKernel(ProductKernel(PeriodicKernel, RBFKernel))``.

    The quasi-periodic kernel is the elementwise product of a
    :class:`~gpytorch.kernels.PeriodicKernel` (oscillatory component)
    and an :class:`~gpytorch.kernels.RBFKernel` (long-term decay).
    No custom kernel subclass is required.

    Parameters
    ----------
    period : float, optional
        Initial period, in the same units as the input time axis.
        By default 1.0.
    lengthscale : float, optional
        Periodic lengthscale of the :class:`~gpytorch.kernels.PeriodicKernel`
        (same units as ``period``).  Small values produce sharp, narrow peaks
        within each cycle; large values produce broad, smooth sinusoidal
        variations.  By default 0.5.
    decay : float, optional
        Lengthscale of the :class:`~gpytorch.kernels.RBFKernel` envelope
        (same units as the time axis).  Controls how quickly the periodic
        pattern decorrelates over time: roughly the timescale over which the
        amplitude or phase of the oscillation can change.  By default 10.0.
    outputscale : float, optional
        Overall amplitude scaling (approximately the standard deviation of the
        GP output).  By default 1.0.

    Returns
    -------
    gpytorch.kernels.ScaleKernel
        ``ScaleKernel(ProductKernel(PeriodicKernel(), RBFKernel()))``.
    """
    periodic_k = PeriodicKernel()
    periodic_k.period_length = period
    periodic_k.lengthscale = lengthscale
    rbf_k = RBFKernel()
    rbf_k.lengthscale = decay
    kernel = ScaleKernel(ProductKernel(periodic_k, rbf_k))
    kernel.outputscale = outputscale
    return kernel


def make_matern_kernel(nu=1.5, lengthscale=1.0, outputscale=1.0):
    """Create a scaled Matérn kernel.

    Parameters
    ----------
    nu : float, optional
        Smoothness parameter (0.5, 1.5, or 2.5), by default 1.5.
    lengthscale : float, optional
        Correlation length in the same units as the time axis.  Observations
        separated by much less than ``lengthscale`` are strongly correlated;
        observations separated by much more than ``lengthscale`` are
        essentially uncorrelated.  By default 1.0.
    outputscale : float, optional
        Overall amplitude scaling (approximately the standard deviation of the
        GP output).  By default 1.0.

    Returns
    -------
    gpytorch.kernels.ScaleKernel
        Scaled Matérn kernel.
    """
    base = MaternKernel(nu=nu)
    base.lengthscale = lengthscale
    kernel = ScaleKernel(base)
    kernel.outputscale = outputscale
    return kernel


def make_rbf_kernel(lengthscale=1.0, outputscale=1.0):
    """Create a scaled RBF (squared-exponential) kernel.

    Parameters
    ----------
    lengthscale : float, optional
        Correlation length in the same units as the time axis.  Observations
        separated by much less than ``lengthscale`` are strongly correlated;
        observations separated by much more than ``lengthscale`` are
        essentially uncorrelated.  By default 1.0.
    outputscale : float, optional
        Overall amplitude scaling (approximately the standard deviation of the
        GP output).  By default 1.0.

    Returns
    -------
    gpytorch.kernels.ScaleKernel
        Scaled RBF kernel.
    """
    base = RBFKernel()
    base.lengthscale = lengthscale
    kernel = ScaleKernel(base)
    kernel.outputscale = outputscale
    return kernel
