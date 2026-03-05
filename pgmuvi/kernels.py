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
        Initial period, by default 1.0.
    lengthscale : float, optional
        Initial periodic lengthscale, by default 0.5.
    decay : float, optional
        Initial RBF decay timescale, by default 10.0.
    outputscale : float, optional
        Initial output scale, by default 1.0.

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
        Initial lengthscale, by default 1.0.
    outputscale : float, optional
        Initial output scale, by default 1.0.

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
        Initial lengthscale, by default 1.0.
    outputscale : float, optional
        Initial output scale, by default 1.0.

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
