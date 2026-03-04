"""Custom kernel implementations for pgmuvi.

This module provides alternative GP kernels beyond the default spectral mixture
kernel, including quasi-periodic, separable (for 2D multiwavelength data), and
achromatic kernels.
"""

import math
import torch
import gpytorch
from gpytorch.kernels import Kernel, RBFKernel, MaternKernel


class QuasiPeriodicKernel(Kernel):
    """Quasi-periodic kernel for slowly-evolving periodic signals.

    Product of a periodic part and a squared-exponential (RBF) decay:

    .. code-block:: text

        k(x,x') = exp(-sin^2(pi*|x-x'|/T) / ls^2) * exp(-|x-x'|^2 / (2*D^2))

    where ``T`` is the period, ``ls`` the periodic lengthscale and ``D`` the
    long-term decay timescale.

    Parameters
    ----------
    period_constraint : gpytorch.constraints.Constraint, optional
        Constraint on the period parameter. Defaults to Positive().
    lengthscale_constraint : gpytorch.constraints.Constraint, optional
        Constraint on the periodic lengthscale. Defaults to Positive().
    decay_constraint : gpytorch.constraints.Constraint, optional
        Constraint on the long-term decay. Defaults to Positive().

    Notes
    -----
    Best suited for pulsating stars, eclipsing binaries with stable periods,
    and other sources with approximately periodic variability.

    References
    ----------
    MacKay, D. J. C. (1998). Introduction to Gaussian processes.
    Rasmussen and Williams (2006). Gaussian Processes for Machine Learning.
    """

    has_lengthscale = False

    def __init__(
        self,
        period_constraint=None,
        lengthscale_constraint=None,
        decay_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.register_parameter(
            name="raw_period",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        self.register_parameter(
            name="raw_decay",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )

        if period_constraint is None:
            period_constraint = gpytorch.constraints.Positive()
        if lengthscale_constraint is None:
            lengthscale_constraint = gpytorch.constraints.Positive()
        if decay_constraint is None:
            decay_constraint = gpytorch.constraints.Positive()

        self.register_constraint("raw_period", period_constraint)
        self.register_constraint("raw_lengthscale", lengthscale_constraint)
        self.register_constraint("raw_decay", decay_constraint)

    @property
    def period(self):
        return self.raw_period_constraint.transform(self.raw_period)

    @period.setter
    def period(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period)
        self.initialize(
            raw_period=self.raw_period_constraint.inverse_transform(value)
        )

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(
            raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value)
        )

    @property
    def decay(self):
        return self.raw_decay_constraint.transform(self.raw_decay)

    @decay.setter
    def decay(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_decay)
        self.initialize(
            raw_decay=self.raw_decay_constraint.inverse_transform(value)
        )

    def forward(self, x1, x2, diag=False, **params):
        # Compute absolute differences for periodic part
        x1_ = x1.unsqueeze(-2)  # (..., n, 1, d)
        x2_ = x2.unsqueeze(-3)  # (..., 1, m, d)
        diff = x1_ - x2_  # (..., n, m, d)

        if x1.shape[-1] > 1:
            diff = diff[..., 0:1]  # Use first dimension (time)

        dist = diff.squeeze(-1)  # (..., n, m)

        # Periodic part: exp(-sin^2(pi * dist / period) / lengthscale^2)
        period = self.period.squeeze(-1).squeeze(-1)  # scalar or batch
        lengthscale = self.lengthscale.squeeze(-1).squeeze(-1)
        decay = self.decay.squeeze(-1).squeeze(-1)

        sin_term = torch.sin(math.pi * dist / period)
        periodic_part = torch.exp(-sin_term.pow(2) / lengthscale.pow(2))

        # RBF decay part: exp(-dist^2 / (2 * decay^2))
        rbf_part = torch.exp(-dist.pow(2) / (2.0 * decay.pow(2)))

        result = periodic_part * rbf_part

        if diag:
            return result.diagonal(dim1=-1, dim2=-2)
        return result


class SeparableKernel(Kernel):
    """Separable kernel for 2D multiwavelength data.

    Factorizes as ``k(t, l, t', l') = k_time(t, t') * k_wavelength(l, l')``.
    This assumes temporal and wavelength variability are independent.

    Parameters
    ----------
    time_kernel : gpytorch.kernels.Kernel
        Kernel for the temporal dimension (first column of input).
    wavelength_kernel : gpytorch.kernels.Kernel
        Kernel for the wavelength dimension (second column of input).

    Notes
    -----
    Reduces the number of parameters compared to a full 2D spectral mixture
    kernel by exploiting the separability assumption.
    """

    def __init__(self, time_kernel, wavelength_kernel, **kwargs):
        super().__init__(**kwargs)
        self.time_kernel = time_kernel
        self.wavelength_kernel = wavelength_kernel

    def forward(self, x1, x2, diag=False, **params):
        # x1, x2 shape: (..., n, 2) — columns are [time, wavelength]
        t1 = x1[..., 0:1]
        t2 = x2[..., 0:1]
        w1 = x1[..., 1:2]
        w2 = x2[..., 1:2]

        time_cov = self.time_kernel(t1, t2)
        wavelength_cov = self.wavelength_kernel(w1, w2)

        # Elementwise product of the two covariance matrices
        time_mat = time_cov.to_dense()
        wavelength_mat = wavelength_cov.to_dense()
        result = time_mat * wavelength_mat

        if diag:
            return result.diagonal(dim1=-1, dim2=-2)
        return result


class AchromaticKernel(SeparableKernel):
    """Separable kernel for achromatic (wavelength-independent) variability.

    Uses a white-noise kernel in the wavelength dimension, meaning all
    wavelengths share the same temporal covariance structure (e.g. eclipses,
    transits).

    Parameters
    ----------
    time_kernel : gpytorch.kernels.Kernel
        Kernel for the temporal dimension.

    Notes
    -----
    Best suited for eclipses, transits, and other geometric effects that affect
    all wavelengths equally.
    """

    def __init__(self, time_kernel, **kwargs):
        wavelength_kernel = gpytorch.kernels.ConstantKernel()
        super().__init__(
            time_kernel=time_kernel,
            wavelength_kernel=wavelength_kernel,
            **kwargs,
        )


def make_quasi_periodic_kernel(
    period=1.0,
    lengthscale=0.5,
    decay=10.0,
    outputscale=1.0,
):
    """Create a scaled quasi-periodic kernel with initial hyperparameters.

    Parameters
    ----------
    period : float, optional
        Initial period, by default 1.0.
    lengthscale : float, optional
        Initial periodic lengthscale, by default 0.5.
    decay : float, optional
        Initial long-term decay timescale, by default 10.0.
    outputscale : float, optional
        Initial output scale (amplitude), by default 1.0.

    Returns
    -------
    gpytorch.kernels.ScaleKernel
        Scaled quasi-periodic kernel.
    """
    base = QuasiPeriodicKernel()
    base.period = period
    base.lengthscale = lengthscale
    base.decay = decay

    kernel = gpytorch.kernels.ScaleKernel(base)
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
    kernel = gpytorch.kernels.ScaleKernel(base)
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
    kernel = gpytorch.kernels.ScaleKernel(base)
    kernel.outputscale = outputscale
    return kernel
