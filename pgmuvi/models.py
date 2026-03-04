"""Alternative GP model classes for pgmuvi.

This module provides model classes using simpler, physically-motivated kernels
as alternatives to the spectral mixture kernel in gps.py.

Available models:

- ``QuasiPeriodicGPModel`` — 1D model for periodic signals with slow evolution
- ``MaternGPModel`` — 1D stochastic model for red-noise variability
- ``PeriodicPlusStochasticGPModel`` — 1D model combining periodic and stochastic
  components
- ``SeparableGPModel`` — 2D separable model for multiwavelength data
- ``AchromaticGPModel`` — 2D achromatic (wavelength-independent) model
- ``WavelengthDependentGPModel`` — 2D model with smooth wavelength correlation
"""

import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, AdditiveKernel
from gpytorch.distributions import MultivariateNormal as MVN
from gpytorch.models import ExactGP

from .kernels import QuasiPeriodicKernel, SeparableKernel


class QuasiPeriodicGPModel(ExactGP):
    """1D GP model using a quasi-periodic kernel.

    Models signals that are approximately periodic but whose amplitude or
    phase evolves slowly over time.

    Parameters
    ----------
    train_x : torch.Tensor
        Time stamps (1D tensor).
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    period : float, optional
        Initial period guess. If None, defaults to half the data span.

    Notes
    -----
    Recommended for pulsating stars and eclipsing binaries with stable periods.
    Uses far fewer parameters than a SpectralMixture kernel (4 vs 12 for
    ``num_mixtures=4``).

    Examples
    --------
    >>> import torch
    >>> import gpytorch
    >>> from pgmuvi.models import QuasiPeriodicGPModel
    >>> t = torch.linspace(0, 20, 100)
    >>> y = torch.sin(2 * torch.pi * t / 5)
    >>> lik = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = QuasiPeriodicGPModel(t, y, lik, period=5.0)
    """

    def __init__(self, train_x, train_y, likelihood, period=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        qp_kernel = QuasiPeriodicKernel()
        self.covar_module = ScaleKernel(qp_kernel)
        self.sci_kernel = self.covar_module

        if period is None:
            span = float(train_x.max() - train_x.min())
            period = span / 2.0

        self.covar_module.base_kernel.period = period
        self.covar_module.base_kernel.lengthscale = 0.5
        self.covar_module.base_kernel.decay = period * 5.0

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class MaternGPModel(ExactGP):
    """1D GP model using a Matérn kernel for stochastic variability.

    Suitable for red-noise processes such as accretion disk variability in AGN
    and other stochastic astronomical sources.

    Parameters
    ----------
    train_x : torch.Tensor
        Time stamps (1D tensor).
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    nu : float, optional
        Matérn smoothness parameter (0.5, 1.5, or 2.5), by default 1.5.
        - 0.5: Ornstein-Uhlenbeck (damped random walk)
        - 1.5: Once-differentiable
        - 2.5: Twice-differentiable
    lengthscale : float, optional
        Initial lengthscale. If None, defaults to one-quarter of the data span.

    Notes
    -----
    Recommended for accretion disk variability, AGN, and other sources with
    stochastic red-noise characteristics.

    Examples
    --------
    >>> import torch
    >>> import gpytorch
    >>> from pgmuvi.models import MaternGPModel
    >>> t = torch.linspace(0, 20, 100)
    >>> y = torch.randn(100)
    >>> lik = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = MaternGPModel(t, y, lik, nu=0.5)
    """

    def __init__(self, train_x, train_y, likelihood, nu=1.5, lengthscale=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        matern_kernel = MaternKernel(nu=nu)
        self.covar_module = ScaleKernel(matern_kernel)
        self.sci_kernel = self.covar_module

        if lengthscale is None:
            span = float(train_x.max() - train_x.min())
            lengthscale = span / 4.0

        self.covar_module.base_kernel.lengthscale = lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class PeriodicPlusStochasticGPModel(ExactGP):
    """1D GP model combining a quasi-periodic kernel and an RBF kernel.

    Decomposes the signal into a periodic component (quasi-periodic kernel)
    and a stochastic/red-noise component (RBF kernel):
    ``k(x,x') = k_periodic(x,x') + k_rbf(x,x')``

    Parameters
    ----------
    train_x : torch.Tensor
        Time stamps (1D tensor).
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    period : float, optional
        Initial period guess. If None, defaults to half the data span.

    Notes
    -----
    Recommended for spotted stars with rotation plus stochastic variations,
    active galactic nuclei with quasi-periodic oscillations plus red noise, etc.

    Examples
    --------
    >>> import torch
    >>> import gpytorch
    >>> from pgmuvi.models import PeriodicPlusStochasticGPModel
    >>> t = torch.linspace(0, 20, 100)
    >>> y = torch.sin(2 * torch.pi * t / 5) + 0.3 * torch.randn(100)
    >>> lik = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = PeriodicPlusStochasticGPModel(t, y, lik, period=5.0)
    """

    def __init__(self, train_x, train_y, likelihood, period=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        if period is None:
            span = float(train_x.max() - train_x.min())
            period = span / 2.0

        qp_kernel = ScaleKernel(QuasiPeriodicKernel())
        qp_kernel.base_kernel.period = period
        qp_kernel.base_kernel.lengthscale = 0.5
        qp_kernel.base_kernel.decay = period * 5.0

        rbf_kernel = ScaleKernel(RBFKernel())
        rbf_kernel.base_kernel.lengthscale = period

        self.covar_module = AdditiveKernel(qp_kernel, rbf_kernel)
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class SeparableGPModel(ExactGP):
    """2D GP model with a separable kernel for multiwavelength data.

    The covariance factorizes as:
    ``k(t, l, t', l') = k_time(t, t') * k_wavelength(l, l')``

    This reduces the number of parameters compared to a full 2D spectral
    mixture kernel while still modelling both temporal and wavelength
    correlations.

    Parameters
    ----------
    train_x : torch.Tensor
        Input data of shape ``(n, 2)`` where column 0 is time and column 1 is
        wavelength.
    train_y : torch.Tensor
        Observed values (1D tensor of length n).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    time_kernel : gpytorch.kernels.Kernel, optional
        Kernel for the temporal dimension. Defaults to a scaled Matérn(1.5).
    wavelength_kernel : gpytorch.kernels.Kernel, optional
        Kernel for the wavelength dimension. Defaults to a scaled RBF.

    Notes
    -----
    Use this as a base class or when you want full control over both kernels.
    For common cases see ``AchromaticGPModel`` and ``WavelengthDependentGPModel``.
    """

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        time_kernel=None,
        wavelength_kernel=None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        if time_kernel is None:
            time_kernel = ScaleKernel(MaternKernel(nu=1.5))
        if wavelength_kernel is None:
            wavelength_kernel = ScaleKernel(RBFKernel())

        self.covar_module = SeparableKernel(time_kernel, wavelength_kernel)
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class AchromaticGPModel(ExactGP):
    """2D GP model for achromatic (wavelength-independent) variability.

    Uses a separable kernel where the wavelength kernel is a white-noise term,
    meaning all wavelengths share the same temporal variability pattern.

    Parameters
    ----------
    train_x : torch.Tensor
        Input data of shape ``(n, 2)`` where column 0 is time and column 1 is
        wavelength.
    train_y : torch.Tensor
        Observed values (1D tensor of length n).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    time_kernel_type : str, optional
        Type of time kernel to use. Options: ``'quasi_periodic'``, ``'matern'``,
        ``'rbf'``. Default is ``'matern'``.
    period : float, optional
        Initial period for the quasi-periodic kernel (only used when
        ``time_kernel_type='quasi_periodic'``). Defaults to half the data span.

    Notes
    -----
    Best suited for eclipses, transits, and geometric effects that affect
    all wavelengths equally.

    Examples
    --------
    >>> import torch
    >>> import gpytorch
    >>> from pgmuvi.models import AchromaticGPModel
    >>> t = torch.linspace(0, 10, 50)
    >>> wl = torch.ones(50) * 550.0
    >>> x = torch.stack([t, wl], dim=1)
    >>> y = torch.sin(2 * torch.pi * t / 3)
    >>> lik = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = AchromaticGPModel(
    ...     x, y, lik, time_kernel_type='quasi_periodic', period=3.0
    ... )
    """

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        time_kernel_type="matern",
        period=None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        if period is None:
            span = float(train_x[:, 0].max() - train_x[:, 0].min())
            period = span / 2.0

        if time_kernel_type == "quasi_periodic":
            qp = QuasiPeriodicKernel()
            qp.period = period
            qp.lengthscale = 0.5
            qp.decay = period * 5.0
            time_kernel = ScaleKernel(qp)
        elif time_kernel_type == "matern":
            time_kernel = ScaleKernel(MaternKernel(nu=1.5))
        elif time_kernel_type == "rbf":
            time_kernel = ScaleKernel(RBFKernel())
        else:
            raise ValueError(
                f"Unknown time_kernel_type '{time_kernel_type}'. "
                "Choose from 'quasi_periodic', 'matern', 'rbf'."
            )

        wavelength_kernel = gpytorch.kernels.ConstantKernel()
        self.covar_module = SeparableKernel(time_kernel, wavelength_kernel)
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class WavelengthDependentGPModel(ExactGP):
    """2D GP model with smooth wavelength-dependent variability.

    Uses a separable kernel with an RBF wavelength kernel, capturing smooth
    correlations across wavelengths (e.g. temperature variations producing
    correlated flux changes at nearby wavelengths).

    Parameters
    ----------
    train_x : torch.Tensor
        Input data of shape ``(n, 2)`` where column 0 is time and column 1 is
        wavelength.
    train_y : torch.Tensor
        Observed values (1D tensor of length n).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    time_kernel_type : str, optional
        Type of time kernel to use. Options: ``'quasi_periodic'``, ``'matern'``,
        ``'rbf'``. Default is ``'matern'``.
    period : float, optional
        Initial period for the quasi-periodic kernel. Defaults to half the time
        span.
    wavelength_lengthscale : float, optional
        Initial wavelength correlation length. Defaults to half the wavelength
        span.

    Notes
    -----
    Best suited for temperature-driven variability, spot models with
    wavelength-dependent contrast, or any effect where the variability
    amplitude changes smoothly with wavelength.

    Examples
    --------
    >>> import torch
    >>> import gpytorch
    >>> from pgmuvi.models import WavelengthDependentGPModel
    >>> t = torch.linspace(0, 10, 50)
    >>> wl = torch.linspace(400, 900, 50)
    >>> x = torch.stack([t, wl], dim=1)
    >>> y = torch.sin(2 * torch.pi * t / 3) * (wl / wl.mean())
    >>> lik = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = WavelengthDependentGPModel(x, y, lik)
    """

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        time_kernel_type="matern",
        period=None,
        wavelength_lengthscale=None,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        if period is None:
            span = float(train_x[:, 0].max() - train_x[:, 0].min())
            period = span / 2.0
        if wavelength_lengthscale is None:
            wl_span = float(train_x[:, 1].max() - train_x[:, 1].min())
            wavelength_lengthscale = max(wl_span / 2.0, 1.0)

        if time_kernel_type == "quasi_periodic":
            qp = QuasiPeriodicKernel()
            qp.period = period
            qp.lengthscale = 0.5
            qp.decay = period * 5.0
            time_kernel = ScaleKernel(qp)
        elif time_kernel_type == "matern":
            time_kernel = ScaleKernel(MaternKernel(nu=1.5))
        elif time_kernel_type == "rbf":
            time_kernel = ScaleKernel(RBFKernel())
        else:
            raise ValueError(
                f"Unknown time_kernel_type '{time_kernel_type}'. "
                "Choose from 'quasi_periodic', 'matern', 'rbf'."
            )

        wl_kernel = ScaleKernel(RBFKernel())
        wl_kernel.base_kernel.lengthscale = wavelength_lengthscale
        self.covar_module = SeparableKernel(time_kernel, wl_kernel)
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class LinearMeanQuasiPeriodicGPModel(ExactGP):
    """1D GP model with linear mean and quasi-periodic kernel.

    Like ``QuasiPeriodicGPModel`` but assumes a linear trend as the mean
    function, suitable for sources with long-term trends plus periodic
    variations.

    Parameters
    ----------
    train_x : torch.Tensor
        Time stamps (1D tensor).
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    period : float, optional
        Initial period guess. If None, defaults to half the data span.
    """

    def __init__(self, train_x, train_y, likelihood, period=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean(input_size=1)

        qp_kernel = QuasiPeriodicKernel()
        self.covar_module = ScaleKernel(qp_kernel)
        self.sci_kernel = self.covar_module

        if period is None:
            span = float(train_x.max() - train_x.min())
            period = span / 2.0

        self.covar_module.base_kernel.period = period
        self.covar_module.base_kernel.lengthscale = 0.5
        self.covar_module.base_kernel.decay = period * 5.0

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)
