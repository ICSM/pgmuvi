import torch as t
import gpytorch as gpt
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import (
    SpectralMixtureKernel as SMK,
    GridInterpolationKernel as GIK,
    AdditiveKernel,
    ConstantKernel,
    MaternKernel,
    PeriodicKernel,
    ProductKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.distributions import MultivariateNormal as MVN
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


# FIRST WE HAVE SOME Naive GPs
class SpectralMixtureGPModel(ExactGP):
    """A one-dimensional GP model using a spectral mixture kernel

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    This model assumes the mean is constant.

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps)
    train_y : Tensor
        The data for the dependent variable (typically flux)
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model
    num_mixtures : int
        Number of components in the Mixture Model. More mixtures gives more
        flexibility, but more hyperparameters and more complex inference

    Examples
    --------


    Notes
    ------



    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = SMK(num_mixtures=num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        # Now we alias the covariance kernel so that we can exploit the
        # same object properties in different classes with different kernel
        # structure
        # Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class SpectralMixtureLinearMeanGPModel(ExactGP):
    """A one-dimensional GP model using a spectral mixture kernel

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    This model assumes the mean is a linear function.

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps)
    train_y : Tensor
        The data for the dependent variable (typically flux)
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model
    num_mixtures : int
        Number of components in the Mixture Model. More mixtures gives more
        flexibility, but more hyperparameters and more complex inference

    Examples
    --------


    Notes
    ------



    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean()
        self.covar_module = SMK(num_mixtures=num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        # Now we alias the covariance kernel so that we can exploit the same
        # object properties in different classes with different kernel structure
        # Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class TwoDSpectralMixtureGPModel(ExactGP):
    """A two-dimensional GP model using a spectral mixture kernel

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    This model assumes the mean is constant. It supports datasets with two
    independent variables (e.g. time and wavelength).

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps and
        wavelengths)
    train_y : Tensor
        The data for the dependent variable (typically flux)
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model
    num_mixtures : int
        Number of components in the Mixture Model. More mixtures gives more
        flexibility, but more hyperparameters and more complex inference

    Examples
    --------


    Notes
    ------



    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = SMK(ard_num_dims=2, num_mixtures=num_mixtures)
        # Note: initialize_from_data can fail for 2D data due to constraint violations
        # Users should set hyperparameters manually or via set_hypers
        # self.covar_module.initialize_from_data(train_x, train_y)

        # Now we alias the covariance kernel so that we can exploit the
        # same object properties in different classes with different kernel
        # structure. Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class TwoDSpectralMixtureLinearMeanGPModel(ExactGP):
    """A two-dimensional GP model using a spectral mixture kernel

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    This model assumes the mean is a linear function.  It supports datasets
    with two independent variables (e.g. time and wavelength).

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps and
        wavelengths)
    train_y : Tensor
        The data for the dependent variable (typically flux)
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model
    num_mixtures : int
        Number of components in the Mixture Model. More mixtures gives more
        flexibility, but more hyperparameters and more complex inference

    Examples
    --------


    Notes
    ------



    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4):
        super().__init__(train_x, train_y, likelihood)
        # LinearMean requires input_size for 2D data
        input_size = train_x.shape[-1] if train_x.dim() > 1 else 1
        self.mean_module = LinearMean(input_size=input_size)
        self.covar_module = SMK(ard_num_dims=2, num_mixtures=num_mixtures)
        # Note: initialize_from_data can fail for 2D data due to constraint violations
        # Users should set hyperparameters manually or via set_hypers
        # self.covar_module.initialize_from_data(train_x, train_y)

        # Now we alias the covariance kernel so that we can exploit the same
        # object properties in different classes with different kernel
        # structure. Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


# Now we define some that use KISS-GP/SKI to try to accelerate inference
class SpectralMixtureKISSGPModel(ExactGP):
    """A one-dimensional GP model using a spectral mixture kernel

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    This model assumes the mean is constant. It uses the Kernel interpolation
    for scalable structured Gaussian processes (KISS-GP) approximation to
    enable scaling to much larger datasets. This means it becomes effective
    when your dataset exceeds ~10,000 entries; for smaller datasets, the
    overhead of interpolation is typically not worth the effort.

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps)
    train_y : Tensor
        The data for the dependent variable (typically flux)
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model
    num_mixtures : int
        Number of components in the Mixture Model. More mixtures gives more
        flexibility, but more hyperparameters and more complex inference
    grid_size : int
        The number of points to use in the kernel interpolation grid.

    Examples
    --------


    Notes
    ------



    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4, grid_size=2000):
        super().__init__(train_x, train_y, likelihood)
        if not grid_size:
            grid_size = gpt.utils.grid.choose_grid_size(train_x, 1.0)
            print(f"Using a grid of size {grid_size} for SKI")
        grid_bounds = [[t.min(train_x), t.max(train_x)]]
        self.mean_module = ConstantMean()
        self.covar_module = GIK(
            SMK(num_mixtures=num_mixtures),
            grid_size=grid_size,
            num_dims=1,
            grid_bounds=grid_bounds,
        )

        self.covar_module.base_kernel.initialize_from_data(train_x, train_y)

        self.sci_kernel = self.covar_module.base_kernel
        # self.covar_module.base_kernel.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class SpectralMixtureLinearMeanKISSGPModel(ExactGP):
    """A one-dimensional GP model using a spectral mixture kernel

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    This model assumes the mean is a linear function. It uses the Kernel
    interpolation for scalable structured Gaussian processes (KISS-GP)
    approximation to enable scaling to much larger datasets. This means it
    becomes effective when your dataset exceeds ~10,000 entries; for smaller
    datasets, the overhead of interpolation is typically not worth the effort.

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps)
    train_y : Tensor
        The data for the dependent variable (typically flux)
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model
    num_mixtures : int
        Number of components in the Mixture Model. More mixtures gives more
        flexibility, but more hyperparameters and more complex inference
    grid_size : int
        The number of points to use in the kernel interpolation grid.

    Examples
    --------


    Notes
    ------



    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4, grid_size=2000):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean()
        self.covar_module = GIK(SMK(num_mixtures=num_mixtures), grid_size=grid_size)
        self.covar_module.base_kernel.initialize_from_data(train_x, train_y)

        # Now we alias the covariance kernel so that we can exploit the
        # same object properties in different classes with different kernel
        # structure. Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class TwoDSpectralMixtureKISSGPModel(ExactGP):
    """A two-dimensional GP model using a spectral mixture kernel

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    This model assumes the mean is constant. It supports datasets with two
    independent variables (e.g. time and wavelength). It uses the Kernel
    interpolation for scalable structured Gaussian processes (KISS-GP)
    approximation to enable scaling to much larger datasets. This means it
    becomes effective when your dataset exceeds ~10,000 entries; for smaller
    datasets, the overhead of interpolation is typically not worth the effort.

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps)
    train_y : Tensor
        The data for the dependent variable (typically flux)
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model
    num_mixtures : int
        Number of components in the Mixture Model. More mixtures gives more
        flexibility, but more hyperparameters and more complex inference
    grid_size : (2x1) iterable of ints
        The number of points to use in the kernel interpolation grid, with one
        value per dimension.

    Examples
    --------


    Notes
    ------



    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4, grid_size=None):
        if grid_size is None:
            grid_size = [5000, 20]
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = GIK(
            SMK(ard_num_dims=2, num_mixtures=num_mixtures),
            num_dims=2,
            grid_size=grid_size,
        )
        # Note: initialize_from_data can fail for 2D data due to constraint violations
        # Users should set hyperparameters manually or via set_hypers
        # self.covar_module.base_kernel.initialize_from_data(train_x, train_y)

        # Now we alias the covariance kernel so that we can exploit the
        # same object properties in different classes with different kernel
        # structure. Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class TwoDSpectralMixtureLinearMeanKISSGPModel(ExactGP):
    """A two-dimensional GP model using a spectral mixture kernel

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    This model assumes the mean is a linear function. It supports datasets
    with two independent variables (e.g. time and wavelength). It uses the
    Kernel interpolation for scalable structured Gaussian processes (KISS-GP)
    approximation to enable scaling to much larger datasets. This means it
    becomes effective when your dataset exceeds ~10,000 entries; for smaller
    datasets, the overhead of interpolation is typically not worth the effort.

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps)
    train_y : Tensor
        The data for the dependent variable (typically flux)
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model
    num_mixtures : int
        Number of components in the Mixture Model. More mixtures gives more
        flexibility, but more hyperparameters and more complex inference
    grid_size : (2x1) iterable of ints
        The number of points to use in the kernel interpolation grid, with one
        value per dimension.

    Examples
    --------


    Notes
    ------



    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4, grid_size=None):
        if grid_size is None:
            grid_size = [5000, 20]
        super().__init__(train_x, train_y, likelihood)
        # LinearMean requires input_size for 2D data
        input_size = train_x.shape[-1] if train_x.dim() > 1 else 1
        self.mean_module = LinearMean(input_size=input_size)
        self.covar_module = GIK(
            SMK(ard_num_dims=2, num_mixtures=num_mixtures),
            num_dims=2,
            grid_size=grid_size,
        )
        # Note: initialize_from_data can fail for 2D data due to constraint violations
        # Users should set hyperparameters manually or via set_hypers
        # self.covar_module.base_kernel.initialize_from_data(train_x, train_y)

        # Now we alias the covariance kernel so that we can exploit the
        # same object properties in different classes with different kernel
        # structure. Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


# We can also implement sparse/variational GPs here
class SparseSpectralMixtureGPModel(ApproximateGP):
    """A one-dimensional GP model using a spectral mixture kernel

    A longer description goes here

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps)
    train_y : Tensor
        The data for the dependent variable (typically flux)
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model
    num_mixtures : int
        Number of components in the Mixture Model. More mixtures gives more
        flexibility, but more hyperparameters and more complex inference

    Examples
    --------


    Notes
    ------



    """

    def __init__(
        self, train_x, train_y, likelihood, num_mixtures=4, inducing_points=None
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = SMK(num_mixtures=num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        # Now we alias the covariance kernel so that we can exploit the
        # same object properties in different classes with different kernel
        # structure. Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


# TBD: Implement SVGP and VNN-GP versions of the above models


# ---------------------------------------------------------------------------
# Alternative physically-motivated GP models
#
# These models compose standard GPyTorch kernels to capture specific patterns
# common in cool evolved stars (pulsators, eclipsing binaries, AGN, etc.)
# with far fewer parameters than a full SpectralMixture kernel.
# ---------------------------------------------------------------------------


def _make_qp_kernel(period):
    """Return ``ScaleKernel(ProductKernel(PeriodicKernel, RBFKernel))``.

    The quasi-periodic kernel ``k_qp = k_periodic * k_rbf`` captures signals
    that are approximately periodic but slowly evolving.  It uses only
    GPyTorch built-in kernels (no custom code).

    Parameters
    ----------
    period : float
        Initial period_length for the PeriodicKernel.

    Returns
    -------
    gpytorch.kernels.ScaleKernel
    """
    periodic_k = PeriodicKernel()
    periodic_k.period_length = period
    rbf_k = RBFKernel()
    rbf_k.lengthscale = period * 5.0  # long-term decay timescale
    return ScaleKernel(ProductKernel(periodic_k, rbf_k))


class QuasiPeriodicGPModel(ExactGP):
    """1D GP model using a quasi-periodic kernel.

    Models signals that are approximately periodic but whose amplitude or
    phase evolves slowly over time.  The covariance is:

    .. code-block:: text

        k(x,x') = σ² · k_periodic(x,x') · k_rbf(x,x')

    where ``k_periodic`` is GPyTorch's :class:`~gpytorch.kernels.PeriodicKernel`
    and ``k_rbf`` provides the long-term amplitude decay.

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
    Recommended for pulsating stars and eclipsing binaries with stable
    periods. Uses far fewer parameters than a SpectralMixture kernel.

    Examples
    --------
    >>> import torch, gpytorch
    >>> from pgmuvi.gps import QuasiPeriodicGPModel
    >>> t = torch.linspace(0, 20, 100)
    >>> y = torch.sin(2 * torch.pi * t / 5)
    >>> lik = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = QuasiPeriodicGPModel(t, y, lik, period=5.0)
    """

    def __init__(self, train_x, train_y, likelihood, period=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        if period is None:
            span = float(train_x.max() - train_x.min())
            period = span / 2.0

        self.covar_module = _make_qp_kernel(period)
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class MaternGPModel(ExactGP):
    """1D GP model using a Matérn kernel for stochastic variability.

    Suitable for red-noise processes such as accretion disk variability in
    AGN and other stochastic astronomical sources.

    Parameters
    ----------
    train_x : torch.Tensor
        Time stamps (1D tensor).
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    nu : float, optional
        Matérn smoothness (0.5 = Ornstein-Uhlenbeck, 1.5, or 2.5).
        Default 1.5.
    lengthscale : float, optional
        Initial lengthscale. Defaults to one-quarter of the data span.

    Notes
    -----
    Recommended for accretion disk variability, AGN, and stochastic
    red-noise characteristics.

    Examples
    --------
    >>> import torch, gpytorch
    >>> from pgmuvi.gps import MaternGPModel
    >>> t = torch.linspace(0, 20, 100)
    >>> y = torch.randn(100)
    >>> lik = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = MaternGPModel(t, y, lik, nu=0.5)
    """

    def __init__(self, train_x, train_y, likelihood, nu=1.5, lengthscale=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        if lengthscale is None:
            span = float(train_x.max() - train_x.min())
            lengthscale = span / 4.0

        matern_k = MaternKernel(nu=nu)
        matern_k.lengthscale = lengthscale
        self.covar_module = ScaleKernel(matern_k)
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class PeriodicPlusStochasticGPModel(ExactGP):
    """1D GP model combining quasi-periodic and stochastic components.

    Decomposes the signal as:
    ``k(x,x') = k_quasi_periodic(x,x') + k_rbf(x,x')``

    Parameters
    ----------
    train_x : torch.Tensor
        Time stamps (1D tensor).
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    period : float, optional
        Initial period guess. Defaults to half the data span.

    Notes
    -----
    Recommended for spotted stars with rotation plus stochastic variations.

    Examples
    --------
    >>> import torch, gpytorch
    >>> from pgmuvi.gps import PeriodicPlusStochasticGPModel
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

        qp_part = _make_qp_kernel(period)
        rbf_stochastic = ScaleKernel(RBFKernel())
        rbf_stochastic.base_kernel.lengthscale = period

        self.covar_module = AdditiveKernel(qp_part, rbf_stochastic)
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class LinearMeanQuasiPeriodicGPModel(ExactGP):
    """1D GP model with a linear trend mean and quasi-periodic kernel.

    Like :class:`QuasiPeriodicGPModel` but assumes a linear mean function,
    suited for sources with a long-term trend plus periodic variations.

    Parameters
    ----------
    train_x : torch.Tensor
        Time stamps (1D tensor).
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    period : float, optional
        Initial period guess. Defaults to half the data span.
    """

    def __init__(self, train_x, train_y, likelihood, period=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean(input_size=1)

        if period is None:
            span = float(train_x.max() - train_x.min())
            period = span / 2.0

        self.covar_module = _make_qp_kernel(period)
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class SeparableGPModel(ExactGP):
    """2D GP model with a separable (product) kernel for multiwavelength data.

    The covariance factorizes as:

    .. code-block:: text

        k(t, l, t', l') = k_time(t, t') * k_wavelength(l, l')

    This is implemented natively via GPyTorch's
    :class:`~gpytorch.kernels.ProductKernel` with ``active_dims`` set on
    each component, with no custom kernel code required.

    Parameters
    ----------
    train_x : torch.Tensor
        Input of shape ``(n, 2)`` — column 0 is time, column 1 is wavelength.
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    time_kernel : gpytorch.kernels.Kernel, optional
        Kernel for the temporal dimension (``active_dims`` will be set to
        ``[0]`` automatically). Defaults to a scaled Matérn-1.5.
    wavelength_kernel : gpytorch.kernels.Kernel, optional
        Kernel for the wavelength dimension (``active_dims`` will be set to
        ``[1]`` automatically). Defaults to a scaled RBF.

    Notes
    -----
    For common special cases see :class:`AchromaticGPModel` and
    :class:`WavelengthDependentGPModel`.
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

        # Restrict each sub-kernel to its own dimension via active_dims.
        time_kernel.register_buffer(
            "active_dims", t.tensor([0], dtype=t.long)
        )
        wavelength_kernel.register_buffer(
            "active_dims", t.tensor([1], dtype=t.long)
        )

        # ProductKernel with active_dims gives the separable 2D covariance
        # without any custom forward() code.
        self.covar_module = time_kernel * wavelength_kernel
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class AchromaticGPModel(ExactGP):
    """2D GP model for achromatic (wavelength-independent) variability.

    Uses a separable kernel where the wavelength component is a
    :class:`~gpytorch.kernels.ConstantKernel`, enforcing perfect correlation
    across wavelengths so that all bands share the same temporal variability
    pattern (e.g. eclipses, transits, geometric occultations).

    Parameters
    ----------
    train_x : torch.Tensor
        Input of shape ``(n, 2)`` — column 0 is time, column 1 is wavelength.
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    time_kernel_type : str, optional
        Type of time kernel: ``'quasi_periodic'``, ``'matern'`` (default),
        or ``'rbf'``.
    period : float, optional
        Initial period for the quasi-periodic option. Defaults to half the
        time span.

    Notes
    -----
    Best suited for eclipses, transits, and other geometric effects that
    affect all wavelengths equally.

    Examples
    --------
    >>> import torch, gpytorch
    >>> from pgmuvi.gps import AchromaticGPModel
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
            time_kernel = _make_qp_kernel(period)
        elif time_kernel_type == "matern":
            time_kernel = ScaleKernel(MaternKernel(nu=1.5))
        elif time_kernel_type == "rbf":
            time_kernel = ScaleKernel(RBFKernel())
        else:
            raise ValueError(
                f"Unknown time_kernel_type '{time_kernel_type}'. "
                "Choose from 'quasi_periodic', 'matern', 'rbf'."
            )

        time_kernel.register_buffer(
            "active_dims", t.tensor([0], dtype=t.long)
        )
        wl_kernel = ConstantKernel()
        wl_kernel.register_buffer(
            "active_dims", t.tensor([1], dtype=t.long)
        )

        self.covar_module = time_kernel * wl_kernel
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class WavelengthDependentGPModel(ExactGP):
    """2D GP model with smooth wavelength-dependent variability.

    Uses a separable kernel with an RBF wavelength component, capturing
    smooth correlations across wavelengths (e.g. temperature-driven flux
    changes that are stronger at certain wavelengths).

    Parameters
    ----------
    train_x : torch.Tensor
        Input of shape ``(n, 2)`` — column 0 is time, column 1 is wavelength.
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    time_kernel_type : str, optional
        Type of time kernel: ``'quasi_periodic'``, ``'matern'`` (default),
        or ``'rbf'``.
    period : float, optional
        Initial period for the quasi-periodic option. Defaults to half the
        time span.
    wavelength_lengthscale : float, optional
        Initial wavelength correlation length. Defaults to half the wavelength
        span.

    Notes
    -----
    Best suited for temperature-driven variability and spot models where the
    variability amplitude changes smoothly with wavelength.

    Examples
    --------
    >>> import torch, gpytorch
    >>> from pgmuvi.gps import WavelengthDependentGPModel
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
            time_kernel = _make_qp_kernel(period)
        elif time_kernel_type == "matern":
            time_kernel = ScaleKernel(MaternKernel(nu=1.5))
        elif time_kernel_type == "rbf":
            time_kernel = ScaleKernel(RBFKernel())
        else:
            raise ValueError(
                f"Unknown time_kernel_type '{time_kernel_type}'. "
                "Choose from 'quasi_periodic', 'matern', 'rbf'."
            )

        time_kernel.register_buffer(
            "active_dims", t.tensor([0], dtype=t.long)
        )
        wl_kernel = ScaleKernel(RBFKernel())
        wl_kernel.base_kernel.lengthscale = wavelength_lengthscale
        wl_kernel.register_buffer(
            "active_dims", t.tensor([1], dtype=t.long)
        )

        self.covar_module = time_kernel * wl_kernel
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)
