import math

import torch as t
import gpytorch as gpt
from gpytorch.means import Mean, ConstantMean, LinearMean
from gpytorch.kernels import (
    SpectralMixtureKernel as SMK,
    GridInterpolationKernel as GIK,
    AdditiveKernel,
    ConstantKernel,
    MaternKernel,
    PeriodicKernel,
    ProductKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
from gpytorch.distributions import MultivariateNormal as MVN
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

# Module-level constant: log(1.7) used as default initial value for
# DustMean.log_alpha, corresponding to a typical interstellar dust-extinction
# power-law index of alpha ≈ 1.7.
_LOG_1_7 = math.log(1.7)


class PowerLawMean(gpt.means.Mean):
    """Mean function with power-law wavelength dependence for 2D GP models.

    Computes the mean flux as a power law in the second input dimension
    (wavelength):

        m(t, λ) = offset + weight * λ^exponent

    This is more realistic than a linear mean function for AGB stars and
    other variable stars where the variability amplitude and mean flux can
    vary steeply with wavelength. A negative ``exponent`` gives higher flux
    at shorter (optical) wavelengths; a positive ``exponent`` gives higher
    flux at longer (infrared) wavelengths.

    Parameters
    ----------
    batch_shape : torch.Size, optional
        Batch shape for the mean function parameters.

    Attributes
    ----------
    offset : torch.nn.Parameter
        Constant offset term.
    weight : torch.nn.Parameter
        Amplitude scaling factor (unconstrained; sign determines whether
        flux increases or decreases with wavelength).
    exponent : torch.nn.Parameter
        Power-law exponent for the wavelength dependence. Defaults to
        ``-2.0``, which gives a steep decline from optical to infrared
        (i.e., high optical amplitude).

    Notes
    -----
    The ``exponent`` parameter is unconstrained and can be learned to any
    real value during optimisation. Initialising it to a physically
    motivated value (e.g. ``-1.7`` for a typical dust-extinction law, or
    ``2.0`` for a Rayleigh-Jeans tail) can help convergence.
    """

    def __init__(self, batch_shape=None):
        super().__init__()
        if batch_shape is None:
            batch_shape = t.Size()
        self.register_parameter(
            "offset", t.nn.Parameter(t.zeros(*batch_shape, 1))
        )
        self.register_parameter(
            "weight", t.nn.Parameter(t.ones(*batch_shape, 1))
        )
        # Default exponent of -2 gives a steep optical-to-IR decline
        self.register_parameter(
            "exponent", t.nn.Parameter(t.full((*batch_shape, 1), -2.0))
        )

    def forward(self, x):
        wavelength = x[..., 1]  # second column is wavelength
        return (
            self.offset.squeeze(-1)
            + self.weight.squeeze(-1) * wavelength.pow(self.exponent.squeeze(-1))
        )


class DustMean(gpt.means.Mean):
    """Mean function with dust-extinction wavelength dependence for 2D GP
    models.

    Computes the mean flux following a dust-attenuation law:

        m(t, λ) = amplitude * exp(-tau * λ^(-alpha)) + offset

    where ``tau`` is the dust optical depth (a positive constant controlling
    the overall obscuration), and ``alpha`` is the power-law index of the
    wavelength-dependent extinction (``alpha > 0`` gives stronger extinction
    at shorter wavelengths, as observed for interstellar dust).

    This form is physically motivated for dust-obscured AGB stars, where
    optical fluxes can be two or three orders of magnitude fainter than
    infrared fluxes due to circumstellar dust shells.

    Parameters
    ----------
    batch_shape : torch.Size, optional
        Batch shape for the mean function parameters.

    Attributes
    ----------
    offset : torch.nn.Parameter
        Constant offset (background) term.
    log_amplitude : torch.nn.Parameter
        Log of the amplitude; the amplitude itself is ``exp(log_amplitude)``,
        ensuring it is always positive.
    log_tau : torch.nn.Parameter
        Log of the dust optical depth ``tau``; the optical depth itself is
        ``exp(log_tau)``, ensuring it is always positive.
    log_alpha : torch.nn.Parameter
        Log of the extinction power-law index ``alpha``; the index itself is
        ``exp(log_alpha)``, ensuring it is always positive.  Defaults to
        ``log(1.7)`` ≈ 0.53, corresponding to a typical interstellar
        dust-extinction law.

    Notes
    -----
    The wavelength axis is expected to be the second column of the input
    tensor ``x``.  It should be strictly positive (as is the case for
    physical wavelengths in microns).  Wavelength values below ``1e-6``
    (microns) are clamped to ``1e-6`` to avoid numerical overflow in
    ``λ^(-alpha)``; in practice any physically meaningful wavelength
    (optical ~0.4 μm or longer) is well above this threshold.  Applying a
    wavelength transform (e.g. ``MinMax``) before fitting is still
    recommended to keep wavelength values in a numerically convenient range.
    """

    def __init__(self, batch_shape=None):
        super().__init__()
        if batch_shape is None:
            batch_shape = t.Size()
        self.register_parameter(
            "offset", t.nn.Parameter(t.zeros(*batch_shape, 1))
        )
        self.register_parameter(
            "log_amplitude", t.nn.Parameter(t.zeros(*batch_shape, 1))
        )
        self.register_parameter(
            "log_tau", t.nn.Parameter(t.zeros(*batch_shape, 1))
        )
        # Default alpha ≈ 1.7, consistent with a typical dust-extinction law
        self.register_parameter(
            "log_alpha",
            t.nn.Parameter(t.full((*batch_shape, 1), _LOG_1_7)),
        )

    def forward(self, x):
        # Clamp wavelength to avoid overflow in λ^(-alpha): any physical
        # wavelength in microns is far above 1e-6, so this guard is only
        # triggered for pathological inputs.
        wavelength = x[..., 1].clamp(min=1e-6)
        amplitude = self.log_amplitude.squeeze(-1).exp()
        tau = self.log_tau.squeeze(-1).exp()
        alpha = self.log_alpha.squeeze(-1).exp()
        extinction = tau * wavelength.pow(-alpha)
        return self.offset.squeeze(-1) + amplitude * (-extinction).exp()


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


class TwoDSpectralMixturePowerLawMeanGPModel(ExactGP):
    """A two-dimensional GP model with a power-law wavelength mean function.

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    The mean function follows a power law in the wavelength dimension:

        m(t, λ) = offset + weight * λ^exponent

    This is more physically realistic than a linear mean for AGB stars and
    other variable stars where the variability amplitude (case a) varies
    steeply with wavelength.  It supports datasets with two independent
    variables (e.g. time and wavelength).

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps and
        wavelengths), shape (N, 2).
    train_y : Tensor
        The data for the dependent variable (typically flux).
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model.
    num_mixtures : int
        Number of components in the Mixture Model.

    Examples
    --------


    Notes
    -----
    See :class:`PowerLawMean` for details of the mean-function
    parameterisation.

    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = PowerLawMean()
        self.covar_module = SMK(ard_num_dims=2, num_mixtures=num_mixtures)

        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class TwoDSpectralMixturePowerLawMeanKISSGPModel(ExactGP):
    """A two-dimensional KISS-GP model with a power-law wavelength mean
    function.

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    The mean function follows a power law in the wavelength dimension:

        m(t, λ) = offset + weight * λ^exponent

    This is more physically realistic than a linear mean for AGB stars and
    other variable stars where the variability amplitude (case a) varies
    steeply with wavelength.  It supports datasets with two independent
    variables (e.g. time and wavelength) and uses the Kernel Interpolation
    for Scalable Structured Gaussian Processes (KISS-GP) approximation to
    scale to larger datasets.

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps and
        wavelengths), shape (N, 2).
    train_y : Tensor
        The data for the dependent variable (typically flux).
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model.
    num_mixtures : int
        Number of components in the Mixture Model.
    grid_size : list of int, optional
        The number of grid points per dimension for the KISS-GP
        approximation.  Defaults to ``[5000, 20]``.

    Examples
    --------


    Notes
    -----
    See :class:`PowerLawMean` for details of the mean-function
    parameterisation.

    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4, grid_size=None):
        if grid_size is None:
            grid_size = [5000, 20]
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = PowerLawMean()
        self.covar_module = GIK(
            SMK(ard_num_dims=2, num_mixtures=num_mixtures),
            num_dims=2,
            grid_size=grid_size,
        )

        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class TwoDSpectralMixtureDustMeanGPModel(ExactGP):
    """A two-dimensional GP model with a dust-extinction wavelength mean
    function.

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    The mean function follows a dust-attenuation law in the wavelength
    dimension:

        m(t, λ) = amplitude * exp(-tau * λ^(-alpha)) + offset

    where ``tau > 0`` is the dust optical depth and ``alpha > 0`` is the
    power-law index of the wavelength-dependent extinction.  This is
    physically motivated for dust-obscured AGB stars where optical (short
    wavelength) fluxes can be two to three orders of magnitude fainter than
    infrared fluxes (case b).  It supports datasets with two independent
    variables (e.g. time and wavelength).

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps and
        wavelengths), shape (N, 2).
    train_y : Tensor
        The data for the dependent variable (typically flux).
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model.
    num_mixtures : int
        Number of components in the Mixture Model.

    Examples
    --------


    Notes
    -----
    See :class:`DustMean` for details of the mean-function parameterisation.

    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = DustMean()
        self.covar_module = SMK(ard_num_dims=2, num_mixtures=num_mixtures)

        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class TwoDSpectralMixtureDustMeanKISSGPModel(ExactGP):
    """A two-dimensional KISS-GP model with a dust-extinction wavelength mean
    function.

    A Gaussian Process which uses a Spectral Mixture Kernel to model the Power
    Spectral Density of the covariance matrix as a Gaussian Mixture Model.
    The mean function follows a dust-attenuation law in the wavelength
    dimension:

        m(t, λ) = amplitude * exp(-tau * λ^(-alpha)) + offset

    where ``tau > 0`` is the dust optical depth and ``alpha > 0`` is the
    power-law index of the wavelength-dependent extinction.  This is
    physically motivated for dust-obscured AGB stars where optical (short
    wavelength) fluxes can be two to three orders of magnitude fainter than
    infrared fluxes (case b).  It supports datasets with two independent
    variables (e.g. time and wavelength) and uses the Kernel Interpolation
    for Scalable Structured Gaussian Processes (KISS-GP) approximation to
    scale to larger datasets.

    Parameters
    ----------
    train_x : Tensor
        The data for the independent variable (typically timestamps and
        wavelengths), shape (N, 2).
    train_y : Tensor
        The data for the dependent variable (typically flux).
    likelihood : a Likelihood object or subclass
        The likelihood that will be used to evaluate the model.
    num_mixtures : int
        Number of components in the Mixture Model.
    grid_size : list of int, optional
        The number of grid points per dimension for the KISS-GP
        approximation.  Defaults to ``[5000, 20]``.

    Examples
    --------


    Notes
    -----
    See :class:`DustMean` for details of the mean-function parameterisation.

    """

    def __init__(self, train_x, train_y, likelihood, num_mixtures=4, grid_size=None):
        if grid_size is None:
            grid_size = [5000, 20]
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = DustMean()
        self.covar_module = GIK(
            SMK(ard_num_dims=2, num_mixtures=num_mixtures),
            num_dims=2,
            grid_size=grid_size,
        )

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


def _build_time_kernel(time_kernel_type, period, num_mixtures=4, **kwargs):
    """Build a time kernel from a string name or a Kernel instance.

    Parameters
    ----------
    time_kernel_type : str or gpytorch.kernels.Kernel
        Kernel type or a pre-built Kernel instance.  Supported string values:

        - ``'quasi_periodic'``: ``ScaleKernel(PeriodicKernel * RBFKernel)``
        - ``'matern'`` (default): ``ScaleKernel(MaternKernel(nu=1.5))``
        - ``'rbf'``: ``ScaleKernel(RBFKernel())``
        - ``'spectral_mixture'`` / ``'sm'``:
          ``SpectralMixtureKernel(num_mixtures, ard_num_dims=1)``

        If a :class:`gpytorch.kernels.Kernel` instance is passed directly, it
        is used as-is (a warning is emitted to remind the user to check that
        the kernel is appropriate for the time dimension).
    period : float
        Initial period for the quasi-periodic option. Ignored for other types.
    num_mixtures : int, optional
        Number of mixture components for ``'spectral_mixture'``.  Default 4.

    Returns
    -------
    gpytorch.kernels.Kernel
    """
    if isinstance(time_kernel_type, gpt.kernels.Kernel):
        import warnings
        warnings.warn(
            "A Kernel instance was supplied for time_kernel_type. "
            "Ensure this kernel is appropriate for the time (first) dimension "
            "before fitting.",
            UserWarning,
            stacklevel=3,
        )
        return time_kernel_type
    if time_kernel_type == "quasi_periodic":
        return _make_qp_kernel(period)
    if time_kernel_type == "matern":
        return ScaleKernel(MaternKernel(nu=1.5))
    if time_kernel_type == "rbf":
        return ScaleKernel(RBFKernel())
    if time_kernel_type in ("spectral_mixture", "sm"):
        # SMK incorporates its own output scale; no wrapping ScaleKernel needed.
        return SMK(num_mixtures=num_mixtures, ard_num_dims=1)
    raise ValueError(
        f"Unknown time_kernel_type '{time_kernel_type}'. "
        "Choose from 'quasi_periodic', 'matern', 'rbf', "
        "'spectral_mixture'/'sm', or supply a gpytorch.kernels.Kernel instance."
    )


def _build_wavelength_kernel(wavelength_kernel_type, wavelength_lengthscale):
    """Build a wavelength kernel from a string name or a Kernel instance.

    Parameters
    ----------
    wavelength_kernel_type : str or gpytorch.kernels.Kernel
        Kernel type or a pre-built Kernel instance.  Supported string values:

        - ``'rbf'`` (default): ``ScaleKernel(RBFKernel())``
        - ``'matern'``: ``ScaleKernel(MaternKernel(nu=1.5))``
        - ``'rational_quadratic'`` / ``'rq'``:
          ``ScaleKernel(RQKernel())``

        If a :class:`gpytorch.kernels.Kernel` instance is passed, it is used
        as-is (a warning is emitted to remind the user to check suitability).
    wavelength_lengthscale : float
        Initial lengthscale applied to ``base_kernel.lengthscale`` for
        string-constructed kernels.

    Returns
    -------
    gpytorch.kernels.Kernel
    """
    if isinstance(wavelength_kernel_type, gpt.kernels.Kernel):
        import warnings
        warnings.warn(
            "A Kernel instance was supplied for wavelength_kernel_type. "
            "Ensure this kernel is appropriate for the wavelength (second) "
            "dimension before fitting.",
            UserWarning,
            stacklevel=3,
        )
        return wavelength_kernel_type
    if wavelength_kernel_type == "rbf":
        k = ScaleKernel(RBFKernel())
        k.base_kernel.lengthscale = wavelength_lengthscale
        return k
    if wavelength_kernel_type == "matern":
        k = ScaleKernel(MaternKernel(nu=1.5))
        k.base_kernel.lengthscale = wavelength_lengthscale
        return k
    if wavelength_kernel_type in ("rational_quadratic", "rq"):
        k = ScaleKernel(RQKernel())
        k.base_kernel.lengthscale = wavelength_lengthscale
        return k
    raise ValueError(
        f"Unknown wavelength_kernel_type '{wavelength_kernel_type}'. "
        "Choose from 'rbf', 'matern', 'rational_quadratic'/'rq', "
        "or supply a gpytorch.kernels.Kernel instance."
    )


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

    def __init__(self, train_x, train_y, likelihood, period=None, **kwargs):
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

    def __init__(
        self, train_x, train_y, likelihood, nu=1.5, lengthscale=None, **kwargs
    ):
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

    def __init__(self, train_x, train_y, likelihood, period=None, **kwargs):
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

    def __init__(self, train_x, train_y, likelihood, period=None, **kwargs):
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
        mean_module=None,
        **kwargs
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean() if mean_module is None else mean_module

        if time_kernel is None:
            time_kernel = ScaleKernel(MaternKernel(nu=1.5))
        if wavelength_kernel is None:
            wavelength_kernel = ScaleKernel(RBFKernel())

        # Restrict each sub-kernel to its own dimension via active_dims.
        time_kernel.register_buffer(
            "active_dims", t.tensor([1], dtype=t.long)
        )
        wavelength_kernel.register_buffer(
            "active_dims", t.tensor([0], dtype=t.long)
        )

        # ProductKernel with active_dims gives the separable 2D covariance
        # without any custom forward() code.
        self.covar_module = time_kernel * wavelength_kernel
        self.sci_kernel = self.covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


class AchromaticGPModel(SeparableGPModel):
    """2D GP model for achromatic (wavelength-independent) variability.

    Subclass of :class:`SeparableGPModel` that uses a
    :class:`~gpytorch.kernels.ConstantKernel` for the wavelength dimension,
    enforcing perfect correlation across wavelengths so that all bands share
    the same temporal variability pattern (e.g. eclipses, transits, geometric
    occultations).

    Parameters
    ----------
    train_x : torch.Tensor
        Input of shape ``(n, 2)`` — column 0 is time, column 1 is wavelength.
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    time_kernel_type : str or gpytorch.kernels.Kernel, optional
        Type of time kernel.  Accepted string values:

        - ``'matern'`` (default): ``ScaleKernel(MaternKernel(nu=1.5))``
        - ``'quasi_periodic'``: ``ScaleKernel(PeriodicKernel * RBFKernel)``
        - ``'rbf'``: ``ScaleKernel(RBFKernel())``
        - ``'spectral_mixture'`` / ``'sm'``:
          ``SpectralMixtureKernel(num_mixtures, ard_num_dims=1)``

        Alternatively, supply any :class:`gpytorch.kernels.Kernel` instance
        directly.
    period : float, optional
        Initial period for the quasi-periodic option. Defaults to half the
        time span.
    num_mixtures : int, optional
        Number of mixture components when ``time_kernel_type='spectral_mixture'``.
        Default 4.

    Notes
    -----
    Best suited for eclipses, transits, and other geometric effects that
    affect all wavelengths equally.

    Examples
    --------
    >>> import torch, gpytorch
    >>> from pgmuvi.gps import AchromaticGPModel
    >>> t_data = torch.linspace(0, 10, 50)
    >>> wl = torch.ones(50) * 550.0
    >>> x = torch.stack([t_data, wl], dim=1)
    >>> y = torch.sin(2 * torch.pi * t_data / 3)
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
        num_mixtures=4,
        mean_module=None,
        **kwargs
    ):
        if period is None:
            span = float(train_x[:, 0].max() - train_x[:, 0].min())
            period = span / 2.0

        time_kernel = _build_time_kernel(time_kernel_type, period, num_mixtures)
        wl_kernel = ConstantKernel()

        super().__init__(
            train_x,
            train_y,
            likelihood,
            time_kernel=time_kernel,
            wavelength_kernel=wl_kernel,
        )

class CustomLinearConstantMean(Mean):
    """ Custom mean function that is linear in wavelength and constant in time.

    This is useful for modelling stars whose mean flux changes with wavelength but not
    time, as most stars do (because they are approximately blackbodies with a fixed
    temperature).
    """

    def __init__(self):
        super().__init__()
        self.register_parameter(
            name="wavelength_slope",
            parameter=t.nn.Parameter(t.tensor(0.0)),
        )
        self.register_parameter(
            name="bias",
            parameter=t.nn.Parameter(t.tensor(0.0)),
        )

    def forward(self, x):
        return self.bias + self.wavelength_slope * x[:, 1]


class CustomQuadConstantMean(Mean):
    """ Custom mean function that is quadratic in wavelength and constant in time.

    This is useful for modelling stars whose mean flux changes with wavelength but not
    time, as most stars do (because they are approximately blackbodies with a fixed
    temperature).
    """

    def __init__(self):
        super().__init__()
        self.register_parameter(
            name="weights",
            parameter=t.nn.Parameter(t.tensor([0.0, 0.0])),
        )
        self.register_parameter(
            name="bias",
            parameter=t.nn.Parameter(t.tensor(0.0)),
        )

    def forward(self, x):
        x_wl = x[:, 1]
        powers = t.arange(1, len(self.weights) + 1,
                          device=self.weights.device,
                          dtype=self.weights.dtype)
        x_powers = x_wl.unsqueeze(-1) ** powers
        return self.bias + t.sum(self.weights * x_powers, dim=-1)


class WavelengthDependentGPModel(SeparableGPModel):
    """2D GP model with smooth wavelength-dependent variability.

    Subclass of :class:`SeparableGPModel` that uses a smooth kernel for the
    wavelength dimension, capturing correlated flux changes across nearby
    wavelengths (e.g. temperature-driven variability, spot models with
    wavelength-dependent contrast).

    Parameters
    ----------
    train_x : torch.Tensor
        Input of shape ``(n, 2)`` — column 0 is time, column 1 is wavelength.
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    time_kernel_type : str or gpytorch.kernels.Kernel, optional
        Type of time kernel.  Accepted string values:

        - ``'matern'`` (default): ``ScaleKernel(MaternKernel(nu=1.5))``
        - ``'quasi_periodic'``: ``ScaleKernel(PeriodicKernel * RBFKernel)``
        - ``'rbf'``: ``ScaleKernel(RBFKernel())``
        - ``'spectral_mixture'`` / ``'sm'``:
          ``SpectralMixtureKernel(num_mixtures, ard_num_dims=1)``

        Alternatively, supply any :class:`gpytorch.kernels.Kernel` instance.
    wavelength_kernel_type : str or gpytorch.kernels.Kernel, optional
        Type of wavelength kernel.  Accepted string values:

        - ``'rbf'`` (default): ``ScaleKernel(RBFKernel())``
        - ``'matern'``: ``ScaleKernel(MaternKernel(nu=1.5))``
        - ``'rational_quadratic'`` / ``'rq'``: ``ScaleKernel(RQKernel())``

        Alternatively, supply any :class:`gpytorch.kernels.Kernel` instance
        directly (a warning will be emitted to prompt a suitability check).
    period : float, optional
        Initial period for the quasi-periodic option. Defaults to half the
        time span.
    wavelength_lengthscale : float, optional
        Initial wavelength correlation length. Defaults to half the wavelength
        span (minimum 1.0).
    num_mixtures : int, optional
        Number of mixture components when ``time_kernel_type='spectral_mixture'``.
        Default 4.
    mean_module : str or gpytorch.means.Mean, optional
        Mean function for the GP.  Accepted string values:

        - ``None`` / ``'quad'`` / ``'quadratic'``: :class:`CustomQuadConstantMean`
          (default — quadratic in wavelength, constant in time).
        - ``'linear'`` / ``'linear_mean'``: :class:`CustomLinearConstantMean`.
        - ``'constant'`` / ``'constant_mean'``: :class:`~gpytorch.means.ConstantMean`.
        - ``'dust'`` / ``'dust_mean'``: :class:`DustMean` — physically motivated
          dust-attenuation law.
        - ``'power_law'`` / ``'power_law_mean'``: :class:`PowerLawMean` — power-law
          wavelength dependence.

        Alternatively, supply any :class:`gpytorch.means.Mean` instance directly.

    Notes
    -----
    Best suited for temperature-driven variability and spot models where the
    variability amplitude changes smoothly with wavelength.

    To model dust-obscured sources (e.g. AGB stars) combine ``mean_module='dust'``
    with this model's separable Matérn/RBF kernel structure — this is equivalent
    to :class:`DustMeanGPModel`.  For a simple power-law wavelength dependence use
    ``mean_module='power_law'`` (equivalent to :class:`PowerLawMeanGPModel`).

    Examples
    --------
    >>> import torch, gpytorch
    >>> from pgmuvi.gps import WavelengthDependentGPModel
    >>> t_data = torch.linspace(0, 10, 50)
    >>> wl = torch.linspace(400, 900, 50)
    >>> x = torch.stack([t_data, wl], dim=1)
    >>> y = torch.sin(2 * torch.pi * t_data / 3) * (wl / wl.mean())
    >>> lik = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = WavelengthDependentGPModel(x, y, lik)
    """

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        time_kernel_type="matern",
        wavelength_kernel_type="rbf",
        period=None,
        wavelength_lengthscale=None,
        num_mixtures=4,
        mean_module=None,
        **kwargs
    ):
        if period is None:
            span = float(train_x[:, 0].max() - train_x[:, 0].min())
            period = span / 2.0
        if wavelength_lengthscale is None:
            wl_span = float(train_x[:, 1].max() - train_x[:, 1].min())
            wavelength_lengthscale = max(wl_span / 2.0, 1.0)

        if mean_module is None or mean_module in ('quad', 'quadratic', 'quad_constant'):
            mean_module = CustomQuadConstantMean()
        elif mean_module in ('linear', 'linear_mean'):
            mean_module = CustomLinearConstantMean()
            # mean_module = LinearMean(input_size=2)
        elif mean_module in ('constant', 'constant_mean'):
            mean_module = ConstantMean()
        elif mean_module in ('dust', 'dust_mean'):
            mean_module = DustMean()
        elif mean_module in ('power_law', 'power_law_mean'):
            mean_module = PowerLawMean()
        elif isinstance(mean_module, Mean):
            mean_module = mean_module
        else:
            raise ValueError(
                "mean_module must be a gpytorch.means.Mean instance or None."
            )


        time_kernel = _build_time_kernel(time_kernel_type, period, num_mixtures)
        wl_kernel = _build_wavelength_kernel(
            wavelength_kernel_type, wavelength_lengthscale
        )

        super().__init__(
            train_x,
            train_y,
            likelihood,
            time_kernel=time_kernel,
            wavelength_kernel=wl_kernel,
            mean_module=mean_module,
            **kwargs
        )


class DustMeanGPModel(WavelengthDependentGPModel):
    """2D GP model combining a dust-extinction mean with separable simple kernels.

    A convenience subclass of :class:`WavelengthDependentGPModel` that
    replaces the default quadratic mean function with :class:`DustMean`:

    .. code-block:: text

        mean: m(t, λ) = amplitude * exp(-tau * λ^(-alpha)) + offset
        cov:  k(t,λ,t',λ') = k_time(t,t') * k_wavelength(λ,λ')

    This provides a physically-motivated model for dust-obscured AGB stars
    built entirely from standard Matérn/RBF kernels rather than the more
    complex :class:`~gpytorch.kernels.SpectralMixtureKernel`.

    Parameters
    ----------
    train_x : torch.Tensor
        Input of shape ``(n, 2)`` — column 0 is time, column 1 is wavelength.
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    time_kernel_type : str or gpytorch.kernels.Kernel, optional
        Type of time kernel (see :class:`WavelengthDependentGPModel`).
        Defaults to ``'matern'``.
    wavelength_kernel_type : str or gpytorch.kernels.Kernel, optional
        Type of wavelength kernel (see :class:`WavelengthDependentGPModel`).
        Defaults to ``'rbf'``.
    period : float, optional
        Initial period for the quasi-periodic time kernel option.
    wavelength_lengthscale : float, optional
        Initial wavelength correlation length.
    num_mixtures : int, optional
        Number of mixture components when ``time_kernel_type='spectral_mixture'``.
        Default 4.
    **kwargs
        Additional keyword arguments forwarded to :class:`WavelengthDependentGPModel`.

    Notes
    -----
    See :class:`DustMean` for the parameterisation of the mean function.
    For a spectral-mixture covariance variant see
    :class:`TwoDSpectralMixtureDustMeanGPModel`.

    Examples
    --------
    >>> import torch, gpytorch
    >>> from pgmuvi.gps import DustMeanGPModel
    >>> t_data = torch.linspace(0, 10, 50)
    >>> wl = torch.linspace(0.4, 2.5, 50)   # microns (optical to IR)
    >>> x = torch.stack([t_data, wl], dim=1)
    >>> y = torch.exp(-1.0 * wl ** -1.7) + 0.05 * torch.randn(50)
    >>> lik = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = DustMeanGPModel(x, y, lik)
    """

    def __init__(self, train_x, train_y, likelihood, **kwargs):
        kwargs.setdefault("time_kernel_type", "matern")
        kwargs.setdefault("wavelength_kernel_type", "rbf")
        super().__init__(
            train_x,
            train_y,
            likelihood,
            mean_module="dust",
            **kwargs,
        )


class PowerLawMeanGPModel(WavelengthDependentGPModel):
    """2D GP model combining a power-law wavelength mean with separable simple kernels.

    A convenience subclass of :class:`WavelengthDependentGPModel` that
    replaces the default quadratic mean function with :class:`PowerLawMean`:

    .. code-block:: text

        mean: m(t, λ) = offset + weight * λ^exponent
        cov:  k(t,λ,t',λ') = k_time(t,t') * k_wavelength(λ,λ')

    This provides a physically-motivated model for variable stars with
    steep wavelength-dependent mean flux, built entirely from standard
    Matérn/RBF kernels rather than the more complex
    :class:`~gpytorch.kernels.SpectralMixtureKernel`.

    Parameters
    ----------
    train_x : torch.Tensor
        Input of shape ``(n, 2)`` — column 0 is time, column 1 is wavelength.
    train_y : torch.Tensor
        Observed values (1D tensor).
    likelihood : gpytorch.likelihoods.Likelihood
        Likelihood function for the model.
    time_kernel_type : str or gpytorch.kernels.Kernel, optional
        Type of time kernel (see :class:`WavelengthDependentGPModel`).
        Defaults to ``'matern'``.
    wavelength_kernel_type : str or gpytorch.kernels.Kernel, optional
        Type of wavelength kernel (see :class:`WavelengthDependentGPModel`).
        Defaults to ``'rbf'``.
    period : float, optional
        Initial period for the quasi-periodic time kernel option.
    wavelength_lengthscale : float, optional
        Initial wavelength correlation length.
    num_mixtures : int, optional
        Number of mixture components when ``time_kernel_type='spectral_mixture'``.
        Default 4.
    **kwargs
        Additional keyword arguments forwarded to :class:`WavelengthDependentGPModel`.

    Notes
    -----
    See :class:`PowerLawMean` for the parameterisation of the mean function.
    For a spectral-mixture covariance variant see
    :class:`TwoDSpectralMixturePowerLawMeanGPModel`.

    Examples
    --------
    >>> import torch, gpytorch
    >>> from pgmuvi.gps import PowerLawMeanGPModel
    >>> t_data = torch.linspace(0, 10, 50)
    >>> wl = torch.linspace(0.4, 2.5, 50)   # microns
    >>> x = torch.stack([t_data, wl], dim=1)
    >>> y = wl ** -1.7 + 0.05 * torch.randn(50)
    >>> lik = gpytorch.likelihoods.GaussianLikelihood()
    >>> model = PowerLawMeanGPModel(x, y, lik)
    """

    def __init__(self, train_x, train_y, likelihood, **kwargs):
        kwargs.setdefault("time_kernel_type", "matern")
        kwargs.setdefault("wavelength_kernel_type", "rbf")
        super().__init__(
            train_x,
            train_y,
            likelihood,
            mean_module="power_law",
            **kwargs,
        )

