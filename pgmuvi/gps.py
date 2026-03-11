import torch as t
import gpytorch as gpt
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import SpectralMixtureKernel as SMK
from gpytorch.kernels import GridInterpolationKernel as GIK
from gpytorch.distributions import MultivariateNormal as MVN
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


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
    physical wavelengths).  Very small wavelength values may cause numerical
    overflow in ``λ^(-alpha)``; applying a suitable wavelength transform
    (e.g. ``MinMax``) before fitting is recommended.
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
            t.nn.Parameter(t.full((*batch_shape, 1), t.tensor(1.7).log().item())),
        )

    def forward(self, x):
        wavelength = x[..., 1].clamp(min=1e-6)  # guard against λ ≤ 0
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
