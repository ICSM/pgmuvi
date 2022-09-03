import numpy as np
import torch as t
import gpytorch as gpt
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import SpectralMixtureKernel as SMK
from gpytorch.kernels import GridInterpolationKernel as GIK
from gpytorch.distributions import MultivariateNormal as MVN
from gpytorch.models import ExactGP, ApproximateGP



#### FIRST WE HAVE SOME Naive GPs
class SpectralMixtureGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = SMK(num_mixtures = num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)

class SpectralMixtureLinearMeanGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4):
        super(SpectralMixtureLinearMeanGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean()
        self.covar_module = SMK(num_mixtures = num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)

class TwoDSpectralMixtureGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4):
        super(TwoDSpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = SMK(ard_num_dims = 2, num_mixtures = num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)
        

class TwoDSpectralMixtureLinearMeanGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4):
        super(TwoDSpectralMixtureLinearMeanGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean()
        self.covar_module = SMK(ard_num_dims = 2, num_mixtures = num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


#Now we define some that use KISS-GP/SKI to try to accelerate inference
class SpectralMixtureKISSGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4, grid_size = 2000):
        super(SpectralMixtureKISSGPModel, self).__init__(train_x, train_y, likelihood)
        if not grid_size:
            grid_size = gpt.utils.grid.choose_grid_size(train_x,1.0)
            print("Using a grid of size ", grid_size," for SKI")
        grid_bounds = [[t.min(train_x), t.max(train_x)]]
        self.mean_module = ConstantMean()
        self.covar_module = GIK(SMK(num_mixtures = num_mixtures), grid_size = grid_size, num_dims=1, grid_bounds = grid_bounds
        )

        self.covar_module.base_kernel.initialize_from_data(train_x, train_y)

        self.sci_kernel = self.covar_module.base_kernel
        #self.covar_module.base_kernel.initialize_from_data(train_x, train_y)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)
        

class SpectralMixtureLinearMeanKISSGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4, grid_size = 2000):
        super(SpectralMixtureLinearMeanKISSGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean()
        self.covar_module = GIK(SMK(num_mixtures = num_mixtures), grid_size = grid_size
        )
        self.covar_module.base_kernel.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)

    
class TwoDSpectralMixtureKISSGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4, grid_size = [5000,20]):
        super(SpectralMixtureKISSGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = GIK(SMK(ard_num_dims = 2, num_mixtures = num_mixtures), num_dims = 2, grid_size = grid_size
        )
        self.covar_module.base_kernel.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)
        

class TwoDSpectralMixtureLinearMeanKISSGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4, grid_size = [5000,20]):
        super(TwoDSpectralMixtureLinearMeanKISSGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean()
        self.covar_module = GIK(SMK(ard_num_dims = 2, num_mixtures = num_mixtures), num_dims = 2, grid_size = grid_size
        )
        self.covar_module.base_kernel.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)


from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
    
#We can also implement sparse/variational GPs here
class SparseSpectralMixtureGPModel(ApproximateGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SpectralMixtureGPModel, self).__init__(variational_strategy)
        #super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = SMK(num_mixtures = num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)

class SparseSpectralMixtureLinearMeanGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4):
        super(SpectralMixtureLinearMeanGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean()
        self.covar_module = SMK(num_mixtures = num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)

class SparseTwoDSpectralMixtureGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4):
        super(TwoDSpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = SMK(ard_num_dims = 2, num_mixtures = num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)
        

class SparseTwoDSpectralMixtureLinearMeanGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4):
        super(TwoDSpectralMixtureLinearMeanGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean()
        self.covar_module = SMK(ard_num_dims = 2, num_mixtures = num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)

        #Now we alias the covariance kernel so that we can exploit the same object properties in different classes with different kernel structure
        #Will turn this into an @property at some point.
        self.sci_kernel = self.covar_module


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)
