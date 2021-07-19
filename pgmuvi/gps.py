import numpy as np
import torch as t
import gpytorch as gpt
from gpytorch.means import ConstantMean
from gpytorch.kernels import SpectralMixtureKernel as SMK
from gpytorch.distributions import MultivariateNormal as MVN
from gpytorch.models import ExactGP




class TwoDSpectralMixtureGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_mixtures = 4):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = SMK(ard_num_dims = 2, num_mixtures = num_mixtures)
        self.covar_module.initialize_from_data(train_x, train_y)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MVN(mean_x, covar_x)
        
