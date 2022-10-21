import numpy as np
import torch
import gpytorch
import pandas as pd
# from gps import SpectralMixtureGPModel as SMG
# from gps import SpectralMixtureKISSGPModel as SMKG
# from gps import TwoDSpectralMixtureGPModel as TMG
from .gps import *  # FIX THIS LATER!
import matplotlib.pyplot as plt
from trainers import train
from gpytorch.constraints import Interval
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from scipy.stats import multivariate_normal


class Transformer(object):
    def transform(self, data, **kwargs):
        """ Transform some data and return it, storing the parameters required to repeat or reverse the transform

		This is a baseclass with no implementations, your subclass should
		implement the transform itself
		"""
        raise NotImplementedError

    def inverse(self, data, **kwargs):
        """ Invert a transform based on saved parameters

		This is a baseclass with no implementation, your subclass should
		implement the inverse transform itself
		"""
        raise NotImplementedError


class MinMax(Transformer):
    def transform(self, data, dim=0, recalc=False, **kwargs):
        """ Perform a MinMax transformation

		Transform the data such that each dimension is rescaled to the [0,1]
		interval. It stores the min and range of the data for the inverse
		transformation.

		Parameters
		----------
		data : Tensor of floats
			The data to be transformed
		recalc : bool, default False
			Should the min and range of the transform be recalculated, or reused from previously?
		"""
        if recalc or self.min is None:
            self.min = torch.min(data, dim=dim, keepdim=True)
            self.range = torch.max(data, dim=dim, keepdim=True) - self.min
        return (data - self.min) / self.range

    def inverse(self, data, **kwargs):
        """ Invert a MinMax transformation based on saved state

		Invert the transformation of the data from  the [0,1] interval.
		It used the stored min and range of the data for the inverse
		transformation.

		Parameters
		----------
		data : Tensor of floats
			The data to be reverse-transformed
		"""
        return (data * self.range) + self.min


class ZScore(Transformer):
    def transform(self, data, dim=0, recalc=False, **kwargs):
        """ Perform a z-score transformation

		Transform the data such that each dimension is rescaled to the such that
		its mean is 0 and its standard deviation is 1.

		Parameters
		----------
		data : Tensor of floats
			The data to be transformed
		recalc : bool, default False
			Should the parameters of the transform be recalculated, or reused from previously?
		"""
        if recalc or self.mean is None:
            self.mean = torch.mean(data, dim=dim, keepdim=True)
            self.sd = torch.std(data, dim=dim, keepdim=True)
        return (data - self.mean) / self.sd

    def inverse(self, data, **kwargs):
        """ Invert a z-score transformation based on saved state

		Invert the z-scoring of the data based on the saved mean and standard
		deviation

		Parameters
		----------
		data : Tensor of floats
			The data to be reverse-transformed
		"""
        return (data * self.sd) + self.mean


class RobustZScore(Transformer):
    def transform(self, data, dim=0, recalc=False, **kwargs):
        """ Perform a robust z-score transformation

		Transform the data such that each dimension is rescaled to the such that
		its median is 0 and its median absolute deviation is 1.

		Parameters
		----------
		data : Tensor of floats
			The data to be transformed
		recalc : bool, default False
			Should the parameters of the transform be recalculated, or reused from previously?
		"""
        if recalc or self.mad is None:
            self.median = torch.median(data, dim=dim, keepdim=True)
            self.mad = torch.median(torch.abs(data - self.median), dim=dim, keepdim=True)
        return (data - self.median) / self.mad

    def inverse(self, data, **kwargs):
        """ Invert a robust z-score transformation based on saved state

		Invert the robust z-scoring of the data based on the saved median and
		median absolute deviation.

		Parameters
		----------
		data : Tensor of floats
			The data to be reverse-transformed
		"""
        return (data * self.mad) + self.median


def minmax(data, dim=0):
    m = torch.min(data, dim=dim, keepdim=True)
    r = torch.max(data, dim=dim, keepdim=True) - m
    return (data - m) / r, m, r


class Lightcurve(object):
    """ A class for storing, manipulating and fitting light curves

	Long description goes here

	Parameters
	----------



	Examples
	--------


	Notes
	-----
	"""

    def __init__(self, xdata, ydata, yerr=None,
                 xtransform='minmax', ytransform=None,
                 **kwargs):
        if xtransform is 'minmax':
            self.xtransform = MinMax()
        elif xtransform is 'zscore':
            self.xtransform = ZScore()
        elif xtransform is 'robust_zscore':
            self.xtransform = RobustZScore()
        elif xtransform is None or isinstance(xtransform, Transformer):
            self.xtransform = xtransform

        if ytransform is 'minmax':
            self.ytransform = MinMax()
        elif ytransform is 'zscore':
            self.ytransform = ZScore()
        elif ytransform is 'robust_zscore':
            self.ytransform = RobustZScore()
        elif ytransform is None or isinstance(ytransform, Transformer):
            self.ytransform = ytransform
        self.xdata = xdata
        self.ydata = ydata
        if yerr is not None:
            self.yerr = yerr
        pass

    @property
    def magnitudes(self):
        pass

    @magnitudes.setter
    def magnitudes(self, value):
        pass

    @property
    def xdata(self):
        """ The independent variable data

		:getter: Returns the independent variable data in its raw (untransformed) state
		:setter: Takes the input data and transforms it as requested by the user
		:type: torch.Tensor
		"""
        return self._xdata_raw

    @xdata.setter
    def xdata(self, values):
        # first, store the raw data internally
        self._xdata_raw = values
        # then, apply the transformation to the values, so it can be used to train the GP
        if self.xtransform is None:
            self._xdata_transformed = values
        elif isinstance(self.xtransform, Transformer):
            self._xdata_transformed = self.xtransform.transform(values)

    @property
    def ydata(self):
        return self._ydata_raw

    @ydata.setter
    def ydata(self, values):
        # first, store the raw data internally
        self._ydata_raw = values
        # then, apply the transformation to the values
        if self.ytransform is None:
            self._ydata_transformed = values
        elif isinstance(self.ytransform, Transformer):
            self._ydata_transformed = self.ytransform.transform(values)

    @property
    def yerr(self):
        return self._yerr_raw

    @yerr.setter
    def yerr(self, values):
        self._yerr_raw = values
        # now apply the same transformation that was applied to the ydata
        if self.ytransform is None:
            self._yerr_transformed = values
        elif isinstance(self.ytransform, Transformer):
            self._yerr_transformed = self.ytransform.transform(values)

    def append_data(self, new_values_x, new_values_y):
        pass

    def transform_x(self, values):
        if self.xtransform is None:
            return values
        elif isinstance(self.xtransform, Transformer):
            return self.xtransform.transform(values)

    def transform_y(self, values):
        if self.ytransform is None:
            return values
        elif isinstance(self.xtransform, Transformer):
            return self.xtransform.transform(values)

    def fit(self, model=None, likelihood=None, num_mixtures=4,
            guess=None, grid_size=2000, cuda=False,
            training_iter=300, max_cg_iterations=None,
            optim="AdamW", miniter=100, stop=1e-5, lr=0.1,
            stopavg=30,
            **kwargs):
        if self._yerr_transformed is not None and likelihood is None:
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                self._yerr_transformed)  # , learn_additional_noise = True)
        elif self._yerr_transformed is not None and likelihood is "learn":
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(self._yerr_transformed,
                                                                                learn_additional_noise=True)
        elif "Constraint" in [t.__name__ for t in type(likelihood).__mro__]:
            # In this case, the likelihood has been passed a constraint, which means we want a constrained GaussianLikelihood
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=likelihood)
        elif likelihood is None:
            # We're just going to make the simplest assumption
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # Also add a case for if it is a Likelihood object

        model_dic_1 = {
            "2D": TwoDSpectralMixtureGPModel,
            "1D": SpectralMixtureGPModel,
            "1DLinear": SpectralMixtureLinearMeanGPModel,
            "2DLinear": TwoDSpectralMixtureLinearMeanGPModel
        }

        model_dic_2 = {
            "1DSKI": SpectralMixtureKISSGPModel,
            "2DSKI": TwoDSpectralMixtureKISSGPModel,
            "1DLinearSKI": SpectralMixtureLinearMeanKISSGPModel,
            "2DLinearSKI": TwoDSpectrakMixtureLinearMeanKISSGPModel
        }

        if "GP" in [t.__name__ for t in type(model).__mro__]:  # check if it is or inherets from a GPyTorch model
            self.model = model

        elif model in model_dic_1.keys():
            self.model = model_dic_1[model](self._xdata_transformed,
                                            self._ydata_transformed,
                                            self.likelihood,
                                            num_mixtures=num_mixtures)
        elif model in model_dic_2.keys():
            self.model = model_dic_2[model](self._xdata_transformed,
                                            self._ydata_transformed,
                                            self.likelihood,
                                            num_mixtures=num_mixtures)  # Add missing arguments

        else:
            raise ValueError("Insert a valid model")

        if cuda:
            self.likelihood.cuda()
            self.model.cuda()

        if guess is not None:
            self.model.initialize(**guess)

        # Next we probably want to report some setup info

        # Train the model
        self.model.train()
        self.likelihood.train()

        for param_name, param in self.model.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.data}')

        with gpytorch.settings.max_cg_iterations(10000):
            self.results = train(self.model, self.likelihood, self._xdata_transformed, self._ydata_transformed,
                                 maxiter=training_iter, miniter=miniter, stop=stop, lr=lr, optim=optim, stopavg=stopavg)

        return self.results

    # Now we're going

    def plot_psd(self, means, freq, scales, weights, show=True, wavelengths=None):
        if means.ndim == 1:  # 1D case
            # Computing the psd for frequencies freq
            psd = compute_psd_1D(means, freq, scales, weights)

            # Initialise plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            # plotting psd
            ax.plot(freq, psd)
            if show:
                plt.show()
            else:
                return fig, ax

    if means.ndim == 2:  # 2D case
        # Computing the psd for frequencies and wavelengths
        psd = compute_psd_2D(means, freq, scales, weights, wavelengths)

        # Initialise plot
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Plot the surface
        surf = ax.plot_surface(freq, wavelengths, psd, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Add a colour bar which maps values to colours
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def compute_psd_2D(self, means, freq, scales, weights, wavelengths):
        c = np.zeros((len(means[0, :]),) + [freq,
                                            wavelengths].shape, )  # CHECK THIS IS CORRECT SHAPE #mean = mean of each gaussian in the psd (the kernel we use uses only gaussians).
        for i in range(len(means[0, :])):  # freq = array of frequencies that we want to plot
            s = scales[:, i]  # s.d
            w = weights[i]  # how much power is given to each gaussian
            m = means[:, i]
            c[i] = np.sqrt(w) * (multivariate_normal.pdf([freq, wavelengths], m, s) - multivariate_normal.pdf(
                [-freq, wavelengths], m,
                s))  # CHECK THIS FOR 2D - DOES WAVELENGTH ALSO NEED TO BE SUBTRACTED? ALSO SHOULD IT BE WAVENUMBER?? #subtracting negative side of psd - otherwise it would cause interference
        # Each component of the PSD is the weight times the difference of the forward and reverse PDFs
        # In this case, the weights are square-rooted, because the original AGW formula for the kernel uses weights**2 while gpytorch implements weights, and therefore we must adjust our interpretation of the output.
        # Now we just have to some over the components
        psd = np.sum(c, axis=0)  # Check axis to sum over (want to sum over power axis)
        return psd

    def compute_psd_1D(self, means, freq, scales, weights):
        c = np.zeros((len(
            means),) + freq.shape, )  # mean = mean of each gaussian in the psd (the kernel we use uses only gaussians).
        for i, m in enumerate(means):  # f = array of frequencies that we want to plot
            s = scales[i]  # s.d
            w = weights[i]  # how much power is given to each gaussian
            c[i] = np.sqrt(w) * (norm.pdf(freq, m, s) - norm.pdf(-freq, m,
                                                                  s))  # subtracting negative side of psd - otherwise it would cause interference
        # Each component of the PSD is the weight times the difference of the forward and reverse PDFs
        # In this case, the weights are square-rooted, because the original AGW formula for the kernel uses weights**2 while gpytorch implements weights, and therefore we must adjust our interpretation of the output.
        # Now we just have to some over the components
        psd = np.sum(c, axis=0)
        return psd

    def plot(self):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get into evaluation (predictive posterior) mode
            self.model.eval()
            self.likelihood.eval()

            # Importing raw x and y training data from xdata and ydata functions
            x_raw = self.xdata()
            y_raw = self.ydata()

            # creating array of 10000 test points across the range of the data
            x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), 10000)

            # transforming the x_fine_raw data to the space that the GP was trained in (so it can predict)
            if self.xtransform is None:
                x_fine_transformed = x_fine_raw
            elif isinstance(self.xtransform, Transformer):
                x_fine_transformed = self.xtransform.transform(x_fine_raw)

            # Make predictions
            observed_pred = self.likelihood(self.model(x_fine_transformed))

            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(8, 6))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()

            # Plot training data as black stars
            ax.plot(x_raw.numpy(), y_raw.numpy(), 'k*')

            # Plot predictive GP mean as blue line
            ax.plot(x_fine_raw.numpy(), observed_pred.mean.numpy(), 'b')

            # Shade between the lower and upper confidence bounds
            ax.fill_between(x_fine_raw.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_ylim([3, -3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            plt.show()
