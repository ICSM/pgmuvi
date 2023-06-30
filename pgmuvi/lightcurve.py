import contextlib
import numpy as np
import torch
import gpytorch
# from gps import SpectralMixtureGPModel as SMG
# from gps import SpectralMixtureKISSGPModel as SMKG
# from gps import TwoDSpectralMixtureGPModel as TMG
from .gps import * #FIX THIS LATER!
import matplotlib.pyplot as plt
from .trainers import train
from gpytorch.constraints import Interval
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from inspect import isclass


def _reraise_with_note(e, note):
    """Reraise an exception with a note added to the message

    This function is to provide a way to add a note to an exception, without
    losing the traceback, and without requiring python 3.11, which has
    added notes. It is based on this answer on stackoverflow:
    https://stackoverflow.com/a/75549200/16164384

    Parameters
    ----------
    e : Exception
        The exception to reraise
    note : str
        The note to add to the exception message
    """
    try:
        e.add_note(note)
    except AttributeError:
        args = e.args
        arg0 = f"{args[0]}\n{note}" if args else note
        e.args = (arg0,) + args[1:]
    raise e


class Transformer(object):
    def transform(self, data, **kwargs):
        """ Transform some data and return it, storing the parameters required
        to repeat or reverse the transform 

        This is a baseclass with no implementations, your subclass should
        implement the transform itself
        """
        raise NotImplementedError

    def inverse(self, data, shift=True, **kwargs):
        """ Invert a transform based on saved parameters

        This is a baseclass with no implementation, your subclass should
        implement the inverse transform itself
        """
        raise NotImplementedError


class MinMax(Transformer):
    def transform(self, data, dim=0, apply_to=None,
                  recalc=False, shift=True, **kwargs):
        """ Perform a MinMax transformation

        Transform the data such that each dimension is rescaled to the [0,1]
        interval. It stores the min and range of the data for the inverse
        transformation.

        Parameters
        ----------
        data : Tensor of floats
            The data to be transformed
        apply_to : tensor of ints or slice objects, optional
            Which dimensions to apply the transform to. If None, apply to all
        recalc : bool, default False
            Should the min and range of the transform be recalculated, or
            reused from previously?
        shift : bool, default True
            Should the data be shifted such that the minimum value is 0?
            This is mainly included so that data or parameters can be
            transformed when they apply to a single period - in this case,
            only the range needs to be applied.
        """
        if recalc or not hasattr(self, "min"):
            self.min = torch.min(data, dim=dim, keepdim=True)[0]
            self.range = torch.max(data, dim=dim, keepdim=True)[0] - self.min
            shift = True  # if we're recalculating, we need to shift
        if apply_to is not None:
            return (data-(shift*self.min[apply_to]))/self.range[apply_to]
        return (data-(shift*self.min))/self.range

    def inverse(self, data, shift=True, **kwargs):
        """ Invert a MinMax transformation based on saved state

        Invert the transformation of the data from  the [0,1] interval.
        It used the stored min and range of the data for the inverse
        transformation.

        Parameters
        ----------
        data : Tensor of floats
            The data to be reverse-transformed
        """
        return (data * self.range)+(shift*self.min)


class ZScore(Transformer):
    def transform(self, data, dim=0, apply_to=None,
                  recalc=False, shift=True, **kwargs):
        """ Perform a z-score transformation

        Transform the data such that each dimension is rescaled such that
        its mean is 0 and its standard deviation is 1.

        Parameters
        ----------
        data : Tensor of floats
            The data to be transformed
        apply_to : int or tensor of ints, optional
            Which dimensions to apply the transform to. If None, apply to all
        recalc : bool, default False
            Should the parameters of the transform be recalculated, or reused
            from previously?
        shift : bool, default True
            Should the data be shifted such that the mean value is 0?
            This is mainly included so that data or parameters can be
            transformed when they apply to a single period - in this case,
            only the standard deviation needs to be applied.
        """
        if recalc or not hasattr(self, 'mean'):
            self.mean = torch.mean(data, dim=dim, keepdim=True)[0]
            self.sd = torch.std(data, dim=dim, keepdim=True)[0]
            shift = True  # if we're recalculating, we need to shift
        if apply_to is not None:
            return (data-(shift*self.mean[apply_to]))/self.sd[apply_to]
        return (data - shift*self.mean)/self.sd

    def inverse(self, data, shift=True, **kwargs):
        """ Invert a z-score transformation based on saved state

        Invert the z-scoring of the data based on the saved mean and standard
        deviation

        Parameters
        ----------
        data : Tensor of floats
            The data to be reverse-transformed
        """
        return (data*self.sd) + (self.mean*shift)


class RobustZScore(Transformer):
    def transform(self, data, dim=0, apply_to=None,
                  recalc=False, shift=True, **kwargs):
        """ Perform a robust z-score transformation

        Transform the data such that each dimension is rescaled such that
        its median is 0 and its median absolute deviation is 1.

        Parameters
        ----------
        data : Tensor of floats
            The data to be transformed
        apply_to : int or tensor of ints, optional
            Which dimensions to apply the transform to. If None, apply to all
        recalc : bool, default False
            Should the parameters of the transform be recalculated, or reused
            from previously?
        shift : bool, default True
            Should the data be shifted such that the median value is 0?
            This is mainly included so that data or parameters can be
            transformed when they apply to a single period - in this case,
            only the median absolute deviation needs to be applied.
        """
        if recalc or not hasattr(self, 'mad'):
            self.median = torch.median(data, dim=dim, keepdim=True)[0]
            self.mad = torch.median(torch.abs(data - self.median),
                                    dim=dim, keepdim=True)[0]
            shift = True  # if we're recalculating, we need to shift
        if apply_to is not None:
            return (data-shift*self.median[apply_to])/self.mad[apply_to]
        return (data - shift*self.median)/self.mad

    def inverse(self, data, shift=True, **kwargs):
        """ Invert a robust z-score transformation based on saved state

        Invert the robust z-scoring of the data based on the saved median and
        median absolute deviation.

        Parameters
        ----------
        data : Tensor of floats
            The data to be reverse-transformed
        """
        return (data * self.mad) + (self.median*shift)


def minmax(data, dim=0):
    m = torch.min(data, dim=dim, keepdim=True)
    r = torch.max(data, dim=dim, keepdim=True) - m
    return (data-m)/r, m, r


class Lightcurve(object):
    """ A class for storing, manipulating and fitting light curves

    Long description goes here

    Parameters
    ----------
    xdata : Tensor of floats
    ydata : Tensor of floats
    yerr : Tensor of floats, optional
    xtransform : str, optional
    ytransform : str, optional


    Examples
    --------


    Notes
    -----
    """
    def __init__(self, xdata, ydata, yerr=None,
                 xtransform='minmax', ytransform=None,
                 name=None,
                 **kwargs):
        """_summary_

        Parameters
        ----------
        xdata : _type_
            _description_
        ydata : _type_
            _description_
        yerr : _type_, optional
            _description_, by default None
        xtransform : str, optional
            _description_, by default 'minmax'
        ytransform : _type_, optional
            _description_, by default None
        """
        
        transform_dic = {'minmax': MinMax(),
                         'zscore': ZScore(),
                         'robust_score': RobustZScore()}

        if xtransform is None or isinstance(xtransform, Transformer):
            self.xtransform = xtransform
        else:
            self.xtransform = transform_dic[xtransform]

        if ytransform is None or isinstance(ytransform, Transformer):
            self.ytransform = ytransform
        else:
            self.ytransform = transform_dic[ytransform]

        self.xdata = xdata
        self.ydata = ydata
        if yerr is not None:
            self.yerr = yerr

        self.name = "Lightcurve" if name is None else name

    @property
    def ndim(self):
        return self.xdata.shape[-1] if self.xdata.dim() > 1 else 1

    @property
    def magnitudes(self):
        pass

    @magnitudes.setter
    def magnitudes(self, value):
        pass

    @property
    def xdata(self):
        """ The independent variable data

        :getter: Returns the independent variable data in its raw
        (untransformed) state
        :setter: Takes the input data and transforms it as requested by the
        user 
        :type: torch.Tensor
        """
        return self._xdata_raw

    @xdata.setter
    def xdata(self, values):
        # first, store the raw data internally
        self._xdata_raw = values
        # then, apply the transformation to the values, so it can be used to
        # train the GP
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

    def set_likelihood(self, likelihood=None, **kwargs):
        """Set the likelihood function for the model

        Parameters
        ----------
        likelihood : string, None or instance of
                     gpytorch.likelihoods.likelihood.Likelihood or Constraint,
                     optional
            The likelihood function to use for the GP, by default None. If
            None, a Gaussian likelihood will be used. If a string, it must be
            'learn', in which case a Gaussian likelihood with learnable noise
            will be used. If an instance of a Likelihood object, that object
            will be used. If a Constraint object, a GaussianLikelihood will be
            used, with the constraint being passed to the likelihood as the
            noise_constraint argument. If a Constraint object is passed, the
            noise_constraint argument will override any other arguments passed
            to the likelihood function. You can also provide a class as input,
            in which case the class will be instantiated with the kwargs
            provided under the assumption that it is a Likelihood object, and
            it will also be passed the uncertainties on the y data, if available.
        """
        if hasattr(self, '_yerr_transformed') and likelihood is None:
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(self._yerr_transformed)
        elif hasattr(self, '_yerr_transformed') and likelihood == "learn":
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(self._yerr_transformed,
                                                                                learn_additional_noise = True)
        elif "Constraint" in [t.__name__ for t in type(likelihood).__mro__]:
            # In this case, the likelihood has been passed a constraint, which
            # means we want a constrained GaussianLikelihood
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=likelihood)
        elif likelihood is None:
            # We're just going to make the simplest assumption
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # Also add a case for if it is a Likelihood object
        elif isinstance(likelihood, gpytorch.likelihoods.likelihood.Likelihood):
            self.likelihood = likelihood
        elif isclass(likelihood):
            if hasattr(self, '_yerr_transformed'):
                self.likelihood = likelihood(self._yerr_transformed, **kwargs)
            else:
                self.likelihood = likelihood(**kwargs)
        else:
            raise ValueError(f"""Expected a string, a constraint, a Likelihood
                              instance or a class to be instantiated as a
                              Likelihood instance, but got {type(likelihood)}.
                              Please provide a suitable likelihood input.""")

    def set_model(self, model=None, likelihood=None,
                  num_mixtures=None, **kwargs):
        """Set the model for the lightcurve

        Parameters
        ----------
        model : string or instance of gpytorch.models.GP, optional
            The model to use for the GP, by default None. If None, an
            error will be raised. If a string, it must be one of the
            following:
                '1D': SpectralMixtureGPModel
                '2D': TwoDSpectralMixtureGPModel
                '1DLinear': SpectralMixtureLinearMeanGPModel
                '2DLinear': TwoDSpectralMixtureLinearMeanGPModel
                '1DSKI': SpectralMixtureKISSGPModel
                '2DSKI': TwoDSpectralMixtureKISSGPModel
                '1DLinearSKI': SpectralMixtureLinearMeanKISSGPModel
                '2DLinearSKI': TwoDSpectralMixtureLinearMeanKISSGPModel
            If an instance of a GP class, that object will be used.
            _description_, by default None
        likelihood : string, None or instance of
                     gpytorch.likelihoods.likelihood.Likelihood or Constraint,
                     optional
            If likelihood is passed, it will be passed along to `set_likelihood()`
            and used to set the likelihood function for the model. For details, see
            the documentation for `set_likelihood()`.
        num_mixtures : int, optional
            The number of mixtures to use in the spectral mixture kernel, by
            default None. If None, a default value will be used. This value
            is passed to the constructor for the model if a string is passed
            as the model argument.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the model constructor.
        """

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
            "2DLinearSKI": TwoDSpectralMixtureLinearMeanKISSGPModel
        }

        if likelihood is None and not hasattr(self, 'likelihood'):
            raise ValueError("""You must provide a likelihood function""")
        elif likelihood is not None:
            self.set_likelihood(likelihood, **kwargs)

        if "GP" in [t.__name__ for t in type(model).__mro__]:
            # check if it is or inherets from a GPyTorch model
            self.model = model
        elif model in model_dic_1:
            self.model = model_dic_1[model](self._xdata_transformed,
                                            self._ydata_transformed,
                                            self.likelihood,
                                            num_mixtures=num_mixtures,
                                            **kwargs)
        elif model in model_dic_2:
            self.model = model_dic_2[model](self._xdata_transformed,
                                            self._ydata_transformed,
                                            self.likelihood,
                                            num_mixtures=num_mixtures,
                                            **kwargs)
            # Add missing arguments to the model call
        else:
            raise ValueError("Insert a valid model")
        
    def set_model_prior(self, prior=None, **kwargs):
        '''Set the prior for the model parameters
        
        Parameters
        ----------
        prior : dict, optional
            A dictionary of the priors to use for the model parameters. The
            keys should be the names of the parameters, and the values should
            be instances of gpytorch.priors.Prior. If None, no priors will be
            used. If a prior is passed for a parameter that is not a model
            parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the Prior constructors.
        '''
        pass

    def set_model_constraint(self, constraint=None, **kwargs):
        '''Set the constraint for the model parameters

        Parameters
        ----------
        constraint : dict, optional
            A dictionary of the constraints to use for the model parameters. The
            keys should be the names of the parameters, and the values should
            be instances of gpytorch.constraints.Constraint. If None, no
            constraints will be used. If a constraint is passed for a parameter
            that is not a model parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the Constraint
            constructors.
        '''
        pass

    def set_likelihood_prior(self, prior=None, **kwargs):
        '''Set the prior for the likelihood parameters

        Parameters
        ----------
        prior : dict, optional
            A dictionary of the priors to use for the likelihood parameters. The
            keys should be the names of the parameters, and the values should
            be instances of gpytorch.priors.Prior. If None, no priors will be
            used. If a prior is passed for a parameter that is not a likelihood
            parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the Prior constructors.
        '''
        pass

    def set_likelihood_constraint(self, constraint=None, **kwargs):
        '''Set the constraint for the likelihood parameters

        Parameters
        ----------
        constraint : dict, optional
            A dictionary of the constraints to use for the likelihood parameters.
            The keys should be the names of the parameters, and the values
            should be instances of gpytorch.constraints.Constraint. If None, no
            constraints will be used. If a constraint is passed for a parameter
            that is not a likelihood parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the Constraint
            constructors.
        '''
        pass

    def set_hypers(self, hypers=None, debug=False, **kwargs):
        '''Set the hyperparameters for the model and likelihood. This is a
        convenience function that calls the model.initialize() to set the 
        hyperparameters. However, first it applies any transforms to the
        hyperparameters, so that the user can pass the hyperparameters in
        the original data space if they wish.

        Parameters
        ----------
        hypers : dict, optional
            A dictionary of the hyperparameters to use for the model and
            likelihood. The keys should be the names of the parameters, and the
            values should be Tensors containing the values of the parameters.
            If None, no hyperparameters will be set. If a hyperparameter is
            passed for a parameter that is not a model or likelihood
            parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the initialize.
        '''

        if hypers is None:
            return
        pars_to_transform = {'x': ['mixture_means', 'mixture_scales'],
                             'y': ['noise', 'mean_module']}
        if debug:
            print("hypers before transform:")
            print(hypers)
        for key in hypers:
            # first, check if the parameter needs to be transformed:
            if any(p in key for p in pars_to_transform['x']):
                # now apply the x transform
                # remember that the means and scales are in fourier space
                # so we need to transform them back to real space
                # before applying the transform
                # and then transform them back to fourier space
                # luckily, when the shift is removed from the transform,
                # the factors of 2pi cancel out for the scales
                # so we can just do 1/ for both means and scales
                if self.xtransform is not None:
                    if debug:
                        print(f"Applying x-transform to {key}")
                    hypers[key] = 1/self.xtransform.transform(1/hypers[key],
                                                          shift=False)
            elif any(p in key for p in pars_to_transform['y']):
                # now apply the y transform
                # the mean function and noise are not defined in fourier
                # space, so we can just apply the transform directly
                if self.ytransform is not None:
                    if debug:
                        print(f"Applying y-transform to {key}")
                    hypers[key] = self.ytransform.transform(hypers[key])
        if debug:
            print("hypers after transform:")
            print(hypers)
        self.model.initialize(**hypers, **kwargs)

    def init_hypers_from_LS(self, **kwargs):
        pass

    def _set_hypers_raw(self, hypers=None, **kwargs):
        pass

    def cuda(self):
        try:
            self.model.cuda()
            self.likelihood.cuda()
        except AttributeError as e:
            errmsg = "You must first set the model and likelihood"
            _reraise_with_note(e, errmsg)

    def _train(self):
        try:
            self.model.train()
            self.likelihood.train()
        except AttributeError as e:
            errmsg = "You must first set the model and likelihood"
            _reraise_with_note(e, errmsg)

    def fit(self, model=None, likelihood=None, num_mixtures=4,
            guess=None, grid_size=2000, cuda=False,
            training_iter=300, max_cg_iterations=None,
            optim="AdamW", miniter=None, stop=1e-5, lr=0.1,
            stopavg=30,
            **kwargs):
        """Fit the lightcurve

        Parameters
        ----------
        model : _type_, optional
            _description_, by default None
        likelihood : _type_, optional
            _description_, by default None
        num_mixtures : int, optional
            _description_, by default 4
        guess : _type_, optional
            _description_, by default None
        grid_size : int, optional
            _description_, by default 2000
        cuda : bool, optional
            _description_, by default False
        training_iter : int, optional
            _description_, by default 300
        max_cg_iterations : _type_, optional
            _description_, by default None
        optim : str, optional
            _description_, by default "AdamW"
        miniter : int, optional
            _description_, by default None
        stop : _type_, optional
            _description_, by default 1e-5
        lr : float, optional
            _description_, by default 0.1
        stopavg : int, optional
            _description_, by default 30

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if likelihood is None and not hasattr(self, 'likelihood'):
            raise ValueError("""You must provide a likelihood function""")
        elif likelihood is not None:
            self.set_likelihood(likelihood, **kwargs)

        if model is None and not hasattr(self, 'model'):
            raise ValueError("""You must provide a model""")
        elif model is not None:
            self.set_model(model, self.likelihood, 
                           num_mixtures=num_mixtures, **kwargs)

        if cuda:
            self.cuda()

        if guess is not None:
            #self.model.initialize(**guess)
            self.set_hypers(guess)

        if miniter is None:
            miniter = training_iter

        # Next we probably want to report some setup info
        # later...

        # Train the model
        # self.model.train()
        # self.likelihood.train()

        # set training mode:
        self._train()

        #for param_name, param in self.model.named_parameters():
        #    print(f'Parameter name: {param_name:42} value = {param.data}')
        self.print_parameters()

        # Now actually call the trainer!
        with gpytorch.settings.max_cg_iterations(10000):
            self.results = train(self,
                                 maxiter=training_iter,
                                 miniter=miniter,
                                 stop=stop, lr=lr,
                                 optim=optim, stopavg=stopavg)

        return self.results

    def print_periods(self):
        if self.ndim == 1:
            for i in range(len(self.model.covar_module.mixture_means)):
                if self.xtransform is None:
                    p = 1/self.model.covar_module.mixture_means[i]
                else:
                    p = self.xtransform.inverse(1/self.model.covar_module.mixture_means[i],
                                                shift=False).detach().numpy()[0]
                print(f"Period {i}: "
                      f"{p}"
                      f" weight: {self.model.covar_module.mixture_weights[i]}")
        elif self.ndim == 2:
            for i in range(len(self.model.covar_module.mixture_means[:, 0])):
                if self.xtransform is None:
                    p = 1/self.model.covar_module.mixture_means[i, 0]
                else:
                    p = self.xtransform.inverse(1/self.model.covar_module.mixture_means[i, 0],
                                                shift=False).detach().numpy()[0, 0]
                print(f"Period {i}: "
                      f"{p}"
                      f" weight: {self.model.covar_module.mixture_weights[i]}")

    def get_periods(self):
        '''
        Returns a list of the periods, scales and weights of the model. This 
        is useful for getting the periods after training, for example.
        '''
        periods = []
        scales = []
        weights = []
        if self.ndim == 1:
            for i in range(len(self.model.sci_kernel.mixture_means)):
                if self.xtransform is None:
                    p = 1/self.model.sci_kernel.mixture_means[i]
                    scales.append(1/(2*torch.pi*self.model.sci_kernel.mixture_scales[i]))
                else:
                    p = self.xtransform.inverse(1/self.model.sci_kernel.mixture_means[i],
                                                shift=False).detach().numpy()[0]
                    scales.append(self.xtransform.inverse(1/(2*torch.pi*self.model.sci_kernel.mixture_scales[i]),
                                                shift=False).detach().numpy()[0])
                periods.append(p)
                weights.append(self.model.sci_kernel.mixture_weights[i])
        elif self.ndim == 2:
            for i in range(len(self.model.sci_kernel.mixture_means[:, 0])):
                if self.xtransform is None:
                    p = 1/self.model.sci_kernel.mixture_means[i, 0]
                    scales.append(1/(2*torch.pi*self.model.sci_kernel.mixture_scales[i, 0]))
                else:
                    p = self.xtransform.inverse(1/self.model.sci_kernel.mixture_means[i, 0],
                                                shift=False).detach().numpy()[0, 0]
                    scales.append(self.xtransform.inverse(1/(2*torch.pi*self.model.sci_kernel.mixture_scales[i, 0]),
                                                shift=False).detach().numpy()[0, 0])
                periods.append(p)
                weights.append(self.model.sci_kernel.mixture_weights[i])

        return periods, weights, scales
                
    def get_parameters(self, raw=False):
        '''
        Returns a dictionary of the parameters of the model, with the keys
        being the names of the parameters and the values being the values of
        the parameters. This is useful for getting the values of the parameters
        after training, for example.

        The routine is rather hacky, since there is no built-in way to get the
        unconstrained values of the parameters from the model without knowing
        exactly what they are ahead of time. This routine therefore gets the
        names of the raw parameters, and then uses those names with string
        manipulation and `__getattr__` to get the values of the constrained
        parameters.

        Parameters
        ----------
        raw : bool, default False
            If True, returns the raw values of the parameters, otherwise
            returns the constrained values of the parameters.
        
        Returns
        -------
        pars : dict
            A dictionary of the parameters of the model, with the keys
            being the names of the parameters and the values being the values
            of the parameters.
        '''
        pars = {}
        for param_name, param in self.model.named_parameters():
            comps = list(param_name.split('.'))
            if not raw and 'raw' in param_name:
                # This is a constrained parameter, so we need to get the
                # unconstrained value
                pn = '.'.join([c.lstrip('raw_') for c in comps])
                tmp = self.model.__getattr__(comps[0])
                for i in range(1,len(comps)):
                    c = comps[i] if 'raw' not in comps[i] else comps[i].lstrip('raw_')
                    try:
                        tmp = tmp.__getattr__(c)
                    except AttributeError:
                        tmp = tmp.__getattribute__(c)
                pars[pn] = tmp.data
            else:
                # Either we actually want the raw values, or it's not a
                # constrained parameter
                pars[param_name] = param.data
        return pars
                
    def print_parameters(self, raw=False):
        '''
        Prints the parameters of the model, with the keys being the names of
        the parameters and the values being the values of the parameters. This
        is useful for getting the values of the parameters after training, for
        example.

        Parameters
        ----------
        raw : bool, default False
            If True, prints the raw values of the parameters, otherwise prints
            the constrained values of the parameters.

        '''
        pars = self.get_parameters(raw=raw)
        for key, value in pars.items():
            print(f"{key}: {value}")
        # for param_name, param in self.model.named_parameters():
        #     if raw:
        #         print(f'Parameter name: {param_name:42} value = {param.data}')
        #     else:

            #if 'raw' in param_name:
            #    print(f'Constrained Parameter name: {param_name[3:]:42} value = {param.constraint.transform(param.data)}')

    def print_results(self):
        for key in self.results.keys():
            results_tmp = self.results[key][-1]
            results_tmp_shape = results_tmp.shape  # e.g. (4,1,1)
            results_tmp_shape_len = len(results_tmp.shape)
            if results_tmp_shape_len == 1:
                print(f"{key}: {results_tmp}")
            else:
                sum_over_shape = sum(j > 1 for j in results_tmp_shape)
                if sum_over_shape in [0,1]:
                    print(f"{key}: {results_tmp.flatten()}")
                elif sum_over_shape == 2:
                    for i in range(results_tmp.shape[-1]):
                        print(f"{key}: {results_tmp[...,i].flatten()}")
                    # print(f"{key}: {results_tmp[...,0].flatten()}, {results_tmp[...,2].flatten()}")
            
    def plot_psd(self, freq=None, means=None, scales=None, weights=None,
                 show=True, raw=False, log=(True, False), **kwargs):
        
        if freq is None:
            if self.ndim == 1:
                if raw:
                    # our step size only needs to be small enough to resolve
                    # the width of the narrowest gaussian
                    step = self.model.sci_kernel.mixture_scales.min()/5
                    # this isn't really the correct way to do this, but it will
                    # do for now
                    diffs = self._xdata_transformed.sort().values[1:] - self._xdata_transformed.sort().values[:-1]
                    mindelta = (diffs[diffs>0]).min().item()
                    freq = torch.arange(1/(self._xdata_transformed.max() - self._xdata_transformed.min()).item(),
                                        1/(mindelta),
                                        step.item())  
                else:
                    # we have to transform the step size to the original space
                    # to get the correct frequency range
                    step = 1/self.xtransform.inverse(1/(self.model.sci_kernel.mixture_scales.min()/5),
                                                     shift=False)
                    # this isn't really the correct way to do this, but it will
                    # do for now
                    diffs = self._xdata_raw.sort().values[1:] - self._xdata_raw.sort().values[:-1]
                    mindelta = (diffs[diffs>0]).min().item()
                    freq = torch.arange(1/(self._xdata_raw.max() - self._xdata_raw.min()).item(),
                                        1/(mindelta/2),
                                        step.item())
            elif self.ndim == 2:
                raise NotImplementedError("""Plotting models and data in more than 1 dimension is not
                currently supported. Please get in touch if you need this
                functionality!
                """)
            else:
                raise NotImplementedError("""Plotting models and data in more than 2 dimensions is not
                currently supported. Please get in touch if you need this
                functionality!
                """)
        # Computing the psd for frequencies f
        psd = self.compute_psd(freq, means=means, scales=scales,
                               weights=weights, raw=raw)

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        #plotting psd
        ax.plot(freq, psd)
        if log[0]:
            ax.set_xscale('log')
        if log[1]:
            ax.set_yscale('log')
        if show:
            plt.show()
        else:
            return fig, ax

    def compute_psd(self, freq, means=None, scales=None, weights=None,
                    raw=False, **kwargs):
        if means is None:
            means = self.model.sci_kernel.mixture_means
            # now apply the transform too!
            if self.xtransform is not None and not raw:
                # there's probably an easier way to do this than converting to
                # a period and back, but this will do for now
                means = 1/self.xtransform.inverse(1/means, shift=False).detach().numpy()
        if scales is None:
            scales = self.model.sci_kernel.mixture_scales
            # now apply the transform too!
            if self.xtransform is not None and not raw:
                scales = 1/(2*np.pi*self.xtransform.inverse(1/(2*torch.pi*scales), shift=False).detach().numpy())
        if weights is None:
            weights = self.model.sci_kernel.mixture_weights.detach().numpy()
        from scipy.stats import norm
        c = np.atleast_1d(np.zeros((len(means),) + freq.shape, ))  # mean = mean of each gaussian in the psd (the kernel we use uses only gaussians).
        for i, m in enumerate(means): # f = array of frequencies that we want to plot
            s = scales[i]  # s.d
            w = weights[i]  # how much power is given to each gaussian
            c[i] = np.atleast_2d(np.sqrt(w)) * (norm.pdf(freq, m, s) - norm.pdf(-freq, m, s))  #subtracting negative side of psd - otherwise it would cause interference
            # Each component of the PSD is the weight times the difference of the forward and reverse PDFs
            # In this case, the weights are square-rooted, because the original AGW formula for the kernel uses weights**2 while gpytorch implements weights, and therefore we must adjust our interpretation of the output.
        # Now we just have to some over the components
        psd = np.sum(c, axis=0)
        return psd

    def plot(self, ylim=None, show=True):
        if ylim is None:
            ylim = [-3, 3]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get into evaluation (predictive posterior) mode
            self.model.eval()
            self.likelihood.eval()

            # Importing raw x and y training data from xdata and
            # ydata functions
            if self.ndim == 1:
                x_raw = self.xdata
            elif self.ndim == 2:
                x_raw = self.xdata[:, 0]
            y_raw = self.ydata

            # creating array of 10000 test points across the range of the data
            x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), 10000)

            if self.ndim == 1:
                fig = self._plot_1d(x_fine_raw, ylim=ylim, 
                                    show=show)
            elif self.ndim == 2:
                fig = self._plot_2d(x_fine_raw, ylim=ylim,
                                    show=show)
            else:
                raise NotImplementedError("""
                Plotting models and data in more than 2 dimensions is not
                currently supported. Please get in touch if you need this
                functionality!
                """)
        return fig

    def _plot_1d(self, x_fine_raw, ylim=None, show=False,
                 save=True):
        # transforming the x_fine_raw data to the space that the GP was
        # trained in (so it can predict)
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
        ax.plot(self.xdata.numpy(), self.ydata.numpy(), 'k*')

        # Plot predictive GP mean as blue line
        ax.plot(x_fine_raw.numpy(), observed_pred.mean.numpy(), 'b')

        # Shade between the lower and upper confidence bounds
        ax.fill_between(x_fine_raw.numpy(),
                        lower.numpy(), upper.numpy(),
                        alpha=0.5)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        if save:
            plt.savefig(f"{self.name}_fit.png")
        if show:
            plt.show()
        return f

    def _plot_2d(self, x_fine_raw, ylim=None, show=False,
                 save=True):
        if self.xtransform is None:
            x_fine_transformed = x_fine_raw
        elif isinstance(self.xtransform, Transformer):
            x_fine_transformed = self.xtransform.transform(x_fine_raw,
                                                           apply_to=(0, 0))
        unique_values_axis2 = torch.unique(self.xdata[:,1])
        figs = []
        for val in unique_values_axis2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.xdata[self.xdata[:, 1] == val, 0],
                    self.ydata[self.xdata[:, 1] == val],
                    "k*")

            vals = torch.ones_like(x_fine_transformed)*val
            x_fine_tmp = torch.cat((x_fine_transformed[:, None],
                                    vals[:, None]),
                                   dim=1)

            observed_pred = self.likelihood(self.model(x_fine_tmp))
            ax.plot(x_fine_raw.numpy(), observed_pred.mean.numpy(), 'b')

            lower, upper = observed_pred.confidence_region()
            ax.fill_between(x_fine_raw.numpy(),
                            lower.numpy(), upper.numpy(),
                            alpha=0.5)
            ax.legend(['Observed Data', 'Mean', 'Confidence'])

            ax.set_ylabel("y")
            ax.set_xlabel("x")
            ax.set_title(f"y vs x for {val}")
            if ylim is not None:
                ax.set_ylim(ylim)
            if save:
                plt.savefig(f"{self.name}_{val}_fit.png")
        
            if show:
                plt.show()
            figs.append(fig)
        return figs
    
    def _plot_nd(self):
        raise NotImplementedError("""
        Plotting models and data in more than 2 dimensions is not currently supported.
        Please get in touch if you need this functionality!
        """)
            
    def plot_results(self):
        for key, value in self.results.item():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            with contextlib.suppress(ValueError):
                ax.plot(value, "-")
            ax.set_ylabel(key)
            ax.set_xlabel("Iteration")

            if "means" in key:
                self.value_inversed = self.xtransform.inverse(value)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(torch.Tensor(self.value_reversed),"-")
                ax.set_ylabel(key)
                ax.set_xlabel("Iteration")
        plt.show()

