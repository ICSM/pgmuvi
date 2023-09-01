import contextlib
import numpy as np
import torch
import gpytorch
from .gps import (SpectralMixtureLinearMeanGPModel,
                  SpectralMixtureLinearMeanKISSGPModel,
                  TwoDSpectralMixtureLinearMeanGPModel,
                  TwoDSpectralMixtureLinearMeanKISSGPModel,
                  SpectralMixtureGPModel,
                  SpectralMixtureKISSGPModel,
                  TwoDSpectralMixtureGPModel,
                  TwoDSpectralMixtureKISSGPModel)
import matplotlib.pyplot as plt
from .trainers import train
from gpytorch.constraints import Interval, GreaterThan, LessThan, Positive  # noqa: F401
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior  # noqa: F401
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from inspect import isclass
import xarray as xr
import arviz as az


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

    This class is designed to be a convenient way to store and manipulate
    light curve data, and to fit Gaussian Processes to that data. It is
    designed to be used with the GPyTorch library, and to be compatible with
    the Pyro library for MCMC fitting.

    Parameters
    ----------
    xdata : Tensor of floats
        The independent variable data
    ydata : Tensor of floats
        The dependent variable data
    yerr : Tensor of floats, optional
        The uncertainties on the dependent variable data, by default None
    xtransform : str, optional
        The transform to apply to the x data, by default 'minmax'
    ytransform : str, optional
        The transform to apply to the y data, by default None


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
        xdata : torch.Tensor
            The independent variable data
        ydata : torch.Tensor
            The dependent variable data
        yerr : torch.Tensor, optional
            The uncertainties on the dependent variable data, by default None
        xtransform : str or Transformer, optional
            The transform to apply to the x data, by default 'minmax'
        ytransform : str or Transformer, optional
            The transform to apply to the y data, by default None
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

        self.__SET_LIKELIHOOD_CALLED = False
        self.__SET_MODEL_CALLED = False
        self.__CONTRAINTS_SET = False
        self.__PRIORS_SET = False
        self.__FITTED_MAP = False
        self.__FITTED_MCMC = False

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
        """ The dependent variable data

        :getter: Returns the dependent variable data in its raw
        (untransformed) state
        :setter: Takes the input data and transforms it as requested by the
        user
        :type: torch.Tensor"""
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
        """ The uncertainties on the dependent variable data

        :getter: Returns the uncertainties on the dependent variable data in
        its raw (untransformed) state
        :setter: Takes the input data and transforms it as requested by the
        user
        :type: torch.Tensor
        """
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

        self.__SET_LIKELIHOOD_CALLED = True
        if hasattr(self, '_yerr_transformed') and likelihood is None:
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(self._yerr_transformed)  # noqa: E501
        elif hasattr(self, '_yerr_transformed') and likelihood == "learn":
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(self._yerr_transformed,  # noqa: E501
                                                                                learn_additional_noise=True)  # noqa: E501
        elif likelihood == "learn":
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(learn_additional_noise=True)  # noqa: E501
        elif "Constraint" in [t.__name__ for t in type(likelihood).__mro__]:
            # In this case, the likelihood has been passed a constraint, which
            # means we want a constrained GaussianLikelihood
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=likelihood)  # noqa: E501
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
        self.__SET_MODEL_CALLED = True
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

        if not hasattr(self, 'likelihood'):
            self.set_likelihood(likelihood, **kwargs)
        elif not self.__SET_LIKELIHOOD_CALLED and likelihood is None:
            # if no likelihood is passed, we only want to set the likelihood
            # if it hasn't already been set
            self.set_likelihood(likelihood, **kwargs)
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

        # now we've got a model set up, we're going to make some handy lookups
        # for the parameters and modules that we'll need to access later
        self._make_parameter_dict()
        # self.set_default_constraints()

    def _make_parameter_dict(self):
        ''' Make a dictionary of the model parameters

        This function is used to make a dictionary of the model parameters,
        providing a convenient way to access them. The dictionary is stored
        in the _model_pars attribute.
        '''
        self._model_pars = {}
        # there are a few parameters that we want to make sure we expose a
        # direct link to if we need them!
        _special_pars = ['noise', 'mixture_means',
                         'mixture_scales', 'mixture_weights']

        for param_name, param in self.model.named_parameters():
            comps = list(param_name.split('.'))
            pn_root = comps[-1]
            param_dict = {
                'full_name': param_name,
                'root_name': pn_root,
                'chain': [],
                'constrained': False,
            }
            if 'raw' in param_name:
                # This is a constrained parameter, so we need to get the
                # unconstrained value
                pn_const = comps[-1].lstrip('raw_')
                param_dict['constrained'] = True
                param_dict['constrained_name'] = pn_const
                pn = '.'.join([c.lstrip('raw_') for c in comps])
                param_dict['constrained_full_name'] = pn
            tmp = self.model.__getattr__(comps[0])
            param_dict['chain'].append(tmp)
            for i in range(1,len(comps)):
                c = comps[i] if 'raw' not in comps[i] else comps[i].lstrip('raw_')
                try:
                    tmp = tmp.__getattr__(c)
                except AttributeError:
                    tmp = tmp.__getattribute__(c)
                param_dict['chain'].append(tmp)
            param_dict['module'] = param_dict['chain'][-2]
            if param_dict['constrained']:
                param_dict['param'] = param_dict['chain'][-1]
                try:
                    param_dict['raw_param'] = param_dict['chain'][-2].__getattr__(comps[-1])  # noqa: E501
                except AttributeError:
                    param_dict['raw_param'] = param_dict['chain'][-2].__getattribute__(comps[-1])  # noqa: E501
            else:
                param_dict['param'] = param_dict['chain'][-1]
                param_dict['raw_param'] = param_dict['param']
            if any(s in pn_root for s in _special_pars):
                # it's a special parameter that we want extra easy access to!
                param_dict['special'] = True
                j = np.argmax([s in pn_root for s in _special_pars])
                self._model_pars[_special_pars[j]] = param_dict

            #     pars[pn] = tmp.data
            # else:
            #     # Either we actually want the raw values, or it's not a
            #     # constrained parameter
            #     pars[param_name] = param.data
            self._model_pars[param_name] = param_dict
            if param_dict['constrained']:
                self._model_pars[pn] = param_dict  # so we also alias the full
                                                   # name for the constrained
                                                   # parameter

    def set_prior(self, prior=None, **kwargs):
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
        self.__PRIORS_SET = True
        pass

    def set_constraint(self, constraint, debug=False, **kwargs):
        '''Set the constraint for the model parameters

        Parameters
        ----------
        constraint : dict, optional
            A dictionary of the constraints to use for the model parameters.
            The keys should be the names of the parameters, and the values
            should be instances of gpytorch.constraints.Constraint. If None, no
            constraints will be used. If a constraint is passed for a parameter
            that is not a model parameter, it will be ignored.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the Constraint
            constructors.
        '''
        # which paramaters need to have their constraints transformed? and how?
        pars_to_transform = {'x': ['mixture_means', 'mixture_scales'],
                             'y': ['noise', 'mean_module']}

        for key in constraint:
            if key in self._model_pars:
                if debug:
                    print(f"Found parameter {key} in model parameters")
                    print(f"Parameter {key} will have constraint: {constraint[key]}")
                    print("which may be transformed")
                # constraints must be registered to raw parameters!
                k = key.split('.')[-1] if 'raw_' in key else f"raw_{key.split('.')[-1]}"
                if all(p not in key
                       for p in pars_to_transform['y'] + pars_to_transform['x']):  # no transform needed! # noqa: E501
                    self._model_pars[key]['module'].register_constraint(
                        k, constraint[key]
                    )
                elif any(p in key for p in pars_to_transform['x']):
                    # now apply the x transform
                    # remember that the means and scales are in fourier space
                    # so we need to transform them back to real space
                    # before applying the transform
                    # and then transform them back to fourier space
                    # luckily, when the shift is removed from the transform,
                    # the factors of 2pi cancel out for the scales
                    # so we can just do 1/ for both means and scales
                    if self.xtransform is not None:
                        # now things get complicated...
                        # if we have gotten to here, we know that the parameter
                        # is a mixture mean or scale, so we need to transform
                        # it to real space, apply the constraint, and then
                        # transform it back to fourier space
                        # luckily, when the shift is removed from the transform,
                        # the factors of 2pi cancel out for the scales
                        # so we can just do 1/ for both means and scales
                        if debug:
                            print(constraint[key])
                        if (constraint[key].lower_bound
                           not in [torch.tensor(0), torch.tensor(-torch.inf)]):
                            # we need to transform the lower bound
                            if debug:
                                print(1./self.xtransform.transform(1./constraint[key].lower_bound,
                                                                    apply_to=[0],
                                                                    shift=False))
                            constraint[key].lower_bound = torch.tensor(1./self.xtransform.transform(1./constraint[key].lower_bound,  # noqa: E501
                                                                                                    apply_to=[0],  # noqa: E501
                                                                                                    shift=False).item())  # noqa: E501
                            if debug:
                                print(constraint[key].lower_bound)
                                print(constraint[key])
                        if (constraint[key].upper_bound
                            not in [torch.tensor(0), torch.tensor(torch.inf)]):
                            # we need to transform the upper bound
                            constraint[key].upper_bound = torch.tensor(1./self.xtransform.transform(1./constraint[key].upper_bound,  # noqa: E501
                                                                                                    apply_to=[0],  # noqa: E501
                                                                                                    shift=False).item())  # noqa: E501
                            if debug:
                                print(constraint[key].upper_bound)
                                print(constraint[key])
                        if debug:
                            print(constraint[key])
                    self._model_pars[key]['module'].register_constraint(
                        k, constraint[key]
                    )
                elif any(p in key for p in pars_to_transform['y']):
                    if self.ytransform is not None:
                        if debug:
                            print(constraint[key])
                        if isinstance(constraint[key], Positive) and (
                            isinstance(self.ytransform, (ZScore, RobustZScore))
                        ):
                            # convert constraint to an interval with minimum equal to
                            # what zero is in the untransformed space
                            constraint[key] = Interval(self.ytransform.transform(0),
                                                       torch.inf)
                            if debug:
                                print(constraint[key])

                        elif (constraint[key].lower_bound
                             not in [torch.tensor(0), torch.tensor(-torch.inf)]):
                            # we need to transform the lower bound
                            constraint[key].lower_bound = torch.tensor(self.ytransform.transform(constraint[key].lower_bound).item())  # noqa: E501
                            if debug:
                                print(constraint[key].lower_bound)
                                print(constraint[key])
                        if (constraint[key].upper_bound
                           not in [torch.tensor(0), torch.tensor(torch.inf)]):
                            # we need to transform the upper bound
                            constraint[key].upper_bound = torch.tensor(self.ytransform.transform(constraint[key].upper_bound).item())  # noqa: E501
                            if debug:
                                print(constraint[key].upper_bound)
                                print(constraint[key])
                    if debug:
                            print(constraint[key])
                    self._model_pars[key]['module'].register_constraint(
                        k, constraint[key]
                    )
                if debug:
                    try:
                        print(f"Registered constraint {constraint[key]}")
                    except TypeError:
                        print("Registered constraint")
                        print(constraint[key])
                    print(f"to parameter {key}")
            else:
                print(f"Parameter {key} not found in model parameters,")
                print("this constraint will be ignored.")
                print("Available parameters are:")
                print(self._model_pars.keys())
                print("(Beware, several of these are aliases!)")

    def set_default_priors(self, **kwargs):
        '''Set the default priors for the model and likelihood parameters

        The default priors are as follows:
            - Parameters that must be positive are given LogNormal, HalfNormal
            or HalfCauchy priors, depending on the parameter.
            - The noise is given a HalfNormal prior with a scale of 1/10 of the
            smallest uncertainty on the y-data, if uncertainties are given, or
            1/10 of the standard deviation of the y data.
            - The mean of the GP is given a Gaussian prior with a mean of the
            mean of the y-data and a standard deviation of 1/10 of the standard
            deviation of the y-data.


        Parameters
        ----------
        **kwargs : dict, optional
            Any keyword arguments to be passed to the Prior constructors.
        '''

        # Gpytorch currently crashes if you try to do MCMC while learning additional
        # diagonal noise with the FixedNoiseGaussianLikelihood. So we only need to
        # set priors for the noise if we don't have uncertainties on the data.
        if not hasattr(self, '_yerr_transformed'):
            try:
                noise_scale = np.minimum(1e-4, self._yerr_transformed.min()/10)
            except AttributeError:
                noise_scale = 1e-4*self._ydata_transformed.std()
            # noise_prior = gpytorch.priors.HalfCauchyPrior(noise_scale)
            noise_prior = gpytorch.priors.LogNormalPrior(torch.log(noise_scale),
                                                         noise_scale)
            self._model_pars['noise']['module'].register_prior("noise_prior",
                                                               noise_prior,
                                                               'noise')
        with contextlib.suppress(RuntimeError):
            mean_prior = gpytorch.priors.NormalPrior(self._ydata_transformed.mean(),
                                                     self._ydata_transformed.std()/10)
            for key in self._model_pars:
                if 'mean_module.constant' in key:
                    self._model_pars[key]['module'].register_prior("mean_prior",
                                                                   mean_prior,
                                                                   'constant')
        # we use a lognormal prior for the means, because we want to make sure
        # that the means are positive, but we don't want to restrict them to
        # be close to zero. In fact, we want to penalise both very high and very low
        # frequencies, so we use a lognormal prior with mu = 0 and sigma = 1
        mixture_means_prior = gpytorch.priors.LogNormalPrior(0, 1)  # /self._xdata_transformed.max())  # noqa: E501
        self._model_pars['mixture_means']['module'].register_prior("mixture_means_prior",
                                                                   mixture_means_prior,
                                                                   'mixture_means')

        # now we need a prior for the mixture scales
        # we want to penalise very large scales, so we use a half-cauchy prior
        # with a scale of 1/10 of the maximum frequency
        # mixture_scales_prior = gpytorch.priors.HalfCauchyPrior(1/self._xdata_transformed.max())  # noqa: E501
        mixture_scales_prior = gpytorch.priors.LogNormalPrior(0, 1) #1/self._xdata_transformed.max())  # noqa: E501
        self._model_pars['mixture_scales']['module'].register_prior("mixture_scales_prior",
                                                                    mixture_scales_prior,
                                                                    'mixture_scales')
        # we use a LogNormal prior for the mixture weights, because we want to
        # make sure that they are positive (but never zero) and we don't want
        # to restrict them to be close to zero. In fact, we want to penalise
        # both very high and very low weights, so we use a LogNormal prior
        # with a scale of 1/10 of the maximum frequency
        mixture_weights_prior = gpytorch.priors.LogNormalPrior(0, 1)  # 1/self._xdata_transformed.max())  # noqa: E501
        self._model_pars['mixture_weights']['module'].register_prior("mixture_weights_prior",
                                                                     mixture_weights_prior,
                                                                     'mixture_weights')

        # need a more general way to assign default priors to everything, but for now
        # let's see if this works!
        self.__PRIORS_SET = True

    def set_default_constraints(self, **kwargs):
        '''Set the default constraints for the model and likelihood parameters

        The default constraints are as follows:
            - All parameters are constrained to be positive, except the mean of
            the GP, which is constrained to be in the range of the y-data (a correction
            will be needed if the data are censored!)
            - The noise is constrained to be less than the standard deviation of
            the y-data, and greater than either 1e-4 or 1/10 of the smallest
            uncertainty on the y-data, if uncertainties are given, or 1e-4
            times the standard deviation of the y data.
            - The mixture means greater than the frequency corresponding to
            the separation between the earliest and latest points in the data
            and less than the frequency corresponding to the separation between
            the two closest data points (should be updated to account for the
            window function and whatever we're really sensitive to)
            - The mixture scales and weights are left with their default
            constraints as defined in GPyTorch.

        Parameters
        ----------
        **kwargs : dict, optional
            Any keyword arguments to be passed to the Constraint constructors.
        '''
        try:
            noise_min = np.minimum(1e-4, self._yerr_transformed.min()/10)
        except AttributeError:
            noise_min = 1e-4*self._ydata_transformed.std()
        noise_max = self._ydata_transformed.std()  # for a non-periodic source,
                                                # the noise should be less than
                                                # the standard deviation
        noise_constraint = Interval(noise_min, noise_max)
        self._model_pars['noise']['module'].register_constraint("raw_noise",
                                                                noise_constraint)
        with contextlib.suppress(RuntimeError):
            mean_const_constraint = Interval(self._ydata_transformed.min(),
                                             self._ydata_transformed.max())
            for key in self._model_pars:
                if 'mean_module.constant' in key:
                    self._model_pars[key]['module'].register_constraint("raw_constant",
                                                                        mean_const_constraint)
        # this should correspond to the longest frequency entirely
        # contained in the dataset:
        mixture_means_constraint = GreaterThan(1/self._xdata_transformed.max())
        self._model_pars['mixture_means']['module'].register_constraint("raw_mixture_means",
                                                                        mixture_means_constraint)

        # to-do - check if constraints on mixture scales are useful!

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

    def init_hypers_from_LombScargle(self, **kwargs):
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

    def _eval(self):
        try:
            self.model.eval()
            self.likelihood.eval()
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
        likelihood : string, None or instance of
                        gpytorch.likelihoods.likelihood.Likelihood or Constraint,
                        optional
            If likelihood is passed, it will be passed along to `set_likelihood()`
            and used to set the likelihood function for the model. For details, see
            the documentation for `set_likelihood()`.
        num_mixtures : int, optional
            The number of mixtures to use in the spectral mixture kernel, by
            default 4. If None, a default value will be used. This value
            is passed to the constructor for the model if a string is passed
            as the model argument.
        guess : dict, optional
            A dictionary of the hyperparameters to use for the model and
            likelihood. The keys should be the names of the parameters, and the
            values should be Tensors containing the values of the parameters.
            If None, no hyperparameters will be set. If a hyperparameter is
            passed for a parameter that is not a model or likelihood
            parameter, it will be ignored.
        grid_size : int, optional
            The number of points to use in the grid for the KISS-GP models,
            by default 2000.
        cuda : bool, optional
            Whether to use CUDA, by default False.
        training_iter : int, optional
            The number of iterations to use for training, by default 300.
        max_cg_iterations : int, optional
            The maximum number of conjugate gradient iterations to use, by
            default None. If None, gpytorch.settings.max_cg_iterations will
            be used.
        optim : str or torch.optim.Optimizer, optional
            The optimizer to use for training, by default "AdamW". If a string,
            it must be one of the following:
                'AdamW': torch.optim.AdamW
                'Adam': torch.optim.Adam
                'SGD': torch.optim.SGD
            Otherwise, it must be an instance of torch.optim.Optimizer.
        miniter : int, optional
            The minimum number of iterations to use for training, by default
            None. If None, training_iter will be used.
        stop : float, optional
            The stopping criterion for the training, by default 1e-5.
        lr : float, optional
            The learning rate to use for the optimizer, by default 0.1.
        stopavg : int, optional
            The number of iterations to use for the stopping criterion, by
            default 30.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the model constructor,
            likelihood constructor, or the optimizer.

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if not hasattr(self, 'likelihood'):
            self.set_likelihood(likelihood, **kwargs)
        elif not self.__SET_LIKELIHOOD_CALLED and likelihood is None:
            # if no likelihood is passed, we only want to set the likelihood
            # if it hasn't already been set
            self.set_likelihood(likelihood, **kwargs)
        elif likelihood is not None:
            self.set_likelihood(likelihood, **kwargs)
        # if likelihood is None and not hasattr(self, 'likelihood'):
        #     raise ValueError("""You must provide a likelihood function""")
        # elif likelihood is not None:
        #     self.set_likelihood(likelihood, **kwargs)

        if model is None and not hasattr(self, 'model'):
            raise ValueError("""You must provide a model""")
        elif model is not None:
            self.set_model(model, self.likelihood,
                           num_mixtures=num_mixtures, **kwargs)

        if not self.__CONTRAINTS_SET:
            self.set_default_constraints()

        if cuda:
            self.cuda()

        if guess is not None:
            # self.model.initialize(**guess)
            self.set_hypers(guess)

        if miniter is None:
            miniter = training_iter

        if max_cg_iterations is None:
            max_cg_iterations = 10000

        # Next we probably want to report some setup info
        # later...

        # Train the model
        # self.model.train()
        # self.likelihood.train()

        # set training mode:
        self._train()

        # for param_name, param in self.model.named_parameters():
        #    print(f'Parameter name: {param_name:42} value = {param.data}')
        self.print_parameters()

        # Now actually call the trainer!
        with gpytorch.settings.max_cg_iterations(max_cg_iterations):
            self.results = train(self,
                                 maxiter=training_iter,
                                 miniter=miniter,
                                 stop=stop, lr=lr,
                                 optim=optim, stopavg=stopavg)
        self.__FITTED_MAP = True

        return self.results

    def mcmc(self, sampler=None, num_samples=500,
             warmup_steps=100, num_chains=1,
             disable_progbar=False,
             max_cg_iterations=None,
             **kwargs):
        '''Run an MCMC sampler on the model

        This function runs an MCMC sampler on the model, using the sampler
        specified in the `sampler` attribute. The results are stored in the
        `mcmc_results` attribute.

        Parameters
        ----------
        sampler : str or MCMC, optional
            The name of the sampler to use. If None, pyro.infer.mcmc.NUTS will
            be used. If a string, it must be one of the following:
                'NUTS': pyro.infer.mcmc.NUTS
                'HMC': pyro.infer.mcmc.HMC
            Otherwise, it must be an instance of pyro.infer.mcmc.MCMC.
        num_samples : int, optional
            The number of samples to draw from the posterior, by default 500.
        warmup_steps : int, optional
            The number of warmup steps to use, by default 100.
        disable_progbar : bool, optional
            Whether to disable the progress bar, by default False.
        **kwargs : dict, optional

        Returns
        -------
        mcmc_results : dict
            A dictionary containing the results of the MCMC sampling. The
            keys are the names of the parameters, and the values are the
            samples of the parameters.
        '''
        if sampler is None:
            sampler = NUTS
        elif isinstance(sampler, str):
            if sampler == 'NUTS':
                sampler = NUTS
            elif sampler == 'HMC':
                sampler = HMC
            else:
                raise ValueError("sampler must be one of 'NUTS' or 'HMC'")
        elif not isinstance(sampler, MCMC):
            raise TypeError("sampler must be either None, a string, or an instance of pyro.infer.mcmc.MCMC")  # noqa: E501

        # we need to make sure that the model is in train mode
        # self._train()
        # self._eval()

        if not self.__PRIORS_SET:
            self.set_default_priors()

        if max_cg_iterations is None:
            max_cg_iterations = 10000

        model = self.model

        def pyro_model(x, y):
            with (gpytorch.settings.fast_computations(False, False, False),
                  gpytorch.settings.max_cg_iterations(max_cg_iterations)):
                sampled_model = model.pyro_sample_from_prior()  # .detatch()
                output = sampled_model.likelihood(sampled_model(x))  # .detatch()
                pyro.sample("obs", output, obs=y)
            return y

        self.num_samples = num_samples

        nuts_kernel = sampler(pyro_model)
        self.mcmc_run = MCMC(nuts_kernel,
                             num_samples=num_samples,
                             warmup_steps=warmup_steps,
                             num_chains=num_chains,
                             disable_progbar=disable_progbar)
        import linear_operator.utils.errors as linear
        try:
            self.mcmc_run.run(self._xdata_transformed, self._ydata_transformed)
        except linear.NanError as e:
            print("NaNError encountered, returning None")
            print(list(model.named_parameters()))
            self.print_parameters()
            raise e

        self.__FITTED_MCMC = True

        # self.mcmc_run.summary(prob=0.683)
        self.inference_data = az.from_pyro(self.mcmc_run)
        samples = self.mcmc_run.get_samples()
        self.model.pyro_load_from_samples(samples)

        self.post = self.inference_data.posterior
        transformed_mcmc_periods = 1/self.post['covar_module.mixture_means_prior']
        raw_mcmc_periods = self.xtransform.inverse(torch.as_tensor(transformed_mcmc_periods.to_numpy()), shift=False)  # noqa: E501
        raw_mcmc_frequencies = 1/raw_mcmc_periods
        transformed_mcmc_period_scales = 1/(2*torch.pi*self.post['covar_module.mixture_scales_prior'])  # noqa: E501
        raw_mcmc_period_scales = self.xtransform.inverse(torch.as_tensor(transformed_mcmc_period_scales.to_numpy()), shift=False)  # noqa: E501
        raw_mcmc_frequency_scales = 1/(2*torch.pi*raw_mcmc_period_scales)

        self.inference_data.posterior['transformed_periods'] = transformed_mcmc_periods
        self.inference_data.posterior['raw_periods'] = xr.DataArray(raw_mcmc_periods.reshape(self.post['covar_module.mixture_means_prior'].shape),  # noqa: E501
                                                                    coords=self.inference_data.posterior['covar_module.mixture_means_prior'].indexes)  # noqa: E501
        self.inference_data.posterior['raw_frequencies'] = xr.DataArray(raw_mcmc_frequencies.reshape(self.post['covar_module.mixture_means_prior'].shape),  # noqa: E501
                                                                        coords=self.inference_data.posterior['covar_module.mixture_means_prior'].indexes) # noqa: E501
        self.inference_data.posterior['transformed_period_scales'] = transformed_mcmc_period_scales  # noqa: E501
        self.inference_data.posterior['raw_period_scales'] = xr.DataArray(raw_mcmc_period_scales.reshape(self.post['covar_module.mixture_scales_prior'].shape),  # noqa: E501
                                                                          coords=self.inference_data.posterior['covar_module.mixture_scales_prior'].indexes)  # noqa: E501
        self.inference_data.posterior['raw_frequency_scales'] = xr.DataArray(raw_mcmc_frequency_scales.reshape(self.post['covar_module.mixture_scales_prior'].shape),  # noqa: E501
                                                                             coords=self.inference_data.posterior['covar_module.mixture_scales_prior'].indexes)  # noqa: E501

        # self.mcmc_results = mcmc(self, sampler, **kwargs)

    def summary(self, prob=0.683, use_arviz=True,
                var_names=None, filter_vars='like',
                stat_focus='median', **kwargs):
        '''Print a summary of the results of the MCMC sampling

        Parameters
        ----------
        prob : float, optional
            The probability to use for the credible intervals, by default
            0.683.
        use_arviz : bool, optional
            Whether to use arviz to print the summary, by default True.
        var_names : list, optional
            A list of the names of the variables to include in the summary. If
            None, the variables for the mean function and the covariance
            function will be included, by default None.
        filter_vars : str, optional
            A string specifying how to filter the variables, based on
            `arviz.summary`. If None, the default behaviour of `arviz.summary`
            will be used, by default 'like'.
        stat_focus : str, optional
            A string specifying which statistic to focus on, based on
            `arviz.summary`. If None, the default behaviour of `arviz.summary`
            will be used ('mean'), by default 'median'.
        '''
        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        if var_names is None:
            var_names=['mean_module', 'covar_module.mixture_weights', 'raw']
        elif var_names == 'all':
            var_names = None
        if stat_focus is None:
            stat_focus = 'mean'
        if use_arviz:
            self.summary = az.summary(self.inference_data, round_to=2,
                                      hdi_prob=prob,
                                      var_names=var_names,
                                      filter_vars=filter_vars,
                                      stat_focus=stat_focus,
                                      **kwargs)
        # self.mcmc_run.summary(prob=prob)
        self.diagnostics = self.mcmc_run.diagnostics()
        # figure out how to filter these before printing!
        # print(self.diagnostics)
        return self.summary

    def plot_corner(self, kind='scatter', var_names=None, filter_vars='like',
                    marginals=True, point_estimate='median', **kwargs):
        '''Plot a corner plot of the results of the MCMC sampling

        Parameters
        ----------
        kind : str, optional
            The kind of plot to use, based on `arviz.plot_pair`. If None, the
            default behaviour of `arviz.plot_pair` will be used, by default
            'scatter'. Other options are 'kde' and 'hexbin'.
        var_names : list, optional
            A list of the names of the variables to include in the corner plot.
            If None, the variables for the mean function and the covariance
            function will be included, by default None.
        filter_vars : str, optional
            A string specifying how to filter the variables, based on
            `arviz.plot_pair`. If None, the default behaviour of
            `arviz.plot_pair` will be used, by default 'like'.
        marginals : bool, optional
            Whether to include the marginal distributions, by default True.
        point_estimate : str, optional
            The point estimate to plot, based on `arviz.plot_pair`. If None,
            no point estimate will be plotted, by default 'median'.
        '''
        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        if var_names is None:
            var_names = ['mean_module', 'covar_module.mixture_weights', 'raw']
        if point_estimate is None:
            point_estimate = 'median'
        az.plot_pair(self.inference_data, kind=kind,
                     var_names=var_names, filter_vars=filter_vars,
                     marginals=marginals, point_estimate=point_estimate,
                     **kwargs)

    def plot_trace(self, var_names=None, filter_vars='like',
                   figsize=None, **kwargs):
        '''Plot a trace plot of the results of the MCMC sampling

        Parameters
        ----------
        var_names : list, optional
            A list of the names of the variables to include in the trace plot.
            If None, the variables for the mean function and the covariance
            function will be included, by default None.
        '''
        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        if var_names is None:
            # we carefully choose the default variables to plot
            # we want to plot all parameters relating to the mean function
            # but we don't want to plot all the covariance parameters
            # because those are in the transformed space. Instead, we want
            # to plot the extra parameters we have created, which are in the
            # raw space, as well as the periods and the mixture weights
            var_names = ['mean_module', 'covar_module.mixture_weights', 'raw']  # ['mean_module', 'covar_module']  # noqa: E501
        az.plot_trace(self.inference_data, var_names=var_names,
                      filter_vars=filter_vars,
                      figsize=figsize,
                      **kwargs)

    def print_periods(self):
        if self.ndim == 1:
            for i in range(len(self.model.covar_module.mixture_means)):
                if self.xtransform is None:
                    p = 1/self.model.covar_module.mixture_means[i]
                else:
                    p = self.xtransform.inverse(1/self.model.covar_module.mixture_means[i],  # noqa: E501
                                                shift=False).detach().numpy()[0]
                print(f"Period {i}: "
                      f"{p}"
                      f" weight: {self.model.covar_module.mixture_weights[i]}")
        elif self.ndim == 2:
            for i in range(len(self.model.covar_module.mixture_means[:, 0])):
                if self.xtransform is None:
                    p = 1/self.model.covar_module.mixture_means[i, 0]
                else:
                    p = self.xtransform.inverse(1/self.model.covar_module.mixture_means[i, 0],  # noqa: E501
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
                    p = self.xtransform.inverse(1/self.model.sci_kernel.mixture_means[i],  # noqa: E501
                                                shift=False).detach().numpy()[0]
                    scales.append(self.xtransform.inverse(1/(2*torch.pi*self.model.sci_kernel.mixture_scales[i]),
                                                          shift=False).detach().numpy()[0])
                periods.append(p)
                weights.append(self.model.sci_kernel.mixture_weights[i])
        elif self.ndim == 2:
            for i in range(len(self.model.sci_kernel.mixture_means[:, 0])):
                if self.xtransform is None:
                    p = 1/self.model.sci_kernel.mixture_means[i, 0]
                    scales.append(1/(2*torch.pi*self.model.sci_kernel.mixture_scales[i, 0]))  # noqa: E501
                else:
                    p = self.xtransform.inverse(1/self.model.sci_kernel.mixture_means[i, 0],  # noqa: E501
                                                shift=False).detach().numpy()[0, 0]
                    scales.append(self.xtransform.inverse(1/(2*torch.pi*self.model.sci_kernel.mixture_scales[i, 0]),  # noqa: E501
                                                shift=False).detach().numpy()[0, 0])
                periods.append(p)
                weights.append(self.model.sci_kernel.mixture_weights[i])

        return torch.as_tensor(periods), torch.as_tensor(weights), torch.as_tensor(scales)  # noqa: E501

    def get_parameters(self, raw=False, transform=True):
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
        pars_to_transform = {'x': ['mixture_means', 'mixture_scales'],
                             'y': ['noise', 'mean_module']}
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
                if (any(p in pn for p in pars_to_transform['x'])
                   and transform
                   and self.xtransform is not None):
                    d = 1/self.xtransform.inverse(1/tmp.data, shift=False)
                elif (any(p in pn for p in pars_to_transform['y'])
                      and transform
                      and self.ytransform is not None):
                    d = self.ytransform.inverse(tmp.data)
                else:
                    d = tmp.data
                pars[pn] = d
            else:
                # Either we actually want the raw values, or it's not a
                # constrained parameter
                if (any(p in param_name for p in pars_to_transform['x'])
                   and transform
                   and self.xtransform is not None):
                    d = 1/self.xtransform.inverse(1/param.data, shift=False)
                elif (any(p in param_name for p in pars_to_transform['y'])
                      and transform
                      and self.ytransform is not None):
                    d = self.ytransform.inverse(param.data)
                else:
                    d = param.data
                pars[param_name] = d
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

    def plot_psd(self, freq=None, means=None, scales=None, weights=None,
                 show=True, raw=False, log=(True, False),
                 truncate_psd=True, logpsd=False,
                 mcmc_samples=False, **kwargs):
        '''Plot the power spectral density of the model

        Parameters
        ----------
        freq : array_like, optional
            The frequencies at which to compute the PSD, by default None. If
            None, the frequencies will be computed automatically.
        means : array_like, optional
            The means of the gaussians in the spectral mixture kernel, by
            default None. If None, the means from the model will be used.
        scales : array_like, optional
            The scales of the gaussians in the spectral mixture kernel, by
            default None. If None, the scales from the model will be used.
        weights : array_like, optional
            The weights of the gaussians in the spectral mixture kernel, by
            default None. If None, the weights from the model will be used.
        show : bool, optional
            Whether to show the plot, by default True.
        raw : bool, optional
            If True, the PSD will be computed in the space that the model was
            trained in, by default False. If False, the PSD will be computed
            in the original space of the data.
        log : tuple, optional
            A tuple of two booleans, indicating whether to plot the x-axis and
            y-axis on a log scale, respectively, by default (True, False).
        truncate_psd : float or bool, optional
            If not False, the PSD will be truncated at this value, by default
            True. This is useful for speeding up plotting when the frequency
            range is large. If logpsd is True, this value should be given in
            (natural) log space. If truncate_psd is True, the PSD will be
            truncated at 1e-6 times the maximum PSD for logpsd=False, and 1e-15
            of the maximum PSD (i.e. max(ln(psd)) - 34.5388) for logpsd=True.
        logpsd : bool, optional
            If True, the PSD will be plotted on a log scale, by default False.
            If True, truncate_psd must be given in (natural) log space.
        mcmc_samples : bool, optional
            If True, many sample PSDs will be plotted using the MCMC samples,
            by default False. This will only work if the model has been fitted
            using MCMC.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
            The figure and axes objects of the plot.
        '''

        if freq is None:
            if self.ndim == 1:
                if raw:
                    # our step size only needs to be small enough to resolve
                    # the width of the narrowest gaussian
                    step = self.model.sci_kernel.mixture_scales.min()/5
                    # this isn't really the correct way to do this, but it will
                    # do for now
                    diffs = self._xdata_transformed.sort().values[1:] - self._xdata_transformed.sort().values[:-1]  # noqa: E501
                    mindelta = (diffs[diffs>0]).min().item()
                    freq = torch.arange(1/(self._xdata_transformed.max() - self._xdata_transformed.min()).item(),  # noqa: E501
                                        1/(mindelta),
                                        step.item())
                else:
                    # we have to transform the step size to the original space
                    # to get the correct frequency range
                    step = 1/self.xtransform.inverse(1/(self.model.sci_kernel.mixture_scales.min()/5),  # noqa: E501
                                                     shift=False)
                    # this isn't really the correct way to do this, but it will
                    # do for now
                    diffs = self._xdata_raw.sort().values[1:] - self._xdata_raw.sort().values[:-1]  # noqa: E501
                    mindelta = (diffs[diffs>0]).min().item()
                    print(step, mindelta, 1/(mindelta/2))

                    # we want to sample a set of frequencies that are spaced
                    # in the range covered by the gaussian mixture, but we
                    # want to sample them densely enough to resolve the
                    # narrowest gaussian so we want a minimum frequency

                    freq = torch.arange(1/(self._xdata_raw.max()
                                           - self._xdata_raw.min()
                                           ).item(),
                                        1/(mindelta/2),
                                        step.item())
                    print(freq.shape)

            elif self.ndim == 2:
                raise NotImplementedError("""Plotting PSDs in more than 1 dimension is
                                          not currently supported. Please get in touch
                                          if you need this functionality!
                """)
            else:
                raise NotImplementedError("""Plotting PSDs in more than 2 dimensions
                                          is not currently supported. Please get in
                                          touch if you need this functionality!
                """)

        if mcmc_samples:
            fig, ax = self._plot_psd_mcmc(freq, means=means, scales=scales,
                                          weights=weights, show=show, raw=raw,
                                          log=log, truncate_psd=truncate_psd,
                                          logpsd=logpsd, **kwargs)
            return fig, ax
        # Computing the psd for frequencies f
        psd = self.compute_psd(freq, means=means, scales=scales,
                               weights=weights, raw=raw, log=logpsd, **kwargs)

        if truncate_psd is True:
            if logpsd:
                freq = freq[psd > psd.max()-34.5388]
                psd = psd[psd > psd.max()-34.5388]
            else:
                freq = freq[psd > 1e-6*psd.max()]
                psd = psd[psd > 1e-6*psd.max()]
        elif truncate_psd:
            freq = freq[psd > truncate_psd]
            psd = psd[psd > truncate_psd]

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # plotting psd
        ax.plot(freq, psd)
        if log[0]:
            ax.set_xscale('log')
        if log[1] and not logpsd:  # we don't need to double-log the Y axis (I hope!)
            ax.set_yscale('log')
        if show:
            plt.show()
        else:
            return fig, ax

    def _plot_psd_mcmc(self, freq, means=None, scales=None, weights=None,
                       show=True, raw=False, log=(True, True),
                       truncate_psd=True, logpsd=False, n_samples_to_plot=25,
                       **kwargs):
        '''Plot the power spectral density of the model using MCMC samples

        Parameters
        ----------
        freq : array_like
            The frequencies at which to compute the PSD
        means : array_like, optional
            The means of the gaussians in the spectral mixture kernel, by
            default None. If None, the means from the model will be used.
        scales : array_like, optional
            The scales of the gaussians in the spectral mixture kernel, by
            default None. If None, the scales from the model will be used.
        weights : array_like, optional
            The weights of the gaussians in the spectral mixture kernel, by
            default None. If None, the weights from the model will be used.
        show : bool, optional
            Whether to show the plot, by default True.
        raw : bool, optional
            If True, the PSD will be computed in the space that the model was
            trained in, by default False. If False, the PSD will be computed
            in the original space of the data.
        log : tuple, optional
            A tuple of two booleans, indicating whether to plot the x-axis and
            y-axis on a log scale, respectively, by default (True, False).
        truncate_psd : float or bool, optional
            If not False, the PSD will be truncated at this value, by default
            True. This is useful for speeding up plotting when the frequency
            range is large. If logpsd is True, this value should be given in
            (natural) log space. If truncate_psd is True, the PSD will be
            truncated at 1e-6 times the maximum PSD for logpsd=False, and 1e-15
            of the maximum PSD (i.e. max(ln(psd)) - 34.5388) for logpsd=True.
        logpsd : bool, optional
            If True, the PSD will be plotted on a log scale, by default False.
            If True, truncate_psd must be given in (natural) log space.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
            The figure and axes objects of the plot.
        '''

        if not self.__FITTED_MCMC:
            raise RuntimeError("You must first run the MCMC sampler")
        n_samples = min(self.num_samples, n_samples_to_plot)
        if means is None:
            # this approach is slightly bugged - if more than one chain is used,
            # it will only draw samples from the first chain
            # will change this to generate random indices instead
            # at some point!
            # right now, this will end up having shape (1, chains, samples) (I thinkk)
            means = torch.as_tensor(self.inference_data.posterior['raw_frequencies'].values).squeeze()[:n_samples].unsqueeze(0)  # noqa: E501
            # print(means.shape)
            # print(freq.shape)
        if scales is None:
            scales = torch.as_tensor(self.inference_data.posterior['raw_frequency_scales'].values).squeeze()[:n_samples].unsqueeze(0)  # .unsqueeze(-1)  # noqa: E501
        if weights is None:
            weights = torch.as_tensor(self.inference_data.posterior['covar_module.mixture_weights_prior'].values).squeeze()[:n_samples].unsqueeze(0)  # .unsqueeze(-1)  # noqa: E501

        # computing the psd for all samples simultaneously is very expensive,
        # so we're just going to loop over them and plot them individually
        # this means we have to do things in a differnet order to the other
        # plotting routines

        # Initialize plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for i in range(n_samples):

            # Computing the psd for frequencies f
            psd = self.compute_psd(freq,
                                   means=means[..., i],
                                   scales=scales[..., i],
                                   weights=weights[..., i],
                                   raw=raw, log=logpsd,
                                   **kwargs)

            if truncate_psd is True:
                mask = psd > psd.max()-34.5388 if logpsd else psd > 1e-6*psd.max()
            elif truncate_psd:
                mask = psd > truncate_psd
            # now we can plot it:
            ax.plot(freq[mask], psd[mask], alpha=0.2, color='b')

        # final plot formatting
        if log[0]:
            ax.set_xscale('log')
        if log[1] and not logpsd:  # we don't need to double-log the Y axis (I hope!)
            ax.set_yscale('log')
        if show:
            plt.show()
        return fig, ax

    def compute_psd(self, freq, means=None, scales=None, weights=None,
                    raw=False, log=False, debug=False, **kwargs):
        '''Compute the power spectral density for the model

        Parameters
        ----------
        freq : array_like
            The frequencies at which to compute the PSD
        means : array_like, optional
            The means of the gaussians in the spectral mixture kernel, by
            default None. If None, the means from the model will be used.
        scales : array_like, optional
            The scales of the gaussians in the spectral mixture kernel, by
            default None. If None, the scales from the model will be used.
        weights : array_like, optional
            The weights of the gaussians in the spectral mixture kernel, by
            default None. If None, the weights from the model will be used.
        raw : bool, optional
            If True, the PSD will be computed in the space that the model was
            trained in, by default False. If False, the PSD will be computed
            in the original space of the data.
        **kwargs : dict, optional
            Any other keyword arguments to be passed.

        Returns
        -------
        psd : array_like
            The power spectral density of the model at the frequencies given
            by freq.
        '''
        if means is None:
            means = self.model.sci_kernel.mixture_means
            # now apply the transform too!
            if self.xtransform is not None and not raw:
                # there's probably an easier way to do this than converting to
                # a period and back, but this will do for now
                means = 1/self.xtransform.inverse(1/means, shift=False).detach()
        if scales is None:
            scales = self.model.sci_kernel.mixture_scales
            # now apply the transform too!
            if self.xtransform is not None and not raw:
                scales = 1/(2*np.pi*self.xtransform.inverse(1/(2*torch.pi*scales),
                                                            shift=False
                                                            ).detach()
                            )
        if weights is None:
            weights = self.model.sci_kernel.mixture_weights.detach()  # .numpy()

        from torch.distributions import Normal as torchnorm
        # Computing the psd for frequencies f
        if debug:
            print(freq.shape, means.shape, scales.shape, weights.shape)
        norm = torchnorm(means, scales)
        if debug:
            print(norm)
        psd = norm.log_prob(freq.unsqueeze(-1))
        if debug:
            print(psd.shape)
        try:
            psd = torch.logsumexp(torch.log(weights) + psd, dim=-1).squeeze()
        except RuntimeError:  # logsumexp tries to allocate a large array and
            # then do the summation so let's do it in a loop instead and see
            # if that avoids the problem
            logweights = torch.log(weights)
            for i in range(means.shape[-2]):
                if i == 0:
                    psd = logweights[..., i] + psd[..., i]
                else:
                    psd += logweights[..., i] + psd[..., i]
            psd = logweights + psd
            for i in range(len(weights)):
                if i == 0:
                    psd = weights[i] * torch.exp(psd[i])
                else:
                    psd += weights[i] * torch.exp(psd[i])
        if debug:
            print(psd.shape)
        if not log:
            psd = psd.exp().detach().numpy()
        return psd

    def plot(self, ylim=None, show=True,
             mcmc_samples=False, **kwargs):
        '''Plot the model and data

        Parameters
        ----------
        ylim : list, optional
            The y-limits of the plot, by default None. If None, the y-limits
            will be set automatically.
        show : bool, optional
            Whether to show the plot, by default True.
        mcmc_samples : bool, optional
            Whether to plot the samples from the MCMC run, by default False.
            This will only work if the MCMC sampler has been run.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure object of the plot.
        '''
        if ylim is None:
            ylim = [-3, 3]
        if mcmc_samples:
            if self.__FITTED_MCMC:
                return self._plot_mcmc(ylim=ylim, show=show, **kwargs)
            else:
                raise RuntimeError("You must first run the MCMC sampler")
        elif not self.__FITTED_MAP:
            raise RuntimeError("You must first fit the GP")
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get into evaluation (predictive posterior) mode
            # self.model.eval()
            # self.likelihood.eval()

            self._eval()

            # Importing raw x and y training data from xdata and
            # ydata functions
            if self.ndim == 1:
                x_raw = self.xdata
            elif self.ndim == 2:
                x_raw = self.xdata[:, 0]
            # y_raw = self.ydata

            # creating array of 10000 test points across the range of the data
            x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), 10000)

            if self.ndim == 1:
                fig = self._plot_1d(x_fine_raw, ylim=ylim,
                                    show=show, **kwargs)
            elif self.ndim == 2:
                fig = self._plot_2d(x_fine_raw, ylim=ylim,
                                    show=show, **kwargs)
            else:
                raise NotImplementedError("""
                Plotting models and data in more than 2 dimensions is not
                currently supported. Please get in touch if you need this
                functionality!
                """)
        return fig

    def _plot_mcmc(self, ylim=None, show=False,
                   n_samples_to_plot=25,
                   **kwargs):
        '''Plot the model and data, including samples from the MCMC run

        Parameters
        ----------
        ylim : list, optional
            The y-limits of the plot, by default None. If None, the y-limits
            will be set automatically.
        show : bool, optional
            Whether to show the plot, by default True.
        n_samples_to_plot : int, optional
            The number of samples to plot, by default 25.
        **kwargs : dict, optional
            Any other keyword arguments to be passed to the plotting routine.

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure object of the plot.
        '''
        # Get into evaluation (predictive posterior) mode
        self._eval()

        if self.ndim > 1:
            raise NotImplementedError("""
            Plotting models and data in more than 1 dimension is not
            currently supported. Please get in touch if you need this
            functionality!
            """)
        # Importing raw x and y training data from xdata and
        # ydata functions
        x_raw = self.xdata
        # y_raw = self.ydata

        # creating array of 10000 test points across the range of the data
        x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), 10000).unsqueeze(-1)

        # transforming the x_fine_raw data to the space that the GP was
        # trained in (so it can predict)
        if self.xtransform is None:
            self.x_fine_transformed = x_fine_raw
        elif isinstance(self.xtransform, Transformer):
            self.x_fine_transformed = self.xtransform.transform(x_fine_raw)

        self.expanded_test_x = self.x_fine_transformed.unsqueeze(0).repeat(self.num_samples, 1, 1)  # .unsqueeze(0)  # noqa: E501
        print(self.x_fine_transformed.shape)
        print(self.expanded_test_x.shape)
        output = self.model(self.expanded_test_x)
        with torch.no_grad():
            f, ax = plt.subplots(1, 1, figsize=(8, 6))
            # Plot training data as black stars
            ax.plot(self.xdata.numpy(), self.ydata.numpy(), 'k*')
            for i in range(min(n_samples_to_plot, self.num_samples)):
                # Plot predictive samples as colored lines
                ax.plot(x_fine_raw.numpy(), output[i].sample().numpy(), 'b', alpha=0.2)

            ax.legend(['Observed Data', 'Sample means'])
            if ylim is not None:
                ax.set_ylim(ylim)
            if show:
                plt.show()
        return f

    def _plot_1d(self, x_fine_raw, ylim=None, show=False,
                 save=True, **kwargs):
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
                 save=True, **kwargs):
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


    def to_table(self):
        """Create an astropy table with the results.

        Parameters
        ----------
        none

        Returns
        -------
        tab_results : astropy.table.Table
            Astropy table with the results.
        """
        from astropy.table import Table
        t = Table()
        t['x'] = [np.asarray(self.xdata)]
        t['y'] = [np.asarray(self.ydata)]
        if hasattr(self, 'yerr'):
            t['yerr'] = [np.asarray(self.yerr)]
        if self.__FITTED_MCMC or self.__FITTED_MAP:
            # These outputs can only be produced if a fit has been run.
            periods, weights, scales = self.get_periods()
            t['period'] = [np.asarray(periods)]
            try:
                t['weights'] = [np.asarray(weights)]
            except RuntimeError:
                t['weights'] = [torch.as_tensor(weights).detach().numpy()]
            try:
                t['scales'] = [np.asarray(scales)]
            except RuntimeError:
                t['scales'] = [torch.as_tensor(scales).detach().numpy()]
            for key, value in self.results.items():
                try:
                    t[key] = [np.asarray(value)]
                except RuntimeError:
                    t[key] = [torch.as_tensor(value).detach().numpy()]
            if self.__FITTED_MAP:
                # Loss isn't relevant for MCMC, I think
                t['loss'] = [np.asarray(self.results['loss'])]
            # Now we want the model predictions for the input times:
            if self.__FITTED_MAP:
                self._eval()
                with torch.no_grad():
                    observed_pred = self.likelihood(self.model(self._xdata_transformed))
                    t['y_pred_mean_obs'] = [np.asarray(observed_pred.mean)]
                    t['y_pred_lower_obs'] = [np.asarray(observed_pred.confidence_region()[0])]  # noqa: E501
                    t['y_pred_upper_obs'] = [np.asarray(observed_pred.confidence_region()[1])]   # noqa: E501

                    if self.ndim == 1:
                        x_raw = self.xdata
                    elif self.ndim == 2:
                        x_raw = self.xdata[:, 0]
                    # y_raw = self.ydata

                    # creating array of 10000 test points across the range of the data
                    x_fine_raw = torch.linspace(x_raw.min(), x_raw.max(), 10000)
                    if self.xtransform is None:
                        x_fine_transformed = x_fine_raw
                    elif isinstance(self.xtransform, Transformer):
                        x_fine_transformed = self.xtransform.transform(x_fine_raw)

                    # Make predictions
                    observed_pred = self.likelihood(self.model(x_fine_transformed))
                    t['x_fine'] = [np.asarray(x_fine_raw)]
                    t['y_pred_mean'] = [np.asarray(observed_pred.mean)]
                    t['y_pred_lower'] = [np.asarray(observed_pred.confidence_region()[0])]  # noqa: E501
                    t['y_pred_upper'] = [np.asarray(observed_pred.confidence_region()[1])]   # noqa: E501
            elif self.__FITTED_MCMC:
                raise NotImplementedError("MCMC predictions not yet implemented")
                # with torch.no_grad():

        return t

    def write_votable(self, filename):
        """Write the results to a votable file.

        Parameters
        ----------
        filename : str
            The name of the file to write to.
        """
        t = self.to_table()
        t.write(filename, format='votable', overwrite=True)
