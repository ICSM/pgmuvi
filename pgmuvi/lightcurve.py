import numpy as np
import torch
import gpytorch
import pandas as pd
from gps import SpectralMixtureGPModel as SMG
from gps import SpectralMixtureKISSGPModel as SMKG
from gps import TwoDSpectralMixtureGPModel as TMG
import matplotlib.pyplot as plt
from trainers import train
from gpytorch.constraints import Interval
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist

def zscale(data, dim=0):
    m = torch.min(data, dim=dim, keepdim=True)
    r = torch.max(data, dim=dim, keepdim=True) - m
    return (data-m)/r, m, r

class Lightcurve(object):
    def __init__(self, *args):
        pass


    @property
    def magnitudes(self):
        pass

    @magnitudes.setter
    def magnitudes(self, value):
        pass
