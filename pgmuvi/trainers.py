import numpy as np
import torch
import gpytorch
import pyro
#import tqdm



def train(model, likelihood, train_x, train_y,
          maxiter = 100, miniter = 10, stop = None, lr = 1e-4,
          lossfn='mll', optim = "SGD",
          **kwargs):
    ''' Given a GP model, a likelihood, and some training data, optimise a loss function to fit the training data.

    Parameters
    ----------
    model : an instance of gpytorch.models.gp.GP or a subcluss thereof
        The GP model whose (hyper-)parameters will be optimised.
    likelihood : an instance of gpytorch.likelihoods.likelihood.Likelihood
        The likelihood function for the Gaussian Process.
    train_x : torch.Tensor or array-like
        The values of the independent variables for training.
    train_y : torch.Tensor or array-like
        The values of the dependent variables for training.
    maxiter : int, default 100
        The maximum number of training iterations to use. If stop is not a positive number, this will be the number of iterations used to train.
    miniter : int, default 10
        The minimum number of training iterations to use. This parameter is only 
        used if stop is a positive real number, in which case it is used to 
        ensure that a sufficient number of iterations have been performed before
        terminating training.
    stop : float, default None
        The fractional change in the loss function below which training will be 
        terminated. If set to None, a negative value, not a number of a 
        non-numerical type, training will continue until maxiter is reached.
    lr : float, default 1e-4
        The learning rate for the optimiser.
    lossfn : string or instance of gpytorch.mlls.marginal_log_likelihood.MarginalLogLikelihood, default 'mll'
        The loss function that will be used to evaluate the training. 
        If a string, it must take one of the values 'mll' or 'elbo'.
    optim : string or instance of torch.optim.optimizer.Optimizer, default 'SGD'
        The optimizer that will be used to train the model.
        If a string, it must take one of the values 'SGD', 'Adam', 'AdamW', 'NUTS'
        Otherwise, it may be any torch or pyro optimiser. 

    Examples
    --------

    '''
    #Idea here is to provide a convenience function, so most of the time users only need to interact with this routine. It calls the other routines to do the training, which users can interact with if they choose to.
    if isinstance(lossfn, str):
        #Loss function is passed as a string, must be one of the values we understand:
        if lossfn is 'mll':
            #loss = -1* marginal log-likelihood
            pass
        elif lossfn is 'elbo':
            #loss = -1* variational elbo, variational inference to be performed!
            pass
    elif isinstance(lossfn, ):
        pass
    else:
        raise ValueError("lossfn must be either ")
    pass


def train_mll():
    pass

def train_variational():
    pass

def train_variational_uncertain():
    pass
