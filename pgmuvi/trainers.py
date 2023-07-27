import numpy as np
import torch
import gpytorch
from tqdm import tqdm


class Trainer:
    def __init__():
        pass


def train(lightcurve=None, model=None, likelihood=None,
          train_x=None, train_y=None,
          maxiter=100, miniter=10, stop=None, lr=1e-4,
          lossfn='mll', optim="SGD", eps=1e-8, stopavg=9,
          **kwargs):
    ''' Given a GP model, a likelihood, and some training data, optimise a
    loss function to fit the training data.

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
        The maximum number of training iterations to use. If stop is not a
        positive number, this will be the number of iterations used to train.
    miniter : int, default 10
        The minimum number of training iterations to use. This parameter is
        only used if stop is a positive real number, in which case it is used
        to ensure that a sufficient number of iterations have been performed
        before terminating training.
    stop : float, default None
        The fractional change in the loss function below which training will be
        terminated. If set to None, a negative value, not a number of a
        non-numerical type, training will continue until maxiter is reached.
    lr : float, default 1e-4
        The learning rate for the optimiser. Increasing this number will
        result in larger steps in the parameters each iteration. This will
        make it easier to escape local minima, but may also result in
        instability.
    lossfn : string or instance of
             gpytorch.mlls.marginal_log_likelihood.MarginalLogLikelihood,
             default 'mll'
        The loss function that will be used to evaluate the training.
        If a string, it must take one of the values 'mll' or 'elbo'.
    optim : string or instance of torch.optim.optimizer.Optimizer,
            default 'SGD'
        The optimizer that will be used to train the model.
        If a string, it must take one of the values 'SGD', 'Adam', 'AdamW',
        'NUTS'. Otherwise, it may be any torch or pyro optimiser. If passing a
        torch or pyro optimiser, it should already have been initialised with
        all arguments set
    eps : float, default 1e-8.
        term added to the denominator to improve numerical stability in some
        optimisers (e.g. AdamW)

    Examples
    --------

    '''

    if lightcurve is not None:
        if any([model is not None,
                likelihood is not None,
                train_x is not None,
                train_y is not None]):
            print("""A lightcurve object was passed to train(), but one or
                  more of model, likelihood, train_x and train_y were also
                  passed. The lightcurve object will be used, and the other
                  parameters will be ignored.""")
        model = lightcurve.model
        likelihood = lightcurve.likelihood
        train_x = lightcurve._xdata_transformed
        train_y = lightcurve._ydata_transformed
    elif any([model is None,
              likelihood is None,
              train_x is None,
              train_y is None]):
        raise ValueError("""If a lightcurve object is not passed to train(),
                         **all** of model, likelihood, train_x and train_y
                         **must** be passed to train().""")

    # We're going to be doing some training, so our first step should be to
    # put the model and likelihood into training mode:
    model.train()
    likelihood.train()

    # Idea here is to provide a convenience function, so most of the time
    # users only need to interact with this routine. It calls the other
    # routines to do the training, which users can interact with if they
    # choose to.
    if isinstance(lossfn, str):
        # Loss function is passed as a string, must be one of the values we
        # understand:
        if lossfn == 'mll':
            # loss = -1* marginal log-likelihood
            lossfn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,
                                                              model)
        elif lossfn == 'elbo':
            # loss = -1* variational elbo,
            # variational inference to be performed!
            raise NotImplementedError("Currently only maximisation of the marginal log-likelihood is implemented. Using elbo will be implemented soon")  # noqa: E501
    elif isinstance(lossfn,
                    gpytorch.mlls.marginal_log_likelihood.MarginalLogLikelihood):
        raise NotImplementedError("Currently only maximisation of the marginal log-likelihood is implemented. Passing arbitrary MLL objects will be implemented soon.")  # noqa: E501
    else:
        raise ValueError("lossfn must be either 'mll', 'elbo', or a gpytorch, torch or pyro loss function.")  # noqa: E501

    if isinstance(optim, str):
        if optim == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif optim == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)
        elif optim == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps)
        elif optim == "NUTS":
            raise NotImplementedError("Optimisation with NUTS/MCMC is not yet implemented.")  # noqa: E501
        else:
            raise ValueError("""optim must be either 'SGD', 'Adam', 'AdamW',
                            'NUTS', or an instance of a torch or pyro optimiser.
                            """)
    elif isinstance(optim, torch.optim.Optimizer):
        optimizer = optim
    else:
        raise ValueError("""optim must be either 'SGD', 'Adam', 'AdamW',
                        'NUTS', or an instance of a torch or pyro optimiser.
                        """)

    results = {'loss': [], 'delta_loss': []}
    if lightcurve is not None:
        pars = lightcurve.get_parameters()
        for key, value in pars.items():
            results[key] = [value]
    else:
        for param_name, param in model.named_parameters():
            p = param_name.split('.')[1] if 'raw' in param_name else param_name
            results[p] = []
    # for param_name, param in
    for i in tqdm(range(maxiter)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -lossfn(output, train_y)
        loss.backward()
        optimizer.step()
        # Now update list of parameters
        if i > 0:
            results['delta_loss'].append(loss.detach().numpy() - results['loss'][-1])
        results['loss'].append(loss.detach().numpy())

        if lightcurve is not None:
            for key, value in lightcurve.get_parameters().items():
                results[key] = [value]
        else:
            for param_name, param in model.named_parameters():
                results[param_name].append(param.detach().numpy())
            # print(i, param_name," = ",param.item())
        # Finally check if convergence criterion is met
        # optimisers are stochastic, so we average the change in loss
        # function over a few iterations
        if stop and i > miniter:
            stopval = np.std(results['loss'][-stopavg:])
            if stopval < stop:
                print(
                    f"""Average change in loss over the last {stopavg} iterations
                    was {stopval}.\n This is < {stop}, so we will end training here."""
                )
                break  # break out of the training loop early

    return results


def train_mll():
    pass


def train_variational():
    pass


def train_variational_uncertain():
    pass
