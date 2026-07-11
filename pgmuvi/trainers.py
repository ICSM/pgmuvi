import copy
import numbers

import numpy as np
import torch
import gpytorch
from tqdm import tqdm

try:
    from linear_operator.utils.errors import NotPSDError
except Exception:  # pragma: no cover - compatibility with older GPyTorch stacks
    NotPSDError = None


def _collect_trainable_parameters(*modules):
    """Return unique trainable parameters from model/likelihood modules.

    GPyTorch likelihoods may own trainable parameters, e.g. the standard
    GaussianLikelihood noise parameter or the additional noise term in
    FixedNoiseGaussianLikelihood(learn_additional_noise=True).  Optimizers
    built from model.parameters() alone silently leave those parameters fixed.
    """

    params = []
    seen = set()

    for module in modules:
        if module is None or not hasattr(module, "parameters"):
            continue
        for param in module.parameters():
            if not getattr(param, "requires_grad", False):
                continue
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            params.append(param)

    return params




def _iter_named_trainable_parameters(model, likelihood=None):
    """Yield stable result-history names and unique trainable parameters.

    The trainer records parameter histories after each optimization step.  Once
    likelihood parameters are included in string-built optimizers, the history
    dictionary must use the same model+likelihood parameter set.
    """

    seen = set()

    for prefix, module in (("", model), ("likelihood.", likelihood)):
        if module is None or not hasattr(module, "named_parameters"):
            continue
        for name, param in module.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            yield f"{prefix}{name}", param



def _recoverable_training_errors():
    """Return numerical training exceptions that may have a usable best state."""

    errors = [FloatingPointError]
    if NotPSDError is not None:
        errors.append(NotPSDError)
    return tuple(errors)


def _snapshot_module_state(module):
    """Return a detached deep copy of a module state_dict."""

    if module is None or not hasattr(module, "state_dict"):
        return None
    return copy.deepcopy(module.state_dict())


def _snapshot_training_state(model, likelihood=None):
    """Capture model/likelihood state so a failed fit can be restored."""

    return {
        "model": _snapshot_module_state(model),
        "likelihood": _snapshot_module_state(likelihood),
    }


def _restore_module_state(module, state):
    """Restore a module state_dict if both module and snapshot exist."""

    if module is None or state is None or not hasattr(module, "load_state_dict"):
        return
    module.load_state_dict(state)


def _restore_training_state(model, likelihood, state):
    """Restore model/likelihood state captured by _snapshot_training_state."""

    if state is None:
        return
    _restore_module_state(model, state.get("model"))
    _restore_module_state(likelihood, state.get("likelihood"))


def _validate_training_controls(maxiter, miniter, stopavg, lr):
    """Validate basic scalar trainer controls before optimizer setup."""

    if not isinstance(maxiter, numbers.Integral) or int(maxiter) < 0:
        raise ValueError("maxiter must be a non-negative integer.")
    if not isinstance(miniter, numbers.Integral) or int(miniter) < 0:
        raise ValueError("miniter must be a non-negative integer.")
    if not isinstance(stopavg, numbers.Integral) or int(stopavg) < 1:
        raise ValueError("stopavg must be a positive integer.")
    if (
        not isinstance(lr, numbers.Real)
        or not np.isfinite(float(lr))
        or float(lr) <= 0
    ):
        raise ValueError("lr must be a positive finite real number.")

    return int(maxiter), int(miniter), int(stopavg), float(lr)


def _raise_if_nonfinite_tensor(value, *, label, iteration=None):
    """Raise a clear error if a tensor/scalar contains NaN or Inf."""

    tensor = torch.as_tensor(value)
    if torch.isfinite(tensor).all():
        return

    where = f" at iteration {iteration}" if iteration is not None else ""
    raise FloatingPointError(f"Non-finite {label}{where}.")


def _raise_if_nonfinite_gradients(model, likelihood=None, *, iteration=None):
    """Raise a clear error if any optimized parameter has NaN/Inf gradients."""

    bad_names = []

    def _check_named_parameters(prefix, module):
        if module is None or not hasattr(module, "named_parameters"):
            return
        for name, param in module.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                bad_names.append(f"{prefix}{name}")

    _check_named_parameters("model.", model)
    _check_named_parameters("likelihood.", likelihood)

    if bad_names:
        where = f" at iteration {iteration}" if iteration is not None else ""
        listed = ", ".join(bad_names[:10])
        extra = "" if len(bad_names) <= 10 else f", ... ({len(bad_names)} total)"
        raise FloatingPointError(
            f"Non-finite parameter gradient(s){where}: {listed}{extra}."
        )


class Trainer:
    def __init__():
        pass


def train(
    lightcurve=None,
    model=None,
    likelihood=None,
    train_x=None,
    train_y=None,
    maxiter=100,
    miniter=10,
    stop=None,
    lr=1e-4,
    lossfn="mll",
    optim="SGD",
    eps=1e-8,
    stopavg=9,
    verbose=True,
    restore_best_on_failure=True,
    **kwargs,
):
    """Given a GP model, a likelihood, and some training data, optimise a
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
    restore_best_on_failure : bool, default True
        If training fails after at least one finite-loss iteration because of a
        recoverable numerical error, restore the best finite-loss model and
        likelihood state and return the partial results instead of discarding
        the fit.  This currently covers non-finite-loss/gradient failures and
        GPyTorch/linear-operator NotPSDError failures.

    Examples
    --------

    """

    if lightcurve is not None:
        if any(
            [
                model is not None,
                likelihood is not None,
                train_x is not None,
                train_y is not None,
            ]
        ):
            print(
                """A lightcurve object was passed to train(), but one or
                  more of model, likelihood, train_x and train_y were also
                  passed. The lightcurve object will be used, and the other
                  parameters will be ignored."""
            )
        model = lightcurve.model
        likelihood = lightcurve.likelihood
        train_x = lightcurve._xdata_transformed
        train_y = lightcurve._ydata_transformed
    elif any([model is None, likelihood is None, train_x is None, train_y is None]):
        raise ValueError(
            """If a lightcurve object is not passed to train(),
                         **all** of model, likelihood, train_x and train_y
                         **must** be passed to train()."""
        )

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
        if lossfn == "mll":
            # loss = -1* marginal log-likelihood
            lossfn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        elif lossfn == "elbo":
            # loss = -1* variational elbo,
            # variational inference to be performed!
            raise NotImplementedError(
                "Currently only maximisation of the marginal log-likelihood is "
                "implemented. Using elbo will be implemented soon"
            )
    elif isinstance(
        lossfn, gpytorch.mlls.marginal_log_likelihood.MarginalLogLikelihood
    ):
        raise NotImplementedError(
            "Currently only maximisation of the marginal log-likelihood is "
            "implemented. Passing arbitrary MLL objects will be implemented "
            "soon."
        )
    else:
        raise ValueError(
            "lossfn must be either 'mll', 'elbo', or a gpytorch, torch or "
            "pyro loss function."
        )

    maxiter, miniter, stopavg, lr = _validate_training_controls(
        maxiter=maxiter, miniter=miniter, stopavg=stopavg, lr=lr
    )

    if isinstance(optim, str):
        optim_params = _collect_trainable_parameters(model, likelihood)
        if not optim_params:
            raise ValueError(
                "No trainable model or likelihood parameters were found for "
                "the optimizer."
            )

        if optim == "SGD":
            optimizer = torch.optim.SGD(optim_params, lr=lr)
        elif optim == "Adam":
            optimizer = torch.optim.Adam(optim_params, lr=lr, eps=eps)
        elif optim == "AdamW":
            optimizer = torch.optim.AdamW(optim_params, lr=lr, eps=eps)
        elif optim == "NUTS":
            raise NotImplementedError(
                "Optimisation with NUTS/MCMC is not yet implemented."
            )
        else:
            raise ValueError(
                """optim must be either 'SGD', 'Adam', 'AdamW',
                            'NUTS', or an instance of a torch or pyro optimiser.
                            """
            )
    elif isinstance(optim, torch.optim.Optimizer):
        optimizer = optim
    else:
        raise ValueError(
            """optim must be either 'SGD', 'Adam', 'AdamW',
                        'NUTS', or an instance of a torch or pyro optimiser.
                        """
        )

    results = {
        "loss": [],
        "delta_loss": [],
        "training_recovered_from_failure": False,
        "training_failure_iteration": None,
        "training_failure_type": None,
        "training_failure_message": None,
        "training_restored_best_iteration": None,
        "training_restored_best_loss": None,
        "training_best_iteration": None,
        "training_best_loss": None,
    }
    best_state = None
    best_iteration = None
    best_loss = None
    if lightcurve is not None:
        pars = lightcurve.get_parameters()
        for key, value in pars.items():
            results[key] = [value.cpu().detach().numpy()]
    else:
        for param_name, _param in _iter_named_trainable_parameters(model, likelihood):
            results[param_name] = []
    # for param_name, param in
    for i in tqdm(range(maxiter), disable=not verbose):
        try:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -lossfn(output, train_y)
            _raise_if_nonfinite_tensor(
                loss.detach(), label="training loss", iteration=i
            )
            loss_value = float(loss.detach().cpu().item())
            loss.backward()
            _raise_if_nonfinite_gradients(model, likelihood, iteration=i)

            # Snapshot the pre-step state that produced the finite loss.
            # If the optimizer step moves into a numerically invalid region,
            # the next iteration can restore this known-good state.
            if best_loss is None or loss_value < best_loss:
                best_loss = loss_value
                best_iteration = i
                best_state = _snapshot_training_state(model, likelihood)
                results["training_best_iteration"] = best_iteration
                results["training_best_loss"] = best_loss

            optimizer.step()
        except _recoverable_training_errors() as exc:
            if restore_best_on_failure and best_state is not None:
                _restore_training_state(model, likelihood, best_state)
                model.train()
                likelihood.train()
                results["training_recovered_from_failure"] = True
                results["training_failure_iteration"] = i
                results["training_failure_type"] = type(exc).__name__
                results["training_failure_message"] = str(exc)
                results["training_restored_best_iteration"] = best_iteration
                results["training_restored_best_loss"] = best_loss
                if verbose:
                    print(
                        "Training recovered from "
                        f"{type(exc).__name__} at iteration {i}; "
                        f"restored best finite-loss state from iteration "
                        f"{best_iteration} (loss={best_loss})."
                    )
                break
            raise

        # Now update list of parameters
        if i > 0:
            results["delta_loss"].append(
                loss.cpu().detach().numpy() - results["loss"][-1]
            )
        results["loss"].append(loss.cpu().detach().numpy())

        if lightcurve is not None:
            for key, value in lightcurve.get_parameters().items():
                results[key].append(value.cpu().detach().numpy())
        else:
            for param_name, param in _iter_named_trainable_parameters(model, likelihood):
                results[param_name].append(param.cpu().detach().numpy())
            # print(i, param_name," = ",param.item())
        # Finally check if convergence criterion is met
        # optimisers are stochastic, so we average the change in loss
        # function over a few iterations
        if stop and i > miniter:
            stopval = np.std(results["loss"][-stopavg:])
            if stopval < stop:
                if verbose:
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
