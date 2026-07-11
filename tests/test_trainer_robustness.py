import unittest
from unittest import mock

import torch
import gpytorch

from pgmuvi.trainers import (
    NotPSDError,
    _collect_trainable_parameters,
    _raise_if_nonfinite_tensor,
    _snapshot_training_state,
    _restore_training_state,
    _validate_training_controls,
    train,
)


class ToyExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestTrainerRobustness(unittest.TestCase):
    def test_collect_trainable_parameters_includes_likelihood(self):
        train_x = torch.linspace(0, 1, 5)
        train_y = torch.sin(train_x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ToyExactGP(train_x, train_y, likelihood)

        params = _collect_trainable_parameters(model, likelihood)
        ids = {id(param) for param in params}

        self.assertTrue(params)
        for param in model.parameters():
            if param.requires_grad:
                self.assertIn(id(param), ids)
        for param in likelihood.parameters():
            if param.requires_grad:
                self.assertIn(id(param), ids)

    def test_string_optimizer_receives_model_and_likelihood_parameters(self):
        train_x = torch.linspace(0, 1, 5)
        train_y = torch.sin(train_x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ToyExactGP(train_x, train_y, likelihood)
        captured = {}

        class CapturingAdam:
            def __init__(self, params, lr, eps):
                captured["params"] = list(params)
                captured["lr"] = lr
                captured["eps"] = eps

            def zero_grad(self):
                for param in captured["params"]:
                    if param.grad is not None:
                        param.grad.zero_()

            def step(self):
                pass

        with mock.patch("torch.optim.Adam", CapturingAdam):
            train(
                model=model,
                likelihood=likelihood,
                train_x=train_x,
                train_y=train_y,
                maxiter=1,
                miniter=0,
                optim="Adam",
                lr=0.01,
                verbose=False,
            )

        param_ids = {id(param) for param in captured["params"]}
        self.assertTrue(param_ids)
        for param in model.parameters():
            if param.requires_grad:
                self.assertIn(id(param), param_ids)
        for param in likelihood.parameters():
            if param.requires_grad:
                self.assertIn(id(param), param_ids)


    def test_training_state_snapshot_restores_model_and_likelihood(self):
        train_x = torch.linspace(0, 1, 5)
        train_y = torch.sin(train_x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ToyExactGP(train_x, train_y, likelihood)

        original_mean = model.mean_module.raw_constant.detach().clone()
        original_noise = likelihood.noise_covar.raw_noise.detach().clone()
        state = _snapshot_training_state(model, likelihood)

        with torch.no_grad():
            model.mean_module.raw_constant.add_(3.0)
            likelihood.noise_covar.raw_noise.add_(2.0)

        _restore_training_state(model, likelihood, state)

        self.assertTrue(torch.equal(model.mean_module.raw_constant, original_mean))
        self.assertTrue(torch.equal(likelihood.noise_covar.raw_noise, original_noise))

    @unittest.skipIf(NotPSDError is None, "linear_operator NotPSDError unavailable")
    def test_train_recovers_notpsd_failure_by_restoring_best_state(self):
        train_x = torch.linspace(0, 1, 5)
        train_y = torch.sin(train_x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ToyExactGP(train_x, train_y, likelihood)

        with torch.no_grad():
            model.mean_module.raw_constant.fill_(1.0)
        expected_restored = model.mean_module.raw_constant.detach().clone()

        class FailingMLL:
            def __init__(self):
                self.calls = 0

            def __call__(self, output, target):
                self.calls += 1
                if self.calls == 1:
                    return -(model.mean_module.raw_constant ** 2).sum()
                raise NotPSDError("synthetic not-PSD failure")

        with mock.patch(
            "gpytorch.mlls.ExactMarginalLogLikelihood",
            return_value=FailingMLL(),
        ):
            results = train(
                model=model,
                likelihood=likelihood,
                train_x=train_x,
                train_y=train_y,
                maxiter=3,
                miniter=0,
                optim="SGD",
                lr=0.1,
                verbose=False,
            )

        self.assertTrue(results["training_recovered_from_failure"])
        self.assertEqual(results["training_failure_iteration"], 1)
        self.assertEqual(results["training_restored_best_iteration"], 0)
        self.assertEqual(results["training_failure_type"], "NotPSDError")
        self.assertTrue(torch.equal(model.mean_module.raw_constant, expected_restored))

    @unittest.skipIf(NotPSDError is None, "linear_operator NotPSDError unavailable")
    def test_train_can_disable_notpsd_recovery(self):
        train_x = torch.linspace(0, 1, 5)
        train_y = torch.sin(train_x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ToyExactGP(train_x, train_y, likelihood)

        class FailingMLL:
            def __init__(self):
                self.calls = 0

            def __call__(self, output, target):
                self.calls += 1
                if self.calls == 1:
                    return -(model.mean_module.raw_constant ** 2).sum()
                raise NotPSDError("synthetic not-PSD failure")

        with mock.patch(
            "gpytorch.mlls.ExactMarginalLogLikelihood",
            return_value=FailingMLL(),
        ):
            with self.assertRaises(NotPSDError):
                train(
                    model=model,
                    likelihood=likelihood,
                    train_x=train_x,
                    train_y=train_y,
                    maxiter=3,
                    miniter=0,
                    optim="SGD",
                    lr=0.1,
                    verbose=False,
                    restore_best_on_failure=False,
                )

    def test_training_control_validation_rejects_invalid_values(self):
        # Zero-iteration fits are intentionally supported by consensus smoke
        # tests that exercise initialization/failure handling without GP training.
        _validate_training_controls(maxiter=0, miniter=0, stopavg=1, lr=0.1)

        with self.assertRaises(ValueError):
            _validate_training_controls(maxiter=-1, miniter=0, stopavg=1, lr=0.1)
        with self.assertRaises(ValueError):
            _validate_training_controls(maxiter=1, miniter=-1, stopavg=1, lr=0.1)
        with self.assertRaises(ValueError):
            _validate_training_controls(maxiter=1, miniter=0, stopavg=0, lr=0.1)
        with self.assertRaises(ValueError):
            _validate_training_controls(maxiter=1, miniter=0, stopavg=1, lr=0.0)
        with self.assertRaises(ValueError):
            _validate_training_controls(maxiter=1, miniter=0, stopavg=1, lr=float("nan"))

    def test_nonfinite_loss_check_raises_clear_error(self):
        with self.assertRaisesRegex(FloatingPointError, "Non-finite training loss"):
            _raise_if_nonfinite_tensor(
                torch.tensor(float("nan")), label="training loss", iteration=3
            )


if __name__ == "__main__":
    unittest.main()
