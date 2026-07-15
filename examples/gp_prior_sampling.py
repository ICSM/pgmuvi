#!/usr/bin/env python3
"""Generate reproducible mock light curves from PGMUVI GP priors.

This example instantiates supported one-dimensional GP model classes directly,
applies user-specified physical hyperparameters through the shared parameter
application layer, samples the latent prior, adds observational noise, and wraps
the result in :class:`pgmuvi.lightcurve.Lightcurve`.

No optimizer or fitting workflow is started.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import torch

from pgmuvi.dtypes import DEFAULT_DTYPE
from pgmuvi.gps import MaternGPModel
from pgmuvi.gps import QuasiPeriodicGPModel
from pgmuvi.gps import SpectralMixtureGPModel
from pgmuvi.lightcurve import Lightcurve
from pgmuvi.parameter_estimates import ParameterEstimate
from pgmuvi.parameter_estimates import ParameterEstimateCollection
from pgmuvi.parameter_workflow import apply_parameter_estimates


DEFAULT_SEED = 20260715
SUPPORTED_KERNELS = (
    "quasi_periodic",
    "matern",
    "spectral_mixture",
)


def _user_estimates(model, values: dict[str, Any]) -> ParameterEstimateCollection:
    """Build physical-space parameter estimates from a model schema."""
    schema = model.parameter_schema()
    unknown = sorted(set(values) - set(schema.names()))
    if unknown:
        raise KeyError(f"Unknown parameter name(s): {unknown}")

    return ParameterEstimateCollection(
        [
            ParameterEstimate(
                spec=schema[name],
                value=value,
                value_source="tutorial_user_specified",
            )
            for name, value in values.items()
        ]
    )


def _configure_model(model, values: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """Apply user-specified values without touching raw parameters directly."""
    report = apply_parameter_estimates(model, _user_estimates(model, values))
    if report is None or not all(item["value"] for item in report.values()):
        raise RuntimeError(f"Unable to apply all requested parameters: {report}")
    return report


def _sample_latent_prior(model, x: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Draw one latent realization from ``model.forward(x)``."""
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        return model.forward(x).sample()


def _observation_times(*, seed: int, n_points: int, span: float) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    values = np.sort(rng.uniform(0.0, span, n_points))
    return torch.as_tensor(values, dtype=DEFAULT_DTYPE)


def _build_model(
    kernel: str,
    x: torch.Tensor,
) -> tuple[gpytorch.models.ExactGP, dict[str, Any], dict[str, Any]]:
    placeholder = torch.sin(2.0 * torch.pi * x / 180.0)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    if kernel == "quasi_periodic":
        truth = {
            "period_days": 180.0,
            "coherence_days": 650.0,
            "output_variance": 1.4,
        }
        model = QuasiPeriodicGPModel(
            x,
            placeholder,
            likelihood,
            period=truth["period_days"],
        )
        values = {
            "covar_module.outputscale": truth["output_variance"],
            "covar_module.base_kernel.kernels.0.period_length": truth[
                "period_days"
            ],
            "covar_module.base_kernel.kernels.1.lengthscale": truth[
                "coherence_days"
            ],
        }
    elif kernel == "matern":
        truth = {
            "nu": 1.5,
            "lengthscale_days": 85.0,
            "output_variance": 1.8,
        }
        model = MaternGPModel(
            x,
            placeholder,
            likelihood,
            nu=truth["nu"],
            lengthscale=truth["lengthscale_days"],
        )
        values = {
            "covar_module.outputscale": truth["output_variance"],
            "covar_module.base_kernel.lengthscale": truth[
                "lengthscale_days"
            ],
        }
    elif kernel == "spectral_mixture":
        periods = [180.0, 65.0]
        truth = {
            "periods_days": periods,
            "frequencies_per_day": [1.0 / period for period in periods],
            "frequency_widths_per_day": [7.0e-4, 2.8e-3],
            "variance_weights": [0.9, 0.35],
        }
        model = SpectralMixtureGPModel(
            x,
            placeholder,
            likelihood,
            num_mixtures=2,
        )
        values = {
            "covar_module.mixture_means": truth["frequencies_per_day"],
            "covar_module.mixture_scales": truth[
                "frequency_widths_per_day"
            ],
            "covar_module.mixture_weights": truth["variance_weights"],
        }
    else:
        raise ValueError(
            f"Unsupported kernel {kernel!r}; choose from {SUPPORTED_KERNELS}."
        )

    return model, truth, values


def generate_prior_sample(
    kernel: str,
    *,
    seed: int = DEFAULT_SEED,
    n_points: int = 72,
    span: float = 720.0,
    noise_sigma: float = 0.12,
) -> tuple[Lightcurve, dict[str, Any]]:
    """Generate one observed mock light curve and a JSON-safe summary."""
    if kernel not in SUPPORTED_KERNELS:
        raise ValueError(
            f"Unsupported kernel {kernel!r}; choose from {SUPPORTED_KERNELS}."
        )
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if span <= 0:
        raise ValueError("span must be positive")
    if noise_sigma <= 0:
        raise ValueError("noise_sigma must be positive")

    torch.set_default_dtype(DEFAULT_DTYPE)
    torch.manual_seed(seed)

    x = _observation_times(seed=seed, n_points=n_points, span=span)
    model, truth, values = _build_model(kernel, x)
    application_report = _configure_model(model, values)
    latent = _sample_latent_prior(model, x, seed=seed + 101)

    torch.manual_seed(seed + 202)
    yerr = torch.full_like(latent, noise_sigma)
    observed = latent + noise_sigma * torch.randn_like(latent)
    lightcurve = Lightcurve(
        x,
        observed,
        yerr=yerr,
        max_samples=None,
        check_sampling=False,
        name=f"{kernel}_gp_prior_mock",
    )

    summary = {
        "kernel": kernel,
        "sampling_mode": "latent_gp_prior_plus_observational_noise",
        "fit_started": False,
        "seed": seed,
        "n_points": int(x.numel()),
        "time_span_days": float(x.max() - x.min()),
        "noise_sigma": float(noise_sigma),
        "dtype": str(x.dtype),
        "xtransform": type(lightcurve.xtransform).__name__,
        "latent_mean": float(latent.mean()),
        "latent_std": float(latent.std(unbiased=False)),
        "observed_mean": float(observed.mean()),
        "observed_std": float(observed.std(unbiased=False)),
        "truth": truth,
        "parameter_application": application_report,
    }
    return lightcurve, summary


def build_report(
    kernel: str,
    *,
    seed: int = DEFAULT_SEED,
    n_points: int = 72,
    span: float = 720.0,
    noise_sigma: float = 0.12,
) -> dict[str, Any]:
    """Generate one or all supported mock curves and return summaries."""
    selected = SUPPORTED_KERNELS if kernel == "all" else (kernel,)
    summaries = []
    for index, name in enumerate(selected):
        _, summary = generate_prior_sample(
            name,
            seed=seed + index,
            n_points=n_points,
            span=span,
            noise_sigma=noise_sigma,
        )
        summaries.append(summary)

    return {
        "status": "success",
        "workflow": "gp_prior_sampling",
        "fit_started": False,
        "kernels": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible PGMUVI GP-prior mock summaries."
    )
    parser.add_argument(
        "--kernel",
        choices=("all",) + SUPPORTED_KERNELS,
        default="all",
        help="Kernel family to sample (default: all).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-points", type=int, default=72)
    parser.add_argument("--span", type=float, default=720.0)
    parser.add_argument("--noise-sigma", type=float, default=0.12)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the machine-readable report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        args.kernel,
        seed=args.seed,
        n_points=args.n_points,
        span=args.span,
        noise_sigma=args.noise_sigma,
    )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
