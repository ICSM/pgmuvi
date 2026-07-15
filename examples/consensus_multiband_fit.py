#!/usr/bin/env python3
"""Run one direct or consensus multiband PGMUVI fit.

The script uses either a CSV file or a deterministic synthetic light curve.  It
supports a no-training ``--dry-run`` mode and writes structured diagnostics only
when an explicit ``--output-dir`` is supplied.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pgmuvi.lightcurve import ConsensusFitError, Lightcurve


LPV_CONFIGURABLE_MODELS = {
    "2DWavelengthDependent",
    "2DDustMean",
    "2DPowerLawMean",
}
SUPPORTED_MODELS = (
    "2D",
    "2DSeparable",
    "2DWavelengthDependent",
    "2DDustMean",
    "2DPowerLawMean",
)


def make_synthetic_lpv_lightcurve(n_per_band: int = 60) -> Lightcurve:
    """Create a small coherent 2-D light curve for demonstration."""
    rng = np.random.default_rng(12345)
    wavelengths = np.array([0.55, 0.80, 1.25], dtype=float)
    period = 120.0

    x_rows = []
    y_vals = []
    yerr_vals = []
    band_labels = []

    for index, wavelength in enumerate(wavelengths):
        time = np.sort(rng.uniform(0.0, 4.0 * period, size=n_per_band))
        amplitude = 1.0 + 0.35 * index
        mean = 10.0 + 0.4 * index
        err = np.full_like(time, 0.08 + 0.02 * index, dtype=float)
        phase = 2.0 * np.pi * time / period
        flux = mean + amplitude * np.sin(phase) + rng.normal(0.0, err)

        x_rows.append(np.column_stack([time, np.full_like(time, wavelength)]))
        y_vals.append(flux)
        yerr_vals.append(err)
        band_labels.append(
            np.full(time.shape, f"band_{wavelength:g}", dtype=object)
        )

    return Lightcurve(
        xdata=np.vstack(x_rows),
        ydata=np.concatenate(y_vals),
        yerr=np.concatenate(yerr_vals),
        band=np.concatenate(band_labels),
        check_sampling=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one direct or consensus multiband PGMUVI fit.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Optional source CSV. If omitted, deterministic synthetic data are used.",
    )
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="2D")
    parser.add_argument(
        "--fit-strategy",
        choices=("standard", "consensus", "consensus_multicomp"),
        default="consensus",
        help="Use 'standard' for a direct fit without consensus initialization.",
    )
    parser.add_argument(
        "--time-kernel-type",
        choices=("quasi_periodic", "spectral_mixture", "matern", "rbf"),
        default=None,
    )
    parser.add_argument(
        "--wavelength-kernel-type",
        choices=("rbf", "matern", "rational_quadratic"),
        default=None,
    )
    parser.add_argument("--num-mixtures", type=int, default=None)
    parser.add_argument("--constraint-set", choices=("LPV",), default=None)
    parser.add_argument("--training-iter", type=int, default=50)
    parser.add_argument("--miniter", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-samples-per-band", type=int, default=100)
    parser.add_argument(
        "--learn-additional-noise",
        action="store_true",
        default=True,
        help="Learn one additional variance term on top of supplied yerr values.",
    )
    parser.add_argument(
        "--no-learn-additional-noise",
        dest="learn_additional_noise",
        action="store_false",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for JSON run, result, and failure artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the source and print the resolved fit kwargs without training.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_configuration(args: argparse.Namespace) -> None:
    """Reject combinations that would otherwise be silently misleading."""
    if args.training_iter < 0:
        raise ValueError("--training-iter must be non-negative")
    if args.miniter < 0:
        raise ValueError("--miniter must be non-negative")
    if args.miniter > args.training_iter:
        raise ValueError("--miniter cannot exceed --training-iter")
    if args.num_mixtures is not None and args.num_mixtures < 1:
        raise ValueError("--num-mixtures must be a positive integer")

    if args.model == "2D":
        if args.time_kernel_type is not None or args.wavelength_kernel_type is not None:
            raise ValueError(
                "model='2D' has a built-in full 2-D spectral-mixture kernel; "
                "do not pass time/wavelength kernel selectors"
            )

    if args.model == "2DSeparable":
        if args.fit_strategy != "standard":
            raise ValueError(
                "2DSeparable is documented here as a direct product-kernel baseline; "
                "use --fit-strategy standard"
            )
        if args.time_kernel_type is not None or args.wavelength_kernel_type is not None:
            raise ValueError(
                "2DSeparable does not use the string kernel selectors; choose "
                "2DWavelengthDependent for configurable time/wavelength kernels"
            )

    if args.fit_strategy == "consensus_multicomp" and args.model != "2D":
        raise ValueError(
            "The maintained consensus_multicomp example path currently requires "
            "model='2D'"
        )

    if (
        args.fit_strategy == "consensus"
        and args.model in LPV_CONFIGURABLE_MODELS
        and args.time_kernel_type in {"matern", "rbf"}
    ):
        raise ValueError(
            "Consensus requires a spectral-mixture or period-based time-kernel "
            "target; use --time-kernel-type quasi_periodic or spectral_mixture"
        )


def load_lightcurve(args: argparse.Namespace) -> Lightcurve:
    if args.csv_path:
        return Lightcurve.from_csv(
            args.csv_path,
            check_sampling=True,
            max_samples=args.max_samples,
            max_samples_per_band=args.max_samples_per_band,
            verbose=args.verbose,
        )
    return make_synthetic_lpv_lightcurve()


def build_fit_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    fit_kwargs: dict[str, Any] = {
        "model": args.model,
        "training_iter": args.training_iter,
        "miniter": args.miniter,
        "learn_additional_noise": args.learn_additional_noise,
        "verbose": args.verbose,
    }

    if args.fit_strategy != "standard":
        fit_kwargs["fit_strategy"] = args.fit_strategy

    time_kernel_type = args.time_kernel_type
    if (
        time_kernel_type is None
        and args.fit_strategy == "consensus"
        and args.model in LPV_CONFIGURABLE_MODELS
    ):
        time_kernel_type = "quasi_periodic"

    if time_kernel_type is not None:
        fit_kwargs["time_kernel_type"] = time_kernel_type
    if args.wavelength_kernel_type is not None:
        fit_kwargs["wavelength_kernel_type"] = args.wavelength_kernel_type
    if args.num_mixtures is not None:
        fit_kwargs["num_mixtures"] = args.num_mixtures
    if args.constraint_set is not None:
        fit_kwargs["constraint_set"] = args.constraint_set

    return fit_kwargs


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)


def _format_value(value: Any, precision: int = 6) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{precision}g}"
    except Exception:
        return str(value)


def _summary_dict(period_summary: Any) -> dict[str, Any]:
    if hasattr(period_summary, "as_dict"):
        value = period_summary.as_dict()
        return dict(value) if isinstance(value, dict) else {"value": value}
    if isinstance(period_summary, dict):
        return dict(period_summary)
    return {}


def print_period_summary(period_summary: Any) -> None:
    """Print selected PeriodSummaryResult fields in a user-readable form."""
    summary = _summary_dict(period_summary)

    print("\nPeriod summary:")
    for key in [
        "method",
        "model_name",
        "backend",
        "kernel_family",
        "time_kernel_family",
        "dominant_period",
        "dominant_frequency",
        "q_factor",
        "n_peaks",
        "n_peaks_detected",
        "n_significant_peaks",
    ]:
        value = summary.get(key, _get_field(period_summary, key))
        print(f"  {key}: {_format_value(value)}")

    primary_peak = None
    if hasattr(period_summary, "get_primary_peak"):
        primary_peak = period_summary.get_primary_peak()
    else:
        peaks = _get_field(period_summary, "peaks", []) or summary.get("peaks", [])
        if peaks:
            primary_peak = peaks[0]

    if primary_peak is not None:
        print("  primary_peak:")
        for key in [
            "rank",
            "period",
            "frequency",
            "area_fraction",
            "prominence",
            "coherence_proxy",
        ]:
            print(f"    {key}: {_format_value(_get_field(primary_peak, key))}")

    component_summaries = summary.get("component_summaries") or []
    if component_summaries:
        print(f"  component_summaries: {len(component_summaries)} component(s)")
        for component in component_summaries[:3]:
            print(
                "    "
                f"component={_get_field(component, 'component_index')} "
                f"period={_format_value(_get_field(component, 'consensus_period'))} "
                "strength="
                f"{_format_value(_get_field(component, 'consensus_component_strength'))} "
                f"bands={_get_field(component, 'member_bands', [])}"
            )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            pass
    if hasattr(value, "as_dict"):
        try:
            return _json_safe(value.as_dict())
        except Exception:
            pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote: {path}")


def _configuration_payload(args: argparse.Namespace, fit_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "csv_path": args.csv_path,
        "synthetic_input": args.csv_path is None,
        "dry_run": bool(args.dry_run),
        "fit_kwargs": fit_kwargs,
        "load_kwargs": {
            "check_sampling": True if args.csv_path else False,
            "max_samples": args.max_samples,
            "max_samples_per_band": args.max_samples_per_band,
        },
    }


def _failure_payload(
    lc: Any,
    exc: Exception,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "failed",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "failure_diagnostics": getattr(exc, "failure_diagnostics", None)
        or getattr(lc, "failure_diagnostics", None),
        "failure_summary": getattr(exc, "failure_summary", None)
        or getattr(lc, "failure_summary", None),
        "failure_reason": getattr(lc, "failure_reason", None),
        "fit_failed": getattr(lc, "fit_failed", None),
        "consensus_diagnostics": getattr(lc, "consensus_diagnostics", None),
        "configuration": configuration,
    }


def _write_success_artifacts(
    output_dir: Path,
    lc: Any,
    period_summary: Any | None,
) -> None:
    if period_summary is not None:
        write_json(output_dir / "period_summary.json", _summary_dict(period_summary))

    consensus_diagnostics = getattr(lc, "consensus_diagnostics", None)
    if consensus_diagnostics is not None:
        write_json(
            output_dir / "consensus_diagnostics.json",
            consensus_diagnostics,
        )

    if hasattr(lc, "get_fit_history"):
        try:
            write_json(output_dir / "fit_history.json", lc.get_fit_history())
        except Exception as exc:
            print("fit history export unavailable:", type(exc).__name__, exc)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        validate_configuration(args)
    except ValueError as exc:
        print("CONFIGURATION ERROR:", exc)
        return 1

    lc = load_lightcurve(args)
    fit_kwargs = build_fit_kwargs(args)
    configuration = _configuration_payload(args, fit_kwargs)

    print("Resolved fit configuration:")
    for key, value in fit_kwargs.items():
        print(f"  {key}: {value}")

    if args.output_dir is not None:
        write_json(args.output_dir / "run_configuration.json", configuration)

    if args.dry_run:
        print("\nDRY RUN: light curve loaded; GP training was not started.")
        return 0

    try:
        result = lc.fit(**fit_kwargs)
    except ConsensusFitError as exc:
        print("\nCONSENSUS FIT FAILED")
        print(type(exc).__name__ + ":", exc)
        if args.output_dir is not None:
            write_json(
                args.output_dir / "failure.json",
                _failure_payload(lc, exc, configuration),
            )
        return 2
    except Exception as exc:  # pragma: no cover - depends on numerical backend
        print("\nFIT FAILED")
        traceback.print_exc()
        if args.output_dir is not None:
            write_json(
                args.output_dir / "failure.json",
                _failure_payload(lc, exc, configuration),
            )
        return 1

    print("\nFIT PASSED")
    print("result type:", type(result).__name__)

    period_summary = None
    if hasattr(lc, "get_period_summary"):
        try:
            period_summary = lc.get_period_summary()
            print_period_summary(period_summary)
        except Exception as exc:  # period reporting is not fit success itself
            print("\nPeriod summary unavailable:", type(exc).__name__, exc)

    if args.output_dir is not None:
        _write_success_artifacts(args.output_dir, lc, period_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
