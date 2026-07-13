#!/usr/bin/env python3
"""Minimal consensus multiband fitting example.

This script demonstrates the current flagship direct fitting pattern:

    lc.fit(model="2D", fit_strategy="consensus", learn_additional_noise=True)

It can run on a CSV file or on a small synthetic multi-band light curve.  The
synthetic example is intended as a usage template, not as a scientific benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pgmuvi.lightcurve import Lightcurve


def make_synthetic_lpv_lightcurve(n_per_band: int = 60) -> Lightcurve:
    """Create a small coherent 2-D light curve for demonstration."""
    rng = np.random.default_rng(12345)
    wavelengths = np.array([0.55, 0.80, 1.25], dtype=float)
    period = 120.0

    x_rows = []
    y_vals = []
    yerr_vals = []
    band_labels = []

    for i, wavelength in enumerate(wavelengths):
        time = np.sort(rng.uniform(0.0, 4.0 * period, size=n_per_band))
        amplitude = 1.0 + 0.35 * i
        mean = 10.0 + 0.4 * i
        err = np.full_like(time, 0.08 + 0.02 * i, dtype=float)
        phase = 2.0 * np.pi * time / period
        flux = mean + amplitude * np.sin(phase) + rng.normal(0.0, err)

        x_rows.append(np.column_stack([time, np.full_like(time, wavelength)]))
        y_vals.append(flux)
        yerr_vals.append(err)
        band_labels.append(np.full(time.shape, f"band_{wavelength:g}", dtype=object))

    return Lightcurve(
        xdata=np.vstack(x_rows),
        ydata=np.concatenate(y_vals),
        yerr=np.concatenate(yerr_vals),
        band=np.concatenate(band_labels),
        check_sampling=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal multiband consensus GP fit.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Optional source CSV. If omitted, a synthetic light curve is used.",
    )
    parser.add_argument("--model", default="2D", help="Model name to fit.")
    parser.add_argument("--training-iter", type=int, default=50)
    parser.add_argument("--miniter", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-samples-per-band", type=int, default=100)
    parser.add_argument(
        "--learn-additional-noise",
        action="store_true",
        default=True,
        help="Learn an additional noise term on top of supplied yerr values.",
    )
    parser.add_argument(
        "--no-learn-additional-noise",
        dest="learn_additional_noise",
        action="store_false",
        help="Disable learned additional noise.",
    )
    parser.add_argument(
        "--time-kernel-type",
        default=None,
        help=(
            "Optional time kernel type, e.g. quasi_periodic for separable "
            "LPV-relevant models such as 2DDustMean."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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



def _get_field(obj, key, default=None):
    """Read *key* from either a dict-like result or an attribute object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)


def _format_value(value, precision: int = 6) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{precision}g}"
    except Exception:
        return str(value)


def print_period_summary(period_summary) -> None:
    """Print selected PeriodSummaryResult fields in a user-readable form."""
    if hasattr(period_summary, "as_dict"):
        summary = period_summary.as_dict()
    elif isinstance(period_summary, dict):
        summary = dict(period_summary)
    else:
        summary = {}

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
            index = _get_field(component, "component_index")
            period = _get_field(component, "consensus_period")
            strength = _get_field(component, "consensus_component_strength")
            bands = _get_field(component, "member_bands", [])
            print(
                "    "
                f"component={index} "
                f"period={_format_value(period)} "
                f"strength={_format_value(strength)} "
                f"bands={bands}"
            )

def main() -> None:
    args = parse_args()
    lc = load_lightcurve(args)

    fit_kwargs = {
        "model": args.model,
        "fit_strategy": "consensus",
        "training_iter": args.training_iter,
        "miniter": args.miniter,
        "learn_additional_noise": args.learn_additional_noise,
        "verbose": args.verbose,
    }
    if args.time_kernel_type is not None:
        fit_kwargs["time_kernel_type"] = args.time_kernel_type

    print("Running consensus fit with kwargs:")
    for key, value in fit_kwargs.items():
        print(f"  {key}: {value}")

    result = lc.fit(**fit_kwargs)
    print("\nFIT PASSED")
    print("result type:", type(result).__name__)

    if hasattr(lc, "get_period_summary"):
        try:
            period_summary = lc.get_period_summary()
            print_period_summary(period_summary)
        except Exception as exc:  # pragma: no cover - diagnostic output path
            print("\nPeriod summary unavailable:", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
