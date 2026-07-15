#!/usr/bin/env python3
"""Load, validate, and summarize a PGMUVI light-curve CSV file.

This example deliberately separates input hygiene from sampling gates and
subsampling so users can see how many rows each stage retains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pgmuvi.lightcurve import Lightcurve


def _optional_positive_filter(
    lightcurve: Lightcurve,
    *,
    drop_nonpositive_flux: bool = False,
    drop_nonpositive_errors: bool = False,
) -> tuple[Lightcurve, dict[str, int]]:
    """Optionally remove non-positive flux or uncertainty rows.

    The base Lightcurve constructor intentionally does not impose a universal
    positive-flux rule.  This helper applies only the explicit policy requested
    by the caller.
    """

    x = lightcurve.xdata.detach().cpu().numpy()
    y = lightcurve.ydata.detach().cpu().numpy()
    yerr_tensor = getattr(lightcurve, "_yerr_raw", None)
    yerr = None if yerr_tensor is None else yerr_tensor.detach().cpu().numpy()

    keep = np.ones(y.shape[0], dtype=bool)
    dropped_flux = 0
    dropped_error = 0

    if drop_nonpositive_flux:
        positive_flux = y > 0.0
        dropped_flux = int(np.count_nonzero(~positive_flux))
        keep &= positive_flux

    if drop_nonpositive_errors:
        if yerr is None:
            raise ValueError(
                "--drop-nonpositive-errors requires an uncertainty column."
            )
        positive_error = yerr > 0.0
        dropped_error = int(np.count_nonzero(~positive_error))
        keep &= positive_error

    if not np.any(keep):
        raise ValueError("No rows remain after the requested positive-value filters.")

    band = lightcurve.band
    if band is not None and lightcurve.ndim > 1:
        band = np.asarray(band)[keep]

    filtered = Lightcurve(
        x[keep],
        y[keep],
        None if yerr is None else yerr[keep],
        name=lightcurve.name,
        band=band,
        max_samples=None,
        max_samples_per_band=None,
        check_sampling=False,
        check_variability=False,
    )

    return filtered, {
        "n_rows_before_positive_filter": int(y.shape[0]),
        "n_rows_after_positive_filter": int(np.count_nonzero(keep)),
        "n_dropped_nonpositive_flux": dropped_flux,
        "n_dropped_nonpositive_error": dropped_error,
    }


def load_and_validate(
    csv_path: str | Path,
    *,
    xcol: str | None = None,
    ycol: str | None = None,
    yerrcol: str | None = None,
    wavelcol: str | None = None,
    drop_nonpositive_flux: bool = False,
    drop_nonpositive_errors: bool = False,
    check_sampling: bool = False,
    max_samples: int | None = 1000,
    max_samples_per_band: int | None = None,
    subsample_seed: int | None = None,
) -> tuple[Lightcurve, dict[str, Any]]:
    """Load a CSV, apply explicit hygiene policy, and run validation gates."""

    initial = Lightcurve.from_csv(
        csv_path,
        xcol=xcol,
        ycol=ycol,
        yerrcol=yerrcol,
        wavelcol=wavelcol,
        check_sampling=False,
        check_variability=False,
        max_samples=None,
        max_samples_per_band=None,
    )

    filtered, positive_report = _optional_positive_filter(
        initial,
        drop_nonpositive_flux=drop_nonpositive_flux,
        drop_nonpositive_errors=drop_nonpositive_errors,
    )

    x = filtered.xdata.detach().cpu().numpy()
    y = filtered.ydata.detach().cpu().numpy()
    yerr_tensor = getattr(filtered, "_yerr_raw", None)
    yerr = None if yerr_tensor is None else yerr_tensor.detach().cpu().numpy()

    validated = Lightcurve(
        x,
        y,
        yerr,
        name=filtered.name,
        band=filtered.band,
        check_sampling=check_sampling,
        max_samples=max_samples,
        max_samples_per_band=max_samples_per_band,
        subsample_seed=subsample_seed,
    )

    summary: dict[str, Any] = {
        "csv_path": str(csv_path),
        "ndim": int(validated.ndim),
        "n_rows_final": int(validated.ydata.numel()),
        "has_uncertainties": bool(hasattr(validated, "_yerr_raw")),
        "has_band_labels": bool(validated.band is not None),
        "check_sampling": bool(check_sampling),
        "max_samples": max_samples,
        "max_samples_per_band": max_samples_per_band,
        "subsample_seed": subsample_seed,
        **positive_report,
    }

    if validated.ndim > 1:
        wavelengths = validated.xdata[:, 1].detach().cpu().numpy()
        summary["n_wavelengths"] = int(np.unique(wavelengths).size)
        summary["sampling_metrics"] = validated.compute_sampling_metrics_per_band()
    else:
        summary["n_wavelengths"] = 1
        summary["sampling_metrics"] = validated.compute_sampling_metrics()

    return validated, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Input light-curve CSV file.")
    parser.add_argument("--xcol", help="Explicit time-column name.")
    parser.add_argument("--ycol", help="Explicit measurement-column name.")
    parser.add_argument("--yerrcol", help="Explicit uncertainty-column name.")
    parser.add_argument("--wavelcol", help="Explicit numeric wavelength-column name.")
    parser.add_argument(
        "--drop-nonpositive-flux",
        action="store_true",
        help="Drop rows whose measurement is zero or negative.",
    )
    parser.add_argument(
        "--drop-nonpositive-errors",
        action="store_true",
        help="Drop rows whose uncertainty is zero or negative.",
    )
    parser.add_argument(
        "--check-sampling",
        action="store_true",
        help="Apply constructor-time sampling gates after input filtering.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="1-D subsampling limit or 2-D total-size advisory threshold.",
    )
    parser.add_argument(
        "--max-samples-per-band",
        type=int,
        default=None,
        help="Optional 2-D per-wavelength subsampling limit.",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=None,
        help="Seed for reproducible constructor-time subsampling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _, summary = load_and_validate(
        args.csv_path,
        xcol=args.xcol,
        ycol=args.ycol,
        yerrcol=args.yerrcol,
        wavelcol=args.wavelcol,
        drop_nonpositive_flux=args.drop_nonpositive_flux,
        drop_nonpositive_errors=args.drop_nonpositive_errors,
        check_sampling=args.check_sampling,
        max_samples=args.max_samples,
        max_samples_per_band=args.max_samples_per_band,
        subsample_seed=args.subsample_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
