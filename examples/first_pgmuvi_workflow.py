#!/usr/bin/env python3
"""Create a toy input file and print the first PGMUVI workflow commands.

This example is intentionally lightweight: it creates a small synthetic
multi-band CSV and prints the first commands a user would run. It does not launch
an expensive GP fit by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_toy_multiband_csv(path: Path, n_per_band: int = 40) -> None:
    """Write a small, positive-flux, multi-band toy light curve."""
    rng = np.random.default_rng(20260714)
    wavelengths = [0.55, 0.80, 1.25]
    bands = ["V", "I", "J"]
    period = 120.0

    rows = ["time,wavelength,band,flux,flux_error"]
    for index, (wavelength, band) in enumerate(zip(wavelengths, bands)):
        time = np.sort(rng.uniform(0.0, 4.0 * period, n_per_band))
        mean = 10.0 + 0.5 * index
        amplitude = 0.8 + 0.25 * index
        flux_error = 0.08 + 0.02 * index
        phase = 2.0 * np.pi * time / period
        flux = mean + amplitude * np.sin(phase) + rng.normal(0.0, flux_error, n_per_band)
        for t, f in zip(time, flux):
            rows.append(f"{t:.8f},{wavelength:.6g},{band},{f:.8f},{flux_error:.8f}")

    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a toy multi-band CSV and print first PGMUVI commands."
    )
    parser.add_argument(
        "--output-dir",
        default="first_workflow_demo",
        help="Directory where the toy CSV and suggested-output folders are placed.",
    )
    parser.add_argument(
        "--csv-name",
        default="toy_multiband_lpv.csv",
        help="Name of the generated toy CSV file.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.csv_name
    write_toy_multiband_csv(csv_path)

    print(f"Wrote toy multi-band CSV: {csv_path}")
    print("\nFirst commands to try from the repository root:\n")
    print("1. Load and inspect the source in Python:")
    print("   from pgmuvi.lightcurve import Lightcurve")
    print(
        "   lc = Lightcurve.from_csv(" 
        f"{str(csv_path)!r}, xcol='time', ycol='flux', "
        "yerrcol='flux_error', wavelcol='wavelength', "
        "check_sampling=True, verbose=True)"
    )
    print("\n2. Run the wavelength advisory batch wrapper on the toy source:")
    print(
        "   PYTHONPATH=. python3 examples/run_wavelength_advisory_batch.py "
        f"toy={csv_path} --output-dir {output_dir / 'wavelength_advisory'} "
        "--training-iter 20 --miniter 5"
    )
    print("\n3. Once diagnostics look reasonable, run a conservative baseline fit:")
    print("   result = lc.fit(model='2D', fit_strategy='consensus',")
    print("                   training_iter=500, miniter=100,")
    print("                   learn_additional_noise=True, verbose=True)")
    print("\nThis script is a workflow template, not a scientific benchmark.")


if __name__ == "__main__":
    main()
