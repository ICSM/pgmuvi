#!/usr/bin/env python3
"""Prepare and optionally run a deterministic toy wavelength-advisory batch.

The generated light curves are plumbing examples only.  They are not intended
for scientific validation or model-ranking claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable


BANDS = (
    ("J", 1.25, 1.00, 0.18),
    ("H", 1.65, 0.86, 0.14),
    ("K", 2.20, 0.74, 0.10),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create two deterministic toy multiwavelength light curves and "
            "optionally run the existing wavelength-advisory batch CLI."
        )
    )
    parser.add_argument(
        "--workspace",
        default="wavelength_advisory_batch_walkthrough",
        help="Directory for generated inputs, manifest, and batch outputs.",
    )
    parser.add_argument(
        "--n-points-per-band",
        type=int,
        default=36,
        help="Number of synthetic observations generated for each band.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1729,
        help="Base random seed used for deterministic cadence jitter and noise.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing walkthrough workspace.",
    )
    parser.add_argument(
        "--run-batch",
        action="store_true",
        help="Run the existing batch CLI after preparing the toy inputs.",
    )
    parser.add_argument(
        "--training-iter",
        type=int,
        default=20,
        help="Training iterations passed to each toy advisory config.",
    )
    parser.add_argument(
        "--miniter",
        type=int,
        default=5,
        help="Minimum iterations passed to each toy advisory config.",
    )
    parser.add_argument(
        "--model-kernel-config-limit",
        type=int,
        default=1,
        help="Maximum configs evaluated per source during the toy smoke run.",
    )
    parser.add_argument(
        "--allow-source-failures",
        action="store_true",
        help=(
            "Pass through the batch CLI option that returns status 0 even when "
            "one or more source rows fail."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_points_per_band < 8:
        raise ValueError("--n-points-per-band must be at least 8")
    if args.training_iter < 0:
        raise ValueError("--training-iter must be non-negative")
    if args.miniter < 0:
        raise ValueError("--miniter must be non-negative")
    if args.miniter > args.training_iter:
        raise ValueError("--miniter cannot exceed --training-iter")
    if args.model_kernel_config_limit < 1:
        raise ValueError("--model-kernel-config-limit must be at least 1")


def _source_rows(
    *,
    period: float,
    phase: float,
    seed: int,
    n_points_per_band: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    for band_index, (band, wavelength, baseline, amplitude) in enumerate(BANDS):
        for point_index in range(n_points_per_band):
            cadence = 23.0 + 1.7 * band_index
            time = (
                point_index * cadence
                + 4.0 * band_index
                + rng.uniform(-2.5, 2.5)
            )
            flux_error = 0.025 * baseline
            signal = baseline * (
                1.0
                + amplitude
                * math.sin((2.0 * math.pi * time / period) + phase)
            )
            flux = signal + rng.gauss(0.0, flux_error)
            rows.append(
                {
                    "time": time,
                    "wavelength": wavelength,
                    "flux": max(flux, 0.05 * baseline),
                    "flux_error": flux_error,
                    "band": band,
                }
            )

    rows.sort(key=lambda row: (float(row["time"]), str(row["band"])))
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time", "wavelength", "flux", "flux_error", "band"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "time": f"{float(row['time']):.8f}",
                    "wavelength": f"{float(row['wavelength']):.8f}",
                    "flux": f"{float(row['flux']):.8f}",
                    "flux_error": f"{float(row['flux_error']):.8f}",
                    "band": str(row["band"]),
                }
            )


def _prepare_workspace(
    workspace: Path,
    *,
    overwrite: bool,
    seed: int,
    n_points_per_band: int,
) -> dict[str, object]:
    if workspace.exists():
        if not overwrite:
            raise FileExistsError(
                f"workspace already exists: {workspace}; use --overwrite to replace it"
            )
        import shutil

        shutil.rmtree(workspace)

    inputs_dir = workspace / "inputs"
    outputs_dir = workspace / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    source_definitions = (
        ("toy_lpv_a", 310.0, 0.25, seed),
        ("toy_lpv_b", 455.0, 1.10, seed + 1),
    )

    sources = []
    for source_id, period, phase, source_seed in source_definitions:
        csv_path = inputs_dir / f"{source_id}.csv"
        rows = _source_rows(
            period=period,
            phase=phase,
            seed=source_seed,
            n_points_per_band=n_points_per_band,
        )
        _write_csv(csv_path, rows)
        sources.append(
            {
                "source_id": source_id,
                "csv_path": str(csv_path.resolve()),
                "period_used_to_generate_days": period,
                "n_rows": len(rows),
            }
        )

    source_list_path = workspace / "sources.txt"
    source_list_path.write_text(
        "".join(
            f"{item['source_id']}={item['csv_path']}\n" for item in sources
        ),
        encoding="utf-8",
    )

    return {
        "kind": "wavelength_advisory_batch_walkthrough",
        "scientific_use_warning": (
            "Synthetic inputs exercise file layout, batch orchestration, exports, "
            "and failure handling only; they do not validate scientific ranking."
        ),
        "workspace": str(workspace.resolve()),
        "inputs_dir": str(inputs_dir.resolve()),
        "outputs_dir": str(outputs_dir.resolve()),
        "source_list_path": str(source_list_path.resolve()),
        "n_sources": len(sources),
        "n_points_per_band": n_points_per_band,
        "bands": [band for band, _, _, _ in BANDS],
        "wavelengths": [wavelength for _, wavelength, _, _ in BANDS],
        "sources": sources,
    }


def _batch_command(
    manifest: dict[str, object],
    *,
    training_iter: int,
    miniter: int,
    model_kernel_config_limit: int,
    allow_source_failures: bool,
) -> list[str]:
    script_path = Path(__file__).resolve().with_name(
        "run_wavelength_advisory_batch.py"
    )
    command = [
        sys.executable,
        str(script_path),
        "--source-list",
        str(manifest["source_list_path"]),
        "--output-dir",
        str(manifest["outputs_dir"]),
        "--batch-prefix",
        "toy_batch",
        "--training-iter",
        str(training_iter),
        "--miniter",
        str(miniter),
        "--model-kernel-config-limit",
        str(model_kernel_config_limit),
        "--no-check-sampling",
        "--no-plots",
    ]
    if allow_source_failures:
        command.append("--allow-source-failures")
    return command


def _write_manifest(
    workspace: Path,
    manifest: dict[str, object],
    command: list[str],
) -> Path:
    payload = dict(manifest)
    payload["batch_command"] = command
    path = workspace / "walkthrough_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _run_command(command: list[str]) -> int:
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        _validate_args(args)
        workspace = Path(args.workspace)
        manifest = _prepare_workspace(
            workspace,
            overwrite=bool(args.overwrite),
            seed=args.seed,
            n_points_per_band=args.n_points_per_band,
        )
        command = _batch_command(
            manifest,
            training_iter=args.training_iter,
            miniter=args.miniter,
            model_kernel_config_limit=args.model_kernel_config_limit,
            allow_source_failures=bool(args.allow_source_failures),
        )
        manifest_path = _write_manifest(workspace, manifest, command)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))

    print("prepared workspace:", manifest["workspace"])
    print("source list:", manifest["source_list_path"])
    print("manifest:", manifest_path.resolve())
    print("batch command:")
    print(" ".join(command))

    if not args.run_batch:
        print("\nPREPARE ONLY: batch fitting was not started.")
        return 0

    print("\nRUNNING TOY BATCH")
    return _run_command(command)


if __name__ == "__main__":
    raise SystemExit(main())
