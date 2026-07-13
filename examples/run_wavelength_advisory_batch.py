#!/usr/bin/env python3
"""Run the period-independent wavelength advisory workflow for one or more CSVs.

This is an example/CLI wrapper around
``Lightcurve.run_period_independent_wavelength_advisory_workflow_batch``.  It
runs the advisory model/kernel-config workflow per source, exports per-source
reports/figures when requested, and writes a batch JSON/CSV summary.  It remains
advisory-only: no model is selected or installed automatically.

Examples
--------
Run one source::

    PYTHONPATH=. python3 examples/run_wavelength_advisory_batch.py \
        10131+3049.csv \
        --output-dir wavelength_advisory_batch \
        --training-iter 50 \
        --miniter 10

Run several sources from a text file::

    PYTHONPATH=. python3 examples/run_wavelength_advisory_batch.py \
        --source-list sources.txt \
        --output-dir wavelength_advisory_batch

``sources.txt`` may contain one source per line as either ``path/to/file.csv``
or ``source_id=path/to/file.csv``.  CSV source lists with ``source_id`` and
``csv_path`` columns are also accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

def _default_source_id(csv_path: str) -> str:
    """Return a readable default source id for a CSV path."""
    stem = Path(csv_path).stem
    return stem or str(csv_path)


def _source_spec_from_token(token: str) -> dict[str, str]:
    """Parse one CLI source token.

    Accepted forms are ``path/to/source.csv`` and
    ``source_id=path/to/source.csv``.
    """
    text = str(token).strip()
    if not text:
        raise ValueError("empty source token")

    if "=" in text:
        source_id, csv_path = text.split("=", 1)
        source_id = source_id.strip()
        csv_path = csv_path.strip()
        if not source_id or not csv_path:
            raise ValueError(
                "source tokens using '=' must have the form source_id=csv_path"
            )
        return {"source_id": source_id, "csv_path": csv_path}

    return {"source_id": _default_source_id(text), "csv_path": text}


def _source_spec_from_csv_row(row: dict[str, Any]) -> dict[str, str]:
    """Parse one CSV source-list row."""
    csv_path = (
        row.get("csv_path")
        or row.get("path")
        or row.get("filename")
        or row.get("file")
    )
    if csv_path is None or not str(csv_path).strip():
        raise ValueError(
            "CSV source-list rows must include a csv_path, path, filename, or file column"
        )
    csv_path = str(csv_path).strip()
    source_id = str(row.get("source_id") or row.get("id") or _default_source_id(csv_path)).strip()
    return {"source_id": source_id, "csv_path": csv_path}


def _read_source_list(path: str | Path) -> list[dict[str, str]]:
    """Read source specs from JSON, CSV, or newline-delimited text."""
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")

    if source_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON source lists must contain a list")
        out: list[dict[str, str]] = []
        for item in payload:
            if isinstance(item, str):
                out.append(_source_spec_from_token(item))
            elif isinstance(item, dict):
                out.append(_source_spec_from_csv_row(item))
            else:
                raise ValueError("JSON source-list entries must be strings or objects")
        return out

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return []

    first_fields = [field.strip() for field in lines[0].split(",")]
    header_names = {field.lower() for field in first_fields}
    looks_like_header = bool(
        header_names & {"csv_path", "path", "filename", "file"}
    )

    if looks_like_header:
        reader = csv.DictReader(lines)
        return [_source_spec_from_csv_row(row) for row in reader]

    out = []
    for line in lines:
        if "," in line and "=" not in line:
            source_id, csv_path = [part.strip() for part in line.split(",", 1)]
            if not source_id or not csv_path:
                raise ValueError(
                    "comma source-list rows without a header must be source_id,csv_path"
                )
            out.append({"source_id": source_id, "csv_path": csv_path})
        else:
            out.append(_source_spec_from_token(line))
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the period-independent wavelength advisory model/kernel-config "
            "workflow for one or more light-curve CSV files."
        )
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help="CSV paths, optionally as source_id=path/to/source.csv.",
    )
    parser.add_argument(
        "--source-list",
        action="append",
        default=[],
        help=(
            "Text/CSV/JSON file listing sources. Text lines may be csv_path, "
            "source_id=csv_path, or source_id,csv_path. CSV files should include "
            "csv_path/path and optionally source_id. May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="wavelength_advisory_batch",
        help="Directory for per-source exports and batch summaries.",
    )
    parser.add_argument(
        "--batch-prefix",
        default="wavelength_advisory_batch",
        help="Prefix for the batch summary JSON/CSV files.",
    )
    parser.add_argument(
        "--training-iter",
        type=int,
        default=50,
        help="Training iterations for each advisory model/kernel config.",
    )
    parser.add_argument(
        "--miniter",
        type=int,
        default=10,
        help="Minimum iterations for each advisory model/kernel config.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples passed to Lightcurve.from_csv.",
    )
    parser.add_argument(
        "--max-samples-per-band",
        type=int,
        default=100,
        help="Maximum samples per band passed to Lightcurve.from_csv.",
    )
    parser.add_argument(
        "--no-check-sampling",
        action="store_true",
        help="Disable sampling checks in Lightcurve.from_csv.",
    )
    parser.add_argument(
        "--model-kernel-config-limit",
        type=int,
        default=None,
        help="Optional maximum number of advisory model/kernel configs to evaluate.",
    )
    parser.add_argument(
        "--no-include-2d-baseline",
        action="store_true",
        help="Do not include the full 2D baseline model/kernel config.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not make/export workflow comparison figures.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run the batch workflow but do not export per-source workflow outputs.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Re-raise the first source failure instead of recording it and continuing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass verbose=True to from_csv and per-config fits.",
    )
    parser.add_argument(
        "--keep-figures-open",
        action="store_true",
        help="Do not close matplotlib figures after exporting them.",
    )
    parser.add_argument(
        "--allow-source-failures",
        action="store_true",
        help="Exit with status 0 even if one or more source rows fail.",
    )
    return parser


def _collect_sources(args: argparse.Namespace) -> list[dict[str, str]]:
    sources = [_source_spec_from_token(token) for token in args.csv_paths]
    for path in args.source_list:
        sources.extend(_read_source_list(path))
    return sources


def _print_batch_summary(report: dict[str, Any]) -> None:
    """Print a concise terminal summary of a batch manifest."""
    print("kind:", report.get("kind"))
    print("advisory_only:", report.get("advisory_only"))
    print("runs_fits:", report.get("runs_fits"))
    print("applies_to_fit:", report.get("applies_to_fit"))
    print("model_kernel_config_state_isolated:", report.get("model_kernel_config_state_isolated"))
    print("mutates_input_lightcurve:", report.get("mutates_input_lightcurve"))
    print("automatic_model_selection_applied:", report.get("automatic_model_selection_applied"))
    print("selected_model:", report.get("selected_model"))
    print("n_sources:", report.get("n_sources"))
    print("n_succeeded:", report.get("n_succeeded"))
    print("n_failed:", report.get("n_failed"))
    print("batch_json_path:", report.get("batch_json_path"))
    print("batch_csv_path:", report.get("batch_csv_path"))

    print("\nSource rows:")
    for row in report.get("source_results", []):
        print(
            row.get("source_id"),
            row.get("status"),
            row.get("top_ranked_model"),
            row.get("top_ranked_fit_quality_score"),
            row.get("score_kind"),
            "n_model_kernel_configs=", row.get("n_model_kernel_configs"),
            "n_successful_model_kernel_configs=", row.get("n_successful_model_kernel_configs"),
            "n_failed_model_kernel_configs=", row.get("n_failed_model_kernel_configs"),
            "error=", row.get("exception_type"), row.get("exception_message"),
        )


def _run_batch(sources, **kwargs):
    """Delegate to the package batch helper with a lazy import.

    Keeping this import lazy lets parser/unit tests exercise this script without
    importing the full GPyTorch stack.
    """
    from pgmuvi.lightcurve import Lightcurve as LC

    return LC.run_period_independent_wavelength_advisory_workflow_batch(
        sources, **kwargs
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    sources = _collect_sources(args)
    if not sources:
        parser.error("provide at least one CSV path or --source-list file")

    from_csv_kwargs = {
        "check_sampling": not args.no_check_sampling,
        "max_samples": args.max_samples,
        "max_samples_per_band": args.max_samples_per_band,
        "verbose": bool(args.verbose),
    }
    workflow_kwargs = {
        "include_2d_baseline": not args.no_include_2d_baseline,
        "base_fit_kwargs": {
            "training_iter": args.training_iter,
            "miniter": args.miniter,
            "verbose": bool(args.verbose),
        },
        "model_kernel_config_limit": args.model_kernel_config_limit,
        "make_text_report": True,
        "make_plots": not args.no_plots,
    }
    export_kwargs = {
        "close_figures": not args.keep_figures_open,
    }

    report = _run_batch(
        sources,
        from_csv_kwargs=from_csv_kwargs,
        workflow_kwargs=workflow_kwargs,
        output_dir=args.output_dir,
        export=not args.no_export,
        export_kwargs=export_kwargs,
        batch_prefix=args.batch_prefix,
        stop_on_error=args.stop_on_error,
    )
    _print_batch_summary(report)

    if int(report.get("n_failed") or 0) and not args.allow_source_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
