#!/usr/bin/env python3
"""Interpret exported PGMUVI JSON without running or importing a GP fit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def detect_report_type(payload: dict[str, Any]) -> str:
    kind = payload.get("kind")
    if kind == "period_independent_wavelength_advisory_workflow":
        return "advisory_workflow"
    if kind == "period_independent_wavelength_advisory_workflow_batch":
        return "advisory_batch"
    if "dominant_period" in payload and "peaks" in payload:
        return "period_summary"
    raise ValueError(
        "Unsupported JSON report. Expected a period summary, a single-source "
        "wavelength advisory workflow, or a wavelength advisory batch report."
    )


def interpret_period_summary(payload: dict[str, Any]) -> dict[str, Any]:
    peaks = [item for item in _as_list(payload.get("peaks")) if isinstance(item, dict)]
    primary_rank = payload.get("primary_peak_rank")
    largest_rank = payload.get("largest_area_peak_rank")
    warnings: list[str] = []

    if payload.get("dominant_period") is None:
        warnings.append("No dominant period is available for this kernel/report.")
    if primary_rank is not None and largest_rank is not None and primary_rank != largest_rank:
        warnings.append(
            "The physically ranked primary peak differs from the largest-area PSD feature; report both."
        )
    if len(peaks) > 1:
        warnings.append("Multiple analyzed peaks are present; inspect aliases, harmonics, and alternatives.")
    if payload.get("q_factor") is None:
        warnings.append("No finite q_factor/coherence proxy is available.")
    if isinstance(payload.get("component_diagnostics"), dict):
        warnings.append(
            "Spectral-mixture component diagnostics are internal kernel quantities, not independent final periods."
        )
    if payload.get("is_multicomponent"):
        component_summaries = [
            item
            for item in _as_list(payload.get("component_summaries"))
            if isinstance(item, dict)
        ]
        if any(
            item.get("fitted_period_drift_flag")
            or item.get("fitted_frequency_drift_flag")
            or item.get("component_identity_preserved") is False
            for item in component_summaries
        ):
            warnings.append("At least one consensus component has a drift or identity warning.")

    return {
        "report_type": "period_summary",
        "model_name": payload.get("model_name"),
        "method": payload.get("method"),
        "kernel_family": payload.get("kernel_family"),
        "time_kernel_family": payload.get("time_kernel_family"),
        "dominant_period": _finite_number(payload.get("dominant_period")),
        "dominant_frequency": _finite_number(payload.get("dominant_frequency")),
        "period_interval": payload.get("period_interval"),
        "interval_definition": payload.get("interval_definition"),
        "q_factor": _finite_number(payload.get("q_factor")),
        "primary_peak_rank": primary_rank,
        "largest_area_peak_rank": largest_rank,
        "largest_area_period": _finite_number(payload.get("largest_area_period")),
        "n_peaks": len(peaks),
        "n_significant_peaks": payload.get("n_significant_peaks"),
        "is_multicomponent": bool(payload.get("is_multicomponent")),
        "warnings": warnings,
    }


def _extract_advisory_outcomes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    run_report = _as_dict(payload.get("run_report"))
    outcomes = run_report.get("model_kernel_config_results")
    if not isinstance(outcomes, list):
        outcomes = run_report.get("outcomes")
    return [item for item in _as_list(outcomes) if isinstance(item, dict)]


def interpret_advisory_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    quality = _as_dict(payload.get("quality_report"))
    fallback = _as_dict(payload.get("fallback_report"))
    ranked = [item for item in _as_list(quality.get("ranked_results")) if isinstance(item, dict)]
    outcomes = _extract_advisory_outcomes(payload)

    constrained_rows = []
    for item in outcomes:
        count = item.get("n_constrained_sm_ard_components")
        try:
            count_value = int(count or 0)
        except (TypeError, ValueError):
            count_value = 0
        if count_value > 0:
            constrained_rows.append(
                {
                    "model": item.get("model"),
                    "model_kernel_config_id": item.get("model_kernel_config_id"),
                    "n_constrained_sm_ard_components": count_value,
                    "constrained_sm_ard_dimension_counts": item.get(
                        "constrained_sm_ard_dimension_counts"
                    ),
                }
            )

    failed = [
        item
        for item in outcomes
        if item.get("status") == "failed" or item.get("fit_failed") is True
    ]
    warnings = [
        "The ranking uses training-coordinate residual diagnostics, not held-out validation or model evidence.",
        "top_ranked_model is advisory; selected_model remains None unless a separate selection procedure is applied.",
    ]
    if fallback.get("available"):
        warnings.append("All attempted configurations failed; use fallback diagnostics instead of fit ranking.")
    if constrained_rows:
        warnings.append("One or more spectral-mixture ARD scales are near a consensus upper bound.")

    top_row = ranked[0] if ranked else {}
    return {
        "report_type": "advisory_workflow",
        "advisory_only": bool(payload.get("advisory_only", True)),
        "automatic_model_selection_applied": bool(
            payload.get("automatic_model_selection_applied", False)
        ),
        "selected_model": payload.get("selected_model"),
        "top_ranked_model": payload.get("top_ranked_model") or quality.get("top_ranked_model"),
        "top_ranked_fit_quality_score": _finite_number(
            payload.get("top_ranked_fit_quality_score")
            if payload.get("top_ranked_fit_quality_score") is not None
            else quality.get("top_ranked_fit_quality_score")
        ),
        "score_kind": payload.get("score_kind") or quality.get("score_kind"),
        "top_training_metrics": {
            "training_nrmse_by_target_scale": top_row.get(
                "training_nrmse_by_target_scale"
            ),
            "training_median_abs_standardized_residual": top_row.get(
                "training_median_abs_standardized_residual"
            ),
            "training_outlier_fraction_3sigma": top_row.get(
                "training_outlier_fraction_3sigma"
            ),
            "training_reduced_chi2": top_row.get("training_reduced_chi2"),
        },
        "n_model_kernel_configs": len(outcomes),
        "n_failed_model_kernel_configs": len(failed),
        "fallback_available": bool(fallback.get("available")),
        "fallback_reason": fallback.get("reason"),
        "failure_stage_counts": fallback.get("failure_stage_counts") or {},
        "exception_type_counts": fallback.get("exception_type_counts") or {},
        "constrained_sm_ard_rows": constrained_rows,
        "warnings": warnings,
    }


def interpret_advisory_batch(payload: dict[str, Any]) -> dict[str, Any]:
    source_results = [
        item for item in _as_list(payload.get("source_results")) if isinstance(item, dict)
    ]
    config_results = [
        item
        for item in _as_list(payload.get("model_kernel_config_results"))
        if isinstance(item, dict)
    ]
    source_failures = [item for item in source_results if item.get("status") == "failed"]
    config_failures = [
        item
        for item in config_results
        if item.get("status") == "failed" or item.get("fit_failed") is True
    ]
    ard_hits = []
    for item in config_results:
        try:
            count = int(item.get("n_constrained_sm_ard_components") or 0)
        except (TypeError, ValueError):
            count = 0
        if count:
            ard_hits.append(
                {
                    "source_id": item.get("source_id"),
                    "model": item.get("model"),
                    "n_constrained_sm_ard_components": count,
                    "constrained_sm_ard_dimension_counts": item.get(
                        "constrained_sm_ard_dimension_counts"
                    ),
                }
            )

    warnings = [
        "Batch score summaries do not correct for heterogeneous sampling or uncertainty calibration.",
        "The batch workflow is advisory and does not select a final model.",
    ]
    if source_failures:
        warnings.append("At least one source-level workflow failed.")
    if config_failures:
        warnings.append("At least one model/kernel configuration failed.")
    if ard_hits:
        warnings.append("At least one fitted configuration has an ARD scale-ceiling hit.")

    return {
        "report_type": "advisory_batch",
        "advisory_only": bool(payload.get("advisory_only", True)),
        "automatic_model_selection_applied": bool(
            payload.get("automatic_model_selection_applied", False)
        ),
        "selected_model": payload.get("selected_model"),
        "n_sources": payload.get("n_sources", len(source_results)),
        "n_succeeded": payload.get("n_succeeded"),
        "n_failed": payload.get("n_failed", len(source_failures)),
        "n_source_failure_rows": len(source_failures),
        "n_model_kernel_config_rows": len(config_results),
        "n_model_kernel_config_failures": len(config_failures),
        "ard_ceiling_hits": ard_hits,
        "warnings": warnings,
    }


def interpret_payload(payload: dict[str, Any]) -> dict[str, Any]:
    report_type = detect_report_type(payload)
    if report_type == "period_summary":
        return interpret_period_summary(payload)
    if report_type == "advisory_workflow":
        return interpret_advisory_workflow(payload)
    return interpret_advisory_batch(payload)


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def format_interpretation(summary: dict[str, Any]) -> str:
    report_type = summary["report_type"]
    lines = ["PGMUVI OUTPUT INTERPRETATION", "============================", f"report_type: {report_type}"]

    if report_type == "period_summary":
        for key in [
            "model_name",
            "method",
            "kernel_family",
            "time_kernel_family",
            "dominant_period",
            "dominant_frequency",
            "period_interval",
            "interval_definition",
            "q_factor",
            "primary_peak_rank",
            "largest_area_peak_rank",
            "largest_area_period",
            "n_peaks",
            "n_significant_peaks",
            "is_multicomponent",
        ]:
            lines.append(f"{key}: {_display(summary.get(key))}")
    elif report_type == "advisory_workflow":
        for key in [
            "top_ranked_model",
            "top_ranked_fit_quality_score",
            "score_kind",
            "selected_model",
            "automatic_model_selection_applied",
            "n_model_kernel_configs",
            "n_failed_model_kernel_configs",
            "fallback_available",
            "fallback_reason",
        ]:
            lines.append(f"{key}: {_display(summary.get(key))}")
        lines.append(
            "constrained_sm_ard_rows: "
            f"{len(_as_list(summary.get('constrained_sm_ard_rows')))}"
        )
    else:
        for key in [
            "n_sources",
            "n_succeeded",
            "n_failed",
            "n_source_failure_rows",
            "n_model_kernel_config_rows",
            "n_model_kernel_config_failures",
            "selected_model",
            "automatic_model_selection_applied",
        ]:
            lines.append(f"{key}: {_display(summary.get(key))}")
        lines.append(f"ard_ceiling_hits: {len(_as_list(summary.get('ard_ceiling_hits')))}")

    warnings = _as_list(summary.get("warnings"))
    lines.append("")
    lines.append("Interpretation cautions:")
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none recorded")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interpret exported PGMUVI JSON without running a fit."
    )
    parser.add_argument("report", type=Path, help="Path to an exported JSON report.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the normalized interpretation dictionary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("The top-level JSON value must be an object.")
        summary = interpret_payload(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(format_interpretation(summary))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
