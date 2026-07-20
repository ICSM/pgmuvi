#!/usr/bin/env python3
"""Run the failure-aware D3 representative observed-LPV protocol."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pgmuvi.wavelength_validation_real_lpv import (
    DEFAULT_REPRESENTATIVE_LPV_MODELS,
    RepresentativeLPVSourceSpecification,
    export_representative_lpv_batch_report,
    run_representative_lpv_validation_batch,
    validate_representative_lpv_source_manifest,
)


def _read_json(json_path: Path) -> Any:
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exception:
        raise ValueError(
            f"JSON file does not exist: {json_path}"
        ) from exception
    except json.JSONDecodeError as exception:
        raise ValueError(
            f"Invalid JSON in {json_path}: {exception}"
        ) from exception


def load_representative_lpv_manifest(
    manifest_path: str | Path,
) -> tuple[RepresentativeLPVSourceSpecification, ...]:
    """Load, validate, and resolve one ordered source manifest."""
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    payload = _read_json(resolved_manifest_path)

    if isinstance(payload, Mapping):
        entries = payload.get("sources")
        if entries is None:
            raise ValueError(
                "Manifest objects must contain a 'sources' list."
            )
    else:
        entries = payload

    specifications = validate_representative_lpv_source_manifest(
        entries
    )
    resolved_specifications = []

    for specification in specifications:
        source_path = Path(specification.source_path).expanduser()
        if not source_path.is_absolute():
            source_path = (
                resolved_manifest_path.parent / source_path
            ).resolve()

        resolved_specifications.append(
            RepresentativeLPVSourceSpecification(
                source_id=specification.source_id,
                source_path=str(source_path),
                description=specification.description,
                sample_role=specification.sample_role,
                selection_reason=specification.selection_reason,
                seed=specification.seed,
                metadata={
                    **dict(specification.metadata),
                    "manifest_path": str(resolved_manifest_path),
                    "manifest_source_path": (
                        specification.source_path
                    ),
                },
            )
        )

    return tuple(resolved_specifications)


def load_representative_lpv_workflow_configuration(
    configuration_path: str | Path | None,
) -> dict[str, Any]:
    """Load optional workflow keyword arguments from strict JSON."""
    if configuration_path is None:
        return {}

    resolved_path = Path(configuration_path).expanduser().resolve()
    payload = _read_json(resolved_path)
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Workflow configuration JSON must contain an object."
        )
    return dict(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the failure-aware, advisory-only D3 representative "
            "observed-LPV validation protocol."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help=(
            "JSON source manifest. Relative source paths are resolved "
            "relative to this file."
        ),
    )
    parser.add_argument(
        "--workflow-config",
        help=(
            "Optional JSON mapping passed as workflow_kwargs to the "
            "D3 batch runner."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="validation_outputs/d3_real_lpv",
        help=(
            "Output directory used by the report exporter. Default: "
            "validation_outputs/d3_real_lpv"
        ),
    )
    parser.add_argument(
        "--batch-id",
        default="d3-representative-observed-lpv",
        help="Stable identifier stored in the batch report.",
    )
    parser.add_argument(
        "--validate-manifest-only",
        action="store_true",
        help=(
            "Validate and resolve the manifest without loading sources, "
            "running fits, or writing outputs."
        ),
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help=(
            "Run the protocol but do not serialize its report artifacts."
        ),
    )
    parser.add_argument(
        "--fail-on-source-failure",
        action="store_true",
        help=(
            "Return exit status 2 after reporting when one or more "
            "sources failed. Later sources are still attempted."
        ),
    )
    return parser


def _print_json(payload: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    batch_runner: Callable[..., Any] = (
        run_representative_lpv_validation_batch
    ),
    exporter: Callable[..., Mapping[str, Any]] = (
        export_representative_lpv_batch_report
    ),
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        specifications = load_representative_lpv_manifest(
            arguments.manifest
        )
        workflow_configuration = (
            load_representative_lpv_workflow_configuration(
                arguments.workflow_config
            )
        )
    except (TypeError, ValueError) as exception:
        parser.error(str(exception))

    if arguments.validate_manifest_only:
        _print_json(
            {
                "kind": (
                    "representative_lpv_manifest_validation"
                ),
                "valid": True,
                "manifest_path": str(
                    Path(arguments.manifest)
                    .expanduser()
                    .resolve()
                ),
                "n_sources": len(specifications),
                "source_ids": [
                    item.source_id for item in specifications
                ],
                "models": list(
                    DEFAULT_REPRESENTATIVE_LPV_MODELS
                ),
                "runs_fits": False,
                "writes_outputs": False,
                "automatic_model_selection_applied": False,
                "selected_model": None,
            }
        )
        return 0

    batch_report = batch_runner(
        specifications,
        workflow_kwargs=workflow_configuration,
        output_root=arguments.output_root,
        batch_id=arguments.batch_id,
    )
    batch_payload = batch_report.to_dict()

    export_manifest = None
    if not arguments.no_export:
        export_manifest = exporter(
            batch_report,
            arguments.output_root,
        )

    result = {
        "kind": "representative_lpv_cli_result",
        "batch_id": batch_payload["batch_id"],
        "n_sources": batch_payload["n_sources"],
        "n_completed_sources": (
            batch_payload["n_completed_sources"]
        ),
        "n_failed_sources": batch_payload["n_failed_sources"],
        "source_status_counts": (
            batch_payload["source_status_counts"]
        ),
        "models": batch_payload["models"],
        "output_root": arguments.output_root,
        "exported": export_manifest is not None,
        "export_manifest": export_manifest,
        "advisory_only": True,
        "automatic_model_selection_applied": False,
        "selected_model": None,
    }
    _print_json(result)

    if (
        arguments.fail_on_source_failure
        and batch_payload["n_failed_sources"] > 0
    ):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
