"""Validation script for the wavelength model-selection diagnostics workflow.

This script is intentionally not a unit test.  It is a deterministic smoke /
validation run that exercises the public science-facing workflow on a synthetic
multiwavelength long-period-variable-like light curve:

1. build a 2D chromatic sinusoid,
2. run pre-fit wavelength diagnostics,
3. format the Markdown report,
4. create diagnostic plots,
5. optionally run candidate GP model comparison,
6. format and plot the comparison report.

The default mode avoids GP fitting and should run quickly.  Use
``--run-model-comparison`` for a fuller, slower validation that exercises the
normal Lightcurve.fit() pathway through compare_wavelength_models().

Examples
--------
Quick pre-fit validation::

    PYTHONPATH=. python3 examples/validate_wavelength_model_selection_workflow.py

Full comparison validation::

    PYTHONPATH=. python3 examples/validate_wavelength_model_selection_workflow.py \
        --run-model-comparison --training-iter 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    """Convert nested report values into JSON-serialisable objects."""
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _assert_prefit_report(diag: dict[str, Any]) -> None:
    """Fail early if the pre-fit diagnostics did not produce useful evidence."""
    summary = diag.get("summary", {})
    rows = diag.get("band_table", [])
    classification = diag.get("classification", {})
    candidates = diag.get("recommended_candidate_models", [])
    amp_summary = diag.get("amplitude_phase_summary", {})

    if diag.get("kind") != "wavelength_dependence_prefit_diagnostics":
        raise AssertionError("unexpected diagnostic report kind")
    if summary.get("n_bands", 0) < 3:
        raise AssertionError("expected at least three wavelength bands")
    if len(rows) != summary.get("n_bands"):
        raise AssertionError("band_table length does not match n_bands")
    if not amp_summary.get("available"):
        raise AssertionError("fixed-period amplitude/phase diagnostics unavailable")
    if classification.get("primary_class") in {None, "insufficient_wavelength_data"}:
        raise AssertionError("classification did not produce a usable primary class")
    if not candidates:
        raise AssertionError("no candidate model recommendations were produced")

    fixed_amplitudes = []
    for row in rows:
        fixed = row.get("fixed_frequency_diagnostics", {})
        amp = fixed.get("amplitude")
        if amp is not None and np.isfinite(float(amp)):
            fixed_amplitudes.append(float(amp))
    if len(fixed_amplitudes) < 3:
        raise AssertionError("expected fixed-frequency amplitudes for >=3 bands")
    if max(fixed_amplitudes) <= min(fixed_amplitudes):
        raise AssertionError("fixed-frequency amplitudes show no wavelength spread")


def _assert_comparison_report(comparison: dict[str, Any]) -> None:
    """Fail early if the optional model-comparison run did not execute."""
    if comparison.get("kind") != "wavelength_model_comparison":
        raise AssertionError("unexpected comparison report kind")
    summary = comparison.get("summary", {})
    if summary.get("n_fit_candidates", 0) < 1:
        raise AssertionError("comparison did not include any fit candidates")
    if not comparison.get("results"):
        raise AssertionError("comparison returned no candidate results")
    if "interpretation" not in comparison:
        raise AssertionError("comparison report is missing interpretation block")


def _save_figures(figures: dict[str, Any], output_dir: Path, prefix: str) -> None:
    """Save Matplotlib figures if plotting is available."""
    for name, fig in figures.items():
        path = output_dir / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the wavelength model-selection diagnostics workflow."
    )
    parser.add_argument(
        "--output-dir",
        default="wavelength_diagnostics_validation",
        help="Directory for JSON, Markdown, and plot outputs.",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=120.0,
        help="Synthetic period used for the generated light curve.",
    )
    parser.add_argument(
        "--run-model-comparison",
        action="store_true",
        help="Also run candidate GP fits through compare_wavelength_models().",
    )
    parser.add_argument(
        "--training-iter",
        type=int,
        default=8,
        help="Training iterations for optional model-comparison fits.",
    )
    parser.add_argument(
        "--miniter",
        type=int,
        default=0,
        help="Minimum iterations for optional model-comparison fits.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip Matplotlib figure creation and saving.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_plots:
        # Keep validation usable on headless systems.
        import matplotlib

        matplotlib.use("Agg", force=True)

    from pgmuvi.synthetic import make_chromatic_sinusoid_2d

    lc = make_chromatic_sinusoid_2d(
        n_per_band=[45, 48, 50, 46, 43],
        period=args.period,
        amplitude=1.0,
        wavelengths=[0.9, 1.25, 1.65, 2.2, 3.4],
        amplitude_law="linear",
        amplitude_slope=0.35,
        wl_ref=0.9,
        phase_law="linear",
        phase_slope=0.04,
        noise_level=0.04,
        noise_type="gaussian",
        t_span=4.0 * args.period,
        irregular=True,
        seed=4927,
    )

    diag = lc.diagnose_wavelength_dependence(period=args.period)
    _assert_prefit_report(diag)

    diagnostic_markdown = lc.format_wavelength_diagnostics_report(
        diag,
        max_band_rows=10,
    )
    (output_dir / "wavelength_diagnostics_prefit.md").write_text(
        diagnostic_markdown,
        encoding="utf-8",
    )
    (output_dir / "wavelength_diagnostics_prefit.json").write_text(
        json.dumps(_json_safe(diag), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not args.no_plots:
        diagnostic_figs = lc.plot_wavelength_diagnostics(diag, show=False)
        _save_figures(diagnostic_figs, output_dir, "prefit")
    else:
        diagnostic_figs = {}

    comparison = None
    comparison_figs = {}
    if args.run_model_comparison:
        # Use a deliberately small candidate set and short training run so this
        # remains a smoke validation rather than a full scientific comparison.
        candidates = [
            {
                "name": "baseline_2d_consensus_smoke",
                "model": "2D",
                "fit_strategy": "consensus",
                "reason": "Stabilized baseline 2D consensus model.",
            },
            {
                "name": "smooth_wavelength_dependent_smoke",
                "model": "2DWavelengthDependent",
                "reason": "Interpretable smooth wavelength-dependent candidate.",
            },
        ]
        comparison = lc.compare_wavelength_models(
            diagnostic_report=diag,
            candidates=candidates,
            base_fit_kwargs={
                "training_iter": args.training_iter,
                "miniter": args.miniter,
                "learn_additional_noise": True,
                "verbose": False,
            },
            residual_diagnostic_kwargs={"period": args.period},
            stop_on_error=False,
        )
        _assert_comparison_report(comparison)
        comparison_markdown = lc.format_wavelength_diagnostics_report(
            diag,
            comparison_report=comparison,
            max_band_rows=10,
        )
        (output_dir / "wavelength_model_comparison.md").write_text(
            comparison_markdown,
            encoding="utf-8",
        )
        (output_dir / "wavelength_model_comparison.json").write_text(
            json.dumps(_json_safe(comparison), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not args.no_plots:
            comparison_figs = lc.plot_wavelength_model_comparison(
                comparison,
                show=False,
            )
            _save_figures(comparison_figs, output_dir, "comparison")

    print("Wavelength diagnostics validation passed.")
    print(f"Output directory: {output_dir}")
    print(f"Primary class: {diag.get('classification', {}).get('primary_class')}")
    print(f"Recommended candidates: {len(diag.get('recommended_candidate_models', []))}")
    print(f"Diagnostic figures: {sorted(diagnostic_figs)}")
    if comparison is not None:
        summary = comparison.get("summary", {})
        print(
            "Comparison candidates: "
            f"{summary.get('n_successful', 0)} successful, "
            f"{summary.get('n_failed', 0)} failed, "
            f"{summary.get('n_skipped', 0)} skipped"
        )
        print(f"Comparison figures: {sorted(comparison_figs)}")


if __name__ == "__main__":
    main()
