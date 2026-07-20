"""Public API and documentation contracts for D3 real-LPV validation."""

from __future__ import annotations

import unittest
from pathlib import Path

import pgmuvi


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "source"


def _normalized(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


class TestRepresentativeLPVPublicAPI(unittest.TestCase):
    def test_module_is_listed_in_package_public_api(self):
        self.assertIn(
            "wavelength_validation_real_lpv",
            pgmuvi.__all__,
        )


class TestRepresentativeLPVAPIReference(unittest.TestCase):
    def test_api_toctree_includes_real_lpv_module(self):
        text = (DOCS / "api.rst").read_text(encoding="utf-8")
        self.assertIn(
            "pgmuvi.wavelength_validation_real_lpv",
            text,
        )

    def test_real_lpv_api_page_exists_and_uses_automodule(self):
        path = DOCS / "pgmuvi.wavelength_validation_real_lpv.rst"
        self.assertTrue(path.is_file())

        normalized = _normalized(path)
        self.assertIn(
            ".. automodule:: pgmuvi.wavelength_validation_real_lpv",
            normalized,
        )
        self.assertIn(":members:", normalized)
        self.assertIn(":undoc-members:", normalized)
        self.assertIn(":show-inheritance:", normalized)


class TestRepresentativeLPVHowToGuide(unittest.TestCase):
    def test_howto_index_includes_real_lpv_validation_guide(self):
        text = (DOCS / "howto" / "index.rst").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "representative_lpv_validation",
            text,
        )

    def test_real_lpv_validation_guide_documents_contract(self):
        path = DOCS / "howto" / "representative_lpv_validation.rst"
        self.assertTrue(path.is_file())

        normalized = _normalized(path)

        required = (
            "D3",
            "representative observed LPV",
            "examples/data/10131+3049.csv",
            "linear flux",
            "observational channel",
            "physical wavelength",
            "TBD[instrument-channel-calibration]",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
            "2DSeparable",
            "2D",
            'fit_strategy="consensus"',
            'time_kernel_type="quasi_periodic"',
            "learn_additional_noise=True",
            "run_representative_lpv_validation",
            "automatic model selection",
            "selected_model",
            "truth-recovery",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, normalized)

        self.assertNotIn("2DAchromatic", normalized)

    def test_guide_distinguishes_ranked_evidence_from_selection(self):
        normalized = _normalized(
            DOCS / "howto" / "representative_lpv_validation.rst"
        )
        self.assertIn(
            "A top-ranked model is descriptive evidence",
            normalized,
        )
        self.assertIn(
            "does not install or select that model",
            normalized,
        )

    def test_guide_documents_shared_wavelength_policy(self):
        normalized = _normalized(
            DOCS / "howto" / "representative_lpv_validation.rst"
        )
        self.assertIn(
            "multiple observational channels may share one physical wavelength",
            normalized,
        )
        self.assertIn(
            "does not average, merge, calibrate, correct, or reassign",
            normalized,
        )



    def test_guide_documents_failure_aware_batch_protocol(self):
        normalized = _normalized(
            DOCS / "howto" / "representative_lpv_validation.rst"
        )

        required = (
            "RepresentativeLPVSourceSpecification",
            "run_representative_lpv_validation_batch",
            "Manifest order is preserved",
            "Duplicate source identifiers are rejected",
            "does not abort later manifest entries",
            "validation_outputs/d3_real_lpv/sources/<source_id>",
            "writes_outputs=False",
            "does not automatically apply wavelength-derived constraints",
            "initialization",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, normalized)

    def test_guide_documents_reproducible_execution_provenance(self):
        normalized = _normalized(
            DOCS / "howto" / "representative_lpv_validation.rst"
        )

        required = (
            "seed is applied across source loading and advisory execution",
            "Python, NumPy, and PyTorch",
            "RNG streams advanced",
            "exact workflow configuration",
            "compact runtime provenance",
            "GPyTorch",
            "package-version information",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, normalized)

    def test_guide_documents_command_line_runner(self):
        normalized = _normalized(
            DOCS / "howto" / "representative_lpv_validation.rst"
        )

        required = (
            "scripts/run_representative_lpv_validation.py",
            "d3_representative_lpv_manifest.json",
            "d3_representative_lpv_workflow.json",
            "--validate-manifest-only",
            "resolves relative source paths",
            "writes no outputs",
            "--fail-on-source-failure",
            "status 2",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, normalized)

    def test_guide_documents_report_export_layer(self):
        normalized = _normalized(
            DOCS / "howto" / "representative_lpv_validation.rst"
        )

        required = (
            "export_representative_lpv_batch_report",
            "strict JSON",
            "source-summary CSV",
            "source_result.json",
            "report.json",
            "failure.json",
            "Filesystem path components are sanitized",
            "does not rerun fits",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, normalized)

    def test_guide_does_not_claim_batch_runner_writes_outputs(self):
        normalized = _normalized(
            DOCS / "howto" / "representative_lpv_validation.rst"
        )

        self.assertIn(
            "The protocol runner does not create those directories or write report files",
            normalized,
        )
        self.assertNotIn(
            "The protocol runner writes report files",
            normalized,
        )


class TestRepresentativeLPVFutureWorkStatus(unittest.TestCase):
    def test_future_work_no_longer_calls_d3_the_next_unimplemented_step(self):
        normalized = _normalized(DOCS / "future_work.rst")

        self.assertNotIn(
            "The next validation step is D3 application",
            normalized,
        )
        self.assertIn(
            "D3 representative observed-LPV validation infrastructure",
            normalized,
        )
        self.assertIn(
            "TBD[instrument-channel-calibration]",
            normalized,
        )


if __name__ == "__main__":
    unittest.main()
