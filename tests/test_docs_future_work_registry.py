"""Regression contracts for the documentation future-work registry."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "source"
REGISTRY = SOURCE / "future_work.rst"


class TestDocumentationFutureWorkRegistry(unittest.TestCase):
    def test_registry_exists_and_is_linked(self) -> None:
        registry = REGISTRY.read_text(encoding="utf-8")
        index = (SOURCE / "index.rst").read_text(encoding="utf-8")
        roadmap = (SOURCE / "documentation_roadmap.rst").read_text(
            encoding="utf-8"
        )
        maintenance = (SOURCE / "docs_maintenance.rst").read_text(
            encoding="utf-8"
        )

        self.assertIn("Documentation status and future work", registry)
        self.assertIn("future_work", index)
        self.assertIn(":doc:`future_work`", roadmap)
        self.assertIn("audit_docs_tbd_markers.py", maintenance)

    def test_required_future_work_markers_are_registered(self) -> None:
        text = REGISTRY.read_text(encoding="utf-8")
        required = [
            "TBD[held-out-validation]",
            "TBD[batch-validation]",
            "TBD[batch-notebook]",
            "TBD[automatic-model-selection]",
            "TBD[multi-periodic-wavelength-models]",
            "TBD[non-monotonic-wavelength-kernels]",
            "TBD[physical-wavelength-kernels]",
            "TBD[wavelength-dependent-lags]",
            "TBD[mcmc-implementation]",
            "TBD[native-magnitude-input]",
            "TBD[multidimensional-psd-plotting]",
        ]
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, text)
        self.assertNotIn("TBD[result-interpretation-notebook]", text)
        self.assertIn(
            "Completed result-interpretation notebook coverage (PR174)",
            text,
        )

    def test_relevant_pages_own_the_registered_markers(self) -> None:
        advisory = (SOURCE / "howto" / "wavelength_advisory.rst").read_text(
            encoding="utf-8"
        )
        batch = (SOURCE / "howto" / "wavelength_advisory_batch.rst").read_text(
            encoding="utf-8"
        )
        loading = (SOURCE / "howto" / "loading_data.rst").read_text(
            encoding="utf-8"
        )
        multiband = (SOURCE / "howto" / "multiband.rst").read_text(
            encoding="utf-8"
        )
        interpretation = (
            SOURCE / "howto" / "interpreting_results.rst"
        ).read_text(encoding="utf-8")
        notebook_status = (SOURCE / "notebook_status.rst").read_text(
            encoding="utf-8"
        )

        self.assertIn("TBD[held-out-validation]", advisory)
        self.assertIn("TBD[automatic-model-selection]", advisory)
        self.assertIn("TBD[multi-periodic-wavelength-models]", advisory)
        self.assertIn("TBD[non-monotonic-wavelength-kernels]", advisory)
        self.assertIn("TBD[physical-wavelength-kernels]", advisory)
        self.assertIn("TBD[wavelength-dependent-lags]", advisory)
        self.assertIn("TBD[batch-validation]", batch)
        self.assertIn("TBD[batch-notebook]", batch)
        self.assertNotIn(
            "TBD[result-interpretation-notebook]",
            interpretation,
        )
        self.assertIn(
            "tutorial_single_source_analysis",
            interpretation,
        )
        self.assertIn("TBD[native-magnitude-input]", loading)
        self.assertIn("TBD[multidimensional-psd-plotting]", multiband)
        self.assertIn("TBD[mcmc-implementation]", notebook_status)

    def test_roadmap_marks_all_expansion_areas_complete(self) -> None:
        text = (SOURCE / "documentation_roadmap.rst").read_text(
            encoding="utf-8"
        )
        for pr in ["PR97", "PR98", "PR99", "PR100", "PR101", "PR102", "PR103--PR108", "PR109"]:
            with self.subTest(pr=pr):
                self.assertIn(pr, text)
        self.assertIn("the PR97--PR109 expansion sequence is complete", text)

    def test_stale_planned_documentation_promises_are_removed(self) -> None:
        consensus = (SOURCE / "howto" / "consensus_fitting.rst").read_text(
            encoding="utf-8"
        )
        legacy = (
            SOURCE / "howto" / "legacy_wavelength_candidates.rst"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            "results/reporting documentation planned for a later PR",
            consensus,
        )
        self.assertIn(":doc:`interpreting_results`", consensus)
        self.assertNotIn(
            "A dedicated guide for that current workflow is planned",
            legacy,
        )
        self.assertNotIn(
            "Updated wavelength-advisory and consensus-fitting tutorials are planned",
            legacy,
        )
        self.assertIn("tutorial_wavelength_advisory", legacy)
        self.assertIn("pgmuvi_tutorial_2d", legacy)


if __name__ == "__main__":
    unittest.main()
