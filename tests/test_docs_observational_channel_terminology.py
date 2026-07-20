"""Documentation contracts for observational-channel terminology."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LOADING = ROOT / "docs/source/howto/loading_data.rst"
MULTIBAND = ROOT / "docs/source/howto/multiband.rst"
ESTIMATION = ROOT / "docs/source/pgmuvi.wavelength_estimation.rst"
FUTURE = ROOT / "docs/source/future_work.rst"


class TestObservationalChannelDocumentation(unittest.TestCase):
    def test_loading_guide_defines_channel_and_wavelength_roles(self):
        text = " ".join(LOADING.read_text(encoding="utf-8").split())

        self.assertIn("observational-channel label", text)
        self.assertIn("physical wavelength coordinate", text)
        self.assertIn(
            "multiple observational channels may share one physical wavelength",
            text,
        )
        self.assertIn(
            "Lightcurve.band is retained as the legacy attribute",
            text,
        )

    def test_multiband_guide_uses_distinct_terms(self):
        text = " ".join(MULTIBAND.read_text(encoding="utf-8").split())

        self.assertIn("observational channel", text)
        self.assertIn("physical wavelength", text)
        self.assertIn(
            "observational-channel labels do not replace the numeric physical "
            "wavelength coordinate",
            text,
        )
        self.assertIn("TBD[instrument-channel-calibration]", text)

    def test_wavelength_estimation_documents_channel_level_evidence(self):
        text = " ".join(ESTIMATION.read_text(encoding="utf-8").split())

        self.assertIn("per-observational-channel", text)
        self.assertIn("distinct physical wavelengths", text)
        self.assertIn("TBD[instrument-channel-calibration]", text)
        self.assertIn("No instrument-channel calibration is performed", text)

    def test_future_work_registers_calibration_tbd(self):
        text = FUTURE.read_text(encoding="utf-8")

        self.assertIn("TBD[instrument-channel-calibration]", text)
        self.assertIn("observational channels", text)
        self.assertIn("physical wavelength", text)


if __name__ == "__main__":
    unittest.main()
