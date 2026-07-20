"""Documentation contracts for instrument-channel calibration."""

from __future__ import annotations

from pathlib import Path
import unittest

import pgmuvi


DOCS = Path("docs/source")
API = DOCS / "api.rst"
PAGE = DOCS / "pgmuvi.instrument_channel_calibration.rst"
FUTURE = DOCS / "future_work.rst"
ESTIMATION = DOCS / "pgmuvi.wavelength_estimation.rst"
MULTIBAND = DOCS / "howto/multiband.rst"


class TestInstrumentChannelCalibrationDocumentation(unittest.TestCase):
    def test_module_is_public_and_has_an_api_page(self):
        self.assertIn(
            "instrument_channel_calibration",
            pgmuvi.__all__,
        )
        self.assertTrue(PAGE.exists())

        api_text = API.read_text(encoding="utf-8")
        page_text = PAGE.read_text(encoding="utf-8")

        self.assertIn(
            "pgmuvi.instrument_channel_calibration",
            api_text,
        )
        self.assertIn(
            ".. automodule:: pgmuvi.instrument_channel_calibration",
            page_text,
        )

    def test_page_documents_explicit_unsupported_callables(self):
        text = " ".join(PAGE.read_text(encoding="utf-8").split())

        self.assertIn("TBD[instrument-channel-calibration]", text)
        self.assertIn("fit_instrument_channel_calibration", text)
        self.assertIn("apply_instrument_channel_calibration", text)
        self.assertIn("NotImplementedError", text)
        self.assertIn("must not silently infer", text)

    def test_future_work_marker_remains_open(self):
        text = " ".join(FUTURE.read_text(encoding="utf-8").split())

        self.assertIn("TBD[instrument-channel-calibration]", text)
        self.assertIn(
            "does not close the marker or constitute a calibration "
            "implementation",
            text,
        )

    def test_estimation_and_multiband_guides_reference_contract(self):
        estimation = ESTIMATION.read_text(encoding="utf-8")
        multiband = MULTIBAND.read_text(encoding="utf-8")

        self.assertIn(
            "pgmuvi.instrument_channel_calibration",
            estimation,
        )
        self.assertIn(
            "pgmuvi.instrument_channel_calibration",
            multiband,
        )
        self.assertIn("NotImplementedError", multiband)


if __name__ == "__main__":
    unittest.main()
