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

        from pgmuvi import instrument_channel_calibration

        self.assertIn(
            "InstrumentChannelCalibration",
            instrument_channel_calibration.__all__,
        )
        self.assertIn(
            "INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION",
            instrument_channel_calibration.__all__,
        )

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

    def test_page_documents_explicit_paired_calibration(self):
        text = PAGE.read_text(encoding="utf-8")
        self.assertIn("caller-supplied paired", text)
        self.assertIn("does not construct time pairs", text)
        self.assertIn("affine", text.lower())
        self.assertIn(
            "does not propagate uncertainty in the fitted offset or scale",
            " ".join(text.split()),
        )

    def test_future_work_marker_remains_open(self):
        text = " ".join(FUTURE.read_text(encoding="utf-8").split())

        self.assertIn("TBD[instrument-channel-calibration]", text)
        self.assertIn("caller-supplied paired measurements", text)
        self.assertIn("does not close", text)
        self.assertIn("scientific validation", text)

    def test_estimation_and_multiband_guides_reference_contract(self):
        estimation = ESTIMATION.read_text(encoding="utf-8")
        multiband = MULTIBAND.read_text(encoding="utf-8")

        self.assertIn(
            "pgmuvi.instrument_channel_calibration",
            estimation,
        )
        self.assertIn(
            "No instrument-channel calibration is performed by this "
            "wavelength-estimation workflow",
            " ".join(estimation.split()),
        )
        self.assertIn("caller-supplied paired measurements", estimation)
        self.assertIn(
            "pgmuvi.instrument_channel_calibration",
            multiband,
        )
        normalized_multiband = " ".join(multiband.split())
        self.assertIn("caller-supplied paired", normalized_multiband)
        self.assertIn(
            "do not construct time pairs",
            normalized_multiband,
        )
        self.assertNotIn("NotImplementedError", normalized_multiband)


if __name__ == "__main__":
    unittest.main()
