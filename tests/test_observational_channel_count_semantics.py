"""Regression tests for wavelength evidence and channel counts."""

import unittest

from pgmuvi.parameter_context import (
    WavelengthMeanEstimationDiagnostics,
)


class TestObservationalChannelCountSemantics(unittest.TestCase):
    """Keep channel counts distinct from mean-fit evidence counts."""

    def test_preferred_property_uses_channel_metadata(self):
        diagnostics = WavelengthMeanEstimationDiagnostics(
            n_usable_bands=3,
            metadata={
                "n_usable_observational_channels": 5,
            },
        )

        self.assertEqual(
            diagnostics.n_usable_observational_channels,
            5,
        )
        self.assertEqual(diagnostics.n_usable_bands, 3)

    def test_legacy_payload_falls_back_to_evidence_count(self):
        diagnostics = WavelengthMeanEstimationDiagnostics(
            n_usable_bands=3,
        )

        self.assertEqual(
            diagnostics.n_usable_observational_channels,
            3,
        )


if __name__ == "__main__":
    unittest.main()
