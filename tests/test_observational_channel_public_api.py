"""Public API contracts for observational-channel terminology."""

from __future__ import annotations

import unittest

import numpy as np

from pgmuvi.lightcurve import Lightcurve
from pgmuvi.parameter_context import (
    BandDiagnostics,
    ObservationalChannelDiagnostics,
)


class TestObservationalChannelPublicAPI(unittest.TestCase):
    def test_preferred_diagnostics_name_is_public_alias(self):
        self.assertIs(ObservationalChannelDiagnostics, BandDiagnostics)

    def test_lightcurve_exposes_observational_channel_labels(self):
        xdata = np.asarray(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 2.0],
                [1.0, 2.0],
            ]
        )
        flux = np.asarray([1.0, 1.1, 2.0, 2.1])
        error = np.full(4, 0.1)
        labels = np.asarray(
            ["channel-a", "channel-a", "channel-b", "channel-b"]
        )

        lightcurve = Lightcurve(
            xdata,
            flux,
            error,
            band=labels,
            center_time=False,
            max_samples=None,
        )

        self.assertIs(
            lightcurve.observational_channel_labels,
            lightcurve.band,
        )
        self.assertEqual(
            lightcurve.observational_channel_labels.tolist(),
            labels.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
