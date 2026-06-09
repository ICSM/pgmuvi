import unittest

from pgmuvi.parameter_context import (
    BandDiagnostics,
    ConsensusDiagnostics,
    LightcurveDiagnostics,
    ParameterEstimationContext,
)


class TestParameterContext(unittest.TestCase):
    def test_lightcurve_diagnostics_stores_global_quantities(self):
        diagnostics = LightcurveDiagnostics(
            baseline=1000.0,
            cadence=10.0,
            median_flux=123.0,
            mad_flux=5.0,
            flux_percentiles={2.5: 100.0, 50.0: 125.0, 97.5: 150.0},
            n_points=42,
        )

        self.assertEqual(diagnostics.baseline, 1000.0)
        self.assertEqual(diagnostics.cadence, 10.0)
        self.assertEqual(diagnostics.median_flux, 123.0)
        self.assertEqual(
            diagnostics.flux_percentiles,
            {
                2.5: 100.0,
                50.0: 125.0,
                97.5: 150.0,
            },
        )

    def test_band_diagnostics_stores_per_band_quantities(self):
        diagnostics = BandDiagnostics(
            band="WISE_W1",
            wavelength=3.4,
            baseline=900.0,
            cadence=8.0,
            median_flux=200.0,
            mad_flux=12.0,
            n_points=30,
        )

        self.assertEqual(diagnostics.band, "WISE_W1")
        self.assertEqual(diagnostics.wavelength, 3.4)
        self.assertEqual(diagnostics.median_flux, 200.0)

    def test_consensus_diagnostics_stores_period_information(self):
        diagnostics = ConsensusDiagnostics(
            method="consensus_multicomp",
            periods=[300.0, 1200.0],
            frequencies=[1.0 / 300.0, 1.0 / 1200.0],
            powers=[0.8, 0.5],
            component_indices=[0, 1],
        )

        self.assertEqual(diagnostics.method, "consensus_multicomp")
        self.assertEqual(diagnostics.periods, [300.0, 1200.0])
        self.assertEqual(diagnostics.component_indices, [0, 1])

    def test_context_stores_global_and_band_diagnostics(self):
        global_diagnostics = LightcurveDiagnostics(median_flux=100.0)

        w1 = BandDiagnostics(band="WISE_W1", median_flux=120.0)
        w2 = BandDiagnostics(band="WISE_W2", median_flux=140.0)

        context = ParameterEstimationContext(
            is_multiband=True,
            global_diagnostics=global_diagnostics,
            band_diagnostics={
                "WISE_W1": w1,
                "WISE_W2": w2,
            },
        )

        self.assertIs(context.is_multiband, True)
        self.assertIs(context.global_diagnostics, global_diagnostics)
        self.assertEqual(context.bands(), ["WISE_W1", "WISE_W2"])
        self.assertIs(context.get_band("WISE_W1"), w1)

    def test_context_raises_for_missing_band(self):
        context = ParameterEstimationContext(
            is_multiband=True,
            band_diagnostics={"WISE_W1": BandDiagnostics(band="WISE_W1")},
        )

        with self.assertRaisesRegex(KeyError, "No diagnostics found"):
            context.get_band("WISE_W2")
