import pytest

from pgmuvi.parameter_context import (
    BandDiagnostics,
    ConsensusDiagnostics,
    LightcurveDiagnostics,
    ParameterEstimationContext,
)


def test_lightcurve_diagnostics_stores_global_quantities():
    diagnostics = LightcurveDiagnostics(
        baseline=1000.0,
        cadence=10.0,
        median_flux=123.0,
        mad_flux=5.0,
        flux_percentiles=(100.0, 150.0),
        n_points=42,
    )

    assert diagnostics.baseline == 1000.0
    assert diagnostics.cadence == 10.0
    assert diagnostics.median_flux == 123.0
    assert diagnostics.flux_percentiles == (100.0, 150.0)


def test_band_diagnostics_stores_per_band_quantities():
    diagnostics = BandDiagnostics(
        band="WISE_W1",
        wavelength=3.4,
        baseline=900.0,
        cadence=8.0,
        median_flux=200.0,
        mad_flux=12.0,
        n_points=30,
    )

    assert diagnostics.band == "WISE_W1"
    assert diagnostics.wavelength == 3.4
    assert diagnostics.median_flux == 200.0


def test_consensus_diagnostics_stores_period_information():
    diagnostics = ConsensusDiagnostics(
        method="consensus_multicomp",
        periods=[300.0, 1200.0],
        frequencies=[1.0 / 300.0, 1.0 / 1200.0],
        powers=[0.8, 0.5],
        component_indices=[0, 1],
    )

    assert diagnostics.method == "consensus_multicomp"
    assert diagnostics.periods == [300.0, 1200.0]
    assert diagnostics.component_indices == [0, 1]


def test_parameter_estimation_context_stores_global_and_band_diagnostics():
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

    assert context.is_multiband is True
    assert context.global_diagnostics is global_diagnostics
    assert context.bands() == ["WISE_W1", "WISE_W2"]
    assert context.get_band("WISE_W1") is w1


def test_parameter_estimation_context_raises_for_missing_band():
    context = ParameterEstimationContext(
        is_multiband=True,
        band_diagnostics={"WISE_W1": BandDiagnostics(band="WISE_W1")},
    )

    with pytest.raises(KeyError, match="No diagnostics found"):
        context.get_band("WISE_W2")