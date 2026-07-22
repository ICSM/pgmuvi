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
            "InstrumentChannelPairing",
            instrument_channel_calibration.__all__,
        )
        self.assertIn(
            "INSTRUMENT_CHANNEL_CALIBRATION_MODEL_SCHEMA_VERSION",
            instrument_channel_calibration.__all__,
        )
        self.assertIn(
            "InstrumentChannelCalibrationCoefficientUncertainty",
            instrument_channel_calibration.__all__,
        )
        self.assertIn(
            "INSTRUMENT_CHANNEL_CALIBRATION_UNCERTAINTY_SCHEMA_VERSION",
            instrument_channel_calibration.__all__,
        )
        self.assertIn(
            "INSTRUMENT_CHANNEL_CALIBRATION_SCALE_DEPENDENT_UNCERTAINTY_SCHEMA_VERSION",
            instrument_channel_calibration.__all__,
        )
        self.assertIn(
            "InstrumentChannelCalibrationScaleDependentUncertaintyEstimate",
            instrument_channel_calibration.__all__,
        )
        self.assertIn(
            "estimate_scale_dependent_instrument_channel_calibration_coefficient_uncertainty",
            instrument_channel_calibration.__all__,
        )
        self.assertIn(
            "InstrumentChannelCalibrationPredictiveUncertainty",
            instrument_channel_calibration.__all__,
        )
        self.assertIn(
            "INSTRUMENT_CHANNEL_CALIBRATION_PREDICTIVE_UNCERTAINTY_SCHEMA_VERSION",
            instrument_channel_calibration.__all__,
        )
        self.assertIn(
            "apply_instrument_channel_calibration_with_predictive_uncertainty",
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
        self.assertIn("construct_instrument_channel_pairing", text)
        self.assertIn("row indices", text)
        self.assertIn("time-separation", text)
        self.assertIn("nearest_within_tolerance", text)
        self.assertIn(
            "define_instrument_channel_calibration_plan",
            text,
        )
        self.assertIn("planned", text)
        self.assertIn("skipped", text)
        self.assertIn("unavailable", text)
        self.assertIn("Dataset-level orchestration", text)
        self.assertIn("does not construct pairs", text)
        self.assertIn(
            "execute_instrument_channel_calibration_plan",
            text,
        )
        self.assertIn(
            "InstrumentChannelCalibrationExecution",
            text,
        )
        self.assertIn("does not mutate input arrays", text)
        self.assertIn(
            "fitted-coefficient uncertainty",
            text,
        )
        self.assertIn("O(n_reference * n_channel)", text)
        self.assertIn("time and memory cost", text)
        self.assertIn("affine", text.lower())
        normalized = " ".join(text.split())
        self.assertIn(
            "InstrumentChannelCalibrationCoefficientUncertainty",
            text,
        )
        self.assertIn("fixed coefficient order", normalized)
        self.assertIn("offset, scale", normalized)
        self.assertIn("complete symmetric 2-by-2 covariance", normalized)
        self.assertIn("explicit uncertainty source", normalized)
        self.assertIn("unavailable", normalized)
        self.assertIn(
            "ordinary-least-squares normal-matrix inverse",
            normalized,
        )
        self.assertIn(
            "Coefficient covariance is unavailable when channel-axis errors "
            "are supplied",
            normalized,
        )
        self.assertIn(
            "does not expose a full-objective Hessian",
            normalized,
        )
        self.assertIn(
            "local and conditional",
            normalized,
        )
        self.assertIn(
            "Scale-dependent channel-axis uncertainty contract",
            text,
        )
        self.assertIn(
            "InstrumentChannelCalibrationScaleDependentUncertaintyEstimate",
            text,
        )
        self.assertIn(
            "estimate_scale_dependent_instrument_channel_calibration_"
            "coefficient_uncertainty",
            text,
        )
        self.assertIn(
            "estimator is implemented with analytic objective derivatives",
            normalized,
        )
        self.assertIn(
            "log-scale parameterization",
            normalized,
        )
        self.assertIn(
            "inverse observed Hessian of this full objective",
            normalized,
        )
        self.assertIn(
            "not a frozen-weight normal-matrix inverse",
            normalized,
        )
        self.assertIn(
            "v_i=\\sigma_{y,i}^2+a^2\\sigma_{x,i}^2",
            normalized,
        )
        self.assertIn(
            "does not propagate uncertainty in the fitted offset or scale",
            normalized,
        )
        self.assertIn(
            "InstrumentChannelCalibrationPredictiveUncertainty",
            text,
        )
        self.assertIn(
            "apply_instrument_channel_calibration_with_predictive_uncertainty",
            text,
        )
        self.assertIn(
            "returns an ``InstrumentChannelCalibrationPredictiveUncertainty`` "
            "record",
            normalized,
        )
        predictive_section = text.split(
            "Predictive-uncertainty propagation",
            maxsplit=1,
        )[1].split(
            "``TBD[instrument-channel-calibration]`` remains open",
            maxsplit=1,
        )[0]
        self.assertNotIn("NotImplementedError", predictive_section)
        self.assertIn("J_i=[1, x_i]", normalized)
        self.assertIn("offset-scale covariance cross-term", normalized)
        self.assertIn("full_covariance", text)
        self.assertIn("marginal_variance", text)
        self.assertIn(
            "``not_requested``, ``available``, ``skipped``, and "
            "``unavailable``",
            normalized,
        )
        self.assertIn(
            "no silent fallback to measurement-only uncertainty",
            normalized.lower(),
        )

    def test_future_work_marker_remains_open(self):
        text = " ".join(FUTURE.read_text(encoding="utf-8").split())

        self.assertIn("TBD[instrument-channel-calibration]", text)
        self.assertIn("low-level API", text)
        self.assertIn("caller-tolerance", text)
        self.assertIn("does not close", text)
        self.assertIn(
            "representative calibration validation",
            text,
        )
        self.assertIn(
            "implements the dedicated full-objective",
            text,
        )
        self.assertIn(
            "without activating it in the fitter or orchestration layer",
            text,
        )
        self.assertNotIn(
            "predictive-propagation implementation",
            text,
        )
        self.assertIn(
            "implements dedicated marginal or full-covariance predictive "
            "propagation without orchestration activation",
            text,
        )
        self.assertNotIn(
            "uncertainty estimation and propagation",
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
        self.assertIn(
            "preserve their provenance",
            normalized_multiband,
        )
        self.assertIn(
            "construct deterministic one-to-one",
            normalized_multiband,
        )
        self.assertIn(
            "does not choose the reference channel",
            normalized_multiband,
        )
        self.assertNotIn("NotImplementedError", normalized_multiband)


if __name__ == "__main__":
    unittest.main()
