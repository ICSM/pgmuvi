"""Prospective instrument-channel calibration validation contract tests."""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from pgmuvi.instrument_channel_calibration_validation import (
    InstrumentChannelCalibrationValidationAcceptanceCriteria,
    InstrumentChannelCalibrationValidationDataset,
    InstrumentChannelCalibrationValidationDisposition,
    InstrumentChannelCalibrationValidationFoldResult,
    InstrumentChannelCalibrationValidationProtocol,
    InstrumentChannelCalibrationValidationResult,
    InstrumentChannelCalibrationValidationSourceResult,
    assess_instrument_channel_calibration_validation_result,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "examples"
    / "validation"
    / "kelt_r3_pairing_validation_protocol_v1.json"
)
SOURCE_PATH = ROOT / "examples" / "data" / "10131+3049.csv"


class TestInstrumentChannelCalibrationValidationProtocolContract(
    unittest.TestCase
):
    """Contract tests for prospective calibration validation."""

    @staticmethod
    def _anchor_dataset():
        return InstrumentChannelCalibrationValidationDataset(
            dataset_id="10131+3049-full-public-v1",
            astrophysical_source_id="10131+3049",
            dataset_reference="examples/data/10131+3049.csv",
            dataset_sha256=(
                "a54aa419c889e14ff238532349b752c27"
                "ada71c028b9eb1fffb85f382c519f62"
            ),
        )

    @staticmethod
    def _criteria():
        return InstrumentChannelCalibrationValidationAcceptanceCriteria(
            minimum_independent_astrophysical_sources=2,
            minimum_matched_pairs_per_source=100,
            temporal_fold_count=5,
            minimum_holdout_pairs_per_fold=20,
            minimum_holdout_normalization_amplitude_to_median_reference_error=5.0,
            maximum_median_holdout_normalized_rmse=0.10,
            maximum_worst_fold_holdout_normalized_rmse=0.20,
            maximum_absolute_holdout_median_bias_normalized=0.05,
        )

    @classmethod
    def _protocol(cls, **overrides):
        values = {
            "protocol_id": "kelt-osn-r3-pairing-validation-v1",
            "protocol_version": "1.0",
            "rule_id": "kelt-osn-r3-0-reference-r3-1-nearest-0p05d",
            "reference_instrument": "KELT",
            "reference_channel": "KELT/OSN_Johnson.Cousins_R3_0",
            "channel_instrument": "KELT",
            "channel": "KELT/OSN_Johnson.Cousins_R3_1",
            "physical_wavelength": 0.6561154962791801,
            "pairing_method": "nearest_within_tolerance",
            "time_unit": "day",
            "maximum_time_separation": 0.05,
            "anchor_dataset": cls._anchor_dataset(),
            "reference_channel_justification": (
                "R3_0 has more observations, a longer baseline, a finer "
                "median cadence, and a lower median reported uncertainty."
            ),
            "pairing_method_justification": (
                "The channels have no exact shared timestamps, so the "
                "prospective protocol uses deterministic one-to-one nearest "
                "pairing without reuse or interpolation."
            ),
            "time_tolerance_justification": (
                "The fixed 0.05 day window is frozen prospectively and is "
                "small relative to the hundreds-day LPV variability scale."
            ),
            "acceptance_criteria_justification": (
                "Every fold must retain at least 20 holdout pairs and a "
                "reference-flux dynamic range of at least five median "
                "reference error bars. Complete evidence must keep median "
                "normalized RMSE at or below 0.10, every fold at or below "
                "0.20, and absolute normalized median bias at or below 0.05. "
                "These gates are fixed before maintained execution and are "
                "not derived from exploratory outcomes."
            ),
            "calibration_family": "affine",
            "sigma_clip": 3.5,
            "maximum_fit_iterations": 8,
            "minimum_fit_pairs": 20,
            "measurement_error_policy": (
                "reference_and_channel_measurement_errors_required"
            ),
            "temporal_holdout_method": (
                "five_contiguous_equal_count_pair_midpoint_folds"
            ),
            "temporal_holdout_ordering_field": "pair_midpoint_time",
            "holdout_metric_normalization": (
                "reference_flux_q05_q95_amplitude_within_each_held_out_fold"
            ),
            "source_independence_unit": "astrophysical_source_id",
            "derived_datasets_count_as_independent": False,
            "acceptance_criteria": cls._criteria(),
            "applicability_boundaries": (
                "KELT OSN Johnson-Cousins R3_0 and R3_1 only.",
                "Linear positive flux and positive measurement errors only.",
                "Derived or downsampled datasets do not count as independent "
                "astrophysical sources.",
                "All exploratory runs preceding this protocol are excluded "
                "from validation evidence.",
            ),
        }
        values.update(overrides)
        return InstrumentChannelCalibrationValidationProtocol(**values)

    @staticmethod
    def _second_dataset(**overrides):
        values = {
            "dataset_id": "independent-source-full-v1",
            "astrophysical_source_id": "independent-source",
            "dataset_reference": "external:independent-source.csv",
            "dataset_sha256": "b" * 64,
        }
        values.update(overrides)
        return InstrumentChannelCalibrationValidationDataset(**values)

    @staticmethod
    def _fold_result(fold_index, **overrides):
        values = {
            "fold_index": fold_index,
            "n_training_pairs": 120,
            "n_holdout_pairs": 30,
            "successful": True,
            "holdout_reference_flux_q05_q95_amplitude": 0.20,
            "holdout_median_reference_flux_error": 0.02,
            "holdout_normalized_rmse": (
                0.06 + 0.01 * fold_index
            ),
            "holdout_median_bias_normalized": (
                (-1.0 if fold_index % 2 == 0 else 1.0)
                * (0.01 + 0.005 * fold_index)
            ),
            "maximum_absolute_time_separation": 0.049,
            "fitted_offset": -0.02 + 0.001 * fold_index,
            "fitted_scale": 0.65 + 0.01 * fold_index,
        }
        values.update(overrides)
        return InstrumentChannelCalibrationValidationFoldResult(**values)

    @classmethod
    def _source_result(
        cls,
        dataset,
        *,
        fold_overrides=None,
        **overrides,
    ):
        fold_overrides = fold_overrides or {}
        values = {
            "dataset": dataset,
            "n_matched_pairs": 150,
            "fold_results": tuple(
                cls._fold_result(
                    fold_index,
                    **fold_overrides.get(fold_index, {}),
                )
                for fold_index in range(5)
            ),
        }
        values.update(overrides)
        return InstrumentChannelCalibrationValidationSourceResult(**values)

    @classmethod
    def _result(cls, protocol, source_results, **overrides):
        values = {
            "result_id": "execution-v1",
            "result_version": "1.0",
            "protocol_id": protocol.protocol_id,
            "protocol_version": protocol.protocol_version,
            "protocol_sha256": protocol.canonical_sha256,
            "rule_id": protocol.rule_id,
            "execution_reference": "result:execution-v1",
            "package_version": "test",
            "package_commit": "1" * 40,
            "executed_at_utc": "2026-08-01T00:00:00Z",
            "source_results": tuple(source_results),
            "execution_completed": True,
        }
        values.update(overrides)
        return InstrumentChannelCalibrationValidationResult(**values)

    def test_dataset_contract_is_strict_immutable_and_records_lineage(self):
        dataset = self._anchor_dataset()
        payload = dataset.to_dict()
        self.assertFalse(payload["is_derived"])
        json.dumps(payload, allow_nan=False)
        self.assertEqual(
            InstrumentChannelCalibrationValidationDataset.from_dict(
                payload
            ).to_dict(),
            payload,
        )
        with self.assertRaises(FrozenInstanceError):
            dataset.dataset_id = "changed"
        with self.assertRaises(ValueError):
            InstrumentChannelCalibrationValidationDataset.from_dict(
                {**payload, "unexpected": True}
            )

    def test_fold_contract_is_strict_immutable_and_auditable(self):
        fold = self._fold_result(0)
        payload = fold.to_dict()
        self.assertTrue(payload["complete"])
        self.assertEqual(
            payload[
                "normalization_amplitude_to_median_reference_error"
            ],
            10.0,
        )
        self.assertEqual(
            InstrumentChannelCalibrationValidationFoldResult.from_dict(
                payload
            ).to_dict(),
            payload,
        )
        json.dumps(payload, allow_nan=False)
        with self.assertRaises(FrozenInstanceError):
            fold.fold_index = 3
        with self.assertRaises(ValueError):
            InstrumentChannelCalibrationValidationFoldResult.from_dict(
                {**payload, "unexpected": True}
            )

    def test_successful_fold_requires_every_auditable_metric(self):
        with self.assertRaisesRegex(ValueError, "every auditable metric"):
            self._fold_result(
                0,
                holdout_median_reference_flux_error=None,
            )
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            self._fold_result(
                0,
                successful=False,
                holdout_reference_flux_q05_q95_amplitude=None,
                holdout_median_reference_flux_error=None,
                holdout_normalized_rmse=None,
                holdout_median_bias_normalized=None,
                maximum_absolute_time_separation=None,
                fitted_offset=None,
                fitted_scale=None,
            )

    def test_protocol_is_strict_prospective_and_binds_exact_configuration(self):
        protocol = self._protocol()
        payload = protocol.to_dict()
        json.dumps(payload, allow_nan=False)
        restored = InstrumentChannelCalibrationValidationProtocol.from_dict(
            payload
        )
        self.assertEqual(restored.to_dict(), payload)
        self.assertEqual(restored.canonical_sha256, protocol.canonical_sha256)
        self.assertFalse(
            payload[
                "prior_exploratory_runs_eligible_as_validation_evidence"
            ]
        )
        self.assertFalse(payload["protocol_execution_performed"])
        self.assertFalse(payload["populated_catalogue_created"])
        self.assertEqual(protocol.maximum_time_separation, 0.05)
        self.assertEqual(protocol.acceptance_criteria.temporal_fold_count, 5)
        self.assertEqual(
            protocol.acceptance_criteria.minimum_holdout_pairs_per_fold,
            20,
        )
        self.assertEqual(
            (
                protocol.acceptance_criteria
                .minimum_holdout_normalization_amplitude_to_median_reference_error
            ),
            5.0,
        )

    def test_protocol_rejects_execution_catalogue_and_derived_source_claims(self):
        with self.assertRaisesRegex(ValueError, "cannot claim execution"):
            self._protocol(protocol_execution_performed=True)
        with self.assertRaisesRegex(ValueError, "cannot create"):
            self._protocol(populated_catalogue_created=True)
        derived = InstrumentChannelCalibrationValidationDataset(
            dataset_id="sampled",
            astrophysical_source_id="10131+3049",
            dataset_reference="derived:sampled.csv",
            dataset_sha256="c" * 64,
            derivation_parent_dataset_id="10131+3049-full-public-v1",
        )
        with self.assertRaisesRegex(ValueError, "anchor.*cannot be derived"):
            self._protocol(anchor_dataset=derived)

    def test_source_aggregates_are_derived_from_exact_fold_records(self):
        source_result = self._source_result(self._anchor_dataset())
        payload = source_result.to_dict()
        self.assertTrue(source_result.complete)
        self.assertEqual(source_result.n_temporal_folds, 5)
        self.assertEqual(source_result.n_successful_temporal_folds, 5)
        self.assertAlmostEqual(
            source_result.median_holdout_normalized_rmse,
            0.08,
        )
        self.assertAlmostEqual(
            source_result.worst_fold_holdout_normalized_rmse,
            0.10,
        )
        self.assertAlmostEqual(
            source_result.maximum_absolute_holdout_median_bias_normalized,
            0.03,
        )
        self.assertAlmostEqual(
            (
                source_result
                .minimum_holdout_normalization_amplitude_to_median_reference_error
            ),
            10.0,
        )
        self.assertEqual(
            InstrumentChannelCalibrationValidationSourceResult.from_dict(
                payload
            ).to_dict(),
            payload,
        )
        tampered = dict(payload)
        tampered["worst_fold_holdout_normalized_rmse"] = 0.01
        with self.assertRaisesRegex(ValueError, "does not match"):
            InstrumentChannelCalibrationValidationSourceResult.from_dict(
                tampered
            )

    def test_single_source_result_is_inconclusive(self):
        protocol = self._protocol()
        result = self._result(
            protocol,
            [self._source_result(protocol.anchor_dataset)],
        )
        report = assess_instrument_channel_calibration_validation_result(
            protocol,
            result,
        )
        self.assertEqual(
            report.disposition,
            InstrumentChannelCalibrationValidationDisposition.INCONCLUSIVE,
        )
        self.assertEqual(report.independent_astrophysical_source_count, 1)
        self.assertIn(
            "insufficient_independent_astrophysical_sources",
            report.reasons,
        )
        self.assertFalse(report.passed)

    def test_derived_copy_does_not_increase_independent_source_count(self):
        protocol = self._protocol()
        derived = InstrumentChannelCalibrationValidationDataset(
            dataset_id="10131+3049-sampled-v1",
            astrophysical_source_id="10131+3049",
            dataset_reference="derived:sampled.csv",
            dataset_sha256="d" * 64,
            derivation_parent_dataset_id=protocol.anchor_dataset.dataset_id,
        )
        result = self._result(
            protocol,
            [
                self._source_result(protocol.anchor_dataset),
                self._source_result(derived),
            ],
        )
        report = assess_instrument_channel_calibration_validation_result(
            protocol,
            result,
        )
        self.assertEqual(report.independent_astrophysical_source_count, 1)
        self.assertEqual(
            report.disposition,
            InstrumentChannelCalibrationValidationDisposition.INCONCLUSIVE,
        )

    def test_two_independent_sources_can_pass_every_preregistered_gate(self):
        protocol = self._protocol()
        result = self._result(
            protocol,
            [
                self._source_result(protocol.anchor_dataset),
                self._source_result(self._second_dataset()),
            ],
        )
        report = assess_instrument_channel_calibration_validation_result(
            protocol,
            result,
        )
        self.assertEqual(
            report.disposition,
            InstrumentChannelCalibrationValidationDisposition.PASSED,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.reasons, ())
        self.assertEqual(report.independent_astrophysical_source_count, 2)
        json.dumps(report.to_dict(), allow_nan=False)
        self.assertFalse(report.to_dict()["catalogue_population_performed"])

    def test_metric_failure_is_failed_and_not_upgraded(self):
        protocol = self._protocol()
        result = self._result(
            protocol,
            [
                self._source_result(protocol.anchor_dataset),
                self._source_result(
                    self._second_dataset(),
                    fold_overrides={
                        4: {"holdout_normalized_rmse": 0.21},
                    },
                ),
            ],
        )
        report = assess_instrument_channel_calibration_validation_result(
            protocol,
            result,
        )
        self.assertEqual(
            report.disposition,
            InstrumentChannelCalibrationValidationDisposition.FAILED,
        )
        self.assertIn(
            "source_acceptance_criteria_failed",
            report.reasons,
        )
        self.assertEqual(
            report.failed_astrophysical_source_ids,
            ("independent-source",),
        )

    def test_inadequate_fold_dynamic_range_is_inconclusive_not_failed(self):
        protocol = self._protocol()
        result = self._result(
            protocol,
            [
                self._source_result(protocol.anchor_dataset),
                self._source_result(
                    self._second_dataset(),
                    fold_overrides={
                        2: {
                            "holdout_reference_flux_q05_q95_amplitude": 0.08,
                            "holdout_median_reference_flux_error": 0.02,
                        },
                    },
                ),
            ],
        )
        report = assess_instrument_channel_calibration_validation_result(
            protocol,
            result,
        )
        self.assertEqual(
            report.disposition,
            InstrumentChannelCalibrationValidationDisposition.INCONCLUSIVE,
        )
        self.assertIn("source_results_inconclusive", report.reasons)
        self.assertEqual(
            report.inconclusive_astrophysical_source_ids,
            ("independent-source",),
        )

    def test_unsuccessful_fold_is_inconclusive_not_failed(self):
        protocol = self._protocol()
        failed_fold = {
            "successful": False,
            "holdout_reference_flux_q05_q95_amplitude": None,
            "holdout_median_reference_flux_error": None,
            "holdout_normalized_rmse": None,
            "holdout_median_bias_normalized": None,
            "maximum_absolute_time_separation": None,
            "fitted_offset": None,
            "fitted_scale": None,
            "failure_reasons": ("fit_failed",),
        }
        result = self._result(
            protocol,
            [
                self._source_result(protocol.anchor_dataset),
                self._source_result(
                    self._second_dataset(),
                    fold_overrides={2: failed_fold},
                ),
            ],
        )
        report = assess_instrument_channel_calibration_validation_result(
            protocol,
            result,
        )
        self.assertEqual(
            report.disposition,
            InstrumentChannelCalibrationValidationDisposition.INCONCLUSIVE,
        )
        self.assertIn("source_results_inconclusive", report.reasons)

    def test_protocol_digest_and_anchor_mismatches_are_inconclusive(self):
        protocol = self._protocol()
        altered_anchor = InstrumentChannelCalibrationValidationDataset(
            dataset_id="other-anchor",
            astrophysical_source_id="other-anchor-source",
            dataset_reference="external:other-anchor.csv",
            dataset_sha256="e" * 64,
        )
        result = self._result(
            protocol,
            [
                self._source_result(altered_anchor),
                self._source_result(self._second_dataset()),
            ],
            protocol_sha256="f" * 64,
        )
        report = assess_instrument_channel_calibration_validation_result(
            protocol,
            result,
        )
        self.assertEqual(
            report.disposition,
            InstrumentChannelCalibrationValidationDisposition.INCONCLUSIVE,
        )
        self.assertIn("result_protocol_digest_mismatch", report.reasons)
        self.assertIn(
            "anchor_dataset_missing_or_mismatched",
            report.reasons,
        )

    def test_result_contract_is_strict_and_cannot_embed_validation_claim(self):
        protocol = self._protocol()
        result = self._result(
            protocol,
            [self._source_result(protocol.anchor_dataset)],
        )
        payload = result.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertEqual(
            InstrumentChannelCalibrationValidationResult.from_dict(
                payload
            ).to_dict(),
            payload,
        )
        with self.assertRaisesRegex(ValueError, "cannot embed"):
            InstrumentChannelCalibrationValidationResult.from_dict(
                {**payload, "scientific_validation_claim_embedded": True}
            )

    def test_committed_protocol_and_evidence_are_exact_and_source_bound(self):
        payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        protocol = InstrumentChannelCalibrationValidationProtocol.from_dict(
            payload
        )
        self.assertEqual(protocol.to_dict(), payload)
        self.assertEqual(protocol.anchor_dataset, self._anchor_dataset())
        self.assertEqual(
            hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            protocol.anchor_dataset.dataset_sha256,
        )

        # The frozen protocol remains prospectively unexecuted in its own
        # payload.  Execution is recorded only in the separate result and
        # report artifacts committed by the implementation that ran it.
        self.assertFalse(payload["protocol_execution_performed"])
        self.assertFalse(payload["populated_catalogue_created"])

        result_path = (
            PROTOCOL_PATH.parent
            / "kelt_r3_pairing_validation_result_v1.json"
        )
        report_path = (
            PROTOCOL_PATH.parent
            / "kelt_r3_pairing_validation_report_v1.json"
        )
        self.assertTrue(result_path.is_file())
        self.assertTrue(report_path.is_file())

        result_payload = json.loads(
            result_path.read_text(encoding="utf-8")
        )
        report_payload = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        result = InstrumentChannelCalibrationValidationResult.from_dict(
            result_payload
        )

        self.assertEqual(result.to_dict(), result_payload)
        self.assertEqual(
            result.protocol_sha256,
            protocol.canonical_sha256,
        )
        self.assertTrue(result.execution_completed)
        self.assertEqual(len(result.source_results), 1)
        self.assertEqual(
            result.source_results[0].dataset,
            protocol.anchor_dataset,
        )

        recomputed_report = (
            assess_instrument_channel_calibration_validation_result(
                protocol,
                result,
            )
        )
        self.assertEqual(recomputed_report.to_dict(), report_payload)
        self.assertEqual(
            recomputed_report.disposition,
            InstrumentChannelCalibrationValidationDisposition.INCONCLUSIVE,
        )
        self.assertEqual(
            recomputed_report.independent_astrophysical_source_count,
            1,
        )
        self.assertEqual(
            recomputed_report.reasons,
            (
                "source_results_inconclusive",
                "insufficient_independent_astrophysical_sources",
            ),
        )
        self.assertFalse(
            report_payload["catalogue_population_performed"]
        )
        self.assertFalse(
            result_payload["catalogue_population_performed"]
        )
        self.assertFalse(
            result_payload["scientific_validation_claim_embedded"]
        )

    def test_docs_preserve_protocol_only_boundary(self):
        module_docs = (
            ROOT
            / "docs"
            / "source"
            / "pgmuvi.instrument_channel_calibration_validation.rst"
        ).read_text(encoding="utf-8")
        calibration_docs = (
            ROOT
            / "docs"
            / "source"
            / "pgmuvi.instrument_channel_calibration.rst"
        ).read_text(encoding="utf-8")
        future_work = (
            ROOT / "docs" / "source" / "future_work.rst"
        ).read_text(encoding="utf-8")
        combined = "\n".join((module_docs, calibration_docs, future_work))
        for token in (
            "prospectively frozen",
            "fold-level",
            "five median reference error bars",
            "derived datasets do not count as independent",
            "prior exploratory runs",
            "does not execute the protocol",
            "does not populate a pairing-rule catalogue",
            "TBD[instrument-channel-calibration]",
        ):
            with self.subTest(token=token):
                self.assertIn(token, combined)


if __name__ == "__main__":
    unittest.main()
