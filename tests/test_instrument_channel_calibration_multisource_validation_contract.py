from __future__ import annotations

import json
import unittest
from pathlib import Path

from pgmuvi.instrument_channel_calibration_multisource_validation import (
    InstrumentChannelCalibrationCrossSourceValidationMethod,
    InstrumentChannelCalibrationMultiSourceSelectionMethod,
    InstrumentChannelCalibrationMultiSourceValidationDisposition,
    InstrumentChannelCalibrationMultiSourceValidationProtocol,
    InstrumentChannelCalibrationMultiSourceValidationSummary,
    InstrumentChannelCalibrationSourceBalanceMethod,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "examples"
    / "validation"
    / "kelt_r3_maintainer_multisource_validation_protocol_v1.json"
)
VALIDATION_DOC = (
    ROOT
    / "docs"
    / "source"
    / "pgmuvi.instrument_channel_calibration_validation.rst"
)
EXECUTION_DOC = (
    ROOT
    / "docs"
    / "source"
    / "pgmuvi.instrument_channel_calibration_validation_execution.rst"
)
FUTURE_WORK = ROOT / "docs" / "source" / "future_work.rst"


class TestInstrumentChannelCalibrationMultiSourceValidationContract(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.protocol = (
            InstrumentChannelCalibrationMultiSourceValidationProtocol.from_dict(
                cls.payload
            )
        )

    def test_committed_protocol_round_trips_strictly(self) -> None:
        self.assertEqual(self.protocol.to_dict(), self.payload)
        self.assertEqual(
            self.protocol.canonical_sha256,
            "85228615f8d68684ab1c5a473990b34bf2294cf2f43808575b4e989b3675100a",
        )

    def test_protocol_binds_exact_candidate_pair(self) -> None:
        self.assertEqual(
            self.protocol.candidate_protocol_id,
            "kelt-osn-r3-pairing-validation-v1",
        )
        self.assertEqual(
            self.protocol.candidate_protocol_sha256,
            "198d1ed53990bf1f7b146b89b299f972fedf99459178457486b76ec9d174d708",
        )
        self.assertEqual(
            self.protocol.reference_channel,
            "KELT/OSN_Johnson.Cousins_R3_0",
        )
        self.assertEqual(
            self.protocol.channel,
            "KELT/OSN_Johnson.Cousins_R3_1",
        )
        self.assertAlmostEqual(
            self.protocol.physical_wavelength,
            0.6561154962791801,
        )

    def test_selection_is_reproducible_and_outcome_independent(self) -> None:
        self.assertEqual(self.protocol.required_source_count, 5)
        self.assertIs(
            self.protocol.selection_method,
            InstrumentChannelCalibrationMultiSourceSelectionMethod.
            SEEDED_PERMUTATION_FIRST_N,
        )
        self.assertEqual(
            self.protocol.selection_source_ordering_field,
            "astrophysical_source_id",
        )
        self.assertTrue(self.protocol.selection_without_replacement)
        self.assertTrue(
            self.protocol.eligibility_evaluated_before_selection
        )
        self.assertTrue(
            self.protocol.calibration_outcomes_excluded_from_eligibility
        )

    def test_eligibility_requires_exact_usable_candidate_pair(self) -> None:
        self.assertTrue(
            self.protocol.require_exact_observational_channel_identity
        )
        self.assertTrue(
            self.protocol.require_finite_strictly_positive_flux_and_error
        )
        self.assertTrue(
            self.protocol.require_independent_primary_astrophysical_sources
        )
        self.assertTrue(self.protocol.exclude_derived_sources)
        self.assertEqual(
            self.protocol.minimum_matched_pairs_per_source,
            100,
        )
        self.assertEqual(self.protocol.temporal_fold_count, 5)
        self.assertEqual(
            self.protocol.minimum_holdout_pairs_per_fold,
            20,
        )
        self.assertEqual(
            self.protocol.
            minimum_holdout_amplitude_to_median_reference_error,
            5.0,
        )

    def test_cross_source_test_is_balanced_leave_one_out(self) -> None:
        self.assertIs(
            self.protocol.cross_source_validation_method,
            InstrumentChannelCalibrationCrossSourceValidationMethod.
            LEAVE_ONE_SOURCE_OUT,
        )
        self.assertIs(
            self.protocol.source_balance_method,
            InstrumentChannelCalibrationSourceBalanceMethod.
            EQUAL_PAIR_COUNT,
        )
        self.assertTrue(
            self.protocol.require_every_selected_source_held_out_once
        )
        self.assertTrue(
            self.protocol.require_every_heldout_temporal_fold_informative
        )
        self.assertTrue(
            self.protocol.coefficient_stability_reported_not_gated
        )

    def test_aggregate_acceptance_gates_are_frozen(self) -> None:
        self.assertEqual(
            self.protocol.maximum_median_source_holdout_normalized_rmse,
            0.1,
        )
        self.assertEqual(
            self.protocol.maximum_worst_source_holdout_normalized_rmse,
            0.2,
        )
        self.assertEqual(
            self.protocol.
            maximum_absolute_median_source_holdout_bias_normalized,
            0.05,
        )

    def test_private_data_boundary_is_explicit(self) -> None:
        self.assertFalse(self.protocol.private_parquet_runner_distributed)
        self.assertFalse(
            self.protocol.public_package_parquet_support_required
        )
        self.assertFalse(self.protocol.catalogue_population_performed)
        self.assertIn(
            "maintainer-owned Parquet input is not distributed",
            self.protocol.private_input_policy,
        )

    def test_redacted_summary_round_trips_without_raw_ids(self) -> None:
        selected = tuple(
            f"{index:064x}"
            for index in range(1, 6)
        )
        summary = InstrumentChannelCalibrationMultiSourceValidationSummary(
            summary_id="private-kelt-r3-run-v1",
            summary_version="1.0",
            protocol_id=self.protocol.protocol_id,
            protocol_version=self.protocol.protocol_version,
            protocol_sha256=self.protocol.canonical_sha256,
            candidate_protocol_id=self.protocol.candidate_protocol_id,
            candidate_protocol_version=(
                self.protocol.candidate_protocol_version
            ),
            candidate_protocol_sha256=(
                self.protocol.candidate_protocol_sha256
            ),
            rule_id=self.protocol.rule_id,
            private_input_sha256="a" * 64,
            selection_seed=20260726,
            eligible_source_count=12,
            selected_source_id_sha256s=selected,
            successful_source_holdout_count=5,
            median_source_holdout_normalized_rmse=0.08,
            worst_source_holdout_normalized_rmse=0.15,
            median_source_holdout_bias_normalized=0.02,
            fitted_scale_median=1.01,
            fitted_scale_mad=0.01,
            fitted_offset_median=0.0,
            fitted_offset_mad=0.001,
            package_version="test",
            package_commit="b" * 40,
            executed_at_utc="2026-07-26T00:00:00Z",
            disposition=(
                InstrumentChannelCalibrationMultiSourceValidationDisposition.
                PASSED
            ),
            reasons=(),
            all_acceptance_criteria_passed=True,
        )
        payload = summary.to_dict()
        restored = (
            InstrumentChannelCalibrationMultiSourceValidationSummary.from_dict(
                payload
            )
        )
        self.assertEqual(restored.to_dict(), payload)
        self.assertNotIn("astrophysical_source_id", json.dumps(payload))
        self.assertFalse(payload["private_input_distributed"])
        self.assertFalse(payload["private_detailed_report_distributed"])
        self.assertFalse(payload["catalogue_population_performed"])

    def test_summary_rejects_privacy_or_catalogue_claims(self) -> None:
        kwargs = {
            "summary_id": "summary",
            "summary_version": "1",
            "protocol_id": self.protocol.protocol_id,
            "protocol_version": self.protocol.protocol_version,
            "protocol_sha256": self.protocol.canonical_sha256,
            "candidate_protocol_id": self.protocol.candidate_protocol_id,
            "candidate_protocol_version": (
                self.protocol.candidate_protocol_version
            ),
            "candidate_protocol_sha256": (
                self.protocol.candidate_protocol_sha256
            ),
            "rule_id": self.protocol.rule_id,
            "private_input_sha256": "a" * 64,
            "selection_seed": 1,
            "eligible_source_count": 0,
            "selected_source_id_sha256s": (),
            "successful_source_holdout_count": 0,
            "median_source_holdout_normalized_rmse": None,
            "worst_source_holdout_normalized_rmse": None,
            "median_source_holdout_bias_normalized": None,
            "fitted_scale_median": None,
            "fitted_scale_mad": None,
            "fitted_offset_median": None,
            "fitted_offset_mad": None,
            "package_version": "test",
            "package_commit": "b" * 40,
            "executed_at_utc": "2026-07-26T00:00:00Z",
            "disposition": (
                InstrumentChannelCalibrationMultiSourceValidationDisposition.
                INCONCLUSIVE
            ),
            "reasons": ("insufficient_eligible_astrophysical_sources",),
            "all_acceptance_criteria_passed": False,
        }
        for field in (
            "raw_source_identifiers_included",
            "private_input_distributed",
            "private_detailed_report_distributed",
            "catalogue_population_performed",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    InstrumentChannelCalibrationMultiSourceValidationSummary(
                        **kwargs,
                        **{field: True},
                    )

    def test_docs_describe_public_smoke_and_private_decision(self) -> None:
        validation = VALIDATION_DOC.read_text(encoding="utf-8")
        execution = EXECUTION_DOC.read_text(encoding="utf-8")
        future = FUTURE_WORK.read_text(encoding="utf-8")
        combined = "\n".join((validation, execution, future))

        for token in (
            "public reproducibility smoke test",
            "maintainer-private multi-source",
            "leave-one-source-out",
            "five",
            "private Parquet",
            "not distributed",
            "redacted",
        ):
            with self.subTest(token=token):
                self.assertIn(token, combined)

        self.assertIn(
            "protocol-approved candidate observational-channel pair",
            combined,
        )
        self.assertNotIn(
            "Obtaining scientifically adequate anchor evidence",
            future,
        )


if __name__ == "__main__":
    unittest.main()
