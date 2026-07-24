"""Explicit catalogue loading and local-activation implementation tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_SCHEMA_VERSION,
    InstrumentChannelPairingMethod,
    InstrumentChannelPairingRule,
    InstrumentChannelPairingRuleCatalogue,
    InstrumentChannelPairingRuleCatalogueActivationRequest,
    InstrumentChannelPairingRuleCatalogueActivationSnapshot,
    InstrumentChannelPairingRuleCatalogueCompatibilityError,
    InstrumentChannelPairingRuleCatalogueDiscoveryRequest,
    InstrumentChannelPairingRuleCatalogueLoadedSnapshot,
    InstrumentChannelPairingRuleCatalogueParseError,
    InstrumentChannelPairingRuleCatalogueSource,
    InstrumentChannelPairingRuleCatalogueSourceAccessError,
    InstrumentChannelPairingRuleCatalogueValidationError,
    InstrumentChannelPairingRuleCatalogueValidationEvidence,
    InstrumentChannelPairingRuleScientificValidationDisposition,
    InstrumentChannelPairingRuleValidationEvidence,
    InstrumentChannelPairingRuleValidationStatus,
    InstrumentChannelPairingToleranceProvenance,
    activate_instrument_channel_pairing_rule_catalogue,
    discover_instrument_channel_pairing_rule_catalogue,
)


def _rule_validation_evidence(values):
    return InstrumentChannelPairingRuleValidationEvidence(
        validation_id=f"validation:{values['rule_id']}",
        validation_version="1",
        rule_id=values["rule_id"],
        reference_instrument=values["reference_instrument"],
        reference_channel=values["reference_channel"],
        channel_instrument=values["channel_instrument"],
        channel=values["channel"],
        physical_wavelength=values["physical_wavelength"],
        pairing_method=values["pairing_method"],
        time_unit=values["time_unit"],
        maximum_time_separation=values["maximum_time_separation"],
        tolerance_provenance=values["tolerance_provenance"],
        validation_dataset_reference="dataset:synthetic-contract-fixture",
        validation_dataset_sha256="a" * 64,
        validation_protocol_reference="protocol:pairing-validation-v1",
        validation_result_reference=values["evidence_reference"],
        reference_channel_justification=(
            "Reference-channel choice is explicitly justified."
        ),
        pairing_method_justification=(
            "Pairing method is explicitly justified."
        ),
        time_tolerance_justification=(
            "Maximum time separation is explicitly justified."
        ),
        applicability_boundaries=(
            "Applies only to the exact named observational-channel pair.",
        ),
        acceptance_criteria=(
            "All declared deterministic acceptance criteria pass.",
        ),
        n_validation_sources=3,
        n_matched_pairs=30,
        disposition=(
            InstrumentChannelPairingRuleScientificValidationDisposition
            .PASSED
        ),
    )



def _catalogue_validation_evidence(values):
    rules = tuple(values["rules"])
    return InstrumentChannelPairingRuleCatalogueValidationEvidence(
        validation_id=(
            f"catalogue-validation:{values['catalogue_id']}:"
            f"{values['catalogue_version']}"
        ),
        validation_version="1",
        catalogue_id=values["catalogue_id"],
        catalogue_version=values["catalogue_version"],
        catalogue_schema_version=(
            INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_SCHEMA_VERSION
        ),
        member_rule_ids=tuple(rule.rule_id for rule in rules),
        member_validation_ids=tuple(
            (
                rule.validation_evidence.validation_id
                if rule.validation_evidence is not None
                else f"missing:{rule.rule_id}"
            )
            for rule in rules
        ),
        validation_protocol_reference=(
            "protocol:catalogue-validation-v1"
        ),
        validation_result_reference=values["evidence_reference"],
        applicability_boundaries=(
            "Applies only to the exact ordered catalogue membership.",
        ),
        acceptance_criteria=(
            "Every member rule has complete passed typed evidence.",
        ),
        disposition=(
            InstrumentChannelPairingRuleScientificValidationDisposition
            .PASSED
        ),
    )


class TestPairingRuleCatalogueLoadingActivation(unittest.TestCase):
    @staticmethod
    def _validated_rule(**overrides):
        values = {
            "rule_id": "survey-a-to-survey-b-v1",
            "reference_instrument": "Survey A",
            "reference_channel": "SurveyA/V",
            "channel_instrument": "Survey B",
            "channel": "SurveyB/V",
            "physical_wavelength": 0.55,
            "pairing_method": (
                InstrumentChannelPairingMethod
                .NEAREST_WITHIN_TOLERANCE
            ),
            "time_unit": "day",
            "maximum_time_separation": 0.25,
            "tolerance_provenance": (
                InstrumentChannelPairingToleranceProvenance
                .EMPIRICAL_VALIDATION
            ),
            "validation_status": (
                InstrumentChannelPairingRuleValidationStatus
                .SCIENTIFICALLY_VALIDATED
            ),
            "evidence_reference": "validation:representative-lpv-v1",
        }
        values.update(overrides)
        if "validation_evidence" not in overrides:
            if (
                str(values["validation_status"])
                == "scientifically_validated"
                and values["evidence_reference"] is not None
            ):
                values["validation_evidence"] = (
                    _rule_validation_evidence(values)
                )
            else:
                values["validation_evidence"] = None
        return InstrumentChannelPairingRule(**values)

    @classmethod
    def _catalogue(cls, **overrides):
        values = {
            "catalogue_id": "pgmuvi-pairing-rules",
            "catalogue_version": "2026.1",
            "provenance_reference": "catalogue:design-record-v1",
            "rules": (cls._validated_rule(),),
            "validation_status": (
                InstrumentChannelPairingRuleValidationStatus
                .SCIENTIFICALLY_VALIDATED
            ),
            "evidence_reference": "catalogue:validation-report-v1",
        }
        values.update(overrides)
        if "validation_evidence" not in overrides:
            rules = tuple(values["rules"])
            rule_ids = tuple(
                getattr(rule, "rule_id", None)
                for rule in rules
            )
            identity_keys = tuple(
                getattr(rule, "identity_key", None)
                for rule in rules
            )
            if (
                str(values["validation_status"])
                == "scientifically_validated"
                and values["evidence_reference"] is not None
                and bool(rules)
                and len(set(rule_ids)) == len(rule_ids)
                and len(set(identity_keys)) == len(identity_keys)
                and all(
                    getattr(rule, "validation_evidence", None)
                    is not None
                    for rule in rules
                )
            ):
                values["validation_evidence"] = (
                    _catalogue_validation_evidence(values)
                )
            else:
                values["validation_evidence"] = None
        return InstrumentChannelPairingRuleCatalogue(**values)

    @staticmethod
    def _discovery_request(path: Path, **overrides):
        values = {
            "source": (
                InstrumentChannelPairingRuleCatalogueSource
                .EXPLICIT_PATH
            ),
            "source_reference": str(path),
            "expected_catalogue_id": "pgmuvi-pairing-rules",
            "expected_catalogue_version": "2026.1",
            "expected_catalogue_schema_version": (
                INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_SCHEMA_VERSION
            ),
        }
        values.update(overrides)
        return InstrumentChannelPairingRuleCatalogueDiscoveryRequest(
            **values
        )

    @staticmethod
    def _activation_request(**overrides):
        values = {
            "catalogue_id": "pgmuvi-pairing-rules",
            "catalogue_version": "2026.1",
            "catalogue_schema_version": (
                INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_SCHEMA_VERSION
            ),
        }
        values.update(overrides)
        return InstrumentChannelPairingRuleCatalogueActivationRequest(
            **values
        )

    @staticmethod
    def _write_catalogue(
        path: Path,
        catalogue: InstrumentChannelPairingRuleCatalogue,
    ) -> bytes:
        payload = json.dumps(
            catalogue.to_dict(),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        path.write_bytes(payload)
        return payload

    def test_explicit_path_load_returns_immutable_provenance_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "catalogue.json"
            source_bytes = self._write_catalogue(
                source_path,
                self._catalogue(),
            )
            request = self._discovery_request(source_path)

            snapshot = (
                discover_instrument_channel_pairing_rule_catalogue(
                    request
                )
            )

            self.assertIsInstance(
                snapshot,
                InstrumentChannelPairingRuleCatalogueLoadedSnapshot,
            )
            self.assertEqual(snapshot.discovery_request, request)
            self.assertEqual(snapshot.catalogue, self._catalogue())
            self.assertEqual(snapshot.source_reference, str(source_path))
            self.assertEqual(
                snapshot.resolved_source_reference,
                str(source_path.resolve()),
            )
            self.assertEqual(
                snapshot.source_size_bytes,
                len(source_bytes),
            )
            self.assertEqual(len(snapshot.source_sha256), 64)
            self.assertTrue(snapshot.compatibility_report.compatible)
            self.assertTrue(
                snapshot.compatibility_report.activation_permitted
            )

            payload = snapshot.to_dict()
            json.dumps(payload, allow_nan=False)
            self.assertTrue(payload["catalogue_loaded"])
            self.assertFalse(payload["catalogue_activated"])
            self.assertFalse(payload["process_global_state_modified"])
            self.assertFalse(payload["workflow_integration_active"])

            with self.assertRaises(FrozenInstanceError):
                snapshot.source_sha256 = "replacement"

    def test_local_activation_returns_distinct_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "catalogue.json"
            self._write_catalogue(source_path, self._catalogue())
            loaded = discover_instrument_channel_pairing_rule_catalogue(
                self._discovery_request(source_path)
            )
            request = self._activation_request()

            activation = (
                activate_instrument_channel_pairing_rule_catalogue(
                    request,
                    loaded,
                )
            )

            self.assertIsInstance(
                activation,
                InstrumentChannelPairingRuleCatalogueActivationSnapshot,
            )
            self.assertIsNot(activation, loaded)
            self.assertEqual(activation.loaded_snapshot, loaded)
            self.assertEqual(activation.catalogue, loaded.catalogue)
            self.assertEqual(
                activation.source_reference,
                str(source_path),
            )
            self.assertTrue(activation.local_only)
            self.assertFalse(activation.process_global_state_modified)
            self.assertFalse(activation.workflow_integration_active)

            payload = activation.to_dict()
            json.dumps(payload, allow_nan=False)
            self.assertTrue(payload["catalogue_activated"])
            self.assertTrue(payload["local_only"])
            self.assertFalse(payload["process_global_state_modified"])
            self.assertFalse(payload["workflow_integration_active"])

            second_activation = (
                activate_instrument_channel_pairing_rule_catalogue(
                    request,
                    loaded,
                )
            )
            self.assertIsNot(second_activation, activation)
            self.assertEqual(second_activation, activation)

            with self.assertRaises(FrozenInstanceError):
                activation.local_only = False

    def test_missing_or_non_file_source_has_source_access_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing = root / "missing.json"

            with self.assertRaises(
                InstrumentChannelPairingRuleCatalogueSourceAccessError
            ):
                discover_instrument_channel_pairing_rule_catalogue(
                    self._discovery_request(missing)
                )

            with self.assertRaises(
                InstrumentChannelPairingRuleCatalogueSourceAccessError
            ):
                discover_instrument_channel_pairing_rule_catalogue(
                    self._discovery_request(root)
                )

    def test_invalid_utf8_and_json_have_parse_failures(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            invalid_utf8 = root / "invalid-utf8.json"
            invalid_utf8.write_bytes(b"\xff\xfe")
            with self.assertRaises(
                InstrumentChannelPairingRuleCatalogueParseError
            ):
                discover_instrument_channel_pairing_rule_catalogue(
                    self._discovery_request(invalid_utf8)
                )

            malformed = root / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            with self.assertRaises(
                InstrumentChannelPairingRuleCatalogueParseError
            ):
                discover_instrument_channel_pairing_rule_catalogue(
                    self._discovery_request(malformed)
                )

    def test_duplicate_keys_and_nonfinite_constants_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"catalogue_id": "first", "catalogue_id": "second"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                InstrumentChannelPairingRuleCatalogueParseError,
                "Duplicate JSON object key",
            ):
                discover_instrument_channel_pairing_rule_catalogue(
                    self._discovery_request(duplicate)
                )

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text(
                '{"value": NaN}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                InstrumentChannelPairingRuleCatalogueParseError,
                "Non-finite JSON numeric constant",
            ):
                discover_instrument_channel_pairing_rule_catalogue(
                    self._discovery_request(nonfinite)
                )

    def test_semantic_contract_failures_are_distinct_from_parse_failures(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            top_level_list = root / "list.json"
            top_level_list.write_text("[]", encoding="utf-8")
            with self.assertRaises(
                InstrumentChannelPairingRuleCatalogueValidationError
            ):
                discover_instrument_channel_pairing_rule_catalogue(
                    self._discovery_request(top_level_list)
                )

            incomplete = root / "incomplete.json"
            incomplete.write_text(
                '{"catalogue_id": "pgmuvi-pairing-rules"}',
                encoding="utf-8",
            )
            with self.assertRaises(
                InstrumentChannelPairingRuleCatalogueValidationError
            ):
                discover_instrument_channel_pairing_rule_catalogue(
                    self._discovery_request(incomplete)
                )

    def test_discovery_compatibility_failure_preserves_reason_codes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "catalogue.json"
            self._write_catalogue(source_path, self._catalogue())

            with self.assertRaises(
                InstrumentChannelPairingRuleCatalogueCompatibilityError
            ) as context:
                discover_instrument_channel_pairing_rule_catalogue(
                    self._discovery_request(
                        source_path,
                        expected_catalogue_version="different-version",
                    )
                )

            self.assertEqual(
                context.exception.reasons,
                ("catalogue_version_mismatch",),
            )

    def test_unvalidated_catalogue_fails_compatibility_not_semantics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "catalogue.json"
            catalogue = self._catalogue(
                validation_status=(
                    InstrumentChannelPairingRuleValidationStatus
                    .DEFINED_NOT_VALIDATED
                ),
                evidence_reference=None,
            )
            self._write_catalogue(source_path, catalogue)

            with self.assertRaises(
                InstrumentChannelPairingRuleCatalogueCompatibilityError
            ) as context:
                discover_instrument_channel_pairing_rule_catalogue(
                    self._discovery_request(source_path)
                )

            self.assertEqual(
                context.exception.reasons,
                ("catalogue_not_scientifically_validated",),
            )

    def test_activation_rechecks_exact_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "catalogue.json"
            self._write_catalogue(source_path, self._catalogue())
            loaded = discover_instrument_channel_pairing_rule_catalogue(
                self._discovery_request(source_path)
            )

            with self.assertRaises(
                InstrumentChannelPairingRuleCatalogueCompatibilityError
            ) as context:
                activate_instrument_channel_pairing_rule_catalogue(
                    self._activation_request(
                        catalogue_version="different-version"
                    ),
                    loaded,
                )

            self.assertEqual(
                context.exception.reasons,
                ("catalogue_version_mismatch",),
            )

    def test_activation_rejects_raw_catalogue_to_preserve_boundary(self):
        with self.assertRaisesRegex(TypeError, "loaded_snapshot"):
            activate_instrument_channel_pairing_rule_catalogue(
                self._activation_request(),
                self._catalogue(),
            )


if __name__ == "__main__":
    unittest.main()
