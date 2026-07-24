"""Explicit catalogue loading and local-activation implementation tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

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
    InstrumentChannelPairingRuleValidationStatus,
    InstrumentChannelPairingToleranceProvenance,
    activate_instrument_channel_pairing_rule_catalogue,
    discover_instrument_channel_pairing_rule_catalogue,
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
