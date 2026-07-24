"""Pairing-rule catalogue discovery and activation contract tests."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_ACTIVATION_REQUEST_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_COMPATIBILITY_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_DISCOVERY_REQUEST_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_SCHEMA_VERSION,
    InstrumentChannelPairingMethod,
    InstrumentChannelPairingRule,
    InstrumentChannelPairingRuleCatalogue,
    InstrumentChannelPairingRuleCatalogueActivationRequest,
    InstrumentChannelPairingRuleCatalogueDiscoveryRequest,
    InstrumentChannelPairingRuleCatalogueSource,
    InstrumentChannelPairingRuleCatalogueValidationEvidence,
    InstrumentChannelPairingRuleScientificValidationDisposition,
    InstrumentChannelPairingRuleValidationEvidence,
    InstrumentChannelPairingRuleValidationStatus,
    InstrumentChannelPairingToleranceProvenance,
    activate_instrument_channel_pairing_rule_catalogue,
    assess_instrument_channel_pairing_rule_catalogue_compatibility,
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


class TestPairingRuleCatalogueDiscoveryActivationContract(
    unittest.TestCase
):
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
    def _discovery_request(**overrides):
        values = {
            "source": (
                InstrumentChannelPairingRuleCatalogueSource
                .EXPLICIT_PATH
            ),
            "source_reference": "/science/catalogues/pairing-rules.json",
            "expected_catalogue_id": "pgmuvi-pairing-rules",
            "expected_catalogue_version": "2026.1",
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

    def test_discovery_request_is_explicit_immutable_and_json_safe(self):
        request = self._discovery_request()

        self.assertEqual(
            request.schema_version,
            INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_DISCOVERY_REQUEST_SCHEMA_VERSION,
        )
        self.assertEqual(
            request.source,
            InstrumentChannelPairingRuleCatalogueSource.EXPLICIT_PATH,
        )

        payload = request.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertTrue(payload["explicit_source_required"])
        self.assertFalse(payload["allow_ambient_search"])
        self.assertFalse(payload["environment_lookup_permitted"])
        self.assertFalse(payload["working_directory_scan_permitted"])
        self.assertFalse(
            payload["packaged_catalogue_fallback_permitted"]
        )
        self.assertTrue(payload["discovery_implemented"])

        with self.assertRaises(FrozenInstanceError):
            request.source_reference = "replacement.json"

    def test_discovery_request_rejects_ambient_or_ambiguous_sources(self):
        with self.assertRaisesRegex(ValueError, "Ambient"):
            self._discovery_request(allow_ambient_search=True)

        with self.assertRaisesRegex(ValueError, "Unsupported.*source"):
            self._discovery_request(source="environment")

        with self.assertRaisesRegex(ValueError, "non-empty"):
            self._discovery_request(source_reference="  ")

        with self.assertRaises(TypeError):
            self._discovery_request(allow_ambient_search=1)

    def test_activation_request_is_explicit_validated_only_and_json_safe(self):
        request = self._activation_request()

        self.assertEqual(
            request.schema_version,
            INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_ACTIVATION_REQUEST_SCHEMA_VERSION,
        )

        payload = request.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertTrue(payload["explicit_activation"])
        self.assertTrue(payload["require_scientifically_validated"])
        self.assertFalse(payload["allow_unvalidated_rules"])
        self.assertFalse(payload["allow_catalogue_fallback"])
        self.assertFalse(payload["workflow_integration_requested"])
        self.assertFalse(payload["silent_activation_permitted"])
        self.assertTrue(payload["activation_implemented"])

        with self.assertRaises(FrozenInstanceError):
            request.catalogue_version = "replacement"

    def test_activation_request_rejects_unsafe_policy_relaxations(self):
        cases = (
            (
                {"explicit_activation": False},
                "explicitly requested",
            ),
            (
                {"require_scientifically_validated": False},
                "requires scientific validation",
            ),
            (
                {"allow_unvalidated_rules": True},
                "cannot be activated",
            ),
            (
                {"allow_catalogue_fallback": True},
                "cannot fall back",
            ),
            (
                {"workflow_integration_requested": True},
                "workflow integration is not available",
            ),
        )

        for updates, message in cases:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(ValueError, message):
                    self._activation_request(**updates)

    def test_exact_compatibility_is_json_safe_and_side_effect_free(self):
        request = self._activation_request()
        catalogue = self._catalogue()

        report = (
            assess_instrument_channel_pairing_rule_catalogue_compatibility(
                request,
                catalogue,
            )
        )
        payload = report.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertEqual(
            report.schema_version,
            INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_COMPATIBILITY_SCHEMA_VERSION,
        )
        self.assertTrue(report.compatible)
        self.assertTrue(report.activation_permitted)
        self.assertEqual(report.reasons, ())
        self.assertTrue(report.catalogue_id_matches)
        self.assertTrue(report.catalogue_version_matches)
        self.assertTrue(report.catalogue_schema_version_matches)
        self.assertTrue(report.catalogue_scientifically_validated)
        self.assertTrue(report.all_rules_scientifically_validated)
        self.assertFalse(
            payload["compatibility_check_has_side_effects"]
        )
        self.assertFalse(payload["catalogue_activated"])
        self.assertFalse(payload["workflow_integration_implemented"])

    def test_compatibility_rejects_identity_version_and_schema_mismatch(self):
        cases = (
            (
                {"catalogue_id": "different-catalogue"},
                "catalogue_id_mismatch",
            ),
            (
                {"catalogue_version": "2027.1"},
                "catalogue_version_mismatch",
            ),
            (
                {"catalogue_schema_version": "unsupported-schema"},
                "catalogue_schema_version_mismatch",
            ),
        )
        catalogue = self._catalogue()

        for updates, expected_reason in cases:
            with self.subTest(updates=updates):
                report = (
                    assess_instrument_channel_pairing_rule_catalogue_compatibility(
                        self._activation_request(**updates),
                        catalogue,
                    )
                )
                self.assertFalse(report.compatible)
                self.assertFalse(report.activation_permitted)
                self.assertIn(expected_reason, report.reasons)

    def test_compatibility_rejects_unvalidated_catalogue_or_members(self):
        unvalidated_catalogue = self._catalogue(
            validation_status=(
                InstrumentChannelPairingRuleValidationStatus
                .DEFINED_NOT_VALIDATED
            ),
            evidence_reference=None,
        )
        catalogue_report = (
            assess_instrument_channel_pairing_rule_catalogue_compatibility(
                self._activation_request(),
                unvalidated_catalogue,
            )
        )
        self.assertFalse(catalogue_report.compatible)
        self.assertIn(
            "catalogue_not_scientifically_validated",
            catalogue_report.reasons,
        )

        unvalidated_rule = self._validated_rule(
            validation_status=(
                InstrumentChannelPairingRuleValidationStatus
                .DEFINED_NOT_VALIDATED
            ),
            evidence_reference=None,
        )
        unvalidated_members = self._catalogue(
            rules=(unvalidated_rule,),
            validation_status=(
                InstrumentChannelPairingRuleValidationStatus
                .DEFINED_NOT_VALIDATED
            ),
            evidence_reference=None,
        )
        member_report = (
            assess_instrument_channel_pairing_rule_catalogue_compatibility(
                self._activation_request(),
                unvalidated_members,
            )
        )
        self.assertFalse(member_report.compatible)
        self.assertIn(
            "member_rules_not_all_scientifically_validated",
            member_report.reasons,
        )

    def test_compatibility_report_rejects_forged_reason_codes(self):
        compatible_report = (
            assess_instrument_channel_pairing_rule_catalogue_compatibility(
                self._activation_request(),
                self._catalogue(),
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "reasons must exactly describe",
        ):
            replace(
                compatible_report,
                reasons=("arbitrary_reason",),
            )

        mismatch_report = (
            assess_instrument_channel_pairing_rule_catalogue_compatibility(
                self._activation_request(
                    catalogue_version="different-version"
                ),
                self._catalogue(),
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "reasons must exactly describe",
        ):
            replace(
                mismatch_report,
                reasons=("catalogue_id_mismatch",),
            )

    def test_discovery_and_activation_callables_require_typed_inputs(self):
        with self.assertRaises(TypeError):
            discover_instrument_channel_pairing_rule_catalogue(object())

        with self.assertRaises(TypeError):
            activate_instrument_channel_pairing_rule_catalogue(
                object(),
                object(),
            )

    def test_public_documentation_records_implemented_boundary(self):
        repo = Path(__file__).resolve().parents[1]
        api_text = (
            repo
            / "docs/source/pgmuvi.instrument_channel_calibration.rst"
        ).read_text(encoding="utf-8")
        future_text = (
            repo / "docs/source/future_work.rst"
        ).read_text(encoding="utf-8")

        for token in (
            "InstrumentChannelPairingRuleCatalogueLoadedSnapshot",
            "InstrumentChannelPairingRuleCatalogueActivationSnapshot",
            "strict UTF-8 JSON",
            "duplicate object keys",
            "source-access, parse, semantic-validation, and compatibility",
            "process-global",
            "never falls back",
        ):
            self.assertIn(token, api_text)

        for token in (
            "explicit-source loading and local activation",
            "immutable loaded-catalogue snapshot",
            "immutable local activation snapshot",
            "process-global",
            "populated scientifically",
            "TBD[instrument-channel-calibration]",
        ):
            self.assertIn(token, future_text)


if __name__ == "__main__":
    unittest.main()
