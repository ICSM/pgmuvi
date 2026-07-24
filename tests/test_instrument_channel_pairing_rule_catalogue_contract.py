"""Instrument-specific pairing-rule catalogue contract tests."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_SCHEMA_VERSION,
    InstrumentChannelPairingMethod,
    InstrumentChannelPairingRule,
    InstrumentChannelPairingRuleCatalogue,
    InstrumentChannelPairingRuleCatalogueValidationEvidence,
    InstrumentChannelPairingRuleRequest,
    InstrumentChannelPairingRuleScientificValidationDisposition,
    InstrumentChannelPairingRuleValidationEvidence,
    InstrumentChannelPairingRuleValidationStatus,
    InstrumentChannelPairingToleranceProvenance,
    resolve_instrument_channel_pairing_rule,
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


class TestInstrumentChannelPairingRuleCatalogueContract(
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
            "rules": (
                cls._validated_rule(),
                cls._validated_rule(
                    rule_id="survey-a-to-survey-c-v1",
                    channel_instrument="Survey C",
                    channel="SurveyC/V",
                ),
            ),
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

    def test_catalogue_is_immutable_json_safe_and_round_trips(self):
        catalogue = self._catalogue()

        self.assertEqual(
            catalogue.schema_version,
            INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_SCHEMA_VERSION,
        )
        self.assertEqual(len(catalogue), 2)
        self.assertTrue(catalogue.scientifically_validated)
        self.assertTrue(
            catalogue.all_rules_scientifically_validated
        )
        self.assertEqual(tuple(catalogue), catalogue.rules)

        payload = catalogue.to_dict()
        json.dumps(payload, allow_nan=False)

        self.assertEqual(payload["n_rules"], 2)
        self.assertFalse(
            payload["automatic_catalogue_discovery_implemented"]
        )
        self.assertFalse(
            payload["automatic_catalogue_activation_implemented"]
        )
        self.assertFalse(payload["workflow_integration_implemented"])
        self.assertFalse(
            payload["populated_builtin_catalogue_available"]
        )

        restored = InstrumentChannelPairingRuleCatalogue.from_dict(
            payload
        )
        self.assertEqual(restored, catalogue)
        self.assertEqual(restored.to_dict(), payload)

        with self.assertRaises(FrozenInstanceError):
            catalogue.catalogue_version = "replacement"

    def test_catalogue_requires_nonempty_typed_unique_rules(self):
        with self.assertRaisesRegex(ValueError, "at least one rule"):
            self._catalogue(rules=())

        with self.assertRaisesRegex(
            TypeError,
            "only InstrumentChannelPairingRule",
        ):
            self._catalogue(rules=(self._validated_rule(), object()))

        rule = self._validated_rule()
        with self.assertRaisesRegex(
            ValueError,
            "identifiers must be unique",
        ):
            self._catalogue(rules=(rule, rule))

        duplicate_identity = self._validated_rule(
            rule_id="different-rule-id"
        )
        with self.assertRaisesRegex(
            ValueError,
            "identities must be unique",
        ):
            self._catalogue(rules=(rule, duplicate_identity))

    def test_validated_catalogue_cannot_upgrade_unvalidated_rules(self):
        unvalidated_rule = self._validated_rule(
            validation_status="defined_not_validated",
            evidence_reference=None,
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires an evidence_reference",
        ):
            self._catalogue(evidence_reference=None)

        with self.assertRaisesRegex(
            ValueError,
            "cannot contain rules that are not scientifically validated",
        ):
            self._catalogue(rules=(unvalidated_rule,))

        contract_only = self._catalogue(
            rules=(unvalidated_rule,),
            validation_status="defined_not_validated",
            evidence_reference=None,
        )
        self.assertFalse(contract_only.scientifically_validated)
        self.assertFalse(
            contract_only.all_rules_scientifically_validated
        )

    def test_strict_deserialization_rejects_schema_and_flag_changes(self):
        payload = self._catalogue().to_dict()

        changed_schema = dict(payload)
        changed_schema["schema_version"] = "unsupported"
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported.*catalogue schema version",
        ):
            InstrumentChannelPairingRuleCatalogue.from_dict(
                changed_schema
            )

        changed_flag = dict(payload)
        changed_flag[
            "automatic_catalogue_activation_implemented"
        ] = True
        with self.assertRaisesRegex(
            ValueError,
            "strict contract representation",
        ):
            InstrumentChannelPairingRuleCatalogue.from_dict(
                changed_flag
            )

        missing_field = dict(payload)
        missing_field.pop("provenance_reference")
        with self.assertRaisesRegex(
            ValueError,
            "exactly the contract fields",
        ):
            InstrumentChannelPairingRuleCatalogue.from_dict(
                missing_field
            )

    def test_catalogue_uses_existing_explicit_resolver_boundary(self):
        catalogue = self._catalogue()
        request = InstrumentChannelPairingRuleRequest(
            reference_instrument="Survey A",
            reference_channel="SurveyA/V",
            channel_instrument="Survey C",
            channel="SurveyC/V",
            physical_wavelength=0.55,
        )

        resolved = resolve_instrument_channel_pairing_rule(
            request,
            catalogue,
        )

        self.assertIs(resolved, catalogue.rules[1])
        self.assertEqual(
            resolved.rule_id,
            "survey-a-to-survey-c-v1",
        )

        absent = InstrumentChannelPairingRuleRequest(
            reference_instrument="Survey A",
            reference_channel="SurveyA/V",
            channel_instrument="Survey D",
            channel="SurveyD/V",
            physical_wavelength=0.55,
        )
        with self.assertRaisesRegex(
            ValueError,
            "No pairing rule matches the exact requested",
        ):
            resolve_instrument_channel_pairing_rule(
                absent,
                catalogue,
            )

    def test_public_documentation_preserves_activation_boundary(self):
        repo = Path(__file__).resolve().parents[1]
        api_text = (
            repo
            / "docs/source/pgmuvi.instrument_channel_calibration.rst"
        ).read_text(encoding="utf-8")
        future_text = (
            repo / "docs/source/future_work.rst"
        ).read_text(encoding="utf-8")

        for token in (
            "InstrumentChannelPairingRuleCatalogue",
            "catalogue identifier",
            "catalogue version",
            "provenance reference",
            "strict JSON-safe",
            "automatic discovery",
            "automatic activation",
            "caller-supplied",
        ):
            self.assertIn(token, api_text)

        self.assertIn(
            "pairing-rule catalogue contract",
            future_text,
        )
        self.assertIn(
            "populated scientifically validated rule catalogue",
            future_text,
        )
        self.assertIn(
            "workflow integration",
            future_text,
        )
        self.assertIn(
            "TBD[instrument-channel-calibration]",
            future_text,
        )


if __name__ == "__main__":
    unittest.main()
