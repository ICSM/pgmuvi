"""Scientific-validation evidence contract tests for pairing catalogues."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_VALIDATION_EVIDENCE_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_PAIRING_RULE_VALIDATION_EVIDENCE_SCHEMA_VERSION,
    InstrumentChannelPairingMethod,
    InstrumentChannelPairingRule,
    InstrumentChannelPairingRuleCatalogue,
    InstrumentChannelPairingRuleCatalogueValidationEvidence,
    InstrumentChannelPairingRuleScientificValidationDisposition,
    InstrumentChannelPairingRuleValidationEvidence,
    InstrumentChannelPairingRuleValidationStatus,
    InstrumentChannelPairingToleranceProvenance,
    assess_instrument_channel_pairing_rule_catalogue_scientific_validation,
    assess_instrument_channel_pairing_rule_scientific_validation,
)


class TestPairingRuleCatalogueScientificValidationContract(
    unittest.TestCase
):
    @staticmethod
    def _rule_values(**overrides):
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
            "evidence_reference": "result:rule-validation-v1",
        }
        values.update(overrides)
        return values

    @classmethod
    def _rule_evidence(cls, **overrides):
        values = cls._rule_values()
        evidence_values = {
            "validation_id": "rule-validation-v1",
            "validation_version": "1",
            "rule_id": values["rule_id"],
            "reference_instrument": values["reference_instrument"],
            "reference_channel": values["reference_channel"],
            "channel_instrument": values["channel_instrument"],
            "channel": values["channel"],
            "physical_wavelength": values["physical_wavelength"],
            "pairing_method": values["pairing_method"],
            "time_unit": values["time_unit"],
            "maximum_time_separation": (
                values["maximum_time_separation"]
            ),
            "tolerance_provenance": values["tolerance_provenance"],
            "validation_dataset_reference": "dataset:validation-v1",
            "validation_dataset_sha256": "b" * 64,
            "validation_protocol_reference": "protocol:validation-v1",
            "validation_result_reference": (
                values["evidence_reference"]
            ),
            "reference_channel_justification": (
                "Survey A is the explicitly justified reference."
            ),
            "pairing_method_justification": (
                "Nearest matching is justified for this cadence."
            ),
            "time_tolerance_justification": (
                "The tolerance is justified by validation results."
            ),
            "applicability_boundaries": (
                "Only SurveyA/V to SurveyB/V at 0.55.",
            ),
            "acceptance_criteria": (
                "Bias and residual acceptance thresholds pass.",
            ),
            "n_validation_sources": 4,
            "n_matched_pairs": 80,
            "disposition": (
                InstrumentChannelPairingRuleScientificValidationDisposition
                .PASSED
            ),
        }
        evidence_values.update(overrides)
        return InstrumentChannelPairingRuleValidationEvidence(
            **evidence_values
        )

    @classmethod
    def _validated_rule(cls, **overrides):
        values = cls._rule_values()
        values.update(overrides)
        if "validation_evidence" not in overrides:
            if values["evidence_reference"] is not None:
                values["validation_evidence"] = cls._rule_evidence(
                rule_id=values["rule_id"],
                reference_instrument=values["reference_instrument"],
                reference_channel=values["reference_channel"],
                channel_instrument=values["channel_instrument"],
                channel=values["channel"],
                physical_wavelength=values["physical_wavelength"],
                pairing_method=values["pairing_method"],
                time_unit=values["time_unit"],
                maximum_time_separation=(
                    values["maximum_time_separation"]
                ),
                tolerance_provenance=values["tolerance_provenance"],
                validation_result_reference=(
                    values["evidence_reference"]
                ),
            )
            else:
                values["validation_evidence"] = None
        return InstrumentChannelPairingRule(**values)

    @classmethod
    def _catalogue_evidence(cls, rules, **overrides):
        rules = tuple(rules)
        values = {
            "validation_id": "catalogue-validation-v1",
            "validation_version": "1",
            "catalogue_id": "pgmuvi-pairing-rules",
            "catalogue_version": "2026.1",
            "catalogue_schema_version": (
                INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_SCHEMA_VERSION
            ),
            "member_rule_ids": tuple(rule.rule_id for rule in rules),
            "member_validation_ids": tuple(
                (
                    rule.validation_evidence.validation_id
                    if rule.validation_evidence is not None
                    else f"missing:{rule.rule_id}"
                )
                for rule in rules
            ),
            "validation_protocol_reference": (
                "protocol:catalogue-validation-v1"
            ),
            "validation_result_reference": (
                "result:catalogue-validation-v1"
            ),
            "applicability_boundaries": (
                "Only the exact ordered member-rule set.",
            ),
            "acceptance_criteria": (
                "Every member rule passes its evidence contract.",
            ),
            "disposition": (
                InstrumentChannelPairingRuleScientificValidationDisposition
                .PASSED
            ),
        }
        values.update(overrides)
        return InstrumentChannelPairingRuleCatalogueValidationEvidence(
            **values
        )

    @classmethod
    def _validated_catalogue(cls, **overrides):
        rules = overrides.pop(
            "rules",
            (
                cls._validated_rule(),
                cls._validated_rule(
                    rule_id="survey-a-to-survey-c-v1",
                    channel_instrument="Survey C",
                    channel="SurveyC/V",
                    evidence_reference="result:rule-validation-c-v1",
                    validation_evidence=cls._rule_evidence(
                        validation_id="rule-validation-c-v1",
                        rule_id="survey-a-to-survey-c-v1",
                        channel_instrument="Survey C",
                        channel="SurveyC/V",
                        validation_result_reference=(
                            "result:rule-validation-c-v1"
                        ),
                    ),
                ),
            ),
        )
        values = {
            "catalogue_id": "pgmuvi-pairing-rules",
            "catalogue_version": "2026.1",
            "provenance_reference": "catalogue:design-record-v1",
            "rules": tuple(rules),
            "validation_status": (
                InstrumentChannelPairingRuleValidationStatus
                .SCIENTIFICALLY_VALIDATED
            ),
            "evidence_reference": "result:catalogue-validation-v1",
        }
        values.update(overrides)
        if "validation_evidence" not in overrides:
            rules = tuple(values["rules"])
            if (
                values["evidence_reference"] is not None
                and bool(rules)
                and all(
                    rule.validation_evidence is not None
                    for rule in rules
                )
            ):
                values["validation_evidence"] = cls._catalogue_evidence(
                    rules,
                    catalogue_id=values["catalogue_id"],
                    catalogue_version=values["catalogue_version"],
                    validation_result_reference=(
                        values["evidence_reference"]
                    ),
                )
            else:
                values["validation_evidence"] = None
        return InstrumentChannelPairingRuleCatalogue(**values)

    def test_rule_evidence_is_immutable_strict_and_round_trips(self):
        evidence = self._rule_evidence()

        self.assertEqual(
            evidence.schema_version,
            INSTRUMENT_CHANNEL_PAIRING_RULE_VALIDATION_EVIDENCE_SCHEMA_VERSION,
        )
        self.assertTrue(evidence.passed)

        payload = evidence.to_dict()
        json.dumps(payload, allow_nan=False)
        restored = InstrumentChannelPairingRuleValidationEvidence.from_dict(
            payload
        )

        self.assertEqual(restored, evidence)
        self.assertEqual(restored.to_dict(), payload)
        self.assertFalse(
            payload[
                "scientific_validation_execution_performed_by_pgmuvi"
            ]
        )
        self.assertFalse(payload["populated_builtin_evidence"])

        with self.assertRaises(FrozenInstanceError):
            evidence.validation_id = "replacement"

    def test_rule_validation_requires_exact_identity_configuration_and_pass(self):
        rule = self._validated_rule()
        report = (
            assess_instrument_channel_pairing_rule_scientific_validation(
                rule
            )
        )

        self.assertTrue(rule.scientifically_validated)
        self.assertTrue(report.scientifically_validated)
        self.assertEqual(report.reasons, ())
        json.dumps(report.to_dict(), allow_nan=False)
        self.assertFalse(report.to_dict()["scientific_validation_executed"])

        mismatches = (
            (
                self._rule_evidence(channel="SurveyX/V"),
                "identity_mismatch",
            ),
            (
                self._rule_evidence(maximum_time_separation=0.5),
                "configuration_mismatch",
            ),
            (
                self._rule_evidence(
                    disposition=(
                        InstrumentChannelPairingRuleScientificValidationDisposition
                        .FAILED
                    )
                ),
                "not_passed",
            ),
        )
        for evidence, reason_fragment in mismatches:
            with self.subTest(reason_fragment=reason_fragment):
                with self.assertRaisesRegex(
                    ValueError,
                    reason_fragment,
                ):
                    self._validated_rule(validation_evidence=evidence)

    def test_unvalidated_rule_reports_deterministic_reasons(self):
        rule = InstrumentChannelPairingRule(
            **self._rule_values(
                validation_status=(
                    InstrumentChannelPairingRuleValidationStatus
                    .DEFINED_NOT_VALIDATED
                ),
                evidence_reference=None,
                validation_evidence=None,
            )
        )
        report = (
            assess_instrument_channel_pairing_rule_scientific_validation(
                rule
            )
        )

        self.assertFalse(rule.scientifically_validated)
        self.assertEqual(
            report.reasons,
            (
                "rule_status_not_scientifically_validated",
                "rule_validation_evidence_missing",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "reasons must exactly describe",
        ):
            replace(report, reasons=("forged",))

    def test_catalogue_evidence_is_immutable_strict_and_round_trips(self):
        rules = (self._validated_rule(),)
        evidence = self._catalogue_evidence(rules)

        self.assertEqual(
            evidence.schema_version,
            INSTRUMENT_CHANNEL_PAIRING_RULE_CATALOGUE_VALIDATION_EVIDENCE_SCHEMA_VERSION,
        )
        payload = evidence.to_dict()
        json.dumps(payload, allow_nan=False)

        restored = (
            InstrumentChannelPairingRuleCatalogueValidationEvidence.from_dict(
                payload
            )
        )
        self.assertEqual(restored, evidence)
        self.assertEqual(restored.to_dict(), payload)

        with self.assertRaises(FrozenInstanceError):
            evidence.member_rule_ids = ("replacement",)

    def test_catalogue_validation_binds_exact_ordered_members_and_evidence(self):
        catalogue = self._validated_catalogue()
        report = (
            assess_instrument_channel_pairing_rule_catalogue_scientific_validation(
                catalogue
            )
        )

        self.assertTrue(catalogue.scientifically_validated)
        self.assertTrue(
            catalogue.all_rules_scientifically_validated
        )
        self.assertTrue(report.scientifically_validated)
        self.assertEqual(report.reasons, ())
        self.assertEqual(report.unvalidated_rule_ids, ())
        json.dumps(report.to_dict(), allow_nan=False)

        reversed_evidence = self._catalogue_evidence(
            catalogue.rules,
            member_rule_ids=tuple(
                reversed(
                    tuple(rule.rule_id for rule in catalogue.rules)
                )
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "member_rule_ids_mismatch",
        ):
            self._validated_catalogue(
                rules=catalogue.rules,
                validation_evidence=reversed_evidence,
            )

    def test_catalogue_cannot_upgrade_unvalidated_member_evidence(self):
        unvalidated = InstrumentChannelPairingRule(
            **self._rule_values(
                validation_status=(
                    InstrumentChannelPairingRuleValidationStatus
                    .DEFINED_NOT_VALIDATED
                ),
                evidence_reference=None,
                validation_evidence=None,
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            "cannot contain rules that are not scientifically validated",
        ):
            self._validated_catalogue(rules=(unvalidated,))

    def test_public_docs_preserve_contract_only_boundary(self):
        repo = Path(__file__).resolve().parents[1]
        api_text = (
            repo
            / "docs/source/pgmuvi.instrument_channel_calibration.rst"
        ).read_text(encoding="utf-8")
        future_text = (
            repo / "docs/source/future_work.rst"
        ).read_text(encoding="utf-8")

        for token in (
            "InstrumentChannelPairingRuleValidationEvidence",
            "InstrumentChannelPairingRuleCatalogueValidationEvidence",
            "identity-bound",
            "configuration-bound",
            "applicability boundaries",
            "acceptance criteria",
            "does not execute scientific validation",
            "no populated built-in evidence",
        ):
            self.assertIn(token, api_text)

        for token in (
            "typed scientific-validation evidence contract",
            "populated scientifically validated rule catalogue",
            "representative calibration validation",
            "TBD[instrument-channel-calibration]",
        ):
            self.assertIn(token, future_text)


if __name__ == "__main__":
    unittest.main()
