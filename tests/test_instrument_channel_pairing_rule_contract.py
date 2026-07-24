"""Instrument-specific pairing and tolerance contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    INSTRUMENT_CHANNEL_PAIRING_RULE_REQUEST_SCHEMA_VERSION,
    INSTRUMENT_CHANNEL_PAIRING_RULE_SCHEMA_VERSION,
    InstrumentChannelPairingMethod,
    InstrumentChannelPairingRule,
    InstrumentChannelPairingRuleRequest,
    InstrumentChannelPairingRuleValidationStatus,
    InstrumentChannelPairingToleranceProvenance,
    resolve_instrument_channel_pairing_rule,
)


class TestInstrumentChannelPairingRuleContract(unittest.TestCase):
    @staticmethod
    def _nearest_rule(**overrides):
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

    def test_exact_rule_is_immutable_and_json_safe(self):
        rule = InstrumentChannelPairingRule(
            rule_id="instrument-a-exact-v1",
            reference_instrument="Instrument A",
            reference_channel="InstrumentA/stream-0",
            channel_instrument="Instrument A",
            channel="InstrumentA/stream-1",
            physical_wavelength=1.25,
            pairing_method="exact_timestamp",
            time_unit="day",
            maximum_time_separation=None,
            tolerance_provenance="exact_timestamp",
            validation_status="defined_not_validated",
            notes="Contract-only exact-time rule.",
        )

        self.assertEqual(
            rule.schema_version,
            INSTRUMENT_CHANNEL_PAIRING_RULE_SCHEMA_VERSION,
        )
        self.assertEqual(
            rule.pairing_method,
            InstrumentChannelPairingMethod.EXACT_TIMESTAMP,
        )
        self.assertFalse(rule.scientifically_validated)

        payload = rule.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertTrue(
            payload["observational_channel_identity_explicit"]
        )
        self.assertTrue(payload["instrument_identity_explicit"])
        self.assertFalse(
            payload["physical_wavelength_used_as_rule_identity"]
        )
        self.assertFalse(
            payload["automatic_rule_selection_permitted"]
        )
        self.assertEqual(
            payload["activation_status"],
            "defined_not_activated",
        )

        with self.assertRaises(FrozenInstanceError):
            rule.time_unit = "second"

    def test_validated_nearest_rule_preserves_evidence(self):
        rule = self._nearest_rule()

        self.assertTrue(rule.scientifically_validated)
        self.assertEqual(
            rule.tolerance_provenance,
            InstrumentChannelPairingToleranceProvenance
            .EMPIRICAL_VALIDATION,
        )
        self.assertEqual(
            rule.identity_key,
            (
                "Survey A",
                "SurveyA/V",
                "Survey B",
                "SurveyB/V",
                0.55,
            ),
        )

        payload = rule.to_dict()
        self.assertEqual(
            payload["evidence_reference"],
            "validation:representative-lpv-v1",
        )
        self.assertEqual(
            payload["maximum_time_separation"],
            0.25,
        )

    def test_exact_rule_rejects_nonzero_tolerance(self):
        with self.assertRaisesRegex(
            ValueError,
            "permit no non-zero",
        ):
            InstrumentChannelPairingRule(
                rule_id="bad-exact",
                reference_instrument="A",
                reference_channel="A0",
                channel_instrument="B",
                channel="B0",
                physical_wavelength=1.0,
                pairing_method="exact_timestamp",
                time_unit="day",
                maximum_time_separation=0.1,
                tolerance_provenance="exact_timestamp",
                validation_status="defined_not_validated",
            )

    def test_exact_rule_requires_exact_provenance(self):
        with self.assertRaisesRegex(
            ValueError,
            "require.*exact_timestamp",
        ):
            InstrumentChannelPairingRule(
                rule_id="bad-exact-provenance",
                reference_instrument="A",
                reference_channel="A0",
                channel_instrument="B",
                channel="B0",
                physical_wavelength=1.0,
                pairing_method="exact_timestamp",
                time_unit="day",
                maximum_time_separation=None,
                tolerance_provenance="caller_supplied",
                validation_status="defined_not_validated",
            )

    def test_nearest_rule_requires_positive_tolerance(self):
        for value in (None, 0.0, -0.1, np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite (positive|and non-negative)",
                ):
                    self._nearest_rule(
                        maximum_time_separation=value,
                    )

    def test_validated_rule_requires_noncaller_evidence(self):
        with self.assertRaisesRegex(
            ValueError,
            "evidence_reference",
        ):
            self._nearest_rule(evidence_reference=None)

        with self.assertRaisesRegex(
            ValueError,
            "cannot use caller_supplied",
        ):
            self._nearest_rule(
                tolerance_provenance="caller_supplied",
            )

    def test_identity_and_numeric_inputs_are_strict(self):
        with self.assertRaisesRegex(
            ValueError,
            "different observational channels",
        ):
            self._nearest_rule(channel="SurveyA/V")

        with self.assertRaisesRegex(
            TypeError,
            "not boolean",
        ):
            self._nearest_rule(physical_wavelength=True)

        with self.assertRaisesRegex(
            TypeError,
            "not boolean",
        ):
            self._nearest_rule(maximum_time_separation=True)

    def test_request_is_explicit_immutable_and_json_safe(self):
        request = InstrumentChannelPairingRuleRequest(
            reference_instrument="Survey A",
            reference_channel="SurveyA/V",
            channel_instrument="Survey B",
            channel="SurveyB/V",
            physical_wavelength=0.55,
        )

        self.assertEqual(
            request.schema_version,
            INSTRUMENT_CHANNEL_PAIRING_RULE_REQUEST_SCHEMA_VERSION,
        )
        self.assertEqual(
            request.identity_key,
            (
                "Survey A",
                "SurveyA/V",
                "Survey B",
                "SurveyB/V",
                0.55,
            ),
        )

        payload = request.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertTrue(payload["selection_requested"])
        self.assertTrue(
            payload["scientifically_validated_rule_required"]
        )
        self.assertFalse(
            payload["automatic_rule_selection_implemented"]
        )

        with self.assertRaises(FrozenInstanceError):
            request.channel = "replacement"

    def test_resolver_validates_inputs_then_remains_inactive(self):
        request = InstrumentChannelPairingRuleRequest(
            reference_instrument="Survey A",
            reference_channel="SurveyA/V",
            channel_instrument="Survey B",
            channel="SurveyB/V",
            physical_wavelength=0.55,
        )
        rule = self._nearest_rule()

        with self.assertRaisesRegex(TypeError, "request must be"):
            resolve_instrument_channel_pairing_rule(
                object(),
                (rule,),
            )

        with self.assertRaisesRegex(ValueError, "at least one"):
            resolve_instrument_channel_pairing_rule(
                request,
                (),
            )

        with self.assertRaisesRegex(
            NotImplementedError,
            "defined but not implemented",
        ):
            resolve_instrument_channel_pairing_rule(
                request,
                (rule,),
            )

    def test_resolver_rejects_ambiguous_rule_sets(self):
        request = InstrumentChannelPairingRuleRequest(
            reference_instrument="Survey A",
            reference_channel="SurveyA/V",
            channel_instrument="Survey B",
            channel="SurveyB/V",
            physical_wavelength=0.55,
        )
        rule = self._nearest_rule()

        with self.assertRaisesRegex(
            ValueError,
            "identifiers must be unique",
        ):
            resolve_instrument_channel_pairing_rule(
                request,
                (rule, rule),
            )

        second = self._nearest_rule(rule_id="second-id")
        with self.assertRaisesRegex(
            ValueError,
            "identities must be unique",
        ):
            resolve_instrument_channel_pairing_rule(
                request,
                (rule, second),
            )

    def test_public_documentation_records_contract_boundary(self):
        repo = Path(__file__).resolve().parents[1]
        api_text = (
            repo
            / "docs/source/pgmuvi.instrument_channel_calibration.rst"
        ).read_text(encoding="utf-8")
        future_text = (
            repo / "docs/source/future_work.rst"
        ).read_text(encoding="utf-8")

        for token in (
            "InstrumentChannelPairingRule",
            "InstrumentChannelPairingRuleRequest",
            "resolve_instrument_channel_pairing_rule",
            "defined but not activated",
            "physical wavelength",
            "observational channel",
        ):
            self.assertIn(token, api_text)

        self.assertIn(
            "instrument-specific pairing-rule contract",
            future_text,
        )
        self.assertIn(
            "automatic rule resolution remains unimplemented",
            future_text,
        )
        self.assertIn(
            "TBD[instrument-channel-calibration]",
            future_text,
        )


if __name__ == "__main__":
    unittest.main()
