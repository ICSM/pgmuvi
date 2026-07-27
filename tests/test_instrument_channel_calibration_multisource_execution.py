from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from pgmuvi.instrument_channel_calibration_multisource_execution import (
    InstrumentChannelCalibrationMultiSourceChannelData,
    InstrumentChannelCalibrationMultiSourceSourceData,
    execute_instrument_channel_calibration_multisource_validation,
)
from pgmuvi.instrument_channel_calibration_multisource_validation import (
    InstrumentChannelCalibrationMultiSourceValidationDisposition,
    InstrumentChannelCalibrationMultiSourceValidationProtocol,
)
from pgmuvi.instrument_channel_calibration_validation import (
    InstrumentChannelCalibrationValidationProtocol,
)


ROOT = Path(__file__).resolve().parents[1]
MULTISOURCE_PROTOCOL_PATH = (
    ROOT
    / "examples"
    / "validation"
    / "kelt_r3_maintainer_multisource_validation_protocol_v1.json"
)
CANDIDATE_PROTOCOL_PATH = (
    ROOT
    / "examples"
    / "validation"
    / "kelt_r3_pairing_validation_protocol_v1.json"
)


class TestInstrumentChannelCalibrationMultiSourceExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = (
            InstrumentChannelCalibrationMultiSourceValidationProtocol.from_dict(
                json.loads(
                    MULTISOURCE_PROTOCOL_PATH.read_text(encoding="utf-8")
                )
            )
        )
        cls.candidate = InstrumentChannelCalibrationValidationProtocol.from_dict(
            json.loads(CANDIDATE_PROTOCOL_PATH.read_text(encoding="utf-8"))
        )

    def source(
        self,
        source_id: str,
        *,
        n_pairs: int = 100,
        offset: float = 0.3,
        scale: float = 1.2,
        phase: float = 0.0,
        reference_channel: str | None = None,
        target_channel: str | None = None,
        error: float = 0.01,
        is_primary: bool = True,
    ) -> InstrumentChannelCalibrationMultiSourceSourceData:
        time = np.arange(n_pairs, dtype=float)
        target = (
            2.0
            + 0.5 * np.sin(2.0 * np.pi * time / 20.0 + phase)
            + 0.1 * np.cos(2.0 * np.pi * time / 7.0)
        )
        reference = offset + scale * target
        uncertainty = np.full(n_pairs, error)
        wavelength = np.full(n_pairs, self.protocol.physical_wavelength)
        return InstrumentChannelCalibrationMultiSourceSourceData(
            astrophysical_source_id=source_id,
            reference=InstrumentChannelCalibrationMultiSourceChannelData(
                channel=(
                    reference_channel or self.protocol.reference_channel
                ),
                time=tuple(time),
                flux=tuple(reference),
                flux_error=tuple(uncertainty),
                wavelength=tuple(wavelength),
            ),
            channel=InstrumentChannelCalibrationMultiSourceChannelData(
                channel=target_channel or self.protocol.channel,
                time=tuple(time + 0.01),
                flux=tuple(target),
                flux_error=tuple(uncertainty),
                wavelength=tuple(wavelength),
            ),
            is_primary_source=is_primary,
        )

    def execute(
        self,
        sources: tuple[
            InstrumentChannelCalibrationMultiSourceSourceData,
            ...,
        ],
        *,
        seed: int = 20260726,
        n_sources: int | str = 5,
    ):
        return execute_instrument_channel_calibration_multisource_validation(
            sources,
            protocol=self.protocol,
            candidate_protocol=self.candidate,
            private_input_sha256="a" * 64,
            selection_seed=seed,
            n_sources=n_sources,
            package_version="test",
            package_commit="b" * 40,
            executed_at_utc="2026-07-26T00:00:00Z",
        )

    def test_five_transferable_sources_pass(self) -> None:
        sources = tuple(
            self.source(
                f"source-{index}",
                n_pairs=100 + 5 * index,
                phase=0.1 * index,
            )
            for index in range(5)
        )
        result = self.execute(sources)

        self.assertIs(
            result.disposition,
            InstrumentChannelCalibrationMultiSourceValidationDisposition.PASSED,
        )
        self.assertEqual(result.summary.successful_source_holdout_count, 5)
        self.assertEqual(len(result.holdout_results), 5)
        self.assertLess(
            result.summary.worst_source_holdout_normalized_rmse,
            0.01,
        )
        self.assertFalse(
            result.summary.to_dict()["raw_source_identifiers_included"]
        )
        private = result.to_private_dict()
        self.assertIn("source-0", json.dumps(private))
        self.assertNotIn(
            "source-0",
            json.dumps(result.summary.to_dict()),
        )

    def test_selection_is_seeded_and_reproducible(self) -> None:
        sources = tuple(
            self.source(f"source-{index}", phase=0.05 * index)
            for index in range(8)
        )
        first = self.execute(sources, seed=314159)
        second = self.execute(tuple(reversed(sources)), seed=314159)
        third = self.execute(sources, seed=271828)

        self.assertEqual(first.selected_source_ids, second.selected_source_ids)
        self.assertNotEqual(first.selected_source_ids, third.selected_source_ids)
        self.assertEqual(
            first.summary.selected_source_id_sha256s,
            second.summary.selected_source_id_sha256s,
        )

    def test_eligibility_is_decided_before_fit(self) -> None:
        sources = (
            *(self.source(f"good-{index}") for index in range(5)),
            self.source("too-short", n_pairs=99),
            self.source(
                "wrong-reference",
                reference_channel="OTHER/reference",
            ),
            self.source("derived", is_primary=False),
        )
        result = self.execute(sources)
        by_id = {
            item.astrophysical_source_id: item
            for item in result.eligibility_results
        }

        self.assertFalse(by_id["too-short"].eligible)
        self.assertIn(
            "insufficient_matched_pairs",
            by_id["too-short"].reasons,
        )
        self.assertFalse(by_id["wrong-reference"].eligible)
        self.assertIn(
            "missing_exact_reference_observational_channel",
            by_id["wrong-reference"].reasons,
        )
        self.assertFalse(by_id["derived"].eligible)
        self.assertIn("source_not_asserted_primary", by_id["derived"].reasons)
        self.assertNotIn("too-short", result.selected_source_ids)

    def test_training_is_source_balanced(self) -> None:
        sources = tuple(
            self.source(
                f"source-{index}",
                n_pairs=(180 if index == 0 else 100 + index),
            )
            for index in range(5)
        )
        result = self.execute(sources)

        for holdout in result.holdout_results:
            training_counts = [
                next(
                    item.n_matched_pairs
                    for item in result.eligibility_results
                    if item.source_id_sha256 == source_hash
                )
                for source_hash in holdout.training_source_id_sha256s
            ]
            self.assertEqual(
                holdout.matched_pairs_per_training_source,
                min(training_counts),
            )
            self.assertEqual(
                holdout.n_training_pairs,
                holdout.matched_pairs_per_training_source * 4,
            )
            self.assertEqual(len(holdout.training_samples), 4)
            for sample in holdout.training_samples:
                self.assertEqual(
                    len(sample.selected_pair_positions),
                    holdout.matched_pairs_per_training_source,
                )
                self.assertEqual(
                    tuple(sorted(sample.selected_pair_positions)),
                    sample.selected_pair_positions,
                )
                self.assertGreaterEqual(sample.derived_seed, 0)
            self.assertEqual(len(holdout.fold_results), 5)

    def test_nontransferable_source_fails_frozen_gates(self) -> None:
        sources = tuple(
            self.source(
                f"source-{index}",
                offset=(3.0 if index == 4 else 0.3),
                scale=(0.5 if index == 4 else 1.2),
                phase=0.1 * index,
            )
            for index in range(5)
        )
        result = self.execute(sources)

        self.assertIs(
            result.disposition,
            InstrumentChannelCalibrationMultiSourceValidationDisposition.FAILED,
        )
        self.assertTrue(result.reasons)
        self.assertGreater(
            result.summary.worst_source_holdout_normalized_rmse,
            self.protocol.maximum_worst_source_holdout_normalized_rmse,
        )

    def test_four_eligible_sources_are_inconclusive(self) -> None:
        result = self.execute(
            tuple(self.source(f"source-{index}") for index in range(4))
        )

        self.assertIs(
            result.disposition,
            InstrumentChannelCalibrationMultiSourceValidationDisposition.
            INCONCLUSIVE,
        )
        self.assertEqual(
            result.reasons,
            ("insufficient_private_validation_sample",),
        )
        self.assertEqual(result.holdout_results, ())
        self.assertEqual(result.summary.successful_source_holdout_count, 0)

    def test_protocol_digest_mismatch_is_rejected(self) -> None:
        mismatched = replace(
            self.protocol,
            candidate_protocol_sha256="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "candidate protocol digest"):
            execute_instrument_channel_calibration_multisource_validation(
                tuple(self.source(f"source-{index}") for index in range(5)),
                protocol=mismatched,
                candidate_protocol=self.candidate,
                private_input_sha256="a" * 64,
                selection_seed=1,
                package_version="test",
                package_commit="b" * 40,
                executed_at_utc="2026-07-26T00:00:00Z",
            )

    def test_requested_count_larger_than_eligible_is_inconclusive(self) -> None:
        result = self.execute(
            tuple(self.source(f"source-{index}") for index in range(6)),
            n_sources=7,
        )
        self.assertIs(
            result.disposition,
            InstrumentChannelCalibrationMultiSourceValidationDisposition.
            INCONCLUSIVE,
        )
        self.assertEqual(
            result.reasons,
            ("insufficient_requested_source_count",),
        )
        self.assertEqual(result.holdout_results, ())

    def test_all_eligible_sources_can_be_evaluated(self) -> None:
        sources = tuple(
            self.source(f"source-{index}", phase=0.03 * index)
            for index in range(7)
        )
        result = self.execute(sources, n_sources="all")
        self.assertEqual(len(result.selected_source_ids), 7)
        self.assertEqual(len(result.holdout_results), 7)


if __name__ == "__main__":
    unittest.main()
