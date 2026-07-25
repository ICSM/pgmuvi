"""Maintained instrument-channel validation execution tests."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pgmuvi.instrument_channel_calibration import (
    construct_instrument_channel_pairing,
    fit_instrument_channel_calibration,
)
from pgmuvi.instrument_channel_calibration_validation import (
    InstrumentChannelCalibrationValidationDataset,
    InstrumentChannelCalibrationValidationDatasetManifest,
    InstrumentChannelCalibrationValidationProtocol,
)
from pgmuvi.instrument_channel_calibration_validation_execution import (
    _construct_partitioned_pair_positions,
    execute_instrument_channel_calibration_validation_dataset_manifest,
    execute_instrument_channel_calibration_validation_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
COMMITTED_PROTOCOL_PATH = (
    ROOT
    / "examples"
    / "validation"
    / "kelt_r3_pairing_validation_protocol_v1.json"
)


class TestInstrumentChannelCalibrationValidationExecution(
    unittest.TestCase
):
    def _repository_fixture(self, root: Path) -> tuple[str, np.ndarray]:
        data_reference = "examples/data/synthetic_validation.csv"
        data_path = root / data_reference
        data_path.parent.mkdir(parents=True)

        channel_flux = np.asarray(
            [
                1.0
                + 0.002 * index
                + 0.15 * math.sin(index / 5.0)
                for index in range(120)
            ],
            dtype=float,
        )

        with data_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time", "flux", "flux_error", "wavelength", "band"]
            )
            for index, target_flux in enumerate(channel_flux):
                reference_flux = 0.02 + 0.68 * target_flux
                writer.writerow(
                    [
                        float(index),
                        reference_flux,
                        0.001,
                        0.6561154962791801,
                        "KELT/OSN_Johnson.Cousins_R3_0",
                    ]
                )
                writer.writerow(
                    [
                        float(index) + 0.01,
                        target_flux,
                        0.001,
                        0.6561154962791801,
                        "KELT/OSN_Johnson.Cousins_R3_1",
                    ]
                )

        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        payload = json.loads(
            COMMITTED_PROTOCOL_PATH.read_text(encoding="utf-8")
        )
        payload["protocol_id"] = "synthetic-execution-protocol-v1"
        payload["rule_id"] = "synthetic-r3-rule-v1"
        payload["anchor_dataset"] = {
            "schema_version": (
                "pgmuvi-instrument-channel-calibration-validation-dataset-v1"
            ),
            "dataset_id": "synthetic-full-v1",
            "astrophysical_source_id": "synthetic-source",
            "dataset_reference": data_reference,
            "dataset_sha256": digest,
            "derivation_parent_dataset_id": None,
            "is_derived": False,
        }

        protocol_reference = "examples/validation/protocol.json"
        protocol_path = root / protocol_reference
        protocol_path.parent.mkdir(parents=True)
        protocol_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        InstrumentChannelCalibrationValidationProtocol.from_dict(payload)
        return protocol_reference, channel_flux

    def test_partitioned_pairing_matches_maintained_global_primitive(self):
        protocol = (
            InstrumentChannelCalibrationValidationProtocol.from_dict(
                json.loads(
                    COMMITTED_PROTOCOL_PATH.read_text(encoding="utf-8")
                )
            )
        )
        reference_times = np.asarray(
            [0.0, 0.04, 10.0, 10.03, 30.0],
            dtype=float,
        )
        channel_times = np.asarray(
            [0.01, 0.05, 10.01, 10.04, 40.0],
            dtype=float,
        )

        direct = construct_instrument_channel_pairing(
            reference_times,
            channel_times,
            reference_channel=protocol.reference_channel,
            channel=protocol.channel,
            wavelength=protocol.physical_wavelength,
            time_unit=protocol.time_unit,
            method=protocol.pairing_method,
            maximum_time_separation=protocol.maximum_time_separation,
            reference_row_indices=np.arange(reference_times.size),
            channel_row_indices=np.arange(channel_times.size),
        )
        partitioned = _construct_partitioned_pair_positions(
            reference_times,
            channel_times,
            protocol=protocol,
        )

        self.assertEqual(
            tuple(partitioned[0]),
            direct.reference_row_indices,
        )
        self.assertEqual(
            tuple(partitioned[1]),
            direct.channel_row_indices,
        )

    def test_execution_is_repository_bound_and_has_no_fold_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_reference, all_channel_flux = (
                self._repository_fixture(root)
            )
            captured_training_flux: list[np.ndarray] = []

            def recording_fit(reference_flux, channel_flux, **keywords):
                captured_training_flux.append(
                    np.asarray(channel_flux, dtype=float).copy()
                )
                return fit_instrument_channel_calibration(
                    reference_flux,
                    channel_flux,
                    **keywords,
                )

            with patch(
                "pgmuvi.instrument_channel_calibration_validation_execution."
                "fit_instrument_channel_calibration",
                side_effect=recording_fit,
            ):
                result, report = (
                    execute_instrument_channel_calibration_validation_protocol(
                        repository_root=root,
                        protocol_reference=protocol_reference,
                        execution_reference=(
                            "examples/validation/result.json"
                        ),
                        package_version="test",
                        package_commit="1" * 40,
                        executed_at_utc="2026-08-01T00:00:00Z",
                    )
                )

            source = result.source_results[0]
            self.assertEqual(source.n_matched_pairs, 120)
            self.assertEqual(source.n_temporal_folds, 5)
            self.assertEqual(source.n_successful_temporal_folds, 5)
            self.assertEqual(len(captured_training_flux), 5)

            all_values = set(float(value) for value in all_channel_flux)
            omitted_sets = []

            for training_values in captured_training_flux:
                self.assertEqual(training_values.size, 96)
                omitted = all_values.difference(
                    float(value) for value in training_values
                )
                self.assertEqual(len(omitted), 24)
                omitted_sets.append(omitted)

            self.assertEqual(
                set().union(*omitted_sets),
                all_values,
            )
            for first_index, first in enumerate(omitted_sets):
                for second in omitted_sets[first_index + 1 :]:
                    self.assertTrue(first.isdisjoint(second))

            self.assertEqual(report.disposition.value, "inconclusive")
            self.assertEqual(
                report.reasons,
                ("insufficient_independent_astrophysical_sources",),
            )
            self.assertFalse(
                result.to_dict()["catalogue_population_performed"]
            )
            self.assertFalse(
                result.to_dict()["scientific_validation_claim_embedded"]
            )

    def test_dataset_digest_mismatch_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_reference, _ = self._repository_fixture(root)
            data_path = root / "examples/data/synthetic_validation.csv"
            data_path.write_text(
                data_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "SHA-256",
            ):
                execute_instrument_channel_calibration_validation_protocol(
                    repository_root=root,
                    protocol_reference=protocol_reference,
                    execution_reference="examples/validation/result.json",
                    package_version="test",
                    package_commit="1" * 40,
                    executed_at_utc="2026-08-01T00:00:00Z",
                )

    def test_dataset_reference_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_reference, _ = self._repository_fixture(root)
            protocol_path = root / protocol_reference
            payload = json.loads(protocol_path.read_text(encoding="utf-8"))
            payload["anchor_dataset"]["dataset_reference"] = "../outside.csv"
            protocol_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "escape",
            ):
                execute_instrument_channel_calibration_validation_protocol(
                    repository_root=root,
                    protocol_reference=protocol_reference,
                    execution_reference="examples/validation/result.json",
                    package_version="test",
                    package_commit="1" * 40,
                    executed_at_utc="2026-08-01T00:00:00Z",
                )


    def _additional_dataset_manifest(
        self,
        root: Path,
        protocol_reference: str,
        *,
        astrophysical_source_id: str = "independent-synthetic-source",
    ) -> tuple[str, Path]:
        protocol_payload = json.loads(
            (root / protocol_reference).read_text(encoding="utf-8")
        )
        protocol = (
            InstrumentChannelCalibrationValidationProtocol.from_dict(
                protocol_payload
            )
        )

        data_reference = "examples/data/independent_validation.csv"
        data_path = root / data_reference
        channel_flux = np.asarray(
            [
                1.2
                + 0.0025 * index
                + 0.18 * math.sin(index / 6.0)
                for index in range(120)
            ],
            dtype=float,
        )

        with data_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time", "flux", "flux_error", "wavelength", "band"]
            )
            for index, target_flux in enumerate(channel_flux):
                reference_flux = 0.02 + 0.68 * target_flux
                writer.writerow(
                    [
                        float(index) + 0.2,
                        reference_flux,
                        0.001,
                        protocol.physical_wavelength,
                        protocol.reference_channel,
                    ]
                )
                writer.writerow(
                    [
                        float(index) + 0.21,
                        target_flux,
                        0.001,
                        protocol.physical_wavelength,
                        protocol.channel,
                    ]
                )

        dataset = InstrumentChannelCalibrationValidationDataset(
            dataset_id="independent-synthetic-full-v1",
            astrophysical_source_id=astrophysical_source_id,
            dataset_reference=data_reference,
            dataset_sha256=hashlib.sha256(
                data_path.read_bytes()
            ).hexdigest(),
        )
        manifest = InstrumentChannelCalibrationValidationDatasetManifest(
            protocol_id=protocol.protocol_id,
            protocol_version=protocol.protocol_version,
            protocol_sha256=protocol.canonical_sha256,
            datasets=(dataset,),
        )

        manifest_reference = "examples/validation/dataset_manifest.json"
        manifest_path = root / manifest_reference
        manifest_path.write_text(
            json.dumps(
                manifest.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest_reference, data_path

    def test_dataset_manifest_contract_is_strict_and_primary_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_reference, _ = self._repository_fixture(root)
            manifest_reference, _ = self._additional_dataset_manifest(
                root,
                protocol_reference,
            )
            payload = json.loads(
                (root / manifest_reference).read_text(encoding="utf-8")
            )
            manifest = (
                InstrumentChannelCalibrationValidationDatasetManifest.from_dict(
                    payload
                )
            )
            self.assertEqual(manifest.to_dict(), payload)
            self.assertEqual(
                len(manifest.datasets),
                1,
            )

            tampered = dict(payload)
            tampered["unexpected"] = True
            with self.assertRaisesRegex(ValueError, "exactly"):
                InstrumentChannelCalibrationValidationDatasetManifest.from_dict(
                    tampered
                )

            derived = InstrumentChannelCalibrationValidationDataset(
                dataset_id="derived-v1",
                astrophysical_source_id="derived-source",
                dataset_reference="examples/data/derived.csv",
                dataset_sha256="a" * 64,
                derivation_parent_dataset_id="parent-v1",
            )
            with self.assertRaisesRegex(ValueError, "primary"):
                InstrumentChannelCalibrationValidationDatasetManifest(
                    protocol_id=manifest.protocol_id,
                    protocol_version=manifest.protocol_version,
                    protocol_sha256=manifest.protocol_sha256,
                    datasets=(derived,),
                )

    def test_manifest_executes_anchor_and_independent_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_reference, _ = self._repository_fixture(root)
            manifest_reference, _ = self._additional_dataset_manifest(
                root,
                protocol_reference,
            )

            result, report = (
                execute_instrument_channel_calibration_validation_dataset_manifest(
                    repository_root=root,
                    protocol_reference=protocol_reference,
                    dataset_manifest_reference=manifest_reference,
                    execution_reference="examples/validation/combined.json",
                    package_version="test",
                    package_commit="1" * 40,
                    executed_at_utc="2026-08-01T00:00:00Z",
                )
            )

            self.assertEqual(len(result.source_results), 2)
            self.assertEqual(
                tuple(
                    source.dataset.astrophysical_source_id
                    for source in result.source_results
                ),
                (
                    "synthetic-source",
                    "independent-synthetic-source",
                ),
            )
            self.assertEqual(
                tuple(
                    source.n_matched_pairs
                    for source in result.source_results
                ),
                (120, 120),
            )
            self.assertEqual(report.disposition.value, "passed")
            self.assertEqual(
                report.independent_astrophysical_source_count,
                2,
            )
            self.assertFalse(
                result.to_dict()["catalogue_population_performed"]
            )

    def test_manifest_rejects_anchor_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_reference, _ = self._repository_fixture(root)
            manifest_reference, _ = self._additional_dataset_manifest(
                root,
                protocol_reference,
                astrophysical_source_id="synthetic-source",
            )

            with self.assertRaisesRegex(ValueError, "independent"):
                execute_instrument_channel_calibration_validation_dataset_manifest(
                    repository_root=root,
                    protocol_reference=protocol_reference,
                    dataset_manifest_reference=manifest_reference,
                    execution_reference="examples/validation/combined.json",
                    package_version="test",
                    package_commit="1" * 40,
                    executed_at_utc="2026-08-01T00:00:00Z",
                )

    def test_manifest_rejects_protocol_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_reference, _ = self._repository_fixture(root)
            manifest_reference, _ = self._additional_dataset_manifest(
                root,
                protocol_reference,
            )
            manifest_path = root / manifest_reference
            payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            payload["protocol_sha256"] = "0" * 64
            manifest_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "digest"):
                execute_instrument_channel_calibration_validation_dataset_manifest(
                    repository_root=root,
                    protocol_reference=protocol_reference,
                    dataset_manifest_reference=manifest_reference,
                    execution_reference="examples/validation/combined.json",
                    package_version="test",
                    package_commit="1" * 40,
                    executed_at_utc="2026-08-01T00:00:00Z",
                )

    def test_manifest_dataset_digest_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_reference, _ = self._repository_fixture(root)
            manifest_reference, data_path = self._additional_dataset_manifest(
                root,
                protocol_reference,
            )
            data_path.write_text(
                data_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                execute_instrument_channel_calibration_validation_dataset_manifest(
                    repository_root=root,
                    protocol_reference=protocol_reference,
                    dataset_manifest_reference=manifest_reference,
                    execution_reference="examples/validation/combined.json",
                    package_version="test",
                    package_commit="1" * 40,
                    executed_at_utc="2026-08-01T00:00:00Z",
                )


if __name__ == "__main__":
    unittest.main()
