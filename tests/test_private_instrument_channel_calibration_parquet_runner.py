from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pgmuvi.instrument_channel_calibration_multisource_validation import (
    InstrumentChannelCalibrationMultiSourceValidationProtocol,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "maintainer_tools"
    / "run_private_instrument_channel_multisource_validation.py"
)
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


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "pgmuvi_private_parquet_runner",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the private Parquet runner.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPrivateInstrumentChannelCalibrationParquetRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()
        cls.protocol = (
            InstrumentChannelCalibrationMultiSourceValidationProtocol.from_dict(
                json.loads(
                    MULTISOURCE_PROTOCOL_PATH.read_text(encoding="utf-8")
                )
            )
        )

    def columns(
        self,
        *,
        n_sources: int = 5,
        n_pairs: int = 100,
    ) -> dict[str, list[object]]:
        result = {
            name: []
            for name in self.runner.REQUIRED_COLUMNS
        }
        for source_index in range(n_sources):
            time = np.arange(n_pairs, dtype=float)
            target = (
                2.0
                + 0.5 * np.sin(2.0 * np.pi * time / 20.0)
                + 0.1 * np.cos(2.0 * np.pi * time / 7.0)
            )
            reference = 0.3 + 1.2 * target
            for index in range(n_pairs):
                for band, row_time, flux in (
                    (
                        self.protocol.reference_channel,
                        time[index],
                        reference[index],
                    ),
                    (
                        self.protocol.channel,
                        time[index] + 0.01,
                        target[index],
                    ),
                ):
                    result["object_id"].append(f"object-{source_index}")
                    result["time"].append(float(row_time))
                    result["flux"].append(float(flux))
                    result["flux_error"].append(0.01)
                    result["wavelength"].append(
                        self.protocol.physical_wavelength
                    )
                    result["band"].append(band)
            result["object_id"].append(f"object-{source_index}")
            result["time"].append(0.0)
            result["flux"].append(1.0)
            result["flux_error"].append(0.1)
            result["wavelength"].append(1.25)
            result["band"].append("OTHER/J")
        return result

    def test_required_columns_match_private_catalogue_contract(self) -> None:
        self.assertEqual(
            self.runner.REQUIRED_COLUMNS,
            (
                "object_id",
                "time",
                "flux",
                "flux_error",
                "wavelength",
                "band",
            ),
        )

    def test_columns_group_by_object_id_and_exact_channels(self) -> None:
        columns = self.columns()
        sources, ingestion = self.runner.build_source_data_from_columns(
            columns,
            protocol=self.protocol,
        )

        self.assertEqual(len(sources), 5)
        self.assertEqual(
            tuple(source.astrophysical_source_id for source in sources),
            tuple(f"object-{index}" for index in range(5)),
        )
        self.assertEqual(len(sources[0].reference.time), 100)
        self.assertEqual(len(sources[0].channel.time), 100)
        self.assertEqual(
            ingestion["ignored_non_candidate_channel_row_count"],
            5,
        )

    def test_sources_without_candidate_pair_remain_auditable(self) -> None:
        columns = self.columns()
        columns["object_id"].append("unmatched-object")
        columns["time"].append(1.0)
        columns["flux"].append(2.0)
        columns["flux_error"].append(0.1)
        columns["wavelength"].append(2.2)
        columns["band"].append("OTHER/K")

        sources, _ = self.runner.build_source_data_from_columns(
            columns,
            protocol=self.protocol,
        )
        by_id = {
            source.astrophysical_source_id: source
            for source in sources
        }
        self.assertIn("unmatched-object", by_id)
        self.assertEqual(by_id["unmatched-object"].reference.time, ())
        self.assertEqual(by_id["unmatched-object"].channel.time, ())

    def test_private_runner_writes_detailed_and_redacted_outputs(self) -> None:
        columns = self.columns()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            parquet_path = workspace / "private.parquet"
            parquet_path.write_bytes(b"private-parquet-placeholder")
            private_output = workspace / "private.json"
            summary_output = workspace / "summary.json"

            result = self.runner.execute_private_parquet_validation(
                parquet_path=parquet_path,
                repository_root=ROOT,
                multisource_protocol_path=MULTISOURCE_PROTOCOL_PATH,
                candidate_protocol_path=CANDIDATE_PROTOCOL_PATH,
                selection_seed=20260726,
                n_sources=5,
                private_report_output=private_output,
                redacted_summary_output=summary_output,
                package_version="test",
                package_commit="b" * 40,
                executed_at_utc="2026-07-26T00:00:00Z",
                parquet_loader=lambda _: columns,
            )

            self.assertEqual(result.disposition.value, "passed")
            private = json.loads(private_output.read_text(encoding="utf-8"))
            summary = json.loads(summary_output.read_text(encoding="utf-8"))

        self.assertIn("object-0", json.dumps(private))
        self.assertNotIn("object-0", json.dumps(summary))
        self.assertEqual(summary["disposition"], "passed")
        self.assertFalse(summary["catalogue_population_performed"])
        self.assertIn("source_independence_assertion", private)

    def test_missing_column_is_rejected_before_grouping(self) -> None:
        columns = self.columns()
        del columns["flux_error"]
        with self.assertRaisesRegex(ValueError, "flux_error"):
            self.runner.build_source_data_from_columns(
                columns,
                protocol=self.protocol,
            )

    def test_runner_is_outside_installed_package(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            'packages = ["pgmuvi", "pgmuvi.preprocess"]',
            pyproject,
        )
        self.assertNotIn('"pyarrow"', pyproject)
        self.assertTrue(RUNNER_PATH.is_file())
        self.assertFalse(
            (ROOT / "pgmuvi" / RUNNER_PATH.name).exists()
        )

    def test_documentation_contains_runnable_private_command(self) -> None:
        execution = (
            ROOT
            / "docs"
            / "source"
            / "pgmuvi.instrument_channel_calibration_validation_execution.rst"
        ).read_text(encoding="utf-8")
        for token in (
            "object_id",
            "flux_error",
            "PRIVATE_CATALOGUE.parquet",
            "--selection-seed 20260726",
            "--n-sources 5",
            "--n-sources all",
            "validation_outputs/",
        ):
            with self.subTest(token=token):
                self.assertIn(token, execution)


if __name__ == "__main__":
    unittest.main()
