import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "run_wavelength_advisory_batch.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("run_wavelength_advisory_batch", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestRunWavelengthAdvisoryBatchExample(unittest.TestCase):
    def setUp(self):
        self.module = _load_example_module()

    def test_source_token_uses_path_stem_by_default(self):
        spec = self.module._source_spec_from_token("data/10131+3049.csv")
        self.assertEqual(spec["source_id"], "10131+3049")
        self.assertEqual(spec["csv_path"], "data/10131+3049.csv")

    def test_source_token_accepts_explicit_source_id(self):
        spec = self.module._source_spec_from_token("target_a=data/source.csv")
        self.assertEqual(spec, {"source_id": "target_a", "csv_path": "data/source.csv"})

    def test_source_list_accepts_headered_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sources.csv"
            path.write_text("source_id,csv_path\nA,a.csv\nB,b.csv\n", encoding="utf-8")
            specs = self.module._read_source_list(path)
        self.assertEqual(
            specs,
            [
                {"source_id": "A", "csv_path": "a.csv"},
                {"source_id": "B", "csv_path": "b.csv"},
            ],
        )

    def test_source_list_accepts_text_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sources.txt"
            path.write_text(
                "# comment\nplain.csv\ncustom=custom.csv\ncomma_id,comma.csv\n",
                encoding="utf-8",
            )
            specs = self.module._read_source_list(path)
        self.assertEqual(
            specs,
            [
                {"source_id": "plain", "csv_path": "plain.csv"},
                {"source_id": "custom", "csv_path": "custom.csv"},
                {"source_id": "comma_id", "csv_path": "comma.csv"},
            ],
        )

    def test_main_delegates_to_lightcurve_batch_helper(self):
        fake_report = {
            "kind": "period_independent_wavelength_advisory_workflow_batch",
            "advisory_only": True,
            "runs_fits": True,
            "applies_to_fit": True,
            "model_kernel_config_state_isolated": True,
            "mutates_input_lightcurve": False,
            "automatic_model_selection_applied": False,
            "selected_model": None,
            "n_sources": 1,
            "n_succeeded": 1,
            "n_failed": 0,
            "batch_json_path": "out/batch_summary.json",
            "batch_csv_path": "out/batch_summary.csv",
            "batch_model_kernel_config_csv_path": "out/batch_model_kernel_configs.csv",
            "batch_model_kernel_config_summary_csv_path": "out/batch_model_kernel_config_summary.csv",
            "batch_markdown_report_path": "out/batch_report.md",
            "model_kernel_config_summary": [
                {
                    "model": "2DDustMean",
                    "fit_strategy": "consensus",
                    "time_kernel_type": "quasi_periodic",
                    "n_sources_evaluated": 1,
                    "n_successful_sources": 1,
                    "n_top_ranked_sources": 1,
                    "median_fit_quality_score": 42.0,
                }
            ],
            "source_results": [
                {
                    "source_id": "one",
                    "status": "passed",
                    "top_ranked_model": "2DDustMean",
                    "top_ranked_fit_quality_score": 12.0,
                    "score_kind": "training_residual_fit_quality",
                    "n_model_kernel_configs": 3,
                    "n_successful_model_kernel_configs": 3,
                    "n_failed_model_kernel_configs": 0,
                    "exception_type": None,
                    "exception_message": None,
                }
            ],
        }

        with patch.object(
            self.module,
            "_run_batch",
            return_value=fake_report,
        ) as mocked:
            rc = self.module.main(
                [
                    "one=one.csv",
                    "--output-dir", "out",
                    "--batch-prefix", "batch",
                    "--training-iter", "7",
                    "--miniter", "3",
                    "--max-samples", "111",
                    "--max-samples-per-band", "22",
                    "--model-kernel-config-limit", "2",
                    "--no-include-2d-baseline",
                    "--no-plots",
                    "--no-export",
                    "--verbose",
                ]
            )

        self.assertEqual(rc, 0)
        mocked.assert_called_once()
        args, kwargs = mocked.call_args
        self.assertEqual(args[0], [{"source_id": "one", "csv_path": "one.csv"}])
        self.assertEqual(
            kwargs["from_csv_kwargs"],
            {
                "check_sampling": True,
                "max_samples": 111,
                "max_samples_per_band": 22,
                "verbose": True,
            },
        )
        self.assertEqual(
            kwargs["positive_data_filter_kwargs"],
            {"require_positive_flux": True, "require_positive_flux_error": True},
        )
        self.assertEqual(kwargs["output_dir"], "out")
        self.assertFalse(kwargs["export"])
        self.assertEqual(kwargs["batch_prefix"], "batch")
        self.assertEqual(kwargs["export_kwargs"], {"close_figures": True})
        self.assertEqual(kwargs["workflow_kwargs"]["model_kernel_config_limit"], 2)
        self.assertFalse(kwargs["workflow_kwargs"]["include_2d_baseline"])
        self.assertFalse(kwargs["workflow_kwargs"]["make_plots"])
        self.assertTrue(kwargs["workflow_kwargs"]["make_text_report"])
        self.assertEqual(
            kwargs["workflow_kwargs"]["base_fit_kwargs"],
            {"training_iter": 7, "miniter": 3, "verbose": True},
        )

    def test_main_can_disable_default_positive_filters(self):
        fake_report = {
            "kind": "period_independent_wavelength_advisory_workflow_batch",
            "n_failed": 0,
            "source_results": [],
        }
        with patch.object(self.module, "_run_batch", return_value=fake_report) as mocked:
            rc = self.module.main(
                [
                    "one.csv",
                    "--no-export",
                    "--allow-nonpositive-flux",
                    "--allow-nonpositive-flux-error",
                ]
            )

        self.assertEqual(rc, 0)
        _, kwargs = mocked.call_args
        self.assertEqual(
            kwargs["positive_data_filter_kwargs"],
            {"require_positive_flux": False, "require_positive_flux_error": False},
        )

    def test_main_returns_failure_status_when_source_rows_fail(self):
        fake_report = {
            "kind": "period_independent_wavelength_advisory_workflow_batch",
            "n_failed": 1,
            "source_results": [],
        }
        with patch.object(
            self.module,
            "_run_batch",
            return_value=fake_report,
        ):
            rc = self.module.main(["bad.csv", "--no-export"])
        self.assertEqual(rc, 1)

    def test_allow_source_failures_returns_success_status(self):
        fake_report = {
            "kind": "period_independent_wavelength_advisory_workflow_batch",
            "n_failed": 1,
            "source_results": [],
        }
        with patch.object(
            self.module,
            "_run_batch",
            return_value=fake_report,
        ):
            rc = self.module.main(["bad.csv", "--no-export", "--allow-source-failures"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
