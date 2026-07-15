"""Tests for the runnable light-curve validation example."""
import importlib.util
import tempfile
import unittest
from pathlib import Path


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "validate_lightcurve_input.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location(
        "validate_lightcurve_input", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestValidateLightcurveInputExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_example_module()

    def _write_csv(self, text):
        tempdir = tempfile.TemporaryDirectory()
        path = Path(tempdir.name) / "source.csv"
        path.write_text(text, encoding="utf-8")
        return tempdir, path

    def test_single_band_summary_and_positive_error_filter(self):
        tempdir, path = self._write_csv(
            "time,flux,flux_error,band\n"
            "0,1.0,0.1,V\n"
            "1,0.8,0.0,V\n"
            "2,-0.2,0.2,V\n"
            "3,1.1,0.1,V\n"
        )
        with tempdir:
            lc, summary = self.module.load_and_validate(
                path,
                drop_nonpositive_errors=True,
                max_samples=None,
            )

        self.assertEqual(lc.ndim, 1)
        self.assertEqual(summary["n_rows_before_positive_filter"], 4)
        self.assertEqual(summary["n_rows_after_positive_filter"], 3)
        self.assertEqual(summary["n_dropped_nonpositive_error"], 1)
        self.assertEqual(summary["n_dropped_nonpositive_flux"], 0)
        self.assertEqual(summary["n_rows_final"], 3)
        self.assertTrue(summary["has_band_labels"])

    def test_multiband_filter_preserves_row_aligned_labels(self):
        tempdir, path = self._write_csv(
            "mjd,wavelength,band,flux,flux_error\n"
            "0,0.55,V,1.0,0.1\n"
            "1,0.80,I,-0.5,0.1\n"
            "2,0.55,V,0.9,0.1\n"
            "3,0.80,I,1.2,0.1\n"
        )
        with tempdir:
            lc, summary = self.module.load_and_validate(
                path,
                drop_nonpositive_flux=True,
                max_samples=None,
            )

        self.assertEqual(lc.ndim, 2)
        self.assertEqual(summary["n_wavelengths"], 2)
        self.assertEqual(summary["n_dropped_nonpositive_flux"], 1)
        self.assertEqual(summary["n_rows_final"], 3)
        self.assertEqual(lc.band.tolist(), ["V", "V", "I"])

    def test_positive_error_filter_requires_uncertainties(self):
        tempdir, path = self._write_csv(
            "time,flux\n0,1.0\n1,0.9\n"
        )
        with tempdir:
            with self.assertRaisesRegex(ValueError, "requires an uncertainty"):
                self.module.load_and_validate(
                    path,
                    drop_nonpositive_errors=True,
                    max_samples=None,
                )

    def test_main_prints_json_summary(self):
        tempdir, path = self._write_csv(
            "time,flux,flux_error\n0,1.0,0.1\n1,0.9,0.1\n"
        )
        with tempdir:
            rc = self.module.main([str(path), "--max-samples", "2"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
