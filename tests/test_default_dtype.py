"""Tests for default floating dtype handling."""

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from pgmuvi.dtypes import DEFAULT_DTYPE
from pgmuvi.lightcurve import Lightcurve


class TestDefaultDtype(unittest.TestCase):
    def test_constructor_coerces_array_inputs_to_default_dtype(self):
        lc = Lightcurve([0.0, 1.0, 2.0], [1.0, 2.0, 1.0], yerr=[0.1, 0.1, 0.1])

        self.assertEqual(DEFAULT_DTYPE, torch.float64)
        self.assertEqual(lc.xdata.dtype, DEFAULT_DTYPE)
        self.assertEqual(lc.ydata.dtype, DEFAULT_DTYPE)
        self.assertEqual(lc.yerr.dtype, DEFAULT_DTYPE)
        self.assertEqual(lc._xdata_transformed.dtype, DEFAULT_DTYPE)
        self.assertEqual(lc._ydata_transformed.dtype, DEFAULT_DTYPE)
        self.assertEqual(lc._yerr_transformed.dtype, DEFAULT_DTYPE)

    def test_from_csv_uses_default_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lc.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time", "flux", "err"])
                writer.writerow([0.0, 1.0, 0.1])
                writer.writerow([1.0, 2.0, 0.1])
                writer.writerow([2.0, 1.0, 0.1])

            lc = Lightcurve.from_csv(path, timecol="time", fluxcol="flux", errcol="err")

        self.assertEqual(lc.xdata.dtype, DEFAULT_DTYPE)
        self.assertEqual(lc.ydata.dtype, DEFAULT_DTYPE)
        self.assertEqual(lc.yerr.dtype, DEFAULT_DTYPE)

    def test_set_likelihood_and_model_match_data_dtype(self):
        x = torch.linspace(0.0, 4.0, 8, dtype=torch.float64)
        y = torch.sin(x)
        yerr = torch.full_like(y, 0.1)
        lc = Lightcurve(x, y, yerr=yerr)

        lc.set_likelihood()
        lc.set_model("1D", num_mixtures=1)

        self.assertEqual(lc.likelihood.noise.dtype, DEFAULT_DTYPE)
        for parameter in lc.model.parameters():
            self.assertEqual(parameter.dtype, DEFAULT_DTYPE)

    def test_explicit_float32_tensor_input_is_preserved(self):
        x = torch.linspace(0.0, 4.0, 8, dtype=torch.float32)
        y = torch.sin(x)
        lc = Lightcurve(x, y)

        self.assertEqual(lc.xdata.dtype, torch.float32)
        self.assertEqual(lc.ydata.dtype, torch.float32)
        lc.set_model("1D", num_mixtures=1)
        for parameter in lc.model.parameters():
            self.assertEqual(parameter.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
