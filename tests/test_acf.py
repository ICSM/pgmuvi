"""Tests for Lightcurve.acf() and Lightcurve.plot_acf()."""
import unittest
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from pgmuvi.lightcurve import Lightcurve, ACFResult


def _make_1d_lc(n=40, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 100, n))
    y = np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.1, n)
    yerr = np.full(n, 0.1)
    return Lightcurve(
        torch.as_tensor(t, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
        yerr=torch.as_tensor(yerr, dtype=torch.float32),
    )


class TestACFResult(unittest.TestCase):
    def test_acf_result_fields(self):
        lag = torch.zeros(5)
        acf = torch.zeros(5)
        result = ACFResult(lag=lag, acf=acf, method="data")
        self.assertEqual(result.method, "data")
        self.assertIsNone(result.counts)
        self.assertTrue(result.normalized)
        self.assertIsNone(result.band)


class TestACFDataMethod(unittest.TestCase):
    def setUp(self):
        self.lc = _make_1d_lc()

    def test_returns_acf_result(self):
        result = self.lc.acf(method="data")
        self.assertIsInstance(result, ACFResult)
        self.assertEqual(result.method, "data")

    def test_lag_shape(self):
        result = self.lc.acf(method="data", n_lags=20)
        self.assertEqual(result.lag.shape[0], 20)
        self.assertEqual(result.acf.shape[0], 20)

    def test_counts_shape(self):
        result = self.lc.acf(method="data", n_lags=20)
        self.assertIsNotNone(result.counts)
        self.assertEqual(result.counts.shape[0], 20)

    def test_normalize_flag(self):
        result = self.lc.acf(method="data", normalize=True)
        self.assertTrue(result.normalized)
        result_no_norm = self.lc.acf(method="data", normalize=False)
        self.assertFalse(result_no_norm.normalized)

    def test_lag_edges(self):
        edges = np.linspace(0, 50, 11)
        result = self.lc.acf(method="data", lag_edges=edges)
        # 10 bins from 11 edges
        self.assertEqual(result.lag.shape[0], 10)

    def test_max_lag(self):
        result = self.lc.acf(method="data", max_lag=30, n_lags=15)
        self.assertGreaterEqual(float(result.lag.max()), 28.0)
        self.assertLessEqual(float(result.lag.max()), 30.0)

    def test_band_attribute_stored(self):
        result = self.lc.acf(method="data", band="V")
        self.assertEqual(result.band, "V")

    def test_lag_is_tensor(self):
        result = self.lc.acf(method="data")
        self.assertIsInstance(result.lag, torch.Tensor)
        self.assertIsInstance(result.acf, torch.Tensor)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            self.lc.acf(method="unknown")


class TestACFData2D(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 20
        t = np.sort(rng.uniform(0, 50, n))
        t2d_v = np.column_stack([t, np.ones(n) * 550.0])
        t2d_r = np.column_stack([t, np.ones(n) * 700.0])
        t2d = np.vstack([t2d_v, t2d_r])
        y = rng.normal(0, 1, 2 * n)
        yerr = np.full(2 * n, 0.1)
        band = np.array(["V"] * n + ["R"] * n)
        self.lc2d = Lightcurve(
            torch.as_tensor(t2d, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            yerr=torch.as_tensor(yerr, dtype=torch.float32),
            band=band,
        )

    def test_2d_requires_band(self):
        with self.assertRaises(ValueError):
            self.lc2d.acf(method="data")

    def test_2d_with_band(self):
        result = self.lc2d.acf(method="data", band="V")
        self.assertIsInstance(result, ACFResult)
        self.assertEqual(result.band, "V")


class TestPlotACF(unittest.TestCase):
    def setUp(self):
        self.lc = _make_1d_lc()

    def test_plot_acf_returns_axes(self):
        import matplotlib.pyplot as plt
        ax = self.lc.plot_acf(method="data")
        self.assertIsNotNone(ax)
        plt.close("all")

    def test_plot_acf_accepts_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        returned_ax = self.lc.plot_acf(method="data", ax=ax)
        self.assertIs(returned_ax, ax)
        plt.close("all")


class TestACFGPNotFitted(unittest.TestCase):
    def setUp(self):
        self.lc = _make_1d_lc()

    def test_gp_requires_fitted_model(self):
        with self.assertRaises(RuntimeError):
            self.lc.acf(method="gp")


if __name__ == "__main__":
    unittest.main()
