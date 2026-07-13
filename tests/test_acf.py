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


def _make_2d_lc(n=20, seed=42):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 50, n))
    t2d = np.vstack([
        np.column_stack([t, np.ones(n) * 550.0]),
        np.column_stack([t, np.ones(n) * 700.0]),
    ])
    y = rng.normal(0, 1, 2 * n)
    yerr = np.full(2 * n, 0.1)
    band = np.array(["V"] * n + ["R"] * n)
    return Lightcurve(
        torch.as_tensor(t2d, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
        yerr=torch.as_tensor(yerr, dtype=torch.float32),
        band=band,
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

    # --- basic return type ---

    def test_returns_acf_result(self):
        result = self.lc.acf(method="data")
        self.assertIsInstance(result, ACFResult)
        self.assertEqual(result.method, "data")

    def test_lag_is_tensor(self):
        result = self.lc.acf(method="data")
        self.assertIsInstance(result.lag, torch.Tensor)
        self.assertIsInstance(result.acf, torch.Tensor)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            self.lc.acf(method="unknown")

    # --- zero-lag prepend behavior ---

    def test_length_is_n_lags_plus_one(self):
        result = self.lc.acf(method="data", n_lags=20)
        self.assertEqual(result.lag.shape[0], 21)
        self.assertEqual(result.acf.shape[0], 21)
        self.assertEqual(result.counts.shape[0], 21)

    def test_lag_zero_first(self):
        result = self.lc.acf(method="data", n_lags=20)
        self.assertAlmostEqual(float(result.lag[0]), 0.0)

    def test_normalized_acf_zero_is_one(self):
        result = self.lc.acf(method="data", normalize=True)
        self.assertAlmostEqual(float(result.acf[0]), 1.0, places=5)

    def test_unnormalized_acf_zero_is_variance(self):
        result = self.lc.acf(method="data", normalize=False)
        y = self.lc.ydata.double()
        mean = float(y.mean())
        y_centered = y - mean
        expected_variance = float((y_centered**2).mean())
        self.assertAlmostEqual(
            float(result.acf[0]), expected_variance, places=4
        )

    def test_counts_zero_is_n(self):
        result = self.lc.acf(method="data")
        n = self.lc.xdata.shape[0]
        self.assertEqual(int(result.counts[0]), n)

    # --- normalize flag ---

    def test_normalize_flag_stored(self):
        result = self.lc.acf(method="data", normalize=True)
        self.assertTrue(result.normalized)
        result_no_norm = self.lc.acf(method="data", normalize=False)
        self.assertFalse(result_no_norm.normalized)

    # --- lag_edges behavior ---

    def test_lag_edges_length(self):
        edges = np.linspace(0, 50, 11)
        result = self.lc.acf(method="data", lag_edges=edges)
        # 10 bins + 1 zero-lag prepend = 11
        self.assertEqual(result.lag.shape[0], 11)

    def test_lag_edges_works_when_max_lag_none(self):
        # max_lag=None should not crash when lag_edges is provided
        edges = np.linspace(0, 40, 6)
        result = self.lc.acf(method="data", lag_edges=edges, max_lag=None)
        self.assertIsInstance(result, ACFResult)

    def test_lag_edges_non_increasing_raises(self):
        edges = np.array([0.0, 5.0, 3.0, 10.0])
        with self.assertRaises(ValueError):
            self.lc.acf(method="data", lag_edges=edges)

    def test_lag_edges_fewer_than_two_raises(self):
        with self.assertRaises(ValueError):
            self.lc.acf(method="data", lag_edges=np.array([5.0]))

    def test_lag_edges_non_finite_raises(self):
        edges = np.array([0.0, np.inf, 10.0])
        with self.assertRaises(ValueError):
            self.lc.acf(method="data", lag_edges=edges)

    def test_lag_edges_non_1d_raises(self):
        edges = np.array([[0.0, 5.0], [10.0, 20.0]])
        with self.assertRaises(ValueError):
            self.lc.acf(method="data", lag_edges=edges)

    # --- n_lags validation ---

    def test_invalid_n_lags_zero_raises(self):
        with self.assertRaises(ValueError):
            self.lc.acf(method="data", n_lags=0)

    def test_invalid_n_lags_negative_raises(self):
        with self.assertRaises(ValueError):
            self.lc.acf(method="data", n_lags=-1)

    def test_invalid_n_lags_float_raises(self):
        with self.assertRaises(ValueError):
            self.lc.acf(method="data", n_lags=20.5)

    # --- max_lag ---

    def test_max_lag(self):
        result = self.lc.acf(method="data", max_lag=30, n_lags=15)
        self.assertGreaterEqual(float(result.lag.max()), 28.0)
        self.assertLessEqual(float(result.lag.max()), 30.0)

    # --- band stored ---

    def test_band_attribute_stored(self):
        result = self.lc.acf(method="data", band="V")
        self.assertEqual(result.band, "V")


class TestACFData2D(unittest.TestCase):
    def setUp(self):
        self.lc2d = _make_2d_lc()

    def test_2d_requires_band_data(self):
        with self.assertRaises(ValueError):
            self.lc2d.acf(method="data")

    def test_2d_with_band_data(self):
        result = self.lc2d.acf(method="data", band="V")
        self.assertIsInstance(result, ACFResult)
        self.assertEqual(result.band, "V")

    def test_2d_gp_with_band_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.lc2d.acf(method="gp", band="V")

    def test_2d_gp_without_band_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.lc2d.acf(method="gp")


class TestACFGPPrecondition(unittest.TestCase):
    def setUp(self):
        self.lc = _make_1d_lc()

    def test_gp_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            self.lc.acf(method="gp")

    def test_gp_after_set_model_only_raises(self):
        self.lc.set_model("1DQuasiPeriodic", period=20.0)
        with self.assertRaises(RuntimeError):
            self.lc.acf(method="gp")

    def test_gp_n_lags_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.lc.acf(method="gp", n_lags=0)


class TestACFGPMocked(unittest.TestCase):
    """Verify GP ACF logic using a mock covariance module.

    This avoids running a full GP fit while still exercising the
    normalization, detach, and tensor-device behaviour of _acf_gp.
    """

    def _make_fitted_lc_with_mock_kernel(self, kernel_fn):
        """Return a 1-D Lightcurve whose MAP flag is set and whose
        covar_module is replaced by a mock that delegates to *kernel_fn*.
        """
        from unittest.mock import MagicMock, patch
        lc = _make_1d_lc()

        # Force the fitted flag to True without actually fitting
        lc._Lightcurve__FITTED_MAP = True

        # Build a minimal mock model with a covar_module
        mock_model = MagicMock()
        mock_model.eval.return_value = None

        def mock_covar(x1, x2):
            # Returns a lazy-tensor-like object whose .diagonal() gives
            # elementwise kernel values K(x1[i], x1[i] + tau[i])
            result = MagicMock()
            result.diagonal.return_value = kernel_fn(x1, x2)
            return result

        mock_model.covar_module.side_effect = mock_covar
        lc.model = mock_model
        return lc

    def test_normalized_acf_zero_is_one(self):
        """When the kernel is RBF-like, acf[0] should equal 1 (normalized)."""
        def rbf(x1, x2):
            # K(x, x') = exp(-||x-x'||^2 / 2)
            diff = (x1 - x2).squeeze(-1)
            return torch.exp(-0.5 * diff**2)

        lc = self._make_fitted_lc_with_mock_kernel(rbf)
        result = lc.acf(method="gp", n_lags=25, normalize=True)
        self.assertEqual(result.lag.shape[0], 25)
        self.assertAlmostEqual(float(result.lag[0]), 0.0)
        self.assertAlmostEqual(float(result.acf[0]), 1.0, places=5)
        self.assertFalse(torch.any(torch.isnan(result.acf)).item())

    def test_unnormalized_acf_zero_equals_variance(self):
        """Without normalization acf[0] should equal K(t_ref, t_ref)."""
        amplitude = 3.0

        def const_kernel(x1, x2):
            # K(x, x') = amplitude * exp(-||x-x'||^2 / 2)
            diff = (x1 - x2).squeeze(-1)
            return amplitude * torch.exp(-0.5 * diff**2)

        lc = self._make_fitted_lc_with_mock_kernel(const_kernel)
        result = lc.acf(method="gp", n_lags=10, normalize=False)
        self.assertAlmostEqual(float(result.acf[0]), amplitude, places=5)

    def test_tensors_are_detached(self):
        """Returned tensors must not require grad."""
        def rbf(x1, x2):
            diff = (x1 - x2).squeeze(-1)
            return torch.exp(-0.5 * diff**2)

        lc = self._make_fitted_lc_with_mock_kernel(rbf)
        result = lc.acf(method="gp", n_lags=10)
        self.assertFalse(result.lag.requires_grad)
        self.assertFalse(result.acf.requires_grad)


class TestACFConstantLightcurve(unittest.TestCase):
    """Tests for zero-variance (constant) input to _acf_data."""

    def setUp(self):
        n = 30
        t = np.linspace(0, 100, n)
        y = np.ones(n) * 5.0  # constant: variance == 0
        yerr = np.full(n, 0.1)
        self.lc_const = Lightcurve(
            torch.as_tensor(t, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            yerr=torch.as_tensor(yerr, dtype=torch.float32),
        )

    def test_normalize_true_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            self.lc_const.acf(method="data", normalize=True)

    def test_normalize_false_succeeds_variance_zero(self):
        result = self.lc_const.acf(method="data", normalize=False)
        self.assertIsInstance(result, ACFResult)
        # Zero-lag acf equals variance, which is 0 for a constant series
        self.assertAlmostEqual(float(result.acf[0]), 0.0, places=6)


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


if __name__ == "__main__":
    unittest.main()
