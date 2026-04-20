import unittest
import numpy as np

from pgmuvi.preprocess.variability import (
    compute_fvar,
    compute_stetson_k,
    is_variable,
    weighted_chi2_test,
)


class TestWeightedChi2(unittest.TestCase):
    def test_constant_data(self):
        """Chi-square on truly constant data should give p-value > 0.05."""
        np.random.seed(42)
        y = np.ones(100) + np.random.normal(0, 0.01, 100)
        yerr = np.full(100, 0.01)

        chi2, dof, ybar, pval = weighted_chi2_test(y, yerr)

        self.assertEqual(dof, 99)
        self.assertGreater(pval, 0.05)

    def test_variable_data(self):
        """Chi-square on sinusoidal variable should give p-value < 0.01."""
        np.random.seed(42)
        t = np.linspace(0, 100, 200)
        y = 1.0 + 0.3 * np.sin(2 * np.pi * t / 10) + np.random.normal(0, 0.02, 200)
        yerr = np.full(200, 0.02)

        chi2, dof, ybar, pval = weighted_chi2_test(y, yerr)

        self.assertLess(pval, 0.01)

    def test_return_types(self):
        """Return types should match the documented signature."""
        y = np.array([1.0, 1.1, 0.9, 1.05])
        yerr = np.full(4, 0.1)
        chi2, dof, ybar, pval = weighted_chi2_test(y, yerr)

        self.assertIsInstance(chi2, float)
        self.assertIsInstance(dof, int)
        self.assertIsInstance(ybar, float)
        self.assertIsInstance(pval, float)
        self.assertGreaterEqual(pval, 0.0)
        self.assertLessEqual(pval, 1.0)

    def test_input_validation_too_few_points(self):
        """weighted_chi2_test should raise ValueError for N < 2."""
        y = np.array([1.0])
        yerr = np.array([0.1])
        with self.assertRaises(ValueError):
            weighted_chi2_test(y, yerr)

    def test_input_validation_non_positive_yerr(self):
        """weighted_chi2_test should raise ValueError for yerr <= 0."""
        y = np.array([1.0, 1.1, 0.9])
        yerr = np.array([0.1, 0.0, 0.1])
        with self.assertRaises(ValueError):
            weighted_chi2_test(y, yerr)

    def test_input_validation_nan(self):
        """weighted_chi2_test should raise ValueError for NaN values."""
        y = np.array([1.0, float("nan"), 0.9])
        yerr = np.full(3, 0.1)
        with self.assertRaises(ValueError):
            weighted_chi2_test(y, yerr)

    def test_input_validation_shape_mismatch(self):
        """weighted_chi2_test should raise ValueError for shape mismatch."""
        y = np.array([1.0, 1.1, 0.9])
        yerr = np.full(4, 0.1)
        with self.assertRaises(ValueError):
            weighted_chi2_test(y, yerr)

    def test_accepts_torch_tensor(self):
        """weighted_chi2_test should accept torch tensors."""
        import torch
        y = torch.tensor([1.0, 1.1, 0.9, 1.05])
        yerr = torch.full((4,), 0.1)
        chi2, dof, ybar, pval = weighted_chi2_test(y, yerr)
        self.assertIsInstance(chi2, float)


class TestComputeFvar(unittest.TestCase):
    def test_no_intrinsic_variability(self):
        """F_var should be near 0 for noise-only data."""
        np.random.seed(42)
        y = np.ones(100) + np.random.normal(0, 0.01, 100)
        yerr = np.full(100, 0.01)
        fvar = compute_fvar(y, yerr)
        self.assertLess(fvar, 0.05)

    def test_strong_variability(self):
        """F_var should be > 0.2 for a large sinusoidal signal."""
        t = np.linspace(0, 100, 200)
        y = 1.0 + 0.3 * np.sin(2 * np.pi * t / 10)
        yerr = np.full(200, 0.02)
        fvar = compute_fvar(y, yerr)
        self.assertGreater(fvar, 0.2)

    def test_returns_float(self):
        """Return type should be float."""
        y = np.array([1.0, 1.1, 0.9, 1.05])
        yerr = np.full(4, 0.1)
        fvar = compute_fvar(y, yerr)
        self.assertIsInstance(fvar, float)
        self.assertGreaterEqual(fvar, 0.0)


class TestComputeStetsonK(unittest.TestCase):
    def test_gaussian_noise(self):
        """Stetson K for Gaussian noise should be near 0.798."""
        np.random.seed(42)
        y = np.random.normal(1.0, 0.1, 1000)
        yerr = np.full(1000, 0.1)
        K = compute_stetson_k(y, yerr)
        self.assertGreater(K, 0.75)
        self.assertLess(K, 0.85)

    def test_outlier_rich_data(self):
        """Square-wave-like data (many same-magnitude deviations) gives K > Gaussian."""
        np.random.seed(42)
        y_gauss = np.random.normal(1.0, 0.1, 1000)
        yerr = np.full(1000, 0.1)
        K_gauss = compute_stetson_k(y_gauss, yerr)

        # Square-wave: alternates between two values -> K approaches 1
        t = np.linspace(0, 100, 1000)
        y_sq = 1.0 + 0.3 * np.sign(np.sin(2 * np.pi * t / 10))
        K_sq = compute_stetson_k(y_sq, yerr)
        self.assertGreater(K_sq, K_gauss)

    def test_returns_float(self):
        """Return type should be float."""
        y = np.array([1.0, 1.1, 0.9, 1.05])
        yerr = np.full(4, 0.1)
        K = compute_stetson_k(y, yerr)
        self.assertIsInstance(K, float)

    def test_pathological_inputs_return_nan(self):
        """Pathological inputs should return NaN rather than raising."""
        y = np.array([1.0, 1.1, 0.9])
        yerr = np.array([np.nan, 0.0, -1.0])
        K = compute_stetson_k(y, yerr)
        self.assertTrue(np.isnan(K))


class TestIsVariable(unittest.TestCase):
    def test_non_variable(self):
        """Non-variable data should be classified as NOT VARIABLE."""
        np.random.seed(42)
        y = np.ones(100) + np.random.normal(0, 0.01, 100)
        yerr = np.full(100, 0.01)
        is_var, diag = is_variable(y, yerr)
        self.assertFalse(is_var)
        self.assertIn("NOT", diag["decision"])

    def test_variable(self):
        """Clearly variable data should be classified as VARIABLE."""
        np.random.seed(42)
        # Use square-wave signal which gives K close to 1 (> 0.95 threshold)
        t = np.linspace(0, 100, 200)
        y = 1.0 + 0.5 * np.sign(np.sin(2 * np.pi * t / 10)) + np.random.normal(0, 0.02, 200)
        yerr = np.full(200, 0.02)
        is_var, diag = is_variable(y, yerr)
        self.assertTrue(is_var)
        self.assertTrue(diag["decision"].startswith("VARIABLE"))

    def test_sinusoidal_variable_not_vetoed_by_stetson(self):
        """Sinusoidal variability should pass even if stetson_test is False."""
        np.random.seed(123)
        t = np.linspace(0, 100, 300)
        y = 1.0 + 0.3 * np.sin(2 * np.pi * t / 10) + np.random.normal(0, 0.02, 300)
        yerr = np.full(300, 0.02)

        # Set the threshold just above the computed value so this test checks
        # the gating logic rather than a hard-coded Stetson-K scale.
        stetson_k = compute_stetson_k(y, yerr)
        is_var, diag = is_variable(y, yerr, stetson_k_min=stetson_k + 1e-6)

        self.assertTrue(diag["tests_passed"]["chi2_test"])
        self.assertTrue(diag["tests_passed"]["fvar_test"])
        self.assertFalse(diag["tests_passed"]["stetson_test"])
        self.assertTrue(is_var)
        self.assertIn("VARIABLE", diag["decision"])
        self.assertIn("DIAGNOSTIC", diag["decision"])
        self.assertIn("stetson_k", diag)
        self.assertIn("stetson_test", diag["tests_passed"])

    def test_insufficient_points(self):
        """Data with fewer than min_points should fail variability check."""
        y = np.array([1.0, 1.1, 0.9])
        yerr = np.full(3, 0.1)
        is_var, diag = is_variable(y, yerr, min_points=6)
        self.assertFalse(is_var)
        self.assertFalse(diag["tests_passed"]["min_points"])

    def test_diagnostics_structure(self):
        """Diagnostics dict should contain all required keys."""
        np.random.seed(42)
        y = np.random.normal(1.0, 0.1, 50)
        yerr = np.full(50, 0.1)
        _is_var, diag = is_variable(y, yerr)

        required_keys = ["n_points", "chi2", "dof", "p_value", "fvar",
                         "stetson_k", "decision", "tests_passed"]
        for key in required_keys:
            self.assertIn(key, diag)

        test_keys = ["chi2_test", "fvar_test", "stetson_test", "min_points"]
        for key in test_keys:
            self.assertIn(key, diag["tests_passed"])

    def test_verbose_output(self):
        """Verbose mode should print expected diagnostic information."""
        import io
        import sys

        np.random.seed(42)
        y = np.random.normal(1.0, 0.1, 50)
        yerr = np.full(50, 0.1)

        captured = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured
        try:
            is_variable(y, yerr, verbose=True)
        finally:
            sys.stdout = original_stdout

        output = captured.getvalue()
        self.assertIn("Variability assessment", output)
        self.assertIn("Decision", output)


class TestLightcurveVariability(unittest.TestCase):
    """Tests for Lightcurve.check_variability and related methods."""

    def setUp(self):
        import torch
        from pgmuvi.lightcurve import Lightcurve

        np.random.seed(42)
        t = np.linspace(0, 100, 50)
        y = np.ones(50) + np.random.normal(0, 0.01, 50)
        yerr = np.full(50, 0.01)

        self.lc = Lightcurve(
            torch.as_tensor(t, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            yerr=torch.as_tensor(yerr, dtype=torch.float32),
        )

        # 2D multiband lightcurve (two bands)
        n = 50
        t2 = np.linspace(0, 100, n)
        y2 = np.ones(n) + np.random.normal(0, 0.01, n)
        wl1 = np.ones(n) * 1.0
        wl2 = np.ones(n) * 2.0
        t_all = np.concatenate([t2, t2])
        y_all = np.concatenate([y2, y2])
        wl_all = np.concatenate([wl1, wl2])
        yerr_all = np.full(2 * n, 0.01)

        xdata_2d = torch.stack([
            torch.as_tensor(t_all, dtype=torch.float32),
            torch.as_tensor(wl_all, dtype=torch.float32),
        ], dim=1)
        self.lc2d = Lightcurve(
            xdata_2d,
            torch.as_tensor(y_all, dtype=torch.float32),
            yerr=torch.as_tensor(yerr_all, dtype=torch.float32),
        )

    def test_check_variability_1d(self):
        """check_variability should return a diagnostics dict for 1D data."""
        diag = self.lc.check_variability()
        self.assertIn("decision", diag)
        self.assertIn("tests_passed", diag)

    def test_check_variability_raises_for_2d(self):
        """check_variability should raise ValueError for multiband data."""
        from pgmuvi.lightcurve import Lightcurve
        with self.assertRaises(ValueError):
            self.lc2d.check_variability()

    def test_check_variability_per_band_raises_for_1d(self):
        """check_variability_per_band should raise ValueError for 1D data."""
        with self.assertRaises(ValueError):
            self.lc.check_variability_per_band()

    def test_check_variability_per_band_summary(self):
        """check_variability_per_band should return a summary dict."""
        results = self.lc2d.check_variability_per_band()
        self.assertIn("summary", results)
        summary = results["summary"]
        self.assertEqual(summary["n_bands"], 2)
        self.assertIn("n_variable", summary)
        self.assertIn("variable_wavelengths", summary)


if __name__ == "__main__":
    unittest.main()
