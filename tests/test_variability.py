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
        self.assertEqual(diag["decision"], "VARIABLE")

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
        sys.stdout = captured
        try:
            is_variable(y, yerr, verbose=True)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("Variability assessment", output)
        self.assertIn("Decision", output)


if __name__ == "__main__":
    unittest.main()
