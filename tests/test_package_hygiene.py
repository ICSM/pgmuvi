"""Packaging behaviour: exports, lazy optional imports, shims, bundled data."""

import subprocess
import sys
import unittest
import warnings


class TestPackageSurface(unittest.TestCase):
    def test_lightcurve_exported_from_top_level(self):
        import pgmuvi

        self.assertIs(pgmuvi.Lightcurve, pgmuvi.lightcurve.Lightcurve)

    def test_models_shim_warns_and_reexports(self):
        sys.modules.pop("pgmuvi.models", None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import pgmuvi.models as models
        self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
        from pgmuvi.gps import QuasiPeriodicGPModel

        self.assertIs(models.QuasiPeriodicGPModel, QuasiPeriodicGPModel)

    def test_import_does_not_load_mcmc_dependencies(self):
        # gpytorch itself imports pyro when it is installed, so only the
        # packages nothing else pulls in are checked via sys.modules; the
        # module-level names are checked for all of them.
        code = (
            "import sys, pgmuvi, pgmuvi.lightcurve as lc; "
            "print(sorted(m for m in ('arviz', 'xarray', 'seaborn') "
            "if m in sys.modules), "
            "[n for n in ('pyro', 'az', 'xr', 'NUTS', 'MCMC', 'HMC') "
            "if hasattr(lc, n)])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(out.stdout.strip(), "[] []")


class TestBundledData(unittest.TestCase):
    def test_get_path_returns_existing_csv(self):
        from pgmuvi.datasets import get_path

        p = get_path("AlfOriAAVSO_Vband.csv")
        self.assertTrue(p.is_file())
        with open(p) as f:
            self.assertEqual(f.readline().strip(), "JD,Magnitude")

    def test_get_path_unknown_name(self):
        from pgmuvi.datasets import get_path

        with self.assertRaises(FileNotFoundError):
            get_path("nope.csv")


if __name__ == "__main__":
    unittest.main()
