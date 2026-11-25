"""Reproducibility of the Monte Carlo multiband FAP estimators (roadmap 1.3)."""

import importlib.util
import unittest

import numpy as np

from pgmuvi.multiband_ls_significance import MultibandLSWithSignificance

METHODS = ("bootstrap", "phase_scramble")
_HAS_JOBLIB = importlib.util.find_spec("joblib") is not None


def _make_ls():
    gen = np.random.default_rng(1)
    n = 60
    t = np.sort(gen.uniform(0.0, 100.0, n))
    bands = np.repeat(np.array(["a", "b"]), n // 2)
    gen.shuffle(bands)
    y = np.sin(2 * np.pi * t / 17.0) + 0.5 * gen.normal(size=n)
    y[bands == "b"] += 1.0
    dy = np.full(n, 0.5)
    ls = MultibandLSWithSignificance(t, y, bands, dy=dy)
    freq = np.linspace(0.01, 0.5, 200)
    power = ls.power(freq)
    # Levels spanning the null max-power range (about 0.15 to 0.45 for both
    # methods here) so the FAPs are not all 0 or 1, plus the observed peak.
    levels = np.append(np.linspace(0.1, 0.6, 11), power.max())
    return ls, freq, levels


class TestMultibandFapRng(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ls, cls.freq, cls.levels = _make_ls()

    def _fap(self, method, rng, n_samples=20, n_jobs=1):
        return self.ls.false_alarm_probability(
            self.levels,
            method=method,
            n_samples=n_samples,
            freq_grid=self.freq,
            n_jobs=n_jobs,
            rng=rng,
        )

    def test_same_seed_identical(self):
        for method in METHODS:
            with self.subTest(method=method):
                np.testing.assert_array_equal(
                    self._fap(method, 5), self._fap(method, 5)
                )

    def test_generator_matches_int_seed(self):
        for method in METHODS:
            with self.subTest(method=method):
                np.testing.assert_array_equal(
                    self._fap(method, 5),
                    self._fap(method, np.random.default_rng(5)),
                )

    @unittest.skipUnless(_HAS_JOBLIB, "joblib not installed")
    def test_serial_equals_parallel(self):
        for method in METHODS:
            with self.subTest(method=method):
                np.testing.assert_array_equal(
                    self._fap(method, 5, n_jobs=1),
                    self._fap(method, 5, n_jobs=2),
                )

    def test_different_seeds_differ(self):
        # 40 continuous null max-powers coinciding for two seeds is
        # implausible; the FAP arrays must differ for at least one method.
        fap_differs = []
        for method in METHODS:
            with self.subTest(method=method):
                self.assertFalse(
                    np.array_equal(self._null(method, 5), self._null(method, 6))
                )
            fap_differs.append(
                not np.array_equal(
                    self._fap(method, 5, n_samples=40),
                    self._fap(method, 6, n_samples=40),
                )
            )
        self.assertTrue(any(fap_differs))

    def _null(self, method, seed, n=40):
        from pgmuvi import multiband_ls_significance as mls

        worker = {
            "bootstrap": mls._compute_bootstrap_sample_max_power,
            "phase_scramble": mls._compute_phase_scramble_sample_max_power,
        }[method]
        ls = self.ls
        children = mls._spawn_generators(np.random.default_rng(seed), n)
        return np.array(
            [
                worker(ls.t, ls.y, ls.bands, ls.dy, self.freq, ls.ls_method,
                       rng=c)
                for c in children
            ]
        )

    def test_fap_matches_bruteforce_null(self):
        # The FAP equals the exceedance fraction of the spawned null draws.
        for method in METHODS:
            with self.subTest(method=method):
                null = self._null(method, 5)
                expected = np.array([(null >= p).mean() for p in self.levels])
                np.testing.assert_array_equal(
                    self._fap(method, 5, n_samples=40), expected
                )

    def test_rng_none_in_unit_interval(self):
        for method in METHODS:
            with self.subTest(method=method):
                fap = self._fap(method, None)
                self.assertEqual(fap.shape, self.levels.shape)
                self.assertTrue(np.all((fap >= 0) & (fap <= 1)))

    def test_rng_none_is_legacy_global_state(self):
        from pgmuvi import multiband_ls_significance as mls

        workers = {
            "bootstrap": mls._compute_bootstrap_sample_max_power,
            "phase_scramble": mls._compute_phase_scramble_sample_max_power,
        }
        ls = self.ls
        for method, worker in workers.items():
            with self.subTest(method=method):
                np.random.seed(0)
                got = self._fap(method, None, n_samples=10)
                np.random.seed(0)
                null = np.array(
                    [
                        worker(ls.t, ls.y, ls.bands, ls.dy, self.freq,
                               ls.ls_method, rng=None)
                        for _ in range(10)
                    ]
                )
                expected = np.array([(null >= p).mean() for p in self.levels])
                np.testing.assert_array_equal(got, expected)

    def test_random_state_rejected(self):
        for method in METHODS:
            with self.subTest(method=method):
                with self.assertRaises(TypeError):
                    self._fap(method, np.random.RandomState(0))

    def test_reused_generator_is_reproducible(self):
        for method in METHODS:
            with self.subTest(method=method):
                gen = np.random.default_rng(5)
                np.testing.assert_array_equal(
                    self._fap(method, gen), self._fap(method, gen)
                )

    def test_worker_without_rng_uses_global_state(self):
        from pgmuvi import multiband_ls_significance as mls

        ls = self.ls
        args = (ls.t, ls.y, ls.bands, ls.dy, self.freq, ls.ls_method)
        for worker in (
            mls._compute_bootstrap_sample_max_power,
            mls._compute_phase_scramble_sample_max_power,
        ):
            with self.subTest(worker=worker.__name__):
                np.random.seed(3)
                a = worker(*args)
                np.random.seed(3)
                b = worker(*args)
                self.assertEqual(a, b)


class TestFitLSFapRng(unittest.TestCase):
    def test_fit_ls_forwards_fap_rng(self):
        from unittest.mock import patch

        from pgmuvi import multiband_ls_significance as mls
        from pgmuvi.lightcurve import Lightcurve
        from pgmuvi.synthetic import make_chromatic_sinusoid_2d

        syn = make_chromatic_sinusoid_2d(
            n_per_band=30, period=2.0, wavelengths=[0.5, 1.5],
            noise_level=0.1, t_span=20.0, seed=42,
        )
        lc = Lightcurve(syn.xdata, syn.ydata)
        orig = mls.MultibandLSWithSignificance.false_alarm_probability
        seen = []

        def _recording(self_, power_values, **kw):
            seen.append(kw.get("rng"))
            kw["n_samples"] = 10
            return orig(self_, power_values, **kw)

        with patch.object(
            mls.MultibandLSWithSignificance, "false_alarm_probability",
            _recording,
        ):
            out1 = lc.fit_LS(num_peaks=2, fap_rng=7)
            out2 = lc.fit_LS(num_peaks=2, fap_rng=7)
        self.assertGreater(len(seen), 0)
        self.assertTrue(all(r == 7 for r in seen))
        for a, b in zip(out1, out2):
            self.assertTrue(bool((a == b).all()))


if __name__ == "__main__":
    unittest.main()
