"""Regression tests for consensus handoff to period-based time kernels.

The single-component consensus path used to require a spectral-mixture time
kernel because it initialized/constrained only mixture_means/mixture_scales.
LPV separable models such as 2DDustMean and 2DPowerLawMean also support
quasi-periodic time kernels, whose consensus target is period_length.
"""

from __future__ import annotations

import unittest

import numpy as np

from pgmuvi.lightcurve import Lightcurve


def _make_lpv_like_lc():
    rng = np.random.default_rng(123)
    period = 5.0
    t = np.linspace(0.0, 30.0, 24)
    x_blocks = []
    y_blocks = []
    yerr_blocks = []
    for band_index, wavelength in enumerate([1.0, 2.0]):
        phase = 2.0 * np.pi * t / period
        amp = 1.0 / wavelength
        y = 10.0 + wavelength + amp * np.sin(phase)
        y = y + rng.normal(0.0, 0.01, size=t.shape)
        x_blocks.append(np.column_stack([t, np.full_like(t, wavelength)]))
        y_blocks.append(y)
        yerr_blocks.append(np.full_like(t, 0.02))
    return Lightcurve(
        np.vstack(x_blocks),
        np.concatenate(y_blocks),
        yerr=np.concatenate(yerr_blocks),
    )


class TestConsensusPeriodKernelHandoff(unittest.TestCase):
    def test_quasi_periodic_lpv_models_resolve_period_length_mode(self):
        for model_name in (
            "2DSeparable",
            "2DWavelengthDependent",
            "2DDustMean",
            "2DPowerLawMean",
        ):
            with self.subTest(model_name=model_name):
                lc = _make_lpv_like_lc()
                lc.set_model(
                    model_name,
                    time_kernel_type="quasi_periodic",
                    wavelength_kernel_type="rbf",
                    period=5.0,
                )
                mode, keys = lc._consensus_resolve_time_kernel_constraint_mode()
                self.assertEqual(mode, "period_length")
                self.assertIn("period_length", keys)
                self.assertTrue(keys["period_length"].endswith("period_length"))

    def test_period_length_guess_uses_inverse_consensus_frequency(self):
        lc = _make_lpv_like_lc()
        lc.set_model(
            "2DDustMean",
            time_kernel_type="quasi_periodic",
            wavelength_kernel_type="rbf",
            period=3.0,
        )
        guess = lc._consensus_build_time_kernel_guess(
            frequencies=np.asarray([0.2]),
            mode="period_length",
        )
        self.assertEqual(len(guess), 1)
        key, value = next(iter(guess.items()))
        self.assertTrue(key.endswith("period_length"))
        self.assertAlmostEqual(float(value.reshape(-1)[0]), 5.0, places=8)

    def test_standard_consensus_manual_frequency_handoff_to_period_kernel(self):
        lc = _make_lpv_like_lc()
        captured = {}

        def fake_inner_fit(**kwargs):
            captured.update(kwargs)
            return {"ok": True}

        # Avoid expensive GP optimization while still exercising the
        # _consensus_standard_fit model-build, constraint, guess, and final
        # validation path.
        lc.fit = fake_inner_fit

        result = lc._consensus_standard_fit(
            model="2DDustMean",
            time_kernel_type="quasi_periodic",
            wavelength_kernel_type="rbf",
            consensus_frequencies=[0.2],
            consensus_frequency_width=[0.01],
            consensus_frequency_k=3.0,
            constrain_consensus=True,
            training_iter=1,
            miniter=0,
        )

        self.assertEqual(result, {"ok": True})
        diag = lc.consensus_diagnostics
        self.assertEqual(
            diag["consensus_time_kernel_constraint_mode"], "period_length"
        )
        self.assertIn("period_length", diag["consensus_constraint_target_key"])
        self.assertIsNone(diag["consensus_scale_constraint_target_key"])
        self.assertIsNone(diag["consensus_scale_constraint_bounds"])

        # Frequency interval is [0.17, 0.23], so period interval is
        # [1 / 0.23, 1 / 0.17].
        expected_period_bounds = [1.0 / 0.23, 1.0 / 0.17]
        self.assertIsNotNone(diag["consensus_period_constraint_bounds"])
        for actual, expected in zip(
            diag["consensus_period_constraint_bounds"], expected_period_bounds
        ):
            self.assertAlmostEqual(float(actual), expected, places=6)

        self.assertIn("guess", captured)
        guess = captured["guess"]
        self.assertEqual(len(guess), 1)
        key, value = next(iter(guess.items()))
        self.assertTrue(key.endswith("period_length"))
        self.assertAlmostEqual(float(value.reshape(-1)[0]), 5.0, places=8)

    def test_spectral_mixture_handoff_remains_available(self):
        lc = _make_lpv_like_lc()
        lc.set_model(
            "2DDustMean",
            time_kernel_type="spectral_mixture",
            wavelength_kernel_type="rbf",
            num_mixtures=1,
        )
        mode, keys = lc._consensus_resolve_time_kernel_constraint_mode()
        self.assertEqual(mode, "spectral_mixture")
        self.assertIn("mixture_means", keys)
        self.assertIn("mixture_scales", keys)


if __name__ == "__main__":
    unittest.main()
