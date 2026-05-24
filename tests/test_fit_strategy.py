"""Tests for fit_strategy='consensus' helper pathways."""

import unittest
from unittest.mock import patch

import torch

from pgmuvi.synthetic import make_chromatic_sinusoid_2d, make_simple_sinusoid_1d

_DUMMY_RESULTS = {"loss": [1.0], "delta_loss": [0.0]}


def _make_1d_lc(seed=42):
    return make_simple_sinusoid_1d(
        n_obs=60,
        period=5.0,
        noise_level=0.0,
        irregular=False,
        seed=seed,
    )


def _make_2d_lc(seed=42):
    return make_chromatic_sinusoid_2d(
        n_per_band=25,
        period=5.0,
        wavelengths=[500.0, 700.0],
        amplitude_slope=0.0,
        noise_level=0.0,
        irregular=False,
        seed=seed,
    )


def _fit_without_training(lc, **kwargs):
    with patch("pgmuvi.lightcurve.train", return_value=_DUMMY_RESULTS):
        with patch.object(lc, "_train"):
            with patch.object(lc, "print_parameters"):
                return lc.fit(**kwargs)


class TestConsensusKeyResolution(unittest.TestCase):
    """Tests for _consensus_resolve_time_sm_keys across model families."""

    def test_1d_model_keys(self):
        lc = _make_1d_lc()
        lc.set_model("1D", num_mixtures=2)
        keys = lc._consensus_resolve_time_sm_keys()
        self.assertEqual(keys["mixture_means"], "covar_module.mixture_means")
        self.assertEqual(keys["mixture_scales"], "covar_module.mixture_scales")

    def test_2d_model_keys(self):
        lc = _make_2d_lc()
        lc.set_model("2D", num_mixtures=2)
        keys = lc._consensus_resolve_time_sm_keys()
        self.assertEqual(keys["mixture_means"], "covar_module.mixture_means")
        self.assertEqual(keys["mixture_scales"], "covar_module.mixture_scales")

    def test_ski_model_keys(self):
        lc = _make_1d_lc()
        lc.set_model("1DSKI", num_mixtures=2)
        keys = lc._consensus_resolve_time_sm_keys()
        self.assertEqual(
            keys["mixture_means"],
            "covar_module.base_kernel.mixture_means",
        )
        self.assertEqual(
            keys["mixture_scales"],
            "covar_module.base_kernel.mixture_scales",
        )

    def test_separable_model_raises(self):
        lc = _make_2d_lc()
        lc.set_model("2DSeparable")
        with self.assertRaisesRegex(RuntimeError, "Could not resolve"):
            lc._consensus_resolve_time_sm_keys()


class TestConsensusInitAndGuessHelpers(unittest.TestCase):
    """Tests for consensus initialization tensor and guess helper behavior."""

    def setUp(self):
        self.lc = _make_1d_lc()
        self.lc.set_model("1D", num_mixtures=2)

    def test_build_sm_initialization_shapes_and_scale_broadcast(self):
        init = self.lc._consensus_build_sm_initialization(
            frequencies=[0.1, 0.2],
            scales=0.05,
        )
        self.assertEqual(init["num_mixtures"], 2)
        self.assertEqual(init["mixture_means"].shape, torch.Size([2]))
        self.assertEqual(init["mixture_scales"].shape, torch.Size([2]))
        torch.testing.assert_close(
            init["mixture_scales"],
            torch.tensor([0.05, 0.05], dtype=init["mixture_scales"].dtype),
        )

    def test_build_sm_initialization_invalid_frequency_inputs(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.lc._consensus_build_sm_initialization(frequencies=[])

        with self.assertRaisesRegex(ValueError, "finite"):
            self.lc._consensus_build_sm_initialization(
                frequencies=[0.1, float("nan")]
            )

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self.lc._consensus_build_sm_initialization(frequencies=[0.1, 0.0])

    def test_build_sm_initialization_invalid_scale_inputs(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.lc._consensus_build_sm_initialization(
                frequencies=[0.1, 0.2],
                scales=[],
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            self.lc._consensus_build_sm_initialization(
                frequencies=[0.1, 0.2],
                scales=[0.1, float("nan")],
            )

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self.lc._consensus_build_sm_initialization(
                frequencies=[0.1, 0.2],
                scales=[0.1, 0.0],
            )

        with self.assertRaisesRegex(ValueError, "same number of elements"):
            self.lc._consensus_build_sm_initialization(
                frequencies=[0.1, 0.2],
                scales=[0.1, 0.2, 0.3],
            )

    def test_build_guess_returns_model_keyed_tensors(self):
        guess = self.lc._consensus_build_guess(
            frequencies=[0.1, 0.2],
            scales=[0.01, 0.02],
        )
        self.assertIn("covar_module.mixture_means", guess)
        self.assertIn("covar_module.mixture_scales", guess)
        self.assertEqual(
            guess["covar_module.mixture_means"].shape,
            torch.Size([2]),
        )
        self.assertEqual(
            guess["covar_module.mixture_scales"].shape,
            torch.Size([2]),
        )

    def test_build_guess_uses_model_inferred_count_when_effective_unset(self):
        lc = _make_1d_lc(seed=7)
        lc.set_model("1D", num_mixtures=4)
        lc._fit_num_mixtures_effective = None

        inferred = lc._infer_num_mixtures_from_model()
        self.assertIsNotNone(inferred)

        mismatch = [0.1] * (int(inferred) + 1)
        with self.assertRaisesRegex(ValueError, "does not match"):
            lc._consensus_build_guess(frequencies=mismatch)


class TestConsensusFitStrategyDispatch(unittest.TestCase):
    """Tests for fit_strategy='consensus' dispatch and unsupported combinations."""

    def test_fit_dispatches_to_consensus_strategy(self):
        lc = _make_1d_lc()
        with patch.object(
            lc,
            "_consensus_standard_fit",
            return_value={"ok": True},
        ) as mock_consensus:
            result = lc.fit(
                fit_strategy="consensus",
                model="1D",
                consensus_frequencies=[0.2],
            )
        self.assertEqual(result, {"ok": True})
        mock_consensus.assert_called_once()

    def test_consensus_fit_end_to_end_applies_consensus_guess(self):
        lc = _make_1d_lc(seed=1)
        _fit_without_training(
            lc,
            fit_strategy="consensus",
            model="1D",
            num_mixtures=2,
            use_mls_init=False,
            training_iter=1,
            consensus_frequencies=[0.11, 0.22],
            consensus_scales=[0.01, 0.02],
        )

        means = lc.model.covar_module.mixture_means.detach().reshape(-1)
        scales = lc.model.covar_module.mixture_scales.detach().reshape(-1)

        torch.testing.assert_close(
            means,
            torch.tensor([0.11, 0.22], dtype=means.dtype),
        )
        torch.testing.assert_close(
            scales,
            torch.tensor([0.01, 0.02], dtype=scales.dtype),
        )

    def test_consensus_without_frequencies_raises_not_implemented(self):
        lc = _make_1d_lc()
        with self.assertRaisesRegex(NotImplementedError, "consensus_frequencies"):
            lc.fit(
                fit_strategy="consensus",
                model="1D",
                training_iter=0,
            )

    def test_unsupported_strategy_variants_raise(self):
        lc = _make_1d_lc(seed=3)
        with self.assertRaisesRegex(NotImplementedError, "consensus_multicomp"):
            lc.fit(
                fit_strategy="consensus_multicomp",
                model="1D",
                consensus_frequencies=[0.2],
                training_iter=0,
            )

        with self.assertRaisesRegex(NotImplementedError, "consensus_relaxed"):
            lc.fit(
                fit_strategy="consensus_relaxed",
                model="1D",
                consensus_frequencies=[0.2],
                training_iter=0,
            )

        with self.assertRaisesRegex(ValueError, "Invalid fit_strategy"):
            lc.fit(
                fit_strategy="not_a_strategy",
                model="1D",
                consensus_frequencies=[0.2],
                training_iter=0,
            )

    def test_consensus_with_unsupported_model_combination_raises(self):
        lc = _make_2d_lc(seed=9)
        with self.assertRaisesRegex(RuntimeError, "Could not resolve"):
            lc.fit(
                fit_strategy="consensus",
                model="2DSeparable",
                consensus_frequencies=[0.1, 0.2],
                training_iter=0,
            )


if __name__ == "__main__":
    unittest.main()
