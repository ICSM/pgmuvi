"""Focused tests for fit_strategy='consensus_multicomp' wiring."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from pgmuvi.lightcurve import ConsensusFitError, Lightcurve


def _make_minimal_multiband_lightcurve():
    t = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=float)
    wl = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=float)
    x = np.column_stack([t, wl])
    y = np.array([0.0, 0.2, -0.1, 0.1, -0.2, 0.05], dtype=float)
    band = np.array(["A", "A", "A", "B", "B", "B"])
    return Lightcurve(x, y, band=band)


def _band_component_candidates():
    return [
        {
            "band_name": "A",
            "wavelength": 1.0,
            "component_candidates": [
                {
                    "frequency": 1.0,
                    "period": 1.0,
                    "ls_rank": 0,
                    "peak_power": 4.0,
                    "peak_prominence": 1.5,
                    "significant": True,
                },
                {
                    "frequency": 3.0,
                    "period": 1.0 / 3.0,
                    "ls_rank": 1,
                    "peak_power": 2.0,
                    "peak_prominence": 0.8,
                    "significant": True,
                },
            ],
        },
        {
            "band_name": "B",
            "wavelength": 2.0,
            "component_candidates": [
                {
                    "frequency": 1.02,
                    "period": 1.0 / 1.02,
                    "ls_rank": 0,
                    "peak_power": 3.0,
                    "peak_prominence": 1.2,
                    "significant": True,
                },
                {
                    "frequency": 2.98,
                    "period": 1.0 / 2.98,
                    "ls_rank": 1,
                    "peak_power": 1.8,
                    "peak_prominence": 0.7,
                    "significant": True,
                },
            ],
        },
    ]


def _component_clusters(accepted=True):
    return [
        {
            "cluster_id": 0,
            "accepted": accepted,
            "rejection_reasons": [] if accepted else ["insufficient_bands (1 < 2)"],
            "member_bands": ["A", "B"],
            "n_member_bands": 2 if accepted else 1,
            "center_frequency": 1.01,
            "center_period": 1.0 / 1.01,
            "log_center_frequency": float(np.log(1.01)),
            "frequency_scatter": 0.01,
            "log_frequency_scatter": 0.01,
            "members": [
                {
                    "band_name": "A",
                    "frequency": 1.0,
                    "peak_power": 4.0,
                    "significant": True,
                },
                {
                    "band_name": "B",
                    "frequency": 1.02,
                    "peak_power": 3.0,
                    "significant": True,
                },
            ],
            "duplicate_band_candidates": [],
        },
        {
            "cluster_id": 1,
            "accepted": accepted,
            "rejection_reasons": [] if accepted else ["insufficient_bands (1 < 2)"],
            "member_bands": ["A", "B"],
            "n_member_bands": 2 if accepted else 1,
            "center_frequency": 2.99,
            "center_period": 1.0 / 2.99,
            "log_center_frequency": float(np.log(2.99)),
            "frequency_scatter": 0.02,
            "log_frequency_scatter": 0.01,
            "members": [
                {
                    "band_name": "A",
                    "frequency": 3.0,
                    "peak_power": 2.0,
                    "significant": True,
                },
                {
                    "band_name": "B",
                    "frequency": 2.98,
                    "peak_power": 1.8,
                    "significant": True,
                },
            ],
            "duplicate_band_candidates": [],
        },
    ]


def _multicomponent_consensus():
    accepted_clusters = _component_clusters(accepted=True)
    return {
        "consensus_frequencies": [1.01, 2.99],
        "consensus_frequency_widths": [0.1, 0.2],
        "consensus_scales": [7.0, 3.8],
        "accepted_clusters": accepted_clusters,
        "rejected_clusters": [],
        "component_summaries": [
            {
                "component_index": 0,
                "source_cluster_id": 0,
                "consensus_frequency": 1.01,
                "consensus_frequency_width": 0.1,
                "consensus_scale": 7.0,
            },
            {
                "component_index": 1,
                "source_cluster_id": 1,
                "consensus_frequency": 2.99,
                "consensus_frequency_width": 0.2,
                "consensus_scale": 3.8,
            },
        ],
    }


def _initialization_diagnostics():
    return {
        "requested_consensus_frequencies": [1.01, 2.99],
        "requested_consensus_scales": [7.0, 3.8],
        "initialized_mixture_means": [1.01, 2.99],
        "initialized_mixture_scales": [7.0, 3.8],
        "initialization_strategy": "per_component_consensus_initialization",
    }


class _DummyInitializationModel:
    def __init__(self):
        self.covar_module = mock.MagicMock()
        self.covar_module.mixture_means = torch.tensor(
            [[[0.0], [0.0]]],
            dtype=torch.float32,
        )
        self.covar_module.mixture_scales = torch.tensor(
            [[[0.0], [0.0]]],
            dtype=torch.float32,
        )

    def initialize(self, **kwargs):
        if "covar_module.mixture_means" in kwargs:
            self.covar_module.mixture_means = kwargs["covar_module.mixture_means"]
        if "covar_module.mixture_scales" in kwargs:
            self.covar_module.mixture_scales = kwargs["covar_module.mixture_scales"]


class TestConsensusMulticompFit(unittest.TestCase):
    def setUp(self):
        self.lc = _make_minimal_multiband_lightcurve()
        self.lc.model = object()
        self.lc._model_pars = {}

    def test_consensus_multicomp_chains_all_helpers(self):
        sentinel = {"status": "ok"}
        with mock.patch.object(
            self.lc,
            "_consensus_collect_band_component_candidates",
            return_value=_band_component_candidates(),
        ) as collect_mock, mock.patch.object(
            self.lc,
            "_consensus_cluster_component_candidates",
            return_value=_component_clusters(),
        ) as cluster_mock, mock.patch.object(
            self.lc,
            "_consensus_build_multicomponent_frequency_consensus",
            return_value=_multicomponent_consensus(),
        ) as consensus_mock, mock.patch.object(
            self.lc,
            "_consensus_build_guess",
            return_value={"dummy_guess": 1.0},
        ) as guess_mock, mock.patch.object(
            self.lc,
            "_consensus_collect_initialization_diagnostics",
            return_value=_initialization_diagnostics(),
        ) as init_diag_mock, mock.patch.object(
            self.lc,
            "fit",
            return_value=sentinel,
        ) as fit_mock:
            result = self.lc._consensus_fit(
                fit_strategy="consensus_multicomp",
                model=None,
                constrain_consensus=False,
                _allow_existing_model_for_consensus=True,
            )

        self.assertIs(result, sentinel)
        collect_mock.assert_called_once()
        cluster_mock.assert_called_once()
        consensus_mock.assert_called_once()
        guess_mock.assert_called_once()
        init_diag_mock.assert_called_once()
        fit_mock.assert_called_once()

    def test_multicomp_fit_infers_num_components_and_passes_frequencies_to_init(self):
        with mock.patch.object(
            self.lc,
            "_consensus_collect_band_component_candidates",
            return_value=_band_component_candidates(),
        ), mock.patch.object(
            self.lc,
            "_consensus_cluster_component_candidates",
            return_value=_component_clusters(),
        ), mock.patch.object(
            self.lc,
            "_consensus_build_multicomponent_frequency_consensus",
            return_value=_multicomponent_consensus(),
        ), mock.patch.object(
            self.lc,
            "_consensus_build_guess",
            return_value={"dummy_guess": 1.0},
        ) as guess_mock, mock.patch.object(
            self.lc,
            "_consensus_collect_initialization_diagnostics",
            return_value=_initialization_diagnostics(),
        ), mock.patch.object(
            self.lc,
            "fit",
            return_value={"status": "ok"},
        ) as fit_mock:
            self.lc._consensus_multicomp_fit(
                model=None,
                constrain_consensus=False,
                _allow_existing_model_for_consensus=True,
            )

        np.testing.assert_allclose(
            guess_mock.call_args.kwargs["frequencies"],
            np.array([1.01, 2.99]),
        )
        np.testing.assert_allclose(
            guess_mock.call_args.kwargs["scales"],
            np.array([7.0, 3.8]),
        )
        self.assertEqual(fit_mock.call_args.kwargs["num_mixtures"], 2)
        self.assertEqual(self.lc.consensus_diagnostics["n_components"], 2)

    def test_empty_consensus_results_raise_informative_consensus_fit_error(self):
        rejected_clusters = _component_clusters(accepted=False)
        empty_consensus = {
            "consensus_frequencies": [],
            "consensus_frequency_widths": [],
            "consensus_scales": [],
            "accepted_clusters": [],
            "rejected_clusters": rejected_clusters,
            "component_summaries": [],
        }
        with mock.patch.object(
            self.lc,
            "_consensus_collect_band_component_candidates",
            return_value=_band_component_candidates(),
        ), mock.patch.object(
            self.lc,
            "_consensus_cluster_component_candidates",
            return_value=rejected_clusters,
        ), mock.patch.object(
            self.lc,
            "_consensus_build_multicomponent_frequency_consensus",
            return_value=empty_consensus,
        ):
            with self.assertRaisesRegex(
                ConsensusFitError,
                "no accepted multicomponent consensus clusters were available",
            ):
                self.lc._consensus_multicomp_fit(
                    model=None,
                    constrain_consensus=False,
                    _allow_existing_model_for_consensus=True,
                )

    def test_diagnostics_include_multicomp_stage_outputs(self):
        with mock.patch.object(
            self.lc,
            "_consensus_collect_band_component_candidates",
            return_value=_band_component_candidates(),
        ), mock.patch.object(
            self.lc,
            "_consensus_cluster_component_candidates",
            return_value=_component_clusters(),
        ), mock.patch.object(
            self.lc,
            "_consensus_build_multicomponent_frequency_consensus",
            return_value=_multicomponent_consensus(),
        ), mock.patch.object(
            self.lc,
            "_consensus_build_guess",
            return_value={"dummy_guess": 1.0},
        ), mock.patch.object(
            self.lc,
            "_consensus_collect_initialization_diagnostics",
            return_value=_initialization_diagnostics(),
        ), mock.patch.object(
            self.lc,
            "fit",
            return_value={"status": "ok"},
        ):
            self.lc._consensus_multicomp_fit(
                model=None,
                constrain_consensus=False,
                _allow_existing_model_for_consensus=True,
            )

        diagnostics = self.lc.consensus_diagnostics
        self.assertIn("band_component_candidates", diagnostics)
        self.assertIn("component_clusters", diagnostics)
        self.assertIn("multicomponent_consensus", diagnostics)
        self.assertEqual(diagnostics["consensus_frequencies"], [1.01, 2.99])
        self.assertEqual(diagnostics["consensus_frequency_widths"], [0.1, 0.2])
        self.assertEqual(diagnostics["consensus_scales"], [7.0, 3.8])
        self.assertEqual(diagnostics["trusted_candidate_count"], 4)
        self.assertIsNone(diagnostics["median_frequency"])
        self.assertIsNone(diagnostics["consensus_frequency"])
        self.assertIsNone(diagnostics["consensus_period"])
        self.assertIsNone(diagnostics["final_consensus_frequency"])
        self.assertIsNone(diagnostics["final_consensus_period"])
        self.assertTrue(diagnostics["consensus_success"])
        self.assertEqual(diagnostics["constraint_strategy"], "constraints_disabled")
        self.assertEqual(
            diagnostics["initialization_strategy"],
            "per_component_consensus_initialization",
        )
        self.assertEqual(diagnostics["requested_consensus_frequencies"], [1.01, 2.99])
        self.assertEqual(diagnostics["initialized_mixture_means"], [1.01, 2.99])

    def test_constraint_strategy_records_global_interval_limitation(self):
        with mock.patch.object(
            self.lc,
            "_consensus_collect_band_component_candidates",
            return_value=_band_component_candidates(),
        ), mock.patch.object(
            self.lc,
            "_consensus_cluster_component_candidates",
            return_value=_component_clusters(),
        ), mock.patch.object(
            self.lc,
            "_consensus_build_multicomponent_frequency_consensus",
            return_value=_multicomponent_consensus(),
        ), mock.patch.object(
            self.lc,
            "_consensus_validate_final_model_supports_sm_time_kernel",
            return_value=None,
        ), mock.patch.object(
            self.lc,
            "_consensus_resolve_time_spectral_mixture_keys",
            return_value={
                "mixture_means": "means",
                "mixture_scales": "scales",
            },
        ), mock.patch.object(
            self.lc,
            "set_default_constraints",
            return_value=None,
        ), mock.patch.object(
            self.lc,
            "set_constraint",
            return_value=None,
        ), mock.patch.object(
            self.lc,
            "_consensus_validate_applied_sm_constraints",
            return_value=None,
        ), mock.patch.object(
            self.lc,
            "_consensus_build_guess",
            return_value={"dummy_guess": 1.0},
        ), mock.patch.object(
            self.lc,
            "_consensus_collect_initialization_diagnostics",
            return_value=_initialization_diagnostics(),
        ), mock.patch.object(
            self.lc,
            "fit",
            return_value={"status": "ok"},
        ):
            self.lc._consensus_multicomp_fit(
                model=None,
                constrain_consensus=True,
                _allow_existing_model_for_consensus=True,
            )

        diagnostics = self.lc.consensus_diagnostics
        self.assertEqual(
            diagnostics["constraint_strategy"],
            "global_frequency_interval",
        )
        self.assertAlmostEqual(diagnostics["consensus_constraint_bounds"][0], 0.71)
        self.assertAlmostEqual(diagnostics["consensus_constraint_bounds"][1], 3.59)
        self.assertEqual(
            diagnostics["initialization_strategy"],
            "per_component_consensus_initialization",
        )

    def test_collect_initialization_diagnostics_matches_requested_frequencies(self):
        self.lc.model = _DummyInitializationModel()
        self.lc._model_pars = {
            "covar_module.mixture_means": {"module": self.lc.model.covar_module},
            "covar_module.mixture_scales": {"module": self.lc.model.covar_module},
        }
        consensus_guess = {
            "covar_module.mixture_means": torch.tensor(
                [[[1.01], [2.99]]],
                dtype=torch.float32,
            ),
            "covar_module.mixture_scales": torch.tensor(
                [[[7.0], [3.8]]],
                dtype=torch.float32,
            ),
        }
        with mock.patch.object(
            self.lc,
            "_consensus_resolve_time_spectral_mixture_keys",
            return_value={
                "mixture_means": "covar_module.mixture_means",
                "mixture_scales": "covar_module.mixture_scales",
            },
        ):
            diagnostics = self.lc._consensus_collect_initialization_diagnostics(
                requested_consensus_frequencies=np.array([1.01, 2.99], dtype=float),
                requested_consensus_scales=np.array([7.0, 3.8], dtype=float),
                consensus_guess=consensus_guess,
            )

        np.testing.assert_allclose(
            diagnostics["initialized_mixture_means"],
            [1.01, 2.99],
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            diagnostics["initialized_mixture_scales"],
            [7.0, 3.8],
            atol=1e-6,
            rtol=0.0,
        )
        self.assertEqual(
            diagnostics["initialization_strategy"],
            "per_component_consensus_initialization",
        )

    def test_collect_initialization_diagnostics_raises_on_component_mismatch(self):
        self.lc.model = _DummyInitializationModel()
        self.lc._model_pars = {
            "covar_module.mixture_means": {"module": self.lc.model.covar_module},
            "covar_module.mixture_scales": {"module": self.lc.model.covar_module},
        }
        bad_guess = {
            "covar_module.mixture_means": torch.tensor([[[1.01]]], dtype=torch.float32),
            "covar_module.mixture_scales": torch.tensor([[[7.0]]], dtype=torch.float32),
        }
        with mock.patch.object(
            self.lc,
            "_consensus_resolve_time_spectral_mixture_keys",
            return_value={
                "mixture_means": "covar_module.mixture_means",
                "mixture_scales": "covar_module.mixture_scales",
            },
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "len\\(initialized_mixture_means\\)=1 does not match "
                "len\\(consensus_frequencies\\)=2",
            ):
                self.lc._consensus_collect_initialization_diagnostics(
                    requested_consensus_frequencies=np.array([1.01, 2.99], dtype=float),
                    requested_consensus_scales=np.array([7.0, 3.8], dtype=float),
                    consensus_guess=bad_guess,
                )

    def test_existing_consensus_dispatch_is_unchanged(self):
        with mock.patch.object(
            self.lc,
            "_consensus_standard_fit",
            return_value="standard-result",
        ) as standard_mock, mock.patch.object(
            self.lc,
            "_consensus_multicomp_fit",
        ) as multicomp_mock:
            result = self.lc._consensus_fit(fit_strategy="consensus")

        self.assertEqual(result, "standard-result")
        standard_mock.assert_called_once()
        multicomp_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
