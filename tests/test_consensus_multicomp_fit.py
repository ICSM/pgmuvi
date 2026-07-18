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
                "member_bands": ["A", "B"],
                "n_member_bands": 2,
            },
            {
                "component_index": 1,
                "source_cluster_id": 1,
                "consensus_frequency": 2.99,
                "consensus_frequency_width": 0.2,
                "consensus_scale": 3.8,
                "member_bands": ["A", "B"],
                "n_member_bands": 2,
            },
        ],
    }


def _initialization_diagnostics():
    initialized_means = np.array([1.01, 2.99], dtype=float)
    initialized_scales = np.array([0.1, 0.2], dtype=float)
    return {
        "requested_consensus_frequencies": [1.01, 2.99],
        "requested_consensus_scales": [0.1, 0.2],
        "requested_consensus_frequency_widths": [0.1, 0.2],
        "requested_consensus_periods": (1.0 / initialized_means).tolist(),
        "requested_consensus_period_widths": (
            initialized_scales / (initialized_means**2)
        ).tolist(),
        "initialized_mixture_means": [1.01, 2.99],
        "initialized_mixture_periods": (1.0 / initialized_means).tolist(),
        "initialized_mixture_scales": [0.1, 0.2],
        "initialized_mixture_period_widths": (
            initialized_scales / (initialized_means**2)
        ).tolist(),
        "initialization_strategy": "per_component_consensus_initialization",
    }


def _mock_fitted_diagnostics():
    fitted_frequencies = np.array([1.02, 2.98], dtype=float)
    fitted_scales = np.array([0.11, 0.21], dtype=float)
    initialized_frequencies = np.array([1.01, 2.99], dtype=float)
    initialized_periods = 1.0 / initialized_frequencies
    fitted_periods = 1.0 / fitted_frequencies
    frequency_shift = fitted_frequencies - initialized_frequencies
    period_shift = fitted_periods - initialized_periods
    return {
        "fitted_mixture_frequencies": fitted_frequencies.tolist(),
        "fitted_mixture_periods": fitted_periods.tolist(),
        "fitted_mixture_scales": fitted_scales.tolist(),
        "fitted_mixture_period_widths": (fitted_scales / (fitted_frequencies**2)).tolist(),
        "fitted_frequency_shift_from_initialization": frequency_shift.tolist(),
        "fitted_period_shift_from_initialization": period_shift.tolist(),
        "fitted_fractional_frequency_shift_from_initialization": (
            frequency_shift / initialized_frequencies
        ).tolist(),
        "fitted_fractional_period_shift_from_initialization": (
            period_shift / initialized_periods
        ).tolist(),
    }


class _MockInitializableSpectralMixtureModel:
    def __init__(self, n_components=2):
        self.covar_module = mock.MagicMock()
        zeros = torch.zeros((1, n_components, 1), dtype=torch.float32)
        self.covar_module.mixture_means = zeros.clone()
        self.covar_module.mixture_scales = zeros.clone()

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
            "_consensus_collect_fitted_mixture_diagnostics",
            return_value=_mock_fitted_diagnostics(),
        ) as fitted_diag_mock, mock.patch.object(
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
        fitted_diag_mock.assert_called_once()
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
            "_consensus_collect_fitted_mixture_diagnostics",
            return_value=_mock_fitted_diagnostics(),
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
            np.array([0.1, 0.2]),
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
            "_consensus_collect_fitted_mixture_diagnostics",
            return_value=_mock_fitted_diagnostics(),
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
        np.testing.assert_allclose(
            diagnostics["consensus_periods"],
            [1.0 / 1.01, 1.0 / 2.99],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            diagnostics["consensus_period_widths"],
            [0.1 / (1.01**2), 0.2 / (2.99**2)],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(diagnostics["consensus_scales"], [7.0, 3.8])
        self.assertEqual(diagnostics["consensus_component_strengths"], [7.0, 3.8])
        self.assertEqual(diagnostics["consensus_mixture_init_scales"], [0.1, 0.2])
        self.assertEqual(diagnostics["primary_component_index"], 0)
        self.assertAlmostEqual(diagnostics["primary_consensus_frequency"], 1.01)
        self.assertAlmostEqual(diagnostics["primary_consensus_period"], 1.0 / 1.01)
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
        self.assertEqual(diagnostics["requested_consensus_frequency_widths"], [0.1, 0.2])
        self.assertEqual(diagnostics["initialized_mixture_means"], [1.01, 2.99])
        self.assertEqual(diagnostics["initialized_mixture_scales"], [0.1, 0.2])
        self.assertEqual(diagnostics["fitted_mixture_frequencies"], [1.02, 2.98])
        self.assertEqual(diagnostics["fitted_mixture_scales"], [0.11, 0.21])
        np.testing.assert_allclose(
            diagnostics["fitted_mixture_periods"],
            [1.0 / 1.02, 1.0 / 2.98],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            diagnostics["fitted_mixture_period_widths"],
            [0.11 / (1.02**2), 0.21 / (2.98**2)],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            diagnostics["fitted_frequency_shift_from_initialization"],
            [0.01, -0.01],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            diagnostics["initialized_mixture_periods"],
            [1.0 / 1.01, 1.0 / 2.99],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            diagnostics["initialized_mixture_period_widths"],
            [0.1 / (1.01**2), 0.2 / (2.99**2)],
            rtol=0.0,
            atol=1e-12,
        )
        summaries = diagnostics["multicomponent_period_summaries"]
        self.assertEqual(len(summaries), 2)
        self.assertEqual(
            [entry["component_index"] for entry in summaries],
            [0, 1],
        )
        self.assertEqual(
            [entry["source_cluster_id"] for entry in summaries],
            [0, 1],
        )
        self.assertEqual(summaries[0]["member_bands"], ["A", "B"])
        self.assertEqual(summaries[1]["member_bands"], ["A", "B"])
        self.assertEqual([entry["n_member_bands"] for entry in summaries], [2, 2])
        self.assertIsNotNone(summaries[0]["fitted_mixture_frequency"])
        self.assertIsNotNone(summaries[1]["fitted_mixture_frequency"])
        self.assertIn(
            "fitted_fractional_period_shift_from_initialization",
            summaries[0],
        )
        expected_abs_period_shift = np.abs(
            np.asarray(
                diagnostics["fitted_fractional_period_shift_from_initialization"],
                dtype=float,
            )
        )
        expected_abs_frequency_shift = np.abs(
            np.asarray(
                diagnostics["fitted_fractional_frequency_shift_from_initialization"],
                dtype=float,
            )
        )
        np.testing.assert_allclose(
            diagnostics["max_abs_fractional_period_shift_from_initialization"],
            float(np.max(expected_abs_period_shift)),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            diagnostics["max_abs_fractional_frequency_shift_from_initialization"],
            float(np.max(expected_abs_frequency_shift)),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(diagnostics["component_fit_drift_flags"], [False, False])
        self.assertEqual(diagnostics["components_with_large_period_drift"], [])
        self.assertEqual(diagnostics["components_with_large_frequency_drift"], [])
        self.assertEqual(diagnostics["nearest_initialized_component_index"], [0, 1])
        np.testing.assert_allclose(
            diagnostics["nearest_initialized_component_fractional_period_distance"],
            [abs((1.0 / 1.02 - 1.0 / 1.01) / (1.0 / 1.01)), abs((1.0 / 2.98 - 1.0 / 2.99) / (1.0 / 2.99))],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(diagnostics["component_identity_preserved"], [True, True])
        self.assertTrue(diagnostics["all_component_identities_preserved"])
        self.assertEqual(diagnostics["possible_component_swaps"], [])
        self.assertEqual(diagnostics["drift_warning_fraction"], 0.10)
        for entry in summaries:
            self.assertIn(
                "fitted_abs_fractional_period_shift_from_initialization",
                entry,
            )
            self.assertIn(
                "fitted_abs_fractional_frequency_shift_from_initialization",
                entry,
            )
            self.assertIn("fitted_period_drift_flag", entry)
            self.assertIn("fitted_frequency_drift_flag", entry)
            self.assertIn("nearest_initialized_component_index", entry)
            self.assertIn(
                "nearest_initialized_component_fractional_period_distance",
                entry,
            )
            self.assertIn("component_identity_preserved", entry)
            self.assertFalse(entry["fitted_period_drift_flag"])

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
            "_consensus_apply_temporal_sm_constraint",
            side_effect=lambda key, lower, upper: {
                "parameter": key,
                "ard_scope": "temporal_only",
                "raw_proposed_bounds": [lower, upper],
                "wavelength_bounds_preserved": True,
            },
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
            "_consensus_collect_fitted_mixture_diagnostics",
            return_value=_mock_fitted_diagnostics(),
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
        self.assertEqual(
            diagnostics["consensus_constraint_ard_scope"],
            "temporal_only",
        )
        self.assertTrue(diagnostics["consensus_wavelength_constraint_preserved"])
        self.assertEqual(
            set(diagnostics["consensus_sm_constraint_provenance"]),
            {"mixture_means", "mixture_scales"},
        )
        self.assertAlmostEqual(diagnostics["consensus_constraint_bounds"][0], 0.71)
        self.assertAlmostEqual(diagnostics["consensus_constraint_bounds"][1], 3.59)
        self.assertEqual(
            diagnostics["initialization_strategy"],
            "per_component_consensus_initialization",
        )

    def test_large_period_drift_is_flagged(self):
        fitted_diag = _mock_fitted_diagnostics()
        fitted_diag["fitted_fractional_period_shift_from_initialization"] = [0.0403, -0.124]
        fitted_diag["fitted_fractional_frequency_shift_from_initialization"] = [
            -0.0387,
            0.1415,
        ]
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
            "_consensus_collect_fitted_mixture_diagnostics",
            return_value=fitted_diag,
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
        self.assertEqual(diagnostics["component_fit_drift_flags"], [False, True])
        self.assertEqual(diagnostics["components_with_large_period_drift"], [1])
        self.assertEqual(diagnostics["components_with_large_frequency_drift"], [1])
        self.assertAlmostEqual(
            diagnostics["max_abs_fractional_period_shift_from_initialization"],
            0.124,
            places=12,
        )

    def test_component_identity_swap_is_detected(self):
        fitted_diag = _mock_fitted_diagnostics()
        fitted_diag["fitted_mixture_frequencies"] = [2.98, 1.02]
        fitted_diag["fitted_mixture_periods"] = [1.0 / 2.98, 1.0 / 1.02]
        fitted_diag["fitted_frequency_shift_from_initialization"] = [1.97, -1.97]
        fitted_diag["fitted_period_shift_from_initialization"] = [
            (1.0 / 2.98) - (1.0 / 1.01),
            (1.0 / 1.02) - (1.0 / 2.99),
        ]
        fitted_diag["fitted_fractional_frequency_shift_from_initialization"] = [
            1.97 / 1.01,
            -1.97 / 2.99,
        ]
        fitted_diag["fitted_fractional_period_shift_from_initialization"] = [
            ((1.0 / 2.98) - (1.0 / 1.01)) / (1.0 / 1.01),
            ((1.0 / 1.02) - (1.0 / 2.99)) / (1.0 / 2.99),
        ]
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
            "_consensus_collect_fitted_mixture_diagnostics",
            return_value=fitted_diag,
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
        self.assertEqual(diagnostics["nearest_initialized_component_index"], [1, 0])
        self.assertEqual(diagnostics["component_identity_preserved"], [False, False])
        self.assertFalse(diagnostics["all_component_identities_preserved"])
        self.assertEqual(diagnostics["possible_component_swaps"], [0, 1])

    def test_component_identity_tie_uses_lower_index(self):
        identity = Lightcurve._consensus_compute_component_identity_diagnostics(
            initialized_periods=[1.0, 3.0],
            fitted_periods=[1.5, 3.0],
            initialized_frequencies=[1.0, 1.0 / 3.0],
            fitted_frequencies=[2.0 / 3.0, 1.0 / 3.0],
        )
        self.assertEqual(identity["nearest_initialized_component_index"], [0, 1])
        self.assertEqual(identity["component_identity_preserved"], [True, True])
        np.testing.assert_allclose(
            identity["nearest_initialized_component_fractional_period_distance"],
            [0.5, 0.0],
            rtol=0.0,
            atol=1e-12,
        )

    def test_collect_initialization_diagnostics_matches_requested_parameters(self):
        self.lc.model = _MockInitializableSpectralMixtureModel()
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
                [[[0.1], [0.2]]],
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
                requested_consensus_scales=np.array([0.1, 0.2], dtype=float),
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
            [0.1, 0.2],
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            diagnostics["initialized_mixture_periods"],
            [1.0 / 1.01, 1.0 / 2.99],
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            diagnostics["initialized_mixture_period_widths"],
            [0.1 / (1.01**2), 0.2 / (2.99**2)],
            atol=1e-6,
            rtol=0.0,
        )
        self.assertEqual(
            diagnostics["initialization_strategy"],
            "per_component_consensus_initialization",
        )

    def test_collect_initialization_diagnostics_raises_on_component_mismatch(self):
        self.lc.model = _MockInitializableSpectralMixtureModel()
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
                "len\\(requested_consensus_frequencies\\)=2",
            ):
                self.lc._consensus_collect_initialization_diagnostics(
                    requested_consensus_frequencies=np.array([1.01, 2.99], dtype=float),
                    requested_consensus_scales=np.array([0.1, 0.2], dtype=float),
                    consensus_guess=bad_guess,
                )

    def test_collect_fitted_mixture_diagnostics_computes_expected_values(self):
        self.lc.model = _MockInitializableSpectralMixtureModel()
        self.lc._model_pars = {
            "covar_module.mixture_means": {"module": self.lc.model.covar_module},
            "covar_module.mixture_scales": {"module": self.lc.model.covar_module},
        }
        with mock.patch.object(
            self.lc,
            "_consensus_resolve_time_spectral_mixture_keys",
            return_value={
                "mixture_means": "covar_module.mixture_means",
                "mixture_scales": "covar_module.mixture_scales",
            },
        ):
            self.lc.model.initialize(
                **{
                    "covar_module.mixture_means": torch.tensor(
                        [[[1.02], [2.98]]], dtype=torch.float32
                    ),
                    "covar_module.mixture_scales": torch.tensor(
                        [[[0.11], [0.21]]], dtype=torch.float32
                    ),
                }
            )
            diagnostics = self.lc._consensus_collect_fitted_mixture_diagnostics(
                initialized_mixture_frequencies=np.array([1.01, 2.99], dtype=float),
                initialized_mixture_scales=np.array([0.1, 0.2], dtype=float),
            )

        np.testing.assert_allclose(
            diagnostics["fitted_mixture_frequencies"], [1.02, 2.98], atol=1e-6, rtol=0.0
        )
        np.testing.assert_allclose(
            diagnostics["fitted_mixture_periods"],
            [1.0 / 1.02, 1.0 / 2.98],
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            diagnostics["fitted_mixture_period_widths"],
            [0.11 / (1.02**2), 0.21 / (2.98**2)],
            atol=1e-6,
            rtol=0.0,
        )
        self.assertEqual(
            len(diagnostics["fitted_fractional_period_shift_from_initialization"]),
            2,
        )

    def test_collect_fitted_mixture_diagnostics_raises_on_component_mismatch(self):
        self.lc.model = _MockInitializableSpectralMixtureModel(n_components=1)
        self.lc._model_pars = {
            "covar_module.mixture_means": {"module": self.lc.model.covar_module},
            "covar_module.mixture_scales": {"module": self.lc.model.covar_module},
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
                "len\\(fitted_mixture_frequencies\\)=1 does not match "
                "len\\(initialized_mixture_means\\)=2",
            ):
                self.lc._consensus_collect_fitted_mixture_diagnostics(
                    initialized_mixture_frequencies=np.array([1.01, 2.99], dtype=float),
                    initialized_mixture_scales=np.array([0.1, 0.2], dtype=float),
                )

    def test_training_iter_zero_keeps_fitted_close_to_initialized(self):
        self.lc.model = _MockInitializableSpectralMixtureModel(n_components=2)
        self.lc._model_pars = {
            "covar_module.mixture_means": {"module": self.lc.model.covar_module},
            "covar_module.mixture_scales": {"module": self.lc.model.covar_module},
        }
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
            "_consensus_resolve_time_spectral_mixture_keys",
            return_value={
                "mixture_means": "covar_module.mixture_means",
                "mixture_scales": "covar_module.mixture_scales",
            },
        ), mock.patch.object(
            self.lc,
            "fit",
            return_value={"status": "ok"},
        ):
            self.lc._consensus_multicomp_fit(
                model=None,
                constrain_consensus=False,
                training_iter=0,
                _allow_existing_model_for_consensus=True,
            )

        diagnostics = self.lc.consensus_diagnostics
        np.testing.assert_allclose(
            diagnostics["fitted_mixture_frequencies"],
            diagnostics["initialized_mixture_means"],
            atol=1e-8,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            diagnostics["fitted_mixture_scales"],
            diagnostics["initialized_mixture_scales"],
            atol=1e-8,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            diagnostics["fitted_frequency_shift_from_initialization"],
            [0.0, 0.0],
            atol=1e-8,
            rtol=0.0,
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


# ---------------------------------------------------------------------------
# Tests for _consensus_multicomp_reconcile_to_n_components (static helper)
# ---------------------------------------------------------------------------

def _make_n_component_summaries(n):
    """Return a list of N synthetic accepted component summaries."""
    summaries = []
    for i in range(n):
        freq = float(i + 1) * 1.0
        summaries.append({
            "component_index": i,
            "source_cluster_id": i,
            "consensus_frequency": freq,
            "consensus_frequency_width": 0.1 * freq,
            "consensus_scale": float(n - i),  # first component strongest
            "n_member_bands": 2 if i < 2 else 1,
            "member_bands": ["A", "B"] if i < 2 else ["A"],
        })
    return summaries


def _make_n_component_init_diagnostics(n, freqs, scales):
    """Return mock initialization diagnostics for N components."""
    freqs = np.asarray(freqs, dtype=float)
    scales = np.asarray(scales, dtype=float)
    return {
        "requested_consensus_frequencies": freqs.tolist(),
        "requested_consensus_scales": scales.tolist(),
        "requested_consensus_frequency_widths": scales.tolist(),
        "requested_consensus_periods": (1.0 / freqs).tolist(),
        "requested_consensus_period_widths": (scales / freqs**2).tolist(),
        "initialized_mixture_means": freqs.tolist(),
        "initialized_mixture_periods": (1.0 / freqs).tolist(),
        "initialized_mixture_scales": scales.tolist(),
        "initialized_mixture_period_widths": (scales / freqs**2).tolist(),
        "initialization_strategy": "per_component_consensus_initialization",
    }


def _make_n_component_fitted_diagnostics(n, freqs, scales):
    """Return mock fitted diagnostics for N components (tiny shift from init)."""
    freqs = np.asarray(freqs, dtype=float)
    fitted_freqs = freqs * 1.001  # tiny shift
    fitted_scales = np.asarray(scales, dtype=float) * 1.01
    init_periods = 1.0 / freqs
    fitted_periods = 1.0 / fitted_freqs
    freq_shift = fitted_freqs - freqs
    period_shift = fitted_periods - init_periods
    return {
        "fitted_mixture_frequencies": fitted_freqs.tolist(),
        "fitted_mixture_periods": fitted_periods.tolist(),
        "fitted_mixture_scales": fitted_scales.tolist(),
        "fitted_mixture_period_widths": (fitted_scales / fitted_freqs**2).tolist(),
        "fitted_frequency_shift_from_initialization": freq_shift.tolist(),
        "fitted_period_shift_from_initialization": period_shift.tolist(),
        "fitted_fractional_frequency_shift_from_initialization": (
            freq_shift / freqs
        ).tolist(),
        "fitted_fractional_period_shift_from_initialization": (
            period_shift / init_periods
        ).tolist(),
    }


class TestReconcileToNComponents(unittest.TestCase):
    """Unit tests for _consensus_multicomp_reconcile_to_n_components."""

    def _call(self, accepted, requested_n, rejected=None, band_cands=None):
        return Lightcurve._consensus_multicomp_reconcile_to_n_components(
            accepted_component_summaries=accepted,
            requested_num_mixtures=requested_n,
            rejected_clusters=rejected or [],
            band_component_candidates=band_cands or [],
        )

    def test_exact_match_m_equals_n(self):
        summaries = _make_n_component_summaries(2)
        reconciled, diag = self._call(summaries, requested_n=2)
        self.assertEqual(len(reconciled), 2)
        self.assertEqual(diag["component_count_reconciliation_strategy"], "exact_match")
        self.assertEqual(diag["accepted_consensus_component_count"], 2)
        self.assertEqual(diag["initialization_component_count"], 2)
        self.assertEqual(diag["requested_num_mixtures"], 2)
        self.assertEqual(diag["dropped_consensus_components"], [])
        self.assertEqual(diag["fallback_initialization_components"], [])
        for s in reconciled:
            self.assertEqual(s["component_source"], "accepted_consensus")

    def test_no_requested_n_uses_m(self):
        summaries = _make_n_component_summaries(2)
        reconciled, diag = self._call(summaries, requested_n=None)
        self.assertEqual(len(reconciled), 2)
        self.assertEqual(diag["requested_num_mixtures"], None)
        self.assertEqual(diag["accepted_consensus_component_count"], 2)
        self.assertEqual(diag["initialization_component_count"], 2)

    def test_m_greater_than_n_drops_weakest(self):
        # M=3 components, keep top N=1
        summaries = [
            {
                "component_index": 0,
                "source_cluster_id": 0,
                "consensus_frequency": 1.0,
                "consensus_frequency_width": 0.1,
                "consensus_scale": 5.0,   # medium
                "n_member_bands": 2,
                "member_bands": ["A", "B"],
            },
            {
                "component_index": 1,
                "source_cluster_id": 1,
                "consensus_frequency": 2.0,
                "consensus_frequency_width": 0.2,
                "consensus_scale": 10.0,  # strongest → should be KEPT
                "n_member_bands": 3,
                "member_bands": ["A", "B", "C"],
            },
            {
                "component_index": 2,
                "source_cluster_id": 2,
                "consensus_frequency": 3.0,
                "consensus_frequency_width": 0.3,
                "consensus_scale": 2.0,   # weakest → dropped
                "n_member_bands": 1,
                "member_bands": ["A"],
            },
        ]
        reconciled, diag = self._call(summaries, requested_n=1)
        self.assertEqual(len(reconciled), 1)
        self.assertEqual(diag["component_count_reconciliation_strategy"], "drop_weakest")
        self.assertEqual(diag["accepted_consensus_component_count"], 3)
        self.assertEqual(diag["initialization_component_count"], 1)
        self.assertEqual(diag["requested_num_mixtures"], 1)
        # Kept component: highest n_member_bands (3) AND highest consensus_scale
        # (10.0) → source_cluster_id=1 (freq=2.0); others have lower support.
        self.assertAlmostEqual(reconciled[0]["consensus_frequency"], 2.0)
        self.assertEqual(reconciled[0]["component_source"], "accepted_consensus")
        # Two components dropped
        self.assertEqual(len(diag["dropped_consensus_components"]), 2)
        dropped_cids = {d["source_cluster_id"] for d in diag["dropped_consensus_components"]}
        self.assertIn(0, dropped_cids)
        self.assertIn(2, dropped_cids)

    def test_m_greater_than_n_keeps_two_of_three(self):
        summaries = _make_n_component_summaries(3)  # scales 3, 2, 1; n_member_bands 2,2,1
        reconciled, diag = self._call(summaries, requested_n=2)
        self.assertEqual(len(reconciled), 2)
        self.assertEqual(diag["component_count_reconciliation_strategy"], "drop_weakest")
        self.assertEqual(len(diag["dropped_consensus_components"]), 1)
        # Third component (index 2, 1 member band, scale 1) should be dropped
        self.assertEqual(diag["dropped_consensus_components"][0]["source_cluster_id"], 2)

    def test_m_greater_than_n_ranking_n_member_bands_first(self):
        # All have same scale; ranking must use n_member_bands then cluster_id
        summaries = [
            {
                "component_index": 0,
                "source_cluster_id": 10,
                "consensus_frequency": 1.0,
                "consensus_frequency_width": 0.1,
                "consensus_scale": 5.0,
                "n_member_bands": 3,
                "member_bands": ["A", "B", "C"],
            },
            {
                "component_index": 1,
                "source_cluster_id": 20,
                "consensus_frequency": 2.0,
                "consensus_frequency_width": 0.2,
                "consensus_scale": 5.0,
                "n_member_bands": 1,
                "member_bands": ["A"],
            },
        ]
        reconciled, diag = self._call(summaries, requested_n=1)
        # Higher n_member_bands kept
        self.assertAlmostEqual(reconciled[0]["consensus_frequency"], 1.0)
        self.assertEqual(diag["dropped_consensus_components"][0]["source_cluster_id"], 20)

    def test_m_less_than_n_uses_rejected_clusters(self):
        summaries = _make_n_component_summaries(1)  # M=1 accepted
        rejected = [
            {
                "cluster_id": 99,
                "center_frequency": 5.0,
                "frequency_scatter": 0.2,
                "n_member_bands": 1,
                "member_bands": ["B"],
                "accepted": False,
            }
        ]
        reconciled, diag = self._call(summaries, requested_n=2, rejected=rejected)
        self.assertEqual(len(reconciled), 2)
        self.assertEqual(diag["component_count_reconciliation_strategy"], "pad_with_fallback")
        self.assertEqual(diag["initialization_component_count"], 2)
        sources = [s["component_source"] for s in reconciled]
        self.assertIn("accepted_consensus", sources)
        self.assertIn("rejected_cluster_fallback", sources)
        fallback_entry = diag["fallback_initialization_components"]
        self.assertEqual(len(fallback_entry), 1)
        self.assertEqual(fallback_entry[0]["component_source"], "rejected_cluster_fallback")
        self.assertAlmostEqual(fallback_entry[0]["consensus_frequency"], 5.0)

    def test_m_less_than_n_uses_per_band_candidates(self):
        summaries = _make_n_component_summaries(1)  # M=1 accepted (freq ~1.0)
        band_cands = [
            {
                "band_name": "A",
                "component_candidates": [
                    {"frequency": 7.0, "peak_power": 3.0, "significant": True},
                    {"frequency": 1.01, "peak_power": 5.0, "significant": True},  # too close to accepted
                ],
            }
        ]
        reconciled, diag = self._call(summaries, requested_n=2, band_cands=band_cands)
        self.assertEqual(len(reconciled), 2)
        sources = [s["component_source"] for s in reconciled]
        self.assertIn("per_band_candidate_fallback", sources)
        fallback_entry = diag["fallback_initialization_components"]
        self.assertEqual(len(fallback_entry), 1)
        self.assertEqual(fallback_entry[0]["component_source"], "per_band_candidate_fallback")
        # freq=7.0 used (1.01 skipped as too close to accepted freq=1.0)
        self.assertAlmostEqual(fallback_entry[0]["consensus_frequency"], 7.0)

    def test_m_less_than_n_uses_broad_fallback(self):
        # No rejected clusters, no band candidates → broad fallback
        summaries = _make_n_component_summaries(2)  # freqs: 1.0, 2.0
        reconciled, diag = self._call(summaries, requested_n=4)
        self.assertEqual(len(reconciled), 4)
        sources = [s["component_source"] for s in reconciled]
        n_broad = sources.count("broad_fallback")
        self.assertEqual(n_broad, 2)
        # Broad fallback components have positive frequencies
        for s in reconciled:
            self.assertGreater(s["consensus_frequency"], 0)

    def test_reconciled_summaries_have_component_source(self):
        summaries = _make_n_component_summaries(2)
        reconciled, _ = self._call(summaries, requested_n=2)
        for s in reconciled:
            self.assertIn("component_source", s)
            self.assertEqual(s["component_source"], "accepted_consensus")

    def test_diagnostic_keys_present(self):
        summaries = _make_n_component_summaries(2)
        _, diag = self._call(summaries, requested_n=2)
        for key in [
            "requested_num_mixtures",
            "accepted_consensus_component_count",
            "initialization_component_count",
            "fitted_num_mixtures",
            "component_count_reconciliation_strategy",
            "dropped_consensus_components",
            "fallback_initialization_components",
        ]:
            self.assertIn(key, diag, msg=f"Missing key: {key}")


class TestConsensusMulticompFitRequestedNumMixtures(unittest.TestCase):
    """Integration tests: requested num_mixtures is preserved through the fit."""

    def setUp(self):
        self.lc = _make_minimal_multiband_lightcurve()
        self.lc.model = object()
        self.lc._model_pars = {}

    def _run_fit(self, num_mixtures, consensus, expected_n):
        """Run _consensus_multicomp_fit with mocks and check num_mixtures."""
        init_diag = _make_n_component_init_diagnostics(
            expected_n,
            freqs=[float(i + 1) * 1.0 for i in range(expected_n)],
            scales=[0.1 * float(i + 1) for i in range(expected_n)],
        )
        fitted_diag = _make_n_component_fitted_diagnostics(
            expected_n,
            freqs=[float(i + 1) * 1.0 for i in range(expected_n)],
            scales=[0.1 * float(i + 1) for i in range(expected_n)],
        )
        fit_kwargs = dict(
            model=None,
            constrain_consensus=False,
            _allow_existing_model_for_consensus=True,
        )
        if num_mixtures is not None:
            fit_kwargs["num_mixtures"] = num_mixtures

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
            return_value=consensus,
        ), mock.patch.object(
            self.lc,
            "_consensus_build_guess",
            return_value={},
        ), mock.patch.object(
            self.lc,
            "_consensus_collect_initialization_diagnostics",
            return_value=init_diag,
        ), mock.patch.object(
            self.lc,
            "_consensus_collect_fitted_mixture_diagnostics",
            return_value=fitted_diag,
        ), mock.patch.object(
            self.lc,
            "fit",
            return_value={"status": "ok"},
        ) as fit_mock:
            self.lc._consensus_multicomp_fit(**fit_kwargs)

        return fit_mock, self.lc.consensus_diagnostics

    def test_num_mixtures_2_exact_match(self):
        """M=2 accepted, N=2 requested → exact match."""
        consensus = _multicomponent_consensus()  # 2 components
        fit_mock, diag = self._run_fit(num_mixtures=2, consensus=consensus, expected_n=2)
        self.assertEqual(fit_mock.call_args.kwargs["num_mixtures"], 2)
        self.assertEqual(diag["fitted_num_mixtures"], 2)
        self.assertEqual(diag["requested_num_mixtures"], 2)
        self.assertEqual(diag["accepted_consensus_component_count"], 2)
        self.assertEqual(diag["initialization_component_count"], 2)
        self.assertEqual(diag["component_count_reconciliation_strategy"], "exact_match")

    def test_num_mixtures_1_drops_weakest(self):
        """M=2 accepted, N=1 requested → drop weakest component."""
        consensus = _multicomponent_consensus()  # 2 components
        fit_mock, diag = self._run_fit(num_mixtures=1, consensus=consensus, expected_n=1)
        self.assertEqual(fit_mock.call_args.kwargs["num_mixtures"], 1)
        self.assertEqual(diag["fitted_num_mixtures"], 1)
        self.assertEqual(diag["requested_num_mixtures"], 1)
        self.assertEqual(diag["accepted_consensus_component_count"], 2)
        self.assertEqual(diag["initialization_component_count"], 1)
        self.assertEqual(diag["component_count_reconciliation_strategy"], "drop_weakest")
        self.assertEqual(len(diag["dropped_consensus_components"]), 1)
        # Period summaries must match N=1
        self.assertEqual(len(diag["multicomponent_period_summaries"]), 1)

    def test_num_mixtures_3_pads_with_fallback(self):
        """M=2 accepted, N=3 requested → one fallback component added."""
        consensus = _multicomponent_consensus()  # 2 components
        fit_mock, diag = self._run_fit(num_mixtures=3, consensus=consensus, expected_n=3)
        self.assertEqual(fit_mock.call_args.kwargs["num_mixtures"], 3)
        self.assertEqual(diag["fitted_num_mixtures"], 3)
        self.assertEqual(diag["requested_num_mixtures"], 3)
        self.assertEqual(diag["accepted_consensus_component_count"], 2)
        self.assertEqual(diag["initialization_component_count"], 3)
        self.assertEqual(diag["component_count_reconciliation_strategy"], "pad_with_fallback")
        self.assertEqual(len(diag["fallback_initialization_components"]), 1)
        # Period summaries must match N=3
        self.assertEqual(len(diag["multicomponent_period_summaries"]), 3)

    def test_num_mixtures_4_pads_with_two_fallbacks(self):
        """M=2 accepted, N=4 requested → two fallback components added."""
        consensus = _multicomponent_consensus()  # 2 components
        fit_mock, diag = self._run_fit(num_mixtures=4, consensus=consensus, expected_n=4)
        self.assertEqual(fit_mock.call_args.kwargs["num_mixtures"], 4)
        self.assertEqual(diag["fitted_num_mixtures"], 4)
        self.assertEqual(diag["requested_num_mixtures"], 4)
        self.assertEqual(diag["accepted_consensus_component_count"], 2)
        self.assertEqual(diag["initialization_component_count"], 4)
        self.assertEqual(diag["component_count_reconciliation_strategy"], "pad_with_fallback")
        self.assertEqual(len(diag["fallback_initialization_components"]), 2)
        # Period summaries must match N=4
        self.assertEqual(len(diag["multicomponent_period_summaries"]), 4)

    def test_no_num_mixtures_uses_m(self):
        """No num_mixtures requested → M=2 used unchanged (legacy behavior)."""
        consensus = _multicomponent_consensus()  # 2 components
        fit_mock, diag = self._run_fit(num_mixtures=None, consensus=consensus, expected_n=2)
        self.assertEqual(fit_mock.call_args.kwargs["num_mixtures"], 2)
        self.assertEqual(diag["fitted_num_mixtures"], 2)
        self.assertIsNone(diag["requested_num_mixtures"])
        self.assertEqual(diag["accepted_consensus_component_count"], 2)
        self.assertEqual(diag["initialization_component_count"], 2)
        self.assertEqual(diag["component_count_reconciliation_strategy"], "exact_match")

    def test_period_summaries_have_component_source(self):
        """Every period summary entry must carry a component_source field."""
        consensus = _multicomponent_consensus()  # 2 components, M=N=2
        _, diag = self._run_fit(num_mixtures=2, consensus=consensus, expected_n=2)
        for entry in diag["multicomponent_period_summaries"]:
            self.assertIn("component_source", entry)
            self.assertEqual(entry["component_source"], "accepted_consensus")

    def test_period_summaries_fallback_component_source(self):
        """Fallback period summary entries must carry their provenance."""
        consensus = _multicomponent_consensus()  # M=2, request N=3
        _, diag = self._run_fit(num_mixtures=3, consensus=consensus, expected_n=3)
        summaries = diag["multicomponent_period_summaries"]
        self.assertEqual(len(summaries), 3)
        accepted_sources = [
            s["component_source"]
            for s in summaries
            if s["component_source"] == "accepted_consensus"
        ]
        fallback_sources = [
            s["component_source"]
            for s in summaries
            if s["component_source"] != "accepted_consensus"
        ]
        self.assertEqual(len(accepted_sources), 2)
        self.assertEqual(len(fallback_sources), 1)

    def test_consensus_frequencies_reflect_reconciled_n(self):
        """consensus_frequencies in diagnostics must have exactly N elements."""
        consensus = _multicomponent_consensus()  # M=2
        for n in [1, 2, 3, 4]:
            with self.subTest(n=n):
                _, diag = self._run_fit(num_mixtures=n, consensus=consensus, expected_n=n)
                self.assertEqual(len(diag["consensus_frequencies"]), n)
                self.assertEqual(len(diag["consensus_periods"]), n)
                self.assertEqual(len(diag["consensus_scales"]), n)


if __name__ == "__main__":
    unittest.main()
