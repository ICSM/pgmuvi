import unittest

import gpytorch
import torch

from pgmuvi.gps import spectral_mixture_parameter_schema
from pgmuvi.lightcurve import Lightcurve
from pgmuvi.parameter_specs import ConstraintStrategy, GuessStrategy


class TestSpectralMixtureArdParameterWorkflow(unittest.TestCase):
    @staticmethod
    def _lightcurve(*, xtransform=None):
        rows = []
        fluxes = []
        for wavelength in (1.0, 2.0, 5.0):
            for time in (0.0, 1.0, 2.0, 100.0):
                rows.append([time, wavelength])
                fluxes.append(wavelength + 0.01 * time)
        kwargs = {"center_time": False} if xtransform is None else {}
        return Lightcurve(
            torch.tensor(rows),
            torch.tensor(fluxes),
            xtransform=xtransform,
            **kwargs,
        )

    def test_2d_schema_uses_dimension_aware_strategy(self):
        schema = spectral_mixture_parameter_schema(
            num_mixtures=2,
            ard_num_dims=2,
        )
        for name in ("covar_module.mixture_means", "covar_module.mixture_scales"):
            spec = schema[name]
            self.assertIs(
                spec.guess_strategy,
                GuessStrategy.DIMENSION_AWARE_SM_ARD,
            )
            self.assertIs(
                spec.constraint_strategy,
                ConstraintStrategy.DIMENSION_AWARE_SM_ARD,
            )
            self.assertEqual(spec.metadata["coordinate_order"][0], "temporal_frequency")
            self.assertEqual(spec.metadata["coordinate_order"][1], "wavelength_frequency")

    def test_1d_schema_keeps_legacy_strategies(self):
        schema = spectral_mixture_parameter_schema(
            num_mixtures=2,
            ard_num_dims=1,
        )
        self.assertIs(
            schema["covar_module.mixture_means"].guess_strategy,
            GuessStrategy.CONSENSUS_FREQUENCY,
        )
        self.assertIs(
            schema["covar_module.mixture_scales"].guess_strategy,
            GuessStrategy.DEFAULT,
        )

    def test_default_constraints_have_broadcast_ard_shape(self):
        lc = self._lightcurve()
        lc.set_model("2D", num_mixtures=2)
        lc.set_default_constraints()

        means = lc.model.covar_module.raw_mixture_means_constraint
        scales = lc.model.covar_module.raw_mixture_scales_constraint
        self.assertIsInstance(means, gpytorch.constraints.Interval)
        self.assertEqual(tuple(means.lower_bound.shape), (1, 1, 2))
        self.assertEqual(tuple(scales.lower_bound.shape), (1, 1, 2))
        self.assertNotEqual(
            float(means.lower_bound[0, 0, 0]),
            float(means.lower_bound[0, 0, 1]),
        )
        self.assertNotEqual(
            float(scales.upper_bound[0, 0, 0]),
            float(scales.upper_bound[0, 0, 1]),
        )

    def test_default_constraints_do_not_widen_existing_tensor_interval(self):
        lc = self._lightcurve()
        lc.set_model("2D", num_mixtures=1)
        existing = gpytorch.constraints.Interval(
            torch.tensor([[[0.02, 0.02]]]),
            torch.tensor([[[0.2, 0.2]]]),
        )
        lc.model.covar_module.register_constraint(
            "raw_mixture_means",
            existing,
        )
        lc.set_default_constraints()
        effective = lc.model.covar_module.raw_mixture_means_constraint

        self.assertAlmostEqual(
            float(effective.lower_bound[0, 0, 0]), 0.02, places=6
        )
        self.assertAlmostEqual(
            float(effective.lower_bound[0, 0, 1]), 0.02, places=6
        )
        self.assertAlmostEqual(
            float(effective.upper_bound[0, 0, 0]), 0.2, places=6
        )
        self.assertAlmostEqual(
            float(effective.upper_bound[0, 0, 1]), 0.2, places=6
        )

    def test_parameter_workflow_applies_values_and_provenance(self):
        lc = self._lightcurve()
        lc.set_model("2D", num_mixtures=2)
        lc.set_default_constraints()
        result = lc._apply_parameter_workflow_estimates()

        for name in (
            "covar_module.mixture_means",
            "covar_module.mixture_scales",
        ):
            self.assertTrue(result[name]["value"])
            self.assertTrue(result[name]["constraint"])
            provenance = result[name]["spectral_mixture_ard_provenance"]
            self.assertEqual(
                provenance["diagnostics"]["coordinate_order"],
                ["temporal_frequency", "wavelength_frequency"],
            )
            self.assertEqual(
                provenance["diagnostics"]["constraint_shape"],
                [1, 1, 2],
            )

        report = lc.get_parameter_workflow_report()
        report_names = {entry["parameter"]: entry for entry in report["applied"]}
        self.assertIn(
            "spectral_mixture_ard_provenance",
            report_names["covar_module.mixture_means"],
        )
        self.assertIn(
            "spectral_mixture_ard_constraint_provenance",
            report,
        )

    def test_lpv_period_limit_changes_only_temporal_mean_upper(self):
        unconstrained = self._lightcurve()
        unconstrained.set_model("2D", num_mixtures=1)
        unconstrained.set_default_constraints()
        base = unconstrained.model.covar_module.raw_mixture_means_constraint

        constrained = self._lightcurve()
        constrained.set_model("2D", num_mixtures=1)
        constrained.set_default_constraints(constraint_set="LPV")
        lpv = constrained.model.covar_module.raw_mixture_means_constraint

        self.assertLessEqual(
            float(lpv.upper_bound[0, 0, 0]),
            float(base.upper_bound[0, 0, 0]),
        )
        self.assertEqual(
            float(lpv.lower_bound[0, 0, 1]),
            float(base.lower_bound[0, 0, 1]),
        )
        self.assertEqual(
            float(lpv.upper_bound[0, 0, 1]),
            float(base.upper_bound[0, 0, 1]),
        )

    def test_minmax_transform_scales_dimensions_independently(self):
        lc = self._lightcurve(xtransform="minmax")
        lc.set_model("2D", num_mixtures=1)
        context = lc._build_parameter_estimation_context()
        diagnostics = context.spectral_mixture_ard_diagnostics
        raw_lower = diagnostics["raw_coordinate"]["mixture_means"][
            "constraint_lower"
        ][0][0]
        model_lower = diagnostics["model_coordinate"]["mixture_means"][
            "constraint_lower"
        ][0][0]

        self.assertAlmostEqual(model_lower[0], 100.0 * raw_lower[0])
        self.assertAlmostEqual(model_lower[1], 4.0 * raw_lower[1])

    def test_set_hypers_transforms_last_ard_axis_independently(self):
        lc = self._lightcurve(xtransform="minmax")
        lc.set_model("2D", num_mixtures=1)
        lc.set_default_constraints()
        raw = torch.tensor([[[0.02, 0.1]]])
        lc.set_hypers({"covar_module.mixture_means": raw})
        value = lc.model.covar_module.mixture_means.detach()

        self.assertAlmostEqual(float(value[0, 0, 0]), 2.0, places=5)
        self.assertAlmostEqual(float(value[0, 0, 1]), 0.4, places=5)

    def test_consensus_updates_only_temporal_ard_bounds(self):
        lc = self._lightcurve(xtransform="minmax")
        lc.set_model("2D", num_mixtures=1)
        lc.set_default_constraints()

        means_before = lc.model.covar_module.raw_mixture_means_constraint
        scales_before = lc.model.covar_module.raw_mixture_scales_constraint
        means_wavelength_before = (
            means_before.lower_bound[..., 1:].clone(),
            means_before.upper_bound[..., 1:].clone(),
        )
        scales_wavelength_before = (
            scales_before.lower_bound[..., 1:].clone(),
            scales_before.upper_bound[..., 1:].clone(),
        )

        means_provenance = lc._consensus_apply_temporal_sm_constraint(
            "covar_module.mixture_means",
            0.02,
            0.1,
        )
        scales_provenance = lc._consensus_apply_temporal_sm_constraint(
            "covar_module.mixture_scales",
            1.0e-4,
            0.05,
        )

        means_after = lc.model.covar_module.raw_mixture_means_constraint
        scales_after = lc.model.covar_module.raw_mixture_scales_constraint
        self.assertAlmostEqual(float(means_after.lower_bound[..., 0]), 2.0)
        self.assertAlmostEqual(float(means_after.upper_bound[..., 0]), 10.0)
        self.assertAlmostEqual(float(scales_after.lower_bound[..., 0]), 0.01)
        self.assertAlmostEqual(float(scales_after.upper_bound[..., 0]), 5.0)
        self.assertTrue(
            torch.equal(
                means_after.lower_bound[..., 1:],
                means_wavelength_before[0],
            )
        )
        self.assertTrue(
            torch.equal(
                means_after.upper_bound[..., 1:],
                means_wavelength_before[1],
            )
        )
        self.assertTrue(
            torch.equal(
                scales_after.lower_bound[..., 1:],
                scales_wavelength_before[0],
            )
        )
        self.assertTrue(
            torch.equal(
                scales_after.upper_bound[..., 1:],
                scales_wavelength_before[1],
            )
        )
        self.assertEqual(means_provenance["ard_scope"], "temporal_only")
        self.assertTrue(means_provenance["wavelength_bounds_preserved"])
        self.assertTrue(scales_provenance["wavelength_bounds_preserved"])

    def test_single_wavelength_does_not_reuse_temporal_bound(self):
        rows = torch.tensor(
            [[0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [4.0, 2.0]]
        )
        lc = Lightcurve(rows, torch.tensor([1.0, 1.1, 0.9, 1.0]), center_time=False)
        lc.set_model("2D", num_mixtures=1)
        lc.set_default_constraints()
        constraint = lc.model.covar_module.raw_mixture_means_constraint

        self.assertEqual(tuple(constraint.lower_bound.shape), (1, 1, 2))
        self.assertNotEqual(
            float(constraint.lower_bound[0, 0, 0]),
            float(constraint.lower_bound[0, 0, 1]),
        )


if __name__ == "__main__":
    unittest.main()
