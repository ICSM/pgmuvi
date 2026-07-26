"""Regression tests for independent temporal-consensus and GP-fit scopes."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_CSV = ROOT / "examples" / "data" / "10131+3049.csv"

DUPLICATE_WAVELENGTH = 0.6561154962791801
KELT_R3_0 = "KELT/OSN_Johnson.Cousins_R3_0"
KELT_R3_1 = "KELT/OSN_Johnson.Cousins_R3_1"


def _make_duplicate_channel_lightcurve() -> Lightcurve:
    rows = []
    flux = []
    error = []
    channels = []

    channel_specs = (
        ("KELT/OSN_Johnson.Cousins_R3_0", DUPLICATE_WAVELENGTH, 0.0),
        ("KELT/OSN_Johnson.Cousins_R3_1", DUPLICATE_WAVELENGTH, 0.3),
        ("GAIA/GAIA3.G", 0.6217711904068691, -0.2),
    )
    for channel_index, (channel, wavelength, phase) in enumerate(channel_specs):
        times = np.linspace(0.0, 40.0, 24) + 0.01 * channel_index
        values = 2.0 + np.sin(2.0 * np.pi * times / 8.0 + phase)
        for time, value in zip(times, values, strict=True):
            rows.append((time, wavelength))
            flux.append(value)
            error.append(0.05)
            channels.append(channel)

    return Lightcurve(
        torch.tensor(rows, dtype=torch.float64),
        torch.tensor(flux, dtype=torch.float64),
        yerr=torch.tensor(error, dtype=torch.float64),
        band=np.asarray(channels, dtype=np.str_),
        center_time=False,
        max_samples=None,
        max_samples_per_band=None,
    )


def _make_wavelength_only_lightcurve() -> Lightcurve:
    times = np.linspace(0.0, 40.0, 24)
    rows = []
    flux = []
    error = []
    for wavelength, phase in ((0.55, 0.0), (0.8, 0.4)):
        values = 2.0 + np.sin(
            2.0 * np.pi * times / 8.0 + phase
        )
        for time, value in zip(times, values, strict=True):
            rows.append((time, wavelength))
            flux.append(value)
            error.append(0.05)

    return Lightcurve(
        torch.tensor(rows, dtype=torch.float64),
        torch.tensor(flux, dtype=torch.float64),
        yerr=torch.tensor(error, dtype=torch.float64),
        band=None,
        center_time=False,
        max_samples=None,
        max_samples_per_band=None,
    )


class TestConsensusDuplicateWavelengthScope(unittest.TestCase):
    def _exercise_public_fit(
        self,
        *,
        fit_strategy,
        policy="first",
        selection=None,
    ):
        lc = _make_duplicate_channel_lightcurve()
        observations = {}

        def fake_fit_core(self, *args, fit_strategy=None, **kwargs):
            if fit_strategy in {"consensus", "consensus_multicomp"}:
                prepared = self._consensus_prepare_band_consensus_inputs(
                    min_points_per_band=2,
                    max_gap_fraction=1.0,
                    min_duty_cycle=0.0,
                    include_wavelengths=(
                        fit_strategy == "consensus_multicomp"
                    ),
                )
                observations["consensus_channels"] = list(
                    prepared["per_band_lc"]
                )
                observations["scope_during_consensus"] = (
                    self._consensus_data_scope_provenance()
                )
                return self.fit(fit_strategy=None)

            observations["training_channels"] = list(
                dict.fromkeys(
                    np.asarray(self.band, dtype=str).tolist()
                )
            )
            observations["training_rows"] = len(self._xdata_raw)
            return {"status": "mock-fit-complete"}

        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            new=fake_fit_core,
        ):
            result = lc.fit(
                fit_strategy=fit_strategy,
                duplicate_wavelength_policy=policy,
                duplicate_wavelength_selection=selection,
            )

        return lc, result, observations

    def test_standard_consensus_uses_duplicate_channels_and_fit_uses_first(
        self,
    ):
        lc, result, observed = self._exercise_public_fit(
            fit_strategy="consensus",
        )

        self.assertEqual(result, {"status": "mock-fit-complete"})
        self.assertEqual(
            observed["consensus_channels"],
            [
                "KELT/OSN_Johnson.Cousins_R3_0",
                "KELT/OSN_Johnson.Cousins_R3_1",
                "GAIA/GAIA3.G",
            ],
        )
        self.assertEqual(
            observed["training_channels"],
            [
                "KELT/OSN_Johnson.Cousins_R3_0",
                "GAIA/GAIA3.G",
            ],
        )
        scope = observed["scope_during_consensus"]
        self.assertEqual(scope["consensus_row_count"], 72)
        self.assertEqual(scope["gp_training_row_count"], 48)
        self.assertEqual(
            lc.temporal_consensus_data_scope[
                "consensus_observational_channels"
            ],
            observed["consensus_channels"],
        )
        self.assertEqual(
            lc.temporal_consensus_data_scope[
                "gp_training_observational_channels"
            ],
            observed["training_channels"],
        )
        self.assertFalse(
            hasattr(lc, "_active_temporal_consensus_source")
        )
        self.assertIsNotNone(
            getattr(
                lc,
                "_temporal_consensus_full_data_source",
                None,
            )
        )

    def test_explicit_selection_changes_fit_scope_not_consensus_scope(self):
        lc, _, observed = self._exercise_public_fit(
            fit_strategy="consensus",
            policy="select",
            selection="KELT/OSN_Johnson.Cousins_R3_1",
        )

        self.assertEqual(
            observed["consensus_channels"],
            [
                "KELT/OSN_Johnson.Cousins_R3_0",
                "KELT/OSN_Johnson.Cousins_R3_1",
                "GAIA/GAIA3.G",
            ],
        )
        self.assertEqual(
            observed["training_channels"],
            [
                "KELT/OSN_Johnson.Cousins_R3_1",
                "GAIA/GAIA3.G",
            ],
        )
        self.assertEqual(
            lc.duplicate_wavelength_channel_resolution["groups"][0][
                "selected_observational_channel"
            ],
            "KELT/OSN_Johnson.Cousins_R3_1",
        )

    def test_multicomponent_consensus_uses_all_channels(self):
        _, _, observed = self._exercise_public_fit(
            fit_strategy="consensus_multicomp",
        )

        self.assertEqual(
            observed["consensus_channels"],
            [
                "KELT/OSN_Johnson.Cousins_R3_0",
                "KELT/OSN_Johnson.Cousins_R3_1",
                "GAIA/GAIA3.G",
            ],
        )
        self.assertEqual(
            observed["training_channels"],
            [
                "KELT/OSN_Johnson.Cousins_R3_0",
                "GAIA/GAIA3.G",
            ],
        )
        groups = observed["scope_during_consensus"][
            "duplicate_physical_wavelength_groups"
        ]
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0]["observational_channels"],
            [
                "KELT/OSN_Johnson.Cousins_R3_0",
                "KELT/OSN_Johnson.Cousins_R3_1",
            ],
        )

    def test_resolved_copy_preserves_full_consensus_source(self):
        source = _make_duplicate_channel_lightcurve()
        with self.assertWarns(UserWarning):
            resolved = source.copy_with_duplicate_wavelength_channels(
                policy="select",
                selection="KELT/OSN_Johnson.Cousins_R3_1",
            )

        self.assertEqual(
            list(
                dict.fromkeys(
                    np.asarray(resolved.band, dtype=str).tolist()
                )
            ),
            [
                "KELT/OSN_Johnson.Cousins_R3_1",
                "GAIA/GAIA3.G",
            ],
        )
        prepared = resolved._consensus_prepare_band_consensus_inputs(
            min_points_per_band=2,
            max_gap_fraction=1.0,
            min_duty_cycle=0.0,
        )
        self.assertEqual(
            list(prepared["per_band_lc"]),
            [
                "KELT/OSN_Johnson.Cousins_R3_0",
                "KELT/OSN_Johnson.Cousins_R3_1",
                "GAIA/GAIA3.G",
            ],
        )
        self.assertEqual(len(source._xdata_raw), 72)

    def test_non_consensus_fit_still_uses_resolved_training_scope(self):
        lc = _make_duplicate_channel_lightcurve()
        observations = {}

        def fake_fit_core(self, *args, **kwargs):
            observations["channels"] = list(
                dict.fromkeys(
                    np.asarray(self.band, dtype=str).tolist()
                )
            )
            return None

        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            new=fake_fit_core,
        ):
            lc.fit()

        self.assertEqual(
            observations["channels"],
            [
                "KELT/OSN_Johnson.Cousins_R3_0",
                "GAIA/GAIA3.G",
            ],
        )
        self.assertFalse(
            hasattr(lc, "temporal_consensus_data_scope")
        )

    def test_all_policy_remains_guarded_for_gp_training(self):
        lc = _make_duplicate_channel_lightcurve()
        with self.assertRaises(NotImplementedError):
            lc.fit(
                fit_strategy="consensus",
                duplicate_wavelength_policy="all",
            )


    def test_consensus_falls_back_to_physical_wavelength_without_labels(
        self,
    ):
        lc = _make_wavelength_only_lightcurve()
        observed = {}

        def fake_fit_core(self, *args, fit_strategy=None, **kwargs):
            if fit_strategy == "consensus":
                prepared = self._consensus_prepare_band_consensus_inputs(
                    min_points_per_band=2,
                    max_gap_fraction=1.0,
                    min_duty_cycle=0.0,
                    include_wavelengths=True,
                )
                observed["groups"] = {
                    label: len(group._xdata_raw)
                    for label, group in prepared["per_band_lc"].items()
                }
                observed["wavelengths"] = prepared["band_to_wavelength"]
                return self.fit(fit_strategy=None)
            return {"status": "mock-fit-complete"}

        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            new=fake_fit_core,
        ):
            result = lc.fit(fit_strategy="consensus")

        expected_labels = [
            Lightcurve._consensus_physical_wavelength_label(value)
            for value in (0.55, 0.8)
        ]
        self.assertEqual(result, {"status": "mock-fit-complete"})
        self.assertEqual(list(observed["groups"]), expected_labels)
        self.assertEqual(
            list(observed["groups"].values()),
            [24, 24],
        )
        self.assertEqual(
            observed["wavelengths"],
            {
                expected_labels[0]: 0.55,
                expected_labels[1]: 0.8,
            },
        )
        scope = lc.temporal_consensus_data_scope
        self.assertEqual(
            scope["consensus_scope"],
            "all_eligible_physical_wavelength_groups",
        )
        self.assertEqual(
            scope["consensus_grouping_key"],
            "physical_wavelength",
        )
        self.assertEqual(
            scope["consensus_group_labels"],
            expected_labels,
        )
        self.assertEqual(
            scope["consensus_observational_channels"],
            [],
        )
        self.assertEqual(
            scope["gp_training_group_labels"],
            expected_labels,
        )

    def test_label_free_gp_validation_group_selection_preserves_transforms(
        self,
    ):
        base = _make_wavelength_only_lightcurve()
        transformed = Lightcurve(
            base._xdata_raw.detach().clone(),
            base._ydata_raw.detach().clone(),
            yerr=base._yerr_raw.detach().clone(),
            xtransform="time_center",
            ytransform="zscore",
            center_time="auto",
            band=None,
            max_samples=None,
            max_samples_per_band=None,
        )

        for wavelength in (0.55, 0.8):
            label = Lightcurve._consensus_physical_wavelength_label(
                wavelength
            )
            selected = (
                transformed._consensus_select_group_for_gp_validation(
                    label
                )
            )
            self.assertEqual(len(selected._xdata_raw), 24)
            self.assertIsNone(selected.band)
            self.assertTrue(
                torch.all(
                    selected._xdata_raw[:, 1]
                    == torch.tensor(
                        wavelength,
                        dtype=selected._xdata_raw.dtype,
                    )
                )
            )
            self.assertIs(
                type(selected.xtransform),
                type(transformed.xtransform),
            )
            self.assertIs(
                type(selected.ytransform),
                type(transformed.ytransform),
            )
            self.assertIsNot(
                selected.xtransform,
                transformed.xtransform,
            )
            self.assertIsNot(
                selected.ytransform,
                transformed.ytransform,
            )

    def test_representative_dataset_keeps_both_r3_consensus_votes(
        self,
    ):
        lc = Lightcurve.from_csv(
            REPRESENTATIVE_CSV,
            max_samples=None,
            max_samples_per_band=None,
        )
        observed = {}

        def fake_fit_core(self, *args, fit_strategy=None, **kwargs):
            if fit_strategy == "consensus":
                consensus_source = self._consensus_data_source()
                consensus_groups = list(
                    consensus_source._consensus_iter_band_lightcurves()
                )
                observed["consensus_channels"] = [
                    label
                    for label, _ in consensus_groups
                ]
                observed["consensus_group_rows"] = {
                    label: len(group._xdata_raw)
                    for label, group in consensus_groups
                }
                observed["scope"] = (
                    self._consensus_data_scope_provenance()
                )
                return self.fit(fit_strategy=None)

            observed["training_channels"] = list(
                dict.fromkeys(
                    np.asarray(self.band, dtype=str).tolist()
                )
            )
            return {"status": "mock-fit-complete"}

        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            new=fake_fit_core,
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "Selected .*R3_0",
            ):
                result = lc.fit(fit_strategy="consensus")

        self.assertEqual(result, {"status": "mock-fit-complete"})
        self.assertIn(KELT_R3_0, observed["consensus_channels"])
        self.assertIn(KELT_R3_1, observed["consensus_channels"])
        self.assertEqual(
            observed["consensus_group_rows"][KELT_R3_0],
            5706,
        )
        self.assertEqual(
            observed["consensus_group_rows"][KELT_R3_1],
            3636,
        )
        self.assertNotIn(
            9342,
            observed["consensus_group_rows"].values(),
        )
        self.assertIn(KELT_R3_0, observed["training_channels"])
        self.assertNotIn(KELT_R3_1, observed["training_channels"])
        self.assertEqual(
            observed["scope"]["consensus_grouping_key"],
            "observational_channel",
        )
        self.assertEqual(
            observed["scope"]["consensus_row_count"],
            10815,
        )
        self.assertEqual(
            observed["scope"]["gp_training_row_count"],
            7179,
        )
        self.assertEqual(
            observed["scope"][
                "duplicate_physical_wavelength_groups"
            ][0]["observational_channels"],
            [KELT_R3_0, KELT_R3_1],
        )

    def test_consensus_copy_preserves_independent_transforms(self):
        base = _make_duplicate_channel_lightcurve()
        transformed = Lightcurve(
            base._xdata_raw.detach().clone(),
            base._ydata_raw.detach().clone(),
            yerr=base._yerr_raw.detach().clone(),
            xtransform="time_center",
            ytransform="zscore",
            center_time="auto",
            band=np.array(base.band, dtype=np.str_, copy=True),
            max_samples=None,
            max_samples_per_band=None,
        )

        source = transformed._copy_for_temporal_consensus()

        self.assertIs(type(source.xtransform), type(transformed.xtransform))
        self.assertIs(type(source.ytransform), type(transformed.ytransform))
        self.assertIsNot(source.xtransform, transformed.xtransform)
        self.assertIsNot(source.ytransform, transformed.ytransform)
        self.assertEqual(
            source._recenter_time_after_data_selection,
            transformed._recenter_time_after_data_selection,
        )
        self.assertTrue(
            torch.equal(source._xdata_raw, transformed._xdata_raw)
        )
        self.assertTrue(
            torch.equal(source._ydata_raw, transformed._ydata_raw)
        )

    def test_later_non_consensus_fit_does_not_reuse_scope_provenance(self):
        lc = _make_duplicate_channel_lightcurve()
        observed_contexts = []

        def fake_fit_core(self, *args, fit_strategy=None, **kwargs):
            observed_contexts.append(
                self._fit_history_context.get(
                    "temporal_consensus_data_scope"
                )
            )
            if fit_strategy == "consensus":
                return self.fit(fit_strategy=None)
            return {"status": "mock-fit-complete"}

        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            new=fake_fit_core,
        ):
            with self.assertWarns(UserWarning):
                lc.fit(fit_strategy="consensus")
            lc.fit()

        self.assertIsNotNone(observed_contexts[0])
        self.assertIsNotNone(observed_contexts[1])
        self.assertIsNone(observed_contexts[-1])
        self.assertIsNotNone(lc.temporal_consensus_data_scope)

    def test_consensus_exception_cleans_active_source(self):
        lc = _make_duplicate_channel_lightcurve()

        def failing_fit_core(self, *args, **kwargs):
            self.assert_active_source_for_test = hasattr(
                self,
                "_active_temporal_consensus_source",
            )
            raise RuntimeError("synthetic consensus failure")

        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            new=failing_fit_core,
        ):
            with self.assertWarns(UserWarning):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "synthetic consensus failure",
                ):
                    lc.fit(fit_strategy="consensus")

        self.assertTrue(lc.assert_active_source_for_test)
        self.assertFalse(
            hasattr(lc, "_active_temporal_consensus_source")
        )
        self.assertEqual(lc._fit_history_context, {})
        self.assertIsNotNone(lc.temporal_consensus_data_scope)


if __name__ == "__main__":
    unittest.main()
