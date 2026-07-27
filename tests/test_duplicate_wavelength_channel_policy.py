"""Duplicate physical-wavelength observational-channel fit policy tests."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock
import warnings

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_CSV = ROOT / "examples" / "data" / "10131+3049.csv"
KELT_WAVELENGTH = 0.6561154962791801
KELT_R3_0 = "KELT/OSN_Johnson.Cousins_R3_0"
KELT_R3_1 = "KELT/OSN_Johnson.Cousins_R3_1"


def _make_duplicate_group_lightcurve(two_groups: bool = False) -> Lightcurve:
    times = torch.arange(12, dtype=torch.get_default_dtype())
    wavelengths = torch.tensor(
        [1.0] * 3 + [1.0] * 2 + [2.0] * 3 + [3.0] * 4,
        dtype=torch.get_default_dtype(),
    )
    bands = np.asarray(
        ["A"] * 3 + ["B"] * 2 + ["C"] * 3 + ["D"] * 4,
        dtype=np.str_,
    )
    if two_groups:
        wavelengths[-4:] = 2.0
    xdata = torch.stack((times, wavelengths), dim=1)
    ydata = torch.sin(times)
    yerr = torch.full_like(ydata, 0.1)
    return Lightcurve(
        xdata,
        ydata,
        yerr=yerr,
        band=bands,
        max_samples=None,
    )


def _make_pair_and_triplet_lightcurve() -> Lightcurve:
    """Return arbitrary channel names with one pair and one triplet."""
    times = torch.arange(18, dtype=torch.get_default_dtype())
    wavelengths = torch.tensor(
        [1.0] * 2
        + [1.0] * 3
        + [2.0] * 2
        + [2.0] * 2
        + [2.0] * 4
        + [3.0] * 5,
        dtype=torch.get_default_dtype(),
    )
    bands = np.asarray(
        ["survey_a/channel_alpha"] * 2
        + ["survey_b/channel_beta"] * 3
        + ["instrument_c/stream_1"] * 2
        + ["instrument_d/stream_2"] * 2
        + ["instrument_e/stream_3"] * 4
        + ["unique/channel"] * 5,
        dtype=np.str_,
    )
    xdata = torch.stack((times, wavelengths), dim=1)
    ydata = 1.0 + torch.sin(times / 3.0)
    yerr = torch.full_like(ydata, 0.1)
    return Lightcurve(
        xdata,
        ydata,
        yerr=yerr,
        band=bands,
        max_samples=None,
    )


class TestDuplicateWavelengthDiscovery(unittest.TestCase):
    def test_detects_channels_in_first_row_order(self):
        lc = _make_duplicate_group_lightcurve()
        groups = lc._duplicate_physical_wavelength_channel_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["physical_wavelength"], 1.0)
        self.assertEqual(groups[0]["observational_channels"], ["A", "B"])
        self.assertEqual(groups[0]["row_counts"], {"A": 3, "B": 2})

    def test_no_band_metadata_has_no_groups(self):
        lc = _make_duplicate_group_lightcurve()
        lc.band = None
        self.assertEqual(
            lc._duplicate_physical_wavelength_channel_groups(),
            [],
        )

    def test_public_report_detects_pair_and_triplet_generically(self):
        lc = _make_pair_and_triplet_lightcurve()
        multiplets = lc.duplicate_physical_wavelength_multiplets()

        self.assertEqual(len(multiplets), 2)
        self.assertEqual(multiplets[0]["physical_wavelength"], 1.0)
        self.assertEqual(multiplets[0]["multiplet_size"], 2)
        self.assertEqual(
            multiplets[0]["observational_channels"],
            ["survey_a/channel_alpha", "survey_b/channel_beta"],
        )
        self.assertEqual(
            multiplets[0]["row_counts"],
            {
                "survey_a/channel_alpha": 2,
                "survey_b/channel_beta": 3,
            },
        )

        self.assertEqual(multiplets[1]["physical_wavelength"], 2.0)
        self.assertEqual(multiplets[1]["multiplet_size"], 3)
        self.assertEqual(
            multiplets[1]["observational_channels"],
            [
                "instrument_c/stream_1",
                "instrument_d/stream_2",
                "instrument_e/stream_3",
            ],
        )
        self.assertEqual(
            multiplets[1]["row_counts"],
            {
                "instrument_c/stream_1": 2,
                "instrument_d/stream_2": 2,
                "instrument_e/stream_3": 4,
            },
        )

        multiplets[0]["observational_channels"].append("mutated")
        self.assertNotIn(
            "mutated",
            lc.duplicate_physical_wavelength_multiplets()[0][
                "observational_channels"
            ],
        )


class TestDuplicateWavelengthPolicy(unittest.TestCase):
    def test_default_requires_explicit_policy_and_preserves_data(self):
        lc = _make_duplicate_group_lightcurve()
        original_x = lc.xdata.clone()
        original_y = lc.ydata.clone()
        original_band = lc.band.copy()

        with self.assertRaisesRegex(
            ValueError,
            "Choose the GP training input explicitly",
        ):
            lc._apply_duplicate_wavelength_channel_policy()

        torch.testing.assert_close(lc.xdata, original_x)
        torch.testing.assert_close(lc.ydata, original_y)
        np.testing.assert_array_equal(lc.band, original_band)

    def test_explicit_first_retains_first_channel_and_warns(self):
        lc = _make_duplicate_group_lightcurve()
        with self.assertWarnsRegex(
            UserWarning,
            "Selected A .* ignored B \\(2 rows\\)",
        ):
            provenance = lc._apply_duplicate_wavelength_channel_policy(
                policy="first",
            )
        self.assertEqual(len(lc.xdata), 10)
        self.assertNotIn("B", set(lc.band.tolist()))
        self.assertIn("A", set(lc.band.tolist()))
        self.assertTrue(provenance["applied"])
        self.assertEqual(provenance["retained_row_count_before"], 12)
        self.assertEqual(provenance["retained_row_count_after"], 10)

    def test_select_string_for_one_group(self):
        lc = _make_duplicate_group_lightcurve()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            provenance = lc._apply_duplicate_wavelength_channel_policy(
                policy="select",
                selection="B",
            )
        self.assertNotIn("A", set(lc.band.tolist()))
        self.assertIn("B", set(lc.band.tolist()))
        self.assertEqual(
            provenance["groups"][0]["selected_observational_channel"],
            "B",
        )

    def test_select_mapping_for_multiple_groups(self):
        lc = _make_duplicate_group_lightcurve(two_groups=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc._apply_duplicate_wavelength_channel_policy(
                policy="select",
                selection={1.0: "B", 2.0: "D"},
            )
        self.assertEqual(set(lc.band.tolist()), {"B", "D"})

    def test_select_mapping_resolves_pair_and_triplet_before_fit(self):
        lc = _make_pair_and_triplet_lightcurve()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            provenance = lc._apply_duplicate_wavelength_channel_policy(
                policy="select",
                selection={
                    1.0: "survey_b/channel_beta",
                    2.0: "instrument_e/stream_3",
                },
            )

        self.assertEqual(
            set(lc.band.tolist()),
            {
                "survey_b/channel_beta",
                "instrument_e/stream_3",
                "unique/channel",
            },
        )
        self.assertEqual(len(lc.xdata), 12)
        self.assertEqual(
            [
                row["selected_observational_channel"]
                for row in provenance["groups"]
            ],
            ["survey_b/channel_beta", "instrument_e/stream_3"],
        )
        self.assertEqual(
            [row["selected_row_count"] for row in provenance["groups"]],
            [3, 4],
        )

    def test_string_is_rejected_for_multiple_groups(self):
        lc = _make_duplicate_group_lightcurve(two_groups=True)
        with self.assertRaisesRegex(ValueError, "only unambiguous"):
            lc._apply_duplicate_wavelength_channel_policy(
                policy="select",
                selection="B",
            )

    def test_missing_mapping_group_is_rejected(self):
        lc = _make_duplicate_group_lightcurve(two_groups=True)
        with self.assertRaisesRegex(ValueError, "Missing"):
            lc._apply_duplicate_wavelength_channel_policy(
                policy="select",
                selection={1.0: "A"},
            )

    def test_unknown_channel_is_rejected(self):
        lc = _make_duplicate_group_lightcurve()
        with self.assertRaisesRegex(ValueError, "not available"):
            lc._apply_duplicate_wavelength_channel_policy(
                policy="select",
                selection="missing",
            )

    def test_all_mode_fails_before_fit(self):
        lc = _make_duplicate_group_lightcurve()
        with self.assertRaisesRegex(
            NotImplementedError,
            "scientifically validated instrument-channel calibration",
        ):
            lc._apply_duplicate_wavelength_channel_policy(policy="all")

    def test_selection_with_first_policy_is_rejected(self):
        lc = _make_duplicate_group_lightcurve()
        with self.assertRaisesRegex(ValueError, "only valid"):
            lc._apply_duplicate_wavelength_channel_policy(
                policy="first",
                selection="A",
            )


class TestDuplicateWavelengthConstructorSubsampling(unittest.TestCase):
    def test_subsamples_each_observational_channel_independently(self):
        times = torch.arange(16, dtype=torch.get_default_dtype())
        wavelengths = torch.tensor(
            [1.0] * 6 + [1.0] * 5 + [2.0] * 5,
            dtype=torch.get_default_dtype(),
        )
        bands = np.asarray(
            ["A"] * 6 + ["B"] * 5 + ["C"] * 5,
            dtype=np.str_,
        )
        xdata = torch.stack((times, wavelengths), dim=1)
        ydata = torch.sin(times)

        with self.assertWarnsRegex(
            UserWarning,
            "observational channels.*channel=A.*channel=B.*channel=C",
        ):
            lc = Lightcurve(
                xdata,
                ydata,
                band=bands,
                max_samples=None,
                max_samples_per_band=3,
                subsample_seed=17,
            )

        retained_counts = {
            channel: int(np.count_nonzero(lc.band == channel))
            for channel in ("A", "B", "C")
        }
        self.assertEqual(retained_counts, {"A": 3, "B": 3, "C": 3})
        self.assertEqual(len(lc.xdata), 9)

        groups = lc._duplicate_physical_wavelength_channel_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["observational_channels"], ["A", "B"])

    def test_first_policy_uses_pre_subsampling_input_order(self):
        times = torch.tensor(
            [100.0, 101.0, 102.0, 103.0, 0.0, 1.0, 2.0, 3.0],
            dtype=torch.get_default_dtype(),
        )
        wavelengths = torch.ones_like(times)
        bands = np.asarray(
            ["B"] * 4 + ["A"] * 4,
            dtype=np.str_,
        )
        xdata = torch.stack((times, wavelengths), dim=1)
        ydata = torch.sin(times)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc = Lightcurve(
                xdata,
                ydata,
                band=bands,
                max_samples=None,
                max_samples_per_band=2,
                subsample_seed=17,
            )

        groups = lc._duplicate_physical_wavelength_channel_groups()
        self.assertEqual(groups[0]["observational_channels"], ["B", "A"])

        with self.assertWarnsRegex(UserWarning, "Selected B"):
            provenance = lc._apply_duplicate_wavelength_channel_policy(
                policy="first",
            )

        self.assertEqual(
            provenance["groups"][0]["selected_observational_channel"],
            "B",
        )
        self.assertEqual(set(lc.band.tolist()), {"B"})

    def test_wavelength_fallback_remains_available_without_labels(self):
        times = torch.arange(10, dtype=torch.get_default_dtype())
        wavelengths = torch.tensor(
            [1.0] * 5 + [2.0] * 5,
            dtype=torch.get_default_dtype(),
        )
        xdata = torch.stack((times, wavelengths), dim=1)
        ydata = torch.sin(times)

        with self.assertWarnsRegex(
            UserWarning,
            "physical wavelengths.*λ=1.0.*λ=2.0",
        ):
            lc = Lightcurve(
                xdata,
                ydata,
                max_samples=None,
                max_samples_per_band=3,
                subsample_seed=17,
            )

        unique, counts = torch.unique(
            lc.xdata[:, 1],
            return_counts=True,
        )
        self.assertEqual(unique.tolist(), [1.0, 2.0])
        self.assertEqual(counts.tolist(), [3, 3])


class TestDuplicateWavelengthNonMutatingCopy(unittest.TestCase):
    def test_default_copy_leaves_source_unchanged(self):
        source = _make_duplicate_group_lightcurve()
        original_x = source.xdata.clone()
        original_y = source.ydata.clone()
        original_band = source.band.copy()

        with self.assertWarnsRegex(UserWarning, "Selected A"):
            resolved = source.copy_with_duplicate_wavelength_channels(
                policy="first",
            )

        self.assertIsNot(resolved, source)
        torch.testing.assert_close(source.xdata, original_x)
        torch.testing.assert_close(source.ydata, original_y)
        np.testing.assert_array_equal(source.band, original_band)
        self.assertEqual(len(source.xdata), 12)
        self.assertEqual(len(resolved.xdata), 10)
        self.assertNotIn("B", set(resolved.band.tolist()))

    def test_alternative_copies_are_independent(self):
        source = _make_duplicate_group_lightcurve()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = source.copy_with_duplicate_wavelength_channels(
                policy="first",
            )
            second = source.copy_with_duplicate_wavelength_channels(
                policy="select",
                selection="B",
            )

        self.assertEqual(set(source.band.tolist()), {"A", "B", "C", "D"})
        self.assertEqual(set(first.band.tolist()), {"A", "C", "D"})
        self.assertEqual(set(second.band.tolist()), {"B", "C", "D"})
        self.assertIsNot(first.xtransform, second.xtransform)
        if first.ytransform is not None or second.ytransform is not None:
            self.assertIsNot(first.ytransform, second.ytransform)

    def test_fit_preserves_copy_resolution_provenance(self):
        source = _make_duplicate_group_lightcurve()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resolved = source.copy_with_duplicate_wavelength_channels(
                policy="select",
                selection="B",
            )

        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            autospec=True,
            return_value="fit-result",
        ):
            result = resolved.fit(model="2D")

        self.assertEqual(result, "fit-result")
        self.assertEqual(
            resolved.duplicate_wavelength_channel_resolution["groups"][0][
                "selected_observational_channel"
            ],
            "B",
        )
        self.assertEqual(
            resolved.fit_history[0]["notes"][
                "duplicate_wavelength_channel_resolution"
            ]["groups"][0]["selected_observational_channel"],
            "B",
        )
        self.assertEqual(set(source.band.tolist()), {"A", "B", "C", "D"})


class TestDuplicateWavelengthFitIntegration(unittest.TestCase):
    @staticmethod
    def _fit_without_training(lc: Lightcurve, **fit_kwargs):
        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            autospec=True,
            return_value="fit-result",
        ) as fit_core:
            result = lc.fit(**fit_kwargs)
        return result, fit_core

    def test_fit_default_requires_explicit_policy_and_never_calls_core(self):
        lc = _make_duplicate_group_lightcurve()
        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            autospec=True,
        ) as fit_core:
            with self.assertRaisesRegex(
                ValueError,
                "Choose the GP training input explicitly",
            ):
                lc.fit(model="2D")
        fit_core.assert_not_called()
        self.assertEqual(len(lc.xdata), 12)
        self.assertIn("B", set(lc.band.tolist()))

    def test_fit_explicit_first_resolves_before_core(self):
        lc = _make_duplicate_group_lightcurve()
        with self.assertWarnsRegex(UserWarning, "ignored B"):
            result, fit_core = self._fit_without_training(
                lc,
                model="2D",
                duplicate_wavelength_policy="first",
            )
        self.assertEqual(result, "fit-result")
        self.assertEqual(len(lc.xdata), 10)
        self.assertNotIn("B", set(lc.band.tolist()))
        fit_core.assert_called_once()

    def test_fit_explicit_select_resolves_before_core(self):
        lc = _make_duplicate_group_lightcurve()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, fit_core = self._fit_without_training(
                lc,
                model="2D",
                duplicate_wavelength_policy="select",
                duplicate_wavelength_selection="B",
            )
        self.assertEqual(set(lc.band.tolist()), {"B", "C", "D"})
        fit_core.assert_called_once()

    def test_fit_all_never_calls_core(self):
        lc = _make_duplicate_group_lightcurve()
        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            autospec=True,
        ) as fit_core:
            with self.assertRaises(NotImplementedError):
                lc.fit(
                    model="2D",
                    duplicate_wavelength_policy="all",
                )
        fit_core.assert_not_called()

    def test_success_history_records_resolution(self):
        lc = _make_duplicate_group_lightcurve()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fit_without_training(
                lc,
                model="2D",
                duplicate_wavelength_policy="first",
            )
        self.assertEqual(len(lc.fit_history), 1)
        notes = lc.fit_history[0]["notes"]
        resolution = notes["duplicate_wavelength_channel_resolution"]
        self.assertTrue(resolution["applied"])
        self.assertEqual(
            resolution["groups"][0]["selected_observational_channel"],
            "A",
        )


class TestRepresentativeDatasetDuplicateWavelengthPolicy(unittest.TestCase):
    def test_representative_dataset_default_selects_r3_0(self):
        lc = Lightcurve.from_csv(REPRESENTATIVE_CSV, max_samples=None)
        groups = lc._duplicate_physical_wavelength_channel_groups()
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertAlmostEqual(
            group["physical_wavelength"],
            KELT_WAVELENGTH,
            places=12,
        )
        self.assertEqual(
            group["observational_channels"],
            [KELT_R3_0, KELT_R3_1],
        )

        with self.assertWarnsRegex(UserWarning, "Selected .*R3_0"):
            provenance = lc._apply_duplicate_wavelength_channel_policy(
                policy="first",
            )

        channels = set(lc.band.tolist())
        self.assertIn(KELT_R3_0, channels)
        self.assertNotIn(KELT_R3_1, channels)
        self.assertEqual(len(lc.xdata), 7179)
        self.assertEqual(
            provenance["groups"][0]["ignored_row_counts"],
            {KELT_R3_1: 3636},
        )

    def test_representative_dataset_default_fit_requires_user_choice(self):
        lc = Lightcurve.from_csv(REPRESENTATIVE_CSV, max_samples=None)

        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            autospec=True,
        ) as fit_core:
            with self.assertRaisesRegex(
                ValueError,
                "Choose the GP training input explicitly",
            ):
                lc.fit(model="2D")

        fit_core.assert_not_called()
        self.assertEqual(len(lc.xdata), 10815)
        self.assertIn(KELT_R3_0, set(lc.band.tolist()))
        self.assertIn(KELT_R3_1, set(lc.band.tolist()))

    def test_representative_dataset_explicit_first_reaches_core(self):
        lc = Lightcurve.from_csv(REPRESENTATIVE_CSV, max_samples=None)

        with mock.patch.object(
            Lightcurve,
            "_fit_core",
            autospec=True,
            return_value="fit-result",
        ) as fit_core:
            with self.assertWarnsRegex(UserWarning, "Selected .*R3_0"):
                result = lc.fit(
                    model="2D",
                    duplicate_wavelength_policy="first",
                )

        self.assertEqual(result, "fit-result")
        fit_core.assert_called_once()
        self.assertEqual(len(lc.xdata), 7179)
        self.assertIn(KELT_R3_0, set(lc.band.tolist()))
        self.assertNotIn(KELT_R3_1, set(lc.band.tolist()))
        self.assertEqual(
            lc.duplicate_wavelength_channel_resolution["groups"][0][
                "selected_observational_channel"
            ],
            KELT_R3_0,
        )

    def test_representative_dataset_can_select_r3_1(self):
        lc = Lightcurve.from_csv(REPRESENTATIVE_CSV, max_samples=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lc._apply_duplicate_wavelength_channel_policy(
                policy="select",
                selection=KELT_R3_1,
            )
        channels = set(lc.band.tolist())
        self.assertNotIn(KELT_R3_0, channels)
        self.assertIn(KELT_R3_1, channels)
        self.assertEqual(len(lc.xdata), 5109)


if __name__ == "__main__":
    unittest.main()
