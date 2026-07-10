"""Tests for default time centering and transform propagation."""

import unittest

import numpy as np
import torch

from pgmuvi.lightcurve import Lightcurve, MinMax, TimeCenter


class TestTimeCenterTransform(unittest.TestCase):
    def test_lightcurve_defaults_to_time_centering_1d(self):
        x = torch.as_tensor([1000.0, 1010.0, 1020.0], dtype=torch.float32)
        y = torch.zeros_like(x)

        lc = Lightcurve(x, y)

        self.assertIsInstance(lc.xtransform, TimeCenter)
        torch.testing.assert_close(lc.xdata, x)
        torch.testing.assert_close(
            lc._xdata_transformed,
            torch.as_tensor([-10.0, 0.0, 10.0], dtype=torch.float32),
        )

    def test_lightcurve_defaults_to_time_centering_2d_time_only(self):
        x = torch.as_tensor(
            [[1000.0, 2.0], [1010.0, 4.0], [1020.0, 6.0]],
            dtype=torch.float32,
        )
        y = torch.zeros(3, dtype=torch.float32)

        lc = Lightcurve(x, y)

        self.assertIsInstance(lc.xtransform, TimeCenter)
        torch.testing.assert_close(lc.xdata, x)
        torch.testing.assert_close(
            lc._xdata_transformed[:, 0],
            torch.tensor([-10.0, 0.0, 10.0]),
        )
        torch.testing.assert_close(lc._xdata_transformed[:, 1], x[:, 1])

    def test_2d_time_center_can_be_reused_for_1d_band_coordinates(self):
        parent_x = torch.as_tensor(
            [[1000.0, 1.0], [1010.0, 1.0], [1020.0, 2.0]],
            dtype=torch.float32,
        )
        parent_y = torch.zeros(3, dtype=torch.float32)
        parent = Lightcurve(parent_x, parent_y)

        band_x = torch.as_tensor([1000.0, 1010.0, 1020.0], dtype=torch.float32)
        transformed = parent.xtransform.transform(band_x)

        torch.testing.assert_close(
            transformed,
            torch.as_tensor([-10.0, 0.0, 10.0], dtype=torch.float32),
        )
        torch.testing.assert_close(parent.xtransform.inverse(transformed), band_x)

    def test_center_time_false_preserves_raw_training_coordinates(self):
        x = torch.as_tensor([1000.0, 1010.0, 1020.0], dtype=torch.float32)
        y = torch.zeros_like(x)

        lc = Lightcurve(x, y, center_time=False)

        self.assertIsNone(lc.xtransform)
        torch.testing.assert_close(lc._xdata_transformed, x)

    def test_explicit_xtransform_is_not_composed_in_auto_mode(self):
        x = torch.as_tensor([1000.0, 1010.0, 1020.0], dtype=torch.float32)
        y = torch.zeros_like(x)

        lc = Lightcurve(x, y, xtransform="minmax")

        self.assertIsInstance(lc.xtransform, MinMax)
        torch.testing.assert_close(
            lc._xdata_transformed,
            torch.tensor([0.0, 0.5, 1.0]),
        )

    def test_explicit_xtransform_and_forced_center_time_raise(self):
        x = torch.as_tensor([1000.0, 1010.0, 1020.0], dtype=torch.float32)
        y = torch.zeros_like(x)

        with self.assertRaisesRegex(ValueError, "cannot currently be combined"):
            Lightcurve(x, y, xtransform="minmax", center_time=True)

    def test_transform_y_uses_ytransform_not_xtransform(self):
        x = torch.as_tensor([1000.0, 1010.0, 1020.0], dtype=torch.float32)
        y = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        lc = Lightcurve(x, y, ytransform="minmax")
        transformed = lc.transform_y(y)

        torch.testing.assert_close(
            transformed,
            torch.tensor([0.0, 0.5, 1.0]),
        )

    def test_select_and_drop_bands_preserve_time_center_transform(self):
        t = torch.as_tensor([1000.0, 1010.0, 1020.0, 1000.0, 1010.0, 1020.0])
        wl = torch.as_tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        x = torch.stack((t, wl), dim=1)
        y = torch.zeros(6)
        band = np.array(["a", "a", "a", "b", "b", "b"])
        lc = Lightcurve(x, y, band=band)

        selected = lc.select_bands(["a"])
        dropped = lc.drop_bands(["b"])

        self.assertIs(selected.xtransform, lc.xtransform)
        self.assertIs(dropped.xtransform, lc.xtransform)
        torch.testing.assert_close(
            selected._xdata_transformed[:, 0],
            torch.tensor([-10.0, 0.0, 10.0]),
        )
        torch.testing.assert_close(
            dropped._xdata_transformed[:, 0],
            torch.tensor([-10.0, 0.0, 10.0]),
        )


if __name__ == "__main__":
    unittest.main()
