import unittest

from pgmuvi.lightcurve import Lightcurve


class TestConsensusScaleConstraintUpper(unittest.TestCase):
    def test_width_cap_wins_when_narrower_than_frequency_fraction(self):
        info = Lightcurve._consensus_resolve_scale_constraint_upper(
            [0.006332046332046332],
            consensus_frequency_width=[7.72201e-05],
            consensus_scale_max_factor=0.2,
            consensus_scale_width_factor=1.0,
        )

        self.assertEqual(
            info["strategy"],
            "min_frequency_fraction_and_frequency_width",
        )
        self.assertAlmostEqual(info["frequency_fraction_upper"], 0.0012664092664092664)
        self.assertAlmostEqual(info["width_upper"], 7.72201e-05)
        self.assertAlmostEqual(info["upper"], 7.72201e-05)

    def test_frequency_fraction_wins_when_width_is_broader(self):
        info = Lightcurve._consensus_resolve_scale_constraint_upper(
            [0.004770134667687988],
            consensus_frequency_width=[0.00120653],
            consensus_scale_max_factor=0.2,
            consensus_scale_width_factor=1.0,
        )

        self.assertEqual(
            info["strategy"],
            "min_frequency_fraction_and_frequency_width",
        )
        self.assertAlmostEqual(info["upper"], 0.0009540269335375977)
        self.assertAlmostEqual(info["frequency_fraction_upper"], info["upper"])

    def test_falls_back_to_frequency_fraction_without_width(self):
        info = Lightcurve._consensus_resolve_scale_constraint_upper(
            [0.0022631349809186328],
            consensus_frequency_width=None,
            consensus_scale_max_factor=0.2,
            consensus_scale_width_factor=1.0,
        )

        self.assertEqual(info["strategy"], "frequency_fraction")
        self.assertIsNone(info["width_upper"])
        self.assertAlmostEqual(info["upper"], 0.00045262699618372656)

    def test_invalid_width_factor_is_rejected(self):
        with self.assertRaises(ValueError):
            Lightcurve._consensus_resolve_scale_constraint_upper(
                [0.001],
                consensus_frequency_width=[1.0e-5],
                consensus_scale_width_factor=0.0,
            )


if __name__ == "__main__":
    unittest.main()
