import unittest
from unittest import mock

import torch

from pgmuvi.lightcurve import Lightcurve


class TestFitStrategyDispatch(unittest.TestCase):
    def setUp(self):
        xdata = torch.as_tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        ydata = torch.as_tensor([1.0, 2.0, 1.5, 2.5], dtype=torch.float32)
        yerr = torch.full_like(ydata, 0.1)
        self.lc = Lightcurve(xdata, ydata, yerr=yerr)

    def test_fit_strategy_consensus_not_implemented(self):
        msg = r"^fit_strategy='consensus' is not implemented yet\.$"
        with self.assertRaisesRegex(NotImplementedError, msg):
            self.lc.fit(fit_strategy="consensus")

    def test_fit_strategy_consensus_multicomp_not_implemented(self):
        msg = r"^fit_strategy='consensus_multicomp' is not implemented yet\.$"
        with self.assertRaisesRegex(NotImplementedError, msg):
            self.lc.fit(fit_strategy="consensus_multicomp")

    def test_fit_strategy_consensus_relaxed_not_implemented(self):
        msg = r"^fit_strategy='consensus_relaxed' is not implemented yet\.$"
        with self.assertRaisesRegex(NotImplementedError, msg):
            self.lc.fit(fit_strategy="consensus_relaxed")

    def test_invalid_fit_strategy_raises_value_error(self):
        msg = (
            r"^Invalid fit_strategy\. Expected None or one of: "
            r"'consensus', 'consensus_multicomp', 'consensus_relaxed'\. "
            r"Got 'invalid_strategy'\.$"
        )
        with self.assertRaisesRegex(ValueError, msg):
            self.lc.fit(fit_strategy="invalid_strategy")

    def test_fit_strategy_none_uses_existing_fit_path(self):
        with mock.patch.object(
            Lightcurve,
            "_consensus_fit",
            side_effect=AssertionError("_consensus_fit should not be called"),
        ):
            with self.assertRaisesRegex(ValueError, r"^You must provide a model$"):
                self.lc.fit(fit_strategy=None)


if __name__ == "__main__":
    unittest.main()
