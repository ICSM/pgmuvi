"""
Integration test for 2D multiwavelength data workflow.

Tests the complete workflow of:
1. Creating synthetic 2D multiwavelength data
2. Setting up a Lightcurve with 2D data
3. Setting a 2D model
4. Running fit() successfully
5. Verifying convergence
"""

import unittest
import torch
import numpy as np
from pgmuvi.lightcurve import Lightcurve
from pgmuvi.gps import TwoDSpectralMixtureGPModel
from gpytorch.likelihoods import GaussianLikelihood
from pgmuvi.synthetic import make_chromatic_sinusoid_2d


class Test2DIntegration(unittest.TestCase):
    """Integration test for complete 2D workflow"""

    def setUp(self):
        """Generate synthetic 2D multiwavelength data"""
        self.true_period = 5.0
        self.true_freq = 1.0 / self.true_period

        lc = make_chromatic_sinusoid_2d(
            n_per_band=[100, 80],
            period=self.true_period,
            wavelengths=[0.5, 1.5],
            amplitude_law="linear",
            amplitude_slope=0.3,
            wl_ref=0.0,
            noise_level=0.1,
            t_span=20.0,
            irregular=False,
            seed=42,
        )
        self.xdata_2d = lc.xdata
        self.ydata_2d = lc.ydata

        # Store true parameters for comparison
        self.true_params = {
            'period': self.true_period,
            'frequency': self.true_freq
        }

    def test_create_2d_lightcurve(self):
        """Test creating a Lightcurve object with 2D data"""
        lightcurve = Lightcurve(self.xdata_2d, self.ydata_2d)

        # Check that it was created successfully
        self.assertIsNotNone(lightcurve)
        self.assertEqual(lightcurve.ndim, 2)
        self.assertEqual(lightcurve._xdata_transformed.shape[1], 2)

    def test_set_2d_model(self):
        """Test setting a 2D model"""
        lightcurve = Lightcurve(self.xdata_2d, self.ydata_2d)

        # Set model using string identifier
        lightcurve.set_model('2D', likelihood=None, num_mixtures=3)

        # Check that model was set
        self.assertIsNotNone(lightcurve.model)
        self.assertIsInstance(lightcurve.model, TwoDSpectralMixtureGPModel)

        # Check that model has correct ard_num_dims
        self.assertEqual(lightcurve.model.covar_module.ard_num_dims, 2)

    def test_default_constraints_set(self):
        """Test that default constraints are set for 2D data"""
        lightcurve = Lightcurve(self.xdata_2d, self.ydata_2d)
        lightcurve.set_model('2D', likelihood=None, num_mixtures=3)

        # Set default constraints
        lightcurve.set_default_constraints()

        # Check that constraints were set
        self.assertTrue(lightcurve._Lightcurve__CONTRAINTS_SET)

        # Check that mixture_means has a constraint
        # GPyTorch adds _constraint suffix
        self.assertIn('raw_mixture_means_constraint',
                      lightcurve._model_pars['mixture_means']['module']._constraints)

    def test_fit_2d_workflow(self):
        """Test complete fit workflow with 2D data"""
        lightcurve = Lightcurve(self.xdata_2d, self.ydata_2d)

        # Run fit with minimal iterations for speed
        results = lightcurve.fit(
            model='2D',
            likelihood=None,
            num_mixtures=3,
            training_iter=50,  # Reduced for speed in testing
            miniter=10,
            lr=0.05,
            stop=1e-3
        )

        # Check that fit completed
        self.assertIsNotNone(results)
        self.assertTrue(lightcurve._Lightcurve__FITTED_MAP)

        # Check that losses were recorded (results dict has 'loss' key)
        self.assertIn('loss', results)
        self.assertGreater(len(results['loss']), 0)

    def test_fit_convergence(self):
        """Test that fit shows convergence (loss decreases)"""
        lightcurve = Lightcurve(self.xdata_2d, self.ydata_2d)

        results = lightcurve.fit(
            model='2D',
            likelihood=None,
            num_mixtures=3,
            training_iter=100,
            miniter=20,
            lr=0.01,
            stop=1e-4
        )

        # Results dict has 'loss' key
        losses = results['loss']

        # Check that final loss is lower than initial loss
        # (allowing for some fluctuation)
        initial_loss = np.mean([float(l) for l in losses[:5]])
        final_loss = np.mean([float(l) for l in losses[-5:]])

        self.assertLess(final_loss, initial_loss,
                        msg=f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}")

    def test_set_hypers_with_2d(self):
        """Test setting hyperparameters for 2D model"""
        lightcurve = Lightcurve(self.xdata_2d, self.ydata_2d)
        lightcurve.set_model('2D', likelihood=None, num_mixtures=2)

        # Create 2D hyperparameters
        # Frequency in transformed space
        hypers = {
            'covar_module.mixture_means': torch.tensor([
                [self.true_freq, 0.5],  # First mixture
                [self.true_freq * 2, 0.3]  # Second mixture (harmonic)
            ], dtype=torch.float32),
            'covar_module.mixture_scales': torch.tensor([
                [0.5, 0.2],
                [0.3, 0.1]
            ], dtype=torch.float32)
        }

        # This should not raise an exception
        lightcurve.set_hypers(hypers)

        # Run a few iterations to ensure it's stable
        results = lightcurve.fit(
            training_iter=20,
            miniter=5,
            lr=0.05
        )

        self.assertIsNotNone(results)

    def test_validation_catches_wrong_model(self):
        """Test that validation catches using 1D model with 2D data"""
        lightcurve = Lightcurve(self.xdata_2d, self.ydata_2d)

        # Try to fit with 1D model (should fail during validation)
        with self.assertRaises((ValueError, RuntimeError)) as context:
            # This should fail either during model setup or validation
            lightcurve.fit(
                model='1D',  # Wrong! Should be '2D'
                likelihood=None,
                num_mixtures=3,
                training_iter=10
            )

        # The error should mention dimensionality or ard_num_dims
        error_msg = str(context.exception).lower()
        self.assertTrue(
            'dimension' in error_msg or 'ard' in error_msg or 'shape' in error_msg,
            msg=f"Error message didn't mention dimensionality issue: {context.exception}"
        )

    def test_2d_with_different_sampling(self):
        """Test 2D data with different sampling in each wavelength band"""
        lc = make_chromatic_sinusoid_2d(
            n_per_band=[150, 50],
            period=5.0,
            wavelengths=[0.5, 1.5],
            amplitude_law="linear",
            amplitude_slope=0.2,
            wl_ref=0.0,
            noise_level=0.1,
            t_span=20.0,
            irregular=False,
            seed=42,
        )

        # Create lightcurve and fit
        lightcurve = Lightcurve(lc.xdata, lc.ydata)
        results = lightcurve.fit(
            model='2D',
            num_mixtures=2,
            training_iter=30,
            miniter=10,
            lr=0.05
        )

        # Should complete successfully
        self.assertIsNotNone(results)
        self.assertTrue(lightcurve._Lightcurve__FITTED_MAP)


class Test2DWithLinearMean(unittest.TestCase):
    """Test 2D models with linear mean function"""

    def setUp(self):
        """Generate synthetic 2D data with linear trend"""
        lc = make_chromatic_sinusoid_2d(
            n_per_band=100,
            period=5.0,
            wavelengths=[1.0],
            amplitude_law="linear",
            amplitude_slope=0.0,
            noise_level=0.1,
            t_span=20.0,
            irregular=False,
            seed=42,
        )
        # Add a linear trend on top of the periodic component
        time = lc.xdata[:, 0]
        linear_trend = 0.05 * time
        self.xdata_2d = lc.xdata
        self.ydata_2d = lc.ydata + linear_trend

    def test_2d_linear_mean_model(self):
        """Test 2D model with linear mean"""
        lightcurve = Lightcurve(self.xdata_2d, self.ydata_2d)

        # Just test that the model can be created
        # Full fit test is covered by other tests
        lightcurve.set_model('2DLinear', likelihood=None, num_mixtures=2)

        # Model should have mean_module
        self.assertTrue(hasattr(lightcurve.model, 'mean_module'))

        # Mean module should be LinearMean
        from gpytorch.means import LinearMean
        self.assertIsInstance(lightcurve.model.mean_module, LinearMean)


if __name__ == '__main__':
    unittest.main()
