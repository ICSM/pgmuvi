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


class Test2DIntegration(unittest.TestCase):
    """Integration test for complete 2D workflow"""
    
    def setUp(self):
        """Generate synthetic 2D multiwavelength data"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Parameters for synthetic data
        n_samples_band1 = 100
        n_samples_band2 = 80
        
        # Time arrays for two wavelength bands
        time_band1 = torch.linspace(0, 20, n_samples_band1, dtype=torch.float32)
        time_band2 = torch.linspace(0, 20, n_samples_band2, dtype=torch.float32)
        
        # Wavelength identifiers (normalized)
        wavelength_band1 = torch.ones(n_samples_band1, dtype=torch.float32) * 0.5  # Band 1
        wavelength_band2 = torch.ones(n_samples_band2, dtype=torch.float32) * 1.5  # Band 2
        
        # Combine into 2D arrays
        time_all = torch.cat([time_band1, time_band2])
        wavelength_all = torch.cat([wavelength_band1, wavelength_band2])
        
        # Stack into (n_samples, 2) format
        self.xdata_2d = torch.stack([time_all, wavelength_all], dim=1)
        
        # Generate y-data with known periodic signal
        # Period = 5 time units, with wavelength-dependent amplitude
        self.true_period = 5.0
        self.true_freq = 1.0 / self.true_period
        
        # Base signal (achromatic)
        base_signal = torch.sin(2 * np.pi * time_all / self.true_period)
        
        # Wavelength-dependent amplitude modulation
        amplitude_factor = 1.0 + 0.3 * wavelength_all
        
        # Combine signal with noise
        signal = amplitude_factor * base_signal
        noise = torch.randn_like(signal) * 0.1
        
        self.ydata_2d = signal + noise
        
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
        self.assertIn('raw_mixture_means',
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
        
        # Check that losses were recorded
        self.assertIn('losses', results)
        self.assertGreater(len(results['losses']), 0)
    
    def test_fit_convergence(self):
        """Test that fit shows convergence (loss decreases)"""
        lightcurve = Lightcurve(self.xdata_2d, self.ydata_2d)
        
        results = lightcurve.fit(
            model='2D',
            likelihood=None,
            num_mixtures=3,
            training_iter=50,
            miniter=10,
            lr=0.05,
            stop=1e-3
        )
        
        losses = results['losses']
        
        # Check that final loss is lower than initial loss
        # (allowing for some fluctuation)
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        
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
        # This is a common real-world scenario
        
        # Band 1: dense sampling
        n1 = 150
        time1 = torch.linspace(0, 20, n1, dtype=torch.float32)
        wavelength1 = torch.ones(n1, dtype=torch.float32) * 0.5
        
        # Band 2: sparse sampling
        n2 = 50
        time2 = torch.linspace(0, 20, n2, dtype=torch.float32)
        wavelength2 = torch.ones(n2, dtype=torch.float32) * 1.5
        
        # Combine
        xdata = torch.stack([
            torch.cat([time1, time2]),
            torch.cat([wavelength1, wavelength2])
        ], dim=1)
        
        # Generate y data
        time_all = xdata[:, 0]
        wavelength_all = xdata[:, 1]
        signal = torch.sin(2 * np.pi * time_all / 5.0) * (1 + 0.2 * wavelength_all)
        ydata = signal + torch.randn_like(signal) * 0.1
        
        # Create lightcurve and fit
        lightcurve = Lightcurve(xdata, ydata)
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
        torch.manual_seed(42)
        np.random.seed(42)
        
        n_samples = 100
        time = torch.linspace(0, 20, n_samples, dtype=torch.float32)
        wavelength = torch.ones(n_samples, dtype=torch.float32) * 1.0
        
        self.xdata_2d = torch.stack([time, wavelength], dim=1)
        
        # Signal with linear trend and periodic component
        linear_trend = 0.05 * time
        periodic = torch.sin(2 * np.pi * time / 5.0)
        noise = torch.randn(n_samples) * 0.1
        
        self.ydata_2d = linear_trend + periodic + noise
    
    def test_2d_linear_mean_model(self):
        """Test 2D model with linear mean"""
        lightcurve = Lightcurve(self.xdata_2d, self.ydata_2d)
        
        results = lightcurve.fit(
            model='2DLinear',  # Model with linear mean
            num_mixtures=2,
            training_iter=30,
            miniter=10,
            lr=0.05
        )
        
        # Should complete successfully
        self.assertIsNotNone(results)
        self.assertTrue(lightcurve._Lightcurve__FITTED_MAP)
        
        # Model should have mean_module
        self.assertTrue(hasattr(lightcurve.model, 'mean_module'))


if __name__ == '__main__':
    unittest.main()
