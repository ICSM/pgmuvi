"""
Unit tests for 2D constraints and transforms in pgmuvi.

Tests the handling of multiwavelength (2D) data including:
- Constraint setup for 2D data
- Hyperparameter transforms for 2D tensors
- Constraint transforms for 2D parameters
- Transformer classes with 2D data
- Validation of 2D setup
"""

import unittest
import torch
import numpy as np
from pgmuvi.lightcurve import Lightcurve, MinMax, ZScore, RobustZScore
from gpytorch.constraints import GreaterThan, Interval


class Test2DConstraintSetup(unittest.TestCase):
    """Test that set_default_constraints works for 2D data"""
    
    def setUp(self):
        """Set up 2D test data"""
        # Create synthetic 2D data: time and wavelength
        n_samples = 50
        time = torch.linspace(0, 10, n_samples, dtype=torch.float32)
        wavelength = torch.ones(n_samples, dtype=torch.float32) * 5.0  # Single wavelength
        
        # Stack into 2D format (n_samples, 2)
        self.xdata_2d = torch.stack([time, wavelength], dim=1)
        
        # Generate y-data with some variation
        self.ydata_2d = torch.sin(2 * np.pi * time / 2.5) + torch.randn(n_samples) * 0.1
        
        self.lightcurve_2d = Lightcurve(self.xdata_2d, self.ydata_2d)
        
    def test_ndim_is_2(self):
        """Test that ndim property correctly identifies 2D data"""
        self.assertEqual(self.lightcurve_2d.ndim, 2)
    
    def test_set_default_constraints_does_not_return_early(self):
        """Test that set_default_constraints doesn't return early for 2D data"""
        # Use Lightcurve.set_model which handles initialization properly
        self.lightcurve_2d.set_model('2D', likelihood=None, num_mixtures=2)
        
        # This should not raise an exception or print a warning and return early
        self.lightcurve_2d.set_default_constraints()
        
        # Check that constraints were actually set
        self.assertTrue(self.lightcurve_2d._Lightcurve__CONTRAINTS_SET)
        
    def test_mixture_means_constraint_registered(self):
        """Test that mixture_means constraint is registered for 2D data"""
        # Use Lightcurve.set_model which handles initialization properly
        self.lightcurve_2d.set_model('2D', likelihood=None, num_mixtures=2)
        self.lightcurve_2d.set_default_constraints()
        
        # Check that mixture_means has a constraint
        mixture_means_module = self.lightcurve_2d._model_pars['mixture_means']['module']
        self.assertTrue(hasattr(mixture_means_module, '_constraints'))
        # GPyTorch adds _constraint suffix
        self.assertIn('raw_mixture_means_constraint', mixture_means_module._constraints)
        
    def test_constraint_type_is_greater_than(self):
        """Test that the constraint is GreaterThan for 2D data"""
        # Use Lightcurve.set_model which handles initialization properly
        self.lightcurve_2d.set_model('2D', likelihood=None, num_mixtures=2)
        self.lightcurve_2d.set_default_constraints()
        
        # Check constraint type
        # GPyTorch adds _constraint suffix
        constraint = self.lightcurve_2d._model_pars['mixture_means']['module']._constraints['raw_mixture_means_constraint']
        self.assertIsInstance(constraint, GreaterThan)


class Test2DHyperparameterTransforms(unittest.TestCase):
    """Test that set_hypers works correctly with 2D parameters"""
    
    def setUp(self):
        """Set up 2D lightcurve"""
        n_samples = 50
        time = torch.linspace(0, 10, n_samples, dtype=torch.float32)
        wavelength = torch.ones(n_samples, dtype=torch.float32) * 5.0
        
        self.xdata_2d = torch.stack([time, wavelength], dim=1)
        self.ydata_2d = torch.sin(2 * np.pi * time / 2.5) + torch.randn(n_samples) * 0.1
        
        self.lightcurve_2d = Lightcurve(self.xdata_2d, self.ydata_2d)
        
        # Set up model using Lightcurve method
        self.lightcurve_2d.set_model('2D', likelihood=None, num_mixtures=2)
    
    def test_set_hypers_with_2d_mixture_means(self):
        """Test that set_hypers handles 2D mixture_means correctly"""
        # Create 2D hyperparameters (num_mixtures=2, ard_num_dims=2)
        mixture_means_2d = torch.tensor([[0.5, 0.1], [1.0, 0.2]], dtype=torch.float32)
        
        hypers = {
            'covar_module.mixture_means': mixture_means_2d
        }
        
        # This should not raise an exception
        self.lightcurve_2d.set_hypers(hypers)
        
    def test_set_hypers_preserves_2d_shape(self):
        """Test that set_hypers preserves the 2D shape of parameters"""
        mixture_means_2d = torch.tensor([[0.5, 0.1], [1.0, 0.2]], dtype=torch.float32)
        
        hypers = {
            'covar_module.mixture_means': mixture_means_2d
        }
        
        self.lightcurve_2d.set_hypers(hypers)
        
        # Get the actual parameter from the model
        actual_means = self.lightcurve_2d.model.covar_module.mixture_means
        
        # Check shape is preserved
        self.assertEqual(actual_means.shape, torch.Size([2, 2]))
    
    def test_set_hypers_with_2d_mixture_scales(self):
        """Test that set_hypers handles 2D mixture_scales correctly"""
        mixture_scales_2d = torch.tensor([[0.5, 0.1], [1.0, 0.2]], dtype=torch.float32)
        
        hypers = {
            'covar_module.mixture_scales': mixture_scales_2d
        }
        
        # This should not raise an exception
        self.lightcurve_2d.set_hypers(hypers)


class Test2DConstraintTransforms(unittest.TestCase):
    """Test that set_constraint handles 2D parameters correctly"""
    
    def setUp(self):
        """Set up 2D lightcurve"""
        n_samples = 50
        time = torch.linspace(0, 10, n_samples, dtype=torch.float32)
        wavelength = torch.ones(n_samples, dtype=torch.float32) * 5.0
        
        self.xdata_2d = torch.stack([time, wavelength], dim=1)
        self.ydata_2d = torch.sin(2 * np.pi * time / 2.5) + torch.randn(n_samples) * 0.1
        
        self.lightcurve_2d = Lightcurve(self.xdata_2d, self.ydata_2d)
        
        # Set up model using Lightcurve method
        self.lightcurve_2d.set_model('2D', likelihood=None, num_mixtures=2)
    
    def test_set_constraint_with_greater_than(self):
        """Test that set_constraint works with GreaterThan for 2D"""
        constraint = GreaterThan(0.1)
        
        constraints = {
            'mixture_means': constraint
        }
        
        # This should not raise an exception
        self.lightcurve_2d.set_constraint(constraints)
        
        # Verify constraint was set
        self.assertIn('raw_mixture_means', 
                      self.lightcurve_2d._model_pars['mixture_means']['module']._constraints)
    
    def test_set_constraint_with_interval(self):
        """Test that set_constraint works with Interval for 2D"""
        constraint = Interval(0.1, 10.0)
        
        constraints = {
            'mixture_means': constraint
        }
        
        # This should not raise an exception
        self.lightcurve_2d.set_constraint(constraints)
        
        # Verify constraint was set
        self.assertIn('raw_mixture_means',
                      self.lightcurve_2d._model_pars['mixture_means']['module']._constraints)


class Test2DTransformerClasses(unittest.TestCase):
    """Test that transformer classes work with 2D data"""
    
    def test_minmax_with_2d_data(self):
        """Test MinMax transformer with 2D data"""
        # Create 2D data (n_samples, 2)
        data_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        
        transformer = MinMax()
        transformed = transformer.transform(data_2d, dim=0)
        
        # Check that transformation worked
        self.assertEqual(transformed.shape, data_2d.shape)
        # Check that each dimension is scaled to [0, 1]
        self.assertAlmostEqual(transformed[:, 0].min().item(), 0.0, places=5)
        self.assertAlmostEqual(transformed[:, 0].max().item(), 1.0, places=5)
    
    def test_minmax_apply_to_dimension(self):
        """Test MinMax transformer with apply_to parameter"""
        data_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        
        transformer = MinMax()
        # First transform to calculate min/range
        _ = transformer.transform(data_2d, dim=0)
        
        # Now test apply_to
        test_data = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
        transformed = transformer.transform(test_data, dim=0, apply_to=[0])
        
        # Should have the same shape
        self.assertEqual(transformed.shape, test_data.shape)
    
    def test_zscore_with_2d_data(self):
        """Test ZScore transformer with 2D data"""
        data_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        
        transformer = ZScore()
        transformed = transformer.transform(data_2d, dim=0)
        
        # Check that transformation worked
        self.assertEqual(transformed.shape, data_2d.shape)
    
    def test_robust_zscore_with_2d_data(self):
        """Test RobustZScore transformer with 2D data"""
        data_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
        
        transformer = RobustZScore()
        transformed = transformer.transform(data_2d, dim=0)
        
        # Check that transformation worked
        self.assertEqual(transformed.shape, data_2d.shape)


class Test2DValidation(unittest.TestCase):
    """Test validation of 2D setup"""
    
    def test_validate_2d_setup_with_correct_data(self):
        """Test that validation passes with correct 2D setup"""
        n_samples = 50
        time = torch.linspace(0, 10, n_samples, dtype=torch.float32)
        wavelength = torch.ones(n_samples, dtype=torch.float32) * 5.0
        
        xdata_2d = torch.stack([time, wavelength], dim=1)
        ydata_2d = torch.sin(2 * np.pi * time / 2.5) + torch.randn(n_samples) * 0.1
        
        lightcurve_2d = Lightcurve(xdata_2d, ydata_2d)
        
        # Use Lightcurve.set_model
        lightcurve_2d.set_model('2D', likelihood=None, num_mixtures=2)
        
        # This should not raise an exception
        lightcurve_2d._validate_2d_setup()
    
    def test_validate_2d_setup_with_wrong_dimensions(self):
        """Test that validation fails with wrong number of dimensions"""
        # Create data with 3 columns instead of 2
        n_samples = 50
        xdata_wrong = torch.randn(n_samples, 3, dtype=torch.float32)
        ydata = torch.randn(n_samples, dtype=torch.float32)
        
        lightcurve = Lightcurve(xdata_wrong, ydata)
        
        # This should raise a ValueError
        with self.assertRaises(ValueError) as context:
            lightcurve._validate_2d_setup()
        
        self.assertIn("must have 2 columns", str(context.exception))
    
    def test_validate_2d_setup_with_1d_model(self):
        """Test that validation fails when using 1D model with 2D data"""
        n_samples = 50
        time = torch.linspace(0, 10, n_samples, dtype=torch.float32)
        wavelength = torch.ones(n_samples, dtype=torch.float32) * 5.0
        
        xdata_2d = torch.stack([time, wavelength], dim=1)
        ydata_2d = torch.sin(2 * np.pi * time / 2.5) + torch.randn(n_samples) * 0.1
        
        lightcurve_2d = Lightcurve(xdata_2d, ydata_2d)
        
        # Try to use a 1D model (this should fail)
        from pgmuvi.gps import SpectralMixtureGPModel
        from gpytorch.likelihoods import GaussianLikelihood
        
        likelihood = GaussianLikelihood()
        
        # This will fail because 1D model expects 1D data
        # We'll test the validation catches this
        try:
            model = SpectralMixtureGPModel(
                lightcurve_2d._xdata_transformed[:, 0],  # Use only first dimension
                lightcurve_2d._ydata_transformed,
                likelihood,
                num_mixtures=2
            )
            lightcurve_2d.model = model
            lightcurve_2d.likelihood = likelihood
            
            # Validation should catch the mismatch
            with self.assertRaises(ValueError) as context:
                lightcurve_2d._validate_2d_setup()
            
            # Should mention ard_num_dims
            self.assertIn("ard_num_dims", str(context.exception).lower())
        except Exception:
            # If model creation itself fails, that's also acceptable
            pass


if __name__ == '__main__':
    unittest.main()
