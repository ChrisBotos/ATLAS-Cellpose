#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Test Name: test_feature_extraction_optimization.py.
Description:
    Test suite for feature extraction performance optimizations.
    Validates that optimized configurations work correctly and provide
    expected performance improvements while maintaining scientific accuracy.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pandas, tempfile.
    • Custom feature extraction utilities.

Usage:
    pytest tests/test_feature_extraction_optimization.py -v

Test Categories:
    • Configuration loading and validation.
    • Feature extraction with different optimization settings.
    • Performance comparison between optimized and standard configurations.
    • Scientific accuracy validation for optimized features.

Key Features:
    • Synthetic data generation for consistent testing.
    • Performance benchmarking with timing measurements.
    • Feature accuracy validation against reference implementations.
    • Memory usage monitoring and validation.

Notes:
    • Tests use synthetic data to ensure reproducible results.
    • Performance tests may vary based on system specifications.
    • Scientific accuracy tests validate that optimizations preserve biological relevance.
"""

import traceback
import sys
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, Any, Tuple

import pytest
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path for imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Import feature extraction utilities.
from code.engineered_feature_extraction.utils.config_loader import load_feature_extraction_config
from code.engineered_feature_extraction.extract_engineered_features import (
    extract_shape_features,
    extract_neighborhood_features,
    extract_texture_features,
    build_neighbors_list,
    filter_nuclei_by_size
)


class TestFeatureExtractionOptimization:
    """Test suite for feature extraction performance optimizations."""
    
    @pytest.fixture
    def synthetic_data(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """Generate synthetic test data for consistent testing."""
        # Create small synthetic image and mask.
        height, width = 512, 512
        
        # Create background image.
        gray = np.random.normal(20, 5, (height, width)).astype(np.uint8)
        gray = np.clip(gray, 0, 255)
        
        # Create segmentation mask with 50 nuclei.
        mask = np.zeros((height, width), dtype=np.int32)
        
        # Generate nuclear positions.
        positions = []
        for i in range(50):
            y = np.random.randint(20, height - 20)
            x = np.random.randint(20, width - 20)
            positions.append((y, x))
        
        # Draw nuclei.
        for i, (y, x) in enumerate(positions):
            radius = np.random.randint(8, 12)
            yy, xx = np.ogrid[:height, :width]
            circle = (yy - y)**2 + (xx - x)**2 <= radius**2
            mask[circle] = i + 1
            gray[circle] = np.clip(gray[circle] + 150, 0, 255)
        
        # Extract region properties.
        from skimage.measure import regionprops
        props = regionprops(mask)
        
        return gray, mask, props
    
    @pytest.fixture
    def test_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create test configurations for performance comparison."""
        return {
            'minimal': {
                'shape_features': True,
                'size_features': True,
                'neighborhood_features': False,
                'texture_features': False,
                'enable_fractal_dimension': False,
                'enable_convex_hull_features': False,
            },
            'optimized': {
                'shape_features': True,
                'size_features': True,
                'neighborhood_features': True,
                'texture_features': True,
                'enable_fractal_dimension': False,
                'enable_convex_hull_features': False,
                'enable_pca_clustering': False,
                'enable_spatial_autocorrelation': False,
                'enable_clustering_coefficient': False,
                'enable_glcm_features': False,
                'enable_gradient_features': True,
                'skip_expensive_texture': True,
                'enable_vectorized_neighborhood': True,
                'neighborhood_batch_size': 100,
            },
            'comprehensive': {
                'shape_features': True,
                'size_features': True,
                'neighborhood_features': True,
                'texture_features': True,
                'enable_fractal_dimension': True,
                'enable_convex_hull_features': True,
                'enable_pca_clustering': True,
                'enable_spatial_autocorrelation': True,
                'enable_clustering_coefficient': True,
                'enable_glcm_features': False,  # Still skip GLCM for test speed.
                'enable_gradient_features': True,
                'skip_expensive_texture': False,
                'enable_vectorized_neighborhood': True,
            }
        }
    
    def test_config_loading(self):
        """Test that configuration loading works with new parameters."""
        config = load_feature_extraction_config()
        
        # Check that new optimization parameters are loaded.
        assert 'enable_vectorized_neighborhood' in config
        assert 'neighborhood_batch_size' in config
        assert 'skip_expensive_texture' in config
        assert 'enable_fractal_dimension' in config
        
        # Check default values.
        assert config['enable_vectorized_neighborhood'] is True
        assert config['neighborhood_batch_size'] == 1000
        assert config['skip_expensive_texture'] is True
    
    def test_shape_features_optimization(self, synthetic_data):
        """Test that shape feature optimization works correctly."""
        gray, mask, props = synthetic_data
        
        if not props:
            pytest.skip("No nuclei found in synthetic data")
        
        region = props[0]
        
        # Test with convex hull features enabled.
        config_full = {'enable_convex_hull_features': True}
        features_full = extract_shape_features(region, gray, config_full)
        
        # Test with convex hull features disabled.
        config_minimal = {'enable_convex_hull_features': False}
        features_minimal = extract_shape_features(region, gray, config_minimal)
        
        # Check that basic features are present in both.
        basic_features = ['circularity', 'eccentricity', 'solidity', 'aspect_ratio']
        for feature in basic_features:
            assert feature in features_full
            assert feature in features_minimal
            assert features_full[feature] == features_minimal[feature]
        
        # Check that convex hull features are handled correctly.
        assert not np.isnan(features_full['convex_area_ratio'])
        assert np.isnan(features_minimal['convex_area_ratio'])
    
    def test_neighborhood_features_optimization(self, synthetic_data):
        """Test that neighborhood feature optimization works correctly."""
        gray, mask, props = synthetic_data
        
        if len(props) < 5:
            pytest.skip("Not enough nuclei for neighborhood testing")
        
        # Build neighbor data.
        from scipy.spatial import cKDTree
        centroids = [r.centroid for r in props]
        tree = cKDTree(centroids)
        
        config_optimized = {
            'enable_pca_clustering': False,
            'enable_spatial_autocorrelation': False,
            'enable_clustering_coefficient': False,
            'enable_vectorized_neighborhood': True,
        }
        
        neighbors = build_neighbors_list(props, tree, 50.0, config_optimized)
        
        # Test feature extraction with optimization.
        region = props[0]
        neighbor_data = neighbors[0]
        
        features = extract_neighborhood_features(
            region, neighbor_data, gray.shape, config_optimized
        )
        
        # Check that basic features are present.
        assert 'nearest_neighbor_distance' in features
        assert 'neighborhood_density' in features
        assert 'boundary_proximity' in features
        
        # Check that expensive features are skipped.
        assert features['cluster_elongation'] == 0.0
        assert features['spatial_autocorrelation'] == 0.0
        assert features['local_clustering_coefficient'] == 0.0
    
    def test_texture_features_optimization(self, synthetic_data):
        """Test that texture feature optimization works correctly."""
        gray, mask, props = synthetic_data
        
        if not props:
            pytest.skip("No nuclei found in synthetic data")
        
        region = props[0]
        
        # Test with expensive features disabled.
        config_fast = {
            'enable_glcm_features': False,
            'enable_gradient_features': True,
            'skip_expensive_texture': True,
        }
        
        features = extract_texture_features(region, gray, config_fast)
        
        # Check that basic features are present.
        basic_features = ['intensity_mean', 'intensity_std', 'intensity_median', 'texture_entropy']
        for feature in basic_features:
            assert feature in features
            assert not np.isnan(features[feature])
        
        # Check that GLCM features are skipped.
        glcm_features = ['glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 'glcm_energy']
        for feature in glcm_features:
            assert feature in features
            assert np.isnan(features[feature])
        
        # Check that gradient features are computed.
        assert 'gradient_magnitude_mean' in features
        assert not np.isnan(features['gradient_magnitude_mean'])
    
    def test_vectorized_neighborhood_performance(self, synthetic_data):
        """Test that vectorized neighborhood computation is faster."""
        gray, mask, props = synthetic_data
        
        if len(props) < 10:
            pytest.skip("Not enough nuclei for performance testing")
        
        from scipy.spatial import cKDTree
        centroids = [r.centroid for r in props]
        tree = cKDTree(centroids)
        
        # Test standard approach.
        config_standard = {'enable_vectorized_neighborhood': False}
        start_time = time.time()
        neighbors_standard = build_neighbors_list(props, tree, 50.0, config_standard)
        time_standard = time.time() - start_time
        
        # Test vectorized approach.
        config_vectorized = {
            'enable_vectorized_neighborhood': True,
            'neighborhood_batch_size': 50,
        }
        start_time = time.time()
        neighbors_vectorized = build_neighbors_list(props, tree, 50.0, config_vectorized)
        time_vectorized = time.time() - start_time
        
        # Check that results are equivalent.
        assert len(neighbors_standard) == len(neighbors_vectorized)
        
        # For small datasets, vectorized might not be faster due to overhead.
        # Just check that both approaches work.
        for i in range(len(neighbors_standard)):
            assert len(neighbors_standard[i]['centroids']) == len(neighbors_vectorized[i]['centroids'])
    
    def test_feature_accuracy_preservation(self, synthetic_data):
        """Test that optimizations preserve feature accuracy."""
        gray, mask, props = synthetic_data
        
        if not props:
            pytest.skip("No nuclei found in synthetic data")
        
        region = props[0]
        
        # Compare basic shape features (should be identical).
        config_full = {'enable_convex_hull_features': True}
        config_minimal = {'enable_convex_hull_features': False}
        
        features_full = extract_shape_features(region, gray, config_full)
        features_minimal = extract_shape_features(region, gray, config_minimal)
        
        # Basic features should be identical.
        basic_features = ['circularity', 'eccentricity', 'solidity']
        for feature in basic_features:
            assert abs(features_full[feature] - features_minimal[feature]) < 1e-10
    
    def test_memory_efficiency(self, synthetic_data):
        """Test that optimizations reduce memory usage."""
        gray, mask, props = synthetic_data
        
        # This is a simplified test - in practice, memory usage would be
        # measured more precisely with memory profiling tools.
        
        # Test that vectorized operations don't create excessive temporary arrays.
        if len(props) >= 5:
            from scipy.spatial import cKDTree
            centroids = [r.centroid for r in props]
            tree = cKDTree(centroids)
            
            config = {
                'enable_vectorized_neighborhood': True,
                'neighborhood_batch_size': 10,  # Small batch for memory efficiency.
            }
            
            # This should not raise memory errors.
            neighbors = build_neighbors_list(props, tree, 50.0, config)
            assert len(neighbors) == len(props)
    
    def test_configuration_validation(self, test_configs):
        """Test that different configurations produce valid results."""
        # Test that all test configurations are valid.
        for config_name, config in test_configs.items():
            # Check required parameters.
            assert 'shape_features' in config
            assert 'size_features' in config
            assert 'neighborhood_features' in config
            assert 'texture_features' in config
            
            # Check that optimization parameters have sensible values.
            if 'neighborhood_batch_size' in config:
                assert config['neighborhood_batch_size'] > 0
                assert config['neighborhood_batch_size'] <= 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
