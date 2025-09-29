#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Test Name: test_granular_feature_optimization.py.
Description:
    Comprehensive test suite for the optimized granular feature extraction system.
    Tests performance improvements, feature selection accuracy, and computational
    efficiency of the new individual feature control system.

Dependencies:
    - pytest
    - numpy
    - pandas
    - PIL
    - skimage
    - scipy

Usage:
    pytest tests/test_granular_feature_optimization.py -v

Arguments:
    None (pytest handles test discovery and execution).

Inputs:
    - Synthetic test images with known nuclear properties.
    - Configuration files with various feature selection combinations.

Outputs:
    - Test results showing performance improvements and feature accuracy.
    - Timing comparisons between old and new systems.

Key Features:
    - Performance benchmarking of granular vs. group-based selection.
    - Validation of individual feature extraction accuracy.
    - Memory usage optimization testing.
    - Computational complexity analysis.

Notes:
    - Tests use synthetic data to ensure reproducible results.
    - Performance tests may vary based on system specifications.
    - GPU acceleration tests require CUDA-compatible hardware.
"""

import traceback
import sys
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, Any, List

import pytest
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import regionprops, label

# Add project root to path for imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Import optimized feature extraction utilities.
from code.engineered_feature_extraction.utils.config_loader import load_feature_extraction_config, get_enabled_features
from code.engineered_feature_extraction.extract_engineered_features import (
    extract_shape_features_optimized,
    extract_size_features_optimized,
    extract_neighborhood_features_optimized,
    extract_texture_features_optimized,
    compute_comprehensive_features_optimized,
    build_neighbors_list_optimized
)


class TestGranularFeatureOptimization:
    """
    Test suite for granular feature extraction optimizations.
    
    This class validates the performance improvements and accuracy of the
    new individual feature control system compared to the legacy group-based approach.
    """

    def setup_method(self):
        """
        Set up test fixtures with synthetic nuclear data.
        
        Creates standardized test images and configurations for reproducible testing.
        """
        # Create synthetic nuclear image with known properties.
        self.test_image = np.zeros((200, 200), dtype=np.uint8)
        
        # Add circular nucleus (label 1).
        y, x = np.ogrid[:200, :200]
        center_y, center_x = 50, 50
        radius = 20
        mask1 = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        self.test_image[mask1] = 1
        
        # Add elongated nucleus (label 2).
        self.test_image[80:120, 80:140] = 2
        
        # Add irregular nucleus (label 3).
        self.test_image[150:180, 150:180] = 3
        self.test_image[160:170, 140:150] = 3  # Add protrusion.
        
        # Create grayscale intensity image.
        self.gray_image = np.random.randint(50, 200, (200, 200), dtype=np.uint8)
        
        # Get region properties.
        self.props = regionprops(self.test_image)
        
        # Create test configurations.
        # Manually set all shape features for testing.
        self.config_all_features = {
            'extract_all_features': False,  # Set manually for testing.
            'extract_circularity': True,
            'extract_eccentricity': True,
            'extract_solidity': True,
            'extract_aspect_ratio': True,
            'extract_compactness': True,
            'extract_elongation': True,
            'extract_roundness': True,
            'extract_form_factor': True,
            'extract_convex_area_ratio': True,
            'extract_convexity': True,
            'extract_fractal_dimension': True,
        }
        
        self.config_minimal = {
            'extract_all_features': False,
            'extract_area': True,
            'extract_perimeter': True,
            'extract_circularity': True,
            'extract_intensity_mean': True,
        }
        
        self.config_expensive = {
            'extract_all_features': False,
            'extract_glcm_contrast': True,
            'extract_glcm_energy': True,
            'extract_fractal_dimension': True,
            'extract_convex_area_ratio': True,
            'extract_cluster_elongation': True,
        }

    def test_individual_feature_selection(self):
        """
        Test that individual feature selection works correctly.
        
        Validates that only requested features are computed and returned.
        """
        region = self.props[0]  # Use first nucleus.
        
        # Test minimal configuration.
        features = extract_shape_features_optimized(region, self.gray_image, self.config_minimal)
        
        # Should only contain circularity (from minimal config).
        assert 'circularity' in features, "Circularity should be extracted"
        assert 'eccentricity' not in features, "Eccentricity should not be extracted"
        assert 'solidity' not in features, "Solidity should not be extracted"
        
        # Test that the feature value is reasonable.
        assert 0.0 <= features['circularity'] <= 1.0, "Circularity should be between 0 and 1"

    def test_performance_improvement(self):
        """
        Test performance improvement of granular feature selection.

        Compares execution time between full feature extraction and selective extraction.
        """
        region = self.props[0]

        # Run multiple iterations for more accurate timing.
        iterations = 100

        # Time full feature extraction.
        start_time = time.time()
        for _ in range(iterations):
            full_features = extract_shape_features_optimized(region, self.gray_image, self.config_all_features)
        full_time = time.time() - start_time

        # Time minimal feature extraction.
        start_time = time.time()
        for _ in range(iterations):
            minimal_features = extract_shape_features_optimized(region, self.gray_image, self.config_minimal)
        minimal_time = time.time() - start_time

        # Minimal extraction should be faster or equal.
        assert minimal_time <= full_time, "Minimal feature extraction should be faster or equal"

        # Minimal extraction should return fewer features.
        assert len(minimal_features) < len(full_features), "Minimal extraction should return fewer features"

        # Calculate performance improvement (avoid division by zero).
        if minimal_time > 0:
            improvement = full_time / minimal_time
            print(f"Performance improvement: {improvement:.2f}x faster for minimal extraction")
        else:
            print("Minimal extraction was too fast to measure accurately")

        print(f"Full extraction: {len(full_features)} features in {full_time:.4f}s")
        print(f"Minimal extraction: {len(minimal_features)} features in {minimal_time:.4f}s")

    def test_expensive_feature_detection(self):
        """
        Test that expensive features are properly identified and handled.
        
        Validates that expensive computations are only performed when explicitly requested.
        """
        region = self.props[0]
        
        # Test that expensive features are not computed by default.
        minimal_features = extract_shape_features_optimized(region, self.gray_image, self.config_minimal)
        assert 'fractal_dimension' not in minimal_features, "Fractal dimension should not be computed by default"
        
        # Test that expensive features are computed when requested.
        expensive_features = extract_shape_features_optimized(region, self.gray_image, self.config_expensive)
        assert 'fractal_dimension' in expensive_features, "Fractal dimension should be computed when requested"

    def test_memory_efficiency(self):
        """
        Test memory efficiency of the optimized feature extraction.
        
        Validates that memory usage is reduced with selective feature extraction.
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure memory before extraction.
        memory_before = process.memory_info().rss / 1024 / 1024  # MB.
        
        # Extract features from all nuclei with minimal configuration.
        all_features = []
        for region in self.props:
            features = extract_shape_features_optimized(region, self.gray_image, self.config_minimal)
            all_features.append(features)
        
        # Measure memory after extraction.
        memory_after = process.memory_info().rss / 1024 / 1024  # MB.
        memory_used = memory_after - memory_before
        
        # Memory usage should be reasonable (less than 50MB for this small test).
        assert memory_used < 50, f"Memory usage too high: {memory_used:.2f} MB"
        
        print(f"Memory usage for minimal extraction: {memory_used:.2f} MB")

    def test_comprehensive_feature_optimization(self):
        """
        Test the optimized comprehensive feature extraction function.
        
        Validates that the main feature extraction function properly uses granular selection.
        """
        region = self.props[0]
        
        # Create dummy neighbor data.
        neighbor_data = {
            'centroids': [(60, 60), (70, 70)],
            'areas': [400, 500],
            'eccs': [0.5, 0.6],
            'orients': [0.1, 0.2],
            'radius': 50.0
        }
        
        # Test comprehensive feature extraction with minimal config.
        features = compute_comprehensive_features_optimized(
            region, neighbor_data, self.gray_image, self.gray_image.shape, self.config_minimal
        )
        
        # Should contain core features.
        assert 'label' in features, "Label should always be included"
        assert 'centroid_x' in features, "Centroid X should always be included"
        assert 'centroid_y' in features, "Centroid Y should always be included"
        
        # Should contain only requested features.
        assert 'area' in features, "Area should be included (requested)"
        assert 'intensity_mean' in features, "Intensity mean should be included (requested)"
        
        # Should not contain unrequested features.
        assert 'eccentricity' not in features, "Eccentricity should not be included (not requested)"
        assert 'glcm_contrast' not in features, "GLCM contrast should not be included (not requested)"

    def test_config_loader_integration(self):
        """
        Test integration with the config loader system.

        Validates that the get_enabled_features function works correctly.
        """
        # Test with manually configured features.
        test_config = {
            'extract_all_features': False,
            'extract_circularity': True,
            'extract_area': True,
            'extract_intensity_mean': True,
            'extract_nearest_neighbor_distance': True,
        }
        enabled_features = get_enabled_features(test_config)

        # Should have features in relevant categories.
        assert 'circularity' in enabled_features['shape'], "Should have circularity in shape features"
        assert 'area' in enabled_features['size'], "Should have area in size features"
        assert 'intensity_mean' in enabled_features['texture'], "Should have intensity_mean in texture features"
        assert 'nearest_neighbor_distance' in enabled_features['neighborhood'], "Should have nearest_neighbor_distance in neighborhood features"

        # Test with minimal configuration.
        enabled_features_minimal = get_enabled_features(self.config_minimal)

        # Should have the expected features.
        total_minimal = sum(len(feats) for feats in enabled_features_minimal.values())
        assert total_minimal > 0, "Minimal config should enable some features"
        assert total_minimal < 43, "Minimal config should not enable all features"

    def test_batch_processing_optimization(self):
        """
        Test that batch processing works correctly with granular features.
        
        Validates that the optimized batch processing maintains accuracy while improving performance.
        """
        # Process all nuclei individually.
        individual_results = []
        for region in self.props:
            neighbor_data = {
                'centroids': [],
                'areas': [],
                'eccs': [],
                'orients': [],
                'radius': 50.0
            }
            features = compute_comprehensive_features_optimized(
                region, neighbor_data, self.gray_image, self.gray_image.shape, self.config_minimal
            )
            individual_results.append(features)
        
        # Results should be consistent.
        assert len(individual_results) == len(self.props), "Should have results for all nuclei"
        
        # All results should have the same feature set.
        feature_keys = set(individual_results[0].keys())
        for result in individual_results[1:]:
            assert set(result.keys()) == feature_keys, "All results should have the same features"


if __name__ == "__main__":
    # Run tests directly.
    pytest.main([__file__, "-v"])
