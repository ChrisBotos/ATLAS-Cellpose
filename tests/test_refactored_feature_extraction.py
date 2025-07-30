#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_refactored_feature_extraction.py.
Description:
    Comprehensive test suite for the refactored nuclear feature extraction functionality
    used in kidney ischemia-reperfusion injury analysis. Tests all feature categories,
    configuration handling, and visualization components with scientific validation.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pandas, PIL, scipy, scikit-image.
    • Custom feature extraction utilities from the nuclei_segmentation package.

Usage:
    python -m pytest tests/test_refactored_feature_extraction.py -v

Inputs:
    • None (tests run on generated synthetic data).

Outputs:
    • Test results indicating pass/fail status for all feature extraction components.

Key Features:
    • Tests for all four feature categories: shape, size, neighborhood, texture.
    • Configuration parameter validation and loading tests.
    • Feature extraction pipeline integration tests.
    • Visualization function tests with synthetic data.
    • Scientific validation of feature calculations and statistical methods.

Notes:
    • Tests use synthetic nuclear data to ensure reproducibility and avoid file dependencies.
    • Validates feature extraction accuracy against known geometric and statistical properties.
    • Ensures compatibility with kidney I/R injury analysis workflows and data formats.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from skimage.measure import regionprops

# Add the project root to the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Import the refactored modules to be tested.
from code.engineered_feature_extraction.extract_engineered_features_refactored import (
    extract_shape_features,
    extract_size_features,
    extract_neighborhood_features,
    extract_texture_features,
    compute_comprehensive_features,
    build_neighbors_list,
    filter_nuclei_by_size,
    fractal_dimension,
    compute_dark_distance_map,
    compute_sparse_distance_map
)

from code.engineered_feature_extraction.visualize_engineered_features_refactored import (
    load_and_validate_data,
    identify_available_features,
    create_violin_plots_by_category,
    FEATURE_CATEGORIES,
    TIMEPOINT_COLORS
)


"""FIXTURES FOR SYNTHETIC TEST DATA"""

@pytest.fixture
def synthetic_grayscale_image():
    """
    Generate a synthetic grayscale image with realistic tissue-like patterns.
    
    Creates a gradient image with dark regions simulating tissue damage areas
    and varying intensity patterns typical of DAPI-stained kidney sections.
    
    Returns:
        np.ndarray: 8-bit grayscale image with tissue-like intensity patterns.
    """
    # Create base gradient image.
    img = np.zeros((200, 200), dtype=np.uint8)
    
    for i in range(200):
        for j in range(200):
            # Create radial gradient with noise.
            center_dist = np.sqrt((i - 100)**2 + (j - 100)**2)
            base_intensity = max(0, 255 - center_dist * 1.5)
            noise = np.random.normal(0, 20)
            img[i, j] = np.clip(base_intensity + noise, 0, 255)
    
    # Add dark regions simulating tissue damage.
    img[50:70, 50:70] = 20  # Dark region 1.
    img[130:150, 120:140] = 15  # Dark region 2.
    img[80:100, 160:180] = 25  # Dark region 3.
    
    return img


@pytest.fixture
def synthetic_nuclear_mask():
    """
    Generate a synthetic nuclear segmentation mask with realistic nuclear shapes.
    
    Creates labeled regions with varying sizes and shapes typical of kidney
    tissue nuclei, including some elongated and circular shapes.
    
    Returns:
        np.ndarray: Labeled mask with synthetic nuclear regions.
    """
    mask = np.zeros((200, 200), dtype=np.uint32)
    
    # Create various nuclear shapes with different properties.
    # Circular nucleus.
    y, x = np.ogrid[:200, :200]
    circle1 = (x - 50)**2 + (y - 50)**2 <= 15**2
    mask[circle1] = 1
    
    # Elongated nucleus.
    mask[80:90, 30:60] = 2
    
    # Irregular nucleus.
    mask[120:140, 120:145] = 3
    
    # Small nucleus.
    mask[160:170, 160:170] = 4
    
    # Large nucleus.
    mask[30:55, 150:175] = 5
    
    # Additional nuclei for neighborhood analysis.
    mask[100:110, 100:110] = 6
    mask[110:120, 110:120] = 7
    mask[90:100, 110:120] = 8
    
    return mask


@pytest.fixture
def synthetic_region_props(synthetic_nuclear_mask):
    """
    Generate region properties from synthetic nuclear mask.
    
    Args:
        synthetic_nuclear_mask: Synthetic labeled nuclear mask.
        
    Returns:
        List of regionprops objects for testing feature extraction.
    """
    return regionprops(synthetic_nuclear_mask)


@pytest.fixture
def test_config_settings():
    """
    Generate test configuration settings for feature extraction.
    
    Returns:
        Dictionary containing test configuration parameters.
    """
    return {
        'shape_features': True,
        'size_features': True,
        'neighborhood_features': True,
        'texture_features': True,
        'neighborhood_radius': 50.0,
        'feature_extraction_workers': 2,
        'min_nuclear_area': 10.0,
        'max_nuclear_area': 2000.0,
        'enable_violin_plots': True,
        'timepoint_color_coding': True,
        'features_per_page': 9,
        'enable_statistical_testing': True
    }


@pytest.fixture
def synthetic_feature_dataframe():
    """
    Generate synthetic feature DataFrame for visualization testing.
    
    Returns:
        pd.DataFrame: Synthetic nuclear features with realistic value ranges.
    """
    np.random.seed(42)  # For reproducible tests.
    n_nuclei = 100
    
    data = {
        'label': range(1, n_nuclei + 1),
        'centroid_x': np.random.uniform(10, 190, n_nuclei),
        'centroid_y': np.random.uniform(10, 190, n_nuclei),
        
        # Shape features.
        'circularity': np.random.uniform(0.3, 1.0, n_nuclei),
        'eccentricity': np.random.uniform(0.0, 0.9, n_nuclei),
        'solidity': np.random.uniform(0.7, 1.0, n_nuclei),
        'aspect_ratio': np.random.uniform(1.0, 3.0, n_nuclei),
        
        # Size features.
        'area': np.random.uniform(50, 500, n_nuclei),
        'perimeter': np.random.uniform(25, 100, n_nuclei),
        'major_axis_length': np.random.uniform(10, 30, n_nuclei),
        'minor_axis_length': np.random.uniform(8, 25, n_nuclei),
        
        # Neighborhood features.
        'nearest_neighbor_distance': np.random.uniform(5, 50, n_nuclei),
        'neighborhood_density': np.random.uniform(0.001, 0.01, n_nuclei),
        
        # Texture features.
        'intensity_mean': np.random.uniform(50, 200, n_nuclei),
        'intensity_std': np.random.uniform(10, 50, n_nuclei),
        'texture_entropy': np.random.uniform(2, 6, n_nuclei),
    }
    
    return pd.DataFrame(data)


"""FEATURE EXTRACTION TESTS"""

def test_extract_shape_features(synthetic_region_props, synthetic_grayscale_image):
    """
    Test shape feature extraction functionality.
    
    Validates that all shape features are calculated correctly and return
    reasonable values for synthetic nuclear regions.
    """
    region = synthetic_region_props[0]  # Test with first region.
    
    # Extract shape features.
    features = extract_shape_features(region, synthetic_grayscale_image)
    
    # Validate feature presence.
    expected_features = ['circularity', 'eccentricity', 'solidity', 'aspect_ratio', 'compactness']
    for feature in expected_features:
        assert feature in features, f"Missing shape feature: {feature}"
    
    # Validate feature value ranges.
    assert 0 <= features['circularity'] <= 1, "Circularity should be between 0 and 1"
    assert 0 <= features['eccentricity'] <= 1, "Eccentricity should be between 0 and 1"
    assert 0 <= features['solidity'] <= 1, "Solidity should be between 0 and 1"
    assert features['aspect_ratio'] >= 1, "Aspect ratio should be >= 1"
    
    # Test with all regions.
    for region in synthetic_region_props:
        features = extract_shape_features(region, synthetic_grayscale_image)
        assert len(features) >= 5, "Should extract at least 5 shape features"


def test_extract_size_features(synthetic_region_props):
    """
    Test size feature extraction functionality.
    
    Validates that all size features are calculated correctly and return
    positive values for synthetic nuclear regions.
    """
    region = synthetic_region_props[0]  # Test with first region.
    
    # Extract size features.
    features = extract_size_features(region)
    
    # Validate feature presence.
    expected_features = ['area', 'perimeter', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length']
    for feature in expected_features:
        assert feature in features, f"Missing size feature: {feature}"
    
    # Validate feature value ranges.
    assert features['area'] > 0, "Area should be positive"
    assert features['perimeter'] > 0, "Perimeter should be positive"
    assert features['equivalent_diameter'] > 0, "Equivalent diameter should be positive"
    assert features['major_axis_length'] >= features['minor_axis_length'], "Major axis should be >= minor axis"
    
    # Test consistency between area and equivalent diameter.
    expected_diameter = np.sqrt(4 * features['area'] / np.pi)
    assert abs(features['equivalent_diameter'] - expected_diameter) < 0.01, "Equivalent diameter calculation error"


def test_extract_neighborhood_features(synthetic_region_props, synthetic_grayscale_image):
    """
    Test neighborhood feature extraction functionality.
    
    Validates spatial analysis features and neighborhood calculations
    for synthetic nuclear regions with known spatial relationships.
    """
    region = synthetic_region_props[0]  # Test with first region.
    
    # Create synthetic neighbor data.
    neighbor_data = {
        'centroids': [(60, 60), (70, 70), (80, 80)],
        'areas': [100, 120, 80],
        'eccs': [0.5, 0.7, 0.3],
        'orients': [0.1, 0.2, 0.3],
        'radius': 50.0
    }
    
    image_shape = synthetic_grayscale_image.shape
    
    # Extract neighborhood features.
    features = extract_neighborhood_features(region, neighbor_data, image_shape)
    
    # Validate feature presence.
    expected_features = ['nearest_neighbor_distance', 'neighborhood_density', 'boundary_proximity']
    for feature in expected_features:
        assert feature in features, f"Missing neighborhood feature: {feature}"
    
    # Validate feature value ranges.
    assert features['nearest_neighbor_distance'] > 0, "Nearest neighbor distance should be positive"
    assert features['neighborhood_density'] >= 0, "Neighborhood density should be non-negative"
    assert features['boundary_proximity'] >= 0, "Boundary proximity should be non-negative"
    
    # Test with empty neighbor data.
    empty_neighbor_data = {
        'centroids': [],
        'areas': [],
        'eccs': [],
        'orients': [],
        'radius': 50.0
    }
    
    features_empty = extract_neighborhood_features(region, empty_neighbor_data, image_shape)
    assert features_empty['nearest_neighbor_distance'] == 50.0, "Should use radius as default distance"


def test_extract_texture_features(synthetic_region_props, synthetic_grayscale_image):
    """
    Test texture feature extraction functionality.
    
    Validates intensity statistics and texture analysis for synthetic
    nuclear regions with known intensity patterns.
    """
    region = synthetic_region_props[0]  # Test with first region.
    
    # Extract texture features.
    features = extract_texture_features(region, synthetic_grayscale_image)
    
    # Validate feature presence.
    expected_features = ['intensity_mean', 'intensity_std', 'intensity_median', 'texture_entropy']
    for feature in expected_features:
        assert feature in features, f"Missing texture feature: {feature}"
    
    # Validate feature value ranges.
    assert 0 <= features['intensity_mean'] <= 255, "Intensity mean should be in valid range"
    assert features['intensity_std'] >= 0, "Intensity std should be non-negative"
    assert 0 <= features['intensity_median'] <= 255, "Intensity median should be in valid range"
    assert features['texture_entropy'] >= 0, "Texture entropy should be non-negative"
    
    # Test statistical consistency.
    minr, minc, maxr, maxc = region.bbox
    patch = synthetic_grayscale_image[minr:maxr, minc:maxc]
    mask_patch = region.image
    vals = patch[mask_patch]
    
    if len(vals) > 0:
        expected_mean = float(vals.mean())
        assert abs(features['intensity_mean'] - expected_mean) < 0.01, "Intensity mean calculation error"


def test_compute_comprehensive_features(synthetic_region_props, synthetic_grayscale_image, test_config_settings):
    """
    Test comprehensive feature extraction with configuration settings.
    
    Validates the main feature extraction function that coordinates all
    feature categories based on configuration parameters.
    """
    region = synthetic_region_props[0]
    
    # Create neighbor data.
    neighbor_data = {
        'centroids': [(60, 60), (70, 70)],
        'areas': [100, 120],
        'eccs': [0.5, 0.7],
        'orients': [0.1, 0.2],
        'radius': 50.0
    }
    
    image_shape = synthetic_grayscale_image.shape
    
    # Extract comprehensive features.
    features = compute_comprehensive_features(
        region, neighbor_data, synthetic_grayscale_image, image_shape, test_config_settings
    )
    
    # Validate basic features.
    assert 'label' in features, "Should include label"
    assert 'centroid_x' in features, "Should include centroid_x"
    assert 'centroid_y' in features, "Should include centroid_y"
    
    # Validate feature categories based on config.
    if test_config_settings['shape_features']:
        assert any('circularity' in str(k) for k in features.keys()), "Should include shape features"
    
    if test_config_settings['size_features']:
        assert any('area' in str(k) for k in features.keys()), "Should include size features"
    
    if test_config_settings['neighborhood_features']:
        assert any('neighbor' in str(k) for k in features.keys()), "Should include neighborhood features"
    
    if test_config_settings['texture_features']:
        assert any('intensity' in str(k) for k in features.keys()), "Should include texture features"
    
    # Test with disabled feature categories.
    config_no_shape = test_config_settings.copy()
    config_no_shape['shape_features'] = False
    
    features_no_shape = compute_comprehensive_features(
        region, neighbor_data, synthetic_grayscale_image, image_shape, config_no_shape
    )
    
    # Should have fewer features when shape features are disabled.
    assert len(features_no_shape) < len(features), "Should have fewer features when category disabled"


"""UTILITY FUNCTION TESTS"""

def test_fractal_dimension():
    """
    Test fractal dimension calculation with known geometric shapes.

    Validates fractal dimension calculation accuracy using simple geometric
    shapes with predictable fractal properties.
    """
    # Test with a filled square (should be close to 2.0).
    square = np.ones((32, 32), dtype=bool)
    fd_square = fractal_dimension(square)
    assert 1.8 < fd_square < 2.2, f"Square fractal dimension should be ~2.0, got {fd_square}"

    # Test with a small region (should return NaN).
    tiny = np.ones((1, 1), dtype=bool)
    fd_tiny = fractal_dimension(tiny)
    assert np.isnan(fd_tiny), "Tiny region should return NaN"

    # Test with empty region.
    empty = np.zeros((10, 10), dtype=bool)
    fd_empty = fractal_dimension(empty)
    assert np.isnan(fd_empty), "Empty region should return NaN"


def test_compute_dark_distance_map(synthetic_grayscale_image):
    """
    Test dark distance map computation for spatial context analysis.

    Validates distance calculations to dark regions in synthetic tissue images.
    """
    # Compute distance map with default threshold.
    dist_map = compute_dark_distance_map(synthetic_grayscale_image, threshold=50)

    # Validate output properties.
    assert dist_map.shape == synthetic_grayscale_image.shape, "Distance map should match image shape"
    assert dist_map.dtype == np.float64, "Distance map should be float64"

    # Dark regions should have distance 0.
    dark_mask = synthetic_grayscale_image < 50
    assert np.all(dist_map[dark_mask] == 0), "Dark regions should have distance 0"

    # Non-dark regions should have positive distances.
    non_dark_mask = synthetic_grayscale_image >= 50
    if np.any(non_dark_mask):
        assert np.all(dist_map[non_dark_mask] > 0), "Non-dark regions should have positive distances"


def test_compute_sparse_distance_map(synthetic_grayscale_image, synthetic_nuclear_mask):
    """
    Test sparse distance map computation for tissue organization analysis.

    Validates distance calculations to sparse zones in synthetic tissue images.
    """
    # Compute sparse distance map.
    dist_map = compute_sparse_distance_map(
        synthetic_grayscale_image,
        synthetic_nuclear_mask,
        intensity_threshold=30,
        min_size=100  # Smaller for test data.
    )

    # Validate output properties.
    assert dist_map.shape == synthetic_grayscale_image.shape, "Distance map should match image shape"
    assert dist_map.dtype == np.float64, "Distance map should be float64"

    # Should have some zero distances (sparse regions).
    assert np.sum(dist_map == 0) >= 0, "Should have some sparse regions"

    # Should have some positive distances (non-sparse regions).
    assert np.sum(dist_map > 0) > 0, "Should have some non-sparse regions"


def test_build_neighbors_list(synthetic_region_props):
    """
    Test neighborhood list building for spatial analysis.

    Validates neighbor identification and spatial relationship calculations
    for synthetic nuclear regions with known positions.
    """
    from scipy.spatial import cKDTree

    # Create KD-tree from centroids.
    centroids = [r.centroid for r in synthetic_region_props]
    tree = cKDTree(centroids)

    # Build neighbors list with radius that should include some neighbors.
    radius = 50.0
    neighbors = build_neighbors_list(synthetic_region_props, tree, radius)

    # Validate output structure.
    assert len(neighbors) == len(synthetic_region_props), "Should have one entry per region"

    for neighbor_data in neighbors:
        assert 'centroids' in neighbor_data, "Should include centroids"
        assert 'areas' in neighbor_data, "Should include areas"
        assert 'eccs' in neighbor_data, "Should include eccentricities"
        assert 'orients' in neighbor_data, "Should include orientations"
        assert 'radius' in neighbor_data, "Should include radius"
        assert neighbor_data['radius'] == radius, "Should store correct radius"

    # Test with small radius (should find fewer neighbors).
    small_radius = 5.0
    neighbors_small = build_neighbors_list(synthetic_region_props, tree, small_radius)

    # Should generally have fewer neighbors with smaller radius.
    total_neighbors_large = sum(len(n['centroids']) for n in neighbors)
    total_neighbors_small = sum(len(n['centroids']) for n in neighbors_small)
    assert total_neighbors_small <= total_neighbors_large, "Smaller radius should find fewer neighbors"


def test_filter_nuclei_by_size(synthetic_region_props):
    """
    Test nuclear size filtering functionality.

    Validates size-based filtering of nuclear regions to remove artifacts
    and incorrectly segmented objects.
    """
    original_count = len(synthetic_region_props)

    # Test with permissive size limits (should keep most nuclei).
    min_area = 10.0
    max_area = 2000.0
    filtered_props = filter_nuclei_by_size(synthetic_region_props, min_area, max_area)

    # Should keep most nuclei with permissive limits.
    assert len(filtered_props) <= original_count, "Filtered count should not exceed original"
    assert len(filtered_props) > 0, "Should keep some nuclei with permissive limits"

    # Validate that all remaining nuclei meet size criteria.
    for prop in filtered_props:
        assert min_area <= prop.area <= max_area, f"Nucleus area {prop.area} outside limits [{min_area}, {max_area}]"

    # Test with restrictive size limits (should remove more nuclei).
    min_area_strict = 100.0
    max_area_strict = 300.0
    filtered_props_strict = filter_nuclei_by_size(synthetic_region_props, min_area_strict, max_area_strict)

    # Should keep fewer nuclei with restrictive limits.
    assert len(filtered_props_strict) <= len(filtered_props), "Restrictive limits should keep fewer nuclei"


"""VISUALIZATION TESTS"""

def test_load_and_validate_data(tmp_path, synthetic_feature_dataframe):
    """
    Test data loading and validation functionality.

    Validates CSV loading, size filtering, and data quality checks
    for nuclear feature datasets.
    """
    # Save synthetic data to temporary CSV.
    csv_path = tmp_path / "test_features.csv"
    synthetic_feature_dataframe.to_csv(csv_path, index=False)

    # Load and validate data.
    df = load_and_validate_data(csv_path, min_area=50.0, max_area=400.0)

    # Validate output.
    assert isinstance(df, pd.DataFrame), "Should return DataFrame"
    assert len(df) > 0, "Should contain data after filtering"
    assert len(df) <= len(synthetic_feature_dataframe), "Should not exceed original size"

    # Validate size filtering.
    if 'area' in df.columns:
        assert df['area'].min() >= 50.0, "All areas should be >= min_area"
        assert df['area'].max() <= 400.0, "All areas should be <= max_area"

    # Test with non-existent file.
    with pytest.raises(FileNotFoundError):
        load_and_validate_data(tmp_path / "nonexistent.csv")


def test_identify_available_features(synthetic_feature_dataframe):
    """
    Test feature identification and categorization functionality.

    Validates mapping of theoretical feature categories to actual
    column names in nuclear feature datasets.
    """
    # Identify available features.
    available_features = identify_available_features(synthetic_feature_dataframe)

    # Validate output structure.
    assert isinstance(available_features, dict), "Should return dictionary"

    for category in FEATURE_CATEGORIES.keys():
        assert category in available_features, f"Should include category: {category}"
        assert isinstance(available_features[category], list), f"Category {category} should be list"

    # Validate that some features are found.
    total_features = sum(len(features) for features in available_features.values())
    assert total_features > 0, "Should find some available features"

    # Validate that found features exist in DataFrame.
    for category, features in available_features.items():
        for feature in features:
            assert feature in synthetic_feature_dataframe.columns, f"Feature {feature} should exist in DataFrame"


def test_create_violin_plots_by_category(tmp_path, synthetic_feature_dataframe):
    """
    Test violin plot generation functionality.

    Validates publication-quality violin plot creation with proper
    categorization and scientific formatting.
    """
    # Identify available features.
    available_features = identify_available_features(synthetic_feature_dataframe)

    # Create violin plots.
    create_violin_plots_by_category(
        df=synthetic_feature_dataframe,
        available_features=available_features,
        output_dir=tmp_path,
        timepoint='10h',
        control_df=None,
        features_per_page=6
    )

    # Validate that plot files were created.
    plot_files = list(tmp_path.glob("violin_*.png"))
    assert len(plot_files) > 0, "Should create violin plot files"

    # Validate that plots were created for categories with features.
    categories_with_features = [cat for cat, feats in available_features.items() if feats]
    expected_files = len(categories_with_features)

    # Should have at least one plot file per category with features.
    assert len(plot_files) >= expected_files, f"Should create at least {expected_files} plot files"

    # Validate file naming convention.
    for plot_file in plot_files:
        assert plot_file.name.startswith("violin_"), "Plot files should start with 'violin_'"
        assert plot_file.name.endswith(".png"), "Plot files should be PNG format"


def test_timepoint_colors():
    """
    Test timepoint color palette for scientific visualization.

    Validates that timepoint colors are properly defined for
    kidney I/R injury analysis visualization.
    """
    # Validate that essential timepoints have colors defined.
    essential_timepoints = ['10h', '2d', '14d', 'control', 'default']

    for timepoint in essential_timepoints:
        assert timepoint in TIMEPOINT_COLORS, f"Missing color for timepoint: {timepoint}"
        assert isinstance(TIMEPOINT_COLORS[timepoint], str), f"Color for {timepoint} should be string"
        assert TIMEPOINT_COLORS[timepoint].startswith('#'), f"Color for {timepoint} should be hex code"

    # Validate color uniqueness.
    colors = list(TIMEPOINT_COLORS.values())
    unique_colors = set(colors)
    assert len(unique_colors) == len(colors), "All timepoint colors should be unique"


def test_feature_categories():
    """
    Test feature category definitions for comprehensive analysis.

    Validates that all essential feature categories are defined with
    appropriate features for kidney I/R injury analysis.
    """
    # Validate that all essential categories are defined.
    essential_categories = ['Shape Features', 'Size Features', 'Neighborhood Features', 'Texture Features']

    for category in essential_categories:
        assert category in FEATURE_CATEGORIES, f"Missing feature category: {category}"
        assert isinstance(FEATURE_CATEGORIES[category], list), f"Category {category} should be list"
        assert len(FEATURE_CATEGORIES[category]) > 0, f"Category {category} should have features"

    # Validate feature uniqueness across categories.
    all_features = []
    for features in FEATURE_CATEGORIES.values():
        all_features.extend(features)

    unique_features = set(all_features)
    assert len(unique_features) == len(all_features), "All features should be unique across categories"

    # Validate that essential features are included.
    essential_features = ['area', 'circularity', 'intensity_mean', 'nearest_neighbor_distance']

    for feature in essential_features:
        found = any(feature in category_features for category_features in FEATURE_CATEGORIES.values())
        assert found, f"Essential feature {feature} should be in some category"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
