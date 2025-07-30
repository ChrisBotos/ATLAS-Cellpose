#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: feature_extraction_test.py.
Description:
    Test suite for the nuclei feature extraction functionality used in the analysis of
    tissue after ischemia-reperfusion injury. Tests the extraction of morphological,
    intensity, texture, and spatial features from segmented nuclei.

Dependencies:
    • Python >= 3.9.
    • numpy, pytest, pandas, PIL, scipy, scikit-image.
    • Custom feature extraction utilities from the nuclei_segmentation package.

Usage:
    python -m pytest tests/nuclei_segmentation_tests/feature_extraction_test.py -v

Inputs:
    • None (tests run on generated test data).

Outputs:
    • Test results indicating pass/fail status.

Key Features:
    • Tests for distance map computation (dark regions, sparse zones).
    • Tests for fractal dimension calculation.
    • Tests for region feature extraction with various input types.
    • Tests for neighborhood analysis and spatial context features.
    • Tests for the full feature extraction pipeline.

Notes:
    • These tests verify the correct behavior of feature extraction functions essential for
      quantitative analysis of nuclear morphology in tissue I/R injury studies.
    • Uses synthetic data to ensure reproducibility and avoid dependencies on real images.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from PIL import Image
from skimage.measure import regionprops

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import the module to be tested
from code.engineered_feature_extraction.extract_engineered_features import (
    compute_dark_distance_map,
    compute_sparse_distance_map,
    fractal_dimension,
    compute_region_features,
    build_neighbors_list,
    process_image
)


"""FIXTURES"""

@pytest.fixture
def dummy_grayscale_image():
    """
    Generate a synthetic grayscale image with varying intensities.
    
    Returns:
        np.ndarray: 8-bit grayscale image with gradient and some dark regions.
    """
    # Create a gradient image
    img = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        img[:, i] = int(i * 2.55)  # 0-255 gradient
    
    # Add some dark regions
    img[20:30, 20:30] = 10  # Dark region 1
    img[70:85, 60:80] = 20  # Dark region 2
    
    return img


@pytest.fixture
def dummy_binary_mask():
    """
    Generate a binary mask with a few labeled regions.
    
    Returns:
        np.ndarray: Binary mask with labeled regions.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    # Add some "nuclei"
    mask[25:35, 25:35] = 1  # Square nucleus
    mask[60:70, 30:45] = 2  # Rectangular nucleus
    mask[40:50, 60:75] = 3  # Another nucleus
    
    return mask


@pytest.fixture
def dummy_region():
    """
    Create a regionprop object for testing feature extraction.
    
    Returns:
        object: A region properties object from skimage.
    """
    # Create a small binary mask for a single nucleus
    binary = np.zeros((20, 20), dtype=bool)
    binary[5:15, 5:15] = True  # 10x10 square
    
    # Create a labeled image
    labeled = np.zeros((20, 20), dtype=np.int32)
    labeled[binary] = 1
    
    # Get the region properties
    return regionprops(labeled)[0]


@pytest.fixture
def dummy_neighbor_data():
    """
    Create dummy neighborhood data for a nucleus.
    
    Returns:
        dict: Dictionary with centroids, areas, eccentricities, and orientations.
    """
    return {
        'centroids': [(10, 30), (15, 25), (20, 20)],
        'areas': [100, 120, 80],
        'eccs': [0.5, 0.7, 0.3],
        'orients': [0.1, 0.2, 0.3],
        'radius': 25.0
    }


@pytest.fixture
def temp_image_and_mask(tmp_path, dummy_grayscale_image, dummy_binary_mask):
    """
    Save temporary image and mask files for testing.
    
    Returns:
        tuple: Paths to the image and mask files.
    """
    image_path = tmp_path / "test_image.tif"
    mask_path = tmp_path / "test_mask.npy"
    
    # Save the image as TIFF
    Image.fromarray(dummy_grayscale_image).save(image_path)
    
    # Save the mask as numpy array
    np.save(mask_path, dummy_binary_mask)
    
    return image_path, mask_path


"""TESTS"""

def test_compute_dark_distance_map(dummy_grayscale_image):
    """
    Test that the dark distance map correctly identifies distances from dark regions.
    
    Args:
        dummy_grayscale_image: Test grayscale image.
    """
    # Compute distance map with default threshold
    dist_map = compute_dark_distance_map(dummy_grayscale_image)
    
    # Check output properties
    assert dist_map.shape == dummy_grayscale_image.shape
    assert dist_map.dtype == np.float64
    
    # Dark regions should have distance 0
    dark_mask = dummy_grayscale_image < 50
    assert np.all(dist_map[dark_mask] == 0)
    
    # Non-dark regions should have positive distances
    assert np.all(dist_map[~dark_mask] > 0)


def test_compute_sparse_distance_map(dummy_grayscale_image, dummy_binary_mask):
    """
    Test that the sparse distance map correctly identifies distances from sparse regions.
    
    Args:
        dummy_grayscale_image: Test grayscale image.
        dummy_binary_mask: Test binary mask.
    """
    # Compute sparse distance map
    dist_map = compute_sparse_distance_map(
        dummy_grayscale_image, 
        dummy_binary_mask,
        intensity_threshold=20,
        min_size=25  # Smaller for test data
    )
    
    # Check output properties
    assert dist_map.shape == dummy_grayscale_image.shape
    assert dist_map.dtype == np.float64
    
    # Verify that some distances are zero (sparse regions)
    assert np.sum(dist_map == 0) > 0
    
    # Verify that some distances are positive (non-sparse regions)
    assert np.sum(dist_map > 0) > 0


def test_fractal_dimension():
    """
    Test fractal dimension calculation with various shapes.
    """
    # Test with a simple square (should be close to 2.0 for a filled square)
    square = np.ones((32, 32), dtype=bool)
    fd_square = fractal_dimension(square)
    assert 1.9 < fd_square < 2.1
    
    # Test with a small region (should return NaN)
    tiny = np.ones((1, 1), dtype=bool)
    fd_tiny = fractal_dimension(tiny)
    assert np.isnan(fd_tiny)
    
    # Test with a more complex shape (fractal-like)
    complex_shape = np.zeros((64, 64), dtype=bool)
    for i in range(0, 64, 4):
        for j in range(0, 64, 4):
            if (i//4 + j//4) % 2 == 0:
                complex_shape[i:i+2, j:j+2] = True
    
    fd_complex = fractal_dimension(complex_shape)
    assert 1.5 < fd_complex < 2.0  # Should be between 1 and 2


def test_compute_region_features(dummy_region, dummy_neighbor_data, dummy_grayscale_image):
    """
    Test feature extraction for a single nucleus region.
    
    Args:
        dummy_region: A region properties object.
        dummy_neighbor_data: Neighborhood data for the region.
        dummy_grayscale_image: Test grayscale image.
    """
    # Create distance maps
    dark_map = compute_dark_distance_map(dummy_grayscale_image)
    sparse_map = compute_sparse_distance_map(dummy_grayscale_image, np.zeros_like(dummy_grayscale_image))
    
    # Extract features
    features = compute_region_features(
        dummy_region,
        dummy_neighbor_data,
        dark_map,
        sparse_map,
        dummy_grayscale_image,
        dummy_grayscale_image.shape
    )
    
    # Check that all expected features are present
    expected_features = [
        'Label', 'Centroid_X', 'Centroid_Y', 'Area', 'Perimeter',
        'Major_Axis_Length', 'Minor_Axis_Length', 'Aspect_Ratio',
        'Circularity', 'Eccentricity', 'Solidity', 'Feret_Diameter',
        'Roughness_Index', 'Bounding_Box_Width', 'Bounding_Box_Height',
        'Fractal_Dimension', 'Intensity_Mean', 'Intensity_Std',
        'Intensity_Median', 'Intensity_Skewness', 'Intensity_Kurtosis',
        'Texture_Entropy', 'Distance_to_Image_Center', 'Distance_to_Image_Edge',
        'Distance_to_Dark_Region', 'Distance_to_Sparse_Zone',
        'Neighborhood_Mean_Area', 'Neighborhood_Std_Area',
        'Neighborhood_Eccentricity_Mean', 'Orientation_Alignment_Std',
        'Distance_to_Nearest_Nucleus', 'Cluster_Density_Index',
        'Cluster_Elongation', 'Cluster_Polarization_Score',
        'Cluster_Area_Ratio'
    ]
    
    # Check for LBP features
    for i in range(11):
        expected_features.append(f'LBP_Bin_{i}')
    
    for feature in expected_features:
        assert feature in features
    
    # Check some specific feature values
    assert features['Label'] == dummy_region.label
    assert features['Area'] == dummy_region.area
    assert features['Neighborhood_Mean_Area'] == np.mean(dummy_neighbor_data['areas'])


def test_build_neighbors_list():
    """
    Test building the neighbors list for spatial context analysis.
    """
    # Create a simple labeled image with a few objects
    labeled = np.zeros((100, 100), dtype=np.int32)
    labeled[20:30, 20:30] = 1
    labeled[40:50, 40:50] = 2
    labeled[60:70, 60:70] = 3
    
    # Get region properties
    props = regionprops(labeled)
    
    # Create KD-tree from centroids
    from scipy.spatial import cKDTree
    centroids = [r.centroid for r in props]
    tree = cKDTree(centroids)
    
    # Build neighbors list with radius that should include some neighbors
    radius = 30.0
    neighbors = build_neighbors_list(props, tree, radius)
    
    # Check that we have the right number of entries
    assert len(neighbors) == len(props)
    
    # Check that at least one region has neighbors
    has_neighbors = False
    for n in neighbors:
        if len(n['centroids']) > 0:
            has_neighbors = True
            break
    
    assert has_neighbors, "At least one region should have neighbors within the radius"
    
    # Test with a very small radius (should find no neighbors)
    small_radius = 5.0
    neighbors_small = build_neighbors_list(props, tree, small_radius)
    
    # Check that no region has neighbors with the small radius
    for n in neighbors_small:
        assert len(n['centroids']) == 0


@pytest.mark.parametrize("jobs", [1, 2, -1])
def test_process_image(temp_image_and_mask, jobs):
    """
    Test the full feature extraction pipeline.
    
    Args:
        temp_image_and_mask: Tuple with paths to test image and mask.
        jobs: Number of parallel workers to use.
    """
    image_path, mask_path = temp_image_and_mask
    output_path = image_path.parent / "features.csv"
    
    # Process the image
    df = process_image(
        image_path,
        mask_path,
        output_path,
        neighbor_radius=25.0,
        jobs=jobs
    )
    
    # Check that the output is a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Check that the CSV file was created
    assert output_path.exists()
    
    # Check that we have one row per labeled region in the mask
    mask = np.load(mask_path)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    assert len(df) == len(unique_labels)
    
    # Check that all expected columns are present
    expected_columns = [
        'Label', 'Centroid_X', 'Centroid_Y', 'Area', 'Perimeter',
        'Major_Axis_Length', 'Minor_Axis_Length', 'Aspect_Ratio',
        'Circularity', 'Eccentricity', 'Solidity', 'Feret_Diameter',
        'Roughness_Index', 'Bounding_Box_Width', 'Bounding_Box_Height',
        'Fractal_Dimension', 'Intensity_Mean', 'Intensity_Std',
        'Intensity_Median', 'Intensity_Skewness', 'Intensity_Kurtosis',
        'Texture_Entropy', 'Distance_to_Image_Center', 'Distance_to_Image_Edge',
        'Distance_to_Dark_Region', 'Distance_to_Sparse_Zone',
        'Neighborhood_Mean_Area', 'Neighborhood_Std_Area',
        'Neighborhood_Eccentricity_Mean', 'Orientation_Alignment_Std',
        'Distance_to_Nearest_Nucleus', 'Cluster_Density_Index',
        'Cluster_Elongation', 'Cluster_Polarization_Score',
        'Cluster_Area_Ratio'
    ]
    
    # Check for LBP features
    for i in range(11):
        expected_columns.append(f'LBP_Bin_{i}')
    
    for col in expected_columns:
        assert col in df.columns


@patch('code.nuclei_segmentation.extract_engineered_features.process_image')
def test_extract_command(mock_process_image, temp_image_and_mask):
    """
    Test the CLI command for feature extraction.
    
    Args:
        mock_process_image: Mocked process_image function.
        temp_image_and_mask: Tuple with paths to test image and mask.
    """
    from code.engineered_feature_extraction.extract_engineered_features import extract
    
    image_path, mask_path = temp_image_and_mask
    output_path = image_path.parent / "features.csv"
    
    # Call the command
    extract(
        image=image_path,
        mask=mask_path,
        output=output_path,
        neighbor_radius=25.0,
        jobs=1
    )
    
    # Check that process_image was called with the correct arguments
    mock_process_image.assert_called_once_with(
        image_path,
        mask_path,
        output_path,
        25.0,
        1
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
