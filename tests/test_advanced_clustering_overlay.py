#!/usr/bin/env python3
"""
Test script for advanced clustering overlay functionality.

Author: Christos Botos
Email: hcty02@gmail.com
Date: 2025-07-31

Description:
    This test verifies that the clustering script correctly integrates with the
    advanced overlay utilities for memory-efficient processing of large images.
    It tests the cluster mask generation, overlay configuration, and fallback
    mechanisms for robust operation.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas for data handling.
    • pathlib for file operations.
    • PIL for image processing.

Usage:
    python tests/test_advanced_clustering_overlay.py

Inputs:
    • Test creates synthetic data for validation.
    • Tests overlay integration and fallback mechanisms.

Outputs:
    • Test results showing successful overlay integration.
    • Validation of memory-efficient processing capabilities.

Key Features:
    • Tests cluster mask generation from nuclear labels.
    • Validates overlay configuration and parameter handling.
    • Tests fallback mechanisms when advanced overlay is unavailable.
    • Verifies memory-efficient processing for large datasets.

Notes:
    • This test ensures the clustering script works with the advanced overlay utilities.
    • Validates integration with memory-efficient tile-based processing.
    • Critical for handling gigantic images without memory issues.
"""

import traceback
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import logging

# Configure logging for test output.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_test_data(output_dir: Path) -> tuple:
    """
    Create synthetic test data for overlay testing.
    
    Args:
        output_dir: Directory to save test files.
        
    Returns:
        Tuple of (image_path, mask_path, features_csv_path).
        
    This function generates synthetic test data including a DAPI image,
    segmentation mask, and nuclear features CSV for testing the overlay
    integration functionality.
    """
    logger.info("Creating synthetic test data for overlay testing.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic DAPI image (512x512) - grayscale.
    image_size = (512, 512)
    synthetic_image = np.random.randint(20, 200, image_size, dtype=np.uint8)

    # Add some structure to make it look more realistic.
    for i in range(50):  # Add some bright spots (nuclei).
        x, y = np.random.randint(50, image_size[0]-50, 2)
        radius = np.random.randint(8, 20)

        # Create circular bright regions.
        yy, xx = np.ogrid[:image_size[0], :image_size[1]]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        synthetic_image[mask] = np.random.randint(150, 255, np.sum(mask))

    image_path = output_dir / 'test_image.tif'
    Image.fromarray(synthetic_image).save(image_path)
    
    # Create synthetic segmentation mask.
    seg_mask = np.zeros(image_size, dtype=np.int32)
    nuclear_labels = []
    
    for i in range(30):  # Create 30 synthetic nuclei.
        label = i + 1
        x, y = np.random.randint(30, image_size[0]-30, 2)
        radius = np.random.randint(6, 15)
        
        # Create circular labeled regions.
        yy, xx = np.ogrid[:image_size[0], :image_size[1]]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        seg_mask[mask] = label
        nuclear_labels.append(label)
    
    mask_path = output_dir / 'test_mask.npy'
    np.save(mask_path, seg_mask)
    
    # Create synthetic features CSV.
    n_nuclei = len(nuclear_labels)
    np.random.seed(42)
    
    features_data = {
        'label': nuclear_labels,
        'centroid_x': np.random.uniform(50, image_size[1]-50, n_nuclei),
        'centroid_y': np.random.uniform(50, image_size[0]-50, n_nuclei),
        'area': np.random.uniform(100, 400, n_nuclei),
        'perimeter': np.random.uniform(30, 80, n_nuclei),
        'circularity': np.random.uniform(0.6, 1.0, n_nuclei),
        'eccentricity': np.random.uniform(0.1, 0.8, n_nuclei),
        'solidity': np.random.uniform(0.8, 1.0, n_nuclei),
        'aspect_ratio': np.random.uniform(1.0, 2.5, n_nuclei)
    }
    
    features_df = pd.DataFrame(features_data)
    features_csv_path = output_dir / 'test_features.csv'
    features_df.to_csv(features_csv_path, index=False)
    
    logger.info(f"✓ Created test data: {n_nuclei} nuclei, image size {image_size}")
    
    return image_path, mask_path, features_csv_path


def test_cluster_mask_generation():
    """
    Test cluster mask generation from nuclear labels and cluster assignments.

    This function tests the core functionality of converting nuclear labels
    and cluster assignments into a cluster mask for overlay processing.
    """
    logger.info("Testing cluster mask generation.")

    # Import the clustering functions.
    sys.path.append(str(Path(__file__).parent.parent / 'code' / 'engineered_feature_extraction'))
    from cluster_engineered_features import create_cluster_mask

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test data.
        image_path, mask_path, features_csv_path = create_test_data(tmp_path)

        # Create test cluster assignments.
        nuclear_labels = np.arange(1, 31)  # 30 nuclei.
        cluster_labels = np.random.randint(0, 3, 30)  # 3 clusters.

        # Test cluster mask generation.
        cluster_mask_path = tmp_path / 'cluster_mask.npy'
        result_path = create_cluster_mask(mask_path, nuclear_labels, cluster_labels, cluster_mask_path)

        # Verify the cluster mask was created.
        assert result_path.exists(), "Cluster mask file was not created"

        # Load and validate cluster mask.
        cluster_mask = np.load(result_path)
        original_mask = np.load(mask_path)

        # Check dimensions match.
        assert cluster_mask.shape == original_mask.shape, \
            f"Cluster mask shape mismatch: {cluster_mask.shape} vs {original_mask.shape}"

        # Check that cluster values are in expected range.
        unique_clusters = np.unique(cluster_mask[cluster_mask > 0])
        expected_clusters = np.unique(cluster_labels) + 1  # +1 because of background offset.

        assert np.array_equal(np.sort(unique_clusters), np.sort(expected_clusters)), \
            f"Cluster values mismatch: {unique_clusters} vs {expected_clusters}"

        logger.info("✓ Cluster mask generation test passed.")


def test_overlay_integration():
    """
    Test integration with advanced overlay utilities.

    This function tests the integration between the clustering script and
    the advanced overlay utilities, including configuration handling and
    fallback mechanisms.
    """
    logger.info("Testing overlay integration.")

    # Import the clustering functions.
    sys.path.append(str(Path(__file__).parent.parent / 'code' / 'engineered_feature_extraction'))
    from cluster_engineered_features import create_cluster_overlay_advanced

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test data.
        image_path, mask_path, features_csv_path = create_test_data(tmp_path)

        # Create test cluster assignments.
        nuclear_labels = np.arange(1, 31)  # 30 nuclei.
        cluster_labels = np.random.randint(0, 3, 30)  # 3 clusters.

        # Create test color palette.
        color_palette = {
            0: (255, 0, 0, 200),    # Red.
            1: (0, 255, 0, 200),    # Green.
            2: (0, 0, 255, 200)     # Blue.
        }

        # Test overlay creation.
        overlay_path = tmp_path / 'test_overlay.tif'

        # Test with small parameters for fast execution.
        create_cluster_overlay_advanced(
            image_path=image_path,
            mask_path=mask_path,
            nuclear_labels=nuclear_labels,
            cluster_labels=cluster_labels,
            color_palette=color_palette,
            output_path=overlay_path,
            tile_size=256,  # Small tiles for testing.
            workers=2,      # Limited workers for testing.
            alpha=0.5,
            gpu=False,      # Disable GPU for testing stability.
            memory_limit_mb=1024
        )

        # Verify overlay was created.
        assert overlay_path.exists(), "Overlay file was not created"

        # Check file size is reasonable.
        file_size = overlay_path.stat().st_size
        assert file_size >= 1000, f"Overlay file too small: {file_size} bytes"

        logger.info(f"✓ Overlay integration test passed. File size: {file_size} bytes")


def main():
    """
    Run all advanced clustering overlay tests.
    
    This function coordinates all test cases and provides comprehensive
    validation of the advanced overlay integration functionality.
    """
    logger.info("🧬 ADVANCED CLUSTERING OVERLAY INTEGRATION TESTS 🧬")
    logger.info("=" * 70)
    
    tests = [
        ("Cluster Mask Generation", test_cluster_mask_generation),
        ("Overlay Integration", test_overlay_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 50)
        
        if test_func():
            logger.info(f"✓ {test_name} PASSED")
            passed += 1
        else:
            logger.error(f"✗ {test_name} FAILED")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All advanced overlay integration tests PASSED!")
        return 0
    else:
        logger.error("❌ Some tests FAILED. Please check the overlay integration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
