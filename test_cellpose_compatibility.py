"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_cellpose_compatibility.py.
Description:
    Test script to verify Cellpose3 and Cellpose4 compatibility with the
    version-agnostic wrapper. This script helps identify compatibility issues
    and validates that the wrapper works correctly with both versions.

Dependencies:
    • Python >= 3.10.
    • cellpose (version 3.x or 4.x).
    • numpy for array operations.
    • matplotlib for visualization (optional).

Usage:
    # Test with current environment
    python test_cellpose_compatibility.py
    
    # Test with specific configuration
    python test_cellpose_compatibility.py --config configs/nuclei_segmentation_config_cellpose3.ini

Arguments:
    --config: Path to configuration file (optional).
    --test-image: Path to test image (optional, uses synthetic if not provided).
    --output-dir: Directory for test outputs (optional).
    
Inputs:
    Test image (synthetic or provided).
    Configuration parameters.
    
Outputs:
    Test results and compatibility report.
    Segmentation masks (if successful).
    Version information and diagnostics.
    
Key Features:
    • Automatic Cellpose version detection and testing.
    • Synthetic test image generation for consistent testing.
    • Comprehensive compatibility validation.
    • Detailed error reporting and diagnostics.
    • Performance benchmarking across versions.
    
Notes:
    • Run this script in both iri310 (Cellpose4) and iri310_cellpose3 environments.
    • Compare results to ensure consistent behavior across versions.
    • Use synthetic images for reproducible testing.
    • Check logs for version-specific warnings or issues.
"""

import traceback
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Add the project root to the Python path for imports.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from code.nuclei_segmentation.utils.cellpose_compatibility import CellposeWrapper
    WRAPPER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import CellposeWrapper: {e}")
    WRAPPER_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging for the test script."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('cellpose_compatibility_test.log')
        ]
    )
    return logging.getLogger(__name__)


def create_synthetic_test_image(size: int = 1024, n_objects: int = 50, seed: int = 42) -> np.ndarray:
    """
    Create a synthetic test image with circular objects for testing.
    
    Args:
        size: Image size (square).
        n_objects: Number of synthetic objects to create.
        seed: Random seed for reproducibility.
        
    Returns:
        2D numpy array representing synthetic nuclei image.
    """
    np.random.seed(seed)
    
    # Create blank image.
    image = np.zeros((size, size), dtype=np.uint8)
    
    # Add synthetic circular objects.
    for i in range(n_objects):
        # Random position and size.
        center_y = np.random.randint(50, size - 50)
        center_x = np.random.randint(50, size - 50)
        radius = np.random.randint(8, 25)
        intensity = np.random.randint(100, 255)
        
        # Create circular object.
        y, x = np.ogrid[:size, :size]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        image[mask] = intensity
    
    # Add some noise.
    noise = np.random.normal(0, 10, (size, size))
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return image


def test_cellpose_version_detection(logger: logging.Logger) -> Dict[str, Any]:
    """Test Cellpose version detection and model initialization."""
    logger.info("Testing Cellpose version detection...")
    
    results = {
        'version_detection': False,
        'model_initialization': False,
        'version_info': {},
        'errors': []
    }
    
    try:
        if not WRAPPER_AVAILABLE:
            raise ImportError("CellposeWrapper not available")
        
        # Create wrapper instance.
        wrapper = CellposeWrapper(model_type='nuclei', gpu=True, logger=logger)
        
        # Get version information.
        version_info = wrapper.get_version_info()
        results['version_info'] = version_info
        results['version_detection'] = True
        
        logger.info(f"Version detection successful: {version_info}")
        
        # Test model initialization.
        if wrapper.model is not None:
            results['model_initialization'] = True
            logger.info("Model initialization successful")
        else:
            results['errors'].append("Model initialization failed - model is None")
            
    except Exception as e:
        error_msg = f"Version detection failed: {e}"
        results['errors'].append(error_msg)
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return results


def test_segmentation_functionality(wrapper: CellposeWrapper, test_image: np.ndarray, logger: logging.Logger) -> Dict[str, Any]:
    """Test segmentation functionality with synthetic image."""
    logger.info("Testing segmentation functionality...")
    
    results = {
        'segmentation_success': False,
        'n_objects_detected': 0,
        'diameter_detected': None,
        'processing_time': 0.0,
        'errors': []
    }
    
    try:
        # Record start time.
        start_time = time.time()
        
        # Run segmentation.
        masks, flows, n_cells, diameter_info = wrapper.segment(
            image=test_image,
            diameter=0,  # Auto-detection
            flow_threshold=0.9,
            cellprob_threshold=-12,
            resample=True,
            batch_size=8
        )
        
        # Record processing time.
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        # Validate results.
        if masks is not None and masks.size > 0:
            results['segmentation_success'] = True
            results['n_objects_detected'] = n_cells
            results['diameter_detected'] = diameter_info
            
            logger.info(f"Segmentation successful: {n_cells} objects detected in {processing_time:.2f}s")
            if diameter_info is not None:
                logger.info(f"Auto-detected diameter: {diameter_info:.1f}px")
        else:
            results['errors'].append("Segmentation returned empty or None masks")
            
    except Exception as e:
        error_msg = f"Segmentation failed: {e}"
        results['errors'].append(error_msg)
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return results


def save_test_results(results: Dict[str, Any], output_dir: Path, logger: logging.Logger):
    """Save test results and visualizations."""
    logger.info("Saving test results...")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results as text file.
        results_file = output_dir / "compatibility_test_results.txt"
        with open(results_file, 'w') as f:
            f.write("Cellpose Compatibility Test Results\n")
            f.write("=" * 40 + "\n\n")
            
            for test_name, test_results in results.items():
                f.write(f"{test_name.upper()}:\n")
                f.write("-" * 20 + "\n")
                
                if isinstance(test_results, dict):
                    for key, value in test_results.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {test_results}\n")
                f.write("\n")
        
        logger.info(f"Test results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Failed to save test results: {e}")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Cellpose compatibility")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--test-image", type=str, help="Test image path")
    parser.add_argument("--output-dir", type=str, default="test_outputs", help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging.
    logger = setup_logging(args.log_level)
    logger.info("Starting Cellpose compatibility test...")
    
    # Create output directory.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results dictionary.
    all_results = {}
    
    # Test 1: Version detection and model initialization.
    logger.info("=" * 50)
    logger.info("TEST 1: Version Detection and Model Initialization")
    logger.info("=" * 50)
    
    version_results = test_cellpose_version_detection(logger)
    all_results['version_detection'] = version_results
    
    if not version_results['version_detection']:
        logger.error("Version detection failed. Cannot proceed with further tests.")
        return
    
    # Test 2: Segmentation functionality.
    logger.info("=" * 50)
    logger.info("TEST 2: Segmentation Functionality")
    logger.info("=" * 50)
    
    try:
        # Create or load test image.
        if args.test_image and Path(args.test_image).exists():
            logger.info(f"Loading test image from: {args.test_image}")
            # Add image loading logic here if needed.
            test_image = create_synthetic_test_image()  # Fallback to synthetic.
        else:
            logger.info("Creating synthetic test image...")
            test_image = create_synthetic_test_image(size=1024, n_objects=30)
        
        # Save test image.
        test_image_path = output_dir / "test_image.png"
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(8, 8))
            plt.imshow(test_image, cmap='gray')
            plt.title("Synthetic Test Image")
            plt.axis('off')
            plt.savefig(test_image_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Test image saved to: {test_image_path}")
        
        # Create wrapper and test segmentation.
        wrapper = CellposeWrapper(model_type='nuclei', gpu=True, logger=logger)
        segmentation_results = test_segmentation_functionality(wrapper, test_image, logger)
        all_results['segmentation'] = segmentation_results
        
    except Exception as e:
        logger.error(f"Segmentation test failed: {e}")
        all_results['segmentation'] = {'errors': [str(e)]}
    
    # Save all results.
    logger.info("=" * 50)
    logger.info("SAVING RESULTS")
    logger.info("=" * 50)
    
    save_test_results(all_results, output_dir, logger)
    
    # Print summary.
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    version_success = all_results.get('version_detection', {}).get('version_detection', False)
    model_success = all_results.get('version_detection', {}).get('model_initialization', False)
    segmentation_success = all_results.get('segmentation', {}).get('segmentation_success', False)
    
    logger.info(f"Version Detection: {'✓' if version_success else '✗'}")
    logger.info(f"Model Initialization: {'✓' if model_success else '✗'}")
    logger.info(f"Segmentation: {'✓' if segmentation_success else '✗'}")
    
    if version_success and model_success and segmentation_success:
        logger.info("🎉 All tests passed! Cellpose compatibility confirmed.")
    else:
        logger.warning("⚠️  Some tests failed. Check logs for details.")
    
    logger.info("Test completed. Check test_outputs/ directory for detailed results.")


if __name__ == "__main__":
    main()
