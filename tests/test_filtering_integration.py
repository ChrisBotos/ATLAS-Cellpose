"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_filtering_integration.py.
Description:
    Test suite for morphological filtering integration in the nuclei segmentation pipeline.
    Validates that filtering parameters are correctly loaded from configuration and that
    the filtering step properly removes artifacts while preserving valid nuclei.

Dependencies:
    • Python >= 3.7.
    • numpy, pytest, skimage.
    • Custom nuclei segmentation modules.

Usage:
    pytest tests/test_filtering_integration.py -v

Inputs:
    • Synthetic segmentation masks for testing.
    • Configuration parameters for filtering thresholds.

Outputs:
    • Test results validating filtering functionality.
    • Temporary test files for verification.

Key Features:
    • Tests configuration loading for filtering parameters.
    • Validates filtering threshold application.
    • Checks integration with pipeline workflow.
    • Verifies output file generation and statistics.

Notes:
    • Uses synthetic data to ensure reproducible test results.
    • Tests both programmatic and configuration-based filtering.
"""

import traceback
import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock

# Import the modules to test.
import sys
sys.path.append(str(Path(__file__).parent.parent / "code" / "nuclei_segmentation"))

from utils.filter_masks import filter_masks_programmatic, Thresholds
from utils.project_setup import load_config


class TestFilteringIntegration:
    """Test suite for morphological filtering integration."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create synthetic segmentation mask with various nucleus shapes.
        self.test_mask = np.zeros((200, 200), dtype=np.uint16)
        
        # Good nucleus: circular, medium size.
        y, x = np.ogrid[:200, :200]
        circle1 = (x - 50)**2 + (y - 50)**2 <= 15**2
        self.test_mask[circle1] = 1
        
        # Bad nucleus: too small.
        circle2 = (x - 100)**2 + (y - 50)**2 <= 3**2
        self.test_mask[circle2] = 2
        
        # Bad nucleus: too large.
        circle3 = (x - 150)**2 + (y - 50)**2 <= 35**2
        self.test_mask[circle3] = 3
        
        # Good nucleus: slightly elongated but within limits.
        ellipse = ((x - 50)**2 / 12**2) + ((y - 150)**2 / 8**2) <= 1
        self.test_mask[ellipse] = 4
        
        # Bad nucleus: too elongated (high aspect ratio).
        line = ((x - 150)**2 / 30**2) + ((y - 150)**2 / 3**2) <= 1
        self.test_mask[line] = 5

    def test_thresholds_validation(self):
        """Test that filtering thresholds are properly validated."""
        # Valid thresholds should not raise an error.
        valid_thresholds = Thresholds(
            min_pixels=20,
            max_pixels=900,
            min_circularity=0.56,
            max_circularity=1.00,
            min_solidity=0.765,
            max_solidity=1.00,
            min_eccentricity=0.00,
            max_eccentricity=0.975,
            min_aspect_ratio=0.50,
            max_aspect_ratio=3.20,
            min_hole_fraction=0.00,
            max_hole_fraction=0.001
        )
        valid_thresholds.validate()  # Should not raise.
        
        # Invalid thresholds should raise an assertion error.
        with pytest.raises(AssertionError):
            invalid_thresholds = Thresholds(min_pixels=100, max_pixels=50)
            invalid_thresholds.validate()

    def test_filtering_programmatic_interface(self):
        """Test the programmatic filtering interface."""
        # Create test settings with filtering parameters.
        settings = {
            "min_pixels": 20,
            "max_pixels": 900,
            "min_circularity": 0.56,
            "max_circularity": 1.00,
            "min_solidity": 0.765,
            "max_solidity": 1.00,
            "min_eccentricity": 0.00,
            "max_eccentricity": 0.975,
            "min_aspect_ratio": 0.50,
            "max_aspect_ratio": 3.20,
            "min_hole_fraction": 0.00,
            "max_hole_fraction": 0.001,
            "exclude_border": False,
            "debug_mode": True
        }
        
        # Create temporary output directory.
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create mock logger.
            logger = Mock()
            
            # Apply filtering.
            filtered_masks = filter_masks_programmatic(
                masks=self.test_mask,
                settings=settings,
                output_dir=output_dir,
                logger=logger
            )
            
            # Check that filtering was applied.
            assert filtered_masks is not None
            assert filtered_masks.shape == self.test_mask.shape
            
            # Check that some nuclei were filtered out.
            original_count = len(np.unique(self.test_mask)) - 1  # Exclude background.
            filtered_count = len(np.unique(filtered_masks)) - 1  # Exclude background.
            assert filtered_count < original_count
            
            # Check that output files were created.
            filter_dir = output_dir / "filtering"
            assert filter_dir.exists()
            assert (filter_dir / "filtered_masks.npy").exists()
            assert (filter_dir / "filtering_stats.json").exists()
            
            # Check statistics file.
            with open(filter_dir / "filtering_stats.json", "r") as f:
                stats = json.load(f)
            assert "original_nuclei" in stats
            assert "passed_nuclei" in stats
            assert "failed_nuclei" in stats
            assert "retention_rate" in stats
            assert stats["original_nuclei"] == original_count
            assert stats["passed_nuclei"] == filtered_count
            
            # Check debug files were created.
            assert (filter_dir / "filtering_metrics.csv").exists()
            assert (filter_dir / "passed_labels.npy").exists()
            assert (filter_dir / "failed_labels.npy").exists()

    def test_configuration_loading(self):
        """Test that filtering parameters are correctly loaded from configuration."""
        # Create a temporary config file with filtering parameters.
        config_content = """
[general]
job_name = test_filtering
image_path = test.tif
output_dir = test_output

[filtering]
use_filtering = True
min_pixels = 25
max_pixels = 800
min_circularity = 0.60
max_circularity = 0.95
min_solidity = 0.80
max_solidity = 0.99
min_eccentricity = 0.10
max_eccentricity = 0.90
min_aspect_ratio = 0.60
max_aspect_ratio = 3.00
min_hole_fraction = 0.00
max_hole_fraction = 0.002
exclude_border = True

[cellpose]
model_type = nuclei
gpu = False
diameter = None

[tiling]
use_tiling = False

[debug]
debug_mode = False

[using_previous_results]
use_previous_results = False
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            # Load configuration.
            settings, cellpose_params, dirs = load_config(config_path)
            
            # Check that filtering parameters were loaded correctly.
            assert settings["use_filtering"] is True
            assert settings["min_pixels"] == 25
            assert settings["max_pixels"] == 800
            assert settings["min_circularity"] == 0.60
            assert settings["max_circularity"] == 0.95
            assert settings["min_solidity"] == 0.80
            assert settings["max_solidity"] == 0.99
            assert settings["min_eccentricity"] == 0.10
            assert settings["max_eccentricity"] == 0.90
            assert settings["min_aspect_ratio"] == 0.60
            assert settings["max_aspect_ratio"] == 3.00
            assert settings["min_hole_fraction"] == 0.00
            assert settings["max_hole_fraction"] == 0.002
            assert settings["exclude_border"] is True
            
        finally:
            # Clean up temporary config file.
            Path(config_path).unlink()

    def test_filtering_with_intensity_image(self):
        """Test filtering with optional intensity image."""
        # Create synthetic intensity image.
        intensity_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        
        settings = {
            "min_pixels": 20,
            "max_pixels": 900,
            "min_circularity": 0.0,  # Relaxed for synthetic data.
            "max_circularity": 1.00,
            "min_solidity": 0.0,     # Relaxed for synthetic data.
            "max_solidity": 1.00,
            "min_eccentricity": 0.00,
            "max_eccentricity": 1.00,
            "min_aspect_ratio": 0.50,
            "max_aspect_ratio": 10.0,  # Relaxed for synthetic data.
            "min_hole_fraction": 0.00,
            "max_hole_fraction": 1.00,
            "exclude_border": False,
            "debug_mode": False
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            logger = Mock()
            
            # Apply filtering with intensity image.
            filtered_masks = filter_masks_programmatic(
                masks=self.test_mask,
                settings=settings,
                output_dir=output_dir,
                logger=logger,
                intensity_image=intensity_image
            )
            
            # Check that filtering completed successfully.
            assert filtered_masks is not None
            assert filtered_masks.shape == self.test_mask.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
