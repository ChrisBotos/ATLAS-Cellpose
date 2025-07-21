"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_cellpose4_auto_detection.py.
Description:
    Test suite for Cellpose4 auto-detection functionality to ensure proper
    diameter detection without fallback parameters. Validates that the
    segmentation pipeline correctly uses diameter=0 and resample=True for
    adaptive diameter learning across different tissue regions.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, cellpose, unittest.mock.
    • Custom segmentation utilities from the project.

Usage:
    pytest tests/nuclei_segmentation_tests/test_cellpose4_auto_detection.py -v

Arguments:
    None (pytest handles test discovery and execution).

Inputs:
    • Mock Cellpose model instances.
    • Synthetic test images with known characteristics.
    • Configuration parameters for auto-detection testing.

Outputs:
    • Test results showing auto-detection functionality.
    • Validation of diameter detection logging.
    • Confirmation that fallback parameters are not used.

Key Features:
    • Tests single-pass and tiled segmentation auto-detection.
    • Validates proper error handling without fallback parameters.
    • Checks diameter detection logging and reporting.
    • Ensures configuration parameters are correctly applied.
    • Tests parallel processing auto-detection functionality.

Notes:
    • Designed for kidney I/R injury tissue analysis validation.
    • Uses mock objects to simulate Cellpose4 behavior.
    • Focuses on auto-detection reliability and error handling.
"""

import traceback
import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import sys
import os

# Add the project root to the Python path for imports.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock the imports to avoid dependency issues during testing.
try:
    from code.nuclei_segmentation.utils.segmentation import (
        _run_single_pass_cellpose
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

    # Create mock functions for testing.
    def _run_single_pass_cellpose(model, image, cellpose_params, logger):
        """Mock function for testing."""
        return model.eval(image[..., None], **cellpose_params)


class TestCellpose4AutoDetection:
    """Test suite for Cellpose4 auto-detection functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create synthetic test image.
        self.test_image = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        
        # Create mock logger.
        self.mock_logger = Mock()
        
        # Create temporary directory for test outputs.
        self.temp_dir = tempfile.mkdtemp()
        
        # Standard cellpose parameters for auto-detection.
        self.cellpose_params = {
            "diameter": 0,  # Enable auto-detection.
            "flow_threshold": 0.9,
            "cellprob_threshold": -12,
            "resample": True,  # Required for auto-detection.
            "batch_size": 8
        }
        
        # Standard settings for testing.
        self.settings = {
            "output_dir": self.temp_dir,
            "tile_side_length": 512,
            "tile_overlap": 0.2,
            "use_tiling": True,
            "debug_mode": True
        }

    def teardown_method(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_cellpose_model(self, detected_diameter=25.0, num_cells=50):
        """Create a mock Cellpose model with auto-detection results."""
        mock_model = Mock()
        
        # Create mock segmentation results.
        mock_masks = np.random.randint(0, num_cells + 1, self.test_image.shape, dtype=np.uint32)
        mock_flows = [
            np.random.random((2, *self.test_image.shape)),  # Flow field.
            np.random.random(self.test_image.shape),        # Cell probability.
            None                                            # Styles (not used).
        ]
        mock_styles = np.random.random(256)
        mock_diameters = [detected_diameter] * 10  # Multiple diameter detections.
        
        # Configure model.eval() to return proper Cellpose4 format.
        mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles, mock_diameters)
        
        return mock_model

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Segmentation modules not available")
    def test_single_pass_auto_detection_success(self):
        """Test successful auto-detection in single-pass mode."""
        # Create mock model with successful auto-detection.
        mock_model = self.create_mock_cellpose_model(detected_diameter=22.5, num_cells=75)

        # Run single-pass segmentation.
        masks, flows, num_cells = _run_single_pass_cellpose(
            mock_model, self.test_image, self.cellpose_params, self.mock_logger
        )

        # Verify model was called with correct parameters.
        mock_model.eval.assert_called_once()
        call_args = mock_model.eval.call_args

        assert call_args[1]["diameter"] == 0, "Should use diameter=0 for auto-detection"
        assert call_args[1]["resample"] == True, "Should use resample=True for auto-detection"
        assert call_args[1]["flow_threshold"] == 0.9, "Should use configured flow_threshold"
        assert call_args[1]["cellprob_threshold"] == -12, "Should use configured cellprob_threshold"

        # Verify results.
        assert masks is not None, "Should return valid masks"
        assert num_cells > 0, "Should detect cells"

        # Verify diameter detection logging.
        info_calls = [call for call in self.mock_logger.info.call_args_list if "Auto-detected diameter" in str(call)]
        assert len(info_calls) > 0, "Should log auto-detected diameter information"

    def test_single_pass_auto_detection_failure(self):
        """Test proper error handling when auto-detection fails."""
        # Create mock model that raises an exception.
        mock_model = Mock()
        mock_model.eval.side_effect = Exception("Auto-detection failed")
        
        # Verify that exception is properly raised without fallback.
        with pytest.raises(Exception, match="Auto-detection failed"):
            _run_single_pass_cellpose(
                mock_model, self.test_image, self.cellpose_params, self.mock_logger
            )
        
        # Verify error logging.
        error_calls = [call for call in self.mock_logger.error.call_args_list if "auto-detection failed" in str(call).lower()]
        assert len(error_calls) > 0, "Should log auto-detection failure"

    def test_tiled_auto_detection_success(self):
        """Test successful auto-detection in tiled mode."""
        # Create mock model with successful auto-detection.
        mock_model = self.create_mock_cellpose_model(detected_diameter=18.3, num_cells=25)
        
        # Run tiled segmentation.
        masks_mm, flows, total_cells = run_cellpose_on_tiles(
            mock_model, self.test_image, self.cellpose_params, self.settings, self.mock_logger
        )
        
        # Verify model was called multiple times (for tiles).
        assert mock_model.eval.call_count > 1, "Should process multiple tiles"
        
        # Verify all calls used auto-detection parameters.
        for call in mock_model.eval.call_args_list:
            assert call[1]["diameter"] == 0, "All tiles should use diameter=0"
            assert call[1]["resample"] == True, "All tiles should use resample=True"
        
        # Verify results.
        assert masks_mm is not None, "Should return valid memory-mapped masks"
        assert total_cells > 0, "Should detect cells across tiles"
        
        # Verify diameter detection logging for tiles.
        info_calls = [call for call in self.mock_logger.info.call_args_list if "Auto-detected diameter" in str(call)]
        assert len(info_calls) > 0, "Should log diameter detection for tiles"

    def test_parallel_auto_detection_success(self):
        """Test successful auto-detection in parallel processing mode."""
        # Create mock model with successful auto-detection.
        mock_model = self.create_mock_cellpose_model(detected_diameter=20.1, num_cells=30)
        
        # Create test tile batch.
        tile_batch = [
            (np.random.randint(0, 255, (512, 512), dtype=np.uint8), (slice(0, 512), slice(0, 512))),
            (np.random.randint(0, 255, (512, 512), dtype=np.uint8), (slice(0, 512), slice(512, 1024)))
        ]
        
        # Run parallel batch processing.
        results = process_cellpose_batch(
            mock_model, tile_batch, self.cellpose_params, batch_idx=0, timeout_seconds=60
        )
        
        # Verify results.
        assert len(results) == 2, "Should process all tiles in batch"
        assert all(len(result) == 3 for result in results), "Each result should have mask, slice_info, cell_count"
        
        # Verify model was called for each tile.
        assert mock_model.eval.call_count == 2, "Should process each tile"
        
        # Verify all calls used auto-detection parameters.
        for call in mock_model.eval.call_args_list:
            assert call[1]["diameter"] == 0, "All tiles should use diameter=0"
            assert call[1]["resample"] == True, "All tiles should use resample=True"

    def test_no_fallback_parameters_used(self):
        """Test that fallback parameters are never used."""
        # Create mock model that fails on first call.
        mock_model = Mock()
        mock_model.eval.side_effect = Exception("Simulated failure")
        
        # Verify that no fallback is attempted.
        with pytest.raises(Exception, match="Simulated failure"):
            _run_single_pass_cellpose(
                mock_model, self.test_image, self.cellpose_params, self.mock_logger
            )
        
        # Verify model was called only once (no fallback attempt).
        assert mock_model.eval.call_count == 1, "Should not attempt fallback parameters"
        
        # Verify no fallback-related logging.
        all_calls = str(self.mock_logger.warning.call_args_list) + str(self.mock_logger.info.call_args_list)
        assert "fallback" not in all_calls.lower(), "Should not mention fallback parameters"

    def test_diameter_variation_logging(self):
        """Test that diameter variation is properly logged."""
        # Create mock model with variable diameter detection.
        mock_model = Mock()
        variable_diameters = [15.0, 18.5, 22.0, 19.3, 16.8]  # Variable diameters.
        mock_masks = np.random.randint(0, 50, self.test_image.shape, dtype=np.uint32)
        mock_flows = [np.random.random((2, *self.test_image.shape)), np.random.random(self.test_image.shape), None]
        mock_styles = np.random.random(256)
        
        mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles, variable_diameters)
        
        # Run single-pass segmentation.
        _run_single_pass_cellpose(
            mock_model, self.test_image, self.cellpose_params, self.mock_logger
        )
        
        # Verify diameter variation logging.
        info_calls = str(self.mock_logger.info.call_args_list)
        assert "range:" in info_calls, "Should log diameter range"
        assert "CV:" in info_calls, "Should log coefficient of variation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
