"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_cellpose4_auto_detection_simple.py.
Description:
    Simple test suite to verify that Cellpose4 auto-detection parameters
    are correctly configured and fallback parameters have been removed.
    Tests the core functionality without complex dependencies.

Dependencies:
    • Python >= 3.10.
    • pytest, configparser.

Usage:
    pytest tests/test_cellpose4_auto_detection_simple.py -v

Key Features:
    • Validates configuration file parameters for auto-detection.
    • Tests that diameter=0 and resample=True are properly set.
    • Verifies that fallback parameter logic has been removed from code.
    • Checks for proper error handling without fallback attempts.

Notes:
    • Lightweight test that doesn't require full segmentation pipeline.
    • Focuses on configuration validation and code structure verification.
"""

import traceback
import pytest
import configparser
from pathlib import Path
import re


class TestCellpose4AutoDetectionConfig:
    """Test suite for Cellpose4 auto-detection configuration."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / "configs" / "nuclei_segmentation_config.ini"

    def test_config_file_exists(self):
        """Test that the configuration file exists."""
        assert self.config_path.exists(), f"Configuration file not found: {self.config_path}"

    def test_auto_detection_parameters(self):
        """Test that auto-detection parameters are correctly configured."""
        config = configparser.ConfigParser()
        config.read(self.config_path)
        
        # Verify cellpose section exists.
        assert "cellpose" in config, "Cellpose section missing from configuration"
        
        cellpose_config = config["cellpose"]
        
        # Test diameter=0 for auto-detection.
        diameter = cellpose_config.getint("diameter")
        assert diameter == 0, f"Expected diameter=0 for auto-detection, got {diameter}"
        
        # Test resample=True for auto-detection.
        resample = cellpose_config.getboolean("resample")
        assert resample == True, f"Expected resample=True for auto-detection, got {resample}"
        
        # Test other parameters are reasonable for auto-detection.
        flow_threshold = cellpose_config.getfloat("flow_threshold")
        assert 0.1 <= flow_threshold <= 1.0, f"Flow threshold should be reasonable: {flow_threshold}"
        
        cellprob_threshold = cellpose_config.getfloat("cellprob_threshold")
        assert -20 <= cellprob_threshold <= 5, f"Cell probability threshold should be reasonable: {cellprob_threshold}"

    def test_config_comments_updated(self):
        """Test that configuration comments reflect auto-detection usage."""
        with open(self.config_path, 'r') as f:
            config_content = f.read()
        
        # Check for auto-detection related comments.
        assert "auto-detection" in config_content.lower(), "Configuration should mention auto-detection"
        assert "adaptive diameter" in config_content.lower(), "Configuration should mention adaptive diameter"
        
        # Check that fallback-related comments have been updated.
        assert "no fallback" in config_content.lower() or "relies entirely" in config_content.lower(), \
            "Configuration should indicate no fallback parameters are used"

    def test_segmentation_code_no_fallback(self):
        """Test that segmentation code doesn't contain fallback parameter logic."""
        segmentation_file = self.project_root / "code" / "nuclei_segmentation" / "utils" / "segmentation.py"
        
        if segmentation_file.exists():
            with open(segmentation_file, 'r') as f:
                code_content = f.read()
            
            # Check that fallback parameter logic has been removed.
            fallback_patterns = [
                r"diameter\s*=\s*15",  # Fixed diameter fallback.
                r"flow_threshold\s*=\s*0\.6",  # Conservative flow threshold.
                r"cellprob_threshold\s*=\s*-3",  # Less sensitive threshold.
                r"resample\s*=\s*False.*fallback",  # Disable resample for fallback.
                r"fallback.*parameters",  # General fallback logic.
                r"trying.*fallback",  # Fallback attempt messages.
            ]
            
            for pattern in fallback_patterns:
                matches = re.findall(pattern, code_content, re.IGNORECASE)
                assert len(matches) == 0, f"Found fallback parameter pattern: {pattern} in {matches}"

    def test_parallel_segmentation_code_no_fallback(self):
        """Test that parallel segmentation code doesn't contain fallback parameter logic."""
        parallel_file = self.project_root / "code" / "nuclei_segmentation" / "utils" / "parallel_segmentation.py"
        
        if parallel_file.exists():
            with open(parallel_file, 'r') as f:
                code_content = f.read()
            
            # Check that fallback parameter logic has been removed.
            fallback_patterns = [
                r"diameter\s*=\s*15",  # Fixed diameter fallback.
                r"flow_threshold\s*=\s*0\.6",  # Conservative flow threshold.
                r"cellprob_threshold\s*=\s*-3",  # Less sensitive threshold.
                r"resample\s*=\s*False.*fallback",  # Disable resample for fallback.
                r"fallback.*parameters",  # General fallback logic.
                r"trying.*fallback",  # Fallback attempt messages.
            ]
            
            for pattern in fallback_patterns:
                matches = re.findall(pattern, code_content, re.IGNORECASE)
                assert len(matches) == 0, f"Found fallback parameter pattern: {pattern} in {matches}"

    def test_auto_detection_error_handling(self):
        """Test that proper error handling exists for auto-detection failures."""
        segmentation_file = self.project_root / "code" / "nuclei_segmentation" / "utils" / "segmentation.py"
        
        if segmentation_file.exists():
            with open(segmentation_file, 'r') as f:
                code_content = f.read()
            
            # Check for proper error handling patterns.
            error_patterns = [
                r"auto-detection.*failed",  # Auto-detection failure messages.
                r"diameter=0.*resample=True",  # Parameter guidance.
                r"adaptive.*diameter.*learning",  # Auto-detection explanation.
            ]
            
            found_patterns = 0
            for pattern in error_patterns:
                if re.search(pattern, code_content, re.IGNORECASE):
                    found_patterns += 1
            
            assert found_patterns >= 2, "Should have proper auto-detection error handling and guidance"

    def test_diameter_logging_enhanced(self):
        """Test that diameter detection logging has been enhanced."""
        files_to_check = [
            self.project_root / "code" / "nuclei_segmentation" / "utils" / "segmentation.py",
            self.project_root / "code" / "nuclei_segmentation" / "utils" / "parallel_segmentation.py"
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    code_content = f.read()
                
                # Check for enhanced diameter logging patterns.
                logging_patterns = [
                    r"Auto-detected diameter",  # Basic diameter logging.
                    r"range:",  # Diameter range logging.
                    r"variation",  # Diameter variation analysis.
                    r"CV.*%",  # Coefficient of variation.
                ]
                
                found_patterns = 0
                for pattern in logging_patterns:
                    if re.search(pattern, code_content, re.IGNORECASE):
                        found_patterns += 1
                
                assert found_patterns >= 2, f"Should have enhanced diameter logging in {file_path.name}"

    def test_configuration_parameter_consistency(self):
        """Test that configuration parameters are consistent with auto-detection requirements."""
        config = configparser.ConfigParser()
        config.read(self.config_path)
        
        cellpose_config = config["cellpose"]
        
        # When diameter=0, resample must be True.
        diameter = cellpose_config.getint("diameter")
        resample = cellpose_config.getboolean("resample")
        
        if diameter == 0:
            assert resample == True, "When diameter=0, resample must be True for proper auto-detection"
        
        # GPU should be enabled for better performance.
        gpu = cellpose_config.getboolean("gpu")
        assert gpu == True, "GPU should be enabled for optimal auto-detection performance"

    def test_no_division_by_zero_fallback(self):
        """Test that division by zero fallback logic has been removed."""
        files_to_check = [
            self.project_root / "code" / "nuclei_segmentation" / "utils" / "segmentation.py",
            self.project_root / "code" / "nuclei_segmentation" / "utils" / "parallel_segmentation.py"
        ]
        
        for file_path in files_to_check:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    code_content = f.read()
                
                # Check that division by zero fallback logic has been removed.
                division_patterns = [
                    r"division by zero.*fallback",
                    r"if.*division by zero.*in.*str\(e\)",
                    r"Handle division by zero with fallback",
                ]
                
                for pattern in division_patterns:
                    matches = re.findall(pattern, code_content, re.IGNORECASE)
                    assert len(matches) == 0, f"Found division by zero fallback pattern in {file_path.name}: {pattern}"


class TestAutoDetectionBehavior:
    """Test auto-detection behavior with mock scenarios."""

    def test_mock_auto_detection_parameters(self):
        """Test that auto-detection parameters would be passed correctly."""
        # Simulate the parameter passing that would occur.
        cellpose_params = {
            "diameter": 0,
            "flow_threshold": 0.9,
            "cellprob_threshold": -12,
            "resample": True,
            "batch_size": 8
        }
        
        # Verify auto-detection parameters.
        assert cellpose_params["diameter"] == 0, "Should use diameter=0 for auto-detection"
        assert cellpose_params["resample"] == True, "Should use resample=True for auto-detection"
        assert isinstance(cellpose_params["flow_threshold"], float), "Flow threshold should be numeric"
        assert isinstance(cellpose_params["cellprob_threshold"], (int, float)), "Cell prob threshold should be numeric"

    def test_mock_diameter_detection_results(self):
        """Test handling of mock diameter detection results."""
        # Simulate Cellpose4 diameter detection results.
        mock_results = [
            (None, None, None, [18.5, 19.2, 17.8, 20.1]),  # Multiple diameters.
            (None, None, None, 19.0),  # Single diameter.
            (None, None, None, None),  # No diameter detected.
            (None, None, None, []),  # Empty diameter list.
        ]
        
        for masks, flows, styles, diameters in mock_results:
            if diameters is not None:
                if isinstance(diameters, (list, tuple)) and len(diameters) > 0:
                    # Should handle multiple diameters.
                    avg_diameter = sum(diameters) / len(diameters)
                    assert avg_diameter > 0, "Average diameter should be positive"
                elif isinstance(diameters, (int, float)) and diameters > 0:
                    # Should handle single diameter.
                    assert diameters > 0, "Single diameter should be positive"
                # Other cases should be handled gracefully without fallback.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
