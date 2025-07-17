"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_config_integration.py.
Description:
    Integration tests for configuration parameter loading and passing through
    the entire nuclei segmentation pipeline from config file to processing functions.

Dependencies:
    • Python >= 3.7.
    • pytest, configparser.
    • Custom project_setup and pipeline modules.

Usage:
    pytest test_config_integration.py -v

Inputs:
    • Configuration files and parameter dictionaries.
    • Mock pipeline components for testing.

Outputs:
    • Validation results for parameter loading and passing.
    • Configuration consistency checks.

Key Features:
    • End-to-end parameter validation.
    • Configuration file parsing verification.
    • Pipeline parameter passing validation.
    • Memory and timeout parameter checks.

Notes:
    • Tests ensure all new parallel processing parameters are properly integrated.
    • Validates configuration consistency across the entire pipeline.
    • Includes fallback value testing for missing parameters.
"""

import pytest
import configparser
import tempfile
from pathlib import Path
import sys
import os

# Add the project root to the path for imports.
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from code.nuclei_segmentation.utils.project_setup import load_config


class TestConfigurationLoading:
    """Test configuration parameter loading and validation."""
    
    @pytest.fixture
    def sample_config_content(self):
        """Create sample configuration content for testing."""
        return """
[general]
image_path = test_image.tif
output_dir = test_output
upscale_factor = 1
crop_image = True
enhance_contrast = True
generate_full_overlay = False
crop_box = 0.5,0.56,0.59,0.68

[cellpose]
model_type = nuclei
gpu = True
diameter = 0
channels = 0,0
flow_threshold = 0.9
cellprob_threshold = -12
resample = True
enable_parallel_processing = True
parallel_batch_size = 4
parallel_max_workers = 2
parallel_memory_limit_gb = 6.0
parallel_timeout_seconds = 300

[tiling]
use_tiling = True
tile_side_length = 512
tile_overlap = 0.2
merge_overlap_threshold = 0.3
gpu_batch_size = 1
gpu_memory_limit_gb = 8.0
gpu_memory_safety_factor = 1.5
gpu_spatial_strategy = adaptive
gpu_adaptive_batching = True
gpu_aggressive_cleanup = True
gpu_max_retries = 3
gpu_timeout_seconds = 300

[using_previous_results]
use_previous_results = False
skip_and_copy_preprocessing = False
skip_and_copy_segmentation = False
skip_and_copy_merging = False
skip_and_copy_postprocessing = False
skip_and_copy_visualization = False
"""
    
    @pytest.fixture
    def temp_config_file(self, sample_config_content):
        """Create a temporary configuration file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(sample_config_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup.
        os.unlink(temp_path)
    
    def test_config_loading_basic(self, temp_config_file):
        """Test basic configuration loading functionality."""
        # Mock the setup_project_structure function to avoid directory creation.
        with pytest.MonkeyPatch().context() as m:
            mock_dirs = {
                "configs": Path("configs"),
                "results": Path("results"),
                "logs": Path("logs"),
                "data": Path("data")
            }
            m.setattr("code.nuclei_segmentation.utils.project_setup.setup_project_structure", 
                     lambda: mock_dirs)
            
            settings, cellpose_params, project_dirs = load_config(temp_config_file)
            
            # Verify basic settings.
            assert settings["image_path"] == "test_image.tif"
            assert settings["output_dir"] == "test_output"
            assert settings["tile_side_length"] == 512
            assert settings["tile_overlap"] == 0.2
    
    def test_cellpose_parallel_parameters(self, temp_config_file):
        """Test that all new Cellpose parallel processing parameters are loaded."""
        with pytest.MonkeyPatch().context() as m:
            mock_dirs = {
                "configs": Path("configs"),
                "results": Path("results"),
                "logs": Path("logs"),
                "data": Path("data")
            }
            m.setattr("code.nuclei_segmentation.utils.project_setup.setup_project_structure", 
                     lambda: mock_dirs)
            
            settings, cellpose_params, project_dirs = load_config(temp_config_file)
            
            # Verify parallel processing parameters.
            assert cellpose_params["enable_parallel_processing"] is True
            assert cellpose_params["parallel_batch_size"] == 4
            assert cellpose_params["parallel_max_workers"] == 2
            assert cellpose_params["parallel_memory_limit_gb"] == 6.0
            assert cellpose_params["parallel_timeout_seconds"] == 300
    
    def test_gpu_timeout_retry_parameters(self, temp_config_file):
        """Test that GPU timeout and retry parameters are loaded."""
        with pytest.MonkeyPatch().context() as m:
            mock_dirs = {
                "configs": Path("configs"),
                "results": Path("results"),
                "logs": Path("logs"),
                "data": Path("data")
            }
            m.setattr("code.nuclei_segmentation.utils.project_setup.setup_project_structure", 
                     lambda: mock_dirs)
            
            settings, cellpose_params, project_dirs = load_config(temp_config_file)
            
            # Verify GPU timeout and retry parameters.
            assert settings["gpu_max_retries"] == 3
            assert settings["gpu_timeout_seconds"] == 300
            assert settings["gpu_memory_safety_factor"] == 1.5
            assert settings["gpu_spatial_strategy"] == "adaptive"
    
    def test_fallback_values(self):
        """Test that fallback values are used when parameters are missing."""
        # Create minimal config without new parameters.
        minimal_config = """
[general]
image_path = test_image.tif
output_dir = test_output

[cellpose]
model_type = nuclei
gpu = True

[tiling]
use_tiling = True
tile_side_length = 512
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(minimal_config)
            temp_path = f.name
        
        try:
            with pytest.MonkeyPatch().context() as m:
                mock_dirs = {
                    "configs": Path("configs"),
                    "results": Path("results"),
                    "logs": Path("logs"),
                    "data": Path("data")
                }
                m.setattr("code.nuclei_segmentation.utils.project_setup.setup_project_structure", 
                         lambda: mock_dirs)
                
                settings, cellpose_params, project_dirs = load_config(temp_path)
                
                # Verify fallback values are used.
                assert cellpose_params["enable_parallel_processing"] is True  # Default fallback.
                assert cellpose_params["parallel_batch_size"] == 4  # Default fallback.
                assert cellpose_params["parallel_max_workers"] == 2  # Default fallback.
                assert cellpose_params["parallel_memory_limit_gb"] == 6.0  # Default fallback.
                assert cellpose_params["parallel_timeout_seconds"] == 300  # Default fallback.
                
                assert settings["gpu_max_retries"] == 3  # Default fallback.
                assert settings["gpu_timeout_seconds"] == 300  # Default fallback.
                
        finally:
            os.unlink(temp_path)
    
    def test_parameter_types(self, temp_config_file):
        """Test that parameters are loaded with correct data types."""
        with pytest.MonkeyPatch().context() as m:
            mock_dirs = {
                "configs": Path("configs"),
                "results": Path("results"),
                "logs": Path("logs"),
                "data": Path("data")
            }
            m.setattr("code.nuclei_segmentation.utils.project_setup.setup_project_structure", 
                     lambda: mock_dirs)
            
            settings, cellpose_params, project_dirs = load_config(temp_config_file)
            
            # Verify data types.
            assert isinstance(cellpose_params["enable_parallel_processing"], bool)
            assert isinstance(cellpose_params["parallel_batch_size"], int)
            assert isinstance(cellpose_params["parallel_max_workers"], int)
            assert isinstance(cellpose_params["parallel_memory_limit_gb"], float)
            assert isinstance(cellpose_params["parallel_timeout_seconds"], int)
            
            assert isinstance(settings["gpu_max_retries"], int)
            assert isinstance(settings["gpu_timeout_seconds"], int)
            assert isinstance(settings["gpu_memory_safety_factor"], float)
            assert isinstance(settings["gpu_spatial_strategy"], str)
            assert isinstance(settings["gpu_adaptive_batching"], bool)
            assert isinstance(settings["gpu_aggressive_cleanup"], bool)
    
    def test_parameter_ranges(self, temp_config_file):
        """Test that parameters are within expected ranges."""
        with pytest.MonkeyPatch().context() as m:
            mock_dirs = {
                "configs": Path("configs"),
                "results": Path("results"),
                "logs": Path("logs"),
                "data": Path("data")
            }
            m.setattr("code.nuclei_segmentation.utils.project_setup.setup_project_structure", 
                     lambda: mock_dirs)
            
            settings, cellpose_params, project_dirs = load_config(temp_config_file)
            
            # Verify parameter ranges.
            assert cellpose_params["parallel_batch_size"] >= 1
            assert cellpose_params["parallel_max_workers"] >= 1
            assert cellpose_params["parallel_memory_limit_gb"] > 0
            assert cellpose_params["parallel_timeout_seconds"] > 0
            
            assert settings["gpu_max_retries"] >= 1
            assert settings["gpu_timeout_seconds"] > 0
            assert settings["gpu_memory_safety_factor"] >= 1.0
            assert settings["gpu_spatial_strategy"] in ["adaptive", "2x2", "spatial", "hybrid"]


class TestConfigurationConsistency:
    """Test configuration consistency and validation."""
    
    def test_memory_limit_consistency(self, temp_config_file):
        """Test that memory limits are consistent across different components."""
        with pytest.MonkeyPatch().context() as m:
            mock_dirs = {
                "configs": Path("configs"),
                "results": Path("results"),
                "logs": Path("logs"),
                "data": Path("data")
            }
            m.setattr("code.nuclei_segmentation.utils.project_setup.setup_project_structure", 
                     lambda: mock_dirs)
            
            settings, cellpose_params, project_dirs = load_config(temp_config_file)
            
            # Verify memory limits are reasonable.
            parallel_memory = cellpose_params["parallel_memory_limit_gb"]
            gpu_memory = settings["gpu_memory_limit_gb"]
            
            # Parallel memory should be less than or equal to GPU memory for consistency.
            assert parallel_memory <= gpu_memory + 2.0  # Allow some tolerance.
            
            # Both should be within system memory constraints (6863MB = ~6.7GB).
            assert parallel_memory <= 6.7
            assert gpu_memory <= 10.0  # GPU memory can be higher.
    
    def test_timeout_consistency(self, temp_config_file):
        """Test that timeout values are consistent."""
        with pytest.MonkeyPatch().context() as m:
            mock_dirs = {
                "configs": Path("configs"),
                "results": Path("results"),
                "logs": Path("logs"),
                "data": Path("data")
            }
            m.setattr("code.nuclei_segmentation.utils.project_setup.setup_project_structure", 
                     lambda: mock_dirs)
            
            settings, cellpose_params, project_dirs = load_config(temp_config_file)
            
            # Verify timeout values are reasonable.
            parallel_timeout = cellpose_params["parallel_timeout_seconds"]
            gpu_timeout = settings["gpu_timeout_seconds"]
            
            # Both timeouts should be reasonable (not too short or too long).
            assert 30 <= parallel_timeout <= 3600  # 30 seconds to 1 hour.
            assert 30 <= gpu_timeout <= 3600  # 30 seconds to 1 hour.


if __name__ == "__main__":
    # Run tests with verbose output.
    pytest.main([__file__, "-v", "--tb=short"])
