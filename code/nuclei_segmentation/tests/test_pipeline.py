"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_pipeline.py.
Description:
    Comprehensive test suite for the pipeline module in kidney I/R injury
    spatial multiomics analysis. This test suite validates the complete
    nuclei segmentation pipeline including preprocessing, segmentation,
    merging, and post-processing steps for bioinformatics workflows.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pathlib, tempfile for testing infrastructure.
    • PIL for image handling and validation.
    • pipeline module from the nuclei_segmentation package.

Usage:
    pytest test_pipeline.py -v
    pytest test_pipeline.py::test_complete_pipeline_workflow -v

Inputs:
    • Synthetic tissue images and configuration files for testing.
    • Various pipeline scenarios for comprehensive validation.

Outputs:
    • Test results with validation of complete pipeline functionality.
    • Memory usage monitoring and workflow validation.

Key Features:
    • Complete pipeline workflow testing.
    • Configuration parameter validation.
    • Memory management for large tissue images.
    • Error handling and edge case testing.
    • Integration testing across all pipeline components.
    • Scientific context validation for kidney tissue analysis.

Notes:
    • This test suite validates the complete nuclei segmentation pipeline
      from raw tissue images to final merged segmentation masks.
    • Tests ensure proper integration of all pipeline components.
    • All tests include scientific context for kidney I/R injury analysis.
"""

import traceback
import pytest
import numpy as np
import tempfile
import time
import psutil
import os
from pathlib import Path
from typing import Dict, Any
from numpy.typing import NDArray
from unittest.mock import patch, MagicMock

# Import the module under test.
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pipeline import (
    run_segmentation_pipeline,
    log_config,
    setup_model,
    save_outputs,
    apply_postprocessing,
    generate_overlays
)


"""MEMORY AND TIMEOUT MONITORING"""

class MemoryMonitor:
    """Monitor memory usage during pipeline tests to prevent RAM overflow."""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
    
    def check_memory(self):
        """Check current memory usage and raise error if exceeded."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - self.initial_memory
        
        if memory_increase > self.max_memory_mb:
            raise MemoryError(f"Pipeline memory usage exceeded {self.max_memory_mb}MB: "
                            f"current increase = {memory_increase:.1f}MB")
        
        return memory_increase


def timeout_protection(timeout_seconds: int = 300):
    """Decorator to add timeout protection to pipeline test functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            def check_timeout():
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(f"Pipeline test {func.__name__} exceeded {timeout_seconds}s timeout")
            
            kwargs['_timeout_checker'] = check_timeout
            return func(*args, **kwargs)
        return wrapper
    return decorator


"""SYNTHETIC DATA GENERATION"""

def create_synthetic_tissue_image(height: int, width: int) -> NDArray[np.uint8]:
    """
    Create realistic synthetic tissue image for pipeline testing.
    
    This function generates tissue-like images that simulate kidney
    tissue sections for comprehensive pipeline testing.
    """
    
    # Create base tissue with realistic H&E-like coloring.
    tissue = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background tissue color (pinkish).
    tissue[:, :, 0] = 220  # Red channel.
    tissue[:, :, 1] = 180  # Green channel.
    tissue[:, :, 2] = 160  # Blue channel.
    
    # Add nuclei-like structures (darker blue regions).
    np.random.seed(42)  # Reproducible for testing.
    nucleus_count = (height * width) // 10000  # Realistic nucleus density.
    
    for _ in range(nucleus_count):
        center_y = np.random.randint(10, height - 10)
        center_x = np.random.randint(10, width - 10)
        radius = np.random.randint(5, 12)
        
        # Create circular nucleus-like structure.
        y_coords, x_coords = np.ogrid[:height, :width]
        mask = (y_coords - center_y)**2 + (x_coords - center_x)**2 <= radius**2
        
        # Make nuclei darker and more blue (hematoxylin staining).
        tissue[mask, 0] = np.clip(tissue[mask, 0] * 0.6, 0, 255)  # Reduce red.
        tissue[mask, 1] = np.clip(tissue[mask, 1] * 0.7, 0, 255)  # Reduce green.
        tissue[mask, 2] = np.clip(tissue[mask, 2] * 1.2, 0, 255)  # Enhance blue.
    
    # Add some texture variation.
    noise = np.random.randint(-20, 21, size=(height, width, 3))
    tissue = np.clip(tissue.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return tissue


def create_test_configuration() -> Dict[str, Any]:
    """
    Create test configuration for pipeline testing.
    
    This function generates a realistic configuration dictionary
    that mimics the actual pipeline configuration for kidney tissue analysis.
    """
    
    return {
        'input': {
            'image_path': 'test_tissue.tif',
            'output_dir': 'test_output'
        },
        'preprocessing': {
            'clahe_clip_limit': 2.0,
            'clahe_tile_grid_size': (8, 8),
            'gaussian_sigma': 1.0,
            'normalize': True
        },
        'segmentation': {
            'model_type': 'cyto',
            'diameter': 30,
            'flow_threshold': 0.4,
            'cellprob_threshold': 0.0,
            'use_gpu': False
        },
        'tiling': {
            'tile_size': 512,
            'overlap': 64,
            'min_tile_size': 256
        },
        'merging': {
            'overlap_threshold': 0.3,
            'use_gpu': False,
            'max_workers': 2
        },
        'output': {
            'save_intermediate': True,
            'generate_qc': True,
            'compression': 'lzw'
        }
    }


"""CONFIGURATION TESTS"""

class TestPipelineConfiguration:
    """Test pipeline configuration handling."""
    
    @timeout_protection(30)
    def test_load_configuration(self, _timeout_checker):
        """Test configuration loading from file."""
        _timeout_checker()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "test_config.yaml"
            
            # Create test configuration file.
            test_config = create_test_configuration()
            
            # Mock YAML loading since we don't want to depend on actual YAML files.
            with patch('yaml.safe_load') as mock_yaml:
                mock_yaml.return_value = test_config
                
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value = "mock file"
                    
                    config = load_configuration(str(config_path))
                    
                    assert config is not None
                    assert 'input' in config
                    assert 'segmentation' in config
                    assert 'merging' in config
        
        _timeout_checker()
    
    @timeout_protection(30)
    def test_validate_configuration(self, _timeout_checker):
        """Test configuration validation."""
        _timeout_checker()
        
        # Test valid configuration.
        valid_config = create_test_configuration()
        assert validate_configuration(valid_config) == True
        
        # Test invalid configuration (missing required sections).
        invalid_config = {'input': {'image_path': 'test.tif'}}
        assert validate_configuration(invalid_config) == False
        
        # Test configuration with invalid values.
        invalid_values_config = create_test_configuration()
        invalid_values_config['segmentation']['diameter'] = -1  # Invalid diameter.
        assert validate_configuration(invalid_values_config) == False
        
        _timeout_checker()
    
    @timeout_protection(30)
    def test_setup_output_directories(self, _timeout_checker):
        """Test output directory setup."""
        _timeout_checker()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "pipeline_output"
            
            # Test directory creation.
            setup_output_directories(str(output_dir))
            
            assert output_dir.exists()
            assert (output_dir / "segmentation_masks").exists()
            assert (output_dir / "tile_masks").exists()
            assert (output_dir / "qc").exists()
            assert (output_dir / "logs").exists()
        
        _timeout_checker()


"""PIPELINE COMPONENT TESTS"""

class TestPipelineComponents:
    """Test individual pipeline components."""
    
    @timeout_protection(60)
    def test_preprocess_image(self, _timeout_checker):
        """Test image preprocessing functionality."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=512)
        
        # Create test image.
        height, width = 512, 512
        test_image = create_synthetic_tissue_image(height, width)
        
        # Test preprocessing.
        config = create_test_configuration()
        preprocessed = preprocess_image(test_image, config['preprocessing'])
        
        # Verify preprocessing results.
        assert preprocessed.shape == test_image.shape
        assert preprocessed.dtype == np.uint8
        assert not np.array_equal(preprocessed, test_image), "Image should be modified by preprocessing"
        
        memory_monitor.check_memory()
        _timeout_checker()
    
    @timeout_protection(120)
    def test_run_segmentation(self, _timeout_checker):
        """Test segmentation functionality with mocked Cellpose."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=1024)
        
        # Create test image.
        height, width = 512, 512
        test_image = create_synthetic_tissue_image(height, width)
        
        # Mock Cellpose model to avoid dependency on actual model.
        # Use CellposeModel for Cellpose 4.0+ compatibility.
        with patch('cellpose.models.CellposeModel') as mock_cellpose:
            # Create mock segmentation result.
            mock_masks = np.random.randint(0, 100, size=(height, width), dtype=np.uint32)
            mock_flows = [np.random.rand(height, width, 2)]
            mock_styles = np.random.rand(256)
            mock_diams = np.array([30.0])
            
            mock_model = MagicMock()
            mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles, mock_diams)
            mock_cellpose.return_value = mock_model
            
            # Test segmentation.
            config = create_test_configuration()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                masks, flows = run_segmentation(
                    test_image, 
                    config['segmentation'], 
                    str(temp_path)
                )
                
                # Verify segmentation results.
                assert masks.shape == (height, width)
                assert masks.dtype == np.uint32
                assert len(flows) > 0
        
        memory_monitor.check_memory()
        _timeout_checker()
    
    @timeout_protection(180)
    def test_merge_segmentation_results(self, _timeout_checker):
        """Test segmentation result merging."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=1024)
        
        # Create test tile masks.
        height, width = 1024, 1024
        tile_h, tile_w = 256, 256
        overlap = 64
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            tiles_dir = temp_path / "tile_masks"
            tiles_dir.mkdir()
            
            # Create synthetic tile masks.
            stride_h = tile_h - overlap
            stride_w = tile_w - overlap
            n_rows = (height + stride_h - 1) // stride_h
            n_cols = (width + stride_w - 1) // stride_w
            
            for r in range(n_rows):
                for c in range(n_cols):
                    global_y = r * stride_h
                    global_x = c * stride_w
                    
                    # Create tile mask with some nuclei.
                    tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint32)
                    if r < n_rows - 1 and c < n_cols - 1:  # Avoid edge complications for this test.
                        tile_mask[50:70, 50:70] = r * n_cols + c + 1
                    
                    # Save tile mask.
                    tile_path = tiles_dir / f"{global_y}_{global_x}.npz"
                    np.savez_compressed(tile_path, mask=tile_mask)
            
            memory_monitor.check_memory()
            _timeout_checker()
            
            # Test merging.
            config = create_test_configuration()
            merged_mask = merge_segmentation_results(
                height, width, tile_h, tile_w, overlap,
                str(tiles_dir), config['merging']
            )
            
            # Verify merging results.
            assert merged_mask.shape == (height, width)
            assert merged_mask.dtype == np.uint32
            assert np.count_nonzero(merged_mask) > 0, "Merged mask should contain nuclei"
        
        memory_monitor.check_memory()
        _timeout_checker()


"""INTEGRATION TESTS"""

class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    @timeout_protection(300)
    def test_complete_pipeline_workflow(self, _timeout_checker):
        """Test complete pipeline workflow from image to final results."""
        _timeout_checker()
        
        memory_monitor = MemoryMonitor(max_memory_mb=2048)
        
        # Create test data.
        height, width = 1024, 1024
        test_image = create_synthetic_tissue_image(height, width)
        config = create_test_configuration()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save test image.
            image_path = temp_path / "test_tissue.tif"
            Image.fromarray(test_image).save(image_path)
            
            # Update config with actual paths.
            config['input']['image_path'] = str(image_path)
            config['input']['output_dir'] = str(temp_path / "output")
            
            memory_monitor.check_memory()
            _timeout_checker()
            
            # Mock Cellpose to avoid dependency.
            # Use CellposeModel for Cellpose 4.0+ compatibility.
            with patch('cellpose.models.CellposeModel') as mock_cellpose:
                # Create realistic mock segmentation.
                mock_masks = np.zeros((height, width), dtype=np.uint32)
                
                # Add some mock nuclei.
                nucleus_id = 1
                for y in range(50, height - 50, 100):
                    for x in range(50, width - 50, 100):
                        mock_masks[y-10:y+10, x-10:x+10] = nucleus_id
                        nucleus_id += 1
                
                mock_flows = [np.random.rand(height, width, 2)]
                mock_styles = np.random.rand(256)
                mock_diams = np.array([30.0])
                
                mock_model = MagicMock()
                mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles, mock_diams)
                mock_cellpose.return_value = mock_model
                
                # Run complete pipeline.
                result = run_nuclei_segmentation_pipeline(config)
                
                # Verify pipeline results.
                assert result is not None
                assert 'merged_mask' in result
                assert 'statistics' in result
                
                # Verify output files.
                output_dir = Path(config['input']['output_dir'])
                assert output_dir.exists()
                assert (output_dir / "segmentation_masks").exists()
                
                # Verify merged mask.
                merged_mask = result['merged_mask']
                assert merged_mask.shape == (height, width)
                assert merged_mask.dtype == np.uint32
                
                memory_monitor.check_memory()
                _timeout_checker()
    
    @timeout_protection(60)
    def test_pipeline_error_handling(self, _timeout_checker):
        """Test pipeline error handling and recovery."""
        _timeout_checker()
        
        # Test with invalid configuration.
        invalid_config = {'invalid': 'config'}
        
        with pytest.raises((ValueError, KeyError, TypeError)):
            run_nuclei_segmentation_pipeline(invalid_config)
        
        # Test with non-existent image file.
        config = create_test_configuration()
        config['input']['image_path'] = '/non/existent/path.tif'
        
        with pytest.raises(FileNotFoundError):
            run_nuclei_segmentation_pipeline(config)
        
        _timeout_checker()
    
    @timeout_protection(120)
    def test_pipeline_memory_management(self, _timeout_checker):
        """Test pipeline memory management with large images."""
        _timeout_checker()
        
        # Use strict memory monitoring.
        memory_monitor = MemoryMonitor(max_memory_mb=1536)
        
        # Test with moderately large image.
        height, width = 2048, 2048
        
        # Create minimal test image to save memory.
        test_image = np.full((height, width, 3), 128, dtype=np.uint8)
        
        # Add minimal structure for segmentation.
        test_image[1000:1100, 1000:1100] = [200, 150, 180]  # Single large structure.
        
        config = create_test_configuration()
        config['tiling']['tile_size'] = 512  # Smaller tiles for memory efficiency.
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save test image.
            image_path = temp_path / "large_tissue.tif"
            Image.fromarray(test_image).save(image_path)
            
            config['input']['image_path'] = str(image_path)
            config['input']['output_dir'] = str(temp_path / "output")
            
            memory_monitor.check_memory()
            _timeout_checker()
            
            # Mock Cellpose with minimal output.
            # Use CellposeModel for Cellpose 4.0+ compatibility.
            with patch('cellpose.models.CellposeModel') as mock_cellpose:
                # Create minimal mock segmentation to save memory.
                mock_masks = np.zeros((height, width), dtype=np.uint32)
                mock_masks[1000:1100, 1000:1100] = 1  # Single nucleus.
                
                mock_flows = [np.zeros((height, width, 2))]  # Minimal flows.
                mock_styles = np.zeros(256)
                mock_diams = np.array([30.0])
                
                mock_model = MagicMock()
                mock_model.eval.return_value = (mock_masks, mock_flows, mock_styles, mock_diams)
                mock_cellpose.return_value = mock_model
                
                # Run pipeline with memory monitoring.
                result = run_nuclei_segmentation_pipeline(config)
                
                # Verify results and memory usage.
                assert result is not None
                memory_increase = memory_monitor.check_memory()
                
                # Memory increase should be reasonable for image size.
                expected_memory_mb = (height * width * 4) / 1024 / 1024  # Rough estimate.
                assert memory_increase < expected_memory_mb * 3, \
                    f"Memory usage too high: {memory_increase:.1f}MB"
                
                _timeout_checker()
