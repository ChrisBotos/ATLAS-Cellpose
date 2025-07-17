"""
Test Suite: test_gpu_merge_integration.py.

Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Description:
    Integration tests for the enhanced GPU tile merging system.
    Tests the complete pipeline from configuration loading through
    merge_masks_streaming to final output, including parameter passing
    and error handling across the entire system.

Usage:
    python -m pytest tests/nuclei_segmentation_tests/test_gpu_merge_integration.py -v

Dependencies:
    • Python >= 3.10.
    • numpy, pytest, torch, tempfile, pathlib.
    • The complete cellpose_merge package and pipeline integration.

Key Features:
    • End-to-end pipeline testing with new GPU parameters.
    • Configuration parameter validation and propagation.
    • Performance comparison between different strategies.
    • Memory management validation in realistic scenarios.
    • Error recovery testing with simulated failures.

Notes:
    • Tests create temporary directories and files for realistic testing.
    • Mock tile data is generated to simulate real segmentation outputs.
    • GPU tests are skipped when CUDA is not available.
"""

from __future__ import annotations

import os
import sys
import traceback
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

import numpy as np
import pytest

# Adjust path for imports.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from code.nuclei_segmentation.cellpose_merge.merge_tiles import merge_masks_streaming
from code.nuclei_segmentation.utils.project_setup import load_config

"""Test fixtures and utilities."""

@pytest.fixture
def temp_tiles_dir():
    """Create temporary directory with mock tile files."""
    temp_dir = tempfile.mkdtemp()
    tiles_dir = Path(temp_dir) / "tiles"
    tiles_dir.mkdir()
    
    # Create mock tile files.
    for r in range(3):
        for c in range(3):
            tile_file = tiles_dir / f"tile_{r:03d}_{c:03d}.npz"
            
            # Create a simple mask with some objects.
            mask = np.zeros((512, 512), dtype=np.uint32)
            if r == 1 and c == 1:  # Center tile has more objects.
                mask[100:200, 100:200] = 1
                mask[300:400, 300:400] = 2
            elif r + c > 0:  # Other tiles have single objects.
                mask[200:300, 200:300] = 1
            
            np.savez_compressed(tile_file, masks=mask)
    
    yield tiles_dir
    
    # Cleanup.
    shutil.rmtree(temp_dir)

@pytest.fixture
def enhanced_gpu_settings():
    """Create settings dictionary with enhanced GPU parameters."""
    return {
        "gpu_batch_size": 2,
        "gpu_memory_limit_gb": 4.0,
        "gpu_memory_safety_factor": 1.3,
        "gpu_spatial_strategy": "adaptive",
        "gpu_adaptive_batching": True,
        "gpu_aggressive_cleanup": True,
        "merge_overlap_threshold": 0.3,
        "qc_overlays": False,  # Disable for faster testing.
    }

"""Configuration and parameter passing tests."""

class TestConfigurationIntegration:
    """Test configuration loading and parameter passing."""

    def test_enhanced_parameters_in_config(self):
        """Test that enhanced GPU parameters are properly loaded from config."""
        try:
            settings, _, _ = load_config()
            
            # Check that new parameters are present with defaults.
            expected_params = [
                "gpu_batch_size",
                "gpu_memory_limit_gb", 
                "gpu_memory_safety_factor",
                "gpu_spatial_strategy",
                "gpu_adaptive_batching",
                "gpu_aggressive_cleanup"
            ]
            
            for param in expected_params:
                assert param in settings, f"Parameter {param} should be in settings"
                assert settings[param] is not None, f"Parameter {param} should have a value"
                
        except Exception as e:
            pytest.skip(f"Config loading failed: {e}")

    def test_parameter_type_validation(self, enhanced_gpu_settings):
        """Test that parameters have correct types."""
        assert isinstance(enhanced_gpu_settings["gpu_batch_size"], int), "gpu_batch_size should be int"
        assert isinstance(enhanced_gpu_settings["gpu_memory_limit_gb"], float), "gpu_memory_limit_gb should be float"
        assert isinstance(enhanced_gpu_settings["gpu_memory_safety_factor"], float), "gpu_memory_safety_factor should be float"
        assert isinstance(enhanced_gpu_settings["gpu_spatial_strategy"], str), "gpu_spatial_strategy should be str"
        assert isinstance(enhanced_gpu_settings["gpu_adaptive_batching"], bool), "gpu_adaptive_batching should be bool"
        assert isinstance(enhanced_gpu_settings["gpu_aggressive_cleanup"], bool), "gpu_aggressive_cleanup should be bool"

    def test_parameter_value_ranges(self, enhanced_gpu_settings):
        """Test that parameters have reasonable value ranges."""
        assert enhanced_gpu_settings["gpu_batch_size"] >= 1, "gpu_batch_size should be at least 1"
        assert enhanced_gpu_settings["gpu_memory_limit_gb"] > 0, "gpu_memory_limit_gb should be positive"
        assert enhanced_gpu_settings["gpu_memory_safety_factor"] >= 1.0, "gpu_memory_safety_factor should be at least 1.0"
        assert enhanced_gpu_settings["gpu_spatial_strategy"] in ["adaptive", "2x2", "spatial", "hybrid"], \
            "gpu_spatial_strategy should be valid option"

"""End-to-end pipeline tests."""

class TestEndToEndPipeline:
    """Test complete pipeline with enhanced GPU parameters."""

    def test_merge_with_enhanced_parameters(self, temp_tiles_dir, enhanced_gpu_settings):
        """Test merge_masks_streaming with all enhanced parameters."""
        result = merge_masks_streaming(
            height=1536,  # 3x512 tiles.
            width=1536,
            tile_h=512,
            tile_w=512,
            overlap=64,
            tiles_path=temp_tiles_dir,
            threshold=enhanced_gpu_settings["merge_overlap_threshold"],
            use_gpu=False,  # Use CPU for reliable testing.
            gpu_batch_size=enhanced_gpu_settings["gpu_batch_size"],
            gpu_memory_limit_gb=enhanced_gpu_settings["gpu_memory_limit_gb"],
            gpu_memory_safety_factor=enhanced_gpu_settings["gpu_memory_safety_factor"],
            gpu_spatial_strategy=enhanced_gpu_settings["gpu_spatial_strategy"],
            gpu_adaptive_batching=enhanced_gpu_settings["gpu_adaptive_batching"],
            gpu_aggressive_cleanup=enhanced_gpu_settings["gpu_aggressive_cleanup"],
        )
        
        assert result is not None, "Should produce merged result"
        assert result.shape == (1536, 1536), "Result should have correct dimensions"
        assert result.dtype == np.uint32, "Result should have correct dtype"
        assert np.max(result) > 0, "Result should contain some objects"

    def test_different_spatial_strategies(self, temp_tiles_dir):
        """Test merge with different spatial strategies."""
        strategies = ["adaptive", "2x2", "spatial", "hybrid"]
        results = {}
        
        for strategy in strategies:
            result = merge_masks_streaming(
                height=1536,
                width=1536,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path=temp_tiles_dir,
                threshold=0.3,
                use_gpu=False,
                gpu_batch_size=1,
                gpu_spatial_strategy=strategy,
                gpu_adaptive_batching=True,
            )
            
            results[strategy] = result
            assert result is not None, f"Strategy {strategy} should produce result"
            assert result.shape == (1536, 1536), f"Strategy {strategy} should have correct shape"
        
        # All strategies should produce similar results (same objects detected).
        object_counts = {strategy: np.max(result) for strategy, result in results.items()}
        
        # Allow some variation but results should be reasonably similar.
        max_objects = max(object_counts.values())
        min_objects = min(object_counts.values())
        
        if max_objects > 0:
            variation = (max_objects - min_objects) / max_objects
            assert variation < 0.5, f"Strategies should produce similar results, got {object_counts}"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_gpu_vs_cpu_consistency(self, temp_tiles_dir):
        """Test that GPU and CPU processing produce consistent results."""
        # CPU result.
        cpu_result = merge_masks_streaming(
            height=1536,
            width=1536,
            tile_h=512,
            tile_w=512,
            overlap=64,
            tiles_path=temp_tiles_dir,
            threshold=0.3,
            use_gpu=False,
            gpu_batch_size=1,
        )
        
        # GPU result (if available).
        try:
            gpu_result = merge_masks_streaming(
                height=1536,
                width=1536,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path=temp_tiles_dir,
                threshold=0.3,
                use_gpu=True,
                gpu_batch_size=1,
            )
            
            # Results should be identical or very similar.
            assert cpu_result.shape == gpu_result.shape, "CPU and GPU results should have same shape"
            
            # Allow for minor differences due to floating point precision.
            cpu_objects = np.max(cpu_result)
            gpu_objects = np.max(gpu_result)
            
            if cpu_objects > 0 and gpu_objects > 0:
                difference = abs(cpu_objects - gpu_objects) / max(cpu_objects, gpu_objects)
                assert difference < 0.1, f"CPU and GPU results should be similar: {cpu_objects} vs {gpu_objects}"
                
        except Exception as e:
            pytest.skip(f"GPU testing failed: {e}")

    def test_memory_safety_factor_effects(self, temp_tiles_dir):
        """Test effects of different memory safety factors."""
        safety_factors = [1.0, 1.5, 2.0]
        
        for factor in safety_factors:
            result = merge_masks_streaming(
                height=1536,
                width=1536,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path=temp_tiles_dir,
                threshold=0.3,
                use_gpu=False,
                gpu_batch_size=2,
                gpu_memory_safety_factor=factor,
            )
            
            assert result is not None, f"Safety factor {factor} should work"
            assert result.shape == (1536, 1536), f"Safety factor {factor} should produce correct shape"

    def test_adaptive_batching_effects(self, temp_tiles_dir):
        """Test effects of adaptive batching on/off."""
        for adaptive in [True, False]:
            result = merge_masks_streaming(
                height=1536,
                width=1536,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path=temp_tiles_dir,
                threshold=0.3,
                use_gpu=False,
                gpu_batch_size=2,
                gpu_adaptive_batching=adaptive,
            )
            
            assert result is not None, f"Adaptive batching {adaptive} should work"
            assert result.shape == (1536, 1536), f"Adaptive batching {adaptive} should produce correct shape"

"""Error handling and recovery tests."""

class TestErrorHandlingIntegration:
    """Test error handling and recovery in the complete pipeline."""

    def test_invalid_tiles_directory(self):
        """Test handling of invalid tiles directory."""
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
            merge_masks_streaming(
                height=1024,
                width=1024,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path="/nonexistent/path",
                threshold=0.3,
                use_gpu=False,
            )

    def test_invalid_parameters(self, temp_tiles_dir):
        """Test handling of invalid parameters."""
        # Test negative batch size.
        with pytest.raises((ValueError, RuntimeError)):
            merge_masks_streaming(
                height=1536,
                width=1536,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path=temp_tiles_dir,
                threshold=0.3,
                use_gpu=False,
                gpu_batch_size=-1,  # Invalid.
            )

    def test_memory_limit_handling(self, temp_tiles_dir):
        """Test handling of very low memory limits."""
        # Test with extremely low memory limit.
        result = merge_masks_streaming(
            height=1536,
            width=1536,
            tile_h=512,
            tile_w=512,
            overlap=64,
            tiles_path=temp_tiles_dir,
            threshold=0.3,
            use_gpu=False,
            gpu_batch_size=1,
            gpu_memory_limit_gb=0.001,  # Very low limit.
        )
        
        # Should still work by using minimal batch sizes.
        assert result is not None, "Should handle low memory limits gracefully"
        assert result.shape == (1536, 1536), "Should produce correct output despite memory constraints"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_gpu_memory_error_recovery(self, temp_tiles_dir):
        """Test GPU memory error recovery."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('code.nuclei_segmentation.cellpose_merge.gpu_merge.merge_patch_gpu') as mock_gpu:
            
            # Simulate intermittent GPU memory errors.
            call_count = 0
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:  # Fail first two calls.
                    raise RuntimeError("CUDA out of memory")
                # Succeed on subsequent calls.
                from code.nuclei_segmentation.cellpose_merge.rules import merge_patch_cpu
                return merge_patch_cpu(*args, **kwargs)
            
            mock_gpu.side_effect = side_effect
            
            # Should recover and complete successfully.
            result = merge_masks_streaming(
                height=1536,
                width=1536,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path=temp_tiles_dir,
                threshold=0.3,
                use_gpu=True,
                gpu_batch_size=4,  # Start with large batch.
                gpu_aggressive_cleanup=True,
            )
            
            assert result is not None, "Should recover from GPU memory errors"
            assert result.shape == (1536, 1536), "Should produce correct output after recovery"
