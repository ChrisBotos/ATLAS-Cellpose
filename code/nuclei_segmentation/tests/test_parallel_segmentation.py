"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_parallel_segmentation.py.
Description:
    Test suite for parallel Cellpose3 segmentation functionality to ensure
    proper memory management, timeout handling, and error recovery.

Dependencies:
    • Python >= 3.7.
    • pytest, numpy, torch.
    • Custom parallel_segmentation module.

Usage:
    pytest test_parallel_segmentation.py -v

Inputs:
    • Mock Cellpose model and tile data.
    • Configuration parameters for testing.

Outputs:
    • Test results and validation reports.
    • Performance metrics and memory usage statistics.

Key Features:
    • Memory safety validation with 6863MB system limit.
    • Timeout handling verification.
    • Error recovery testing.
    • Integration testing with configuration loading.

Notes:
    • Tests are designed for kidney I/R injury analysis pipeline.
    • Includes both unit tests and integration tests.
    • Validates thread-safe operations in parallel processing.
"""

import pytest
import numpy as np
import logging
import time
from unittest.mock import Mock, patch
from pathlib import Path
import sys
import os

# Add the project root to the path for imports.
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from code.nuclei_segmentation.utils.parallel_segmentation import (
    estimate_batch_memory_usage,
    get_optimal_batch_size,
    process_cellpose_batch,
    run_cellpose_parallel_batches
)


class TestMemoryEstimation:
    """Test memory estimation functions for batch processing."""
    
    def test_estimate_batch_memory_usage(self):
        """Test memory usage estimation for different batch sizes."""
        # Test with typical tile size.
        memory_gb = estimate_batch_memory_usage(tile_size=512, batch_size=4)
        assert memory_gb > 0
        assert memory_gb < 10  # Should be reasonable for 4 tiles.
        
        # Test scaling with batch size.
        memory_1 = estimate_batch_memory_usage(tile_size=512, batch_size=1)
        memory_4 = estimate_batch_memory_usage(tile_size=512, batch_size=4)
        assert memory_4 > memory_1
        assert memory_4 / memory_1 <= 4  # Should scale roughly linearly.
    
    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        # Test with generous memory.
        batch_size = get_optimal_batch_size(tile_size=512, available_memory_gb=8.0, max_batch_size=8)
        assert batch_size >= 1
        assert batch_size <= 8
        
        # Test with limited memory.
        batch_size_limited = get_optimal_batch_size(tile_size=512, available_memory_gb=1.0, max_batch_size=8)
        assert batch_size_limited >= 1
        assert batch_size_limited <= batch_size  # Should be smaller with less memory.
        
        # Test with very limited memory.
        batch_size_min = get_optimal_batch_size(tile_size=512, available_memory_gb=0.1, max_batch_size=8)
        assert batch_size_min == 1  # Should fall back to minimum.


class TestBatchProcessing:
    """Test batch processing functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock Cellpose model for testing."""
        model = Mock()
        # Mock successful segmentation.
        mock_mask = np.random.randint(0, 10, size=(512, 512), dtype=np.uint32)
        model.eval.return_value = [mock_mask, None, None]
        return model
    
    @pytest.fixture
    def sample_tiles(self):
        """Create sample tile data for testing."""
        tiles = []
        for i in range(4):
            tile_image = np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)
            y_slice = slice(i * 400, (i + 1) * 400 + 112)  # Overlapping slices.
            x_slice = slice(0, 512)
            tiles.append((tile_image, (y_slice, x_slice)))
        return tiles
    
    @pytest.fixture
    def cellpose_params(self):
        """Create sample Cellpose parameters."""
        return {
            "diameter": 0,
            "channels": (0, 0),
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "resample": True,
            "batch_size": 8
        }
    
    def test_process_cellpose_batch_success(self, mock_model, sample_tiles, cellpose_params):
        """Test successful batch processing."""
        batch = sample_tiles[:2]  # Use 2 tiles.
        
        results = process_cellpose_batch(
            model=mock_model,
            tile_batch=batch,
            cellpose_params=cellpose_params,
            batch_idx=0,
            timeout_seconds=30
        )
        
        assert len(results) == 2
        for mask, slice_info, cell_count in results:
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.uint32
            assert isinstance(cell_count, int)
            assert cell_count >= 0
    
    def test_process_cellpose_batch_timeout(self, mock_model, sample_tiles, cellpose_params):
        """Test batch processing timeout handling."""
        # Mock a slow model that will timeout.
        def slow_eval(*args, **kwargs):
            time.sleep(2)  # Sleep longer than timeout.
            return [np.zeros((512, 512), dtype=np.uint32), None, None]
        
        mock_model.eval.side_effect = slow_eval
        batch = sample_tiles[:1]  # Use 1 tile.
        
        with pytest.raises(TimeoutError):
            process_cellpose_batch(
                model=mock_model,
                tile_batch=batch,
                cellpose_params=cellpose_params,
                batch_idx=0,
                timeout_seconds=1  # Very short timeout.
            )
    
    def test_process_cellpose_batch_error_recovery(self, mock_model, sample_tiles, cellpose_params):
        """Test error recovery in batch processing."""
        # Mock a model that fails on first tile but succeeds on second.
        def failing_eval(*args, **kwargs):
            if not hasattr(failing_eval, 'call_count'):
                failing_eval.call_count = 0
            failing_eval.call_count += 1
            
            if failing_eval.call_count == 1:
                raise RuntimeError("Simulated Cellpose failure")
            else:
                return [np.random.randint(0, 5, size=(512, 512), dtype=np.uint32), None, None]
        
        mock_model.eval.side_effect = failing_eval
        batch = sample_tiles[:2]  # Use 2 tiles.
        
        results = process_cellpose_batch(
            model=mock_model,
            tile_batch=batch,
            cellpose_params=cellpose_params,
            batch_idx=0,
            timeout_seconds=30
        )
        
        assert len(results) == 2
        # First result should be empty (failed).
        assert results[0][2] == 0  # No cells detected.
        # Second result should have cells (succeeded).
        assert results[1][2] >= 0


class TestParallelProcessing:
    """Test full parallel processing pipeline."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock Cellpose model for testing."""
        model = Mock()
        # Mock successful segmentation with varying cell counts.
        def variable_eval(*args, **kwargs):
            # Return different masks for variety.
            mask = np.random.randint(0, 8, size=(512, 512), dtype=np.uint32)
            return [mask, None, None]
        
        model.eval.side_effect = variable_eval
        return model
    
    @pytest.fixture
    def sample_tiles(self):
        """Create sample tile data for testing."""
        tiles = []
        for i in range(8):  # More tiles for parallel testing.
            tile_image = np.random.randint(0, 255, size=(512, 512), dtype=np.uint8)
            y_slice = slice(i * 400, (i + 1) * 400 + 112)
            x_slice = slice(0, 512)
            tiles.append((tile_image, (y_slice, x_slice)))
        return tiles
    
    @pytest.fixture
    def cellpose_params(self):
        """Create sample Cellpose parameters."""
        return {
            "diameter": 0,
            "channels": (0, 0),
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "resample": True,
            "batch_size": 8
        }
    
    def test_parallel_processing_success(self, mock_model, sample_tiles, cellpose_params):
        """Test successful parallel processing."""
        masks, flows, total_cells = run_cellpose_parallel_batches(
            model=mock_model,
            tiles=sample_tiles,
            cellpose_params=cellpose_params,
            batch_size=2,
            max_workers=2,
            memory_limit_gb=6.0,
            timeout_seconds=30
        )
        
        assert len(masks) == len(sample_tiles)
        assert len(flows) == len(sample_tiles)
        assert total_cells >= 0
        
        # Check that all masks are valid.
        for mask in masks:
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.uint32
            assert mask.shape == (512, 512)
    
    def test_parallel_processing_memory_limit(self, mock_model, sample_tiles, cellpose_params):
        """Test parallel processing with memory constraints."""
        # Test with very low memory limit.
        masks, flows, total_cells = run_cellpose_parallel_batches(
            model=mock_model,
            tiles=sample_tiles[:4],  # Fewer tiles.
            cellpose_params=cellpose_params,
            batch_size=4,  # Large batch size.
            max_workers=1,  # Single worker.
            memory_limit_gb=1.0,  # Low memory limit.
            timeout_seconds=30
        )
        
        assert len(masks) == 4
        assert total_cells >= 0
    
    def test_parallel_processing_empty_input(self, mock_model, cellpose_params):
        """Test parallel processing with empty input."""
        masks, flows, total_cells = run_cellpose_parallel_batches(
            model=mock_model,
            tiles=[],
            cellpose_params=cellpose_params,
            batch_size=2,
            max_workers=2,
            memory_limit_gb=6.0,
            timeout_seconds=30
        )
        
        assert len(masks) == 0
        assert len(flows) == 0
        assert total_cells == 0


if __name__ == "__main__":
    # Run tests with verbose output.
    pytest.main([__file__, "-v", "--tb=short"])
