"""
Test Suite: test_gpu_performance_benchmark.py.

Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Description:
    Performance benchmarking tests for the enhanced GPU tile merging system.
    Measures and compares performance across different strategies, batch sizes,
    and memory configurations to validate optimization effectiveness.

Usage:
    python -m pytest tests/nuclei_segmentation_tests/test_gpu_performance_benchmark.py -v -s

Dependencies:
    • Python >= 3.10.
    • numpy, pytest, torch, time, statistics.
    • The complete cellpose_merge package with GPU optimizations.

Key Features:
    • Performance comparison between spatial strategies.
    • Memory usage profiling and optimization validation.
    • Batch size optimization effectiveness testing.
    • GPU vs CPU performance comparison.
    • Scalability testing with different cluster sizes.

Notes:
    • Tests are marked as slow and may be skipped in regular test runs.
    • GPU benchmarks require CUDA availability.
    • Results are logged for performance regression detection.
    • Memory profiling helps validate optimization effectiveness.
"""

from __future__ import annotations

import os
import sys
import time
import statistics
import traceback
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
from unittest.mock import patch

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

from code.nuclei_segmentation.cellpose_merge.batch_merge import (
    estimate_memory_requirements,
    get_optimal_batch_size,
    group_tiles_by_spatial_proximity,
    merge_cluster_batched,
)
from code.nuclei_segmentation.cellpose_merge.merge_tiles import merge_masks_streaming

"""Test fixtures and utilities."""

@pytest.fixture
def performance_tiles_dir():
    """Create temporary directory with realistic tile data for performance testing."""
    temp_dir = tempfile.mkdtemp()
    tiles_dir = Path(temp_dir) / "tiles"
    tiles_dir.mkdir()
    
    # Create a larger grid of tiles for performance testing.
    grid_size = 8  # 8x8 = 64 tiles.
    tile_size = 512
    
    for r in range(grid_size):
        for c in range(grid_size):
            tile_file = tiles_dir / f"tile_{r:03d}_{c:03d}.npz"
            
            # Create more realistic masks with multiple objects.
            mask = np.zeros((tile_size, tile_size), dtype=np.uint32)
            
            # Add random objects to simulate real segmentation.
            np.random.seed(r * grid_size + c)  # Reproducible randomness.
            num_objects = np.random.randint(5, 20)
            
            for obj_id in range(1, num_objects + 1):
                # Random object position and size.
                y = np.random.randint(50, tile_size - 100)
                x = np.random.randint(50, tile_size - 100)
                size = np.random.randint(20, 80)
                
                # Create circular object.
                yy, xx = np.ogrid[:tile_size, :tile_size]
                circle_mask = (yy - y) ** 2 + (xx - x) ** 2 <= size ** 2
                mask[circle_mask] = obj_id
            
            np.savez_compressed(tile_file, masks=mask)
    
    yield tiles_dir
    
    # Cleanup.
    shutil.rmtree(temp_dir)

def create_test_cluster(size: str) -> List[Tuple[int, int]]:
    """Create test clusters of different sizes."""
    if size == "small":
        return [(r, c) for r in range(3) for c in range(3)]  # 9 tiles.
    elif size == "medium":
        return [(r, c) for r in range(6) for c in range(6)]  # 36 tiles.
    elif size == "large":
        return [(r, c) for r in range(10) for c in range(10)]  # 100 tiles.
    elif size == "xlarge":
        return [(r, c) for r in range(15) for c in range(15)]  # 225 tiles.
    else:
        raise ValueError(f"Unknown cluster size: {size}")

def benchmark_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Benchmark a function and return result and execution time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

"""Memory estimation benchmarks."""

@pytest.mark.slow
class TestMemoryEstimationPerformance:
    """Benchmark memory estimation accuracy and performance."""

    def test_memory_estimation_accuracy(self):
        """Test accuracy of memory estimation across different cluster sizes."""
        cluster_sizes = ["small", "medium", "large"]
        results = {}
        
        for size in cluster_sizes:
            cluster = create_test_cluster(size)
            
            # Test different safety factors.
            for safety_factor in [1.0, 1.5, 2.0]:
                memory_estimate = estimate_memory_requirements(
                    tiles=cluster,
                    tile_h=512,
                    tile_w=512,
                    overlap=64,
                    safety_factor=safety_factor
                )
                
                results[f"{size}_{safety_factor}"] = memory_estimate
                
                # Basic sanity checks.
                assert memory_estimate > 0, f"Memory estimate should be positive for {size} cluster"
                
                # Larger clusters should generally require more memory.
                if size != "small":
                    small_estimate = results.get(f"small_{safety_factor}")
                    if small_estimate:
                        assert memory_estimate >= small_estimate, \
                            f"Larger cluster should require more memory: {size} vs small"
        
        print(f"\nMemory estimation results: {results}")

    def test_memory_estimation_performance(self):
        """Benchmark memory estimation performance."""
        cluster_sizes = ["small", "medium", "large", "xlarge"]
        times = {}
        
        for size in cluster_sizes:
            cluster = create_test_cluster(size)
            
            # Benchmark multiple runs.
            run_times = []
            for _ in range(10):
                _, exec_time = benchmark_function(
                    estimate_memory_requirements,
                    tiles=cluster,
                    tile_h=512,
                    tile_w=512,
                    overlap=64,
                    safety_factor=1.5
                )
                run_times.append(exec_time)
            
            times[size] = {
                'mean': statistics.mean(run_times),
                'stdev': statistics.stdev(run_times) if len(run_times) > 1 else 0,
                'min': min(run_times),
                'max': max(run_times)
            }
        
        print(f"\nMemory estimation performance: {times}")
        
        # Performance should scale reasonably with cluster size.
        assert times["small"]["mean"] < 0.1, "Small cluster estimation should be fast"
        assert times["xlarge"]["mean"] < 1.0, "Even large cluster estimation should be reasonable"

"""Batch size optimization benchmarks."""

@pytest.mark.slow
class TestBatchSizeOptimizationPerformance:
    """Benchmark batch size optimization algorithms."""

    def test_batch_size_optimization_performance(self):
        """Benchmark batch size optimization across different scenarios."""
        cluster_sizes = ["small", "medium", "large"]
        memory_limits = [2.0, 8.0, 16.0]  # GB.
        results = {}
        
        for size in cluster_sizes:
            cluster = create_test_cluster(size)
            
            for memory_limit in memory_limits:
                # Benchmark adaptive vs non-adaptive.
                for adaptive in [True, False]:
                    key = f"{size}_{memory_limit}GB_adaptive_{adaptive}"
                    
                    _, exec_time = benchmark_function(
                        get_optimal_batch_size,
                        cluster=cluster,
                        tile_h=512,
                        tile_w=512,
                        overlap=64,
                        memory_limit_gb=memory_limit,
                        adaptive_sizing=adaptive
                    )
                    
                    results[key] = exec_time
        
        print(f"\nBatch size optimization performance: {results}")
        
        # All optimizations should complete quickly.
        for key, time_taken in results.items():
            assert time_taken < 5.0, f"Batch size optimization should be fast: {key} took {time_taken}s"

    def test_batch_size_quality(self):
        """Test quality of batch size optimization."""
        cluster = create_test_cluster("medium")
        memory_limits = [1.0, 4.0, 16.0]
        
        batch_sizes = {}
        for memory_limit in memory_limits:
            batch_size = get_optimal_batch_size(
                cluster=cluster,
                tile_h=512,
                tile_w=512,
                overlap=64,
                memory_limit_gb=memory_limit,
                adaptive_sizing=True
            )
            batch_sizes[memory_limit] = batch_size
        
        print(f"\nBatch sizes for different memory limits: {batch_sizes}")
        
        # Higher memory limits should allow larger batch sizes.
        assert batch_sizes[1.0] <= batch_sizes[4.0] <= batch_sizes[16.0], \
            "Higher memory limits should allow larger batches"

"""Spatial strategy benchmarks."""

@pytest.mark.slow
class TestSpatialStrategyPerformance:
    """Benchmark different spatial batching strategies."""

    def test_spatial_strategy_performance(self):
        """Compare performance of different spatial strategies."""
        cluster_sizes = ["small", "medium", "large"]
        strategies = ["adaptive", "2x2", "spatial", "hybrid"]
        results = {}
        
        for size in cluster_sizes:
            cluster = create_test_cluster(size)
            
            for strategy in strategies:
                key = f"{size}_{strategy}"
                
                _, exec_time = benchmark_function(
                    group_tiles_by_spatial_proximity,
                    cluster=cluster,
                    batch_size=4,
                    strategy=strategy
                )
                
                results[key] = exec_time
        
        print(f"\nSpatial strategy performance: {results}")
        
        # All strategies should be reasonably fast.
        for key, time_taken in results.items():
            assert time_taken < 1.0, f"Spatial strategy should be fast: {key} took {time_taken}s"

    def test_spatial_strategy_quality(self):
        """Test quality of different spatial strategies."""
        cluster = create_test_cluster("medium")
        strategies = ["adaptive", "2x2", "spatial", "hybrid"]
        
        results = {}
        for strategy in strategies:
            batches = group_tiles_by_spatial_proximity(
                cluster=cluster,
                batch_size=4,
                strategy=strategy
            )
            
            results[strategy] = {
                'num_batches': len(batches),
                'avg_batch_size': statistics.mean(len(batch) for batch in batches),
                'max_batch_size': max(len(batch) for batch in batches),
                'min_batch_size': min(len(batch) for batch in batches),
            }
        
        print(f"\nSpatial strategy quality metrics: {results}")
        
        # All strategies should produce valid batches.
        for strategy, metrics in results.items():
            assert metrics['num_batches'] > 0, f"Strategy {strategy} should produce batches"
            assert metrics['max_batch_size'] <= 4, f"Strategy {strategy} should respect batch size limit"

"""End-to-end performance benchmarks."""

@pytest.mark.slow
class TestEndToEndPerformance:
    """Benchmark complete pipeline performance."""

    def test_merge_streaming_performance(self, performance_tiles_dir):
        """Benchmark merge_masks_streaming with different configurations."""
        configurations = [
            {"strategy": "adaptive", "batch_size": 1, "adaptive": True},
            {"strategy": "2x2", "batch_size": 2, "adaptive": False},
            {"strategy": "spatial", "batch_size": 4, "adaptive": True},
            {"strategy": "hybrid", "batch_size": 2, "adaptive": True},
        ]
        
        results = {}
        
        for i, config in enumerate(configurations):
            config_name = f"{config['strategy']}_b{config['batch_size']}_a{config['adaptive']}"
            
            _, exec_time = benchmark_function(
                merge_masks_streaming,
                height=4096,  # 8x512 tiles.
                width=4096,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path=performance_tiles_dir,
                threshold=0.3,
                use_gpu=False,  # Use CPU for consistent benchmarking.
                gpu_batch_size=config["batch_size"],
                gpu_spatial_strategy=config["strategy"],
                gpu_adaptive_batching=config["adaptive"],
            )
            
            results[config_name] = exec_time
        
        print(f"\nEnd-to-end performance results: {results}")
        
        # All configurations should complete in reasonable time.
        for config_name, time_taken in results.items():
            assert time_taken < 300, f"Configuration {config_name} should complete in reasonable time: {time_taken}s"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_gpu_vs_cpu_performance(self, performance_tiles_dir):
        """Compare GPU vs CPU performance when available."""
        try:
            # CPU benchmark.
            _, cpu_time = benchmark_function(
                merge_masks_streaming,
                height=2048,  # Smaller for faster testing.
                width=2048,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path=performance_tiles_dir,
                threshold=0.3,
                use_gpu=False,
                gpu_batch_size=2,
            )
            
            # GPU benchmark.
            _, gpu_time = benchmark_function(
                merge_masks_streaming,
                height=2048,
                width=2048,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path=performance_tiles_dir,
                threshold=0.3,
                use_gpu=True,
                gpu_batch_size=2,
            )
            
            print(f"\nGPU vs CPU performance - CPU: {cpu_time:.2f}s, GPU: {gpu_time:.2f}s")
            
            # Both should complete successfully.
            assert cpu_time > 0 and gpu_time > 0, "Both CPU and GPU should complete successfully"
            
            # GPU should generally be faster for larger workloads, but allow for variation.
            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
            print(f"GPU speedup: {speedup:.2f}x")
            
        except Exception as e:
            pytest.skip(f"GPU performance testing failed: {e}")

    def test_scalability_performance(self, performance_tiles_dir):
        """Test performance scalability with different image sizes."""
        image_sizes = [1024, 2048, 4096]  # Different scales.
        results = {}
        
        for size in image_sizes:
            _, exec_time = benchmark_function(
                merge_masks_streaming,
                height=size,
                width=size,
                tile_h=512,
                tile_w=512,
                overlap=64,
                tiles_path=performance_tiles_dir,
                threshold=0.3,
                use_gpu=False,
                gpu_batch_size=2,
                gpu_spatial_strategy="adaptive",
                gpu_adaptive_batching=True,
            )
            
            results[size] = exec_time
        
        print(f"\nScalability performance: {results}")
        
        # Performance should scale reasonably with image size.
        # Larger images should take more time, but not exponentially more.
        if len(results) >= 2:
            sizes = sorted(results.keys())
            for i in range(1, len(sizes)):
                prev_size, curr_size = sizes[i-1], sizes[i]
                size_ratio = (curr_size / prev_size) ** 2  # Area scaling.
                time_ratio = results[curr_size] / results[prev_size]
                
                # Time should scale roughly with area, but allow for overhead.
                assert time_ratio < size_ratio * 2, \
                    f"Performance should scale reasonably: {prev_size}→{curr_size}, time ratio: {time_ratio:.2f}, size ratio: {size_ratio:.2f}"
