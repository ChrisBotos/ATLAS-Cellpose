#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_memory_fixes.py.
Description:
    Test script to verify that the memory allocation fixes in batch_merge.py
    work correctly for sparse tile distributions that previously caused
    out-of-memory errors.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pytest.

Usage:
    python test_memory_fixes.py

Key Features:
    • Tests memory estimation for sparse tile distributions.
    • Validates batch sizing algorithms.
    • Simulates problematic tile patterns that caused the original error.
    • Verifies conservative spatial chunking works correctly.
"""

import traceback
import logging
import numpy as np
from typing import List, Tuple

# Configure logging for detailed output.
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def test_memory_estimation_fixes():
    """Test the enhanced memory estimation function with sparse distributions."""
    print("\n" + "="*80)
    print("TESTING MEMORY ESTIMATION FIXES")
    print("="*80)
    
    try:
        from code.nuclei_segmentation.cellpose_merge.batch_merge import estimate_memory_requirements
        
        # Test case 1: Dense tile distribution (should work normally).
        dense_tiles = [(0, 0), (0, 1), (1, 0), (1, 1)]
        tile_h, tile_w, overlap = 512, 512, 102
        
        dense_memory = estimate_memory_requirements(
            dense_tiles, tile_h, tile_w, overlap, safety_factor=1.5
        )
        print(f"✓ Dense distribution memory estimate: {dense_memory:.2f} GB")
        assert dense_memory < 10.0, f"Dense memory estimate too high: {dense_memory:.2f} GB"
        
        # Test case 2: Sparse tile distribution (the problematic case).
        # This simulates the original error: tiles spanning from (9,0) to (10,64).
        sparse_tiles = [(9, c) for c in range(0, 65, 4)]  # Every 4th column.
        sparse_tiles.extend([(10, c) for c in range(0, 65, 4)])  # Two rows.
        
        sparse_memory = estimate_memory_requirements(
            sparse_tiles, tile_h, tile_w, overlap, safety_factor=1.5
        )
        print(f"✓ Sparse distribution memory estimate: {sparse_memory:.2f} GB")
        
        # The fix should prevent unreasonably high estimates.
        assert sparse_memory < 50.0, f"Sparse memory estimate still too high: {sparse_memory:.2f} GB"
        
        # Test case 3: Extremely sparse distribution (should use individual tile estimate).
        extreme_sparse_tiles = [(0, 0), (0, 100), (50, 0), (50, 100)]
        
        extreme_memory = estimate_memory_requirements(
            extreme_sparse_tiles, tile_h, tile_w, overlap, safety_factor=1.5
        )
        print(f"✓ Extremely sparse distribution memory estimate: {extreme_memory:.2f} GB")
        assert extreme_memory < 5.0, f"Extreme sparse memory estimate too high: {extreme_memory:.2f} GB"
        
        print("✓ All memory estimation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Memory estimation test failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def test_batch_sizing_fixes():
    """Test the enhanced batch sizing function with sparse distributions."""
    print("\n" + "="*80)
    print("TESTING BATCH SIZING FIXES")
    print("="*80)
    
    try:
        from code.nuclei_segmentation.cellpose_merge.batch_merge import get_optimal_batch_size
        
        tile_h, tile_w, overlap = 512, 512, 102
        memory_limit_gb = 8.0
        
        # Test case 1: Dense cluster (should allow larger batch sizes).
        dense_cluster = [(r, c) for r in range(5) for c in range(5)]
        
        dense_batch_size = get_optimal_batch_size(
            dense_cluster, tile_h, tile_w, overlap, memory_limit_gb, adaptive_sizing=True
        )
        print(f"✓ Dense cluster optimal batch size: {dense_batch_size}")
        assert dense_batch_size >= 1, "Batch size should be at least 1"
        
        # Test case 2: Sparse cluster (should force batch size to 1).
        sparse_cluster = [(9, c) for c in range(0, 65, 4)]
        sparse_cluster.extend([(10, c) for c in range(0, 65, 4)])
        
        sparse_batch_size = get_optimal_batch_size(
            sparse_cluster, tile_h, tile_w, overlap, memory_limit_gb, adaptive_sizing=True
        )
        print(f"✓ Sparse cluster optimal batch size: {sparse_batch_size}")
        assert sparse_batch_size == 1, f"Sparse cluster should have batch size 1, got {sparse_batch_size}"
        
        # Test case 3: Very large sparse cluster (should definitely be 1).
        large_sparse_cluster = [(r, c) for r in range(0, 20, 5) for c in range(0, 100, 10)]
        
        large_sparse_batch_size = get_optimal_batch_size(
            large_sparse_cluster, tile_h, tile_w, overlap, memory_limit_gb, adaptive_sizing=True
        )
        print(f"✓ Large sparse cluster optimal batch size: {large_sparse_batch_size}")
        assert large_sparse_batch_size == 1, f"Large sparse cluster should have batch size 1, got {large_sparse_batch_size}"
        
        print("✓ All batch sizing tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Batch sizing test failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def test_conservative_spatial_chunking():
    """Test the conservative spatial chunking function."""
    print("\n" + "="*80)
    print("TESTING CONSERVATIVE SPATIAL CHUNKING")
    print("="*80)
    
    try:
        from code.nuclei_segmentation.cellpose_merge.batch_merge import _create_conservative_spatial_chunks
        
        # Test case 1: Sparse distribution should create small, compact chunks.
        sparse_tiles = [(0, 0), (0, 50), (10, 0), (10, 50)]
        max_batch_size = 4
        
        chunks = _create_conservative_spatial_chunks(sparse_tiles, max_batch_size)
        print(f"✓ Created {len(chunks)} chunks from {len(sparse_tiles)} sparse tiles")
        
        # Verify that chunks are reasonable.
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {len(chunk)} tiles - {chunk}")
            assert len(chunk) <= max_batch_size, f"Chunk {i+1} exceeds max batch size"
            
            # Check spatial compactness.
            if len(chunk) > 1:
                min_r = min(r for r, _ in chunk)
                max_r = max(r for r, _ in chunk)
                min_c = min(c for _, c in chunk)
                max_c = max(c for _, c in chunk)
                
                row_span = max_r - min_r + 1
                col_span = max_c - min_c + 1
                
                # Chunks should be spatially compact.
                assert row_span <= 8 and col_span <= 8, f"Chunk {i+1} not spatially compact: {row_span}x{col_span}"
        
        print("✓ Conservative spatial chunking test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Conservative spatial chunking test failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def test_problematic_tile_pattern():
    """Test the exact tile pattern that caused the original memory error."""
    print("\n" + "="*80)
    print("TESTING PROBLEMATIC TILE PATTERN")
    print("="*80)
    
    try:
        from code.nuclei_segmentation.cellpose_merge.batch_merge import (
            estimate_memory_requirements, get_optimal_batch_size, group_tiles_by_spatial_proximity
        )
        
        # Recreate the exact problematic pattern from the error log.
        # tile_range=(9,0) to (10,64), which means 66 tiles in a horizontal line.
        problematic_tiles = []
        for r in [9, 10]:  # Two rows.
            for c in range(65):  # Columns 0 to 64.
                problematic_tiles.append((r, c))
        
        print(f"Created problematic pattern: {len(problematic_tiles)} tiles")
        print(f"Tile range: ({min(r for r, _ in problematic_tiles)},{min(c for _, c in problematic_tiles)}) "
              f"to ({max(r for r, _ in problematic_tiles)},{max(c for _, c in problematic_tiles)})")
        
        tile_h, tile_w, overlap = 512, 512, 102  # From the original config.
        memory_limit_gb = 4.0  # Conservative limit.
        
        # Test memory estimation.
        memory_estimate = estimate_memory_requirements(
            problematic_tiles, tile_h, tile_w, overlap, safety_factor=2.0
        )
        print(f"✓ Problematic pattern memory estimate: {memory_estimate:.2f} GB")
        
        # Should not be unreasonably high.
        assert memory_estimate < 100.0, f"Memory estimate still too high: {memory_estimate:.2f} GB"
        
        # Test batch sizing.
        batch_size = get_optimal_batch_size(
            problematic_tiles, tile_h, tile_w, overlap, memory_limit_gb, adaptive_sizing=True
        )
        print(f"✓ Problematic pattern optimal batch size: {batch_size}")
        assert batch_size == 1, f"Should force batch size to 1 for problematic pattern, got {batch_size}"
        
        # Test spatial grouping.
        batches = group_tiles_by_spatial_proximity(
            problematic_tiles, batch_size, strategy="spatial"
        )
        print(f"✓ Created {len(batches)} batches from problematic pattern")
        
        # Verify all batches are safe.
        for i, batch in enumerate(batches[:5]):  # Check first 5 batches.
            batch_memory = estimate_memory_requirements(
                batch, tile_h, tile_w, overlap, safety_factor=2.0
            )
            print(f"  Batch {i+1}: {len(batch)} tiles, memory: {batch_memory:.2f} GB")
            assert batch_memory < memory_limit_gb * 2, f"Batch {i+1} still has high memory requirement"
        
        print("✓ Problematic tile pattern test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Problematic tile pattern test failed: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


def main():
    """Run all memory fix tests."""
    print("MEMORY ALLOCATION FIXES VERIFICATION")
    print("="*80)
    print("Testing fixes for sparse tile distribution memory issues...")
    
    tests = [
        test_memory_estimation_fixes,
        test_batch_sizing_fixes,
        test_conservative_spatial_chunking,
        test_problematic_tile_pattern,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*80)
    print(f"MEMORY FIXES TEST RESULTS: {passed}/{total} tests passed")
    print("="*80)
    
    if passed == total:
        print("✓ All memory allocation fixes are working correctly!")
        print("✓ The system should now handle sparse tile distributions safely.")
        return True
    else:
        print("✗ Some memory fixes are not working correctly.")
        print("✗ Please review the failed tests and fix the issues.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
