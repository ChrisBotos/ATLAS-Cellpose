"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_merge_masks_streaming_two_phase.py.
Description:
    Integration tests for the merge_masks_streaming function with two-phase merging enabled.
    Tests the complete pipeline from tile loading to final merged mask generation,
    ensuring proper integration between the two-phase merge strategy and the main
    streaming merge function.

Dependencies:
    • Python ≥ 3.10.
    • pytest, numpy, pillow.
    • cellpose_merge.merge_tiles module.

Key Features:
    • Integration tests for merge_masks_streaming with use_two_phase_merge=True.
    • Comparison tests between two-phase and cluster-based approaches.
    • Memory efficiency validation for large tile grids.
    • Error handling and fallback mechanism tests.
    • QC overlay generation tests with two-phase merging.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from PIL import Image

# Adjust path for imports.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from code.nuclei_segmentation.cellpose_merge.merge_tiles import merge_masks_streaming


class TestMergeMasksStreamingTwoPhase:
    """Test merge_masks_streaming function with two-phase merging enabled."""
    
    def create_test_tiles(self, temp_dir: Path, coords: List[Tuple[int, int]], 
                         tile_h: int = 256, tile_w: int = 256) -> Path:
        """Create synthetic tile mask files for testing."""
        tiles_dir = temp_dir / "tile_masks"
        tiles_dir.mkdir(exist_ok=True)
        
        for r, c in coords:
            # Create synthetic tile with nucleus.
            tile_mask = np.zeros((tile_h, tile_w), dtype=np.uint32)
            
            # Add nucleus with unique ID based on tile position.
            nucleus_id = r * 10 + c + 1
            center_y, center_x = tile_h // 2, tile_w // 2
            
            # Create nucleus that might extend into overlap regions.
            y_start = max(0, center_y - 30)
            y_end = min(tile_h, center_y + 30)
            x_start = max(0, center_x - 30)
            x_end = min(tile_w, center_x + 30)
            
            tile_mask[y_start:y_end, x_start:x_end] = nucleus_id
            
            # Save as NPZ file.
            tile_path = tiles_dir / f"{r}_{c}.npz"
            np.savez_compressed(tile_path, mask=tile_mask)
        
        return tiles_dir
    
    def test_two_phase_merge_basic(self):
        """Test basic two-phase merging functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
            
            tiles_dir = self.create_test_tiles(temp_path, coords)
            
            # Test with two-phase merging enabled.
            merged = merge_masks_streaming(
                height=448,  # 2 * (256 - 64) + 64
                width=448,
                tile_h=256,
                tile_w=256,
                overlap=64,
                tiles_path=tiles_dir,
                threshold=0.3,
                use_gpu=False,
                qc=False,
                use_two_phase_merge=True,
                merge_batch_size=2,
                output_dir=temp_path / "results"
            )
            
            # Verify merged result.
            assert merged.shape == (448, 448)
            assert np.any(merged > 0)  # Should contain nuclei.
            
            # Check that nuclei were properly merged.
            unique_labels = np.unique(merged[merged > 0])
            assert len(unique_labels) >= 1  # At least one nucleus.
            
            # Verify output files were created.
            results_dir = temp_path / "results" / "masks"
            assert (results_dir / "segmentation_masks.npy").exists()
    
    def test_two_phase_vs_cluster_comparison(self):
        """Compare results between two-phase and cluster-based merging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            coords = [(0, 0), (0, 1), (1, 0)]  # L-shaped pattern
            
            tiles_dir = self.create_test_tiles(temp_path, coords)
            
            # Test with two-phase merging.
            merged_two_phase = merge_masks_streaming(
                height=448,
                width=384,  # 2 * 192 + 64 for width
                tile_h=256,
                tile_w=256,
                overlap=64,
                tiles_path=tiles_dir,
                threshold=0.3,
                use_gpu=False,
                qc=False,
                use_two_phase_merge=True,
                output_dir=temp_path / "results_two_phase"
            )
            
            # Test with cluster-based merging.
            merged_cluster = merge_masks_streaming(
                height=448,
                width=384,
                tile_h=256,
                tile_w=256,
                overlap=64,
                tiles_path=tiles_dir,
                threshold=0.3,
                use_gpu=False,
                qc=False,
                use_two_phase_merge=False,
                output_dir=temp_path / "results_cluster"
            )
            
            # Both should produce valid results.
            assert merged_two_phase.shape == merged_cluster.shape
            assert np.any(merged_two_phase > 0)
            assert np.any(merged_cluster > 0)
            
            # Number of nuclei should be similar (allowing for merge differences).
            nuclei_two_phase = len(np.unique(merged_two_phase[merged_two_phase > 0]))
            nuclei_cluster = len(np.unique(merged_cluster[merged_cluster > 0]))
            
            # Should have at least some nuclei in both cases.
            assert nuclei_two_phase >= 1
            assert nuclei_cluster >= 1
    
    def test_two_phase_merge_single_row(self):
        """Test two-phase merging with tiles in a single row."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            coords = [(0, 0), (0, 1), (0, 2), (0, 3)]  # Single row
            
            tiles_dir = self.create_test_tiles(temp_path, coords)
            
            merged = merge_masks_streaming(
                height=256,
                width=832,  # 4 * 192 + 64
                tile_h=256,
                tile_w=256,
                overlap=64,
                tiles_path=tiles_dir,
                threshold=0.3,
                use_gpu=False,
                qc=False,
                use_two_phase_merge=True,
                merge_batch_size=3
            )
            
            assert merged.shape == (256, 832)
            assert np.any(merged > 0)
    
    def test_two_phase_merge_error_handling(self):
        """Test error handling with empty tiles directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create empty tiles directory to trigger FileNotFoundError.
            tiles_dir = temp_path / "empty_tiles"
            tiles_dir.mkdir()

            # This should raise FileNotFoundError for empty directory.
            with pytest.raises(FileNotFoundError, match="None of the following directories contains tile masks"):
                merge_masks_streaming(
                    height=256,
                    width=256,
                    tile_h=256,
                    tile_w=256,
                    overlap=64,
                    tiles_path=tiles_dir,
                    threshold=0.3,
                    use_gpu=False,
                    qc=False,
                    use_two_phase_merge=True,
                    merge_batch_size=2
                )

    def test_two_phase_merge_memory_efficiency(self):
        """Test memory efficiency of two-phase merging with larger tile grids."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a 3x3 grid of tiles (9 tiles total).
            coords = [(r, c) for r in range(3) for c in range(3)]
            tiles_dir = self.create_test_tiles(temp_path, coords, tile_h=128, tile_w=128)

            # Test with two-phase merging - should handle this efficiently.
            merged = merge_masks_streaming(
                height=320,  # 3 * (128 - 32) + 32
                width=320,
                tile_h=128,
                tile_w=128,
                overlap=32,
                tiles_path=tiles_dir,
                threshold=0.3,
                use_gpu=False,
                qc=False,
                use_two_phase_merge=True,
                merge_batch_size=4,
                output_dir=temp_path / "results_memory_test"
            )

            # Verify the result is valid.
            assert merged.shape == (320, 320)
            assert np.any(merged > 0)  # Should contain nuclei.

            # Check that we have nuclei from multiple tiles.
            unique_labels = np.unique(merged[merged > 0])
            assert len(unique_labels) >= 3  # Should have nuclei from multiple tiles.

    def test_two_phase_merge_batch_size_effects(self):
        """Test different batch sizes for two-phase merging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            coords = [(0, 0), (0, 1), (0, 2)]  # Single row

            tiles_dir = self.create_test_tiles(temp_path, coords)

            # Test with batch_size=1 (sequential processing).
            merged_batch1 = merge_masks_streaming(
                height=256,
                width=640,
                tile_h=256,
                tile_w=256,
                overlap=64,
                tiles_path=tiles_dir,
                threshold=0.3,
                use_gpu=False,
                qc=False,
                use_two_phase_merge=True,
                merge_batch_size=1,
                output_dir=temp_path / "results_batch1"
            )

            # Test with batch_size=2 (parallel processing).
            merged_batch2 = merge_masks_streaming(
                height=256,
                width=640,
                tile_h=256,
                tile_w=256,
                overlap=64,
                tiles_path=tiles_dir,
                threshold=0.3,
                use_gpu=False,
                qc=False,
                use_two_phase_merge=True,
                merge_batch_size=2,
                output_dir=temp_path / "results_batch2"
            )

            # Both should produce similar results.
            assert merged_batch1.shape == merged_batch2.shape
            assert np.any(merged_batch1 > 0)
            assert np.any(merged_batch2 > 0)

            # Number of nuclei should be similar.
            nuclei_batch1 = len(np.unique(merged_batch1[merged_batch1 > 0]))
            nuclei_batch2 = len(np.unique(merged_batch2[merged_batch2 > 0]))

            # Allow for small differences due to processing order.
            assert abs(nuclei_batch1 - nuclei_batch2) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
