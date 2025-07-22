"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: create_tile_overlays.py.
Description:
    Simple script to create memory-efficient tile overlays for kidney I/R injury
    tissue analysis. This script creates before/after merging visualizations with
    unique colors for each tile, handling thousands of tiles without overwhelming RAM.

Dependencies:
    • Python >= 3.10.
    • numpy, pillow for image processing.
    • pathlib for file operations.

Usage:
    python create_tile_overlays.py --results-dir "results/run_name" --image-path "data/image.tif"

Arguments:
    --results-dir   Path to results directory containing masks subdirectory.
    --image-path    Path to the full tissue image file.
    --output-dir    Output directory for overlays (default: overlays).
    --crop-size     Size of central crop for visualization (default: 1300).
    --batch-size    Batch size for memory-efficient processing (default: 100).

Inputs:
    • Results directory with tile_masks_npz and merged_tile_masks_npz subdirectories.
    • Full tissue image for background visualization.

Outputs:
    • before_merging_tiles.tif: Individual tile masks with unique colors.
    • after_merging_tiles.tif: Merged tile masks with unique colors.

Key Features:
    • Before merging: Tile-based colors (each tile gets unique deterministic color).
    • After merging: Nucleus-based colors (each nucleus gets unique random color).
    • Alpha transparency blending with tissue background for scientific visualization.
    • Memory-efficient batch processing for large tile datasets.

Notes:
    • The script is optimized for kidney I/R injury tissue analysis workflows.
    • Color schemes are designed for optimal visibility on tissue backgrounds.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Import the overlay functions.
sys.path.append(str(Path(__file__).parent / "code" / "nuclei_segmentation" / "cellpose_merge"))

from qc import (
    create_before_after_overlays,
    create_tile_overlay_from_directory,
    _load_rgb_image
)

"""MAIN FUNCTIONS"""

def create_overlays_from_results(
    results_dir: Path,
    image_path: Path,
    output_dir: Path,
    crop_size: int = 1300,
    batch_size: int = 100,
    alpha: float = 0.6
) -> None:
    """
    Create both before and after merging overlays from a results directory.
    
    Parameters
    ----------
    results_dir : Path
        Path to results directory containing masks subdirectory.
    image_path : Path
        Path to the full tissue image file.
    output_dir : Path
        Directory to save overlay outputs.
    crop_size : int, default 1300
        Size of central crop for visualization.
    batch_size : int, default 100
        Batch size for memory-efficient processing.
    alpha : float, default 0.6
        Transparency level for overlays.
    """
    
    logging.info(f"Creating tile overlays from: {results_dir}")
    logging.info(f"Using tissue image: {image_path}")
    
    # Validate inputs.
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    masks_dir = results_dir / "masks"
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    # Load tissue image.
    logging.info("Loading tissue image...")
    full_image = _load_rgb_image(image_path)
    logging.info(f"Loaded tissue image: {full_image.shape}")
    
    # Create output directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create overlays.
    start_time = time.time()
    
    try:
        before_overlay, after_overlay = create_before_after_overlays(
            results_dir=results_dir,
            full_image=full_image,
            crop_size=crop_size,
            batch_size=batch_size,
            alpha=alpha,
            output_dir=output_dir
        )
        
        processing_time = time.time() - start_time
        
        logging.info(f"✓ Created before overlay: {before_overlay.shape}")
        logging.info(f"✓ Created after overlay: {after_overlay.shape}")
        logging.info(f"✓ Processing completed in {processing_time:.2f} seconds")
        logging.info(f"✓ Overlays saved to: {output_dir}")
        
    except Exception as e:
        logging.error(f"Failed to create overlays: {e}")
        raise


def create_single_overlay(
    tiles_dir: Path,
    image_path: Path,
    output_path: Path,
    overlay_type: str = "before",
    crop_size: int = 1300,
    batch_size: int = 100,
    alpha: float = 0.6
) -> None:
    """
    Create a single overlay from a tile directory.
    
    Parameters
    ----------
    tiles_dir : Path
        Directory containing tile mask files (.npz format).
    image_path : Path
        Path to the full tissue image file.
    output_path : Path
        Path to save the overlay image.
    overlay_type : str, default "before"
        Type of overlay ("before" or "after").
    crop_size : int, default 1300
        Size of central crop for visualization.
    batch_size : int, default 100
        Batch size for memory-efficient processing.
    alpha : float, default 0.6
        Transparency level for overlay.
    """
    
    logging.info(f"Creating {overlay_type} overlay from: {tiles_dir}")
    
    # Validate inputs.
    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load tissue image.
    full_image = _load_rgb_image(image_path)
    logging.info(f"Loaded tissue image: {full_image.shape}")
    
    # Create output directory.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create overlay.
    start_time = time.time()
    
    overlay = create_tile_overlay_from_directory(
        tiles_dir=tiles_dir,
        full_image=full_image,
        crop_size=crop_size,
        batch_size=batch_size,
        alpha=alpha,
        output_path=output_path,
        overlay_type=overlay_type
    )
    
    processing_time = time.time() - start_time
    
    logging.info(f"✓ Created {overlay_type} overlay: {overlay.shape}")
    logging.info(f"✓ Processing completed in {processing_time:.2f} seconds")
    logging.info(f"✓ Overlay saved to: {output_path}")


"""COMMAND LINE INTERFACE"""

def main() -> int:
    """
    Main entry point for tile overlay creation.
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    
    parser = argparse.ArgumentParser(
        description="Create memory-efficient tile overlays for kidney tissue analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Path to results directory containing masks subdirectory"
    )
    
    parser.add_argument(
        "--image-path",
        required=True,
        type=Path,
        help="Path to the full tissue image file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="overlays",
        type=Path,
        help="Output directory for overlay images"
    )
    
    parser.add_argument(
        "--crop-size",
        type=int,
        default=1300,
        help="Size of central crop for visualization"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for memory-efficient processing"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Transparency level for overlays [0-1]"
    )
    
    parser.add_argument(
        "--single-overlay",
        choices=["before", "after"],
        help="Create only a single overlay type instead of both"
    )
    
    args = parser.parse_args()
    
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    
    try:
        if args.single_overlay:
            # Create single overlay.
            tiles_subdir = "tile_masks_npz" if args.single_overlay == "before" else "merged_tile_masks_npz"
            tiles_dir = args.results_dir / "masks" / tiles_subdir
            output_path = args.output_dir / f"{args.single_overlay}_overlay.tif"
            
            create_single_overlay(
                tiles_dir=tiles_dir,
                image_path=args.image_path,
                output_path=output_path,
                overlay_type=args.single_overlay,
                crop_size=args.crop_size,
                batch_size=args.batch_size,
                alpha=args.alpha
            )
        else:
            # Create both overlays.
            create_overlays_from_results(
                results_dir=args.results_dir,
                image_path=args.image_path,
                output_dir=args.output_dir,
                crop_size=args.crop_size,
                batch_size=args.batch_size,
                alpha=args.alpha
            )
        
        logging.info("Tile overlay creation completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"Tile overlay creation failed: {e}")
        import traceback
        logging.debug(f"Error traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())
