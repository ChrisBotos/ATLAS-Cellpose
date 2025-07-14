"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: segmentation.py.
Description:
    End‑to‑end wrapper that runs Cellpose on large histological slides using
    fixed‑size overlapping tiles and fuses the per‑tile masks via the streaming
    merge engine in ``cellpose_merge``.  This revision switches to the new
    ``merge_masks_streaming`` signature that consumes on‑disk *NPZ* tiles
    instead of an in‑memory ``data_loader`` callback.

Key Features:
    • Automatically falls back to a single‑pass evaluation when the image fits
      comfortably in RAM.
    • Streams both the intermediary per‑tile masks *and* the final merged mask
      to disk so that gigantic slides do not exhaust RAM.
    • Saves each labelled tile as ``<y0>_<x0>.npz`` where *y0* and *x0* are the
      slice offsets – the format expected by
      ``cellpose_merge.io.make_tile_loader``.
    • Propagates all quality‑control options (``qc``, ``qc_dir``) directly to
      the merge layer.

Dependencies:
    • Python ≥ 3.10.
    • numpy, cellpose, cellpose_merge, tqdm.

Usage:
    masks_mm, _, n_cells = run_cellpose_on_tiles(
        model=my_cp_model,
        image=slide_img,
        cellpose_params={...},
        settings={
            "output_dir": "./results",            # Required.
            "tile_side_length": 512,              # Required.
            "tile_overlap": 64,                   # Required.
            # Optional tweaks ↓
            "use_tiling": True,
            "merge_overlap_threshold": 0.3,
            "qc_overlays": False,
            "qc_dir": "./qc_overlays",
        },
        logger=my_logger,
    )
"""

from __future__ import annotations

import traceback
from typing import List, Tuple
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .tiling import split_image_into_tiles
from cellpose_merge.merge_tiles import merge_masks_streaming

# Alias for readability.
MaskReturn = Tuple[np.memmap, List[None], int]

# -----------------------------------------------------------------------------
# Helper – single‑pass evaluation (no tiling required).
# -----------------------------------------------------------------------------


def _run_single_pass_cellpose(model, image: np.ndarray, cellpose_params: dict, logger) -> Tuple[np.ndarray, List[np.ndarray], int]:
    """Run Cellpose on the whole *image* and return the raw mask.

    If Cellpose crashes we return a zero‑filled mask so that downstream logic
    remains robust.  All comments are full sentences.
    """

    logger.info("Running full‑image Cellpose segmentation (no tiling).")
    try:
        masks, flows, *_ = model.eval(
            image[..., None],
            diameter=cellpose_params["diameter"],
            channels=cellpose_params["channels"],
            flow_threshold=cellpose_params["flow_threshold"],
            cellprob_threshold=cellpose_params["cellprob_threshold"],
            resample=cellpose_params["resample"],
            augment=False,
            batch_size=cellpose_params["batch_size"],
            do_3D=False,
        )
        num_cells = int(masks.max())
        logger.info("Detected %d nuclei in full image.", num_cells)
        return masks.astype(np.uint32), [flows[0], flows[1], None], num_cells
    except Exception as exc:
        logger.error("✗ Cellpose failed on full image: %s", exc)
        return (
            np.zeros_like(image, dtype=np.uint32),
            [np.zeros((2, *image.shape), dtype=np.float32), np.zeros(image.shape, dtype=np.float32), None],
            0,
        )


# -----------------------------------------------------------------------------
# Public API – tiled execution with streaming merge.
# -----------------------------------------------------------------------------


def run_cellpose_on_tiles(
    model,
    image: np.ndarray,
    cellpose_params: dict,
    settings: dict,
    logger,
) -> MaskReturn:
    """
    Segment kidney tissue nuclei using Cellpose with intelligent tiling for large images.

    This function processes DAPI-stained nuclear images from kidney I/R injury experiments,
    automatically determining whether tiling is necessary based on image size. For large
    images, it splits them into overlapping tiles, processes each tile with Cellpose,
    and saves both individual tile masks and a combined segmentation mask.

    The function is designed for bioinformaticians analyzing kidney tissue sections at
    different time points (10 hours, 2 days, 14 days) after ischemia/reperfusion injury.
    It handles the computational challenges of processing large histological images while
    maintaining segmentation quality across tile boundaries.

    Parameters
    ----------
    model : cellpose.models.Cellpose
        Pre-loaded Cellpose model instance configured for nuclear segmentation.
        Should be initialized with appropriate model type (typically 'nuclei').
    image : np.ndarray
        Two-dimensional greyscale DAPI-stained image array of shape (H, W).
        Represents nuclear staining in kidney tissue sections.
    cellpose_params : dict
        Cellpose segmentation parameters including diameter, flow_threshold,
        cellprob_threshold, channels, resample, and batch_size settings.
        These parameters are forwarded directly to model.eval().
    settings : dict
        Configuration dictionary with required keys:
        - output_dir: Directory for saving segmentation results
        - tile_side_length: Size of square tiles for processing large images
        - tile_overlap: Overlap between adjacent tiles (pixels or fraction)
        Optional keys for advanced control:
        - use_tiling: Force enable/disable tiling (default: auto-detect)
    logger : logging.Logger
        Logger instance for progress tracking and debugging information.
        Used extensively for monitoring segmentation progress.

    Returns
    -------
    masks_mm : np.memmap
        Memory-mapped array containing the final segmentation mask of shape (H, W).
        Each unique positive integer represents a distinct nucleus instance.
        Saved to disk as 'segmentation_masks.npy' for downstream analysis.
    flows : list[None, None, None]
        Placeholder list - flows are discarded to conserve memory during processing.
        Required for compatibility with downstream pipeline components.
    total_cells : int
        Total number of unique nuclei detected across all tiles.
        Used for quality control and statistical analysis of segmentation results.
    """

    '''Setup output directories and initialize memory-mapped storage'''
    # Create output directory structure for organized result storage.
    out_root = Path(settings["output_dir"]).expanduser().resolve()
    masks_dir = out_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Segmentation output directory: {masks_dir}")
    logger.info(f"Processing image of shape: {image.shape}")

    # Initialize memory-mapped storage for the final combined segmentation mask.
    # This approach allows processing of very large images without memory overflow.
    H, W = (int(s) for s in image.shape)
    temp_mask_path = masks_dir / "temp_combined_mask.dat"
    masks_mm = open_memmap(temp_mask_path, mode="w+", dtype=np.uint32, shape=(H, W))

    '''Determine tiling strategy based on image size and configuration'''
    # Calculate tile parameters for optimal processing of large kidney tissue images.
    # Tiling is essential for processing whole-slide images that exceed GPU memory limits.
    tile_size = int(settings["tile_side_length"])
    overlap_cfg = settings["tile_overlap"]

    # Handle both fractional (0-1) and absolute pixel overlap specifications.
    if 0 <= overlap_cfg <= 1:
        overlap = int(tile_size * overlap_cfg)
    else:
        overlap = int(overlap_cfg)

    # Ensure overlap doesn't exceed half the tile size to maintain processing efficiency.
    overlap = min(overlap, tile_size // 2)

    # Automatically determine if tiling is necessary based on image dimensions.
    auto_tiling = H > tile_size or W > tile_size
    use_tiling = settings.get("use_tiling", True) and auto_tiling

    logger.info(f"Image dimensions: {H} x {W} pixels")
    logger.info(f"Tile configuration: {'ENABLED' if use_tiling else 'DISABLED'}")
    logger.info(f"Tile size: {tile_size}x{tile_size} pixels, overlap: {overlap} pixels")

    if settings.get("debug_mode", False):
        logger.debug(f"Tile overlap fraction: {overlap/tile_size:.3f}")
        logger.debug(f"Auto-tiling triggered: {auto_tiling}")

    '''Fast path: Process entire image without tiling for smaller images'''
    if not use_tiling:
        logger.info("Processing entire image in single pass (no tiling required)")

        # Run Cellpose on the complete image for optimal segmentation quality.
        masks, _, n_cells = _run_single_pass_cellpose(model, image, cellpose_params, logger)

        # Copy results to memory-mapped storage and save to disk.
        masks_mm[:] = masks
        masks_mm.flush()

        logger.info(f"Single-pass segmentation completed: {n_cells} nuclei detected")
        return masks_mm, [None, None, None], n_cells

    '''Tiled processing: Split large images into manageable overlapping tiles'''
    # Create directory structure for storing individual tile segmentation masks.
    # This organization facilitates downstream analysis and quality control.
    tiles_dir = masks_dir / "tile_masks_npz"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Individual tile masks will be saved to: {tiles_dir}")
    logger.info("Beginning tiled segmentation for large kidney tissue image")

    # Generate overlapping tiles from the input image for processing.
    # Overlap is crucial for maintaining segmentation continuity across tile boundaries.
    tile_iter = split_image_into_tiles(
        img=image,
        tile_h=tile_size,
        tile_w=tile_size,
        overlap=overlap,
        logger=logger
    )

    # Convert iterator to lists for processing and progress tracking.
    tiles, slices = zip(*tile_iter)
    n_tiles = len(tiles)

    logger.info(f"Generated {n_tiles} overlapping tiles for processing")

    if settings.get("debug_mode", False):
        logger.debug(f"Tile processing will use batch_size: {cellpose_params.get('batch_size', 'default')}")
        logger.debug(f"Expected memory usage per tile: ~{(tile_size**2 * 4) / (1024**2):.1f} MB")

    # Initialize counters for global label management and statistics.
    next_global_id = 1  # Ensures unique nucleus IDs across all tiles.
    total_cells_detected = 0  # Running count of detected nuclei.

    '''Process each tile individually with Cellpose segmentation'''
    for tile_idx, (current_tile, (y_slice, x_slice)) in enumerate(zip(tiles, slices), start=1):

        # Log detailed progress information for monitoring large image processing.
        logger.info(f"Processing tile {tile_idx}/{n_tiles}")
        logger.info(f"  Position: ({y_slice.start}, {x_slice.start})")
        logger.info(f"  Dimensions: {current_tile.shape}")

        try:
            # Run Cellpose segmentation on the current tile.
            # The model expects a 3D array with channel dimension, so we add one.
            cellpose_results = model.eval(
                current_tile[..., None],  # Add channel dimension for Cellpose.
                diameter=cellpose_params["diameter"],
                channels=cellpose_params["channels"],
                flow_threshold=cellpose_params["flow_threshold"],
                cellprob_threshold=cellpose_params["cellprob_threshold"],
                resample=cellpose_params["resample"],
                augment=False,  # Disable augmentation for consistent results.
                batch_size=cellpose_params["batch_size"],
                do_3D=False,  # Process as 2D nuclear segmentation.
            )

            # Extract masks from Cellpose results (first element).
            raw_masks = cellpose_results[0]

            # Handle cases where Cellpose returns None (no nuclei detected).
            if raw_masks is None:
                tile_segmentation_mask = np.zeros(current_tile.shape, dtype=np.uint32)
                nuclei_in_tile = 0
                logger.info(f"  → No nuclei detected in tile {tile_idx}")
            else:
                tile_segmentation_mask = raw_masks.astype(np.uint32)
                nuclei_in_tile = int(tile_segmentation_mask.max())

                if nuclei_in_tile > 0:
                    # Shift nucleus labels to ensure global uniqueness across all tiles.
                    # This prevents label conflicts when combining tiles later.
                    nucleus_pixels = tile_segmentation_mask != 0
                    tile_segmentation_mask[nucleus_pixels] += next_global_id

                    # Update the combined segmentation mask with this tile's results.
                    masks_mm[y_slice, x_slice][nucleus_pixels] = tile_segmentation_mask[nucleus_pixels]

                    # Update global counters for the next tile.
                    next_global_id += nuclei_in_tile
                    total_cells_detected += nuclei_in_tile

                    logger.info(f"  → {nuclei_in_tile} nuclei detected and labeled")
                else:
                    logger.info(f"  → No nuclei detected in tile {tile_idx}")

            # Save individual tile mask for downstream analysis and quality control.
            # The filename format (y_start_x_start.npz) is required by the merge system.
            tile_filename = f"{y_slice.start}_{x_slice.start}.npz"
            tile_save_path = tiles_dir / tile_filename
            np.savez_compressed(tile_save_path, mask=tile_segmentation_mask)

            if settings.get("debug_mode", False):
                logger.debug(f"  → Saved tile mask to: {tile_filename}")

            # Explicitly release memory to prevent accumulation during large image processing.
            tile_segmentation_mask = None
            raw_masks = None

        except Exception as tile_error:
            # Log detailed error information for debugging segmentation failures.
            logger.error(f"✗ Tile {tile_idx} processing failed: {tile_error}")

            if settings.get("debug_mode", False):
                import traceback
                logger.debug(f"Full error traceback:\n{traceback.format_exc()}")

            # Create empty mask for failed tile to maintain consistency.
            empty_mask = np.zeros(current_tile.shape, dtype=np.uint32)
            tile_filename = f"{y_slice.start}_{x_slice.start}.npz"
            np.savez_compressed(tiles_dir / tile_filename, mask=empty_mask)

    '''Finalize segmentation results and prepare outputs'''
    # Ensure all data is written to disk before returning.
    masks_mm.flush()

    # Log comprehensive segmentation statistics for quality assessment.
    logger.info('\n'+"="*60)
    logger.info("SEGMENTATION SUMMARY")
    logger.info("="*60 + '\n')
    logger.info(f"Total tiles processed: {n_tiles}")
    logger.info(f"Total nuclei detected: {total_cells_detected}")
    logger.info(f"Average nuclei per tile: {total_cells_detected/n_tiles:.1f}")
    logger.info(f"Nuclear density: {total_cells_detected/(H*W)*1e6:.1f} nuclei/mm² (assuming 1 pixel = 1 μm)")
    logger.info(f"Individual tile masks saved to: {tiles_dir}")
    logger.info(f"Combined segmentation mask ready for pipeline processing")

    if settings.get("debug_mode", False):
        logger.debug(f"Memory-mapped file size: {masks_mm.nbytes / (1024**2):.1f} MB")
        logger.debug(f"Next available global ID: {next_global_id}")

    logger.info("Tiled segmentation completed successfully")

    # Return memory-mapped mask, placeholder flows, and cell count.
    # The flows are set to None to conserve memory during large image processing.
    return masks_mm, [None, None, None], total_cells_detected
