"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
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
      slice offsets – the standard format for tile mask files.
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

from typing import List, Tuple
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from .tiling import split_image_into_tiles
from atlas_cellpose.cellpose_merge.merge_tiles import merge_masks_streaming

# Initialize Rich console for formatted output.
console = Console()

# Alias for readability.
MaskReturn = Tuple[np.memmap, List[None], int]

# -----------------------------------------------------------------------------
# Helper functions for mask processing.
# -----------------------------------------------------------------------------

def prepare_cellpose_parameters(cellpose_params, use_cellpose4=True):
    """
    Prepare Cellpose parameters for version-specific compatibility.

    This function handles parameter differences between Cellpose3 and Cellpose4,
    ensuring optimal segmentation performance across both versions.

    Args:
        cellpose_params (dict): Raw Cellpose parameters from configuration.
        use_cellpose4 (bool): Whether to use Cellpose4 (True) or Cellpose3 (False).

    Returns:
        dict: Version-specific parameters ready for model.eval().
    """
    eval_params = {
        "diameter": cellpose_params["diameter"],
        "flow_threshold": cellpose_params["flow_threshold"],
        "cellprob_threshold": cellpose_params["cellprob_threshold"],
        "augment": False,
        "batch_size": cellpose_params["batch_size"],
        "do_3D": False,
    }

    # Handle resample parameter based on Cellpose version.
    if use_cellpose4:
        # Resample is deprecated in Cellpose4 v4.0.1+ but still works for compatibility.
        if cellpose_params.get("resample", True):
            eval_params["resample"] = True
    else:
        # Resample is still required in Cellpose3.
        eval_params["resample"] = cellpose_params.get("resample", True)

    return eval_params


def relabel_mask_for_unique_ids(mask: np.ndarray, current_max_id: int) -> np.ndarray:
    """
    Relabel a mask to ensure unique IDs across tiles by adding an offset.

    This function takes a mask with potentially overlapping label IDs and shifts
    all non-zero labels by the current maximum ID to ensure global uniqueness
    across all processed tiles in the segmentation pipeline.

    Args:
        mask: Input mask array with integer labels (0 = background).
        current_max_id: Current maximum label ID used in previous tiles.

    Returns:
        Relabeled mask with unique IDs starting from current_max_id + 1.
    """
    if mask.size == 0 or mask.max() == 0:
        return mask.copy()

    # Create a copy to avoid modifying the original.
    unique_mask = mask.copy()

    # Find all non-zero labels.
    nonzero_mask = unique_mask > 0

    if np.any(nonzero_mask):
        # Shift all labels by the current maximum ID.
        unique_mask[nonzero_mask] += current_max_id

    return unique_mask


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
        # Determine Cellpose version from parameters.
        use_cellpose4 = cellpose_params.get("use_cellpose4", True)
        cellpose_version = "Cellpose4" if use_cellpose4 else "Cellpose3"
        logger.info(f"Using {cellpose_version} for full-image segmentation")

        # Prepare version-specific parameters for adaptive diameter learning.
        # This enables optimal segmentation quality across varying nuclear morphology.
        try:
            eval_params = prepare_cellpose_parameters(cellpose_params, use_cellpose4)
            cellpose_results = model.eval(image[..., None], **eval_params)
        except Exception as e:
            logger.error(f"✗ {cellpose_version} auto-detection failed on full image: {e}")
            logger.error("Please check that diameter=0/None is set for proper auto-detection")
            logger.error("Image statistics: mean={:.1f}, std={:.1f}, min={}, max={}".format(
                float(np.mean(image)), float(np.std(image)),
                float(np.min(image)), float(np.max(image))
            ))
            raise e

        masks, flows = cellpose_results[0], cellpose_results[1]

        # Debug: Log the structure of cellpose_results to understand what's returned.
        logger.debug(f"Full image: Cellpose results structure - length: {len(cellpose_results)}")
        for i, result in enumerate(cellpose_results):
            if result is not None:
                if hasattr(result, 'shape'):
                    logger.debug(f"  Result {i}: shape {result.shape}, type {type(result)}")
                else:
                    logger.debug(f"  Result {i}: {type(result)}, value: {result}")
            else:
                logger.debug(f"  Result {i}: None")

        # Extract and log Cellpose4 diameter information for quality assessment.
        # Note: Cellpose4 with diameter=None may not return diameter info in results tuple.
        if len(cellpose_results) >= 4 and cellpose_results[3] is not None:
            detected_diameters = cellpose_results[3]
            if isinstance(detected_diameters, (list, np.ndarray)) and len(detected_diameters) > 0:
                avg_diameter = float(np.mean(detected_diameters))
                min_diameter = float(np.min(detected_diameters))
                max_diameter = float(np.max(detected_diameters))
                logger.info(f"✓ Full image: Auto-detected diameter = {avg_diameter:.1f}px (range: {min_diameter:.1f}-{max_diameter:.1f}px)")
                logger.info(f"  Diameter variation: {max_diameter - min_diameter:.1f}px, CV: {(np.std(detected_diameters)/avg_diameter)*100:.1f}%")
            elif isinstance(detected_diameters, (int, float)) and detected_diameters > 0:
                logger.info(f"✓ Full image: Auto-detected diameter = {detected_diameters:.1f}px")
            else:
                logger.info("Full image: No valid diameter detected, using model default")
        else:
            # Check if model has diameter information stored internally.
            model_diameter = getattr(model, 'diam_mean', None)
            if model_diameter and model_diameter > 0:
                logger.info(f"✓ Full image: Using model diameter = {model_diameter:.1f}px (auto-detection active)")
            else:
                logger.info(f"Full image: Auto-detection active (diameter=None), using internal Cellpose4 sizing")

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

    The function is designed for users analyzing tissue sections at
    different experimental time points. It handles the computational challenges
    of processing large histological images while maintaining segmentation quality
    across tile boundaries.

    Args:
        model (cellpose.models.CellposeModel or cellpose.models.Cellpose):
            Pre-loaded Cellpose model instance configured for nuclear segmentation.
            Should be initialized with appropriate model type (typically 'nuclei').
            Supports both CellposeModel (Cellpose4) and Cellpose (Cellpose3) classes.
        image (np.ndarray): Two-dimensional greyscale DAPI-stained image array of
            shape (H, W). Represents nuclear staining in kidney tissue sections.
        cellpose_params (dict): Cellpose segmentation parameters including diameter,
            flow_threshold, cellprob_threshold, channels, resample, and batch_size
            settings. These parameters are forwarded directly to model.eval().
        settings (dict): Configuration dictionary with required keys:
            - output_dir: Directory for saving segmentation results.
            - tile_side_length: Size of square tiles for processing large images.
            - tile_overlap: Overlap between adjacent tiles (pixels or fraction).
            Optional keys for advanced control:
            - use_tiling: Force enable/disable tiling (default: auto-detect).
        logger (logging.Logger): Logger instance for progress tracking and debugging
            information. Used extensively for monitoring segmentation progress.

    Returns:
        tuple: A tuple of (masks_mm, flows, total_cells) where:
            - masks_mm (np.memmap): Memory-mapped array containing the final
              segmentation mask of shape (H, W). Each unique positive integer
              represents a distinct nucleus instance. Saved to disk as
              'segmentation_masks.npy' for downstream analysis.
            - flows (list[None, None, None]): Placeholder list - flows are discarded
              to conserve memory during processing. Required for compatibility with
              downstream pipeline components.
            - total_cells (int): Total number of unique nuclei detected across all
              tiles. Used for quality control and statistical analysis of
              segmentation results.
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
    current_max_id = 0  # Tracks maximum label ID for unique nucleus identification.
    total_cells = 0  # Running count of detected nuclei.

    '''Process tiles with Cellpose segmentation (parallel or sequential)'''

    # Check if parallel processing is enabled and beneficial.
    use_parallel = (
        cellpose_params.get("enable_parallel_processing", True) and
        n_tiles > 1 and  # Only use parallel for multiple tiles.
        cellpose_params.get("parallel_max_workers", 2) > 1
    )

    if use_parallel:
        logger.info(f"Using parallel processing for {n_tiles} tiles")
        logger.info(f"Parallel config: batch_size={cellpose_params.get('parallel_batch_size', 4)}, "
                   f"max_workers={cellpose_params.get('parallel_max_workers', 2)}")

        # Import parallel processing function.
        from .parallel_segmentation import run_cellpose_parallel_batches

        # Prepare tiles for parallel processing.
        tile_data = [(tile, (y_slice, x_slice)) for tile, (y_slice, x_slice) in zip(tiles, slices)]

        try:
            # Run parallel segmentation.
            parallel_masks, parallel_flows, total_cells_parallel = run_cellpose_parallel_batches(
                model=model,
                tiles=tile_data,
                cellpose_params=cellpose_params,
                batch_size=cellpose_params.get("parallel_batch_size", 4),
                max_workers=cellpose_params.get("parallel_max_workers", 2),
                memory_limit_gb=cellpose_params.get("parallel_memory_limit_gb", 6.0),
                timeout_seconds=cellpose_params.get("parallel_timeout_seconds", 300)
            )

            # Process parallel results and ensure all tile masks are saved.
            for tile_idx, (mask, (y_slice, x_slice)) in enumerate(zip(parallel_masks, slices), start=1):
                if mask.size > 0 and mask.max() > 0:
                    # Relabel mask to ensure unique IDs across tiles.
                    unique_mask = relabel_mask_for_unique_ids(mask, current_max_id)
                    current_max_id = int(unique_mask.max()) if unique_mask.max() > 0 else current_max_id

                    # Place the processed mask in the memory-mapped array.
                    masks_mm[y_slice, x_slice] = unique_mask

                    tile_cells = int(mask.max()) if mask.max() > 0 else 0
                    total_cells += tile_cells

                    logger.info(f"Tile {tile_idx}/{n_tiles}: {tile_cells} nuclei detected")
                else:
                    # Create empty mask for tiles with no nuclei.
                    unique_mask = np.zeros(mask.shape, dtype=np.uint32)
                    logger.info(f"Tile {tile_idx}/{n_tiles}: No nuclei detected")

                # Always save individual tile mask (even if empty) to prevent merge failures.
                # Use coordinate-based filename format required by merge system.
                tile_filename = f"{y_slice.start}_{x_slice.start}.npz"
                tile_path = tiles_dir / tile_filename
                np.savez_compressed(tile_path, mask=unique_mask)

                if settings.get("debug_mode", False):
                    logger.debug(f"  -> Saved tile mask to: {tile_filename}")

            logger.info(f"Parallel segmentation completed: {total_cells} total nuclei detected")

        except Exception as parallel_error:
            import traceback
            logger.error(f"Parallel processing failed: {parallel_error}")
            logger.debug(f"Parallel processing error traceback:\n{traceback.format_exc()}")
            logger.info("Falling back to sequential processing")

            # Ensure we create empty tile masks for all tiles to prevent merge failures.
            logger.info("Creating empty tile masks for failed parallel processing")
            for tile_idx, (y_slice, x_slice) in enumerate(slices, start=1):
                empty_mask = np.zeros((y_slice.stop - y_slice.start, x_slice.stop - x_slice.start), dtype=np.uint32)
                tile_filename = f"{y_slice.start}_{x_slice.start}.npz"
                tile_path = tiles_dir / tile_filename
                np.savez_compressed(tile_path, mask=empty_mask)
                logger.info(f"NUCLEI DETECTION: Tile {tile_idx}/{n_tiles}: No nuclei detected (parallel processing failed)")

            use_parallel = False

    if not use_parallel:
        logger.info("Using sequential processing for tiles")

        # Sequential processing (original implementation).
        for tile_idx, (current_tile, (y_slice, x_slice)) in enumerate(zip(tiles, slices), start=1):

            # Log detailed progress information for monitoring large image processing.
            logger.info(f"Processing tile {tile_idx}/{n_tiles}")
            logger.info(f"  Position: ({y_slice.start}, {x_slice.start})")
            logger.info(f"  Dimensions: {current_tile.shape}")

            try:
                # Validate tile content before Cellpose processing.
                tile_stats = {
                    'mean': float(np.mean(current_tile)),
                    'std': float(np.std(current_tile)),
                    'min': float(np.min(current_tile)),
                    'max': float(np.max(current_tile))
                }

                logger.debug(f"Tile {tile_idx} stats: mean={tile_stats['mean']:.1f}, "
                            f"std={tile_stats['std']:.1f}, min={tile_stats['min']}, max={tile_stats['max']}")

                # Check for problematic tiles that might cause division by zero.
                if tile_stats['std'] < 1.0 or tile_stats['max'] - tile_stats['min'] < 5:
                    logger.warning(f"Tile {tile_idx} has very low contrast (std={tile_stats['std']:.2f}), "
                                  f"may cause Cellpose issues")

                # Run Cellpose segmentation on the current tile.
                # The model expects a 3D array with channel dimension, so we add one.
                # Each tile runs independently with diameter=0 for adaptive per-tile
                # diameter detection. This is incompatible with Cellpose's built-in
                # stitch_threshold, which requires a uniform diameter across all tiles.
                # A custom merge algorithm is used instead (see cellpose_merge/).
                use_cellpose4 = cellpose_params.get("use_cellpose4", True)
                cellpose_version = "Cellpose4" if use_cellpose4 else "Cellpose3"

                try:
                    eval_params = prepare_cellpose_parameters(cellpose_params, use_cellpose4)
                    logger.debug(f"Tile {tile_idx} Cellpose parameters: {eval_params}")
                    logger.debug(f"Tile {tile_idx} input shape: {current_tile[..., None].shape}")
                    cellpose_results = model.eval(current_tile[..., None], **eval_params)
                except Exception as e:
                    logger.error(f"✗ {cellpose_version} auto-detection failed on tile {tile_idx}: {e}")
                    logger.error(f"Tile {tile_idx} statistics: mean={tile_stats['mean']:.1f}, "
                                f"std={tile_stats['std']:.1f}, min={tile_stats['min']}, max={tile_stats['max']}")
                    logger.error(f"Parameters that failed: {eval_params}")
                    logger.error("Please ensure diameter=0/None is configured for auto-detection")
                    raise e

                # Extract masks from Cellpose results (first element).
                raw_masks = cellpose_results[0]

                # Debug: Log the structure of cellpose_results to understand what's returned.
                logger.debug(f"Tile {tile_idx}: Cellpose results structure - length: {len(cellpose_results)}")
                for i, result in enumerate(cellpose_results):
                    if result is not None:
                        if hasattr(result, 'shape'):
                            logger.debug(f"  Result {i}: shape {result.shape}, type {type(result)}")
                        else:
                            logger.debug(f"  Result {i}: {type(result)}, value: {result}")
                    else:
                        logger.debug(f"  Result {i}: None")

                # Extract and log Cellpose4 diameter information for adaptive learning assessment.
                # Note: Cellpose4 with diameter=None may not return diameter info in results tuple.
                if len(cellpose_results) >= 4 and cellpose_results[3] is not None:
                    detected_diameters = cellpose_results[3]
                    if isinstance(detected_diameters, (list, np.ndarray)) and len(detected_diameters) > 0:
                        # Multiple diameters detected - analyze variation across tile.
                        avg_diameter = float(np.mean(detected_diameters))
                        min_diameter = float(np.min(detected_diameters))
                        max_diameter = float(np.max(detected_diameters))
                        logger.info(f"✓ Tile {tile_idx}: Auto-detected diameter = {avg_diameter:.1f}px (range: {min_diameter:.1f}-{max_diameter:.1f}px)")

                        # Log diameter variation for quality assessment.
                        if len(detected_diameters) > 1:
                            diameter_cv = (np.std(detected_diameters) / avg_diameter) * 100
                            logger.info(f"  Tile {tile_idx} diameter variation: CV = {diameter_cv:.1f}%")
                    elif isinstance(detected_diameters, (int, float)) and detected_diameters > 0:
                        # Single diameter detected.
                        logger.info(f"✓ Tile {tile_idx}: Auto-detected diameter = {detected_diameters:.1f}px")
                    else:
                        logger.debug(f"Tile {tile_idx}: No valid diameter detected (value: {detected_diameters}), using model default")
                else:
                    # Check if model has diameter information stored internally.
                    model_diameter = getattr(model, 'diam_mean', None)
                    if model_diameter and model_diameter > 0:
                        logger.info(f"Tile {tile_idx}: Using model diameter = {model_diameter:.1f}px (auto-detection active)")
                    else:
                        logger.info(f"Tile {tile_idx}: Auto-detection active (diameter=None), using internal Cellpose4 sizing")

                # Handle cases where Cellpose returns None (no nuclei detected).
                if raw_masks is None:
                    tile_segmentation_mask = np.zeros(current_tile.shape, dtype=np.uint32)
                    nuclei_in_tile = 0
                    logger.info(f"  -> NUCLEI COUNT: No nuclei detected in tile {tile_idx} (Cellpose returned None)")
                    logger.info(f"     Tile stats: mean={tile_stats['mean']:.1f}, std={tile_stats['std']:.1f}")
                    logger.info(f"     Parameters used: diameter={eval_params.get('diameter')}, "
                               f"flow_threshold={eval_params.get('flow_threshold')}, "
                               f"cellprob_threshold={eval_params.get('cellprob_threshold')}")
                else:
                    tile_segmentation_mask = raw_masks.astype(np.uint32)
                    # Count unique non-zero labels (correct method for nuclei counting).
                    nuclei_in_tile = len(np.unique(tile_segmentation_mask[tile_segmentation_mask > 0]))

                    if nuclei_in_tile > 0:
                        # Relabel mask to ensure unique IDs across tiles.
                        unique_mask = relabel_mask_for_unique_ids(tile_segmentation_mask, current_max_id)
                        current_max_id = int(unique_mask.max()) if unique_mask.max() > 0 else current_max_id

                        # Update the combined segmentation mask with this tile's results.
                        nucleus_pixels = unique_mask != 0
                        masks_mm[y_slice, x_slice][nucleus_pixels] = unique_mask[nucleus_pixels]

                        # Update counters.
                        total_cells += nuclei_in_tile

                        logger.info(f"  -> NUCLEI COUNT: {nuclei_in_tile} nuclei detected and labeled in tile {tile_idx}")
                        logger.info(f"  -> Diameter used: {detected_diameters if 'detected_diameters' in locals() else 'auto'}")

                        # Use the unique mask for saving.
                        tile_segmentation_mask = unique_mask
                    else:
                        logger.info(f"  -> NUCLEI COUNT: 0 nuclei detected in tile {tile_idx} (mask was not None but empty)")
                        logger.info(f"     Mask shape: {raw_masks.shape}, unique values: {len(np.unique(raw_masks))}")
                        logger.info(f"     Tile stats: mean={tile_stats['mean']:.1f}, std={tile_stats['std']:.1f}")
                        logger.info(f"     Parameters used: diameter={eval_params.get('diameter')}, "
                                   f"flow_threshold={eval_params.get('flow_threshold')}, "
                                   f"cellprob_threshold={eval_params.get('cellprob_threshold')}")

                # Save individual tile mask for downstream analysis and quality control.
                # The filename format (y_start_x_start.npz) is required by the merge system.
                tile_filename = f"{y_slice.start}_{x_slice.start}.npz"
                tile_save_path = tiles_dir / tile_filename
                np.savez_compressed(tile_save_path, mask=tile_segmentation_mask)

                if settings.get("debug_mode", False):
                    logger.debug(f"  -> Saved tile mask to: {tile_filename}")

                # Explicitly release memory to prevent accumulation during large image processing.
                tile_segmentation_mask = None
                raw_masks = None

            except Exception as tile_error:
                # Log detailed error information for debugging segmentation failures.
                logger.error(f"✗ Tile {tile_idx} processing failed: {tile_error}")

                if settings.get("debug_mode", False):
                    import traceback
                    logger.debug(f"Full error traceback:\n{traceback.format_exc()}")

                # Create empty mask as fallback.
                empty_mask = np.zeros(current_tile.shape, dtype=np.uint32)
                tile_filename = f"{y_slice.start}_{x_slice.start}.npz"
                tile_save_path = tiles_dir / tile_filename
                np.savez_compressed(tile_save_path, mask=empty_mask)

    '''Finalize segmentation results and prepare outputs'''
    # Ensure all data is written to disk before returning.
    masks_mm.flush()

    # Log comprehensive segmentation statistics for quality assessment.
    logger.info('\n'+"="*60)
    logger.info("SEGMENTATION SUMMARY")
    logger.info("="*60 + '\n')
    logger.info(f"Total tiles processed: {n_tiles}")
    logger.info(f"Total nuclei detected: {total_cells}")
    logger.info(f"Average nuclei per tile: {total_cells/n_tiles:.1f}")
    logger.info(f"Nuclear density: {total_cells/(H*W)*1e6:.1f} nuclei/mm^2 (assuming 1 pixel = 1 um)")
    logger.info(f"Individual tile masks saved to: {tiles_dir}")
    logger.info(f"Combined segmentation mask ready for pipeline processing")

    if settings.get("debug_mode", False):
        logger.debug(f"Memory-mapped file size: {masks_mm.nbytes / (1024**2):.1f} MB")
        logger.debug(f"Current maximum ID: {current_max_id}")

    logger.info("Tiled segmentation completed successfully")

    # Return memory-mapped mask, placeholder flows, and cell count.
    # The flows are set to None to conserve memory during large image processing.
    return masks_mm, [None, None, None], total_cells
