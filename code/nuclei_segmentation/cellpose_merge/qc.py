"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: qc.py.
Description:
    Generate publication-quality QC visualizations for tiled nucleus segmentation
    in kidney I/R injury tissue analysis. This module creates comprehensive
    before/after overlays that help bioinformaticians assess tile merging quality
    and identify potential segmentation artifacts at tile boundaries.

    The QC process extracts 1300x1300 pixel crops from image centers and creates
    two critical visualizations:
    1. Before merging: Individual tile masks with unique colors overlaid on tissue
    2. After merging: Final merged masks with random colors for quality assessment

Dependencies:
    • Python >= 3.10.
    • numpy, pillow, tqdm, pytest for core functionality.
    • matplotlib for advanced color generation and visualization.

Usage:
    from qc import write_overlays
    
    write_overlays(
        loader=tile_loader_function,
        merged=merged_segmentation_mask,
        height=image_height,
        width=image_width,
        tile_h=tile_height,
        tile_w=tile_width,
        overlap=tile_overlap,
        qc_dir="./qc_output"
    )

Arguments:
    loader : callable
        Function that loads individual tile masks given slice coordinates.
        Essential for reconstructing pre-merge tile visualization.
    merged : np.ndarray
        Final merged segmentation mask of shape (height, width) with unique
        nucleus labels. Background pixels should be zero.
    height, width : int
        Full tissue image dimensions in pixels for proper coordinate mapping.
    tile_h, tile_w : int
        Individual tile dimensions used during Cellpose segmentation.
    overlap : int
        Spatial overlap between adjacent tiles in pixels.
    qc_dir : str or Path
        Output directory for QC visualization files.

Inputs:
    • Tile loader function providing access to individual segmentation masks.
    • Merged segmentation results from the tile merging pipeline.
    • Tissue image dimensions and tiling parameters for spatial reconstruction.

Outputs:
    • before_merging.tif: Individual tile masks with unique colors per tile.
    • after_merging.tif: Final merged masks with random colors per nucleus.
    • merge_statistics.txt: Quantitative summary of segmentation results.

Key Features:
    • Intelligent image cropping for manageable visualization file sizes.
    • Unique color assignment per tile for easy identification of tile boundaries.
    • Alpha blending for proper overlay visualization on tissue background.
    • Comprehensive error handling for edge cases and missing data.
    • Scientific context-aware logging for bioinformatics workflows.

Notes:
    • This module is specifically designed for kidney I/R injury tissue analysis.
    • Color generation uses deterministic algorithms for reproducible results.
    • All overlays use 16-bit composition buffers to prevent overflow artifacts.
    • The QC process helps identify nucleus boundary alignment issues across tiles.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import traceback
from pathlib import Path
from typing import Final, Iterable, Tuple, Callable, Dict, List
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

"""TYPE DEFINITIONS"""

# Type aliases for better code readability and scientific context.
RGBArray = NDArray[np.uint8]  # RGB image arrays for tissue visualization.
MaskArray = NDArray[np.uint32]  # Nucleus segmentation masks with unique labels.
ColorArray = NDArray[np.uint16]  # Color arrays for overlay composition.

"""CONFIGURATION PARAMETERS"""

# Default crop size for QC visualizations to ensure manageable file sizes.
DEFAULT_CROP_SIZE: Final = 1300

# Alpha transparency value for overlay blending on tissue images.
DEFAULT_ALPHA: Final = 0.5

# Regex patterns for parsing tile coordinate information from filenames.
# These patterns support both pixel coordinates and tile indices.
TILE_FILENAME_PATTERNS: Final = [
    # Direct pixel coordinates: "12345_67890.tif" -> y=12345, x=67890.
    re.compile(r"(?P<y>\d+)[_ ](?P<x>\d+)"),
    # Tile indices: "row5_col7.npz" -> row=5, col=7 (converted to pixels later).
    re.compile(r"row(?P<row>\d+)[_ ]col(?P<col>\d+)")
]

"""MAIN QC GENERATION FUNCTION"""

def write_overlays(
    loader: Callable[[slice, slice], NDArray[np.uint32]],
    merged: NDArray[np.uint32],
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    qc_dir: str | Path,
    image_loader: Callable[[slice, slice], NDArray[np.uint8]] = None,
    use_full_image: bool = False,
    coords: List[Tuple[int, int]] = None,
    original_tiles_path: Path = None,
    merged_tiles_dir: Path = None,
) -> None:
    """
    Generate comprehensive QC overlays for nucleus segmentation merge validation.

    This function creates publication-quality before/after overlay images that help
    bioinformaticians assess the quality of tile merging in kidney I/R injury tissue
    sections. The visualizations highlight potential issues with nucleus boundary
    alignment and merging accuracy across tile boundaries.

    The QC process involves:
    1. Extracting a central crop from the tissue image for manageable visualization
    2. Creating a "before merging" overlay showing individual tile contributions
    3. Creating an "after merging" overlay showing the final segmentation result
    4. Generating quantitative statistics about the segmentation quality

    Parameters
    ----------
    loader : callable
        Function that loads individual tile masks given slice coordinates.
        Should return a 2D numpy array with nucleus labels for the requested region.
    merged : np.ndarray
        Final merged segmentation mask of shape (height, width) with unique
        nucleus labels. Background pixels should be zero.
    height, width : int
        Full tissue image dimensions in pixels for proper spatial coordinate mapping.
    tile_h, tile_w : int
        Individual tile dimensions used during Cellpose segmentation process.
    overlap : int
        Spatial overlap between adjacent tiles in pixels.
    qc_dir : str or Path
        Output directory where QC visualization files will be saved.
    image_loader : callable, optional
        Function that loads the original tissue image given slice coordinates.
        If provided, the actual tissue image will be used as background.
    use_full_image : bool, default True
        If True, generate QC overlays for the entire image. If False, use a central
        crop of DEFAULT_CROP_SIZE for manageable file sizes.

    Returns
    -------
    None
        Function saves QC files directly to the specified directory.

    Raises
    ------
    Exception
        Logs warnings for any errors during QC generation but does not halt execution.
    """

    try:
        qc_dir = Path(qc_dir)
        qc_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Starting QC overlay generation for kidney tissue segmentation analysis.")
        logging.info(f"QC visualizations will be saved to: {qc_dir}")
        logging.info(f"Processing tissue image of size {height}x{width} pixels.")

        # Extract crop region for visualization.
        if use_full_image:
            # Use the entire image for comprehensive QC visualization.
            crop_info = _calculate_crop_region(height, width, max(height, width))
            logging.info(f"Using full image for QC: {crop_info['description']}")
        else:
            # Extract central crop for manageable file sizes.
            crop_info = _calculate_crop_region(height, width, DEFAULT_CROP_SIZE)
            logging.info(f"Using crop region: {crop_info['description']}")

        # Load the tissue image background for proper visualization.
        tissue_background = _load_tissue_background(image_loader, crop_info, height, width)

        # Generate the before merging visualization showing individual tile contributions.
        logging.info("Generating before merging visualization...")
        if original_tiles_path and coords:
            # Use original tile masks for "before" visualization.
            before_overlay = _create_before_merging_overlay_from_files(
                original_tiles_path, coords, crop_info, tile_h, tile_w, overlap, tissue_background
            )
        else:
            # Fallback to loader function.
            before_overlay = _create_before_merging_overlay(
                loader, crop_info, tile_h, tile_w, overlap, tissue_background
            )

        before_path = qc_dir / "before_merging.tif"
        Image.fromarray(before_overlay).save(before_path, compression="tiff_deflate")
        logging.info(f"Saved before merging overlay to: {before_path}")

        # Generate the after merging visualization showing final segmentation results.
        logging.info("Generating after merging visualization...")
        if merged_tiles_dir and coords:
            # Use merged tile masks for "after" visualization.
            after_overlay = _create_after_merging_overlay_from_files(
                merged_tiles_dir, coords, crop_info, tile_h, tile_w, overlap, tissue_background
            )
        else:
            # Fallback to merged mask.
            after_overlay = _create_after_merging_overlay(merged, crop_info, tissue_background)

        after_path = qc_dir / "after_merging.tif"
        Image.fromarray(after_overlay).save(after_path, compression="tiff_deflate")
        logging.info(f"Saved after merging overlay to: {after_path}")

        # Generate quantitative statistics about the segmentation results.
        _generate_merge_statistics(merged, height, width, tile_h, tile_w, overlap, qc_dir)

        logging.info("QC overlay generation completed successfully.")

    except Exception as qc_error:
        logging.warning(f"QC overlay generation encountered an error: {qc_error}")
        logging.debug(f"QC error traceback:\n{traceback.format_exc()}")

"""CROP REGION CALCULATION"""

def _calculate_crop_region(height: int, width: int, crop_size: int) -> Dict:
    """
    Calculate the optimal crop region for QC visualization.

    For large tissue images, this function determines a central crop region that
    provides representative visualization while maintaining manageable file sizes.
    For smaller images, the entire image is used.

    Parameters
    ----------
    height, width : int
        Full tissue image dimensions in pixels.
    crop_size : int
        Target crop size for visualization (typically 1300 pixels).

    Returns
    -------
    Dict
        Dictionary containing crop coordinates and description for logging.
    """

    if height <= crop_size and width <= crop_size:
        # Use entire image if it's smaller than the target crop size.
        return {
            'y_start': 0, 'y_end': height,
            'x_start': 0, 'x_end': width,
            'height': height, 'width': width,
            'description': f"full image ({height}x{width})"
        }
    else:
        # Extract central crop for large images.
        center_y = height // 2
        center_x = width // 2
        half_crop = crop_size // 2
        
        y_start = max(0, center_y - half_crop)
        y_end = min(height, center_y + half_crop)
        x_start = max(0, center_x - half_crop)
        x_end = min(width, center_x + half_crop)
        
        return {
            'y_start': y_start, 'y_end': y_end,
            'x_start': x_start, 'x_end': x_end,
            'height': y_end - y_start, 'width': x_end - x_start,
            'description': f"central crop ({y_end - y_start}x{x_end - x_start}) from ({y_start},{x_start})"
        }


"""TISSUE BACKGROUND LOADING"""

def _load_tissue_background(
    image_loader: Callable[[slice, slice], NDArray[np.uint8]],
    crop_info: Dict,
    height: int,
    width: int
) -> RGBArray:
    """
    Load the actual tissue image background for QC visualization.

    This function loads the original tissue image that serves as the background
    for overlay visualizations. If no image loader is provided, it creates a
    neutral gray background for the visualization.

    Parameters
    ----------
    image_loader : callable or None
        Function that loads the original tissue image given slice coordinates.
        Should return RGB image data as uint8 array.
    crop_info : Dict
        Dictionary containing crop region coordinates and dimensions.
    height, width : int
        Full tissue image dimensions in pixels.

    Returns
    -------
    RGBArray
        RGB tissue image background for the crop region.
    """

    crop_height = crop_info['height']
    crop_width = crop_info['width']

    if image_loader is not None:
        try:
            # Load the actual tissue image for the crop region.
            crop_y_start = crop_info['y_start']
            crop_y_end = crop_info['y_end']
            crop_x_start = crop_info['x_start']
            crop_x_end = crop_info['x_end']

            tissue_slice_y = slice(crop_y_start, crop_y_end)
            tissue_slice_x = slice(crop_x_start, crop_x_end)

            tissue_crop = image_loader(tissue_slice_y, tissue_slice_x)

            # Ensure the image is in RGB format.
            if tissue_crop.ndim == 2:
                # Convert grayscale to RGB.
                tissue_crop = np.stack([tissue_crop] * 3, axis=-1)
            elif tissue_crop.ndim == 3 and tissue_crop.shape[2] == 1:
                # Convert single channel to RGB.
                tissue_crop = np.repeat(tissue_crop, 3, axis=2)
            elif tissue_crop.ndim == 3 and tissue_crop.shape[2] > 3:
                # Take first 3 channels if more than RGB.
                tissue_crop = tissue_crop[:, :, :3]

            # Ensure correct dimensions.
            if tissue_crop.shape[:2] == (crop_height, crop_width):
                logging.debug(f"Loaded tissue background: {tissue_crop.shape}")
                return tissue_crop.astype(np.uint8)
            else:
                logging.warning(f"Tissue crop size mismatch: got {tissue_crop.shape[:2]}, expected ({crop_height}, {crop_width})")

        except Exception as load_error:
            logging.warning(f"Failed to load tissue background: {load_error}")

    # Fallback: Create a neutral gray background.
    logging.info("Using neutral gray background for QC visualization")
    background = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    return background


"""BEFORE MERGING VISUALIZATION"""

def _create_before_merging_overlay(
    loader: Callable[[slice, slice], NDArray[np.uint32]],
    crop_info: Dict,
    tile_h: int,
    tile_w: int,
    overlap: int,
    tissue_background: RGBArray
) -> RGBArray:
    """
    Create the before merging overlay showing individual tile contributions.

    This visualization displays all individual tile masks overlaid on top of each
    other with unique colors corresponding to their tile of origin. This helps
    bioinformaticians identify tile boundaries and potential merging conflicts
    in kidney tissue segmentation.

    Parameters
    ----------
    loader : callable
        Function that loads individual tile masks for given slice coordinates.
    crop_info : Dict
        Dictionary containing crop region coordinates and dimensions.
    tile_h, tile_w : int
        Individual tile dimensions used during segmentation.
    overlap : int
        Spatial overlap between adjacent tiles in pixels.
    tissue_background : RGBArray
        RGB tissue image to use as background for the overlay.

    Returns
    -------
    RGBArray
        RGB image array showing individual tile contributions with unique colors.
    """

    logging.debug("Creating before merging overlay with individual tile colors on tissue background.")

    # Initialize the overlay canvas with the actual tissue background.
    crop_height = crop_info['height']
    crop_width = crop_info['width']
    overlay = tissue_background.astype(np.uint16)  # Use tissue background as base.

    # Calculate tile grid parameters for the crop region.
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    # Determine which tiles intersect with the crop region.
    crop_y_start = crop_info['y_start']
    crop_y_end = crop_info['y_end']
    crop_x_start = crop_info['x_start']
    crop_x_end = crop_info['x_end']

    # Calculate tile index ranges that intersect with the crop.
    tile_row_start = max(0, crop_y_start // stride_h)
    tile_row_end = (crop_y_end + stride_h - 1) // stride_h
    tile_col_start = max(0, crop_x_start // stride_w)
    tile_col_end = (crop_x_end + stride_w - 1) // stride_w

    logging.debug(f"Processing tiles from row {tile_row_start}-{tile_row_end}, col {tile_col_start}-{tile_col_end}")

    tiles_processed = 0

    # Process each tile that intersects with the crop region.
    for tile_row in range(tile_row_start, tile_row_end):
        for tile_col in range(tile_col_start, tile_col_end):

            # Calculate tile position in global coordinates.
            tile_global_y = tile_row * stride_h
            tile_global_x = tile_col * stride_w

            # Load the tile mask for this region.
            tile_slice_y = slice(tile_global_y, tile_global_y + tile_h)
            tile_slice_x = slice(tile_global_x, tile_global_x + tile_w)

            try:
                tile_mask = loader(tile_slice_y, tile_slice_x)

                if tile_mask is None or tile_mask.size == 0:
                    logging.debug(f"Empty tile mask for tile ({tile_row}, {tile_col})")
                    continue

                logging.debug(f"Loaded tile ({tile_row}, {tile_col}): shape={tile_mask.shape}, "
                             f"max_label={tile_mask.max()}, non_zero={np.count_nonzero(tile_mask)}")

                # Generate a unique color for this tile based on its position.
                tile_color = _generate_tile_color(tile_row, tile_col)

                # Calculate intersection with crop region.
                intersect_y_start = max(crop_y_start, tile_global_y)
                intersect_y_end = min(crop_y_end, tile_global_y + tile_mask.shape[0])
                intersect_x_start = max(crop_x_start, tile_global_x)
                intersect_x_end = min(crop_x_end, tile_global_x + tile_mask.shape[1])

                if intersect_y_end <= intersect_y_start or intersect_x_end <= intersect_x_start:
                    continue

                # Extract the relevant portion of the tile mask.
                tile_y_offset = intersect_y_start - tile_global_y
                tile_y_end_offset = intersect_y_end - tile_global_y
                tile_x_offset = intersect_x_start - tile_global_x
                tile_x_end_offset = intersect_x_end - tile_global_x

                mask_region = tile_mask[tile_y_offset:tile_y_end_offset, tile_x_offset:tile_x_end_offset]

                # Calculate position in the crop overlay.
                overlay_y_start = intersect_y_start - crop_y_start
                overlay_y_end = intersect_y_end - crop_y_start
                overlay_x_start = intersect_x_start - crop_x_start
                overlay_x_end = intersect_x_end - crop_x_start

                # Apply the tile color where nuclei are present with improved transparency.
                nucleus_pixels = mask_region > 0
                if np.any(nucleus_pixels):
                    overlay_region = overlay[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]

                    # Use more transparent blending for better visibility of overlaps.
                    tile_alpha = 0.3  # More transparent than default for better overlap visibility.

                    # Add tile boundary visualization for better identification.
                    # Create a slightly larger mask for tile boundaries.
                    boundary_mask = _create_tile_boundary_mask(mask_region, tile_row, tile_col)

                    # Blend the tile color with improved alpha transparency.
                    for c in range(3):
                        # Apply nucleus color.
                        overlay_region[nucleus_pixels, c] = (
                            (1 - tile_alpha) * overlay_region[nucleus_pixels, c] +
                            tile_alpha * tile_color[c]
                        ).astype(np.uint16)

                        # Add subtle tile boundary indicators.
                        if np.any(boundary_mask):
                            boundary_alpha = 0.15  # Very subtle boundary indication.
                            overlay_region[boundary_mask, c] = (
                                (1 - boundary_alpha) * overlay_region[boundary_mask, c] +
                                boundary_alpha * tile_color[c]
                            ).astype(np.uint16)

                tiles_processed += 1

            except Exception as tile_error:
                logging.debug(f"Error processing tile ({tile_row}, {tile_col}): {tile_error}")
                continue

    logging.info(f"Before merging overlay completed with {tiles_processed} tiles processed.")
    logging.debug(f"Final overlay stats: shape={overlay.shape}, min={overlay.min()}, max={overlay.max()}")

    # Convert to uint8 for final output.
    return overlay.clip(0, 255).astype(np.uint8)


def _create_tile_boundary_mask(mask_region: NDArray[np.uint32], tile_row: int, tile_col: int) -> NDArray[np.bool_]:
    """
    Create a subtle boundary mask to help identify tile edges in overlapping regions.

    This function creates a thin boundary around the tile region to help
    bioinformaticians identify which tile each nucleus belongs to, especially
    in overlapping regions where multiple tiles contribute.

    Parameters
    ----------
    mask_region : NDArray[np.uint32]
        The nucleus mask for this tile region.
    tile_row, tile_col : int
        Tile position indices for identification.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask indicating tile boundary pixels.
    """

    if mask_region.size == 0:
        return np.zeros_like(mask_region, dtype=bool)

    # Create boundary mask at tile edges (first/last few pixels).
    boundary_width = 3  # Width of boundary indication in pixels.
    h, w = mask_region.shape
    boundary_mask = np.zeros((h, w), dtype=bool)

    # Add boundaries at tile edges.
    if h > boundary_width * 2 and w > boundary_width * 2:
        # Top and bottom edges.
        boundary_mask[:boundary_width, :] = True
        boundary_mask[-boundary_width:, :] = True
        # Left and right edges.
        boundary_mask[:, :boundary_width] = True
        boundary_mask[:, -boundary_width:] = True

    # Only apply boundary where there are no nuclei to avoid obscuring data.
    boundary_mask = boundary_mask & (mask_region == 0)

    return boundary_mask


def _generate_tile_color(tile_row: int, tile_col: int) -> ColorArray:
    """
    Generate a unique, high-contrast color for a tile based on its position.

    This function creates visually distinct colors for each tile to help
    bioinformaticians identify tile boundaries and origins in the QC overlay.
    The color generation uses improved contrast and saturation for better
    visibility on tissue backgrounds.

    Parameters
    ----------
    tile_row, tile_col : int
        Tile position indices in the grid.

    Returns
    -------
    ColorArray
        RGB color array for the tile as uint16 values.
    """

    # Create a unique identifier for this tile position.
    tile_id = f"{tile_row}_{tile_col}"

    # Generate deterministic color using hash of tile identifier.
    hash_bytes = hashlib.sha256(tile_id.encode()).digest()

    # Use multiple hash bytes for better color distribution.
    r_base = hash_bytes[0]
    g_base = hash_bytes[1]
    b_base = hash_bytes[2]

    # Create high-contrast colors with good visibility on tissue background.
    # Use a color palette approach for better distinction.
    color_palette = [
        [255, 100, 100],  # Bright red.
        [100, 255, 100],  # Bright green.
        [100, 100, 255],  # Bright blue.
        [255, 255, 100],  # Bright yellow.
        [255, 100, 255],  # Bright magenta.
        [100, 255, 255],  # Bright cyan.
        [255, 150, 100],  # Orange.
        [150, 100, 255],  # Purple.
        [255, 100, 150],  # Pink.
        [150, 255, 100],  # Lime.
    ]

    # Select color based on tile position with some variation.
    palette_index = (tile_row * 7 + tile_col * 11) % len(color_palette)
    base_color = color_palette[palette_index]

    # Add some variation based on hash to avoid identical colors for distant tiles.
    variation = 30  # Amount of color variation.
    r = np.clip(base_color[0] + (r_base % (2 * variation)) - variation, 100, 255)
    g = np.clip(base_color[1] + (g_base % (2 * variation)) - variation, 100, 255)
    b = np.clip(base_color[2] + (b_base % (2 * variation)) - variation, 100, 255)

    return np.array([r, g, b], dtype=np.uint16)


"""AFTER MERGING VISUALIZATION"""

def _create_after_merging_overlay(merged: NDArray[np.uint32], crop_info: Dict, tissue_background: RGBArray) -> RGBArray:
    """
    Create the after merging overlay showing final segmentation results.

    This visualization displays the final merged masks with random colors assigned
    to each segmented nucleus. This helps bioinformaticians assess the quality of
    the final merged segmentation and identify potential artifacts or issues.

    Parameters
    ----------
    merged : NDArray[np.uint32]
        Final merged segmentation mask with unique nucleus labels.
    crop_info : Dict
        Dictionary containing crop region coordinates and dimensions.
    tissue_background : RGBArray
        RGB tissue image to use as background for the overlay.

    Returns
    -------
    RGBArray
        RGB image array showing final segmentation with random colors per nucleus.
    """

    logging.debug("Creating after merging overlay with random nucleus colors.")

    # Extract the crop region from the merged mask.
    crop_y_start = crop_info['y_start']
    crop_y_end = crop_info['y_end']
    crop_x_start = crop_info['x_start']
    crop_x_end = crop_info['x_end']

    merged_crop = merged[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # Initialize the overlay canvas with tissue background.
    crop_height, crop_width = merged_crop.shape
    overlay = tissue_background.copy().astype(np.uint16)  # Use tissue background as base.

    # Generate random colors for each nucleus label.
    max_label = int(merged_crop.max())

    if max_label > 0:
        logging.debug(f"Generating colors for {max_label} nucleus labels.")

        # Use reproducible random seed for consistent visualization.
        np.random.seed(42)
        colors = np.random.randint(100, 256, size=(max_label + 1, 3), dtype=np.uint16)
        colors[0] = [0, 0, 0]  # Background remains transparent.

        # Apply colors to each nucleus with alpha blending.
        nucleus_alpha = 0.4  # More transparent for better tissue visibility.

        for label in range(1, max_label + 1):
            nucleus_mask = merged_crop == label
            if np.any(nucleus_mask):
                # Blend nucleus color with tissue background.
                for c in range(3):
                    overlay[nucleus_mask, c] = (
                        (1 - nucleus_alpha) * overlay[nucleus_mask, c] +
                        nucleus_alpha * colors[label, c]
                    ).astype(np.uint16)

        logging.debug(f"Applied colors to nuclei in after merging overlay.")
    else:
        logging.warning("No nuclei found in merged mask crop region.")

    # Convert back to uint8 for final output.
    return overlay.clip(0, 255).astype(np.uint8)


"""IMPROVED QC OVERLAY FUNCTIONS WITH PERSISTENT STORAGE"""

def _create_before_merging_overlay_from_files(
    original_tiles_path: Path,
    coords: List[Tuple[int, int]],
    crop_info: Dict,
    tile_h: int,
    tile_w: int,
    overlap: int,
    tissue_background: RGBArray
) -> RGBArray:
    """
    Create before merging overlay using original tile mask files.

    This function loads individual tile masks from the original tile_masks_npz
    directory and creates a visualization showing overlapping masks with
    tile-specific colors. This provides a clear view of how tiles overlap
    before merging occurs.

    Parameters
    ----------
    original_tiles_path : Path
        Path to directory containing original tile mask files.
    coords : List[Tuple[int, int]]
        List of (row, col) tile coordinates.
    crop_info : Dict
        Dictionary containing crop region information.
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    tissue_background : RGBArray
        Background tissue image for overlay.

    Returns
    -------
    RGBArray
        RGB overlay image showing original tile masks with unique colors.
    """
    logging.info("Creating before merging overlay from original tile files")

    # Initialize overlay with tissue background.
    overlay = tissue_background.astype(np.uint16)

    # Calculate stride and crop parameters.
    stride_h, stride_w = tile_h - overlap, tile_w - overlap
    crop_y, crop_x = crop_info["y_start"], crop_info["x_start"]
    crop_h, crop_w = crop_info["height"], crop_info["width"]

    # Generate unique colors for each tile.
    tile_colors = {}
    for i, coord in enumerate(coords):
        # Generate deterministic color based on tile coordinate.
        r, c = coord
        color_seed = (r * 1000 + c) % 1000
        tile_colors[coord] = _generate_tile_color(color_seed)

    # Process each tile and add to overlay.
    for r, c in coords:
        try:
            # Convert tile indices to pixel coordinates for original tile lookup.
            pixel_r = r * stride_h
            pixel_c = c * stride_w
            tile_filename = f"{pixel_r}_{pixel_c}.npz"
            tile_path = original_tiles_path / tile_filename

            if not tile_path.exists():
                logging.warning(f"Original tile mask not found: {tile_path}")
                continue

            tile_data = np.load(tile_path)["mask"]

            # Calculate tile position in global coordinates.
            tile_y0, tile_x0 = r * stride_h, c * stride_w
            tile_y1 = tile_y0 + tile_data.shape[0]
            tile_x1 = tile_x0 + tile_data.shape[1]

            # Check if tile intersects with crop region.
            intersect_y0 = max(tile_y0, crop_y)
            intersect_y1 = min(tile_y1, crop_y + crop_h)
            intersect_x0 = max(tile_x0, crop_x)
            intersect_x1 = min(tile_x1, crop_x + crop_w)

            if intersect_y1 <= intersect_y0 or intersect_x1 <= intersect_x0:
                continue

            # Extract relevant portion of tile.
            tile_crop_y0 = intersect_y0 - tile_y0
            tile_crop_y1 = intersect_y1 - tile_y0
            tile_crop_x0 = intersect_x0 - tile_x0
            tile_crop_x1 = intersect_x1 - tile_x0

            tile_crop = tile_data[tile_crop_y0:tile_crop_y1, tile_crop_x0:tile_crop_x1]

            # Map to overlay coordinates.
            overlay_y0 = intersect_y0 - crop_y
            overlay_y1 = intersect_y1 - crop_y
            overlay_x0 = intersect_x0 - crop_x
            overlay_x1 = intersect_x1 - crop_x

            # Apply tile-specific color to nuclei.
            tile_color = tile_colors[(r, c)]
            nucleus_mask = tile_crop > 0

            if np.any(nucleus_mask):
                alpha = 0.6  # Semi-transparent for overlapping visualization
                for channel in range(3):
                    overlay[overlay_y0:overlay_y1, overlay_x0:overlay_x1, channel][nucleus_mask] = (
                        (1 - alpha) * overlay[overlay_y0:overlay_y1, overlay_x0:overlay_x1, channel][nucleus_mask] +
                        alpha * tile_color[channel]
                    ).astype(np.uint16)

        except Exception as e:
            logging.error(f"Failed to process original tile ({r},{c}): {e}")
            continue

    logging.info(f"Before merging overlay created with {len(coords)} tiles")
    return overlay.clip(0, 255).astype(np.uint8)


def _create_after_merging_overlay_from_files(
    merged_tiles_dir: Path,
    coords: List[Tuple[int, int]],
    crop_info: Dict,
    tile_h: int,
    tile_w: int,
    overlap: int,
    tissue_background: RGBArray
) -> RGBArray:
    """
    Create after merging overlay using merged tile mask files.

    This function loads merged tile masks from the merged_tile_masks_npz
    directory and creates a visualization showing the final merged results.
    This provides a clear view of how nuclei were merged across tile boundaries.

    Parameters
    ----------
    merged_tiles_dir : Path
        Path to directory containing merged tile mask files.
    coords : List[Tuple[int, int]]
        List of (row, col) tile coordinates.
    crop_info : Dict
        Dictionary containing crop region information.
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    tissue_background : RGBArray
        Background tissue image for overlay.

    Returns
    -------
    RGBArray
        RGB overlay image showing merged tile masks with random colors.
    """
    logging.info("Creating after merging overlay from merged tile files")

    # Initialize overlay with tissue background.
    overlay = tissue_background.astype(np.uint16)

    # Calculate stride and crop parameters.
    stride_h, stride_w = tile_h - overlap, tile_w - overlap
    crop_y, crop_x = crop_info["y_start"], crop_info["x_start"]
    crop_h, crop_w = crop_info["height"], crop_info["width"]

    # Assemble merged mask from individual tiles.
    merged_crop = np.zeros((crop_h, crop_w), dtype=np.uint32)

    for r, c in coords:
        try:
            # Load merged tile mask.
            tile_filename = f"{r}_{c}.npz"
            tile_path = merged_tiles_dir / tile_filename

            if not tile_path.exists():
                logging.warning(f"Merged tile mask not found: {tile_path}")
                continue

            tile_data = np.load(tile_path)["mask"]

            # Calculate tile position in global coordinates.
            tile_y0, tile_x0 = r * stride_h, c * stride_w
            tile_y1 = tile_y0 + tile_data.shape[0]
            tile_x1 = tile_x0 + tile_data.shape[1]

            # Check if tile intersects with crop region.
            intersect_y0 = max(tile_y0, crop_y)
            intersect_y1 = min(tile_y1, crop_y + crop_h)
            intersect_x0 = max(tile_x0, crop_x)
            intersect_x1 = min(tile_x1, crop_x + crop_w)

            if intersect_y1 <= intersect_y0 or intersect_x1 <= intersect_x0:
                continue

            # Extract relevant portion of tile.
            tile_crop_y0 = intersect_y0 - tile_y0
            tile_crop_y1 = intersect_y1 - tile_y0
            tile_crop_x0 = intersect_x0 - tile_x0
            tile_crop_x1 = intersect_x1 - tile_x0

            tile_crop = tile_data[tile_crop_y0:tile_crop_y1, tile_crop_x0:tile_crop_x1]

            # Map to crop coordinates.
            crop_y0 = intersect_y0 - crop_y
            crop_y1 = intersect_y1 - crop_y
            crop_x0 = intersect_x0 - crop_x
            crop_x1 = intersect_x1 - crop_x

            # Place tile data in merged crop (merged tiles should not overlap).
            merged_crop[crop_y0:crop_y1, crop_x0:crop_x1] = tile_crop

        except Exception as e:
            logging.error(f"Failed to process merged tile ({r},{c}): {e}")
            continue

    # Apply random colors to merged nuclei.
    unique_labels = np.unique(merged_crop[merged_crop > 0])
    if len(unique_labels) > 0:
        colors = _generate_random_colors(len(unique_labels))

        for i, label in enumerate(unique_labels):
            nucleus_mask = merged_crop == label
            if np.any(nucleus_mask):
                alpha = 0.7  # More opaque for final results
                for channel in range(3):
                    overlay[nucleus_mask, channel] = (
                        (1 - alpha) * overlay[nucleus_mask, channel] +
                        alpha * colors[i, channel]
                    ).astype(np.uint16)

    logging.info(f"After merging overlay created with {len(unique_labels)} nuclei")
    return overlay.clip(0, 255).astype(np.uint8)


def _generate_random_colors(num_colors: int, seed: int = 42) -> NDArray[np.uint16]:
    """Generate random colors for nucleus visualization."""
    np.random.seed(seed)
    colors = np.zeros((num_colors, 3), dtype=np.uint16)

    for i in range(num_colors):
        # Generate bright, distinguishable colors.
        hue = np.random.random()
        saturation = 0.6 + 0.4 * np.random.random()  # High saturation
        value = 0.7 + 0.3 * np.random.random()  # High brightness

        # Convert HSV to RGB.
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[i] = [r * 255, g * 255, b * 255]

    return colors


"""STATISTICS GENERATION"""

def _generate_merge_statistics(
    merged: NDArray[np.uint32],
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    qc_dir: Path
) -> None:
    """
    Generate quantitative statistics about the segmentation merge results.

    This function creates a comprehensive statistical summary of the segmentation
    results, including nucleus counts, density measurements, and technical details
    about the tiling configuration used during processing.

    Parameters
    ----------
    merged : NDArray[np.uint32]
        Final merged segmentation mask with unique nucleus labels.
    height, width : int
        Full tissue image dimensions in pixels.
    tile_h, tile_w : int
        Individual tile dimensions used during segmentation.
    overlap : int
        Spatial overlap between adjacent tiles in pixels.
    qc_dir : Path
        Output directory for the statistics file.

    Returns
    -------
    None
        Function saves statistics directly to a text file.
    """

    try:
        # Calculate basic segmentation statistics.
        # Count actual unique nuclei, not max label (which can be higher due to merging).
        unique_labels = np.unique(merged[merged > 0])
        total_nuclei = len(unique_labels)
        total_pixels = height * width
        nucleus_pixels = np.count_nonzero(merged)

        # Calculate nucleus density (assuming 1 pixel = 1 micrometer for estimation).
        # This is a rough approximation for kidney tissue analysis.
        area_mm2 = total_pixels / 1e6  # Convert pixels to mm² (rough approximation).
        nucleus_density = total_nuclei / area_mm2 if area_mm2 > 0 else 0

        # Calculate tiling information.
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap
        tiles_per_row = (height + stride_h - 1) // stride_h
        tiles_per_col = (width + stride_w - 1) // stride_w
        total_tiles = tiles_per_row * tiles_per_col

        # Generate comprehensive statistics report.
        stats_path = qc_dir / "merge_statistics.txt"

        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("Kidney I/R Injury Nucleus Segmentation - Merge Statistics\n")
            f.write("=" * 60 + "\n\n")

            f.write("IMAGE INFORMATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Image dimensions: {height} x {width} pixels\n")
            f.write(f"Total image area: {total_pixels:,} pixels\n")
            f.write(f"Estimated tissue area: {area_mm2:.2f} mm²\n\n")

            f.write("TILING CONFIGURATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Tile dimensions: {tile_h} x {tile_w} pixels\n")
            f.write(f"Tile overlap: {overlap} pixels\n")
            f.write(f"Tile stride: {stride_h} x {stride_w} pixels\n")
            f.write(f"Grid dimensions: {tiles_per_row} x {tiles_per_col} tiles\n")
            f.write(f"Total tiles processed: {total_tiles}\n\n")

            f.write("SEGMENTATION RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total nuclei detected: {total_nuclei:,}\n")
            f.write(f"Nucleus pixels: {nucleus_pixels:,}\n")
            f.write(f"Nucleus coverage: {(nucleus_pixels/total_pixels)*100:.2f}%\n")
            f.write(f"Estimated nucleus density: {nucleus_density:.1f} nuclei/mm²\n\n")

            f.write("QUALITY METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average nucleus size: {nucleus_pixels/total_nuclei:.1f} pixels\n")
            f.write(f"Nuclei per tile (average): {total_nuclei/total_tiles:.1f}\n\n")

            f.write("NOTES:\n")
            f.write("-" * 20 + "\n")
            f.write("• Density calculations assume 1 pixel ≈ 1 μm for kidney tissue.\n")
            f.write("• Statistics reflect post-merge results after tile boundary resolution.\n")
            f.write("• QC overlays provide visual validation of merge quality.\n")

        logging.info(f"Generated merge statistics: {total_nuclei:,} nuclei detected.")
        logging.info(f"Saved detailed statistics to: {stats_path}")

    except Exception as stats_error:
        logging.warning(f"Failed to generate merge statistics: {stats_error}")
        logging.debug(f"Statistics error traceback:\n{traceback.format_exc()}")


"""LEGACY COMPATIBILITY FUNCTIONS"""

# The following functions maintain compatibility with existing CLI usage
# while providing the enhanced functionality described above.

def main() -> None:
    """
    Entry point for the qc.py CLI with enhanced functionality.

    This function provides backward compatibility with existing CLI usage
    while incorporating the improved QC visualization features.
    """

    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    try:
        # Load the RGB image for visualization background.
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = _load_rgb_image(image_path)
        logging.info(f"Loaded RGB image: {image.shape}")

        # Load the merged mask for after visualization.
        merged_path = Path(args.merged)
        if not merged_path.exists():
            raise FileNotFoundError(f"Merged mask file not found: {merged_path}")

        merged_mask = _load_mask_file(merged_path)
        logging.info(f"Loaded merged mask: {merged_mask.shape}")

        # Set up output directory.
        outdir = Path(args.outdir or image_path.parent / "qc")
        outdir.mkdir(parents=True, exist_ok=True)

        # Create a simple loader function for CLI usage.
        raw_masks_dir = Path(args.raw_masks)

        def simple_loader(ys: slice, xs: slice) -> NDArray[np.uint32]:
            """Simple loader for CLI compatibility."""
            # This is a simplified version for CLI usage.
            # In practice, the full pipeline provides a more sophisticated loader.
            return np.zeros((ys.stop - ys.start, xs.stop - xs.start), dtype=np.uint32)

        # Generate QC overlays using the enhanced functions.
        write_overlays(
            loader=simple_loader,
            merged=merged_mask,
            height=merged_mask.shape[0],
            width=merged_mask.shape[1],
            tile_h=512,  # Default tile size.
            tile_w=512,
            overlap=64,  # Default overlap.
            qc_dir=outdir
        )

        logging.info(f"QC overlays saved to {outdir.resolve()}")

    except Exception as main_error:
        logging.error(f"QC generation failed: {main_error}")
        logging.debug(f"Main error traceback:\n{traceback.format_exc()}")
        return 1

    return 0


"""UTILITY FUNCTIONS"""

def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser for QC generation.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for CLI usage.
    """

    parser = argparse.ArgumentParser(
        description="Generate enhanced before/after QC overlays for kidney tissue segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the RGB tissue image (TIFF/PNG format)."
    )
    parser.add_argument(
        "--raw_masks",
        required=True,
        help="Directory containing per-tile mask files from segmentation."
    )
    parser.add_argument(
        "--merged",
        required=True,
        help="Path to the merged segmentation mask file."
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for QC overlays (default: next to image file)."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Overlay opacity for alpha blending [0-1]."
    )

    return parser


def _load_rgb_image(path: Path) -> RGBArray:
    """
    Load an RGB tissue image for QC visualization background.

    Parameters
    ----------
    path : Path
        Path to the RGB image file.

    Returns
    -------
    RGBArray
        RGB image array as uint8.

    Raises
    ------
    ValueError
        If the image is not in RGB format.
    """

    try:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {arr.shape}")

        logging.debug(f"Loaded RGB image with shape {arr.shape}")
        return arr

    except Exception as load_error:
        logging.error(f"Failed to load RGB image from {path}: {load_error}")
        raise


def _load_mask_file(path: Path) -> MaskArray:
    """
    Load a segmentation mask from TIFF, PNG, or NPZ format.

    Parameters
    ----------
    path : Path
        Path to the mask file.

    Returns
    -------
    MaskArray
        Segmentation mask as uint32 array.

    Raises
    ------
    ValueError
        If the file format is not supported.
    """

    try:
        if path.suffix.lower() in {".tif", ".tiff", ".png"}:
            arr = np.asarray(Image.open(path), dtype=np.uint32)
        elif path.suffix.lower() == ".npz":
            npz_data = np.load(path)
            # Handle different NPZ file structures.
            if "arr_0" in npz_data:
                arr = npz_data["arr_0"].astype(np.uint32)
            elif "mask" in npz_data:
                arr = npz_data["mask"].astype(np.uint32)
            else:
                # Use the first array in the file.
                arr = npz_data[npz_data.files[0]].astype(np.uint32)
        else:
            raise ValueError(f"Unsupported mask format: {path.suffix}")

        logging.debug(f"Loaded mask with shape {arr.shape}, max label {arr.max()}")
        return arr

    except Exception as load_error:
        logging.error(f"Failed to load mask from {path}: {load_error}")
        raise


"""ENTRY POINT"""

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
