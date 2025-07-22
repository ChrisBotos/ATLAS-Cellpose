"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: merge_tiles_clean.py.
Description:
    Clean implementation of merge_masks_streaming that only uses the two-phase
    merging strategy. This replaces the complex cluster-based approach with
    a simpler, more reliable method.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pillow, tqdm.
    • cellpose_merge.two_phase_merge module.

Key Features:
    • Two-phase merging strategy only (no cluster-based fallback).
    • Reduced verbose logging (debug mode controlled).
    • Clean, maintainable code structure.
"""

from __future__ import annotations

import logging
import math
import re
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None


def merge_masks_streaming_clean(
    *,
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    tiles_path: str | Path,
    threshold: float = 0.3,
    use_gpu: bool = True,
    qc: bool = False,
    qc_dir: str | Path | None = None,
    qc_merge_use_full_image: bool = False,
    merge_batch_size: int = 4,
    output_dir: str | Path | None = None,
    debug_mode: bool = False,
) -> NDArray[np.uint32]:
    """
    Merge per‑tile nucleus‑instance masks into a full‑slide label map using two-phase merging.

    This function uses only the two-phase merging strategy, which processes vertical
    overlaps first, then horizontal overlaps, ensuring systematic merge rule
    application and better handling of cross-boundary nuclei.

    Parameters
    ----------
    height, width : int
        Dimensions of the full image in pixels.
    tile_h, tile_w : int
        Dimensions of each tile in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    tiles_path : str | Path
        Path to directory containing tile mask files.
    threshold : float, default 0.3
        Overlap threshold for merging nuclei across tile boundaries.
    use_gpu : bool, default True
        Whether to use GPU acceleration for merging operations.
    qc : bool, default False
        Whether to generate quality control overlays.
    qc_dir : str | Path | None, default None
        Directory to save QC overlays (required if qc=True).
    qc_merge_use_full_image : bool, default False
        Whether to use full image for QC overlays or center crop.
    merge_batch_size : int, default 4
        Number of tile pairs to process in parallel during each phase.
    output_dir : str | Path | None, default None
        Directory to save merged masks (defaults to tiles_path parent).
    debug_mode : bool, default False
        Whether to enable verbose debug logging.

    Returns
    -------
    NDArray[np.uint32]
        Merged mask with unique nucleus labels.
    """
    from .two_phase_merge import merge_tiles_two_phase

    # Set up paths.
    path = Path(tiles_path)
    if output_dir is None:
        output_dir = path.parent / "results"
    else:
        output_dir = Path(output_dir)

    # Create output directories.
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Set up output file paths.
    final_merged_path = masks_dir / "segmentation_masks.npy"
    temp_merged_path = masks_dir / "segmentation_masks_temp.npy"

    # ------------------------------------------------------------------
    # Tile discovery and coordinate mapping.
    # ------------------------------------------------------------------

    # Look for tile mask files in multiple possible directories.
    search_paths = [path, path / "tile_masks", path / "tile_masks_npz"]
    file_map: Dict[Tuple[int, int], Path] = {}

    for search_path in search_paths:
        if not search_path.exists():
            continue

        # Look for different file patterns.
        patterns = [
            (r"(\d+)_(\d+)\.npz", "tile_index"),
            (r"(\d+)_(\d+)\.tif", "tile_index"),
            (r"(\d+) (\d+)\.npz", "pixel_coord"),
            (r"(\d+) (\d+)\.tif", "pixel_coord"),
        ]

        for pattern, coord_type in patterns:
            for file_path in search_path.glob("*"):
                if file_path.is_file():
                    match = re.match(pattern, file_path.name)
                    if match:
                        r, c = int(match.group(1)), int(match.group(2))
                        if coord_type == "pixel_coord":
                            # Convert pixel coordinates to tile indices.
                            stride_h, stride_w = tile_h - overlap, tile_w - overlap
                            r, c = r // stride_h, c // stride_w
                        file_map[(r, c)] = file_path

        if file_map:
            break

    if not file_map:
        raise FileNotFoundError(
            f"None of the following directories contains tile masks: {search_paths}"
        )

    raw_coords = list(file_map.keys())
    logging.info(f"Found {len(raw_coords)} tile mask files in {search_paths}")

    # Calculate stride for coordinate conversion.
    stride_h, stride_w = tile_h - overlap, tile_w - overlap

    if debug_mode:
        min_r = min(r for r, _ in raw_coords)
        max_r = max(r for r, _ in raw_coords)
        min_c = min(c for _, c in raw_coords)
        max_c = max(c for _, c in raw_coords)
        logging.debug(f"Raw coordinate range: rows {min_r} to {max_r}, cols {min_c} to {max_c}")
        logging.debug(f"First few coordinates: {sorted(raw_coords)[:5]}")

        # Calculate expected tile counts for comparison.
        expected_tiles_r = math.ceil(height / stride_h)
        expected_tiles_c = math.ceil(width / stride_w)
        logging.debug(f"Expected tile counts: {expected_tiles_r} rows x {expected_tiles_c} cols")
        logging.debug(f"Stride: {stride_h}x{stride_w}, Image: {height}x{width}")

    # Convert pixel coordinates to tile indices if necessary.
    max_raw_r = max(r for r, _ in raw_coords)
    max_raw_c = max(c for _, c in raw_coords)
    expected_tiles_r = math.ceil(height / stride_h)
    expected_tiles_c = math.ceil(width / stride_w)

    # If raw coordinates are much larger than expected tile counts, assume pixel coordinates.
    if max_raw_r > expected_tiles_r * 2 or max_raw_c > expected_tiles_c * 2:
        # Convert pixel coordinates to tile indices.
        coords = [(r // stride_h, c // stride_w) for r, c in raw_coords]
        # Create new file mapping with tile indices.
        idx_to_path = {(r // stride_h, c // stride_w): file_map[(r, c)] for r, c in raw_coords}

        if debug_mode:
            logging.debug(f"Converted pixel coordinates to tile indices")
            logging.debug(f"Converted coordinates: {sorted(coords)[:5]}")
    else:
        # Assume they're already tile indices.
        coords = raw_coords
        idx_to_path = file_map
        if debug_mode:
            logging.debug("Using coordinates as tile indices")

    # ------------------------------------------------------------------
    # Create tile loader function.
    # ------------------------------------------------------------------

    def _loader(ys: slice, xs: slice) -> NDArray[np.uint32]:
        """Load tile mask data for the specified region."""
        stride_h, stride_w = tile_h - overlap, tile_w - overlap

        # Handle slice bounds properly.
        y_start = ys.start if ys.start is not None else 0
        y_stop = ys.stop if ys.stop is not None else height
        x_start = xs.start if xs.start is not None else 0
        x_stop = xs.stop if xs.stop is not None else width

        # Determine which tiles overlap with this region.
        r_start = max(0, y_start // stride_h - 1)
        r_end = min(max(r for r, _ in coords) + 1, (y_stop - 1) // stride_h + 2)
        c_start = max(0, x_start // stride_w - 1)
        c_end = min(max(c for _, c in coords) + 1, (x_stop - 1) // stride_w + 2)

        # Create output array.
        region_h = y_stop - y_start
        region_w = x_stop - x_start

        # Ensure positive dimensions.
        if region_h <= 0 or region_w <= 0:
            if debug_mode:
                logging.debug(f"Invalid region dimensions: {region_h}x{region_w} for slice y={y_start}:{y_stop}, x={x_start}:{x_stop}")
            return np.zeros((max(1, region_h), max(1, region_w)), dtype=np.uint32)

        result = np.zeros((region_h, region_w), dtype=np.uint32)

        # Load and place each overlapping tile.
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if (r, c) not in idx_to_path:
                    continue

                # Load tile data.
                tile_path = idx_to_path[(r, c)]
                if tile_path.suffix == ".npz":
                    tile_data = np.load(tile_path)["mask"]
                else:
                    tile_data = np.array(Image.open(tile_path))

                # Calculate tile position in global coordinates.
                tile_y0, tile_x0 = r * stride_h, c * stride_w
                tile_y1 = min(tile_y0 + tile_h, height)
                tile_x1 = min(tile_x0 + tile_w, width)

                # Calculate intersection with requested region.
                intersect_y0 = max(tile_y0, y_start)
                intersect_y1 = min(tile_y1, y_stop)
                intersect_x0 = max(tile_x0, x_start)
                intersect_x1 = min(tile_x1, x_stop)

                if intersect_y1 <= intersect_y0 or intersect_x1 <= intersect_x0:
                    continue

                # Copy tile data to result.
                tile_src_y0 = intersect_y0 - tile_y0
                tile_src_y1 = intersect_y1 - tile_y0
                tile_src_x0 = intersect_x0 - tile_x0
                tile_src_x1 = intersect_x1 - tile_x0

                result_dst_y0 = intersect_y0 - y_start
                result_dst_y1 = intersect_y1 - y_start
                result_dst_x0 = intersect_x0 - x_start
                result_dst_x1 = intersect_x1 - x_start

                result[result_dst_y0:result_dst_y1, result_dst_x0:result_dst_x1] = \
                    tile_data[tile_src_y0:tile_src_y1, tile_src_x0:tile_src_x1]

        return result

    # ------------------------------------------------------------------
    # Execute two-phase merging.
    # ------------------------------------------------------------------

    logging.info("Starting two-phase merging strategy")

    try:
        merged = merge_tiles_two_phase(
            coords=coords,
            loader=_loader,
            height=height,
            width=width,
            tile_h=tile_h,
            tile_w=tile_w,
            overlap=overlap,
            threshold=threshold,
            use_gpu=use_gpu,
            merge_batch_size=merge_batch_size,
            gid_offset=1,  # Start global IDs from 1
            debug_mode=debug_mode,
        )

        # Save the final merged mask.
        np.save(temp_merged_path, merged)

        # Rename temp file to final file name.
        if final_merged_path.exists():
            final_merged_path.unlink()
        temp_merged_path.rename(final_merged_path)

        # Also save as TIFF for compatibility.
        try:
            from skimage import io as skio
            tif_path = final_merged_path.parent / "segmentation_masks.tif"
            skio.imsave(tif_path, merged.astype(np.uint32), plugin="tifffile")
            if debug_mode:
                logging.debug(f"Successfully created TIFF mask: {tif_path}")
        except Exception as e:
            logging.warning(f"Could not write TIFF mask: {e}")

        logging.info(f"Two-phase merge completed successfully: {final_merged_path}")

        # Generate QC overlays if requested.
        if qc and qc_dir is not None:
            try:
                from .qc import write_overlays

                # Create QC directory if it doesn't exist.
                qc_path = Path(qc_dir)
                qc_path.mkdir(parents=True, exist_ok=True)

                logging.info(f"Generating QC overlays in: {qc_path}")

                write_overlays(
                    loader=_loader,
                    merged=merged,
                    height=height,
                    width=width,
                    tile_h=tile_h,
                    tile_w=tile_w,
                    overlap=overlap,
                    qc_dir=qc_path,
                    use_full_image=qc_merge_use_full_image,
                )

                logging.info("QC overlays generated successfully")

            except ImportError as e:
                logging.warning(f"QC overlays requested but qc module not found: {e}. Skipping QC generation.")
            except Exception as e:
                logging.error(f"Failed to generate QC overlays: {e}")
                logging.debug(f"QC error traceback:\n{traceback.format_exc()}")

        return merged

    except Exception as e:
        logging.error(f"Two-phase merge failed: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Two-phase merge failed: {e}") from e
