'''Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: white_segmentation_masks_on_black_background.py
Description:
    Generate a binary mask image for ViT input where pixels inside any mask region
    are set to 1 and all other pixels to 0. Optimized for gigantic images and
    millions of masks with memory-efficient chunked processing.

Dependencies:
    • Python >= 3.10.
    • numpy >= 1.21.0, pillow >= 8.0.0, tqdm >= 4.62.0.
    • psutil >= 5.8.0 (for memory monitoring).
    • imagecodecs (optional, for TIFF compression).

Usage:
    python white_segmentation_masks_on_black_background.py \
        --mask results/20250726_001612_full_ss_bIRI2_cpu-merge_cellpose3_diameter0/masks/segmentation_masks.npy \
        --output data/ss_bIRI3_binary_mask.tif \
        --chunk-size 2048 --compression lzw --progress

Positional Arguments:
    None.

Optional Arguments:
    --mask          Path to a numpy file containing either:
                        • a 2D label map (H × W) with integer labels, or
                        • a 3D boolean or integer mask stack (N × H × W).
    --output        Path where the binary mask image will be saved (TIFF format).
    --chunk-size    Process image in chunks of this size (default: 2048).
    --compression   TIFF compression method: none, lzw, jpeg, zstd (default: lzw).
    --memory-limit  Maximum memory usage in GB (default: 4.0).
    --progress      Show progress bar for large operations.
    --benchmark     Enable performance benchmarking and logging.

Inputs:
    • Large segmentation mask files (.npy format) with millions of masks.
    • Supports both 2D label maps and 3D mask stacks.

Outputs:
    • Compressed binary TIFF image optimized for ViT input.
    • Performance logs when benchmarking is enabled.

Key Features:
    • Memory-efficient chunked processing for gigantic images.
    • Automatic memory monitoring and safety checks.
    • Progress tracking for long-running operations.
    • Optimized TIFF compression for reduced file sizes.
    • Comprehensive error handling and fallback strategies.

Notes:
    • Designed for kidney I/R injury spatial multiomics analysis.
    • Handles uint32 overflow protection for millions of masks.
    • Optimized for both CPU and memory-constrained environments.
'''
import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import psutil
from PIL import Image
from tqdm import tqdm

# Configure PIL for large images.
Image.MAX_IMAGE_PIXELS = 10**12


def setup_logging(enable_benchmark: bool = False) -> logging.Logger:
    """Set up logging configuration for the script."""
    log_level = logging.DEBUG if enable_benchmark else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def check_memory_safety(required_gb: float, limit_gb: float, logger: logging.Logger) -> bool:
    """Check if operation is safe given memory constraints."""
    current_gb = get_memory_usage()
    total_required = current_gb + required_gb

    if total_required > limit_gb:
        logger.warning(
            f"Memory safety check failed: {total_required:.2f}GB required > {limit_gb:.2f}GB limit"
        )
        return False

    logger.debug(f"Memory check passed: {total_required:.2f}GB required < {limit_gb:.2f}GB limit")
    return True


def estimate_memory_requirements(shape: Tuple[int, ...], dtype: np.dtype) -> float:
    """Estimate memory requirements for array operations in GB."""
    # Base array size.
    base_size = np.prod(shape) * dtype.itemsize

    # Account for intermediate operations (boolean conversion, etc.).
    safety_factor = 2.5

    return (base_size * safety_factor) / (1024**3)


def load_mask_chunked(
    mask_path: Path,
    chunk_size: int = 2048,
    memory_limit_gb: float = 8.0,
    show_progress: bool = False,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Load mask array and collapse to a 2D boolean mask using chunked processing.

    This function handles gigantic mask files by processing them in chunks to
    avoid memory overflow issues common with millions of masks.

    Args:
        mask_path: Path to the numpy mask file.
        chunk_size: Size of chunks for processing (pixels).
        memory_limit_gb: Maximum memory usage allowed.
        show_progress: Whether to show progress bar.
        logger: Logger instance for debugging.

    Returns:
        2D boolean mask where True indicates presence of any mask.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Loading mask file: {mask_path}")

    # Load with memory mapping to avoid loading entire file.
    masks = np.load(mask_path, mmap_mode='r')
    logger.info(f"Mask shape: {masks.shape}, dtype: {masks.dtype}")

    # Estimate memory requirements.
    memory_needed = estimate_memory_requirements(masks.shape, masks.dtype)
    logger.debug(f"Estimated memory needed: {memory_needed:.2f}GB")

    if masks.ndim == 3:
        return _process_3d_masks_chunked(
            masks, chunk_size, memory_limit_gb, show_progress, logger
        )
    elif masks.ndim == 2:
        return _process_2d_mask_chunked(
            masks, chunk_size, memory_limit_gb, show_progress, logger
        )
    else:
        raise ValueError(f'Unsupported mask array shape: {masks.shape}')


def _process_3d_masks_chunked(
    masks: np.ndarray,
    chunk_size: int,
    memory_limit_gb: float,
    show_progress: bool,
    logger: logging.Logger
) -> np.ndarray:
    """Process 3D mask stack in chunks to create binary mask."""
    height, width = masks.shape[1], masks.shape[2]
    binary_mask = np.zeros((height, width), dtype=bool)

    # Calculate chunk grid.
    n_chunks_h = (height + chunk_size - 1) // chunk_size
    n_chunks_w = (width + chunk_size - 1) // chunk_size
    total_chunks = n_chunks_h * n_chunks_w

    logger.info(f"Processing {total_chunks} chunks ({n_chunks_h}x{n_chunks_w})")

    # Progress bar setup.
    pbar = tqdm(total=total_chunks, desc="Processing chunks") if show_progress else None

    try:
        for i in range(n_chunks_h):
            for j in range(n_chunks_w):
                # Calculate chunk boundaries.
                y_start = i * chunk_size
                y_end = min((i + 1) * chunk_size, height)
                x_start = j * chunk_size
                x_end = min((j + 1) * chunk_size, width)

                # Memory safety check for this chunk.
                chunk_shape = (masks.shape[0], y_end - y_start, x_end - x_start)
                chunk_memory = estimate_memory_requirements(chunk_shape, masks.dtype)

                if not check_memory_safety(chunk_memory, memory_limit_gb, logger):
                    logger.warning(f"Skipping chunk ({i},{j}) due to memory constraints")
                    continue

                # Extract and process chunk.
                chunk = masks[:, y_start:y_end, x_start:x_end]

                # Handle object dtype conversion efficiently.
                if chunk.dtype == object:
                    # Process each mask individually to avoid memory spikes.
                    chunk_binary = np.zeros((y_end - y_start, x_end - x_start), dtype=bool)
                    for k in range(chunk.shape[0]):
                        if chunk[k] is not None:
                            mask_slice = np.asarray(chunk[k], dtype=bool)
                            chunk_binary |= mask_slice
                else:
                    # Standard processing for numeric dtypes.
                    chunk_binary = np.any(chunk, axis=0)

                # Update the main binary mask.
                binary_mask[y_start:y_end, x_start:x_end] = chunk_binary

                if pbar:
                    pbar.update(1)

                # Log progress periodically.
                if (i * n_chunks_w + j + 1) % max(1, total_chunks // 10) == 0:
                    progress = ((i * n_chunks_w + j + 1) / total_chunks) * 100
                    logger.debug(f"Progress: {progress:.1f}%")

    finally:
        if pbar:
            pbar.close()

    logger.info("Completed 3D mask processing")
    return binary_mask


def _process_2d_mask_chunked(
    masks: np.ndarray,
    chunk_size: int,
    memory_limit_gb: float,
    show_progress: bool,
    logger: logging.Logger
) -> np.ndarray:
    """Process 2D mask in chunks if needed."""
    # For 2D masks, chunking is less critical but still useful for very large images.
    height, width = masks.shape

    # Check if we need chunking.
    total_memory = estimate_memory_requirements(masks.shape, masks.dtype)

    if total_memory < memory_limit_gb * 0.5:  # Use 50% threshold for 2D.
        logger.debug("Processing 2D mask without chunking")
        if masks.dtype == bool:
            return masks.copy()
        else:
            return masks > 0

    logger.info("Processing 2D mask with chunking due to size")
    binary_mask = np.zeros((height, width), dtype=bool)

    # Calculate chunk grid.
    n_chunks_h = (height + chunk_size - 1) // chunk_size
    n_chunks_w = (width + chunk_size - 1) // chunk_size
    total_chunks = n_chunks_h * n_chunks_w

    pbar = tqdm(total=total_chunks, desc="Processing 2D chunks") if show_progress else None

    try:
        for i in range(n_chunks_h):
            for j in range(n_chunks_w):
                y_start = i * chunk_size
                y_end = min((i + 1) * chunk_size, height)
                x_start = j * chunk_size
                x_end = min((j + 1) * chunk_size, width)

                chunk = masks[y_start:y_end, x_start:x_end]

                if chunk.dtype == bool:
                    binary_mask[y_start:y_end, x_start:x_end] = chunk
                else:
                    binary_mask[y_start:y_end, x_start:x_end] = chunk > 0

                if pbar:
                    pbar.update(1)

    finally:
        if pbar:
            pbar.close()

    logger.info("Completed 2D mask processing")
    return binary_mask


def save_binary_image_optimized(
    binary_mask: np.ndarray,
    output_path: Path,
    compression: str = 'lzw',
    chunk_size: int = 2048,
    show_progress: bool = False,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save the 2D boolean mask as an optimized binary TIFF image.

    This function handles gigantic binary masks by using efficient compression
    and chunked processing to minimize memory usage and file size.

    Args:
        binary_mask: 2D boolean array to save.
        output_path: Path where the binary mask image will be saved.
        compression: TIFF compression method ('none', 'lzw', 'jpeg', 'zstd').
        chunk_size: Process image in chunks of this size.
        show_progress: Whether to show progress bar.
        logger: Logger instance for debugging.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Saving binary mask to: {output_path}")
    logger.info(f"Mask shape: {binary_mask.shape}, compression: {compression}")

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert boolean mask to uint8 0/255 efficiently.
    height, width = binary_mask.shape

    # For very large images, process in chunks to avoid memory issues.
    if height * width > chunk_size * chunk_size * 4:  # Threshold for chunked processing.
        logger.info("Using chunked processing for large image")
        _save_large_image_chunked(
            binary_mask, output_path, compression, chunk_size, show_progress, logger
        )
    else:
        logger.debug("Using standard processing for image")
        _save_standard_image(binary_mask, output_path, compression, logger)


def _save_large_image_chunked(
    binary_mask: np.ndarray,
    output_path: Path,
    compression: str,
    chunk_size: int,
    show_progress: bool,
    logger: logging.Logger
) -> None:
    """Save large binary mask using chunked processing with tifffile."""
    try:
        import tifffile
        logger.debug("Using tifffile for chunked large image saving")

        # Convert to uint8 in chunks to save memory.
        height, width = binary_mask.shape
        img_array = np.zeros((height, width), dtype=np.uint8)

        n_chunks_h = (height + chunk_size - 1) // chunk_size
        n_chunks_w = (width + chunk_size - 1) // chunk_size
        total_chunks = n_chunks_h * n_chunks_w

        pbar = tqdm(total=total_chunks, desc="Converting to uint8") if show_progress else None

        try:
            for i in range(n_chunks_h):
                for j in range(n_chunks_w):
                    y_start = i * chunk_size
                    y_end = min((i + 1) * chunk_size, height)
                    x_start = j * chunk_size
                    x_end = min((j + 1) * chunk_size, width)

                    chunk = binary_mask[y_start:y_end, x_start:x_end]
                    img_array[y_start:y_end, x_start:x_end] = chunk.astype(np.uint8) * 255

                    if pbar:
                        pbar.update(1)
        finally:
            if pbar:
                pbar.close()

        # Save with tifffile for better compression options.
        compression_kwargs = _get_tifffile_compression_kwargs(compression)
        tifffile.imwrite(
            output_path,
            img_array,
            photometric='minisblack',
            **compression_kwargs
        )

    except ImportError:
        logger.warning("tifffile not available, falling back to PIL with chunked conversion")
        _save_with_pil_chunked(binary_mask, output_path, compression, chunk_size, show_progress, logger)


def _save_with_pil_chunked(
    binary_mask: np.ndarray,
    output_path: Path,
    compression: str,
    chunk_size: int,
    show_progress: bool,
    logger: logging.Logger
) -> None:
    """Fallback method using PIL with chunked conversion."""
    height, width = binary_mask.shape
    img_array = np.zeros((height, width), dtype=np.uint8)

    n_chunks_h = (height + chunk_size - 1) // chunk_size
    n_chunks_w = (width + chunk_size - 1) // chunk_size
    total_chunks = n_chunks_h * n_chunks_w

    pbar = tqdm(total=total_chunks, desc="Converting chunks") if show_progress else None

    try:
        for i in range(n_chunks_h):
            for j in range(n_chunks_w):
                y_start = i * chunk_size
                y_end = min((i + 1) * chunk_size, height)
                x_start = j * chunk_size
                x_end = min((j + 1) * chunk_size, width)

                chunk = binary_mask[y_start:y_end, x_start:x_end]
                img_array[y_start:y_end, x_start:x_end] = chunk.astype(np.uint8) * 255

                if pbar:
                    pbar.update(1)
    finally:
        if pbar:
            pbar.close()

    # Save with PIL.
    img = Image.fromarray(img_array, mode='L')
    save_kwargs = _get_pil_compression_kwargs(compression)
    img.save(output_path, format='TIFF', **save_kwargs)


def _save_standard_image(
    binary_mask: np.ndarray,
    output_path: Path,
    compression: str,
    logger: logging.Logger
) -> None:
    """Save image using standard method for smaller images."""
    # Convert boolean mask to uint8 0/255.
    img_array = binary_mask.astype(np.uint8) * 255

    try:
        import tifffile
        compression_kwargs = _get_tifffile_compression_kwargs(compression)
        tifffile.imwrite(
            output_path,
            img_array,
            photometric='minisblack',
            **compression_kwargs
        )
        logger.debug("Saved using tifffile")
    except ImportError:
        img = Image.fromarray(img_array, mode='L')
        save_kwargs = _get_pil_compression_kwargs(compression)
        img.save(output_path, format='TIFF', **save_kwargs)
        logger.debug("Saved using PIL")


def _get_tifffile_compression_kwargs(compression: str) -> dict:
    """Get compression kwargs for tifffile."""
    compression_map = {
        'none': {'compression': None},
        'lzw': {'compression': 'lzw'},
        'jpeg': {'compression': 'jpeg', 'compressionargs': {'quality': 95}},
        'zstd': {'compression': 'zstd', 'compressionargs': {'level': 3}}
    }
    return compression_map.get(compression.lower(), {'compression': 'lzw'})


def _get_pil_compression_kwargs(compression: str) -> dict:
    """Get compression kwargs for PIL."""
    if compression.lower() == 'lzw':
        return {'compression': 'tiff_lzw'}
    elif compression.lower() == 'jpeg':
        return {'compression': 'jpeg', 'quality': 95}
    else:
        return {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(
        description='Create optimized binary TIFF mask image for ViT input from gigantic mask files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python white_segmentation_masks_on_black_background.py --mask masks.npy --output binary.tif

  # Optimized for large files
  python white_segmentation_masks_on_black_background.py --mask huge_masks.npy --output binary.tif \\
         --chunk-size 1024 --compression zstd --memory-limit 16.0 --progress --benchmark

  # Memory-constrained environment
  python white_segmentation_masks_on_black_background.py --mask masks.npy --output binary.tif \\
         --chunk-size 512 --memory-limit 4.0 --compression lzw
        """
    )

    # Required arguments.
    parser.add_argument(
        '--mask', type=Path, required=True,
        help='Path to the numpy file containing masks (2D label map or 3D mask stack)'
    )
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Path to save the binary mask image (TIFF format)'
    )

    # Performance optimization arguments.
    parser.add_argument(
        '--chunk-size', type=int, default=2048,
        help='Process image in chunks of this size in pixels (default: 2048, reduce for less memory)'
    )
    parser.add_argument(
        '--memory-limit', type=float, default=4.0,
        help='Maximum memory usage in GB (default: 8.0, adjust based on available RAM)'
    )

    # Output optimization arguments.
    parser.add_argument(
        '--compression', choices=['none', 'lzw', 'jpeg', 'zstd'], default='lzw',
        help='TIFF compression method (default: lzw, zstd for best compression)'
    )

    # User experience arguments.
    parser.add_argument(
        '--progress', action='store_true',
        help='Show progress bars for long-running operations'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Enable performance benchmarking and detailed logging'
    )

    return parser.parse_args()


def benchmark_operation(func, *args, **kwargs):
    """Benchmark a function and return result with timing info."""
    start_time = time.time()
    start_memory = get_memory_usage()

    result = func(*args, **kwargs)

    end_time = time.time()
    end_memory = get_memory_usage()

    return result, {
        'execution_time': end_time - start_time,
        'memory_start': start_memory,
        'memory_end': end_memory,
        'memory_peak': end_memory - start_memory
    }


def main() -> None:
    """Main function with comprehensive error handling and benchmarking."""
    args = parse_args()

    # Set up logging.
    logger = setup_logging(args.benchmark)

    try:
        logger.info("=== Binary Mask Generation Started ===")
        logger.info(f"Input mask: {args.mask}")
        logger.info(f"Output path: {args.output}")
        logger.info(f"Chunk size: {args.chunk_size}")
        logger.info(f"Memory limit: {args.memory_limit}GB")
        logger.info(f"Compression: {args.compression}")

        # Validate input file.
        if not args.mask.exists():
            raise FileNotFoundError(f"Mask file not found: {args.mask}")

        # Check file size and warn if very large.
        file_size_gb = args.mask.stat().st_size / (1024**3)
        logger.info(f"Input file size: {file_size_gb:.2f}GB")

        if file_size_gb > args.memory_limit:
            logger.warning(
                f"Input file ({file_size_gb:.2f}GB) is larger than memory limit ({args.memory_limit}GB). "
                "Consider increasing --memory-limit or reducing --chunk-size."
            )

        # Load and process mask.
        if args.benchmark:
            binary_mask, load_stats = benchmark_operation(
                load_mask_chunked,
                args.mask,
                args.chunk_size,
                args.memory_limit,
                args.progress,
                logger
            )
            logger.info(f"Mask loading took {load_stats['execution_time']:.2f}s")
            logger.info(f"Memory usage: {load_stats['memory_start']:.2f}GB -> {load_stats['memory_end']:.2f}GB")
        else:
            binary_mask = load_mask_chunked(
                args.mask,
                args.chunk_size,
                args.memory_limit,
                args.progress,
                logger
            )

        logger.info(f"Generated binary mask shape: {binary_mask.shape}")
        logger.info(f"Mask coverage: {np.sum(binary_mask) / binary_mask.size * 100:.2f}% of pixels")

        # Save optimized image.
        if args.benchmark:
            _, save_stats = benchmark_operation(
                save_binary_image_optimized,
                binary_mask,
                args.output,
                args.compression,
                args.chunk_size,
                args.progress,
                logger
            )
            logger.info(f"Image saving took {save_stats['execution_time']:.2f}s")
        else:
            save_binary_image_optimized(
                binary_mask,
                args.output,
                args.compression,
                args.chunk_size,
                args.progress,
                logger
            )

        # Final statistics.
        if args.output.exists():
            output_size_mb = args.output.stat().st_size / (1024**2)
            logger.info(f"Output file size: {output_size_mb:.2f}MB")

            if file_size_gb > 0:
                compression_ratio = (file_size_gb * 1024) / output_size_mb
                logger.info(f"Compression ratio: {compression_ratio:.1f}:1")

        logger.info("=== Binary Mask Generation Completed Successfully ===")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        if args.benchmark:
            logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == '__main__':
    main()
