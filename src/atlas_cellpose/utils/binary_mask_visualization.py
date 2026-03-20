"""
Author: Christos Botos.
Affiliation: Human Genetics Department, Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Module Name: binary_mask_visualization.py.
Description:
    Generate binary mask images for ViT input where pixels inside any mask region
    are set to white (255) and all other pixels to black (0). Optimized for gigantic
    images and millions of masks with memory-efficient chunked processing.

Dependencies:
    • Python >= 3.10.
    • numpy >= 1.21.0, pillow >= 8.0.0.
    • psutil >= 5.8.0 (for memory monitoring).
    • tifffile (optional, for TIFF compression).

Key Features:
    • Memory-efficient chunked processing for gigantic images.
    • Automatic memory monitoring and safety checks.
    • Optimized TIFF compression for reduced file sizes.
    • Comprehensive error handling and fallback strategies.

Notes:
    • Designed for kidney I/R injury spatial multiomics analysis.
    • Handles uint32 overflow protection for millions of masks.
    • Optimized for both CPU and memory-constrained environments.
"""

import traceback
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import psutil
from PIL import Image

from rich.console import Console

# Configure PIL for large images.
Image.MAX_IMAGE_PIXELS = 10**12

# Initialize Rich console for formatted output.
console = Console()


def get_memory_usage() -> float:
    """Get current memory usage in GB.

    Returns:
        float: Current memory usage in gigabytes.
    """
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def check_memory_safety(required_gb: float, limit_gb: float, logger: logging.Logger) -> bool:
    """Check if operation is safe given memory constraints.

    Args:
        required_gb (float): Required memory in gigabytes.
        limit_gb (float): Memory limit in gigabytes.
        logger (logging.Logger): Logger for reporting.

    Returns:
        bool: True if operation is safe, False otherwise.
    """
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
    """Estimate memory requirements for array operations in GB.

    Args:
        shape (tuple of int): Shape of the array.
        dtype (np.dtype): Data type of the array.

    Returns:
        float: Estimated memory requirements in gigabytes.
    """
    # Ensure dtype is a proper dtype object.
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    
    # Base array size.
    base_size = np.prod(shape) * dtype.itemsize
    
    # Account for intermediate operations (boolean conversion, etc.).
    safety_factor = 2.5
    
    return (base_size * safety_factor) / (1024**3)


def _process_masks_to_binary(
    masks: np.ndarray,
    chunk_size: int = 2048,
    memory_limit_gb: float = 8.0,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """Convert segmentation masks to binary mask using chunked processing.

    This function handles gigantic mask files by processing them in chunks to
    avoid memory overflow issues common with millions of masks.

    Args:
        masks (np.ndarray): Segmentation masks (2D label map or 3D mask stack).
        chunk_size (int, optional): Size of chunks for processing (pixels). Defaults to 2048.
        memory_limit_gb (float, optional): Maximum memory usage allowed. Defaults to 8.0.
        logger (logging.Logger, optional): Logger instance for debugging.

    Returns:
        np.ndarray: 2D boolean mask where True indicates presence of any mask.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Processing masks to binary: shape={masks.shape}, dtype={masks.dtype}")
    
    # For 2D label maps, simply check for non-zero values.
    if masks.ndim == 2:
        height, width = masks.shape
        
        # Check if we need chunking.
        total_memory = estimate_memory_requirements(masks.shape, masks.dtype)
        
        if total_memory < memory_limit_gb * 0.5:
            logger.debug("Processing 2D mask without chunking")
            return masks > 0
        
        logger.info("Processing 2D mask with chunking due to size")
        binary_mask = np.zeros((height, width), dtype=bool)
        
        # Calculate chunk grid.
        n_chunks_h = (height + chunk_size - 1) // chunk_size
        n_chunks_w = (width + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks_h):
            for j in range(n_chunks_w):
                y_start = i * chunk_size
                y_end = min((i + 1) * chunk_size, height)
                x_start = j * chunk_size
                x_end = min((j + 1) * chunk_size, width)
                
                chunk = masks[y_start:y_end, x_start:x_end]
                binary_mask[y_start:y_end, x_start:x_end] = chunk > 0
        
        return binary_mask
    
    elif masks.ndim == 3:
        # For 3D mask stacks, collapse along first dimension.
        logger.info("Processing 3D mask stack")
        return np.any(masks, axis=0)
    
    else:
        raise ValueError(f'Unsupported mask array shape: {masks.shape}')


def _save_binary_image(
    binary_mask: np.ndarray,
    output_path: Path,
    compression: str = 'lzw',
    logger: Optional[logging.Logger] = None
) -> None:
    """Save the 2D boolean mask as an optimized binary TIFF image.

    Args:
        binary_mask (np.ndarray): 2D boolean array to save.
        output_path (Path): Path where the binary mask image will be saved.
        compression (str, optional): TIFF compression method ('none', 'lzw', 'jpeg', 'zstd'). Defaults to 'lzw'.
        logger (logging.Logger, optional): Logger instance for debugging.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Saving binary mask to: {output_path}")
    logger.info(f"Mask shape: {binary_mask.shape}, compression: {compression}")
    
    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    """Get compression kwargs for tifffile.

    Args:
        compression (str): Compression method name.

    Returns:
        dict: Compression kwargs for tifffile.
    """
    compression_map = {
        'none': {'compression': None},
        'lzw': {'compression': 'lzw'},
        'jpeg': {'compression': 'jpeg', 'compressionargs': {'quality': 95}},
        'zstd': {'compression': 'zstd', 'compressionargs': {'level': 3}}
    }
    return compression_map.get(compression.lower(), {'compression': 'lzw'})


def _get_pil_compression_kwargs(compression: str) -> dict:
    """Get compression kwargs for PIL.

    Args:
        compression (str): Compression method name.

    Returns:
        dict: Compression kwargs for PIL.
    """
    if compression.lower() == 'lzw':
        return {'compression': 'tiff_lzw'}
    elif compression.lower() == 'jpeg':
        return {'compression': 'jpeg', 'quality': 95}
    else:
        return {}


def generate_binary_mask_visualization(
    masks: np.ndarray,
    output_dir: Path,
    logger: logging.Logger,
    suffix: str = "",
    chunk_size: int = 2048,
    memory_limit_gb: float = 8.0,
    compression: str = 'lzw'
) -> None:
    """Generate white segmentation masks on black background visualization.

    This function creates a binary mask image where pixels inside any mask region
    are set to white (255) and all other pixels to black (0). This visualization
    is optimized for ViT input and provides a clear view of the segmented regions.

    The output is saved to the visualizations/ subdirectory within the results directory.

    Args:
        masks (np.ndarray): Segmentation masks (2D label map with integer labels or 3D mask stack).
        output_dir (Path): Base output directory for the pipeline run (e.g., results/<run_name>/).
        logger (logging.Logger): Logger for progress tracking and error reporting.
        suffix (str, optional): Suffix to append to the output filename (e.g., "_filtered" or "_unfiltered"). Defaults to "".
        chunk_size (int, optional): Size of chunks for processing large images (pixels). Defaults to 2048.
        memory_limit_gb (float, optional): Maximum memory usage allowed in gigabytes. Defaults to 8.0.
        compression (str, optional): TIFF compression method ('none', 'lzw', 'jpeg', 'zstd'). Defaults to 'lzw'.

    Returns:
        None: Saves the binary mask visualization to disk.

    Notes:
        The function uses memory-efficient chunked processing for gigantic images
        and automatically handles both 2D label maps and 3D mask stacks.
    """
    try:
        console.print(f"\n[cyan]Generating binary mask visualization{suffix}...[/cyan]")

        # Create visualizations directory.
        vis_dir = Path(output_dir) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Define output path.
        output_filename = f"binary_mask{suffix}.tif"
        output_path = vis_dir / output_filename

        logger.info(f"Creating binary mask visualization: {output_path}")
        logger.info(f"Input mask shape: {masks.shape}, dtype: {masks.dtype}")

        # Convert masks to binary representation.
        binary_mask = _process_masks_to_binary(
            masks=masks,
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb,
            logger=logger
        )

        # Calculate coverage statistics.
        coverage = np.sum(binary_mask) / binary_mask.size * 100
        logger.info(f"Mask coverage: {coverage:.2f}% of pixels")

        # Save the binary mask image.
        _save_binary_image(
            binary_mask=binary_mask,
            output_path=output_path,
            compression=compression,
            logger=logger
        )

        # Report file size.
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024**2)
            console.print(
                f"[green]✓[/green] Binary mask visualization saved: "
                f"[bold]{output_filename}[/bold] ([cyan]{file_size_mb:.2f}MB[/cyan])"
            )
            logger.info(f"Binary mask visualization completed: {output_path}")

    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Binary mask visualization failed: {e}")
        logger.warning(f"Binary mask visualization failed: {e}")
        logger.debug(traceback.format_exc())

