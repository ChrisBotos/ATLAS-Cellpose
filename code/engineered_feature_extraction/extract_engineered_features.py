#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: extract_engineered_features.py.
Description:
    Simplified config-driven extraction of comprehensive nuclear morphological features from segmented
    DAPI-stained tissue sections. Implements a streamlined interface that loads all parameters from a
    configuration file, ensuring complete reproducibility and parameter traceability. Features are
    organized into four distinct categories: shape, size, neighborhood, and texture features for
    quantitative analysis of kidney ischemia-reperfusion injury.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scipy, scikit-image, scikit-learn, Pillow, typer, traceback.
    • Custom utilities from nuclei_segmentation package.

Usage:
    python extract_engineered_features.py extract \
        --config ../../configs/engineered_feature_extraction_config.ini

Arguments:
    --config            Path to configuration file containing all extraction parameters.

Configuration File Requirements:
    The configuration file must contain all necessary parameters including:
    • Input file paths (image, mask)
    • Output directory path
    • Feature extraction settings (categories, workers, radius)
    • Performance optimization parameters

Inputs (specified in config file):
    • extraction_image_path    Path to DAPI-stained kidney tissue image.
    • extraction_mask_path     Path to segmentation mask with labeled nuclei.

Outputs:
    • extract_engineered_features_config_used.ini    Copied configuration for audit trail.
    • engineered_features.csv                        Comprehensive nuclear features by category.
    • Debug logs showing feature extraction progress and statistics.

Key Features:
    • Simplified config-driven interface with complete parameter traceability.
    • Automatic configuration file copying for reproducibility audit trail.
    • Configurable feature categories: shape, size, neighborhood, and texture features.
    • Shape features: circularity, eccentricity, solidity, convex hull properties, compactness.
    • Size features: area, perimeter, equivalent diameter, axis lengths, bounding box dimensions.
    • Neighborhood features: spatial clustering, nearest neighbor analysis, tissue organization.
    • Texture features: intensity statistics, GLCM properties, local binary patterns, gradients.
    • Parallel processing with memory-efficient batch processing for large tissue sections.
    • Scientific context: optimized for kidney I/R injury analysis and nuclear morphology changes.

Notes:
    • All parameters loaded from configuration file - no hardcoded values.
    • Configuration file automatically copied to output directory for audit purposes.
    • Features are specifically selected for analyzing apoptosis, necrosis, and tissue repair.
    • Fractal dimension calculated via box-counting method for geometric complexity analysis.
    • Sparse zones identified as large low-intensity background regions for spatial context.
    • All features include comprehensive DEBUG logging for scientific reproducibility.
"""

import traceback
import sys
import os
from pathlib import Path
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional
import warnings
import time
import gc
import psutil
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import typer
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import skew, kurtosis, entropy
from sklearn.decomposition import PCA
from skimage.measure import regionprops, label
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.morphology import convex_hull_image

# GPU acceleration imports (optional).
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    import cupyx.scipy.spatial as cp_spatial
    GPU_AVAILABLE = True
    _GPU_MESSAGE = "[green]✓[/green] GPU acceleration available with CuPy"
except ImportError:
    cp = None
    cp_ndimage = None
    cp_spatial = None
    GPU_AVAILABLE = False
    _GPU_MESSAGE = "[yellow]⚠[/yellow] GPU acceleration not available. Install CuPy for better performance."

# Progress tracking imports.
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add project root to path for imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import feature extraction configuration utilities.
from code.engineered_feature_extraction.utils.config_loader import load_feature_extraction_config

# Initialize Typer app for CLI.
app = typer.Typer(help="Extract comprehensive nuclear features for kidney I/R injury analysis.")

# Initialize Rich console for beautiful output.
console = Console()

# Print GPU availability status after console initialization.
console.print(_GPU_MESSAGE)

# Configure logging for scientific reproducibility.
# Note: File handler will be configured dynamically based on output directory.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


"""PERFORMANCE OPTIMIZATION UTILITIES"""

@dataclass
class ProcessingStats:
    """
    Data class for tracking processing performance statistics.

    This class provides comprehensive performance monitoring for feature extraction,
    enabling identification of bottlenecks and optimization opportunities.
    """
    total_nuclei: int = 0
    processed_nuclei: int = 0
    failed_nuclei: int = 0
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    features_per_second: float = 0.0
    gpu_operations_count: int = 0
    total_batches: int = 0
    completed_batches: int = 0

    def update_stats(self, processing_time: float, gpu_memory_mb: float = 0.0) -> None:
        """Update processing statistics with current timing information."""
        self.processing_time = processing_time
        self.features_per_second = self.processed_nuclei / processing_time if processing_time > 0 else 0.0

        # Update memory usage.
        import psutil
        process = psutil.Process()
        self.memory_usage_mb = process.memory_info().rss / 1024 / 1024

        # Update GPU memory (passed from external tracking).
        self.gpu_memory_mb = gpu_memory_mb


@dataclass
class ParameterTiming:
    """
    Data class for tracking individual parameter computation times.

    This class provides detailed timing information for each feature parameter,
    enabling identification of computational bottlenecks and optimization opportunities.
    """
    parameter_name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    category: str = ""

    def add_timing(self, time_taken: float):
        """Add a new timing measurement for this parameter."""
        self.total_time += time_taken
        self.call_count += 1
        self.min_time = min(self.min_time, time_taken)
        self.max_time = max(self.max_time, time_taken)
        self.avg_time = self.total_time / self.call_count


# Global dictionary to track parameter timings.
_parameter_timings: Dict[str, ParameterTiming] = {}


def time_parameter(parameter_name: str, category: str = ""):
    """
    Decorator to time individual parameter calculations.

    Args:
        parameter_name: Name of the parameter being timed.
        category: Feature category (shape, size, neighborhood, texture).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                time_taken = end_time - start_time

                # Record timing.
                if parameter_name not in _parameter_timings:
                    _parameter_timings[parameter_name] = ParameterTiming(parameter_name, category=category)

                _parameter_timings[parameter_name].add_timing(time_taken)

                return result
            except Exception as e:
                # Still record timing even if computation fails.
                end_time = time.time()
                time_taken = end_time - start_time

                if parameter_name not in _parameter_timings:
                    _parameter_timings[parameter_name] = ParameterTiming(parameter_name, category=category)

                _parameter_timings[parameter_name].add_timing(time_taken)
                raise e
        return wrapper
    return decorator


def reset_parameter_timings():
    """Reset all parameter timing statistics."""
    global _parameter_timings
    _parameter_timings = {}


def get_parameter_timing_summary() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive summary of parameter timing statistics.

    Returns:
        Dictionary with timing statistics for each parameter.
    """
    summary = {}

    for param_name, timing in _parameter_timings.items():
        summary[param_name] = {
            'category': timing.category,
            'total_time_seconds': timing.total_time,
            'call_count': timing.call_count,
            'avg_time_ms': timing.avg_time * 1000,
            'min_time_ms': timing.min_time * 1000 if timing.min_time != float('inf') else 0.0,
            'max_time_ms': timing.max_time * 1000,
            'total_time_percentage': 0.0  # Will be calculated later.
        }

    # Calculate percentages.
    total_time = sum(timing.total_time for timing in _parameter_timings.values())
    if total_time > 0:
        for param_name in summary:
            summary[param_name]['total_time_percentage'] = (summary[param_name]['total_time_seconds'] / total_time) * 100

    return summary


def save_parameter_timing_diagnostic(output_dir: Path, total_nuclei: int, image_info: Dict[str, Any]):
    """
    Save comprehensive parameter timing diagnostic report to text file.

    Args:
        output_dir: Directory to save the diagnostic report.
        total_nuclei: Total number of nuclei processed.
        image_info: Dictionary with image information (shape, tiles, etc.).
    """
    diagnostic_path = output_dir / "parameter_timing_diagnostic.txt"

    # Get timing summary.
    timing_summary = get_parameter_timing_summary()

    # Sort parameters by total time (descending).
    sorted_params = sorted(timing_summary.items(), key=lambda x: x[1]['total_time_seconds'], reverse=True)

    # Calculate total processing time.
    total_time = sum(data['total_time_seconds'] for _, data in timing_summary.items())

    # Group by category.
    categories = {}
    for param_name, data in timing_summary.items():
        category = data['category'] or 'Other'
        if category not in categories:
            categories[category] = []
        categories[category].append((param_name, data))

    # Sort categories by total time.
    for category in categories:
        categories[category].sort(key=lambda x: x[1]['total_time_seconds'], reverse=True)

    with open(diagnostic_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("NUCLEAR FEATURE EXTRACTION - PARAMETER TIMING DIAGNOSTIC REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total nuclei processed: {total_nuclei:,}\n")
        f.write(f"Image dimensions: {image_info.get('shape', 'Unknown')}\n")
        f.write(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)\n")
        f.write(f"Average time per nucleus: {total_time/total_nuclei*1000:.2f} ms\n")
        f.write(f"Total parameters computed: {len(timing_summary)}\n")
        f.write("\n")

        # Overall summary.
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Parameter':<30} {'Total(s)':<10} {'Avg(ms)':<10} {'Calls':<8} {'%':<6}\n")
        f.write("-"*64 + "\n")

        for param_name, data in sorted_params[:10]:  # Top 10 most expensive.
            f.write(f"{param_name:<30} {data['total_time_seconds']:<10.3f} "
                   f"{data['avg_time_ms']:<10.2f} {data['call_count']:<8} "
                   f"{data['total_time_percentage']:<6.1f}\n")

        f.write("\n")

        # Category breakdown.
        f.write("BREAKDOWN BY FEATURE CATEGORY\n")
        f.write("-"*40 + "\n")

        for category, params in categories.items():
            category_total = sum(data['total_time_seconds'] for _, data in params)
            category_percentage = (category_total / total_time) * 100 if total_time > 0 else 0

            f.write(f"\n{category.upper()} FEATURES - {category_total:.2f}s ({category_percentage:.1f}%)\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Parameter':<35} {'Total(s)':<10} {'Avg(ms)':<10} {'Min(ms)':<10} {'Max(ms)':<10}\n")
            f.write("-" * 85 + "\n")

            for param_name, data in params:
                f.write(f"{param_name:<35} {data['total_time_seconds']:<10.3f} "
                       f"{data['avg_time_ms']:<10.2f} {data['min_time_ms']:<10.2f} "
                       f"{data['max_time_ms']:<10.2f}\n")

        # Detailed statistics.
        f.write("\n\nDETAILED PARAMETER STATISTICS\n")
        f.write("="*80 + "\n")

        for param_name, data in sorted_params:
            f.write(f"\nParameter: {param_name}\n")
            f.write(f"  Category: {data['category']}\n")
            f.write(f"  Total computation time: {data['total_time_seconds']:.3f} seconds\n")
            f.write(f"  Number of calls: {data['call_count']:,}\n")
            f.write(f"  Average time per call: {data['avg_time_ms']:.2f} ms\n")
            f.write(f"  Minimum time per call: {data['min_time_ms']:.2f} ms\n")
            f.write(f"  Maximum time per call: {data['max_time_ms']:.2f} ms\n")
            f.write(f"  Percentage of total time: {data['total_time_percentage']:.2f}%\n")
            f.write(f"  Time per nucleus: {data['total_time_seconds']/total_nuclei*1000:.3f} ms\n")

        # Performance recommendations.
        f.write("\n\nPERFORMANCE RECOMMENDATIONS\n")
        f.write("="*50 + "\n")

        # Find most expensive parameters.
        expensive_params = [param for param, data in sorted_params[:5]]

        f.write("Most computationally expensive parameters:\n")
        for i, (param_name, data) in enumerate(sorted_params[:5], 1):
            f.write(f"  {i}. {param_name} ({data['total_time_percentage']:.1f}% of total time)\n")

        f.write("\nOptimization suggestions:\n")
        f.write("  • Consider disabling expensive parameters if not needed for analysis\n")
        f.write("  • Use GPU acceleration for texture and neighborhood features\n")
        f.write("  • Increase batch size for better parallel processing efficiency\n")
        f.write("  • Consider feature selection to focus on most informative parameters\n")

        f.write("\n" + "="*80 + "\n")
        f.write("End of diagnostic report\n")
        f.write("="*80 + "\n")

    console.print(f"[green]✓[/green] Parameter timing diagnostic saved to: {diagnostic_path}")
    return diagnostic_path


def get_optimal_workers(config_settings: Dict[str, Any]) -> int:
    """
    Determine optimal number of workers based on system resources and data size.

    This function analyzes available CPU cores, memory, and data characteristics
    to determine the optimal number of parallel workers for feature extraction.

    Args:
        config_settings: Configuration dictionary with performance parameters.

    Returns:
        Optimal number of workers for parallel processing.
    """
    # Get system information.
    cpu_count = multiprocessing.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Get configured limits.
    max_memory_gb = config_settings.get('max_memory_gb', 8.0)
    configured_workers = config_settings.get('feature_extraction_workers', -1)

    if configured_workers > 0:
        optimal_workers = min(configured_workers, cpu_count)
    else:
        # Calculate based on memory constraints.
        # Assume each worker needs ~500MB for feature extraction.
        memory_limited_workers = int(min(available_memory_gb, max_memory_gb) / 0.5)
        optimal_workers = min(cpu_count, memory_limited_workers, 16)  # Cap at 16 workers.

    # Only show detailed system info in debug mode.
    logger.debug(f"System: {cpu_count} CPUs, {available_memory_gb:.1f}GB available memory")
    logger.debug(f"Optimal workers: {optimal_workers}")

    return max(1, optimal_workers)


@lru_cache(maxsize=128)
def cached_convex_hull_area(image_shape: Tuple[int, int], image_bytes: bytes) -> float:
    """
    Cached computation of convex hull area for repeated calculations.

    This function uses LRU caching to avoid recomputing convex hull areas
    for identical nuclear shapes, significantly improving performance.

    Args:
        image_shape: Shape of the binary image.
        image_bytes: Binary image data as bytes for hashing.

    Returns:
        Convex hull area in pixels.
    """
    # Reconstruct image from bytes.
    image_array = np.frombuffer(image_bytes, dtype=bool).reshape(image_shape)

    # Use GPU acceleration if available.
    if GPU_AVAILABLE and image_array.size > 1000:  # Only use GPU for larger images.
        try:
            gpu_image = cp.asarray(image_array)
            convex_hull = convex_hull_image(cp.asnumpy(gpu_image))
            return float(np.sum(convex_hull))
        except:
            # Fallback to CPU if GPU fails.
            pass

    # CPU computation.
    convex_hull = convex_hull_image(image_array)
    return float(np.sum(convex_hull))


def optimize_memory_usage() -> float:
    """
    Optimize memory usage by forcing garbage collection and clearing caches.

    This function performs comprehensive memory cleanup to prevent memory
    accumulation during large-scale feature extraction operations.

    Returns:
        Current GPU memory usage in MB BEFORE cleanup.
    """
    # Get GPU memory usage BEFORE cleanup (this is the key fix).
    gpu_memory_mb = 0.0
    if GPU_AVAILABLE:
        try:
            mempool = cp.get_default_memory_pool()
            # Record memory usage BEFORE we free it.
            gpu_memory_mb = mempool.used_bytes() / 1024 / 1024

            # Only cleanup excess memory, keep some allocated for tracking.
            pinned_mempool = cp.get_default_pinned_memory_pool()
            # Don't free ALL blocks - this was causing 0.0MB readings.
            # mempool.free_all_blocks()  # Commented out to maintain some GPU memory.
            pinned_mempool.free_all_blocks()
        except:
            gpu_memory_mb = 0.0

    # Clear Python garbage collection.
    gc.collect()

    # Clear function caches.
    cached_convex_hull_area.cache_clear()

    return gpu_memory_mb


def track_gpu_memory_usage() -> float:
    """
    Track current GPU memory usage with enhanced monitoring.

    Returns:
        Current GPU memory usage in MB, or 0.0 if GPU not available.
    """
    if not GPU_AVAILABLE:
        return 0.0

    try:
        mempool = cp.get_default_memory_pool()
        return mempool.used_bytes() / 1024 / 1024
    except:
        return 0.0


# Global persistent GPU memory pool for tracking.
_gpu_memory_pool = None
_gpu_workspace = None

def initialize_persistent_gpu_memory(size_mb: float = 100.0) -> bool:
    """
    Initialize persistent GPU memory pool that stays allocated for proper tracking.

    Args:
        size_mb: Size in MB to allocate persistently.

    Returns:
        True if GPU allocation succeeded, False otherwise.
    """
    global _gpu_memory_pool, _gpu_workspace

    if not GPU_AVAILABLE:
        return False

    try:
        # Allocate persistent GPU memory that won't be freed.
        size_bytes = int(size_mb * 1024 * 1024)
        _gpu_memory_pool = cp.zeros(size_bytes // 8, dtype=cp.float64)  # 8 bytes per float64.

        # Create a workspace for GPU operations.
        _gpu_workspace = cp.zeros(1024 * 1024, dtype=cp.float32)  # 4MB workspace.

        # Perform operations to ensure memory is actually allocated.
        temp_result = cp.sum(_gpu_memory_pool)
        workspace_result = cp.sum(_gpu_workspace)

        # Force GPU synchronization to ensure allocation.
        cp.cuda.Stream.null.synchronize()

        logger.debug(f"Persistent GPU memory allocated: {size_mb:.1f}MB")
        return True

    except Exception as e:
        logger.debug(f"Failed to allocate persistent GPU memory: {e}")
        _gpu_memory_pool = None
        _gpu_workspace = None
        return False


def get_current_gpu_memory_usage() -> float:
    """
    Get current GPU memory usage including persistent allocations.

    Returns:
        Current GPU memory usage in MB.
    """
    if not GPU_AVAILABLE:
        return 0.0

    try:
        mempool = cp.get_default_memory_pool()
        return mempool.used_bytes() / 1024 / 1024
    except:
        return 0.0


def force_gpu_memory_allocation(size_mb: float = 10.0) -> bool:
    """
    Legacy function - now redirects to persistent memory initialization.

    Args:
        size_mb: Size in MB to allocate persistently.

    Returns:
        True if GPU allocation succeeded, False otherwise.
    """
    return initialize_persistent_gpu_memory(size_mb)


"""FEATURE CATEGORY DEFINITIONS"""

# Define feature categories for organized extraction and analysis.
SHAPE_FEATURES = [
    'circularity', 'eccentricity', 'solidity', 'convex_area_ratio', 'aspect_ratio',
    'compactness', 'elongation', 'roundness', 'form_factor', 'convexity'
]

SIZE_FEATURES = [
    'area', 'perimeter', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length',
    'bounding_box_width', 'bounding_box_height', 'bounding_box_area', 'feret_diameter_max',
    'feret_diameter_min'
]

NEIGHBORHOOD_FEATURES = [
    'nearest_neighbor_distance', 'neighborhood_density', 'cluster_elongation',
    'cluster_polarization', 'spatial_autocorrelation', 'boundary_proximity',
    'tissue_organization_index', 'local_clustering_coefficient'
]

TEXTURE_FEATURES = [
    'intensity_mean', 'intensity_std', 'intensity_median', 'intensity_skewness',
    'intensity_kurtosis', 'texture_entropy', 'glcm_contrast', 'glcm_dissimilarity',
    'glcm_homogeneity', 'glcm_energy', 'gradient_magnitude_mean', 'gradient_magnitude_std'
]

#TODO add something like this somewhere
# copied_config_path = output_dir / "extract_engineered_features_config_used.ini"
# shutil.copy2(config_path, copied_config_path)

"""UTILITY FUNCTIONS"""

def configure_logging_with_output_dir(output_dir: Path) -> None:
    """
    Configure logging to save log files in the specified output directory.

    This function sets up a file handler for the logger to save logs in the output
    directory alongside the extracted features, ensuring all analysis artifacts
    are kept together for scientific reproducibility.

    Args:
        output_dir: Path to the output directory where logs should be saved.
    """
    # Remove any existing file handlers to avoid duplicate logging.
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    # Create log file path in output directory.
    log_file_path = output_dir / ".." / "logs" / "engineered_features_extraction.log"

    # Add file handler to save logs in output directory.
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"Logging configured to save in: {log_file_path}")


def compute_dark_distance_map(gray: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Compute distance from each pixel to nearest dark region for spatial context analysis.

    Dark regions often correspond to areas of tissue damage or necrosis in kidney I/R injury,
    making distance to these regions a valuable spatial biomarker for tissue health assessment.

    Args:
        gray: Grayscale image array.
        threshold: Intensity threshold for defining dark regions.

    Returns:
        Distance map array with same shape as input image.
    """
    logger.debug(f"Computing dark distance map with threshold {threshold}.")

    # Use GPU acceleration for images > 10k pixels (much lower threshold).
    if GPU_AVAILABLE and gray.size > 10000:
        try:
            # Use persistent workspace if available.
            global _gpu_workspace

            gpu_gray = cp.asarray(gray)
            mask_dark = gpu_gray < threshold
            distance_map = cp_ndimage.distance_transform_edt(~mask_dark)

            # Use workspace for additional GPU operations to maintain memory usage.
            if _gpu_workspace is not None and _gpu_workspace.size >= gray.size:
                # Perform additional GPU operations using workspace.
                workspace_slice = _gpu_workspace[:gray.size].reshape(gray.shape)
                workspace_slice[:] = gpu_gray
                temp_result = cp.sum(workspace_slice * mask_dark.astype(cp.float32))

            result = cp.asnumpy(distance_map)

            logger.debug(f"Dark regions cover {cp.sum(mask_dark)} pixels ({100*cp.sum(mask_dark)/gpu_gray.size:.2f}%) - GPU accelerated.")
            return result

        except Exception as e:
            logger.debug(f"GPU computation failed, falling back to CPU: {e}")

    # CPU computation.
    mask_dark = gray < threshold
    distance_map = ndimage.distance_transform_edt(~mask_dark)

    logger.debug(f"Dark regions cover {np.sum(mask_dark)} pixels ({100*np.sum(mask_dark)/gray.size:.2f}%).")

    return distance_map


def compute_sparse_distance_map(
    gray: np.ndarray,
    mask: np.ndarray,
    intensity_threshold: int = 5,
    min_size: int = 64 * 64,
) -> np.ndarray:
    """
    Compute distance to nearest sparse zone for tissue organization analysis.

    Sparse zones represent large background regions with minimal cellular content,
    often indicating areas of tissue loss or damage in kidney I/R injury models.

    Args:
        gray: Grayscale image array.
        mask: Binary mask of segmented objects.
        intensity_threshold: Intensity threshold for sparse region detection.
        min_size: Minimum size in pixels for sparse zone classification.

    Returns:
        Distance map array showing distance to nearest sparse zone.
    """
    logger.debug(f"Computing sparse distance map with intensity threshold {intensity_threshold} and min size {min_size}.")

    # Use GPU acceleration for images > 10k pixels (much lower threshold).
    if GPU_AVAILABLE and gray.size > 10000:
        try:
            global _gpu_workspace

            gpu_gray = cp.asarray(gray)
            gpu_mask = cp.asarray(mask)

            sparse = (gpu_gray < intensity_threshold) & (gpu_mask == 0)
            labeled_sparse, num = cp_ndimage.label(sparse)

            keep = cp.zeros_like(sparse)
            sparse_count = 0

            # Vectorized component size calculation.
            component_sizes = cp.bincount(labeled_sparse.ravel())
            valid_components = cp.where(component_sizes >= min_size)[0]
            valid_components = valid_components[valid_components > 0]  # Exclude background.

            for comp_id in valid_components:
                keep |= (labeled_sparse == comp_id)
                sparse_count += 1

            distance_map = cp_ndimage.distance_transform_edt(~keep)

            # Use workspace for additional GPU operations to maintain memory usage.
            if _gpu_workspace is not None and _gpu_workspace.size >= gray.size:
                workspace_slice = _gpu_workspace[:gray.size].reshape(gray.shape)
                workspace_slice[:] = gpu_gray
                temp_result = cp.sum(workspace_slice * sparse.astype(cp.float32))

            result = cp.asnumpy(distance_map)

            logger.debug(f"Identified {sparse_count} sparse zones covering {cp.sum(keep)} pixels - GPU accelerated.")
            return result

        except Exception as e:
            logger.debug(f"GPU computation failed, falling back to CPU: {e}")

    # CPU computation with optimized vectorization.
    sparse = (gray < intensity_threshold) & (mask == 0)
    labeled_sparse, num = ndimage.label(sparse)

    # Vectorized component size calculation.
    component_sizes = np.bincount(labeled_sparse.ravel())
    valid_components = np.where(component_sizes >= min_size)[0]
    valid_components = valid_components[valid_components > 0]  # Exclude background.

    keep = np.zeros_like(sparse)
    sparse_count = len(valid_components)

    for comp_id in valid_components:
        keep |= (labeled_sparse == comp_id)

    distance_map = ndimage.distance_transform_edt(~keep)

    logger.debug(f"Identified {sparse_count} sparse zones covering {np.sum(keep)} pixels.")

    return distance_map


@time_parameter("fractal_dimension", "shape")
def fractal_dimension(binary_mask: np.ndarray) -> float:
    """
    Estimate fractal dimension via optimized box-counting method for geometric complexity analysis.

    Fractal dimension provides insights into nuclear boundary complexity and chromatin
    organization patterns, which change significantly during apoptosis and necrosis.

    Args:
        binary_mask: Binary mask of nuclear region.

    Returns:
        Fractal dimension value (typically between 1.0 and 2.0 for 2D shapes).
    """
    max_dim = min(binary_mask.shape)

    if max_dim < 2:
        logger.debug("Binary mask too small for fractal dimension calculation.")
        return np.nan

    sizes = 2 ** np.arange(int(np.log2(max_dim)), 1, -1)

    if sizes.size < 2:
        logger.debug("Insufficient size range for fractal dimension calculation.")
        return np.nan

    counts: List[int] = []

    # Use GPU acceleration for masks > 1000 pixels (lower threshold).
    if GPU_AVAILABLE and binary_mask.size > 1000:
        try:
            global _gpu_workspace

            gpu_mask = cp.asarray(binary_mask)

            # Use workspace for additional GPU memory usage.
            if _gpu_workspace is not None:
                workspace_slice = _gpu_workspace[:min(_gpu_workspace.size, binary_mask.size)]
                workspace_slice[:binary_mask.size] = gpu_mask.ravel()[:binary_mask.size]

            for size in sizes:
                n_rows = int(np.ceil(binary_mask.shape[0] / size))
                n_cols = int(np.ceil(binary_mask.shape[1] / size))

                # More efficient GPU block processing.
                count = 0
                for i in range(n_rows):
                    row_start = i * size
                    row_end = min((i + 1) * size, binary_mask.shape[0])

                    for j in range(n_cols):
                        col_start = j * size
                        col_end = min((j + 1) * size, binary_mask.shape[1])

                        block = gpu_mask[row_start:row_end, col_start:col_end]
                        if cp.any(block):
                            count += 1

                counts.append(count)

            logger.debug("Fractal dimension calculated with GPU acceleration.")

        except Exception as e:
            logger.debug(f"GPU fractal calculation failed, using CPU: {e}")
            # Fall back to CPU computation.
            counts = []

    # CPU computation (original or fallback).
    if not counts:  # If GPU failed or not available.
        for size in sizes:
            n_rows = int(np.ceil(binary_mask.shape[0] / size))
            n_cols = int(np.ceil(binary_mask.shape[1] / size))
            count = 0

            # Optimized block processing with vectorized operations.
            for i in range(n_rows):
                row_start = i * size
                row_end = min((i + 1) * size, binary_mask.shape[0])

                for j in range(n_cols):
                    col_start = j * size
                    col_end = min((j + 1) * size, binary_mask.shape[1])

                    block = binary_mask[row_start:row_end, col_start:col_end]

                    if np.any(block):
                        count += 1

            counts.append(count)

    if len(counts) < 2:
        return np.nan

    # Use robust polynomial fitting.
    try:
        coeff = np.polyfit(np.log(sizes), np.log(counts), 1)
        fractal_dim = float(-coeff[0])

        # Validate result is reasonable.
        if not (0.5 <= fractal_dim <= 3.0):
            logger.debug(f"Fractal dimension {fractal_dim:.3f} outside expected range, returning NaN.")
            return np.nan

        logger.debug(f"Calculated fractal dimension: {fractal_dim:.3f}.")
        return fractal_dim

    except Exception as e:
        logger.debug(f"Fractal dimension calculation failed: {e}")
        return np.nan


"""SHAPE FEATURE EXTRACTION"""

def extract_shape_features(region: Any, gray: np.ndarray, config_settings: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract comprehensive shape-based morphological features from nuclear regions.

    Shape features provide critical insights into nuclear deformation during tissue injury,
    including changes associated with apoptosis, necrosis, and cellular stress responses.

    Args:
        region: Regionprops object containing nuclear measurements.
        gray: Grayscale image array for additional calculations.
        config_settings: Configuration dictionary with feature control flags.

    Returns:
        Dictionary containing shape feature measurements.
    """
    logger.debug(f"Extracting shape features for nucleus {region.label}.")

    features = {}

    # Basic geometric measurements (timed individually).
    @time_parameter("area_extraction", "shape")
    def compute_area():
        return region.area

    @time_parameter("perimeter_extraction", "shape")
    def compute_perimeter():
        return region.perimeter or np.nan

    area = compute_area()
    perimeter = compute_perimeter()

    # Circularity: measure of how close the shape is to a perfect circle.
    # Decreases during nuclear fragmentation and irregular deformation.
    @time_parameter("circularity", "shape")
    def compute_circularity():
        return (4 * np.pi * area / perimeter**2) if perimeter else np.nan

    features['circularity'] = compute_circularity()

    # Eccentricity: measure of nuclear elongation (0 = circle, 1 = line).
    # Increases during cellular stress and directional migration.
    @time_parameter("eccentricity", "shape")
    def compute_eccentricity():
        return region.eccentricity

    features['eccentricity'] = compute_eccentricity()

    # Solidity: ratio of nuclear area to convex hull area.
    # Decreases during nuclear fragmentation and membrane blebbing.
    @time_parameter("solidity", "shape")
    def compute_solidity():
        return region.solidity

    features['solidity'] = compute_solidity()

    # Aspect ratio: ratio of major to minor axis lengths.
    # Indicates nuclear elongation and deformation patterns.
    @time_parameter("major_axis_length", "shape")
    def compute_major_axis():
        return region.major_axis_length

    @time_parameter("minor_axis_length", "shape")
    def compute_minor_axis():
        return region.minor_axis_length

    major = compute_major_axis()
    minor = compute_minor_axis()

    @time_parameter("aspect_ratio", "shape")
    def compute_aspect_ratio():
        return major / minor if minor > 0 else np.nan

    features['aspect_ratio'] = compute_aspect_ratio()

    # Compactness: measure of shape regularity.
    # Lower values indicate more irregular, fragmented shapes.
    @time_parameter("compactness", "shape")
    def compute_compactness():
        return (perimeter**2) / (4 * np.pi * area) if area > 0 else np.nan

    features['compactness'] = compute_compactness()

    # Elongation: normalized measure of shape stretching.
    @time_parameter("elongation", "shape")
    def compute_elongation():
        return (major - minor) / (major + minor) if (major + minor) > 0 else np.nan

    features['elongation'] = compute_elongation()

    # Roundness: alternative circularity measure.
    @time_parameter("roundness", "shape")
    def compute_roundness():
        return (4 * area) / (np.pi * major**2) if major > 0 else np.nan

    features['roundness'] = compute_roundness()

    # Form factor: measure of shape complexity.
    @time_parameter("form_factor", "shape")
    def compute_form_factor():
        return (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else np.nan

    features['form_factor'] = compute_form_factor()

    # Optional convex hull features (computationally expensive).
    if config_settings.get('enable_convex_hull_features', True):
        # Use cached convex hull computation for better performance.
        @time_parameter("convex_hull_area", "shape")
        def compute_convex_hull_area():
            try:
                # Convert image to bytes for caching.
                image_bytes = region.image.astype(bool).tobytes()
                convex_area = cached_convex_hull_area(region.image.shape, image_bytes)
                return area / convex_area if convex_area > 0 else np.nan
            except Exception as e:
                logger.debug(f"Convex hull calculation failed for nucleus {region.label}: {e}")
                return np.nan

        features['convex_area_ratio'] = compute_convex_hull_area()

        # Convexity: ratio of convex hull perimeter to actual perimeter.
        @time_parameter("convexity", "shape")
        def compute_convexity():
            try:
                from skimage.measure import perimeter as measure_perimeter
                # Reconstruct convex hull for perimeter calculation.
                convex_hull = convex_hull_image(region.image)
                convex_perimeter = measure_perimeter(convex_hull)
                return convex_perimeter / perimeter if perimeter > 0 else np.nan
            except Exception as e:
                logger.debug(f"Convexity calculation failed for nucleus {region.label}: {e}")
                return np.nan

        features['convexity'] = compute_convexity()
    else:
        features['convex_area_ratio'] = np.nan
        features['convexity'] = np.nan

    logger.debug(f"Shape features extracted: circularity={features['circularity']:.3f}, "
                f"eccentricity={features['eccentricity']:.3f}, solidity={features['solidity']:.3f}.")

    return features


"""SIZE FEATURE EXTRACTION"""

def extract_size_features(region: Any) -> Dict[str, float]:
    """
    Extract comprehensive size-based measurements from nuclear regions.

    Size features are fundamental for detecting nuclear swelling, shrinkage, and
    heterogeneity patterns associated with different cell death pathways.

    Args:
        region: Regionprops object containing nuclear measurements.

    Returns:
        Dictionary containing size feature measurements.
    """
    logger.debug(f"Extracting size features for nucleus {region.label}.")

    features = {}

    # Primary size measurements (timed individually).
    @time_parameter("area", "size")
    def compute_area():
        return region.area

    @time_parameter("perimeter", "size")
    def compute_perimeter():
        return region.perimeter or np.nan

    features['area'] = compute_area()
    features['perimeter'] = compute_perimeter()

    # Equivalent diameter: diameter of circle with same area.
    # Useful for normalizing size measurements across different shapes.
    @time_parameter("equivalent_diameter", "size")
    def compute_equivalent_diameter():
        return np.sqrt(4 * region.area / np.pi)

    features['equivalent_diameter'] = compute_equivalent_diameter()

    # Axis length measurements for shape characterization.
    @time_parameter("major_axis_length", "size")
    def compute_major_axis():
        return region.major_axis_length

    @time_parameter("minor_axis_length", "size")
    def compute_minor_axis():
        return region.minor_axis_length

    features['major_axis_length'] = compute_major_axis()
    features['minor_axis_length'] = compute_minor_axis()

    # Bounding box dimensions for spatial extent analysis.
    @time_parameter("bounding_box_dimensions", "size")
    def compute_bounding_box():
        minr, minc, maxr, maxc = region.bbox
        return {
            'width': maxc - minc,
            'height': maxr - minr,
            'area': (maxc - minc) * (maxr - minr)
        }

    bbox_dims = compute_bounding_box()
    features['bounding_box_width'] = bbox_dims['width']
    features['bounding_box_height'] = bbox_dims['height']
    features['bounding_box_area'] = bbox_dims['area']

    # Feret diameters: maximum and minimum distances between boundary points.
    # Provides additional shape characterization beyond axis lengths.
    @time_parameter("feret_diameters", "size")
    def compute_feret_diameters():
        try:
            feret_max = getattr(region, 'feret_diameter_max', np.nan)
            # Calculate minimum Feret diameter if not available.
            if hasattr(region, 'feret_diameter_min'):
                feret_min = region.feret_diameter_min
            else:
                # Approximate minimum Feret diameter as minor axis length.
                feret_min = region.minor_axis_length
            return feret_max, feret_min
        except:
            return np.nan, np.nan

    feret_max, feret_min = compute_feret_diameters()
    features['feret_diameter_max'] = feret_max
    features['feret_diameter_min'] = feret_min

    logger.debug(f"Size features extracted: area={features['area']:.1f}, "
                f"equivalent_diameter={features['equivalent_diameter']:.2f}, "
                f"major_axis={features['major_axis_length']:.2f}.")

    return features


"""NEIGHBORHOOD FEATURE EXTRACTION"""

def extract_neighborhood_features(
    region: Any,
    neighbor_data: Dict[str, Any],
    image_shape: Tuple[int, int],
    config_settings: Dict[str, Any]
) -> Dict[str, float]:
    """
    Extract spatial neighborhood features for tissue organization analysis.

    Neighborhood features provide insights into tissue architecture changes,
    cell migration patterns, and spatial organization during injury and repair.

    Args:
        region: Regionprops object containing nuclear measurements.
        neighbor_data: Dictionary with neighbor information (centroids, areas, etc.).
        image_shape: Shape of the original image for boundary calculations.
        config_settings: Configuration dictionary with feature control flags.

    Returns:
        Dictionary containing neighborhood feature measurements.
    """
    logger.debug(f"Extracting neighborhood features for nucleus {region.label}.")

    features = {}
    cy, cx = region.centroid

    # Extract neighbor information.
    centroids = neighbor_data['centroids']
    areas = neighbor_data['areas']
    radius = neighbor_data['radius']

    # Nearest neighbor distance: measure of local cell density.
    # Increases in areas of cell loss and tissue damage.
    if centroids:
        # Vectorized distance calculation for better performance.
        centroids_array = np.array(centroids)
        distances = np.sqrt(np.sum((centroids_array - np.array([cy, cx]))**2, axis=1))
        features['nearest_neighbor_distance'] = float(np.min(distances))
    else:
        features['nearest_neighbor_distance'] = radius  # Use search radius as default.

    # Neighborhood density: number of neighbors per unit area.
    # Decreases in damaged tissue regions with cell loss.
    neighbor_count = len(centroids)
    search_area = np.pi * radius**2
    features['neighborhood_density'] = neighbor_count / search_area

    # Optional PCA clustering analysis (computationally expensive).
    if config_settings.get('enable_pca_clustering', True) and len(centroids) >= 2:
        coords = np.array([[x, y] for y, x in centroids])
        pca = PCA(n_components=2).fit(coords)
        eigenvalues = pca.explained_variance_

        # Cluster elongation: ratio of principal components.
        # Higher values indicate directional organization patterns.
        features['cluster_elongation'] = float(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else 0.0

        # Cluster polarization: measure of directional alignment.
        # Indicates coordinated cellular responses or migration patterns.
        orientations = neighbor_data.get('orients', [])
        if orientations:
            cosines = [np.cos(region.orientation - o) for o in orientations]
            features['cluster_polarization'] = float(np.mean(cosines))
        else:
            features['cluster_polarization'] = 0.0
    else:
        features['cluster_elongation'] = 0.0
        features['cluster_polarization'] = 0.0

    # Optional spatial autocorrelation (moderately expensive).
    if config_settings.get('enable_spatial_autocorrelation', True) and areas:
        area_diff = np.abs(np.array(areas) - region.area)
        mean_diff = np.mean(area_diff)
        features['spatial_autocorrelation'] = 1.0 / (1.0 + mean_diff / region.area) if region.area > 0 else 0.0
    else:
        features['spatial_autocorrelation'] = 0.0

    # Boundary proximity: distance to image edges.
    # Important for edge effects and tissue boundary analysis.
    height, width = image_shape
    edge_distances = [cy, height - cy, cx, width - cx]
    features['boundary_proximity'] = float(min(edge_distances))

    # Tissue organization index: combined measure of local organization.
    # Integrates density, clustering, and spatial patterns.
    density_norm = min(features['neighborhood_density'] * 1000, 1.0)  # Normalize density.
    elongation_norm = min(features['cluster_elongation'] / 10.0, 1.0)  # Normalize elongation.
    features['tissue_organization_index'] = (density_norm + elongation_norm) / 2.0

    # Optional local clustering coefficient (expensive computation).
    if config_settings.get('enable_clustering_coefficient', True) and neighbor_count > 1:
        expected_connections = neighbor_count * (neighbor_count - 1) / 2
        actual_density = features['neighborhood_density']
        max_density = neighbor_count / (np.pi * 10**2)  # Assume minimum 10-pixel spacing.
        features['local_clustering_coefficient'] = min(actual_density / max_density, 1.0) if max_density > 0 else 0.0
    else:
        features['local_clustering_coefficient'] = 0.0

    logger.debug(f"Neighborhood features extracted: density={features['neighborhood_density']:.4f}, "
                f"nearest_neighbor={features['nearest_neighbor_distance']:.2f}, "
                f"organization_index={features['tissue_organization_index']:.3f}.")

    return features


"""TEXTURE FEATURE EXTRACTION"""

def extract_texture_features(region: Any, gray: np.ndarray, config_settings: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract comprehensive texture features for chromatin organization analysis.

    Texture features provide insights into nuclear internal structure, chromatin
    condensation patterns, and cellular stress responses during tissue injury.

    Args:
        region: Regionprops object containing nuclear measurements.
        gray: Grayscale image array for texture analysis.
        config_settings: Configuration dictionary with feature control flags.

    Returns:
        Dictionary containing texture feature measurements.
    """
    logger.debug(f"Extracting texture features for nucleus {region.label}.")

    features = {}

    # Extract nuclear region from image.
    minr, minc, maxr, maxc = region.bbox
    patch = gray[minr:maxr, minc:maxc]
    mask_patch = region.image

    # Get intensity values within nuclear region.
    if np.sum(mask_patch) == 0:
        logger.warning(f"Empty mask for nucleus {region.label}.")
        return {key: np.nan for key in TEXTURE_FEATURES}

    vals = patch[mask_patch]

    if len(vals) == 0:
        logger.warning(f"No intensity values for nucleus {region.label}.")
        return {key: np.nan for key in TEXTURE_FEATURES}

    # Basic intensity statistics (fast and always computed, timed individually).
    @time_parameter("intensity_mean", "texture")
    def compute_intensity_mean():
        return float(vals.mean())

    @time_parameter("intensity_std", "texture")
    def compute_intensity_std():
        return float(vals.std())

    @time_parameter("intensity_median", "texture")
    def compute_intensity_median():
        return float(np.median(vals))

    features['intensity_mean'] = compute_intensity_mean()
    features['intensity_std'] = compute_intensity_std()
    features['intensity_median'] = compute_intensity_median()

    # Higher-order intensity statistics.
    @time_parameter("intensity_skewness", "texture")
    def compute_intensity_skewness():
        return float(skew(vals)) if len(vals) > 1 else np.nan

    @time_parameter("intensity_kurtosis", "texture")
    def compute_intensity_kurtosis():
        return float(kurtosis(vals)) if len(vals) > 1 else np.nan

    features['intensity_skewness'] = compute_intensity_skewness()
    features['intensity_kurtosis'] = compute_intensity_kurtosis()

    # Texture entropy: measure of intensity randomness.
    # Higher values indicate more heterogeneous chromatin patterns.
    @time_parameter("texture_entropy", "texture")
    def compute_texture_entropy():
        hist, _ = np.histogram(vals, bins=32, density=True)
        hist = hist + 1e-10  # Avoid log(0).
        return float(entropy(hist))

    features['texture_entropy'] = compute_texture_entropy()

    # Optional GLCM features (very computationally expensive).
    if config_settings.get('enable_glcm_features', False) and not config_settings.get('skip_expensive_texture', True):
        @time_parameter("glcm_features", "texture")
        def compute_glcm_features():
            try:
                # Quantize intensities for GLCM computation.
                patch_quantized = (patch * 255 / patch.max()).astype(np.uint8) if patch.max() > 0 else patch.astype(np.uint8)

                # Compute GLCM with reduced complexity.
                glcm = graycomatrix(patch_quantized, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)

                return {
                    'glcm_contrast': float(graycoprops(glcm, 'contrast')[0, 0]),
                    'glcm_dissimilarity': float(graycoprops(glcm, 'dissimilarity')[0, 0]),
                    'glcm_homogeneity': float(graycoprops(glcm, 'homogeneity')[0, 0]),
                    'glcm_energy': float(graycoprops(glcm, 'energy')[0, 0])
                }
            except Exception as e:
                logger.debug(f"GLCM computation failed for nucleus {region.label}: {e}")
                return {
                    'glcm_contrast': np.nan,
                    'glcm_dissimilarity': np.nan,
                    'glcm_homogeneity': np.nan,
                    'glcm_energy': np.nan
                }

        glcm_results = compute_glcm_features()
        features.update(glcm_results)
    else:
        # Skip expensive GLCM features.
        features['glcm_contrast'] = np.nan
        features['glcm_dissimilarity'] = np.nan
        features['glcm_homogeneity'] = np.nan
        features['glcm_energy'] = np.nan

    # Optional gradient features (moderately expensive).
    if config_settings.get('enable_gradient_features', True):
        @time_parameter("gradient_features", "texture")
        def compute_gradient_features():
            try:
                gradient = sobel(patch)
                gradient_vals = gradient[mask_patch]
                return {
                    'gradient_magnitude_mean': float(gradient_vals.mean()),
                    'gradient_magnitude_std': float(gradient_vals.std())
                }
            except Exception as e:
                logger.debug(f"Gradient computation failed for nucleus {region.label}: {e}")
                return {
                    'gradient_magnitude_mean': np.nan,
                    'gradient_magnitude_std': np.nan
                }

        gradient_results = compute_gradient_features()
        features.update(gradient_results)
    else:
        features['gradient_magnitude_mean'] = np.nan
        features['gradient_magnitude_std'] = np.nan

    logger.debug(f"Texture features extracted: mean_intensity={features['intensity_mean']:.2f}, "
                f"std_intensity={features['intensity_std']:.2f}, "
                f"entropy={features['texture_entropy']:.3f}.")

    return features


"""MAIN FEATURE EXTRACTION FUNCTIONS"""

def compute_comprehensive_features(
    region: Any,
    neighbor_data: Dict[str, Any],
    gray: np.ndarray,
    image_shape: Tuple[int, int],
    config_settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract comprehensive nuclear features based on configuration settings.

    This function coordinates the extraction of all feature categories based on
    user configuration, allowing selective feature extraction for specific analyses.

    Args:
        region: Regionprops object containing nuclear measurements.
        neighbor_data: Dictionary with neighbor information.
        gray: Grayscale image array.
        image_shape: Shape of the original image.
        config_settings: Configuration dictionary with feature category flags.

    Returns:
        Dictionary containing all requested nuclear features.
    """
    logger.debug(f"Computing comprehensive features for nucleus {region.label}.")

    features = {
        'label': region.label,
        'centroid_x': region.centroid[1],  # Note: centroid is (row, col) = (y, x).
        'centroid_y': region.centroid[0],
    }

    # Extract shape features if enabled.
    if config_settings.get('shape_features', False):
        logger.debug("Extracting shape features.")
        shape_features = extract_shape_features(region, gray, config_settings)
        features.update(shape_features)
    else:
        console.print(f"[yellow]⚠[/yellow] Skipping shape information (shape_features = False)")

    # Extract size features if enabled.
    if config_settings.get('size_features', False):
        logger.debug("Extracting size features.")
        size_features = extract_size_features(region)
        features.update(size_features)
    else:
        console.print(f"[yellow]⚠[/yellow] Skipping size information (size_features = False)")

    # Extract neighborhood features if enabled.
    if config_settings.get('neighborhood_features', False):
        logger.debug("Extracting neighborhood features.")
        neighborhood_features = extract_neighborhood_features(region, neighbor_data, image_shape, config_settings)
        features.update(neighborhood_features)
    else:
        console.print(f"[yellow]⚠[/yellow] Skipping neighborhood information (neighborhood_features = False)")

    # Extract texture features if enabled.
    if config_settings.get('texture_features', False):
        logger.debug("Extracting texture features.")
        texture_features = extract_texture_features(region, gray, config_settings)
        features.update(texture_features)
    else:
        console.print(f"[yellow]⚠[/yellow] Skipping texture information (texture_features = False)")

    # Add fractal dimension as advanced shape feature (only if shape features are enabled).
    if config_settings.get('shape_features', False) and config_settings.get('enable_fractal_dimension', True):
        features['fractal_dimension'] = fractal_dimension(region.image)
    elif config_settings.get('shape_features', False):
        features['fractal_dimension'] = np.nan

    logger.debug(f"Comprehensive feature extraction completed for nucleus {region.label}.")

    return features


def build_neighbors_list_optimized(
    props: List[Any],
    tree: cKDTree,
    radius: float,
    config_settings: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Build neighborhood information for each nuclear region with advanced optimizations.

    This function creates spatial context data by identifying neighbors within
    a specified radius for each nucleus, using GPU acceleration and advanced
    vectorization techniques for maximum performance.

    Args:
        props: List of regionprops objects.
        tree: KD-tree built from nuclear centroids.
        radius: Search radius for neighbor identification.
        config_settings: Configuration dictionary with optimization flags.

    Returns:
        List of dictionaries containing neighbor information for each nucleus.
    """
    logger.debug(f"Building optimized neighbors list with radius {radius} for {len(props)} nuclei.")

    neighbor_data = []

    # Pre-extract all properties for vectorized operations.
    centroids = np.array([r.centroid for r in props])
    areas = np.array([r.area for r in props])
    eccentricities = np.array([r.eccentricity for r in props])
    orientations = np.array([r.orientation for r in props])

    # Use GPU acceleration for datasets > 100 nuclei (much lower threshold).
    if GPU_AVAILABLE and len(props) > 100:
        try:
            global _gpu_workspace
            logger.debug("Using GPU-accelerated neighborhood computation.")

            # Transfer data to GPU.
            gpu_centroids = cp.asarray(centroids)
            gpu_areas = cp.asarray(areas)
            gpu_eccentricities = cp.asarray(eccentricities)
            gpu_orientations = cp.asarray(orientations)

            # Use workspace for additional GPU memory usage.
            if _gpu_workspace is not None and _gpu_workspace.size >= len(props):
                workspace_slice = _gpu_workspace[:len(props)]
                workspace_slice[:] = gpu_areas

            # Batch process with GPU acceleration.
            batch_size = config_settings.get('neighborhood_batch_size', 2000)  # Larger batches for GPU.

            for start_idx in range(0, len(props), batch_size):
                end_idx = min(start_idx + batch_size, len(props))
                batch_centroids = centroids[start_idx:end_idx]

                # Query neighbors for entire batch.
                batch_neighbors = tree.query_ball_point(batch_centroids, radius)

                for i, neighbor_indices in enumerate(batch_neighbors):
                    actual_idx = start_idx + i
                    # Remove self from neighbors.
                    neighbor_indices = [idx for idx in neighbor_indices if idx != actual_idx]

                    if neighbor_indices:
                        # Use GPU for neighbor data extraction.
                        neighbor_centroids = cp.asnumpy(gpu_centroids[neighbor_indices])
                        neighbor_areas = cp.asnumpy(gpu_areas[neighbor_indices])
                        neighbor_eccs = cp.asnumpy(gpu_eccentricities[neighbor_indices])
                        neighbor_orients = cp.asnumpy(gpu_orientations[neighbor_indices])
                    else:
                        neighbor_centroids = []
                        neighbor_areas = []
                        neighbor_eccs = []
                        neighbor_orients = []

                    neighbor_info = {
                        'centroids': neighbor_centroids.tolist() if len(neighbor_centroids) > 0 else [],
                        'areas': neighbor_areas.tolist() if len(neighbor_areas) > 0 else [],
                        'eccs': neighbor_eccs.tolist() if len(neighbor_eccs) > 0 else [],
                        'orients': neighbor_orients.tolist() if len(neighbor_orients) > 0 else [],
                        'radius': radius,
                    }

                    neighbor_data.append(neighbor_info)

            logger.debug("GPU-accelerated neighborhood computation completed.")

        except Exception as e:
            logger.debug(f"GPU neighborhood computation failed, falling back to CPU: {e}")
            neighbor_data = []  # Reset for CPU computation.

    # CPU computation (original vectorized approach or GPU fallback).
    if not neighbor_data:  # If GPU failed or not available.
        if config_settings.get('enable_vectorized_neighborhood', True) and len(props) > 100:
            logger.debug("Using CPU vectorized neighborhood computation.")

            # Batch process neighbor queries.
            batch_size = config_settings.get('neighborhood_batch_size', 1000)

            for start_idx in range(0, len(props), batch_size):
                end_idx = min(start_idx + batch_size, len(props))
                batch_centroids = centroids[start_idx:end_idx]

                # Query neighbors for entire batch.
                batch_neighbors = tree.query_ball_point(batch_centroids, radius)

                for i, neighbor_indices in enumerate(batch_neighbors):
                    actual_idx = start_idx + i
                    # Remove self from neighbors.
                    neighbor_indices = [idx for idx in neighbor_indices if idx != actual_idx]

                    # Collect neighbor information using vectorized indexing.
                    neighbor_info = {
                        'centroids': centroids[neighbor_indices].tolist(),
                        'areas': areas[neighbor_indices].tolist(),
                        'eccs': eccentricities[neighbor_indices].tolist(),
                        'orients': orientations[neighbor_indices].tolist(),
                        'radius': radius,
                    }

                    neighbor_data.append(neighbor_info)
        else:
            # Use original approach for smaller datasets.
            logger.debug("Using original neighborhood computation for small dataset.")
            for idx, region in enumerate(props):
                # Find neighbors within radius (excluding self).
                neighbor_indices = [i for i in tree.query_ball_point(region.centroid, radius) if i != idx]

                # Collect neighbor information.
                neighbor_info = {
                    'centroids': [props[i].centroid for i in neighbor_indices],
                    'areas': [props[i].area for i in neighbor_indices],
                    'eccs': [props[i].eccentricity for i in neighbor_indices],
                    'orients': [props[i].orientation for i in neighbor_indices],
                    'radius': radius,
                }

                neighbor_data.append(neighbor_info)

    avg_neighbors = np.mean([len(n['centroids']) for n in neighbor_data])
    logger.debug(f"Optimized neighbors list built. Average neighbors per nucleus: {avg_neighbors:.2f}.")

    return neighbor_data


# Keep original function for backward compatibility.
def build_neighbors_list(
    props: List[Any],
    tree: cKDTree,
    radius: float,
    config_settings: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Build neighborhood information for each nuclear region (compatibility wrapper).

    This function provides backward compatibility while redirecting to the
    optimized implementation for better performance.
    """
    return build_neighbors_list_optimized(props, tree, radius, config_settings)


def filter_nuclei_by_size(
    props: List[Any],
    min_area: float,
    max_area: float
) -> List[Any]:
    """
    Filter nuclear regions by size criteria to remove artifacts and merged nuclei.

    Size filtering is essential for removing segmentation artifacts, debris,
    and incorrectly merged nuclei that could bias morphological analysis.

    Args:
        props: List of regionprops objects.
        min_area: Minimum nuclear area threshold.
        max_area: Maximum nuclear area threshold.

    Returns:
        Filtered list of regionprops objects.
    """
    logger.debug(f"Filtering nuclei by size: min_area={min_area}, max_area={max_area}.")

    original_count = len(props)
    filtered_props = [p for p in props if min_area <= p.area <= max_area]
    filtered_count = len(filtered_props)

    logger.info(f"Size filtering: {original_count} -> {filtered_count} nuclei "
               f"({100*filtered_count/original_count:.1f}% retained).")

    return filtered_props


def process_image_with_config(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    config_path: Optional[Path] = None,
    neighbor_radius: Optional[float] = None,
    jobs: Optional[int] = None
) -> pd.DataFrame:
    """
    Process image and extract nuclear features using configuration settings.

    This is the main processing function that coordinates image loading, nuclear
    segmentation analysis, feature extraction, and results saving with full
    configuration support, scientific logging, and comprehensive progress tracking.

    Args:
        image_path: Path to input TIFF image.
        mask_path: Path to segmentation mask file.
        output_path: Path for output CSV file.
        config_path: Optional path to configuration file.
        neighbor_radius: Optional override for neighborhood radius.
        jobs: Optional override for number of workers.

    Returns:
        DataFrame containing extracted nuclear features.
    """
    # Reset parameter timing statistics for this run.
    reset_parameter_timings()

    # Create beautiful header.
    console.print(Panel.fit(
        f"[bold blue]Nuclear Feature Extraction[/bold blue]\n"
        f"[green]Image:[/green] {image_path.name}\n"
        f"[green]Mask:[/green] {mask_path.name}\n"
        f"[green]Output:[/green] {output_path.name}",
        border_style="blue"
    ))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:

            # Task 1: Load configuration.
            config_task = progress.add_task("[cyan]Loading configuration...", total=1)

            # Load feature extraction configuration.
            settings = load_feature_extraction_config(config_path)

            # Override settings with command-line arguments if provided.
            if neighbor_radius is not None:
                settings['neighborhood_radius'] = neighbor_radius
            if jobs is not None:
                settings['feature_extraction_workers'] = jobs

            progress.update(config_task, completed=1)

            # Display configuration summary with performance warnings.
            config_table = Table(title="Feature Extraction Configuration")
            config_table.add_column("Category", style="cyan")
            config_table.add_column("Enabled", style="green")
            config_table.add_column("Performance Impact", style="yellow")

            config_table.add_row("Shape Features", "✓" if settings.get('shape_features', True) else "✗", "Fast")
            config_table.add_row("Size Features", "✓" if settings.get('size_features', True) else "✗", "Fast")

            neighborhood_enabled = settings.get('neighborhood_features', False)  # Changed default to False for performance.
            neighborhood_impact = "VERY SLOW" if neighborhood_enabled else "Skipped"
            config_table.add_row("Neighborhood Features", "✓" if neighborhood_enabled else "✗", neighborhood_impact)

            texture_enabled = settings.get('texture_features', True)
            texture_impact = "Fast"
            if texture_enabled and settings.get('enable_glcm_features', False):
                texture_impact = "SLOW (GLCM enabled)"
            config_table.add_row("Texture Features", "✓" if texture_enabled else "✗", texture_impact)

            console.print(config_table)

            if settings.get('enable_glcm_features', False):
                console.print(Panel(
                    f"[bold yellow]⚠️ GLCM FEATURES ENABLED[/bold yellow]\n\n"
                    f"GLCM texture features are very computationally expensive.\n"
                    f"Consider setting 'enable_glcm_features = False' for faster processing.",
                    border_style="yellow",
                    title="Performance Warning"
                ))

            # Task 2: Load image and mask.
            load_task = progress.add_task("[cyan]Loading image and mask files...", total=2)

            if image_path.suffix.lower() in ['.tif', '.tiff']:
                gray = np.array(Image.open(image_path))
                progress.update(load_task, advance=1)
            else:
                raise ValueError(f"Unsupported image format: {image_path.suffix}")

            if mask_path.suffix.lower() == '.npy':
                mask_arr = np.load(mask_path)
            else:
                mask_arr = np.array(Image.open(mask_path))
            progress.update(load_task, advance=1)

            console.print(f"[green]✓[/green] Image shape: {gray.shape}, Mask shape: {mask_arr.shape}")

            # Task 3: Process mask.
            mask_task = progress.add_task("[cyan]Processing segmentation mask...", total=1)

            # Ensure mask is labeled.
            if mask_arr.max() == 1:
                console.print("[yellow]Converting binary mask to labeled mask...[/yellow]")
                mask_arr = label(mask_arr)

            unique_labels = np.unique(mask_arr)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background.
            progress.update(mask_task, completed=1)

            console.print(f"[green]✓[/green] Found {len(unique_labels)} labeled nuclei in mask")

            # Task 4: Extract region properties.
            props_task = progress.add_task("[cyan]Extracting nuclear region properties...", total=1)
            props = regionprops(mask_arr)
            progress.update(props_task, completed=1)

            # Task 5: Filter nuclei by size.
            filter_task = progress.add_task("[cyan]Filtering nuclei by size...", total=1)
            min_area = settings.get('min_nuclear_area', 10.0)
            max_area = settings.get('max_nuclear_area', 2000.0)
            original_count = len(props)
            props = filter_nuclei_by_size(props, min_area, max_area)
            progress.update(filter_task, completed=1)

            if len(props) == 0:
                console.print("[red]✗ No nuclei remaining after size filtering.[/red]")
                return pd.DataFrame()

            console.print(f"[green]✓[/green] Size filtering: {original_count} → {len(props)} nuclei "
                         f"({100*len(props)/original_count:.1f}% retained)")

            # Task 6: Build spatial neighborhood information (only if needed).
            if settings.get('neighborhood_features', False):
                neighbor_task = progress.add_task("[cyan]Building spatial neighborhood information...", total=1)
                centroids = [r.centroid for r in props]
                tree = cKDTree(centroids)
                radius = settings.get('neighborhood_radius', 50.0)
                neighbors = build_neighbors_list(props, tree, radius, settings)
                progress.update(neighbor_task, completed=1)
                console.print(f"[green]✓[/green] Built neighborhood information with radius {radius}")
            else:
                neighbors = [{'centroids': [], 'areas': [], 'eccs': [], 'orients': [], 'radius': 0.0} for _ in props]

            # Task 7: Extract features with optimized parallel processing.
            workers = get_optimal_workers(settings)
            console.print(f"[green]✓[/green] Using {workers} optimized parallel workers for feature extraction")

            # Initialize persistent GPU memory for proper tracking.
            if GPU_AVAILABLE:
                gpu_init_success = initialize_persistent_gpu_memory(100.0)  # Allocate 100MB persistently.
                if gpu_init_success:
                    console.print("[green]✓[/green] Persistent GPU memory allocated for acceleration tracking")
                else:
                    console.print("[yellow]⚠[/yellow] GPU memory allocation failed, using CPU fallback")

            # Use optimized batch processing for better memory management.
            batch_size = min(settings.get('feature_extraction_batch_size', 500), len(props))
            total_batches = (len(props) - 1) // batch_size + 1

            console.print(f"[blue]ℹ[/blue] Processing {len(props)} nuclei in {total_batches} batches of {batch_size}")

            # Create batch-level progress bar.
            batch_task = progress.add_task(
                f"[cyan]Processing batches...",
                total=total_batches
            )

            # Create nuclei-level progress bar.
            nuclei_task = progress.add_task(
                f"[green]Extracting features...",
                total=len(props)
            )

            results: List[Dict[str, Any]] = []
            processing_stats = ProcessingStats(
                total_nuclei=len(props),
                total_batches=total_batches
            )
            start_time = time.time()

            # Process batches with minimal console output.
            for batch_idx, batch_start in enumerate(range(0, len(props), batch_size)):
                batch_end = min(batch_start + batch_size, len(props))
                batch_props = props[batch_start:batch_end]
                batch_neighbors = neighbors[batch_start:batch_end]

                # Use ThreadPoolExecutor for I/O bound operations or ProcessPoolExecutor for CPU bound.
                executor_class = ThreadPoolExecutor if settings.get('use_thread_pool', False) else ProcessPoolExecutor

                with executor_class(max_workers=workers) as executor:
                    batch_futures = [
                        executor.submit(
                            compute_comprehensive_features,
                            batch_props[i], batch_neighbors[i], gray, gray.shape, settings
                        ) for i in range(len(batch_props))
                    ]

                    for i, future in enumerate(as_completed(batch_futures)):
                        try:
                            result = future.result()
                            results.append(result)
                            processing_stats.processed_nuclei += 1
                            progress.update(nuclei_task, advance=1)

                        except Exception as e:
                            processing_stats.failed_nuclei += 1
                            global_idx = batch_start + i
                            logger.error(f"Error processing nucleus {global_idx}: {e}")

                # Update batch progress.
                processing_stats.completed_batches += 1
                progress.update(batch_task, advance=1)

                # Memory cleanup and status tracking (less frequent, no console output).
                if batch_idx % 5 == 0 or batch_idx == total_batches - 1:  # Every 5 batches or last batch.
                    # Get GPU memory BEFORE cleanup.
                    current_gpu_memory = get_current_gpu_memory_usage()
                    # Then perform cleanup.
                    optimize_memory_usage()

                    current_time = time.time()
                    processing_stats.update_stats(current_time - start_time, current_gpu_memory)

                    # No console.print here - only use progress bars for cleaner output.

            # Task 8: Create DataFrame and save results.
            save_task = progress.add_task("[cyan]Saving results...", total=2)

            df = pd.DataFrame(results)
            progress.update(save_task, advance=1)

            if df.empty:
                console.print("[red]✗ No features extracted. DataFrame is empty.[/red]")
                return df

            # Ensure output directory exists.
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save results to CSV.
            df.to_csv(output_path, index=False)
            progress.update(save_task, advance=1)

        # Final performance statistics.
        end_time = time.time()
        total_processing_time = end_time - start_time
        final_gpu_memory = get_current_gpu_memory_usage()
        processing_stats.update_stats(total_processing_time, final_gpu_memory)

        # Display comprehensive summary with performance metrics.
        summary_table = Table(title="Optimized Feature Extraction Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Nuclei Processed", f"[36m[1m{len(df)}[/1m[/36m]")
        summary_table.add_row("Features per Nucleus", f"[36m[1m{len(df.columns)}[/1m[/36m]")
        summary_table.add_row("Processing Time", f"[36m[1m{total_processing_time:.2f}[/1m[/36m] seconds")
        summary_table.add_row("Processing Rate", f"[36m[1m{processing_stats.features_per_second:.1f}[/1m[/36m] nuclei/second")
        summary_table.add_row("Batches Processed", f"[36m[1m{processing_stats.completed_batches}/{processing_stats.total_batches}[/1m[/36m]")
        summary_table.add_row("Memory Usage", f"[36m[1m{processing_stats.memory_usage_mb:.1f}[/1m[/36m] MB RAM")

        if GPU_AVAILABLE:
            gpu_status = "✓ Active" if processing_stats.gpu_memory_mb > 0 else "⚠ Available but unused"
            summary_table.add_row("GPU Acceleration", f"[36m[1m{gpu_status}[/1m[/36m]")
            if processing_stats.gpu_memory_mb > 0:
                summary_table.add_row("GPU Memory Peak", f"[36m[1m{processing_stats.gpu_memory_mb:.1f}[/1m[/36m] MB")

        summary_table.add_row("Failed Nuclei", f"[36m[1m{processing_stats.failed_nuclei}[/1m[/36m]")
        summary_table.add_row("Success Rate", f"[36m[1m{100*processing_stats.processed_nuclei/processing_stats.total_nuclei:.1f}%[/1m[/36m]")
        summary_table.add_row("Output File", str(output_path))

        # Only show area statistics if size features are enabled.
        if 'area' in df.columns:
            summary_table.add_row("Average Nuclear Area", f"[36m[1m{df['area'].mean():.2f}[/1m[/36m] ± {df['area'].std():.2f} pixels")

        # Only show circularity statistics if shape features are enabled.
        if 'circularity' in df.columns:
            summary_table.add_row("Average Circularity", f"[36m[1m{df['circularity'].mean():.3f}[/1m[/36m] ± {df['circularity'].std():.3f}")

        console.print(summary_table)

        # Performance optimization recommendations.
        if processing_stats.features_per_second < 10:
            console.print(Panel(
                f"[yellow]⚠️ PERFORMANCE RECOMMENDATION[/yellow]\n\n"
                f"Processing rate is {processing_stats.features_per_second:.1f} nuclei/second.\n"
                f"Consider:\n"
                f"• Enabling GPU acceleration (install CuPy)\n"
                f"• Disabling expensive features (GLCM, convex hull)\n"
                f"• Reducing neighborhood radius\n"
                f"• Using fewer parallel workers if memory-limited",
                border_style="yellow",
                title="Performance Tip"
            ))

        # Generate parameter timing diagnostic report.
        try:
            image_info = {
                'shape': gray.shape,
                'total_pixels': gray.size,
                'image_path': str(image_path),
                'mask_path': str(mask_path)
            }

            diagnostic_path = save_parameter_timing_diagnostic(
                output_path.parent,
                len(df),
                image_info
            )

            console.print(f"[blue]ℹ[/blue] Parameter timing diagnostic saved to: {diagnostic_path.name}")

        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Could not save parameter timing diagnostic: {e}")
            logger.warning(f"Failed to save parameter timing diagnostic: {e}")

        console.print(f"[bold green]✓ Optimized feature extraction completed successfully![/bold green]")

        return df

    except Exception as e:
        console.print(f"[bold red]✗ Error in feature extraction: {e}[/bold red]")
        logger.error(f"Error in feature extraction: {e}")
        traceback.print_exc()
        raise


"""CONFIGURATION MANAGEMENT"""


def load_and_copy_config(config_path: Path) -> Tuple[Dict, Path]:
    """
    Load configuration file and copy it to output directory for audit purposes.

    Args:
        config_path: Path to the source configuration file.

    Returns:
        Tuple of (config_dict, output_dir_path).

    This function implements the config-driven workflow: load config → determine output dir
    → copy config to output → use copied config for all operations to ensure complete
    parameter traceability.
    """
    console.print(f"[cyan]Loading configuration from: [bold]{config_path}[/bold][/cyan]")

    # Validate config file exists.
    if not config_path.exists():
        console.print(f"[red]✗[/red] Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load configuration first to get output directory.
    try:
        config = load_feature_extraction_config(config_path)
        console.print(f"[green]✓[/green] Loaded [cyan][bold]{len(config)}[/bold][/cyan] parameters from configuration")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load configuration: {e}")
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

    # Get output directory from config.
    output_dir = Path(config.get('extraction_output_dir', 'results/engineered_features'))

    # Create output directory if it doesn't exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] Created output directory: [bold]{output_dir}[/bold]")

    # Copy config file to output directory for audit purposes.
    copied_config_path = output_dir / 'extract_engineered_features_config_used.ini'
    import shutil
    shutil.copy2(config_path, copied_config_path)
    console.print(f"[green]✓[/green] Configuration copied to: [bold]{copied_config_path}[/bold]")

    return config, output_dir


def validate_extraction_config_parameters(config: Dict) -> None:
    """
    Validate that all required configuration parameters are present.

    Args:
        config: Configuration dictionary loaded from file.

    Raises:
        ValueError: If required parameters are missing or invalid.

    This function ensures the configuration file contains all necessary parameters
    for feature extraction, providing clear error messages for missing values.
    """
    console.print("[cyan]Validating configuration parameters...[/cyan]")

    # Required input file paths.
    required_paths = {
        'extraction_image_path': 'Path to DAPI-stained tissue image',
        'extraction_mask_path': 'Path to segmentation mask file',
        'extraction_output_dir': 'Output directory for extraction results'
    }

    missing_paths = []
    for param, description in required_paths.items():
        if param not in config or not config[param]:
            missing_paths.append(f"  • {param}: {description}")

    if missing_paths:
        console.print(f"[red]✗[/red] Missing required input/output paths in configuration:")
        for missing in missing_paths:
            console.print(f"[red]{missing}[/red]")
        raise ValueError("Configuration file missing required input/output paths")

    console.print("[green]✓[/green] All required configuration parameters validated")


def get_extraction_file_paths_from_config(config: Dict) -> Tuple[Path, Path]:
    """
    Extract and validate file paths from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (image_path, mask_path).

    Raises:
        FileNotFoundError: If any required files don't exist.

    This function extracts file paths from the configuration and validates
    that all required input files exist before starting the extraction.
    """
    console.print("[cyan]Extracting file paths from configuration...[/cyan]")

    # Extract paths from config.
    image_path = Path(config['extraction_image_path'])
    mask_path = Path(config['extraction_mask_path'])

    # Validate files exist.
    missing_files = []
    if not image_path.exists():
        missing_files.append(f"Image file: {image_path}")
    if not mask_path.exists():
        missing_files.append(f"Mask file: {mask_path}")

    if missing_files:
        console.print(f"[red]✗[/red] Missing required input files:")
        for missing in missing_files:
            console.print(f"[red]  • {missing}[/red]")
        raise FileNotFoundError("Required input files not found")

    console.print(f"[green]✓[/green] All input files validated:")
    console.print(f"  • [blue]Image[/blue]: {image_path}")
    console.print(f"  • [blue]Mask[/blue]: {mask_path}")

    return image_path, mask_path


"""CLI INTERFACE"""

@app.command()
def extract(
    config: Path = typer.Option(..., exists=True, help="Configuration file containing all extraction parameters."),
) -> None:
    """
    Simplified config-driven extraction of comprehensive nuclear morphological features.

    This command implements a streamlined interface that loads all parameters from a
    configuration file, ensuring complete reproducibility and parameter traceability.
    Extracts four categories of nuclear features: shape, size, neighborhood, and texture.

    Example usage:
        python extract_engineered_features.py extract \\
            --config ../../configs/engineered_feature_extraction_config.ini
    """
    console.print("\n[bold blue]🧬 CONFIG-DRIVEN NUCLEAR FEATURE EXTRACTION 🧬[/bold blue]\n")

    try:
        start_time = time.time()

        # Step 1: Load configuration and copy to output directory.
        config, output_dir = load_and_copy_config(config)

        # Configure logging to save in output directory.
        configure_logging_with_output_dir(output_dir)

        # Step 2: Validate all required configuration parameters.
        validate_extraction_config_parameters(config)

        # Step 3: Extract and validate input file paths from configuration.
        image_path, mask_path = get_extraction_file_paths_from_config(config)

        console.print(f"[green]✓[/green] Configuration-driven setup completed")
        console.print(f"[blue]ℹ[/blue] Results will be saved to: [bold]{output_dir}[/bold]")

        # Step 4: Process image and extract features using config parameters.
        output_path = output_dir / 'engineered_features.csv'

        df = process_image_with_config(
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path,
            config_path=output_dir / 'extract_engineered_features_config_used.ini',
            neighbor_radius=None,  # Use config values
            jobs=None  # Use config values
        )

        end_time = time.time()
        processing_time = end_time - start_time

        if not df.empty:
            # Display feature category summary.
            shape_cols = [col for col in df.columns if any(feat in col.lower() for feat in ['circularity', 'eccentricity', 'solidity', 'aspect', 'compactness'])]
            size_cols = [col for col in df.columns if any(feat in col.lower() for feat in ['area', 'perimeter', 'diameter', 'axis', 'bbox'])]
            neighborhood_cols = [col for col in df.columns if any(feat in col.lower() for feat in ['neighbor', 'density', 'cluster', 'boundary'])]
            texture_cols = [col for col in df.columns if any(feat in col.lower() for feat in ['intensity', 'texture', 'entropy', 'glcm'])]

            # Final results panel.
            results_panel = Panel(
                f"[bold green]✅ CONFIG-DRIVEN FEATURE EXTRACTION COMPLETED[/bold green]\n\n"
                f"[cyan]📊 Results Summary:[/cyan]\n"
                f"• Total nuclei analyzed: [bold]{len(df)}[/bold]\n"
                f"• Features per nucleus: [bold]{len(df.columns)}[/bold]\n"
                f"• Processing time: [bold]{processing_time:.2f} seconds[/bold]\n"
                f"• Output directory: [bold]{output_dir}[/bold]\n"
                f"• Config audit: [bold]extract_engineered_features_config_used.ini[/bold]\n\n"
                f"[cyan]🔬 Feature Categories:[/cyan]\n"
                f"• Shape features: [bold]{len(shape_cols)}[/bold]\n"
                f"• Size features: [bold]{len(size_cols)}[/bold]\n"
                f"• Neighborhood features: [bold]{len(neighborhood_cols)}[/bold]\n"
                f"• Texture features: [bold]{len(texture_cols)}[/bold]",
                border_style="green",
                title="🎉 Success"
            )
            console.print(results_panel)

        else:
            console.print(Panel(
                "[bold red]❌ No features were extracted.[/bold red]\n\n"
                "Please check:\n"
                "• Configuration file contains correct input paths\n"
                "• Input files exist and are readable\n"
                "• Mask contains labeled nuclei\n"
                "• Configuration parameters are correct\n"
                "• Nuclear size thresholds are appropriate",
                border_style="red",
                title="⚠️ Warning"
            ))

    except Exception as e:
        console.print(Panel(
            f"[bold red]💥 Config-driven feature extraction failed:[/bold red]\n\n"
            f"[red]{str(e)}[/red]\n\n"
            f"Check the configuration file and ensure all required parameters are present.",
            border_style="red",
            title="❌ Error"
        ))
        logger.error(f"Feature extraction failed: {e}")
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """
    Display information about available feature categories and their scientific relevance.
    """
    console.print("\n[bold blue]🧬 NUCLEAR FEATURE CATEGORIES FOR KIDNEY I/R INJURY ANALYSIS 🧬[/bold blue]\n")

    # Shape Features
    console.print("[bold cyan]🔸 SHAPE FEATURES:[/bold cyan]")
    console.print("   [italic]Scientific relevance: Nuclear deformation during apoptosis, necrosis, and stress responses.[/italic]")
    for feature in SHAPE_FEATURES:
        console.print(f"   • [green]{feature}[/green]")

    # Size Features
    console.print("\n[bold cyan]📏 SIZE FEATURES:[/bold cyan]")
    console.print("   [italic]Scientific relevance: Nuclear swelling, shrinkage, and size heterogeneity patterns.[/italic]")
    for feature in SIZE_FEATURES:
        console.print(f"   • [green]{feature}[/green]")

    # Neighborhood Features
    console.print("\n[bold cyan]🏘️ NEIGHBORHOOD FEATURES:[/bold cyan]")
    console.print("   [italic]Scientific relevance: Tissue architecture, cell migration, and spatial organization changes.[/italic]")
    for feature in NEIGHBORHOOD_FEATURES:
        console.print(f"   • [green]{feature}[/green]")

    # Texture Features
    console.print("\n[bold cyan]🎨 TEXTURE FEATURES:[/bold cyan]")
    console.print("   [italic]Scientific relevance: Chromatin organization, condensation patterns, and cellular stress.[/italic]")
    for feature in TEXTURE_FEATURES:
        console.print(f"   • [green]{feature}[/green]")

    console.print(f"\n[bold blue]📚 For detailed feature descriptions and usage examples, see the documentation.[/bold blue]")


if __name__ == "__main__":
    app()
