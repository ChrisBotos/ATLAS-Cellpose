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
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
import warnings
import time

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
    log_file_path = output_dir / 'feature_extraction.log'

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
    
    sparse = (gray < intensity_threshold) & (mask == 0)
    labeled_sparse, num = ndimage.label(sparse)
    
    keep = np.zeros_like(sparse)
    sparse_count = 0
    
    for idx in range(1, num + 1):
        comp = labeled_sparse == idx
        
        if comp.sum() >= min_size:
            keep |= comp
            sparse_count += 1
    
    distance_map = ndimage.distance_transform_edt(~keep)
    
    logger.debug(f"Identified {sparse_count} sparse zones covering {np.sum(keep)} pixels.")
    
    return distance_map


def fractal_dimension(binary_mask: np.ndarray) -> float:
    """
    Estimate fractal dimension via box-counting method for geometric complexity analysis.
    
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
    
    for size in sizes:
        n_rows = int(np.ceil(binary_mask.shape[0] / size))
        n_cols = int(np.ceil(binary_mask.shape[1] / size))
        count = 0
        
        for i in range(n_rows):
            for j in range(n_cols):
                block = binary_mask[i*size:(i+1)*size, j*size:(j+1)*size]
                
                if block.any():
                    count += 1
                    
        counts.append(count)

    if len(counts) < 2:
        return np.nan

    coeff = np.polyfit(np.log(sizes), np.log(counts), 1)
    fractal_dim = float(-coeff[0])
    
    logger.debug(f"Calculated fractal dimension: {fractal_dim:.3f}.")
    
    return fractal_dim


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

    # Basic geometric measurements.
    area = region.area
    perimeter = region.perimeter or np.nan

    # Circularity: measure of how close the shape is to a perfect circle.
    # Decreases during nuclear fragmentation and irregular deformation.
    features['circularity'] = (4 * np.pi * area / perimeter**2) if perimeter else np.nan

    # Eccentricity: measure of nuclear elongation (0 = circle, 1 = line).
    # Increases during cellular stress and directional migration.
    features['eccentricity'] = region.eccentricity

    # Solidity: ratio of nuclear area to convex hull area.
    # Decreases during nuclear fragmentation and membrane blebbing.
    features['solidity'] = region.solidity

    # Aspect ratio: ratio of major to minor axis lengths.
    # Indicates nuclear elongation and deformation patterns.
    major = region.major_axis_length
    minor = region.minor_axis_length
    features['aspect_ratio'] = major / minor if minor > 0 else np.nan

    # Compactness: measure of shape regularity.
    # Lower values indicate more irregular, fragmented shapes.
    features['compactness'] = (perimeter**2) / (4 * np.pi * area) if area > 0 else np.nan

    # Elongation: normalized measure of shape stretching.
    features['elongation'] = (major - minor) / (major + minor) if (major + minor) > 0 else np.nan

    # Roundness: alternative circularity measure.
    features['roundness'] = (4 * area) / (np.pi * major**2) if major > 0 else np.nan

    # Form factor: measure of shape complexity.
    features['form_factor'] = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else np.nan

    # Optional convex hull features (computationally expensive).
    if config_settings.get('enable_convex_hull_features', True):
        # Convex area ratio: additional measure of shape regularity.
        convex_hull = convex_hull_image(region.image)
        convex_area = np.sum(convex_hull)
        features['convex_area_ratio'] = area / convex_area if convex_area > 0 else np.nan

        # Convexity: ratio of convex hull perimeter to actual perimeter.
        try:
            from skimage.measure import perimeter as measure_perimeter
            convex_perimeter = measure_perimeter(convex_hull)
            features['convexity'] = convex_perimeter / perimeter if perimeter > 0 else np.nan
        except:
            features['convexity'] = np.nan
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

    # Primary size measurements.
    features['area'] = region.area
    features['perimeter'] = region.perimeter or np.nan

    # Equivalent diameter: diameter of circle with same area.
    # Useful for normalizing size measurements across different shapes.
    features['equivalent_diameter'] = np.sqrt(4 * region.area / np.pi)

    # Axis length measurements for shape characterization.
    features['major_axis_length'] = region.major_axis_length
    features['minor_axis_length'] = region.minor_axis_length

    # Bounding box dimensions for spatial extent analysis.
    minr, minc, maxr, maxc = region.bbox
    features['bounding_box_width'] = maxc - minc
    features['bounding_box_height'] = maxr - minr
    features['bounding_box_area'] = (maxc - minc) * (maxr - minr)

    # Feret diameters: maximum and minimum distances between boundary points.
    # Provides additional shape characterization beyond axis lengths.
    try:
        features['feret_diameter_max'] = getattr(region, 'feret_diameter_max', np.nan)
        # Calculate minimum Feret diameter if not available.
        if hasattr(region, 'feret_diameter_min'):
            features['feret_diameter_min'] = region.feret_diameter_min
        else:
            # Approximate minimum Feret diameter as minor axis length.
            features['feret_diameter_min'] = region.minor_axis_length
    except:
        features['feret_diameter_max'] = np.nan
        features['feret_diameter_min'] = np.nan

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

    # Basic intensity statistics (fast and always computed).
    features['intensity_mean'] = float(vals.mean())
    features['intensity_std'] = float(vals.std())
    features['intensity_median'] = float(np.median(vals))

    # Higher-order intensity statistics.
    features['intensity_skewness'] = float(skew(vals)) if len(vals) > 1 else np.nan
    features['intensity_kurtosis'] = float(kurtosis(vals)) if len(vals) > 1 else np.nan

    # Texture entropy: measure of intensity randomness.
    # Higher values indicate more heterogeneous chromatin patterns.
    hist, _ = np.histogram(vals, bins=32, density=True)
    hist = hist + 1e-10  # Avoid log(0).
    features['texture_entropy'] = float(entropy(hist))

    # Optional GLCM features (very computationally expensive).
    if config_settings.get('enable_glcm_features', False) and not config_settings.get('skip_expensive_texture', True):
        try:
            # Quantize intensities for GLCM computation.
            patch_quantized = (patch * 255 / patch.max()).astype(np.uint8) if patch.max() > 0 else patch.astype(np.uint8)

            # Compute GLCM with reduced complexity.
            glcm = graycomatrix(patch_quantized, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)

            features['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
            features['glcm_dissimilarity'] = float(graycoprops(glcm, 'dissimilarity')[0, 0])
            features['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
            features['glcm_energy'] = float(graycoprops(glcm, 'energy')[0, 0])
        except Exception as e:
            logger.debug(f"GLCM computation failed for nucleus {region.label}: {e}")
            features['glcm_contrast'] = np.nan
            features['glcm_dissimilarity'] = np.nan
            features['glcm_homogeneity'] = np.nan
            features['glcm_energy'] = np.nan
    else:
        # Skip expensive GLCM features.
        features['glcm_contrast'] = np.nan
        features['glcm_dissimilarity'] = np.nan
        features['glcm_homogeneity'] = np.nan
        features['glcm_energy'] = np.nan

    # Optional gradient features (moderately expensive).
    if config_settings.get('enable_gradient_features', True):
        try:
            gradient = sobel(patch)
            gradient_vals = gradient[mask_patch]
            features['gradient_magnitude_mean'] = float(gradient_vals.mean())
            features['gradient_magnitude_std'] = float(gradient_vals.std())
        except Exception as e:
            logger.debug(f"Gradient computation failed for nucleus {region.label}: {e}")
            features['gradient_magnitude_mean'] = np.nan
            features['gradient_magnitude_std'] = np.nan
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

    # Extract size features if enabled.
    if config_settings.get('size_features', False):
        logger.debug("Extracting size features.")
        size_features = extract_size_features(region)
        features.update(size_features)

    # Extract neighborhood features if enabled.
    if config_settings.get('neighborhood_features', False):
        logger.debug("Extracting neighborhood features.")
        neighborhood_features = extract_neighborhood_features(region, neighbor_data, image_shape, config_settings)
        features.update(neighborhood_features)

    # Extract texture features if enabled.
    if config_settings.get('texture_features', False):
        logger.debug("Extracting texture features.")
        texture_features = extract_texture_features(region, gray, config_settings)
        features.update(texture_features)

    # Add fractal dimension as advanced shape feature (only if shape features are enabled).
    if config_settings.get('shape_features', False) and config_settings.get('enable_fractal_dimension', True):
        features['fractal_dimension'] = fractal_dimension(region.image)
    elif config_settings.get('shape_features', False):
        features['fractal_dimension'] = np.nan

    logger.debug(f"Comprehensive feature extraction completed for nucleus {region.label}.")

    return features


def build_neighbors_list(
    props: List[Any],
    tree: cKDTree,
    radius: float,
    config_settings: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Build neighborhood information for each nuclear region.

    This function creates spatial context data by identifying neighbors within
    a specified radius for each nucleus, enabling neighborhood feature extraction.

    Args:
        props: List of regionprops objects.
        tree: KD-tree built from nuclear centroids.
        radius: Search radius for neighbor identification.
        config_settings: Configuration dictionary with optimization flags.

    Returns:
        List of dictionaries containing neighbor information for each nucleus.
    """
    logger.debug(f"Building neighbors list with radius {radius} for {len(props)} nuclei.")

    neighbor_data = []

    # Use vectorized approach if enabled and beneficial.
    if config_settings.get('enable_vectorized_neighborhood', True) and len(props) > 100:
        logger.debug("Using vectorized neighborhood computation for better performance.")

        # Pre-extract all properties for vectorized operations.
        centroids = np.array([r.centroid for r in props])
        areas = np.array([r.area for r in props])
        eccentricities = np.array([r.eccentricity for r in props])
        orientations = np.array([r.orientation for r in props])

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

    logger.debug(f"Neighbors list built. Average neighbors per nucleus: "
                f"{np.mean([len(n['centroids']) for n in neighbor_data]):.2f}.")

    return neighbor_data


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
                console.print(f"[yellow]⚠[/yellow] Skipping neighborhood information (neighborhood_features = False)")
                neighbors = [{'centroids': [], 'areas': [], 'eccs': [], 'orients': [], 'radius': 0.0} for _ in props]

            # Task 7: Extract features with parallel processing.
            workers = settings.get('feature_extraction_workers', -1)
            if workers == -1:
                workers = multiprocessing.cpu_count()

            console.print(f"[green]✓[/green] Using {workers} parallel workers for feature extraction")

            feature_task = progress.add_task(
                f"[cyan]Extracting features from {len(props)} nuclei...",
                total=len(props)
            )

            results: List[Dict[str, Any]] = []

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        compute_comprehensive_features,
                        props[i], neighbors[i], gray, gray.shape, settings
                    ) for i in range(len(props))
                ]

                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        results.append(result)
                        progress.update(feature_task, advance=1)

                    except Exception as e:
                        console.print(f"[red]Error processing nucleus {i}: {e}[/red]")
                        logger.error(f"Error processing nucleus {i}: {e}")
                        traceback.print_exc()

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

        # Display final summary.
        summary_table = Table(title="Feature Extraction Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Nuclei Processed", str(len(df)))
        summary_table.add_row("Features per Nucleus", str(len(df.columns)))
        summary_table.add_row("Output File", str(output_path))

        # Only show area statistics if size features are enabled.
        if 'area' in df.columns:
            summary_table.add_row("Average Nuclear Area", f"{df['area'].mean():.2f} ± {df['area'].std():.2f} pixels")

        # Only show circularity statistics if shape features are enabled.
        if 'circularity' in df.columns:
            summary_table.add_row("Average Circularity", f"{df['circularity'].mean():.3f} ± {df['circularity'].std():.3f}")

        console.print(summary_table)
        console.print(f"[bold green]✓ Feature extraction completed successfully![/bold green]")

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
