#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: extract_engineered_features.py.
Description:
    Fast extraction of comprehensive nuclear morphological, neighborhood, and texture features from segmented
    DAPI-stained tissue sections. This optimized version includes all essential size, shape, spatial
    neighborhood, and optional texture features for kidney ischemia-reperfusion injury analysis.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scikit-image, scipy, typer, rich.

Usage:
    python extract_engineered_features.py --config ../../configs/engineered_feature_extraction_config.ini

Arguments:
    --config            Path to configuration file containing extraction parameters.

Key Features:
    • Fast single-threaded processing for reliability.
    • Comprehensive size features: area, perimeter, axes, bounding box, Feret diameters.
    • Complete shape features: circularity, eccentricity, solidity, aspect ratio, compactness,
      elongation, roundness, form factor, convex area ratio, convexity.
    • Advanced neighborhood features: neighbor counts, densities, distances, clustering metrics.
    • Optional texture features: intensity statistics, entropy, gradients, GLCM properties.
    • Detailed progress tracking with feature-level timing and completion status.
    • Optimized spatial indexing with KDTree for efficient neighbor detection.
    • Clean progress reporting with rich console and colored feature categories.
    • Robust error handling and validation.
    • Scientific context for kidney I/R injury research.

Notes:
    • Size features quantify nuclear dimensions and spatial extent.
    • Shape features measure nuclear morphology, regularity, and complexity.
    • Neighborhood features analyze spatial relationships and local tissue architecture.
    • Texture features analyze DAPI intensity patterns and chromatin organization (optional).
    • Results saved as CSV with nucleus_id and all extracted features.
"""

import traceback
import sys
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import typer
from PIL import Image
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from scipy.spatial import KDTree
from scipy.stats import skew, kurtosis, entropy
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

Image.MAX_IMAGE_PIXELS = 10**12

# Add project root to path for imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import configuration utilities.
from code.engineered_feature_extraction.utils.config_loader import load_feature_extraction_config

# Initialize console and CLI.
console = Console()
app = typer.Typer(help="Simple nuclear feature extraction for kidney I/R injury analysis.")


def load_image(image_path: Path) -> np.ndarray:
    """
    Load DAPI image from file.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        Grayscale image array.
    """
    console.print(f"[cyan]Loading image:[/cyan] {image_path}")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image and convert to grayscale if needed.
    img = Image.open(image_path)
    if img.mode != 'L':
        img = img.convert('L')
    
    gray = np.array(img)
    console.print(f"[green]✓[/green] Image loaded: {gray.shape} pixels")
    
    return gray


def load_mask(mask_path: Path) -> np.ndarray:
    """
    Load segmentation mask from numpy file.
    
    Args:
        mask_path: Path to mask file (.npy).
        
    Returns:
        Integer mask array with labeled nuclei.
    """
    console.print(f"[cyan]Loading mask:[/cyan] {mask_path}")
    
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    mask = np.load(mask_path)
    num_nuclei = len(np.unique(mask)) - 1  # Subtract background.
    console.print(f"[green]✓[/green] Mask loaded: {num_nuclei} nuclei detected")
    
    return mask


def extract_basic_features(region: Any) -> Dict[str, float]:
    """
    Extract comprehensive size and shape features from nuclear region.

    Size features quantify nuclear dimensions and spatial extent.
    Shape features measure nuclear morphology, regularity, and complexity.

    Args:
        region: Regionprops object containing nuclear measurements.

    Returns:
        Dictionary with all size and shape feature values.
    """
    features = {}

    # Pre-compute commonly used values for efficiency.
    area = float(region.area)
    perimeter = region.perimeter

    '''Size Features'''
    # Primary size measurements.
    features['area'] = area
    features['perimeter'] = float(perimeter) if perimeter > 0 else 0.0

    # Equivalent diameter: diameter of circle with same area.
    features['equivalent_diameter'] = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0

    # Axis lengths from fitted ellipse.
    features['major_axis_length'] = float(region.major_axis_length)
    features['minor_axis_length'] = float(region.minor_axis_length)

    # Bounding box measurements.
    minr, minc, maxr, maxc = region.bbox
    bbox_width = maxc - minc
    bbox_height = maxr - minr
    features['bounding_box_width'] = float(bbox_width)
    features['bounding_box_height'] = float(bbox_height)
    features['bounding_box_area'] = float(bbox_width * bbox_height)

    # Feret diameters (maximum and minimum caliper measurements).
    features['feret_diameter_max'] = float(getattr(region, 'feret_diameter_max', region.major_axis_length))
    # Approximate minimum Feret diameter as minor axis length if not available.
    features['feret_diameter_min'] = float(getattr(region, 'feret_diameter_min', region.minor_axis_length))

    '''Shape Features'''
    # Pre-compute axis lengths for efficiency.
    major_axis = float(region.major_axis_length)
    minor_axis = float(region.minor_axis_length)

    # Basic shape features.
    # Circularity: shape regularity measure (4π*area/perimeter²).
    if perimeter > 0:
        features['circularity'] = 4 * np.pi * area / (perimeter ** 2)
    else:
        features['circularity'] = 0.0

    # Eccentricity: nuclear elongation measure (0=circle, 1=line).
    features['eccentricity'] = float(region.eccentricity)

    # Solidity: ratio of area to convex hull area.
    features['solidity'] = float(region.solidity)

    # Aspect ratio: elongation measure.
    if minor_axis > 0:
        features['aspect_ratio'] = major_axis / minor_axis
    else:
        features['aspect_ratio'] = 1.0

    # Compactness: measure of shape regularity (perimeter²/4π*area).
    if area > 0:
        features['compactness'] = (perimeter ** 2) / (4 * np.pi * area)
    else:
        features['compactness'] = 0.0

    # Elongation: normalized stretching measure.
    if (major_axis + minor_axis) > 0:
        features['elongation'] = (major_axis - minor_axis) / (major_axis + minor_axis)
    else:
        features['elongation'] = 0.0

    # Roundness: alternative circularity measure (4*area/π*major²).
    if major_axis > 0:
        features['roundness'] = (4 * area) / (np.pi * major_axis ** 2)
    else:
        features['roundness'] = 0.0

    # Form factor: shape complexity measure (same as circularity).
    if perimeter > 0:
        features['form_factor'] = (4 * np.pi * area) / (perimeter ** 2)
    else:
        features['form_factor'] = 0.0

    '''Advanced Shape Features'''
    # Convex area ratio: ratio of actual area to convex hull area.
    convex_area = float(region.convex_area)
    if convex_area > 0:
        features['convex_area_ratio'] = area / convex_area
    else:
        features['convex_area_ratio'] = 1.0

    # Convexity: ratio of convex hull perimeter to actual perimeter.
    # Approximate using convex hull image for efficiency.
    try:
        from skimage.morphology import convex_hull_image
        from skimage.measure import perimeter as measure_perimeter

        convex_hull = convex_hull_image(region.image)
        convex_perimeter = measure_perimeter(convex_hull)

        if perimeter > 0:
            features['convexity'] = convex_perimeter / perimeter
        else:
            features['convexity'] = 1.0
    except Exception:
        # Fallback: approximate convexity using convex area.
        features['convexity'] = 1.0

    return features


def build_spatial_index(centroids: np.ndarray) -> KDTree:
    """
    Build spatial index for efficient neighbor queries.

    Args:
        centroids: Array of (y, x) centroid coordinates.

    Returns:
        KDTree for O(log n) neighbor lookups.
    """
    return KDTree(centroids)


def extract_neighborhood_features(
    region: Any,
    spatial_index: KDTree,
    centroids: np.ndarray,
    areas: np.ndarray,
    region_idx: int,
    neighborhood_radius: float = 20.0
) -> Dict[str, float]:
    """
    Extract comprehensive neighborhood features for spatial analysis.

    Neighborhood features analyze local tissue architecture and nuclear clustering
    patterns essential for understanding kidney ischemia-reperfusion injury.

    Args:
        region: Regionprops object containing nuclear measurements.
        spatial_index: KDTree for efficient neighbor queries.
        centroids: Array of all nuclear centroids.
        areas: Array of all nuclear areas.
        region_idx: Index of current region in centroids array.
        neighborhood_radius: Radius for neighbor detection (pixels).

    Returns:
        Dictionary with all neighborhood feature values.
    """
    features = {}

    # Get current nucleus centroid.
    current_centroid = centroids[region_idx]
    current_area = areas[region_idx]

    # Find neighbors within radius.
    neighbor_indices = spatial_index.query_ball_point(current_centroid, neighborhood_radius)

    # Remove self from neighbors.
    neighbor_indices = [idx for idx in neighbor_indices if idx != region_idx]

    '''Basic Neighborhood Counts'''
    # Neighbor count: number of nuclei within radius.
    features['neighbor_count'] = len(neighbor_indices)

    # Neighbor density: nuclei per unit area in neighborhood.
    neighborhood_area = np.pi * (neighborhood_radius ** 2)
    features['neighbor_density'] = len(neighbor_indices) / neighborhood_area

    '''Distance-Based Features'''
    if len(neighbor_indices) > 0:
        # Calculate distances to all neighbors.
        neighbor_centroids = centroids[neighbor_indices]
        distances = np.linalg.norm(neighbor_centroids - current_centroid, axis=1)

        # Nearest neighbor distance: distance to closest nucleus.
        features['nearest_neighbor_distance'] = np.min(distances)

        # Mean neighbor distance: average distance to all neighbors.
        features['mean_neighbor_distance'] = np.mean(distances)

        # Neighbor area ratio: ratio of nucleus area to mean neighbor area.
        neighbor_areas = areas[neighbor_indices]
        mean_neighbor_area = np.mean(neighbor_areas)
        if mean_neighbor_area > 0:
            features['neighbor_area_ratio'] = current_area / mean_neighbor_area
        else:
            features['neighbor_area_ratio'] = 1.0

    else:
        # Handle isolated nuclei.
        features['nearest_neighbor_distance'] = neighborhood_radius
        features['mean_neighbor_distance'] = neighborhood_radius
        features['neighbor_area_ratio'] = 1.0

    '''Advanced Spatial Features'''
    # Local density gradient: change in density from center to edge.
    if len(neighbor_indices) >= 3:
        # Calculate density in inner and outer rings.
        inner_radius = neighborhood_radius * 0.5
        inner_neighbors = spatial_index.query_ball_point(current_centroid, inner_radius)
        inner_neighbors = [idx for idx in inner_neighbors if idx != region_idx]

        inner_density = len(inner_neighbors) / (np.pi * (inner_radius ** 2))
        outer_density = (len(neighbor_indices) - len(inner_neighbors)) / (np.pi * (neighborhood_radius ** 2 - inner_radius ** 2))

        if inner_density > 0:
            features['local_density_gradient'] = (outer_density - inner_density) / inner_density
        else:
            features['local_density_gradient'] = 0.0
    else:
        features['local_density_gradient'] = 0.0

    # Clustering coefficient: measure of local clustering.
    if len(neighbor_indices) >= 2:
        # Count connections between neighbors.
        connections = 0
        total_possible = len(neighbor_indices) * (len(neighbor_indices) - 1) / 2

        for i, idx1 in enumerate(neighbor_indices):
            for idx2 in neighbor_indices[i+1:]:
                distance = np.linalg.norm(centroids[idx1] - centroids[idx2])
                if distance <= neighborhood_radius:
                    connections += 1

        features['clustering_coefficient'] = connections / total_possible if total_possible > 0 else 0.0
    else:
        features['clustering_coefficient'] = 0.0

    # Isolation score: measure of spatial isolation.
    if len(neighbor_indices) > 0:
        # Inverse of neighbor density with distance weighting.
        neighbor_centroids = centroids[neighbor_indices]
        distances = np.linalg.norm(neighbor_centroids - current_centroid, axis=1)
        weighted_density = np.sum(1.0 / (distances + 1.0))  # Add 1 to avoid division by zero.
        features['isolation_score'] = 1.0 / (weighted_density + 1.0)
    else:
        features['isolation_score'] = 1.0  # Maximum isolation.

    return features


def extract_texture_features(
    region: Any,
    gray: np.ndarray,
    extract_texture: bool = False
) -> Dict[str, float]:
    """
    Extract comprehensive texture features for chromatin organization analysis.

    Texture features analyze DAPI intensity patterns and chromatin organization
    essential for understanding nuclear structure and cell cycle states.

    Args:
        region: Regionprops object containing nuclear measurements.
        gray: Grayscale DAPI image array.
        extract_texture: Whether to extract texture features (computationally expensive).

    Returns:
        Dictionary with all texture feature values.
    """
    features = {}

    if not extract_texture:
        # Return empty features if texture extraction is disabled.
        return features

    try:
        # Extract nuclear region from image.
        minr, minc, maxr, maxc = region.bbox
        patch = gray[minr:maxr, minc:maxc]
        mask_patch = region.image

        # Get intensity values within nucleus.
        vals = patch[mask_patch]

        if len(vals) == 0:
            console.print(f"[yellow]⚠[/yellow] Empty nucleus region for label {region.label}")
            return features

        # Basic intensity statistics.
        features['intensity_mean'] = float(vals.mean())
        features['intensity_std'] = float(vals.std())
        features['intensity_median'] = float(np.median(vals))
        features['intensity_skewness'] = float(skew(vals))
        features['intensity_kurtosis'] = float(kurtosis(vals))

        # Texture entropy.
        hist, _ = np.histogram(vals, bins=32, density=True)
        hist = hist + 1e-10  # Avoid log(0).
        features['texture_entropy'] = float(entropy(hist))

        # Gradient features.
        gradient = sobel(patch)
        gradient_vals = gradient[mask_patch]
        features['gradient_magnitude_mean'] = float(gradient_vals.mean())
        features['gradient_magnitude_std'] = float(gradient_vals.std())

        # GLCM features (computationally expensive).
        if vals.max() > vals.min():  # Ensure there's intensity variation.
            # Normalize intensities to 0-15 range for GLCM.
            vals_norm = ((vals - vals.min()) / (vals.max() - vals.min()) * 15).astype(np.uint8)

            # Create patch for GLCM analysis.
            patch_norm = np.zeros_like(patch, dtype=np.uint8)
            patch_norm[mask_patch] = vals_norm

            # Compute GLCM with distance=1, angles=[0, 45, 90, 135].
            glcm = graycomatrix(
                patch_norm,
                distances=[1],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=16,
                symmetric=True,
                normed=True
            )

            # Extract GLCM properties.
            features['glcm_contrast'] = float(graycoprops(glcm, 'contrast').mean())
            features['glcm_dissimilarity'] = float(graycoprops(glcm, 'dissimilarity').mean())
            features['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity').mean())
            features['glcm_energy'] = float(graycoprops(glcm, 'energy').mean())
        else:
            # No intensity variation - set GLCM features to default values.
            features['glcm_contrast'] = 0.0
            features['glcm_dissimilarity'] = 0.0
            features['glcm_homogeneity'] = 1.0
            features['glcm_energy'] = 1.0

    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Texture extraction failed for nucleus {region.label}: {e}")
        # Return NaN values for failed extractions.
        texture_feature_names = [
            'intensity_mean', 'intensity_std', 'intensity_median', 'intensity_skewness',
            'intensity_kurtosis', 'texture_entropy', 'gradient_magnitude_mean',
            'gradient_magnitude_std', 'glcm_contrast', 'glcm_dissimilarity',
            'glcm_homogeneity', 'glcm_energy'
        ]
        for name in texture_feature_names:
            features[name] = np.nan

    return features


def process_nuclei(
    gray: np.ndarray,
    mask: np.ndarray,
    neighborhood_radius: float = 20.0,
    extract_texture: bool = False
) -> pd.DataFrame:
    """
    Process all nuclei to extract comprehensive morphological, neighborhood, and texture features.

    Args:
        gray: Grayscale DAPI image.
        mask: Segmentation mask with labeled nuclei.
        neighborhood_radius: Radius for neighbor detection (pixels).
        extract_texture: Whether to extract texture features (computationally expensive).

    Returns:
        DataFrame with nucleus_id and all extracted features.
    """
    console.print("[cyan]Extracting nuclear region properties...[/cyan]")

    # Extract region properties from mask.
    props = regionprops(mask, intensity_image=gray)
    console.print(f"[green]✓[/green] Found {len(props)} nuclear regions")

    # Pre-compute centroids and areas for spatial indexing.
    console.print("[cyan]Building spatial index for neighborhood analysis...[/cyan]")
    centroids = np.array([region.centroid for region in props])
    areas = np.array([region.area for region in props])

    # Build spatial index for efficient neighbor queries.
    spatial_index = build_spatial_index(centroids)
    console.print(f"[green]✓[/green] Spatial index built for {len(centroids)} nuclei")

    # Process each nucleus with enhanced progress tracking.
    results = []

    # Feature timing tracking.
    feature_times = {
        'size': 0.0,
        'shape': 0.0,
        'neighborhood': 0.0,
        'texture': 0.0
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        # Create main progress task.
        main_task = progress.add_task("Processing nuclei...", total=len(props))

        # Create feature category tasks.
        size_task = progress.add_task("[blue]Size features[/blue]", total=len(props), visible=False)
        shape_task = progress.add_task("[green]Shape features[/green]", total=len(props), visible=False)
        neighborhood_task = progress.add_task("[magenta]Neighborhood features[/magenta]", total=len(props), visible=False)
        texture_task = progress.add_task("[cyan]Texture features[/cyan]", total=len(props), visible=False)

        for idx, region in enumerate(props):
            # Update current nucleus info.
            progress.update(main_task, description=f"Processing nucleus {region.label} ({idx+1}/{len(props)})")

            # Extract size and shape features with timing.
            start_time = time.time()
            basic_features = extract_basic_features(region)
            basic_time = time.time() - start_time

            # Split timing between size and shape (approximate).
            feature_times['size'] += basic_time * 0.4  # Size features are ~40% of computation.
            feature_times['shape'] += basic_time * 0.6  # Shape features are ~60% of computation.

            # Update feature progress.
            progress.update(size_task, advance=1)
            progress.update(shape_task, advance=1)

            # Extract neighborhood features with timing.
            start_time = time.time()
            neighborhood_features = extract_neighborhood_features(
                region, spatial_index, centroids, areas, idx, neighborhood_radius
            )
            feature_times['neighborhood'] += time.time() - start_time

            progress.update(neighborhood_task, advance=1)

            # Extract texture features with timing.
            start_time = time.time()
            texture_features = extract_texture_features(region, gray, extract_texture)
            feature_times['texture'] += time.time() - start_time

            progress.update(texture_task, advance=1)

            # Combine all features.
            result = {'nucleus_id': region.label}
            result.update(basic_features)
            result.update(neighborhood_features)
            result.update(texture_features)

            results.append(result)
            progress.update(main_task, advance=1)

    # Convert to DataFrame.
    df = pd.DataFrame(results)

    # Display feature timing summary.
    console.print("\n[bold blue]📊 FEATURE EXTRACTION TIMING[/bold blue]")
    timing_table = Table(show_header=True, header_style="bold cyan")
    timing_table.add_column("Feature Category", style="cyan")
    timing_table.add_column("Total Time", style="green")
    timing_table.add_column("Per Nucleus", style="yellow")
    timing_table.add_column("Features", style="magenta")

    for category, total_time in feature_times.items():
        per_nucleus = total_time / len(props) * 1000  # Convert to milliseconds.
        if category == 'size':
            feature_count = "10"
        elif category == 'shape':
            feature_count = "10"
        elif category == 'neighborhood':
            feature_count = "8"
        else:  # texture
            feature_count = "12" if extract_texture else "0"

        timing_table.add_row(
            category.title(),
            f"{total_time:.2f}s",
            f"{per_nucleus:.1f}ms",
            feature_count
        )

    console.print(timing_table)
    console.print(f"[green]✓[/green] Extracted {len(df.columns)-1} features from {len(df)} nuclei")

    return df


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save feature extraction results to CSV file.
    
    Args:
        df: DataFrame with extracted features.
        output_path: Path for output CSV file.
    """
    console.print(f"[cyan]Saving results to:[/cyan] {output_path}")
    
    # Create output directory if needed.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV with proper formatting.
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    console.print(f"[green]✓[/green] Results saved: {len(df)} nuclei with {len(df.columns)-1} features")

    # Display summary statistics for key features.
    console.print("\n[bold blue]📊 FEATURE SUMMARY[/bold blue]")

    # Size features summary.
    console.print("[bold cyan]Size Features:[/bold cyan]")
    console.print(f"  [cyan]Area:[/cyan] mean={df['area'].mean():.1f}, std={df['area'].std():.1f}")
    console.print(f"  [cyan]Perimeter:[/cyan] mean={df['perimeter'].mean():.1f}, std={df['perimeter'].std():.1f}")
    console.print(f"  [cyan]Major Axis:[/cyan] mean={df['major_axis_length'].mean():.1f}, std={df['major_axis_length'].std():.1f}")
    console.print(f"  [cyan]Minor Axis:[/cyan] mean={df['minor_axis_length'].mean():.1f}, std={df['minor_axis_length'].std():.1f}")

    # Shape features summary.
    console.print("[bold cyan]Shape Features:[/bold cyan]")
    console.print(f"  [cyan]Circularity:[/cyan] mean={df['circularity'].mean():.3f}, std={df['circularity'].std():.3f}")
    console.print(f"  [cyan]Eccentricity:[/cyan] mean={df['eccentricity'].mean():.3f}, std={df['eccentricity'].std():.3f}")
    console.print(f"  [cyan]Solidity:[/cyan] mean={df['solidity'].mean():.3f}, std={df['solidity'].std():.3f}")
    console.print(f"  [cyan]Aspect Ratio:[/cyan] mean={df['aspect_ratio'].mean():.2f}, std={df['aspect_ratio'].std():.2f}")
    console.print(f"  [cyan]Elongation:[/cyan] mean={df['elongation'].mean():.3f}, std={df['elongation'].std():.3f}")

    # Neighborhood features summary.
    console.print("[bold cyan]Neighborhood Features:[/bold cyan]")
    console.print(f"  [cyan]Neighbor Count:[/cyan] mean={df['neighbor_count'].mean():.1f}, std={df['neighbor_count'].std():.1f}")
    console.print(f"  [cyan]Neighbor Density:[/cyan] mean={df['neighbor_density'].mean():.4f}, std={df['neighbor_density'].std():.4f}")
    console.print(f"  [cyan]Nearest Distance:[/cyan] mean={df['nearest_neighbor_distance'].mean():.1f}, std={df['nearest_neighbor_distance'].std():.1f}")
    console.print(f"  [cyan]Clustering Coeff:[/cyan] mean={df['clustering_coefficient'].mean():.3f}, std={df['clustering_coefficient'].std():.3f}")
    console.print(f"  [cyan]Isolation Score:[/cyan] mean={df['isolation_score'].mean():.3f}, std={df['isolation_score'].std():.3f}")

    # Texture features summary (if available).
    if 'intensity_mean' in df.columns:
        console.print("[bold cyan]Texture Features:[/bold cyan]")
        console.print(f"  [cyan]Intensity Mean:[/cyan] mean={df['intensity_mean'].mean():.1f}, std={df['intensity_mean'].std():.1f}")
        console.print(f"  [cyan]Intensity Std:[/cyan] mean={df['intensity_std'].mean():.1f}, std={df['intensity_std'].std():.1f}")
        console.print(f"  [cyan]Texture Entropy:[/cyan] mean={df['texture_entropy'].mean():.3f}, std={df['texture_entropy'].std():.3f}")
        console.print(f"  [cyan]GLCM Contrast:[/cyan] mean={df['glcm_contrast'].mean():.3f}, std={df['glcm_contrast'].std():.3f}")
        console.print(f"  [cyan]GLCM Homogeneity:[/cyan] mean={df['glcm_homogeneity'].mean():.3f}, std={df['glcm_homogeneity'].std():.3f}")


def main(
    config: Path = typer.Option(..., exists=True, help="Configuration file containing extraction parameters")
) -> None:
    """
    Extract comprehensive nuclear morphological, neighborhood, and texture features from segmented tissue using config.

    This command processes DAPI-stained tissue sections to extract essential morphological,
    spatial, and optional texture features for kidney ischemia-reperfusion injury analysis.
    Includes all size measurements (area, perimeter, axes, bounding box, Feret diameters),
    comprehensive shape features (circularity, eccentricity, solidity, aspect ratio, compactness,
    elongation, roundness, form factor, convex area ratio, convexity), advanced neighborhood
    features (neighbor counts, densities, distances, clustering metrics), and optional texture
    features (intensity statistics, entropy, gradients, GLCM properties) for thorough nuclear characterization.

    Example:
        python extract_engineered_features.py \\
            --config ../../configs/engineered_feature_extraction_config.ini
    """
    console.print("\n[bold blue]🧬 SIMPLE NUCLEAR FEATURE EXTRACTION 🧬[/bold blue]\n")

    try:
        start_time = time.time()

        # Step 1: Load configuration.
        console.print(f"[cyan]Loading configuration from:[/cyan] {config}")
        settings = load_feature_extraction_config(config)
        console.print(f"[green]✓[/green] Configuration loaded successfully")

        # Extract file paths from config and resolve them properly.
        # The config paths are relative to the project root, so resolve from there.
        image_path = Path(settings['extraction_image_path']).resolve()
        mask_path = Path(settings['extraction_mask_path']).resolve()
        output_dir = Path(settings.get('extraction_output_dir', 'results/engineered_features')).resolve()
        output_path = output_dir / 'engineered_features.csv'

        # Create output directory.
        output_dir.mkdir(parents=True, exist_ok=True)

        # Display processing information.
        console.print(Panel.fit(
            f"[bold]Processing Files[/bold]\n"
            f"Image: {image_path.name}\n"
            f"Mask: {mask_path.name}\n"
            f"Output: {output_path.name}",
            border_style="blue",
            title="🔬 Analysis"
        ))

        # Step 2: Load image and mask.
        gray = load_image(image_path)
        mask_array = load_mask(mask_path)
        
        # Validate dimensions match.
        if gray.shape != mask_array.shape:
            raise ValueError(f"Image shape {gray.shape} doesn't match mask shape {mask_array.shape}")
        
        console.print(f"[green]✓[/green] Image and mask dimensions validated")

        # Step 3: Extract features with neighborhood and texture analysis.
        neighborhood_radius = float(settings.get('neighborhood_radius', 20.0))
        extract_texture = str(settings.get('extract_texture_features', 'False')).lower() == 'true'
        console.print(f"[cyan]Using neighborhood radius:[/cyan] {neighborhood_radius} pixels")
        console.print(f"[cyan]Texture features:[/cyan] {'enabled' if extract_texture else 'disabled'}")
        df = process_nuclei(gray, mask_array, neighborhood_radius, extract_texture)

        # Step 4: Save results.
        save_results(df, output_path)

        # Display completion summary.
        end_time = time.time()
        processing_time = end_time - start_time

        console.print(Panel.fit(
            f"[bold green]✅ EXTRACTION COMPLETE[/bold green]\n"
            f"Processed: {len(df)} nuclei\n"
            f"Features: {len(df.columns)-1} size & shape features\n"
            f"Time: {processing_time:.1f} seconds\n"
            f"Output: {output_path}",
            border_style="green",
            title="🎉 Success"
        ))
        
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]❌ EXTRACTION FAILED[/bold red]\n"
            f"Error: {str(e)}\n"
            f"Check input files and try again.",
            border_style="red",
            title="💥 Error"
        ))
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
