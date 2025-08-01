"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: cluster_engineered_features.py
Description:
    Simplified config-driven clustering of engineered nuclear features with GPU-accelerated overlay generation.
    Implements a streamlined interface that loads all parameters from a configuration file, ensuring complete
    reproducibility and parameter traceability. Features memory-efficient processing, vibrant color palettes,
    and advanced overlay utilities for publication-quality scientific visualizations of kidney I/R injury analysis.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scikit-learn, joblib, matplotlib, PIL, seaborn, scipy.
    • argparse for simplified CLI interface.
    • rich for progress tracking and console output.
    • Advanced overlay utilities with GPU acceleration support.

Usage:
    python cluster_engineered_features.py \
        --config ../../configs/engineered_feature_extraction_config.ini

Arguments:
    --config           Path to configuration file containing all clustering parameters.

Configuration File Requirements:
    The configuration file must contain all necessary parameters including:
    • Input file paths (features CSV, image, mask)
    • Output directory path for clustering results
    • Clustering parameters (number of clusters, methods, seeds)
    • Visualization settings (file names, overlay options)
    • Color and overlay settings (alpha, saturation, GPU options)

Inputs (specified in config file):
    • features_csv_path    Path to comprehensive nuclear feature matrix.
    • image_path          Path to original DAPI-stained tissue image.
    • mask_path           Path to nuclear segmentation masks.

Outputs:
    • engineered_feature_clustering_config_used.ini    Copied configuration for audit trail.
    • nuclear_clusters.csv                             Cluster assignments per nucleus.
    • kmeans_model.joblib                              Trained clustering model.
    • scaler.joblib                                    StandardScaler for feature normalization.
    • cluster_overlay.tif                              GPU-accelerated cluster overlay on tissue.
    • pca_clusters.png                                 PCA scatter plot with cluster colors.
    • feature_importance.png                           Feature importance for clustering.
    • cluster_statistics.csv                           Statistical summary per cluster.

Key Features:
    • Simplified config-driven interface with complete parameter traceability.
    • Automatic configuration file copying for reproducibility audit trail.
    • GPU-accelerated overlay generation with memory-efficient tile processing.
    • Enhanced vibrant color palette with 50+ neon-like colors for maximum visual impact.
    • Memory-efficient streaming processing for datasets of any size.
    • Publication-quality visualizations with scientific formatting and high contrast.
    • Feature importance analysis for biological interpretation.
    • Comprehensive statistical analysis and cluster validation.
    • Optimized for kidney I/R injury research with scientific color standards.

Notes:
    • All parameters loaded from configuration file - no hardcoded values.
    • Configuration file automatically copied to output directory for audit purposes.
    • Supports comprehensive nuclear feature sets (morphological, intensity, texture, neighborhood).
    • Color palette optimized for scientific visualization with WCAG contrast compliance.
    • GPU acceleration provides significant speedup for large image overlay generation.
    • Memory usage scales efficiently through streaming and tile-based processing.
"""
import traceback
import tempfile
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import matplotlib.pyplot as plt
import random
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table
import shutil

# Initialize console for output.
console = Console()

# Import color generation utilities.
from utils.generate_contrast_colors import generate_color_palette, colors_to_hex_list
from utils.color_config import ColorConfig, load_color_config

# Import feature extraction configuration utilities.
from utils.config_loader import load_feature_extraction_config

# Import overlay utilities for memory-efficient large image processing.
overlay_utils_path = Path(__file__).parent.parent / 'nuclei_segmentation' / 'utils'
sys.path.insert(0, str(overlay_utils_path))

import overlay_masks
overlay = overlay_masks.overlay
OverlayConfig = overlay_masks.OverlayConfig


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
    output_dir = Path(config.get('clustering_output_dir', 'results/clustering_analysis'))

    # Create output directory if it doesn't exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/green] Created output directory: [bold]{output_dir}[/bold]")

    # Copy config file to output directory for audit purposes.
    copied_config_path = output_dir / 'engineered_feature_clustering_config_used.ini'
    shutil.copy2(config_path, copied_config_path)
    console.print(f"[green]✓[/green] Configuration copied to: [bold]{copied_config_path}[/bold]")

    return config, output_dir


def validate_config_parameters(config: Dict) -> None:
    """
    Validate that all required configuration parameters are present.

    Args:
        config: Configuration dictionary loaded from file.

    Raises:
        ValueError: If required parameters are missing or invalid.

    This function ensures the configuration file contains all necessary parameters
    for clustering analysis, providing clear error messages for missing values.
    """
    console.print("[cyan]Validating configuration parameters...[/cyan]")

    # Required input file paths.
    required_paths = {
        'features_csv_path': 'Path to engineered features CSV file',
        'image_path': 'Path to original tissue image',
        'mask_path': 'Path to segmentation mask file',
        'clustering_output_dir': 'Output directory for clustering results'
    }

    missing_paths = []
    for param, description in required_paths.items():
        if param not in config or not config[param]:
            missing_paths.append(f"  • {param}: {description}")

    if missing_paths:
        console.print(f"[red]✗[/red] Missing required input file paths in configuration:")
        for missing in missing_paths:
            console.print(f"[red]{missing}[/red]")
        raise ValueError("Configuration file missing required input file paths")

    # Required clustering parameters.
    required_params = {
        'default_clusters': 'Number of clusters for analysis',
        'clustering_seed': 'Random seed for reproducibility',
        'clustering_batch_size': 'Batch size for processing'
    }

    missing_params = []
    for param, description in required_params.items():
        if param not in config:
            missing_params.append(f"  • {param}: {description}")

    if missing_params:
        console.print(f"[red]✗[/red] Missing required clustering parameters in configuration:")
        for missing in missing_params:
            console.print(f"[red]{missing}[/red]")
        raise ValueError("Configuration file missing required clustering parameters")

    console.print("[green]✓[/green] All required configuration parameters validated")


def get_file_paths_from_config(config: Dict) -> Tuple[Path, Path, Path]:
    """
    Extract and validate file paths from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (features_path, image_path, mask_path).

    Raises:
        FileNotFoundError: If any required files don't exist.

    This function extracts file paths from the configuration and validates
    that all required input files exist before starting the analysis.
    """
    console.print("[cyan]Extracting file paths from configuration...[/cyan]")

    # Extract paths from config.
    features_path = Path(config['features_csv_path'])
    image_path = Path(config['image_path'])
    mask_path = Path(config['mask_path'])

    # Validate files exist.
    missing_files = []
    if not features_path.exists():
        missing_files.append(f"Features CSV: {features_path}")
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
    console.print(f"  • [blue]Features[/blue]: {features_path}")
    console.print(f"  • [blue]Image[/blue]: {image_path}")
    console.print(f"  • [blue]Mask[/blue]: {mask_path}")

    return features_path, image_path, mask_path


"""UTILITY FUNCTIONS"""


def configure_logging():
    """Configure root logger for scientific analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def set_global_seed(seed: int):
    """Set global random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Global random seed set to {seed}")


def load_nuclear_features(features_path: Path) -> pd.DataFrame:
    """
    Load nuclear features from CSV file with validation.
    
    Args:
        features_path: Path to CSV file containing nuclear features.
        
    Returns:
        DataFrame with nuclear features and metadata.
        
    This function loads and validates the nuclear feature matrix, ensuring
    all required columns are present for clustering analysis.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    console.print(f"[cyan]Loading nuclear features from {features_path}...[/cyan]")
    
    df = pd.read_csv(features_path)
    
    # Validate essential columns.
    required_cols = ['label']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Identify feature columns (exclude metadata).
    metadata_cols = ['label', 'centroid_x', 'centroid_y']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    console.print(f"[green]✓[/green] Loaded {len(df)} nuclei with {len(feature_cols)} features")
    
    # Display feature categories if recognizable.
    morphological_features = [col for col in feature_cols if any(keyword in col.lower() 
                             for keyword in ['area', 'perimeter', 'axis', 'aspect', 'circularity', 
                                           'eccentricity', 'solidity', 'feret', 'roughness', 'fractal'])]
    
    intensity_features = [col for col in feature_cols if any(keyword in col.lower() 
                         for keyword in ['intensity', 'mean', 'std', 'median', 'skewness', 'kurtosis'])]
    
    texture_features = [col for col in feature_cols if any(keyword in col.lower() 
                       for keyword in ['texture', 'entropy', 'lbp', 'glcm', 'gradient'])]
    
    neighborhood_features = [col for col in feature_cols if any(keyword in col.lower() 
                            for keyword in ['neighborhood', 'cluster', 'distance', 'nearest', 'alignment'])]
    
    # Create feature summary table.
    feature_table = Table(title="Nuclear Feature Categories")
    feature_table.add_column("Category", style="cyan")
    feature_table.add_column("Count", style="green")
    feature_table.add_column("Examples", style="yellow")
    
    feature_table.add_row("Morphological", str(len(morphological_features)), 
                         ", ".join(morphological_features[:3]) + "..." if len(morphological_features) > 3 else ", ".join(morphological_features))
    feature_table.add_row("Intensity", str(len(intensity_features)), 
                         ", ".join(intensity_features[:3]) + "..." if len(intensity_features) > 3 else ", ".join(intensity_features))
    feature_table.add_row("Texture", str(len(texture_features)), 
                         ", ".join(texture_features[:3]) + "..." if len(texture_features) > 3 else ", ".join(texture_features))
    feature_table.add_row("Neighborhood", str(len(neighborhood_features)), 
                         ", ".join(neighborhood_features[:3]) + "..." if len(neighborhood_features) > 3 else ", ".join(neighborhood_features))
    
    console.print(feature_table)
    
    return df


def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Prepare feature matrix for clustering analysis.
    
    Args:
        df: DataFrame with nuclear features.
        
    Returns:
        Tuple of (feature_matrix, feature_names, labels).
        
    This function extracts the numerical feature matrix, handles missing values,
    and prepares data for clustering while preserving nuclear labels.
    """
    # Extract nuclear labels.
    nuclear_labels = df['label'].values

    # Identify feature columns (exclude metadata).
    metadata_cols = ['label', 'centroid_x', 'centroid_y']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Extract feature matrix.
    feature_matrix = df[feature_cols].values
    
    # Handle missing values.
    if np.any(np.isnan(feature_matrix)):
        console.print("[yellow]⚠[/yellow] Found missing values, filling with column medians")
        
        for i in range(feature_matrix.shape[1]):
            col_data = feature_matrix[:, i]
            if np.any(np.isnan(col_data)):
                median_val = np.nanmedian(col_data)
                feature_matrix[np.isnan(col_data), i] = median_val
    
    console.print(f"[green]✓[/green] Prepared feature matrix: {feature_matrix.shape}")
    
    return feature_matrix, feature_cols, nuclear_labels


def choose_optimal_k(features: np.ndarray, k_max: int, criterion: str, 
                    sample_size: int = 5000) -> Tuple[int, pd.DataFrame]:
    """
    Choose optimal number of clusters using silhouette or Davies-Bouldin index.
    
    Args:
        features: Feature matrix for clustering.
        k_max: Maximum number of clusters to test.
        criterion: Optimization criterion ('silhouette' or 'dbi').
        sample_size: Sample size for evaluation (memory efficiency).
        
    Returns:
        Tuple of (optimal_k, scores_dataframe).
        
    This function evaluates different cluster numbers to find the optimal
    configuration based on internal clustering validation metrics.
    """
    console.print(f"[cyan]Evaluating optimal K using {criterion} criterion...[/cyan]")
    
    # Sample data for efficiency.
    if len(features) > sample_size:
        idx = np.random.RandomState(42).choice(len(features), sample_size, replace=False)
        sample_features = features[idx]
    else:
        sample_features = features
    
    scores = []
    
    with Progress() as progress:
        task = progress.add_task(f"Testing K values (2 to {k_max})...", total=k_max-1)
        
        for k in range(2, k_max + 1):
            # Fit MiniBatchKMeans.
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
            cluster_labels = kmeans.fit_predict(sample_features)
            
            # Calculate score.
            if criterion == 'silhouette':
                score = silhouette_score(sample_features, cluster_labels)
            else:  # davies_bouldin
                score = davies_bouldin_score(sample_features, cluster_labels)
            
            scores.append({'k': k, criterion: score})
            progress.update(task, advance=1)
    
    # Create results DataFrame.
    scores_df = pd.DataFrame(scores)
    
    # Find optimal K.
    if criterion == 'silhouette':
        optimal_k = int(scores_df.loc[scores_df[criterion].idxmax(), 'k'])
    else:  # davies_bouldin (lower is better)
        optimal_k = int(scores_df.loc[scores_df[criterion].idxmin(), 'k'])
    
    console.print(f"[green]✓[/green] Optimal K selected: {optimal_k}")
    
    return optimal_k, scores_df


def stream_scale_features(features: np.ndarray, batch_size: int) -> StandardScaler:
    """
    Scale features using streaming StandardScaler for memory efficiency.
    
    Args:
        features: Feature matrix to scale.
        batch_size: Batch size for streaming processing.
        
    Returns:
        Fitted StandardScaler object.
        
    This function fits a StandardScaler using partial_fit to handle large
    datasets that may not fit in memory simultaneously.
    """
    console.print("[cyan]Scaling features with streaming StandardScaler...[/cyan]")
    
    scaler = StandardScaler()
    n_samples = features.shape[0]
    
    with Progress() as progress:
        task = progress.add_task("Fitting scaler...", total=n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_features = features[i:batch_end]
            scaler.partial_fit(batch_features)
            progress.update(task, advance=batch_end - i)
    
    console.print("[green]✓[/green] Feature scaling completed")
    
    return scaler


def stream_cluster_features(features: np.ndarray, scaler: StandardScaler, 
                          n_clusters: int, batch_size: int, seed: int) -> MiniBatchKMeans:
    """
    Cluster features using streaming MiniBatchKMeans.
    
    Args:
        features: Feature matrix to cluster.
        scaler: Fitted StandardScaler for normalization.
        n_clusters: Number of clusters.
        batch_size: Batch size for streaming processing.
        seed: Random seed for reproducibility.
        
    Returns:
        Fitted MiniBatchKMeans model.
        
    This function performs clustering using MiniBatchKMeans with streaming
    processing to handle large datasets efficiently.
    """
    console.print(f"[cyan]Clustering into {n_clusters} clusters...[/cyan]")
    
    # Initialize MiniBatchKMeans.
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=batch_size)
    
    n_samples = features.shape[0]
    
    with Progress() as progress:
        task = progress.add_task("Training clustering model...", total=n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_features = features[i:batch_end]
            
            # Scale and fit batch.
            scaled_batch = scaler.transform(batch_features)
            kmeans.partial_fit(scaled_batch)
            
            progress.update(task, advance=batch_end - i)
    
    console.print("[green]✓[/green] Clustering model trained")
    
    return kmeans


def predict_cluster_labels(features: np.ndarray, scaler: StandardScaler, 
                          kmeans: MiniBatchKMeans, batch_size: int) -> np.ndarray:
    """
    Predict cluster labels for all features using streaming processing.
    
    Args:
        features: Feature matrix for prediction.
        scaler: Fitted StandardScaler.
        kmeans: Fitted MiniBatchKMeans model.
        batch_size: Batch size for streaming processing.
        
    Returns:
        Array of cluster labels.
        
    This function applies the trained clustering model to predict cluster
    assignments for all nuclei using memory-efficient batch processing.
    """
    console.print("[cyan]Predicting cluster labels...[/cyan]")
    
    n_samples = features.shape[0]
    cluster_labels = np.zeros(n_samples, dtype=int)
    
    with Progress() as progress:
        task = progress.add_task("Predicting labels...", total=n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_features = features[i:batch_end]
            
            # Scale and predict batch.
            scaled_batch = scaler.transform(batch_features)
            batch_labels = kmeans.predict(scaled_batch)
            
            cluster_labels[i:batch_end] = batch_labels
            progress.update(task, advance=batch_end - i)
    
    console.print("[green]✓[/green] Cluster labels predicted")

    return cluster_labels


"""VISUALIZATION FUNCTIONS"""


def create_cluster_mask(mask_path: Path, nuclear_labels: np.ndarray,
                       cluster_labels: np.ndarray, output_path: Path) -> Path:
    """
    Create a cluster mask by mapping nuclear labels to cluster assignments.

    Args:
        mask_path: Path to original segmentation mask with nuclear labels.
        nuclear_labels: Array of nuclear labels from feature extraction.
        cluster_labels: Array of cluster assignments for each nucleus.
        output_path: Path where to save the cluster mask.

    Returns:
        Path to the generated cluster mask file.

    This function creates a new mask where each pixel value represents the cluster
    assignment of the corresponding nucleus. This enables memory-efficient overlay
    generation for very large images using the existing overlay utilities.
    """
    console.print("[cyan]Creating cluster mask for memory-efficient overlay...[/cyan]")

    # Load original segmentation mask.
    original_mask = np.load(mask_path, mmap_mode='r')

    # Create lookup table for nuclear label to cluster mapping.
    max_nuclear_label = int(original_mask.max())
    cluster_lut = np.zeros(max_nuclear_label + 1, dtype=np.int32)

    # Populate lookup table.
    for nuclear_label, cluster_id in zip(nuclear_labels, cluster_labels):
        if 0 <= nuclear_label <= max_nuclear_label:
            cluster_lut[int(nuclear_label)] = int(cluster_id) + 1  # +1 to avoid background

    # Apply lookup table to create cluster mask.
    # For memory efficiency, process in chunks if the mask is very large.
    cluster_mask_shape = original_mask.shape
    cluster_mask = np.zeros(cluster_mask_shape, dtype=np.int32)

    # Process in chunks to handle very large masks.
    chunk_size = 10000  # Process 10k rows at a time.

    with Progress() as progress:
        task = progress.add_task("Creating cluster mask...", total=cluster_mask_shape[0])

        for start_row in range(0, cluster_mask_shape[0], chunk_size):
            end_row = min(start_row + chunk_size, cluster_mask_shape[0])

            # Load chunk of original mask.
            mask_chunk = original_mask[start_row:end_row]

            # Apply cluster mapping.
            cluster_mask[start_row:end_row] = cluster_lut[mask_chunk]

            progress.update(task, advance=end_row - start_row)

    # Save cluster mask.
    np.save(output_path, cluster_mask)

    console.print(f"[green]✓[/green] Cluster mask saved → {output_path}")

    return output_path


def create_cluster_overlay_advanced(image_path: Path, mask_path: Path, nuclear_labels: np.ndarray,
                                  cluster_labels: np.ndarray, color_palette: Dict[int, Tuple[int, int, int, int]],
                                  output_path: Path, tile_size: int = 1024, workers: str = "auto",
                                  alpha: float = 0.4, gpu: bool = True,
                                  memory_limit_mb: int = 8192) -> None:
    """
    Create memory-efficient overlay visualization of clusters on tissue image.

    Args:
        image_path: Path to original tissue image.
        mask_path: Path to segmentation mask.
        nuclear_labels: Array of nuclear labels.
        cluster_labels: Array of cluster assignments.
        color_palette: Dictionary mapping cluster to RGBA color.
        output_path: Path for output overlay image.
        tile_size: Tile size for memory-efficient processing.
        workers: Number of worker processes ("auto" or integer).
        alpha: Alpha blending factor for overlay transparency.
        gpu: Enable GPU acceleration if available.
        memory_limit_mb: GPU memory limit in MB.

    This function creates a publication-quality overlay showing cluster assignments
    mapped onto the original tissue morphology using memory-efficient tile-based
    processing that can handle gigantic images without memory issues.
    """

    console.print("[cyan]Creating advanced cluster overlay visualization...[/cyan]")

    try:
        # Create temporary cluster mask.
        with tempfile.NamedTemporaryFile(suffix='_cluster_mask.npy', delete=False) as tmp_file:
            cluster_mask_path = Path(tmp_file.name)

        # Generate cluster mask from nuclear labels and cluster assignments.
        create_cluster_mask(mask_path, nuclear_labels, cluster_labels, cluster_mask_path)

        # Create overlay configuration with cluster-specific color mapping.
        config = OverlayConfig(
            tile_size=tile_size,
            workers=workers,
            alpha=alpha,
            seed=42,  # Use fixed seed for reproducible colors.
            enable_gpu=gpu,
            memory_limit_mb=memory_limit_mb,
            batch_size=4  # Conservative batch size for stability.
        )

        # Use the advanced overlay function for memory-efficient processing.
        overlay(
            image_path=image_path,
            mask_path=cluster_mask_path,
            output_path=output_path,
            config=config
        )

        # Count colored pixels for validation.
        cluster_mask = np.load(cluster_mask_path, mmap_mode='r')
        colored_pixels = np.sum(cluster_mask > 0)
        unique_clusters = len(np.unique(cluster_mask[cluster_mask > 0]))

        console.print(f"[green]✓[/green] Advanced overlay saved: {colored_pixels} colored pixels, {unique_clusters} clusters → {output_path}")

    finally:
        # Clean up temporary cluster mask with retry logic.
        if 'cluster_mask_path' in locals() and cluster_mask_path.exists():
            try:
                cluster_mask_path.unlink()
            except PermissionError:
                # File might still be in use by memory mapping, try again after a short delay.
                import time
                time.sleep(0.5)
                try:
                    cluster_mask_path.unlink()
                except PermissionError:
                    console.print(f"[yellow]⚠[/yellow] Could not delete temporary file: {cluster_mask_path}")
                    console.print("[yellow]⚠[/yellow] File will be cleaned up automatically by the system.")

def create_pca_visualization(features: np.ndarray, cluster_labels: np.ndarray,
                           feature_names: List[str], color_palette: Dict[int, Tuple[int, int, int, int]],
                           output_path: Path, sample_size: int = 5000) -> None:
    """
    Create PCA scatter plot visualization of clusters.

    Args:
        features: Scaled feature matrix.
        cluster_labels: Array of cluster assignments.
        feature_names: List of feature names.
        color_palette: Dictionary mapping cluster to RGBA color.
        output_path: Path for output PCA plot.
        sample_size: Sample size for visualization (memory efficiency).

    This function creates a publication-quality PCA plot showing cluster
    separation in the reduced feature space for validation and interpretation.
    """
    console.print("[cyan]Creating PCA visualization...[/cyan]")

    # Sample data for visualization efficiency.
    if len(features) > sample_size:
        idx = np.random.RandomState(42).choice(len(features), sample_size, replace=False)
        sample_features = features[idx]
        sample_labels = cluster_labels[idx]
    else:
        sample_features = features
        sample_labels = cluster_labels

    # Perform PCA.
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(sample_features)

    # Convert color palette to hex for matplotlib.
    hex_colors = []
    for i in range(len(color_palette)):
        if i in color_palette:
            r, g, b, a = color_palette[i]
            hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
        else:
            hex_colors.append("#000000")  # Fallback black

    # Create high-quality plot.
    plt.figure(figsize=(14, 10))
    plt.style.use('dark_background')

    # Create scatter plot.
    scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1],
                         c=[hex_colors[label] for label in sample_labels],
                         s=60, alpha=0.7, edgecolors='black', linewidth=0.3)

    # Add titles and labels.
    plt.title('PCA Visualization of Nuclear Feature Clusters',
              fontsize=20, fontweight='bold', color='white', pad=25)
    plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.2%})',
               fontsize=14, color='white', fontweight='bold')
    plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.2%})',
               fontsize=14, color='white', fontweight='bold')

    # Improve tick labels.
    plt.xticks(fontsize=12, color='white')
    plt.yticks(fontsize=12, color='white')

    # Add subtle grid.
    plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')

    # Create custom legend.
    unique_labels = np.unique(sample_labels)
    legend_elements = []

    for label in unique_labels:
        if label < len(hex_colors):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=hex_colors[label],
                                            markersize=10, label=f'Cluster {label}'))

    legend = plt.legend(handles=legend_elements, title='Clusters',
                       bbox_to_anchor=(1.02, 1), loc='upper left',
                       frameon=True, fancybox=True, shadow=True,
                       facecolor='black', edgecolor='white', fontsize=10)
    legend.get_title().set_color('white')
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(12)

    for text in legend.get_texts():
        text.set_color('white')
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='black', edgecolor='None')
    plt.close()

    console.print(f"[green]✓[/green] PCA visualization saved → {output_path}")

    # Display PCA summary.
    total_variance = pca.explained_variance_ratio_.sum()
    console.print(f"[blue]PCA Summary:[/blue] First 2 components explain {total_variance:.2%} of variance")


def analyze_feature_importance(features: np.ndarray, cluster_labels: np.ndarray,
                             feature_names: List[str], output_path: Path) -> pd.DataFrame:
    """
    Analyze feature importance for cluster separation.

    Args:
        features: Feature matrix.
        cluster_labels: Array of cluster assignments.
        feature_names: List of feature names.
        output_path: Path for output importance plot.

    Returns:
        DataFrame with feature importance scores.

    This function analyzes which features contribute most to cluster separation,
    providing biological insights into the clustering results.
    """
    console.print("[cyan]Analyzing feature importance for clustering...[/cyan]")

    # Train random forest to predict clusters.
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(features, cluster_labels)

    # Get feature importances.
    importances = rf.feature_importances_

    # Create importance DataFrame.
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Create visualization.
    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')

    # Plot top 20 features.
    top_features = importance_df.head(20)

    bars = plt.barh(range(len(top_features)), top_features['importance'],
                    color='skyblue', alpha=0.8, edgecolor='white', linewidth=0.5)

    plt.yticks(range(len(top_features)), top_features['feature'],
               fontsize=10, color='white')
    plt.xlabel('Feature Importance', fontsize=14, color='white', fontweight='bold')
    plt.title('Top 20 Features for Nuclear Cluster Separation',
              fontsize=16, fontweight='bold', color='white', pad=20)

    # Add value labels on bars.
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center',
                fontsize=9, color='white', fontweight='bold')

    plt.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='black', edgecolor='None')
    plt.close()

    console.print(f"[green]✓[/green] Feature importance analysis saved → {output_path}")

    return importance_df


def compute_cluster_statistics(df: pd.DataFrame, cluster_labels: np.ndarray,
                             feature_names: List[str]) -> pd.DataFrame:
    """
    Compute comprehensive statistics for each cluster.

    Args:
        df: Original DataFrame with nuclear features.
        cluster_labels: Array of cluster assignments.
        feature_names: List of feature names.

    Returns:
        DataFrame with cluster statistics.

    This function computes detailed statistics for each cluster to aid in
    biological interpretation and validation of clustering results.
    """
    console.print("[cyan]Computing cluster statistics...[/cyan]")

    # Add cluster labels to DataFrame.
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels

    # Compute statistics per cluster.
    cluster_stats = []

    for cluster_id in np.unique(cluster_labels):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]

        stats = {
            'cluster_id': cluster_id,
            'nucleus_count': len(cluster_data),
            'percentage': len(cluster_data) / len(df_with_clusters) * 100
        }

        # Compute mean and std for each feature.
        for feature in feature_names:
            if feature in cluster_data.columns:
                stats[f'{feature}_mean'] = cluster_data[feature].mean()
                stats[f'{feature}_std'] = cluster_data[feature].std()

        cluster_stats.append(stats)

    stats_df = pd.DataFrame(cluster_stats)

    console.print(f"[green]✓[/green] Computed statistics for {len(stats_df)} clusters")

    return stats_df


"""MAIN CLUSTERING PIPELINE"""


def main():
    """
    Main clustering pipeline for engineered nuclear features.

    This function coordinates the complete clustering workflow including
    data loading, preprocessing, clustering, and visualization generation.
    """
    parser = argparse.ArgumentParser(
        description='Config-driven clustering of engineered nuclear features.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic clustering using configuration file
    python cluster_engineered_features.py \\
        --config ../../configs/engineered_feature_extraction_config.ini

    # Clustering with custom configuration file
    python cluster_engineered_features.py \\
        --config /path/to/custom_config.ini
        """
    )

    # Required arguments - simplified config-driven interface.
    parser.add_argument('--config', type=Path, required=True,
                       help='Path to configuration file containing all clustering parameters')

    # Parse arguments.
    args = parser.parse_args()

    # Configure logging and reproducibility.
    configure_logging()

    console.print("\n[bold blue]🧬 NUCLEAR FEATURE CLUSTERING ANALYSIS 🧬[/bold blue]\n")

    try:
        start_time = time.time()

        # Step 1: Load configuration and copy to output directory.
        config, output_dir = load_and_copy_config(args.config)

        # Step 2: Validate all required configuration parameters.
        validate_config_parameters(config)

        # Step 3: Extract and validate input file paths from configuration.
        features_path, image_path, mask_path = get_file_paths_from_config(config)

        # Step 4: Set global seed for reproducibility.
        set_global_seed(config['clustering_seed'])

        console.print(f"[green]✓[/green] Configuration-driven setup completed")
        console.print(f"[blue]ℹ[/blue] Results will be saved to: [bold]{output_dir}[/bold]")

        # Step 5: Load nuclear features from config-specified path.
        df = load_nuclear_features(features_path)

        # Step 7: Prepare feature matrix.
        features, feature_names, nuclear_labels = prepare_feature_matrix(df)

        # Step 8: Scale features.
        scaler = stream_scale_features(features, config['clustering_batch_size'])

        # Step 9: Determine optimal number of clusters.
        if config['auto_k_method'] != 'None':
            optimal_k, scores_df = choose_optimal_k(features, config['max_clusters_test'], config['auto_k_method'])
            scores_df.to_csv(output_dir / 'cluster_selection_scores.csv', index=False)
            n_clusters = optimal_k
        else:
            n_clusters = config['default_clusters']

        # Step 10: Perform clustering.
        kmeans = stream_cluster_features(features, scaler, n_clusters, config['clustering_batch_size'], config['clustering_seed'])

        # Step 11: Predict cluster labels.
        cluster_labels = predict_cluster_labels(features, scaler, kmeans, config['clustering_batch_size'])

        # Step 12: Generate enhanced color palette using configuration.
        console.print(f"[cyan]Generating enhanced color palette for {n_clusters} clusters...[/cyan]")

        color_palette = generate_color_palette(
            n=n_clusters,
            alpha=config['color_alpha'],
            background=config['color_background'],
            saturation=config['color_saturation'],
            contrast_ratio=config['color_contrast_ratio'],
            hue_start=config['color_hue_start'],
            custom_colors=config['custom_colors'] if config['custom_colors'] else None
        )

        console.print(f"[green]✓[/green] Generated {len(color_palette)} distinct colors")

        # Step 13: Save results.
        console.print("[cyan]Saving clustering results...[/cyan]")

        # Save cluster assignments.
        results_df = df.copy()
        results_df['Cluster'] = cluster_labels
        results_df.to_csv(output_dir / 'nuclear_clusters.csv', index=False)

        # Save models if configured.
        if config['save_clustering_model']:
            joblib.dump(kmeans, output_dir / 'kmeans_model.joblib')
            joblib.dump(scaler, output_dir / 'scaler.joblib')

        # Step 14: Create visualizations based on configuration.
        console.print("[cyan]Creating visualizations...[/cyan]")

        # Generate cluster overlay if configured.
        if config['generate_cluster_overlay']:
            create_cluster_overlay_advanced(
                image_path, mask_path, nuclear_labels, cluster_labels,
                color_palette, output_dir / 'cluster_overlay.tif',
                tile_size=config.get('overlay_tile_size', 1024),
                workers=config.get('overlay_workers', 'auto'),
                alpha=config.get('overlay_alpha', 0.85),
                gpu=config.get('overlay_gpu', True),
                memory_limit_mb=config.get('overlay_memory_limit_mb', 8192)
            )

        # Generate PCA visualization if configured.
        if config['generate_pca_plot']:
            scaled_features = scaler.transform(features)
            create_pca_visualization(
                scaled_features, cluster_labels, feature_names,
                color_palette, output_dir / 'pca_clusters.png',
                sample_size=config['pca_sample_size']
            )

        # Generate feature importance analysis if configured.
        if config['generate_feature_importance']:
            if 'scaled_features' not in locals():
                scaled_features = scaler.transform(features)
            importance_df = analyze_feature_importance(
                scaled_features, cluster_labels, feature_names,
                output_dir / 'feature_importance.png'
            )
            importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)

        # Generate cluster statistics if configured.
        if config['save_cluster_statistics']:
            stats_df = compute_cluster_statistics(df, cluster_labels, feature_names)
            stats_df.to_csv(output_dir / 'cluster_statistics.csv', index=False)

        # Step 15: Display final summary.
        end_time = time.time()
        processing_time = end_time - start_time

        summary_table = Table(title="Nuclear Feature Clustering Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Nuclei Clustered", str(len(df)))
        summary_table.add_row("Features Used", str(len(feature_names)))
        summary_table.add_row("Number of Clusters", str(n_clusters))
        summary_table.add_row("Clustering Method", config['auto_k_method'] if config['auto_k_method'] != 'None' else 'Fixed K')
        summary_table.add_row("Processing Time", f"{processing_time:.2f} seconds")
        summary_table.add_row("Output Directory", str(output_dir))
        summary_table.add_row("Color Palette", f"{len(color_palette)} distinct colors")

        # Add cluster size distribution.
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = ", ".join([f"C{label}: {count}" for label, count in zip(unique_labels, counts)])
        summary_table.add_row("Cluster Sizes", cluster_sizes[:100] + "..." if len(cluster_sizes) > 100 else cluster_sizes)

        console.print(summary_table)
        console.print(f"[bold green]✓ Nuclear feature clustering completed successfully![/bold green]")

        # Display top contributing features if importance analysis was performed.
        if config['generate_feature_importance'] and 'importance_df' in locals():
            console.print(f"\n[blue]Top 5 Features for Cluster Separation:[/blue]")
            for i, row in importance_df.head(5).iterrows():
                console.print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

        # Display configuration summary.
        console.print(f"\n[blue]Configuration Used:[/blue]")
        console.print(f"  • Batch Size: {config['clustering_batch_size']}")
        console.print(f"  • Random Seed: {config['clustering_seed']}")
        console.print(f"  • Color Background: {config['color_background']}")
        console.print(f"  • Overlay Downsample: {config['overlay_downsample_factor']}x")

    except Exception as e:
        console.print(f"[bold red]✗ Error during clustering analysis: {e}[/bold red]")
        logging.error(f"Clustering failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
