"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: cluster_engineered_features.py
Description:
    Memory-optimized clustering of engineered nuclear features with MiniBatchKMeans and streaming scaling.
    Supports comprehensive morphological, intensity, texture, and neighborhood features extracted from
    DAPI-stained kidney tissue sections. Optimized for I/R injury analysis with publication-quality
    visualizations and scientific color palettes.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scikit-learn, joblib, matplotlib, PIL, seaborn, scipy.
    • typer for modern CLI interface.
    • rich for progress tracking and console output.

Usage:
    python cluster_engineered_features.py cluster \
        --features nuclear_features.csv \
        --image tissue_dapi.tif \
        --mask segmentation_masks.npy \
        --clusters 15 --auto-k silhouette \
        --outdir results/clustering --seed 42

Arguments:
    --features         Path to CSV file with extracted nuclear features.
    --image            Path to raw microscopy image (TIFF format).
    --mask             Path to segmentation mask (.npy format).
    --clusters         Number of clusters for K-means (default: 10).
    --auto-k           Auto K-selection method ('none', 'silhouette', 'dbi').
    --batch-size       Batch size for streaming processing (default: 5000).
    --outdir           Output directory for results.
    --seed             Random seed for reproducibility.
    --region           Crop region for overlay: xmin xmax ymin ymax (fractions).

Inputs:
    • nuclear_features.csv    Comprehensive nuclear feature matrix.
    • tissue_dapi.tif         Original DAPI-stained tissue image.
    • segmentation_masks.npy  Nuclear segmentation masks.

Outputs:
    • nuclear_clusters.csv         Cluster assignments per nucleus.
    • kmeans_model.joblib          Trained MiniBatchKMeans model.
    • scaler.joblib                StandardScaler for feature normalization.
    • cluster_overlay.tif          Overlay of clusters on tissue image.
    • pca_clusters.png             PCA scatter plot with cluster colors.
    • feature_importance.png       Feature importance for clustering.
    • cluster_statistics.csv       Statistical summary per cluster.

Key Features:
    • Handles comprehensive nuclear feature sets (50+ features).
    • Memory-efficient streaming processing for large datasets.
    • Enhanced color palette with 35+ distinct colors for large cluster numbers.
    • Publication-quality visualizations with scientific formatting.
    • Feature importance analysis for biological interpretation.
    • Comprehensive statistical analysis per cluster.
    • Optimized for kidney I/R injury research contexts.

Notes:
    • Features should include morphological, intensity, texture, and neighborhood categories.
    • Color palette optimized for microscopy visualization with high contrast ratios.
    • Supports both automatic and manual cluster number selection.
    • Memory usage scales efficiently with dataset size through streaming processing.
"""
import traceback
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import random
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

# Import color generation utilities.
from code.engineered_feature_extraction.utils.generate_contrast_colors import generate_color_palette, colors_to_hex_list
from code.engineered_feature_extraction.utils.color_config import ColorConfig, load_color_config

# Import feature extraction configuration utilities.
from code.engineered_feature_extraction.utils.config_loader import load_feature_extraction_config

console = Console()


"""UTILITY FUNCTIONS"""


def load_clustering_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load clustering configuration from engineered feature extraction config file.

    Args:
        config_path: Optional path to configuration file.

    Returns:
        Dictionary with clustering configuration parameters.

    This function loads clustering parameters from the dedicated feature extraction
    configuration system, providing comprehensive defaults when configuration is not available.
    """
    try:
        # Load configuration using the dedicated feature extraction config loader.
        config = load_feature_extraction_config(config_path)
        console.print(f"[green]✓[/green] Loaded feature extraction config with {len(config)} parameters")
        return config
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Failed to load config: {e}, using defaults")

        # Return minimal default clustering configuration as fallback.
        default_config = {
            'enable_clustering': True,
            'default_clusters': 12,
            'auto_k_method': 'silhouette',
            'max_clusters_test': 25,
            'clustering_batch_size': 5000,
            'clustering_seed': 42,
            'generate_cluster_overlay': True,
            'generate_pca_plot': True,
            'generate_feature_importance': True,
            'overlay_crop_region': (0.1, 0.9, 0.1, 0.9),
            'overlay_downsample_factor': 1,
            'color_background': 'dark',
            'color_alpha': 200,
            'color_saturation': 0.95,
            'color_contrast_ratio': 4.5,
            'color_hue_start': 0.0,
            'custom_colors': [],
            'clustering_output_subdir': 'clustering_analysis',
            'save_cluster_statistics': True,
            'save_clustering_model': True,
            'pca_sample_size': 5000
        }

        console.print("[blue]ℹ[/blue] Using minimal default clustering configuration")
        return default_config


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
    required_cols = ['Label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Identify feature columns (exclude metadata).
    metadata_cols = ['Label', 'Centroid_X', 'Centroid_Y']
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
    nuclear_labels = df['Label'].values
    
    # Identify feature columns (exclude metadata).
    metadata_cols = ['Label', 'Centroid_X', 'Centroid_Y']
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


def create_cluster_overlay(image_path: Path, mask_path: Path, nuclear_labels: np.ndarray,
                          cluster_labels: np.ndarray, color_palette: Dict[int, Tuple[int, int, int, int]],
                          output_path: Path, region: Optional[Tuple[float, float, float, float]] = None,
                          downsample: int = 1) -> None:
    """
    Create overlay visualization of clusters on tissue image.

    Args:
        image_path: Path to original tissue image.
        mask_path: Path to segmentation mask.
        nuclear_labels: Array of nuclear labels.
        cluster_labels: Array of cluster assignments.
        color_palette: Dictionary mapping cluster to RGBA color.
        output_path: Path for output overlay image.
        region: Optional crop region as (xmin, xmax, ymin, ymax) fractions.
        downsample: Downsampling factor for memory efficiency.

    This function creates a publication-quality overlay showing cluster
    assignments mapped onto the original tissue morphology for biological interpretation.
    """
    console.print("[cyan]Creating cluster overlay visualization...[/cyan]")

    # Load base image.
    base_image = Image.open(image_path).convert('RGBA')
    w, h = base_image.size

    # Apply region cropping if specified.
    if region:
        xmin, xmax, ymin, ymax = region
        x0 = max(0, int(w * xmin))
        x1 = min(w, int(w * xmax))
        y0 = max(0, int(h * ymin))
        y1 = min(h, int(h * ymax))
        base_image = base_image.crop((x0, y0, x1, y1))
        w, h = base_image.size

    # Load segmentation mask.
    mask = np.load(mask_path, mmap_mode='r')

    if region:
        mask = mask[y0:y1, x0:x1]

    # Apply downsampling if specified.
    if downsample > 1:
        mask = mask[::downsample, ::downsample]
        base_image = base_image.resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)

    # Create lookup table for cluster colors.
    max_label = int(mask.max())
    lut = np.zeros(max_label + 1, dtype=np.int32)

    for nuclear_label, cluster_id in zip(nuclear_labels, cluster_labels):
        if 0 <= nuclear_label <= max_label:
            lut[int(nuclear_label)] = int(cluster_id) + 1  # +1 to avoid background

    # Map mask to cluster indices.
    cluster_map = lut[mask]

    # Create RGBA color array.
    n_clusters = len(color_palette)
    rgba_array = np.zeros((n_clusters + 1, 4), dtype=np.uint8)  # +1 for background

    # Populate color array.
    for cluster_id, (r, g, b, a) in color_palette.items():
        array_idx = cluster_id + 1  # Map to 1-based indexing
        if array_idx < len(rgba_array):
            rgba_array[array_idx] = [r, g, b, a]

    # Create overlay image.
    overlay_data = rgba_array[cluster_map]
    overlay = Image.fromarray(overlay_data, mode='RGBA')

    # Composite overlay onto base image.
    composite = Image.alpha_composite(base_image, overlay)
    composite.save(output_path)

    # Count colored pixels for validation.
    colored_pixels = np.sum(cluster_map > 0)
    unique_clusters = len(np.unique(cluster_map[cluster_map > 0]))

    console.print(f"[green]✓[/green] Overlay saved: {colored_pixels} colored pixels, {unique_clusters} clusters → {output_path}")


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
                facecolor='black', edgecolor='none')
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

    from sklearn.ensemble import RandomForestClassifier

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
                facecolor='black', edgecolor='none')
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
        description='Memory-efficient clustering of engineered nuclear features.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic clustering with 10 clusters
    python cluster_engineered_features.py --features nuclear_features.csv \\
        --image tissue_dapi.tif --mask segmentation_masks.npy \\
        --outdir results/clustering

    # Auto-select optimal K using silhouette analysis
    python cluster_engineered_features.py --features nuclear_features.csv \\
        --image tissue_dapi.tif --mask segmentation_masks.npy \\
        --clusters 20 --auto-k silhouette --outdir results/clustering

    # Cluster with custom region and seed
    python cluster_engineered_features.py --features nuclear_features.csv \\
        --image tissue_dapi.tif --mask segmentation_masks.npy \\
        --clusters 15 --seed 42 --region 0.1 0.9 0.1 0.9 \\
        --outdir results/clustering
        """
    )

    # Required arguments.
    parser.add_argument('--features', type=Path, required=True,
                       help='Path to CSV file with extracted nuclear features')
    parser.add_argument('--image', type=Path, required=True,
                       help='Path to raw microscopy image (TIFF format)')
    parser.add_argument('--mask', type=Path, required=True,
                       help='Path to segmentation mask (.npy format)')

    # Configuration and clustering parameters.
    parser.add_argument('--config', type=Path,
                       help='Path to configuration file (uses project defaults if not specified)')
    parser.add_argument('--clusters', type=int,
                       help='Number of clusters for K-means (overrides config)')
    parser.add_argument('--auto-k', choices=['none', 'silhouette', 'dbi'],
                       help='Auto K-selection method (overrides config)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for streaming processing (overrides config)')

    # Output and visualization.
    parser.add_argument('--outdir', type=Path,
                       help='Output directory for results (overrides config)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducibility (overrides config)')
    parser.add_argument('--region', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                       help='Crop region for overlay as fractions (overrides config)')
    parser.add_argument('--downsample', type=int,
                       help='Downsampling factor for overlay (overrides config)')

    # Parse arguments.
    args = parser.parse_args()

    # Configure logging and reproducibility.
    configure_logging()

    console.print("\n[bold blue]🧬 NUCLEAR FEATURE CLUSTERING ANALYSIS 🧬[/bold blue]\n")

    try:
        start_time = time.time()

        # Step 1: Load clustering configuration.
        config = load_clustering_config(args.config)

        # Override config with command-line arguments if provided.
        if args.clusters is not None:
            config['default_clusters'] = args.clusters
        if args.auto_k is not None:
            config['auto_k_method'] = args.auto_k
        if args.batch_size is not None:
            config['clustering_batch_size'] = args.batch_size
        if args.seed is not None:
            config['clustering_seed'] = args.seed
        if args.region is not None:
            config['overlay_crop_region'] = tuple(args.region)
        if args.downsample is not None:
            config['overlay_downsample_factor'] = args.downsample
        if args.outdir is not None:
            output_dir = args.outdir
        else:
            output_dir = Path('results') / config['clustering_output_subdir']

        # Set global seed and create output directory.
        set_global_seed(config['clustering_seed'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 2: Load nuclear features.
        df = load_nuclear_features(args.features)

        # Step 3: Prepare feature matrix.
        features, feature_names, nuclear_labels = prepare_feature_matrix(df)

        # Step 4: Scale features.
        scaler = stream_scale_features(features, config['clustering_batch_size'])

        # Step 5: Determine optimal number of clusters.
        if config['auto_k_method'] != 'none':
            optimal_k, scores_df = choose_optimal_k(features, config['max_clusters_test'], config['auto_k_method'])
            scores_df.to_csv(output_dir / 'cluster_selection_scores.csv', index=False)
            n_clusters = optimal_k
        else:
            n_clusters = config['default_clusters']

        # Step 6: Perform clustering.
        kmeans = stream_cluster_features(features, scaler, n_clusters, config['clustering_batch_size'], config['clustering_seed'])

        # Step 7: Predict cluster labels.
        cluster_labels = predict_cluster_labels(features, scaler, kmeans, config['clustering_batch_size'])

        # Step 8: Generate enhanced color palette using configuration.
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

        # Step 9: Save results.
        console.print("[cyan]Saving clustering results...[/cyan]")

        # Save cluster assignments.
        results_df = df.copy()
        results_df['Cluster'] = cluster_labels
        results_df.to_csv(output_dir / 'nuclear_clusters.csv', index=False)

        # Save models if configured.
        if config['save_clustering_model']:
            joblib.dump(kmeans, output_dir / 'kmeans_model.joblib')
            joblib.dump(scaler, output_dir / 'scaler.joblib')

        # Step 10: Create visualizations based on configuration.
        console.print("[cyan]Creating visualizations...[/cyan]")

        # Generate cluster overlay if configured.
        if config['generate_cluster_overlay']:
            create_cluster_overlay(
                args.image, args.mask, nuclear_labels, cluster_labels,
                color_palette, output_dir / 'cluster_overlay.tif',
                region=config['overlay_crop_region'],
                downsample=config['overlay_downsample_factor']
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

        # Step 11: Display final summary.
        end_time = time.time()
        processing_time = end_time - start_time

        summary_table = Table(title="Nuclear Feature Clustering Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Nuclei Clustered", str(len(df)))
        summary_table.add_row("Features Used", str(len(feature_names)))
        summary_table.add_row("Number of Clusters", str(n_clusters))
        summary_table.add_row("Clustering Method", config['auto_k_method'] if config['auto_k_method'] != 'none' else 'Fixed K')
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
