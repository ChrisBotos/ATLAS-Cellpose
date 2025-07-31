"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: example_clustering_workflow.py
Description:
    Example workflow demonstrating nuclear feature clustering for kidney I/R injury analysis.
    Shows complete pipeline from feature extraction to clustering visualization with
    publication-quality outputs and scientific interpretation.

Dependencies:
    • Python >= 3.10.
    • All dependencies from cluster_engineered_features.py.
    • Project configuration system.

Usage:
    python example_clustering_workflow.py --help
    python example_clustering_workflow.py --demo

Key Features:
    • Complete workflow demonstration with synthetic data.
    • Integration with project configuration system.
    • Publication-quality visualization examples.
    • Scientific interpretation guidelines.
    • Memory-efficient processing for large datasets.

Notes:
    • Designed for kidney I/R injury research contexts.
    • Demonstrates best practices for nuclear morphology analysis.
    • Includes comprehensive error handling and validation.
"""
import traceback
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from rich.console import Console
from rich.panel import Panel

# Add project root to path.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from code.engineered_feature_extraction.cluster_engineered_features import (
    load_nuclear_features, prepare_feature_matrix, stream_scale_features,
    stream_cluster_features, predict_cluster_labels, create_cluster_overlay,
    create_pca_visualization, analyze_feature_importance, compute_cluster_statistics,
    load_clustering_config
)
from code.engineered_feature_extraction.utils.generate_contrast_colors import generate_color_palette
from code.engineered_feature_extraction.utils.config_loader import load_feature_extraction_config

console = Console()


def create_synthetic_dataset(output_dir: Path, n_nuclei: int = 2000) -> tuple[Path, Path, Path]:
    """
    Create synthetic nuclear feature dataset for demonstration.
    
    Args:
        output_dir: Directory for synthetic data files.
        n_nuclei: Number of synthetic nuclei to generate.
        
    Returns:
        Tuple of (features_path, image_path, mask_path).
        
    This function generates realistic synthetic nuclear features and associated
    image data for testing and demonstration purposes.
    """
    console.print(f"[cyan]Creating synthetic dataset with {n_nuclei} nuclei...[/cyan]")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility.
    np.random.seed(42)
    
    # Generate realistic nuclear features with biological variation.
    # Simulate different tissue regions with varying characteristics.
    
    # Create three distinct tissue regions with different nuclear properties.
    region_sizes = [n_nuclei // 3, n_nuclei // 3, n_nuclei - 2 * (n_nuclei // 3)]
    
    features_list = []
    
    for region_id, region_size in enumerate(region_sizes):
        # Region-specific parameters to simulate tissue heterogeneity.
        if region_id == 0:  # Healthy tissue region.
            area_mean, area_std = 200, 50
            circularity_alpha, circularity_beta = 3, 2  # Higher circularity
            intensity_mean, intensity_std = 180, 25
        elif region_id == 1:  # Moderately damaged region.
            area_mean, area_std = 150, 60
            circularity_alpha, circularity_beta = 2, 3  # Lower circularity
            intensity_mean, intensity_std = 140, 35
        else:  # Severely damaged region.
            area_mean, area_std = 100, 70
            circularity_alpha, circularity_beta = 1, 4  # Much lower circularity
            intensity_mean, intensity_std = 100, 45
        
        # Generate features for this region.
        region_features = {
            'Label': np.arange(sum(region_sizes[:region_id]) + 1, 
                             sum(region_sizes[:region_id + 1]) + 1),
            'Centroid_X': np.random.uniform(region_id * 1000, (region_id + 1) * 1000, region_size),
            'Centroid_Y': np.random.uniform(0, 3000, region_size),
            'Area': np.random.lognormal(np.log(area_mean), 0.3, region_size),
            'Perimeter': np.sqrt(region_features['Area'] if 'Area' in locals() else 
                               np.random.lognormal(np.log(area_mean), 0.3, region_size)) * 3.5 + np.random.normal(0, 5, region_size),
            'Major_Axis_Length': np.sqrt(region_features['Area'] if 'Area' in locals() else 
                                       np.random.lognormal(np.log(area_mean), 0.3, region_size)) * 1.8 + np.random.normal(0, 3, region_size),
            'Minor_Axis_Length': np.sqrt(region_features['Area'] if 'Area' in locals() else 
                                       np.random.lognormal(np.log(area_mean), 0.3, region_size)) * 1.2 + np.random.normal(0, 2, region_size),
            'Circularity': np.random.beta(circularity_alpha, circularity_beta, region_size),
            'Eccentricity': np.random.beta(2, 3, region_size),
            'Solidity': np.random.beta(4, 1, region_size),
            'Intensity_Mean': np.random.normal(intensity_mean, intensity_std, region_size),
            'Intensity_Std': np.random.exponential(15, region_size),
            'Texture_Entropy': np.random.gamma(2, 0.5, region_size),
            'Neighborhood_Density': np.random.exponential(0.008, region_size),
            'Distance_to_Nearest_Nucleus': np.random.exponential(30, region_size),
            'Cluster_Density_Index': np.random.gamma(1.5, 2, region_size),
            'Distance_to_Image_Center': np.sqrt((region_features['Centroid_X'] - 1500)**2 + 
                                              (region_features['Centroid_Y'] - 1500)**2) if 'Centroid_X' in locals() else np.random.uniform(0, 2000, region_size)
        }
        
        # Fix the reference issues by recalculating dependent features.
        areas = np.random.lognormal(np.log(area_mean), 0.3, region_size)
        centroids_x = np.random.uniform(region_id * 1000, (region_id + 1) * 1000, region_size)
        centroids_y = np.random.uniform(0, 3000, region_size)
        
        region_features = {
            'Label': np.arange(sum(region_sizes[:region_id]) + 1, 
                             sum(region_sizes[:region_id + 1]) + 1),
            'Centroid_X': centroids_x,
            'Centroid_Y': centroids_y,
            'Area': areas,
            'Perimeter': np.sqrt(areas) * 3.5 + np.random.normal(0, 5, region_size),
            'Major_Axis_Length': np.sqrt(areas) * 1.8 + np.random.normal(0, 3, region_size),
            'Minor_Axis_Length': np.sqrt(areas) * 1.2 + np.random.normal(0, 2, region_size),
            'Circularity': np.random.beta(circularity_alpha, circularity_beta, region_size),
            'Eccentricity': np.random.beta(2, 3, region_size),
            'Solidity': np.random.beta(4, 1, region_size),
            'Intensity_Mean': np.random.normal(intensity_mean, intensity_std, region_size),
            'Intensity_Std': np.random.exponential(15, region_size),
            'Texture_Entropy': np.random.gamma(2, 0.5, region_size),
            'Neighborhood_Density': np.random.exponential(0.008, region_size),
            'Distance_to_Nearest_Nucleus': np.random.exponential(30, region_size),
            'Cluster_Density_Index': np.random.gamma(1.5, 2, region_size),
            'Distance_to_Image_Center': np.sqrt((centroids_x - 1500)**2 + (centroids_y - 1500)**2)
        }
        
        features_list.append(pd.DataFrame(region_features))
    
    # Combine all regions.
    df = pd.concat(features_list, ignore_index=True)
    
    # Add some additional realistic features.
    df['Aspect_Ratio'] = df['Major_Axis_Length'] / np.maximum(df['Minor_Axis_Length'], 1)
    df['Roughness_Index'] = df['Perimeter']**2 / (4 * np.pi * df['Area'])
    df['Fractal_Dimension'] = 1.2 + np.random.normal(0, 0.1, len(df))
    
    # Save features CSV.
    features_path = output_dir / 'synthetic_nuclear_features.csv'
    df.to_csv(features_path, index=False)
    
    # Create synthetic DAPI image.
    image_path = output_dir / 'synthetic_dapi_image.tif'
    synthetic_image = np.random.randint(20, 80, (3000, 3000, 3), dtype=np.uint8)
    
    # Add some bright nuclear regions.
    for _, row in df.iterrows():
        x, y = int(row['Centroid_X']), int(row['Centroid_Y'])
        if 0 <= x < 3000 and 0 <= y < 3000:
            size = int(np.sqrt(row['Area']) / 2)
            x1, x2 = max(0, x - size), min(3000, x + size)
            y1, y2 = max(0, y - size), min(3000, y + size)
            synthetic_image[y1:y2, x1:x2] = np.random.randint(150, 255, (y2-y1, x2-x1, 3))
    
    Image.fromarray(synthetic_image).save(image_path)
    
    # Create synthetic segmentation mask.
    mask_path = output_dir / 'synthetic_segmentation_mask.npy'
    synthetic_mask = np.zeros((3000, 3000), dtype=np.int32)
    
    for _, row in df.iterrows():
        x, y = int(row['Centroid_X']), int(row['Centroid_Y'])
        if 0 <= x < 3000 and 0 <= y < 3000:
            size = int(np.sqrt(row['Area']) / 2)
            x1, x2 = max(0, x - size), min(3000, x + size)
            y1, y2 = max(0, y - size), min(3000, y + size)
            synthetic_mask[y1:y2, x1:x2] = int(row['Label'])
    
    np.save(mask_path, synthetic_mask)
    
    console.print(f"[green]✓[/green] Created synthetic dataset: {len(df)} nuclei")
    console.print(f"  • Features: {features_path}")
    console.print(f"  • Image: {image_path}")
    console.print(f"  • Mask: {mask_path}")
    
    return features_path, image_path, mask_path


def run_clustering_demo():
    """
    Run complete clustering demonstration workflow.
    
    This function demonstrates the complete nuclear feature clustering pipeline
    using synthetic data, showing all major functionality and outputs.
    """
    console.print(Panel.fit(
        "[bold blue]🧬 NUCLEAR FEATURE CLUSTERING DEMONSTRATION 🧬[/bold blue]\n\n"
        "This demo shows the complete workflow for clustering nuclear morphological features\n"
        "extracted from DAPI-stained kidney tissue sections for I/R injury analysis.",
        title="Clustering Demo"
    ))
    
    try:
        # Create output directory.
        demo_dir = Path('results/clustering_demo')
        demo_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Create synthetic dataset.
        features_path, image_path, mask_path = create_synthetic_dataset(demo_dir, n_nuclei=1500)
        
        # Step 2: Load configuration.
        config = load_clustering_config()
        
        # Step 3: Load and prepare features.
        df = load_nuclear_features(features_path)
        features, feature_names, nuclear_labels = prepare_feature_matrix(df)
        
        # Step 4: Scale features.
        scaler = stream_scale_features(features, config['clustering_batch_size'])
        
        # Step 5: Perform clustering.
        n_clusters = 8  # Good number for demonstration
        kmeans = stream_cluster_features(features, scaler, n_clusters, 
                                       config['clustering_batch_size'], config['clustering_seed'])
        cluster_labels = predict_cluster_labels(features, scaler, kmeans, config['clustering_batch_size'])
        
        # Step 6: Generate color palette.
        color_palette = generate_color_palette(
            n=n_clusters,
            alpha=config['color_alpha'],
            background=config['color_background'],
            saturation=config['color_saturation'],
            contrast_ratio=config['color_contrast_ratio']
        )
        
        # Step 7: Create visualizations.
        console.print("[cyan]Creating demonstration visualizations...[/cyan]")
        
        # Cluster overlay.
        create_cluster_overlay(
            image_path, mask_path, nuclear_labels, cluster_labels,
            color_palette, demo_dir / 'demo_cluster_overlay.tif',
            region=(0.2, 0.8, 0.2, 0.8), downsample=2
        )
        
        # PCA visualization.
        scaled_features = scaler.transform(features)
        create_pca_visualization(
            scaled_features, cluster_labels, feature_names,
            color_palette, demo_dir / 'demo_pca_clusters.png'
        )
        
        # Feature importance.
        importance_df = analyze_feature_importance(
            scaled_features, cluster_labels, feature_names,
            demo_dir / 'demo_feature_importance.png'
        )
        
        # Cluster statistics.
        stats_df = compute_cluster_statistics(df, cluster_labels, feature_names)
        
        # Step 8: Save results.
        results_df = df.copy()
        results_df['Cluster'] = cluster_labels
        results_df.to_csv(demo_dir / 'demo_clustered_features.csv', index=False)
        importance_df.to_csv(demo_dir / 'demo_feature_importance.csv', index=False)
        stats_df.to_csv(demo_dir / 'demo_cluster_statistics.csv', index=False)
        
        # Step 9: Display summary.
        console.print(f"\n[bold green]✓ Clustering demonstration completed successfully![/bold green]")
        console.print(f"[blue]Results saved to:[/blue] {demo_dir}")
        console.print(f"[blue]Nuclei clustered:[/blue] {len(df)}")
        console.print(f"[blue]Features used:[/blue] {len(feature_names)}")
        console.print(f"[blue]Clusters found:[/blue] {n_clusters}")
        
        # Display cluster distribution.
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        console.print(f"\n[blue]Cluster Distribution:[/blue]")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(cluster_labels) * 100
            console.print(f"  Cluster {label}: {count} nuclei ({percentage:.1f}%)")
        
        # Display top features.
        console.print(f"\n[blue]Top 5 Discriminative Features:[/blue]")
        for i, row in importance_df.head(5).iterrows():
            console.print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        console.print(f"\n[yellow]💡 Tip:[/yellow] Examine the generated visualizations to understand")
        console.print(f"   tissue organization patterns and nuclear morphology clusters.")
        
    except Exception as e:
        console.print(f"[bold red]✗ Demo failed: {e}[/bold red]")
        traceback.print_exc()


def main():
    """Main function for clustering workflow demonstration."""
    parser = argparse.ArgumentParser(
        description='Nuclear feature clustering workflow demonstration.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run complete clustering demonstration with synthetic data')
    
    args = parser.parse_args()
    
    if args.demo:
        run_clustering_demo()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
