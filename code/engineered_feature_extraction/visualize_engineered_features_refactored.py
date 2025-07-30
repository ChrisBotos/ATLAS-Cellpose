#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: visualize_engineered_features_refactored.py.
Description:
    Generate publication-quality visualizations of nuclear morphological features for kidney
    ischemia-reperfusion injury analysis. Creates organized violin plots grouped by feature
    categories with timepoint-specific color coding and comprehensive statistical analysis.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scipy, statsmodels, matplotlib, seaborn, typer, traceback.
    • Custom utilities from nuclei_segmentation package.

Usage:
    python visualize_engineered_features_refactored.py visualize \
        --features_csv <path/to/features.csv> \
        --output_dir <path/to/output_dir> \
        [--control_csv <path/to/control_features.csv>] \
        [--timepoint <timepoint_label>] \
        [--config <path/to/config.ini>]

Positional Arguments:
    visualize    Command to generate feature visualizations.

Optional Arguments:
    --features_csv    Path to CSV file containing extracted nuclear features.
    --output_dir      Directory to save visualization plots (default: ./feature_plots).
    --control_csv     Path to control group features CSV for comparative analysis.
    --timepoint       Timepoint label for color coding (10h, 2d, 14d, etc.).
    --config          Configuration file path (default: uses project config).

Inputs:
    • CSV file containing comprehensive nuclear features from feature extraction.
    • Optional control group CSV for comparative analysis.
    • Configuration file specifying visualization parameters.

Outputs:
    • Publication-quality violin plots organized by feature categories.
    • Statistical comparison plots with FDR-corrected p-values.
    • Correlation matrix heatmaps for feature relationships.
    • Summary statistics and feature distribution reports.

Key Features:
    • Feature categorization: shape, size, neighborhood, and texture features.
    • Timepoint-specific color coding for injury progression analysis (10h, 2d, 14d).
    • Publication-quality violin plots with proper statistical representations.
    • Comprehensive statistical testing with multiple comparison corrections.
    • Scientific formatting optimized for kidney I/R injury research publications.
    • Configurable visualization parameters for different analysis needs.

Notes:
    • Violin plots show complete density distributions with median overlays.
    • Color schemes are optimized for scientific publications and colorblind accessibility.
    • Statistical tests use non-parametric methods appropriate for biological data.
    • All visualizations include proper axis labels, titles, and scientific formatting.
"""

import traceback
import sys
import os
from pathlib import Path
import warnings
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import typer
from scipy.stats import mannwhitneyu, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Add project root to path for imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from code.nuclei_segmentation.utils.project_setup import load_config

# Initialize Typer app for CLI.
app = typer.Typer(help="Generate publication-quality visualizations for nuclear feature analysis.")

# Configure matplotlib and seaborn for publication-quality plots.
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

# Suppress warnings for cleaner output.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Passing `palette` without assigning `hue`')


"""FEATURE CATEGORY DEFINITIONS"""

# Define feature categories for organized visualization.
FEATURE_CATEGORIES = {
    'Shape Features': [
        'circularity', 'eccentricity', 'solidity', 'convex_area_ratio', 'aspect_ratio',
        'compactness', 'elongation', 'roundness', 'form_factor', 'convexity', 'fractal_dimension'
    ],
    'Size Features': [
        'area', 'perimeter', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length',
        'bounding_box_width', 'bounding_box_height', 'bounding_box_area', 'feret_diameter_max',
        'feret_diameter_min'
    ],
    'Neighborhood Features': [
        'nearest_neighbor_distance', 'neighborhood_density', 'cluster_elongation',
        'cluster_polarization', 'spatial_autocorrelation', 'boundary_proximity',
        'tissue_organization_index', 'local_clustering_coefficient'
    ],
    'Texture Features': [
        'intensity_mean', 'intensity_std', 'intensity_median', 'intensity_skewness',
        'intensity_kurtosis', 'texture_entropy', 'glcm_contrast', 'glcm_dissimilarity',
        'glcm_homogeneity', 'glcm_energy', 'gradient_magnitude_mean', 'gradient_magnitude_std'
    ]
}

# Timepoint-specific color palette for kidney I/R injury analysis.
TIMEPOINT_COLORS = {
    '10h': '#E74C3C',    # Red - acute injury phase.
    '2d': '#F39C12',     # Orange - inflammatory phase.
    '14d': '#27AE60',    # Green - repair/recovery phase.
    'control': '#3498DB', # Blue - healthy control.
    'sham': '#9B59B6',   # Purple - sham operation control.
    'default': '#34495E'  # Dark gray - default/unknown.
}


"""UTILITY FUNCTIONS"""

def load_and_validate_data(csv_path: Path, min_area: float = 10.0, max_area: float = 2000.0) -> pd.DataFrame:
    """
    Load and validate nuclear feature data with quality filtering.
    
    Applies size-based filtering and data validation to ensure high-quality
    feature data for visualization and statistical analysis.
    
    Args:
        csv_path: Path to CSV file containing nuclear features.
        min_area: Minimum nuclear area threshold for filtering.
        max_area: Maximum nuclear area threshold for filtering.
        
    Returns:
        Validated DataFrame with filtered nuclear features.
    """
    print(f"Loading feature data from: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV file not found: {csv_path}")
    
    # Load data.
    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"Loaded {original_count} nuclear feature profiles.")
    
    # Apply size filtering if area column exists.
    if 'area' in df.columns:
        df = df[(df['area'] >= min_area) & (df['area'] <= max_area)]
        filtered_count = len(df)
        print(f"Size filtering: {original_count} -> {filtered_count} nuclei "
              f"({100*filtered_count/original_count:.1f}% retained).")
    
    # Remove rows with excessive NaN values.
    nan_threshold = 0.5  # Remove rows with >50% NaN values.
    df = df.dropna(thresh=int(len(df.columns) * (1 - nan_threshold)))
    
    if len(df) == 0:
        raise ValueError("No valid nuclear data remaining after filtering.")
    
    print(f"Final dataset: {len(df)} nuclei with {len(df.columns)} features each.")
    
    return df


def identify_available_features(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify which features from each category are available in the dataset.
    
    Maps the theoretical feature categories to actual column names in the
    dataset, handling variations in naming conventions and missing features.
    
    Args:
        df: DataFrame containing nuclear features.
        
    Returns:
        Dictionary mapping category names to lists of available feature columns.
    """
    available_features = {}
    
    for category, feature_list in FEATURE_CATEGORIES.items():
        available = []
        
        for feature in feature_list:
            # Check for exact match first.
            if feature in df.columns:
                available.append(feature)
                continue
            
            # Check for case-insensitive partial matches.
            matches = [col for col in df.columns if feature.lower() in col.lower()]
            if matches:
                available.extend(matches)
        
        # Remove duplicates while preserving order.
        available = list(dict.fromkeys(available))
        available_features[category] = available
    
    # Log feature availability.
    print("\nFeature availability by category:")
    for category, features in available_features.items():
        print(f"  {category}: {len(features)} features")
        if len(features) == 0:
            print(f"    Warning: No features found for {category}")
    
    return available_features


def create_violin_plots_by_category(
    df: pd.DataFrame,
    available_features: Dict[str, List[str]],
    output_dir: Path,
    timepoint: Optional[str] = None,
    control_df: Optional[pd.DataFrame] = None,
    features_per_page: int = 9
) -> None:
    """
    Create publication-quality violin plots organized by feature categories.
    
    Generates separate visualization pages for each feature category with
    proper scientific formatting, timepoint color coding, and statistical overlays.
    
    Args:
        df: DataFrame containing nuclear features.
        available_features: Dictionary of available features by category.
        output_dir: Directory to save visualization plots.
        timepoint: Optional timepoint label for color coding.
        control_df: Optional control group DataFrame for comparison.
        features_per_page: Number of features to display per page.
    """
    print("\nGenerating publication-quality violin plots by category...")
    
    # Prepare data for visualization.
    plot_data = df.copy()
    plot_data['Group'] = timepoint if timepoint else 'Treatment'
    
    if control_df is not None:
        control_data = control_df.copy()
        control_data['Group'] = 'Control'
        plot_data = pd.concat([plot_data, control_data], ignore_index=True)
    
    # Create plots for each category.
    for category, features in available_features.items():
        if not features:
            print(f"Skipping {category} - no features available.")
            continue
        
        print(f"Creating violin plots for {category} ({len(features)} features)...")
        
        # Calculate number of pages needed.
        pages = (len(features) + features_per_page - 1) // features_per_page
        
        for page in range(pages):
            start_idx = page * features_per_page
            end_idx = min(start_idx + features_per_page, len(features))
            page_features = features[start_idx:end_idx]
            
            # Create subplot grid.
            rows = int(np.ceil(len(page_features) / 3))
            cols = min(3, len(page_features))
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Create violin plot for each feature.
            for idx, feature in enumerate(page_features):
                ax = axes[idx] if len(page_features) > 1 else axes[0]
                
                # Filter data for current feature.
                feature_data = plot_data[[feature, 'Group']].dropna()
                
                if len(feature_data) == 0:
                    ax.text(0.5, 0.5, f'No data\nfor {feature}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(feature.replace('_', ' ').title())
                    continue
                
                # Apply outlier filtering (central 98%).
                q_low, q_high = feature_data[feature].quantile([0.01, 0.99])
                filtered_data = feature_data[
                    feature_data[feature].between(q_low, q_high)
                ]
                
                # Create violin plot.
                if control_df is not None:
                    palette = {
                        'Control': TIMEPOINT_COLORS.get('control', TIMEPOINT_COLORS['default']),
                        plot_data['Group'].iloc[0]: TIMEPOINT_COLORS.get(timepoint, TIMEPOINT_COLORS['default'])
                    }
                else:
                    palette = [TIMEPOINT_COLORS.get(timepoint, TIMEPOINT_COLORS['default'])]
                
                sns.violinplot(
                    data=filtered_data,
                    x='Group',
                    y=feature,
                    palette=palette,
                    cut=0,
                    ax=ax
                )
                
                # Add median lines.
                for group_idx, group in enumerate(filtered_data['Group'].unique()):
                    group_data = filtered_data[filtered_data['Group'] == group][feature]
                    median_val = group_data.median()
                    
                    ax.hlines(
                        y=median_val,
                        xmin=group_idx - 0.2,
                        xmax=group_idx + 0.2,
                        colors='black',
                        linewidth=3,
                        alpha=0.8
                    )
                
                # Format plot.
                ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel(feature.replace('_', ' ').title())
                
                # Add data range to title.
                ax.set_title(f"{feature.replace('_', ' ').title()}\n[{q_low:.3f}, {q_high:.3f}]")
            
            # Hide unused subplots.
            for idx in range(len(page_features), len(axes)):
                axes[idx].axis('off')
            
            # Add category title and save.
            category_clean = category.replace('_', ' ')
            page_suffix = f"_page_{page + 1}" if pages > 1 else ""
            
            fig.suptitle(f"{category_clean} - Nuclear Morphology Analysis", 
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            
            output_file = output_dir / f"violin_{category.lower().replace(' ', '_')}{page_suffix}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Saved: {output_file.name}")
    
    print("Violin plot generation completed.")


def create_statistical_comparison_plots(
    df: pd.DataFrame,
    control_df: pd.DataFrame,
    available_features: Dict[str, List[str]],
    output_dir: Path
) -> None:
    """
    Create statistical comparison plots between treatment and control groups.

    Generates publication-quality statistical analysis plots including p-value
    distributions, effect size comparisons, and significance summaries.

    Args:
        df: Treatment group DataFrame.
        control_df: Control group DataFrame.
        available_features: Dictionary of available features by category.
        output_dir: Directory to save statistical plots.
    """
    print("\nPerforming statistical analysis and creating comparison plots...")

    # Collect all available features.
    all_features = []
    for features in available_features.values():
        all_features.extend(features)

    if not all_features:
        print("No features available for statistical analysis.")
        return

    # Perform statistical tests.
    results = []

    for feature in all_features:
        if feature not in df.columns or feature not in control_df.columns:
            continue

        # Get clean data for both groups.
        treatment_data = df[feature].dropna()
        control_data = control_df[feature].dropna()

        if len(treatment_data) < 3 or len(control_data) < 3:
            continue

        # Perform Mann-Whitney U test.
        try:
            statistic, p_value = mannwhitneyu(treatment_data, control_data, alternative='two-sided')

            # Calculate effect size (Cohen's d approximation).
            pooled_std = np.sqrt(((len(treatment_data) - 1) * treatment_data.var() +
                                 (len(control_data) - 1) * control_data.var()) /
                                (len(treatment_data) + len(control_data) - 2))

            effect_size = (treatment_data.mean() - control_data.mean()) / pooled_std if pooled_std > 0 else 0

            # Calculate fold change.
            fold_change = treatment_data.mean() / control_data.mean() if control_data.mean() != 0 else np.nan

            results.append({
                'feature': feature,
                'p_value': p_value,
                'effect_size': effect_size,
                'fold_change': fold_change,
                'treatment_mean': treatment_data.mean(),
                'control_mean': control_data.mean(),
                'treatment_std': treatment_data.std(),
                'control_std': control_data.std()
            })

        except Exception as e:
            print(f"Statistical test failed for {feature}: {e}")
            continue

    if not results:
        print("No valid statistical comparisons could be performed.")
        return

    # Convert to DataFrame and apply multiple testing correction.
    stats_df = pd.DataFrame(results)

    # FDR correction.
    _, p_corrected, _, _ = multipletests(stats_df['p_value'], method='fdr_bh')
    stats_df['p_corrected'] = p_corrected
    stats_df['neg_log10_p'] = -np.log10(stats_df['p_corrected'])

    # Create statistical summary plot.
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: P-value distribution.
    axes[0, 0].hist(stats_df['p_value'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(x=0.05, color='red', linestyle='--', label='p = 0.05')
    axes[0, 0].set_xlabel('P-value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of P-values')
    axes[0, 0].legend()

    # Plot 2: Effect size vs significance.
    scatter = axes[0, 1].scatter(stats_df['effect_size'], stats_df['neg_log10_p'],
                                alpha=0.6, c=stats_df['neg_log10_p'], cmap='viridis')
    axes[0, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    axes[0, 1].set_xlabel('Effect Size (Cohen\'s d)')
    axes[0, 1].set_ylabel('-log10(p-value)')
    axes[0, 1].set_title('Effect Size vs Statistical Significance')
    axes[0, 1].legend()
    plt.colorbar(scatter, ax=axes[0, 1], label='-log10(p-value)')

    # Plot 3: Fold change distribution.
    log_fold_change = np.log2(stats_df['fold_change'].replace([np.inf, -np.inf], np.nan).dropna())
    axes[1, 0].hist(log_fold_change, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_xlabel('log2(Fold Change)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Fold Changes')

    # Plot 4: Significance summary by category.
    category_counts = {}
    for category, features in available_features.items():
        significant_features = stats_df[
            (stats_df['feature'].isin(features)) &
            (stats_df['p_corrected'] < 0.05)
        ]
        category_counts[category] = len(significant_features)

    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    bars = axes[1, 1].bar(categories, counts, color=['#E74C3C', '#F39C12', '#27AE60', '#3498DB'])
    axes[1, 1].set_xlabel('Feature Category')
    axes[1, 1].set_ylabel('Number of Significant Features')
    axes[1, 1].set_title('Significant Features by Category (FDR < 0.05)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Add value labels on bars.
    for bar, count in zip(bars, counts):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')

    plt.tight_layout()

    # Save statistical summary plot.
    stats_plot_path = output_dir / "statistical_analysis_summary.png"
    fig.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save detailed statistical results.
    stats_csv_path = output_dir / "statistical_results.csv"
    stats_df.to_csv(stats_csv_path, index=False)

    print(f"Statistical analysis completed:")
    print(f"  Summary plot saved: {stats_plot_path.name}")
    print(f"  Detailed results saved: {stats_csv_path.name}")
    print(f"  Total features tested: {len(stats_df)}")
    print(f"  Significant features (FDR < 0.05): {sum(stats_df['p_corrected'] < 0.05)}")


def create_correlation_heatmap(
    df: pd.DataFrame,
    available_features: Dict[str, List[str]],
    output_dir: Path
) -> None:
    """
    Create correlation heatmap for feature relationships analysis.

    Generates publication-quality correlation matrix heatmaps to visualize
    relationships between different nuclear features and identify redundant measurements.

    Args:
        df: DataFrame containing nuclear features.
        available_features: Dictionary of available features by category.
        output_dir: Directory to save correlation heatmap.
    """
    print("\nCreating feature correlation heatmap...")

    # Collect all numeric features.
    all_features = []
    for features in available_features.values():
        all_features.extend(features)

    # Filter to numeric columns that exist in the data.
    numeric_features = []
    for feature in all_features:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            numeric_features.append(feature)

    if len(numeric_features) < 2:
        print("Insufficient numeric features for correlation analysis.")
        return

    # Calculate correlation matrix.
    feature_data = df[numeric_features].select_dtypes(include=[np.number])
    correlation_matrix = feature_data.corr()

    # Create heatmap.
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create mask for upper triangle to show only lower triangle.
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Generate heatmap.
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=False,
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title('Nuclear Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)

    # Rotate labels for better readability.
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save correlation heatmap.
    heatmap_path = output_dir / "feature_correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Correlation heatmap saved: {heatmap_path.name}")

    # Save correlation matrix as CSV.
    correlation_csv_path = output_dir / "feature_correlations.csv"
    correlation_matrix.to_csv(correlation_csv_path)
    print(f"Correlation matrix saved: {correlation_csv_path.name}")


"""CLI INTERFACE"""

@app.command()
def visualize(
    features_csv: Path = typer.Option(..., exists=True, help="CSV file containing extracted nuclear features."),
    output_dir: Path = typer.Option(Path('./feature_plots'), help="Directory to save visualization plots."),
    control_csv: Optional[Path] = typer.Option(None, exists=True, help="Control group features CSV for comparative analysis."),
    timepoint: Optional[str] = typer.Option(None, help="Timepoint label for color coding (10h, 2d, 14d, etc.)."),
    config: Optional[Path] = typer.Option(None, exists=True, help="Configuration file path (default: uses project config)."),
    min_area: float = typer.Option(10.0, help="Minimum nuclear area for filtering."),
    max_area: float = typer.Option(2000.0, help="Maximum nuclear area for filtering."),
) -> None:
    """
    Generate publication-quality visualizations for nuclear morphological features.

    Creates organized violin plots grouped by feature categories (shape, size, neighborhood,
    texture) with timepoint-specific color coding optimized for kidney I/R injury analysis.
    Includes comprehensive statistical analysis and correlation matrices.

    Example usage:
        python visualize_engineered_features_refactored.py visualize \\
            --features_csv nuclear_features.csv \\
            --output_dir ./plots \\
            --control_csv control_features.csv \\
            --timepoint 10h
    """
    print("="*80)
    print("NUCLEAR FEATURE VISUALIZATION FOR KIDNEY I/R INJURY ANALYSIS")
    print("="*80)

    try:
        # Load configuration if provided.
        if config and config.exists():
            settings, _, _ = load_config(config)
        else:
            settings, _, _ = load_config()

        # Override settings with command-line arguments.
        features_per_page = settings.get('features_per_page', 9)

        print(f"Configuration loaded:")
        print(f"  Features per page: {features_per_page}")
        print(f"  Timepoint color coding: {settings.get('timepoint_color_coding', True)}")
        print(f"  Statistical testing: {settings.get('enable_statistical_testing', True)}")

        # Create output directory.
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Load and validate feature data.
        df = load_and_validate_data(features_csv, min_area, max_area)

        # Load control data if provided.
        control_df = None
        if control_csv:
            print(f"\nLoading control group data...")
            control_df = load_and_validate_data(control_csv, min_area, max_area)

        # Identify available features by category.
        available_features = identify_available_features(df)

        # Generate violin plots by category.
        create_violin_plots_by_category(
            df=df,
            available_features=available_features,
            output_dir=output_dir,
            timepoint=timepoint,
            control_df=control_df,
            features_per_page=features_per_page
        )

        # Generate statistical comparison plots if control data is available.
        if control_df is not None and settings.get('enable_statistical_testing', True):
            create_statistical_comparison_plots(
                df=df,
                control_df=control_df,
                available_features=available_features,
                output_dir=output_dir
            )

        # Generate correlation heatmap.
        create_correlation_heatmap(
            df=df,
            available_features=available_features,
            output_dir=output_dir
        )

        print("\n" + "="*80)
        print("VISUALIZATION GENERATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"All plots saved to: {output_dir}")

        # Summary of generated files.
        plot_files = list(output_dir.glob("*.png"))
        csv_files = list(output_dir.glob("*.csv"))

        print(f"\nGenerated files:")
        print(f"  Visualization plots: {len(plot_files)}")
        print(f"  Data files: {len(csv_files)}")

        for plot_file in sorted(plot_files):
            print(f"    📊 {plot_file.name}")

        for csv_file in sorted(csv_files):
            print(f"    📄 {csv_file.name}")

    except Exception as e:
        print(f"Visualization generation failed: {e}")
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """
    Display information about feature categories and visualization options.
    """
    print("="*80)
    print("NUCLEAR FEATURE VISUALIZATION CATEGORIES")
    print("="*80)

    for category, features in FEATURE_CATEGORIES.items():
        print(f"\n📊 {category.upper()}:")
        print(f"   Features: {len(features)}")
        for feature in features[:5]:  # Show first 5 features.
            print(f"   • {feature}")
        if len(features) > 5:
            print(f"   • ... and {len(features) - 5} more")

    print(f"\n🎨 TIMEPOINT COLOR CODING:")
    for timepoint, color in TIMEPOINT_COLORS.items():
        print(f"   • {timepoint}: {color}")

    print(f"\n📈 VISUALIZATION TYPES:")
    print(f"   • Violin plots by feature category")
    print(f"   • Statistical comparison plots")
    print(f"   • Correlation heatmaps")
    print(f"   • Feature distribution summaries")

    print("\n" + "="*80)


if __name__ == "__main__":
    app()
