#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: visualize_engineered_features.py.
Description:
    Generate violin, bar, scatter, and heatmap visualizations from extracted nuclear features for ischemia-reperfusion kidney injury analysis.

Dependencies:
    • Python >= 3.8.
    • numpy, pandas, scipy, statsmodels, matplotlib, seaborn.

Usage:
    python visualize_engineered_features.py \
        --iri_csv <path/to/iri_features.csv> \
        [--cntl_csv <path/to/cntl_features.csv>] \
        [--output_dir <path/to/output_dir>] \
        [--min_area <float>] [--max_area <float>]

Positional Arguments:
    None.

Optional Arguments:
    --iri_csv      Path to CSV of features from IRI samples.
    --cntl_csv     Path to CSV of features from control samples (optional).
    --output_dir   Directory to save plots (default: ./plots).
    --min_area     Minimum nuclear area for filtering (default: 5.0).
    --max_area     Maximum nuclear area for filtering (default: 1000.0).

Inputs:
    • IRI features CSV file.
    • Control features CSV file (if provided).

Outputs:
    • Violin plot pages comparing feature distributions.
    • Bar plot of FDR-corrected p-values (-log10 scale).
    • Scatter plot of IRI/CNTL mean ratios.
    • Correlation matrix heatmap of features.

Key Features:
    • Non-parametric group comparisons using Mann-Whitney U tests.
    • Multiple testing correction via FDR.
    • Central 98% data trimming to reduce outlier impact.
    • Modular invalid-nuclei filtering based on area, circularity, solidity, and aspect ratio.
Notes:
    • Uses seaborn-whitegrid style for consistency.
"""

from pathlib import Path
import warnings
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

# Apply consistent plotting style.
sns.set(style='whitegrid')

# Suppress seaborn palette deprecation warnings.
warnings.filterwarnings(
    'ignore',
    message='Passing `palette` without assigning `hue`'
)

# ------------------------------------------------------------------
# Define exactly the FINAL list of features in the order to plot.
# ------------------------------------------------------------------
FEATURE_LIST = [
    # Morphological
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Aspect_Ratio",
    "Circularity",
    "Eccentricity",
    "Solidity",
    "Feret_Diameter",
    "Roughness_Index",
    "Bounding_Box_Width",
    "Bounding_Box_Height",
    "Fractal_Dimension",
    # Intensity
    "Mean_Intensity",
    "Intensity_Std",
    "Texture_Entropy",
    # Neighborhood (k-NN)
    "Neighborhood_Mean_Area",
    "Neighborhood_Std_Area",
    "Neighborhood_Eccentricity_Mean",
    "Orientation_Alignment_Std",
    "Distance_to_Nearest_Nucleus",
    "Cluster_Density_Index",
    "Cluster_Elongation",
    "Cluster_Polarization_Score",
    "Cluster_Area_Ratio",
    "Distance_to_Sparse_Zone",
    # Spatial Context
    "Centroid_X",
    "Centroid_Y",
    "Distance_to_Image_Center",
    "Distance_to_Image_Edge",
    # Minimal Texture (LBP bins 0–10)
] + [f"LBP_Bin_{i}" for i in range(0, 11)]


def filter_invalid_masks(
    df: pd.DataFrame,
    min_area: float = 5.0,
    max_area: float = 1000.0
) -> pd.DataFrame:
    """
    Filter out invalid nuclear measurements based on area and morphology constraints.
    """
    mask = (
        (df['Area'] >= min_area)
        & (df['Area'] <= max_area)
    )

    if 'Circularity' in df.columns:
        mask &= (df['Circularity'] <= 1.0)

    if 'Aspect_Ratio' in df.columns:
        mask &= (df['Aspect_Ratio'] <= 10.0)

    if 'Solidity' in df.columns:
        mask &= (df['Solidity'] >= 0.5)

    return df.loc[mask].copy()


def load_data(path: Path) -> pd.DataFrame:
    """
    Load nuclear feature data from CSV file.
    """
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {path}: {e}", file=sys.stderr)
        sys.exit(1)


def plot_group_comparison(
    iri_df: pd.DataFrame,
    cntl_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create comparative plots between IRI and control groups, only on FEATURE_LIST.
    """
    # Tag and concatenate
    iri_df = iri_df.copy()
    cntl_df = cntl_df.copy()
    iri_df['Group'] = 'IRI'
    cntl_df['Group'] = 'CNTL'
    data = pd.concat([iri_df, cntl_df], ignore_index=True)

    # Select only the features we defined, in order, and warn if missing
    feature_cols = []
    for feat in FEATURE_LIST:
        if feat in data.columns:
            feature_cols.append(feat)
        else:
            warnings.warn(f"Feature column '{feat}' not found in data and will be skipped.")

    if not feature_cols:
        raise ValueError("No matching feature columns found in either IRI or CNTL data.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Violin plots paginated
    per_page = 9
    pages = (len(feature_cols) + per_page - 1) // per_page

    for page in range(pages):
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        axes = axes.flatten()

        for idx, feat in enumerate(feature_cols[page*per_page:(page+1)*per_page]):
            ax = axes[idx]
            # clip central 98%
            q_low, q_high = data[feat].quantile([0.01, 0.99])
            subset = data.loc[data[feat].between(q_low, q_high)]

            sns.violinplot(
                x='Group',
                y=feat,
                hue='Group',
                data=subset,
                palette={'IRI': '#4c72b0', 'CNTL': '#dd8452'},
                cut=0,
                ax=ax,
                legend=False
            )

            # overlay medians
            for grp, pos in zip(['IRI', 'CNTL'], [0, 1]):
                med = subset.loc[subset['Group'] == grp, feat].median()
                ax.hlines(
                    y=med,
                    xmin=pos - 0.2,
                    xmax=pos + 0.2,
                    colors='k',
                    linewidth=4
                )

            ax.set_title(f"{feat} [{q_low:.2f}, {q_high:.2f}]")
            ax.set_xlabel('')

        # turn off unused axes
        for ax in axes[len(feature_cols[page*per_page:(page+1)*per_page]):]:
            ax.axis('off')

        plt.tight_layout()
        fig.savefig(output_dir / f'violin_page_{page+1}.png')
        plt.close(fig)

    # 2) Statistical tests
    stats = []
    for feat in feature_cols:
        arr_iri = np.clip(iri_df[feat], *iri_df[feat].quantile([0.01, 0.99]))
        arr_cntl = np.clip(cntl_df[feat], *cntl_df[feat].quantile([0.01, 0.99]))
        mean_iri = arr_iri.mean()
        mean_cntl = arr_cntl.mean()
        ratio = mean_iri / mean_cntl if mean_cntl else np.nan

        try:
            pval = mannwhitneyu(arr_iri, arr_cntl, alternative='two-sided').pvalue
        except Exception:
            pval = np.nan

        stats.append({
            'Feature': feat,
            'IRI Mean': mean_iri,
            'CNTL Mean': mean_cntl,
            'Ratio': ratio,
            'p-value': pval
        })

    comp_df = pd.DataFrame(stats).sort_values('p-value')

    # FDR correction
    mask = comp_df['p-value'].notna()
    if mask.any():
        comp_df.loc[mask, 'FDR p'] = multipletests(comp_df.loc[mask, 'p-value'], method='fdr_bh')[1]
    else:
        comp_df['FDR p'] = np.nan

    comp_df['FDR p'] = comp_df['FDR p'].replace(0, 1e-300)
    comp_df['-log10(FDR p)'] = -np.log10(comp_df['FDR p'])
    comp_df.to_csv(output_dir / 'feature_comparison.csv', index=False)

    # 3) Bar plot of –log10(FDR p)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='-log10(FDR p)',
        y='Feature',
        data=comp_df,
        palette='magma',
        ax=ax
    )
    ax.axvline(-np.log10(0.05), ls='--', lw=2)
    ax.set_title('FDR-Corrected p-values')
    fig.tight_layout()
    fig.savefig(output_dir / 'bar_fdr.png')
    plt.close(fig)

    # 4) Scatter plot of mean ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='Feature',
        y='Ratio',
        data=comp_df,
        s=100,
        ax=ax
    )
    ax.axhline(1.0, ls='--', lw=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('IRI/CNTL Mean Ratio')
    fig.tight_layout()
    fig.savefig(output_dir / 'scatter_ratio.png')
    plt.close(fig)

    # 5) Correlation heatmap
    corr = data[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        corr,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar=True,
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix')
    fig.tight_layout()
    fig.savefig(output_dir / 'heatmap_corr.png')
    plt.close(fig)


def plot_single_group(
    df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create violin plots and heatmap for a single group of nuclear features.
    """
    df = df.copy()
    df['Group'] = 'IRI'

    # Select exactly the same features
    feature_cols = []
    for feat in FEATURE_LIST:
        if feat in df.columns:
            feature_cols.append(feat)
        else:
            warnings.warn(f"Feature column '{feat}' not found and will be skipped.")

    if not feature_cols:
        raise ValueError("No matching feature columns found for single-group plotting.")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_page = 9
    pages = (len(feature_cols) + per_page - 1) // per_page

    # Violin pages
    for page in range(pages):
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        axes = axes.flatten()

        for idx, feat in enumerate(feature_cols[page*per_page:(page+1)*per_page]):
            ax = axes[idx]
            q_low, q_high = df[feat].quantile([0.01, 0.99])
            subset = df.loc[df[feat].between(q_low, q_high)]

            sns.violinplot(
                x='Group',
                y=feat,
                hue='Group',
                data=subset,
                palette='Set2',
                cut=0,
                ax=ax,
                legend=False
            )

            med = subset[feat].median()
            ax.hlines(y=med, xmin=-0.2, xmax=0.2, colors='k', linewidth=4)
            ax.set_title(f"{feat} [{q_low:.2f}, {q_high:.2f}]")
            ax.set_xlabel('')

        for ax in axes[len(feature_cols[page*per_page:(page+1)*per_page]):]:
            ax.axis('off')

        plt.tight_layout()
        fig.savefig(output_dir / f'violin_single_{page+1}.png')
        plt.close(fig)

    # Heatmap
    corr = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        corr,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar=True,
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix')
    fig.tight_layout()
    fig.savefig(output_dir / 'heatmap_single.png')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate visual summaries of nuclear features.'
    )
    parser.add_argument(
        '--iri_csv',
        type=Path,
        required=True,
        help='Path to IRI features CSV file.'
    )
    parser.add_argument(
        '--cntl_csv',
        type=Path,
        default=None,
        help='Path to control features CSV file (optional).'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('plots'),
        help='Directory to save plots.'
    )
    parser.add_argument(
        '--min_area',
        type=float,
        default=5.0,
        help='Minimum nuclear area for filtering.'
    )
    parser.add_argument(
        '--max_area',
        type=float,
        default=1000.0,
        help='Maximum nuclear area for filtering.'
    )
    args = parser.parse_args()

    iri_df = load_data(args.iri_csv)
    iri_df = filter_invalid_masks(iri_df, args.min_area, args.max_area)

    if args.cntl_csv:
        cntl_df = load_data(args.cntl_csv)
        cntl_df = filter_invalid_masks(cntl_df, args.min_area, args.max_area)
        plot_group_comparison(iri_df, cntl_df, args.output_dir)
    else:
        plot_single_group(iri_df, args.output_dir)


if __name__ == '__main__':
    main()
