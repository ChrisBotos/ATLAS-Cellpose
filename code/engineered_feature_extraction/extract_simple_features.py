#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center.
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: extract_simple_features.py.
Description:
    Fast extraction of comprehensive nuclear size and shape features from segmented
    DAPI-stained tissue sections. This optimized version includes all essential size
    measurements and basic shape features for kidney ischemia-reperfusion injury analysis.

Dependencies:
    • Python >= 3.10.
    • numpy, pandas, scikit-image, typer, rich.

Usage:
    python extract_simple_features.py --config ../../configs/engineered_feature_extraction_config.ini

Arguments:
    --config            Path to configuration file containing extraction parameters.

Key Features:
    • Fast single-threaded processing for reliability.
    • Comprehensive size features: area, perimeter, axes, bounding box, Feret diameters.
    • Basic shape features: circularity, aspect ratio.
    • Clean progress reporting with rich console.
    • Robust error handling and validation.
    • Scientific context for kidney I/R injury research.

Notes:
    • Size features quantify nuclear dimensions and spatial extent.
    • Shape features measure nuclear morphology and regularity.
    • Results saved as CSV with nucleus_id and all extracted features.
"""

import traceback
import sys
import os
from pathlib import Path
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import typer
from PIL import Image
from skimage.measure import regionprops
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich import print as rprint

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
    Extract comprehensive size and basic shape features from nuclear region.

    Size features quantify nuclear dimensions and spatial extent.
    Shape features measure nuclear morphology and regularity.

    Args:
        region: Regionprops object containing nuclear measurements.

    Returns:
        Dictionary with all size and basic shape feature values.
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

    '''Basic Shape Features'''
    # Circularity: shape regularity measure (4π*area/perimeter²).
    if perimeter > 0:
        features['circularity'] = 4 * np.pi * area / (perimeter ** 2)
    else:
        features['circularity'] = 0.0

    # Aspect ratio: elongation measure.
    if region.minor_axis_length > 0:
        features['aspect_ratio'] = region.major_axis_length / region.minor_axis_length
    else:
        features['aspect_ratio'] = 1.0

    return features


def process_nuclei(gray: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """
    Process all nuclei and extract comprehensive size and shape features.

    Args:
        gray: Grayscale DAPI image.
        mask: Segmentation mask with labeled nuclei.

    Returns:
        DataFrame with nucleus_id and all extracted size/shape features.
    """
    console.print("[cyan]Extracting nuclear region properties...[/cyan]")

    # Extract region properties from mask.
    props = regionprops(mask, intensity_image=gray)
    console.print(f"[green]✓[/green] Found {len(props)} nuclear regions")

    # Process each nucleus with progress bar.
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task("Extracting features...", total=len(props))

        for region in props:
            # Extract comprehensive features.
            features = extract_basic_features(region)

            # Create result with nucleus identifier and all features.
            result = {'nucleus_id': region.label}
            result.update(features)

            results.append(result)
            progress.update(task, advance=1)

    # Convert to DataFrame.
    df = pd.DataFrame(results)
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
    console.print(f"  [cyan]Aspect Ratio:[/cyan] mean={df['aspect_ratio'].mean():.2f}, std={df['aspect_ratio'].std():.2f}")


def main(
    config: Path = typer.Option(..., exists=True, help="Configuration file containing extraction parameters")
) -> None:
    """
    Extract comprehensive nuclear size and shape features from segmented tissue using config.

    This command processes DAPI-stained tissue sections to extract essential morphological
    features for kidney ischemia-reperfusion injury analysis. Includes all size measurements
    (area, perimeter, axes, bounding box, Feret diameters) and basic shape features
    (circularity, aspect ratio) for comprehensive nuclear characterization.

    Example:
        python extract_simple_features.py \\
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
        output_dir = Path(settings.get('extraction_output_dir', 'results/simple_features')).resolve()
        output_path = output_dir / 'simple_features.csv'

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

        # Step 3: Extract features.
        df = process_nuclei(gray, mask_array)

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
