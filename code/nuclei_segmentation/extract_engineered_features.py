#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: extract_engineered_features.py.
Description:
    Advanced nuclear feature extraction for ischemia-reperfusion kidney injury analysis.

Dependencies:
    • Python >= 3.7.
    • numpy, pandas, PIL, scipy, scikit-image, joblib.
    • cupy (optional, for GPU acceleration).

Usage:
    python extract_engineered_features.py --image <path/to/image> --mask <path/to/mask> [--output <path/to/output.csv>] [--neighbor_radius <radius>] [--jobs <n_jobs>]
    python extract_engineered_features.py \
  --image ../../results/an_example_result/preprocessed/cropped.tif \
  --mask  ../../results/an_example_result/masks/segmentation_masks.npy \
  --output ../../results/an_example_result/features.csv \
  --neighbor_radius 75.0 \
  --jobs 6

Positional Arguments:
    None.

Optional Arguments:
    --image            Path to input grayscale image.
    --mask             Path to segmentation mask (.npy or image).
    --output           Output CSV file (default: features.csv).
    --neighbor_radius  Radius to search for neighboring nuclei in pixels (default: 50.0).
    --jobs             Number of parallel jobs/CPU cores (default: -1 for all cores).

Inputs:
    • Grayscale image of kidney tissue section.
    • Segmentation mask with labeled nuclei.

Outputs:
    • CSV file containing extracted nuclear features including shape metrics, intensity statistics, and neighborhood relationships.

Key Features:
    • GPU acceleration via CuPy when available.
    • Parallel processing for faster analysis.
    • Comprehensive feature extraction including shape, intensity, and neighborhood metrics.
    • Robust handling of edge cases to prevent NaN values.
    • Distance mapping to dark regions for contextual analysis.

Notes:
    • This script is optimized for analyzing nuclear morphology changes during kidney I/R injury.
    • Features extracted are relevant for identifying different cell death mechanisms (apoptosis, pyroptosis, etc.).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import entropy, skew, kurtosis
from scipy.spatial import cKDTree
from skimage.measure import regionprops, label
from joblib import Parallel, delayed

# Attempt GPU acceleration via CuPy.
try:
    import cupy as cp
    xp = cp
    USE_GPU = True
    print("Using GPU acceleration with CuPy for faster nuclear feature extraction!")
except ImportError:
    xp = np
    USE_GPU = False
    print("Using CPU (NumPy) for nuclear feature extraction.")

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_dark_distance_map(gray: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Compute Euclidean distance map from dark regions in the grayscale image.

    Args:
        gray: 2D numpy array of grayscale intensities.
        threshold: Intensity below which pixels are considered dark.

    Returns:
        Distance map array of same shape as input.
    """
    mask = gray < threshold
    return distance_transform_edt(~mask)


def extract_region_intensity(pil_img: Image.Image, region, offset: tuple) -> np.ndarray:
    """
    Crop and return the grayscale intensity patch for a region.

    Args:
        pil_img: Full PIL image in grayscale.
        region: skimage RegionProperties object.
        offset: (x_offset, y_offset) for tiled images.

    Returns:
        2D numpy array of pixel values within region bbox.
    """
    minr, minc, maxr, maxc = region.bbox
    x0, y0 = offset
    crop = pil_img.crop((minc + x0, minr + y0, maxc + x0, maxr + y0))
    return np.array(crop)


def compute_features(region, neighbors, dark_map, pil_gray, offset):
    """
    Compute shape, intensity, and neighborhood features for one nucleus.
    """
    cy, cx = region.centroid
    x_off, y_off = offset
    cy += y_off
    cx += x_off

    # Basic shape metrics.
    area = region.area
    perimeter = region.perimeter or np.nan
    circularity = (4 * np.pi * area / perimeter**2) if perimeter > 0 else np.nan
    major = region.major_axis_length
    minor = region.minor_axis_length
    aspect_ratio = major / minor if minor > 0 else np.nan
    solidity = region.solidity
    eccentricity = region.eccentricity

    # Intensity statistics.
    patch = extract_region_intensity(pil_gray, region, offset)
    mask = region.image
    values = patch[mask]
    intensity = {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'median': float(np.median(values)),
        'skew': float(skew(values)) if values.size else np.nan,
        'kurtosis': float(kurtosis(values)) if values.size else np.nan,
    }
    hist = np.histogram(values, bins=32)[0]
    texture_entropy = float(entropy(hist + 1))

    # Neighbor stats.
    n = len(neighbors['centroids'])
    if n:
        dists = [
            np.hypot(cx - c[1] - x_off, cy - c[0] - y_off)
            for c in neighbors['centroids']
        ]
        neighbor_count = n
        dist_mean, dist_std = float(np.mean(dists)), float(np.std(dists))
        orientation_diffs = [region.orientation - o for o in neighbors['orientations']]
        orient_std = float(np.std(orientation_diffs))
    else:
        neighbor_count = 0
        dist_mean = dist_std = orient_std = 0.0

    # Distance to dark regions.
    h, w = dark_map.shape
    iy, ix = int(round(cy)), int(round(cx))
    if 0 <= iy < h and 0 <= ix < w:
        dist_to_dark = float(dark_map[iy, ix])
    else:
        dist_to_dark = 0.0

    return {
        'Label': region.label,
        'Centroid_Y': cy,
        'Centroid_X': cx,
        'Area': area,
        'Perimeter': perimeter,
        'Circularity': circularity,
        'Aspect_Ratio': aspect_ratio,
        'Eccentricity': eccentricity,
        'Solidity': solidity,
        'Intensity_Mean': intensity['mean'],
        'Intensity_Std': intensity['std'],
        'Intensity_Median': intensity['median'],
        'Intensity_Skew': intensity['skew'],
        'Intensity_Kurtosis': intensity['kurtosis'],
        'Texture_Entropy': texture_entropy,
        'Neighbor_Count': neighbor_count,
        'Neighbor_Dist_Mean': dist_mean,
        'Neighbor_Dist_Std': dist_std,
        'Neighbor_Orient_Std': orient_std,
        'Dist_To_Dark': dist_to_dark,
    }


def process_mask_regions(mask: np.ndarray, radius: float) -> list:
    """
    Precompute neighbor info for each region.

    Args:
        mask: 2D integer array of labeled nuclei.
        radius: Distance threshold to consider nuclei as neighbors.

    Returns:
        Tuple of (region properties list, neighbor info list).
    """
    props = regionprops(mask)
    centroids = [r.centroid for r in props]
    tree = cKDTree(centroids)
    neighbor_list = []
    for i, c in enumerate(centroids):
        ids = [j for j in tree.query_ball_point(c, radius) if j != i]
        neighbor_list.append({
            'centroids': [centroids[j] for j in ids],
            'orientations': [props[j].orientation for j in ids]
        })
    return props, neighbor_list


def extract_features(
    image_path: Path,
    mask_path: Path,
    output_csv: Path,
    radius: float,
    n_jobs: int
) -> pd.DataFrame:
    """Extract features for all nuclei and save to CSV."""
    img = Image.open(image_path).convert('L')
    gray = np.array(img)
    dark_map = compute_dark_distance_map(gray)

    mask = np.load(mask_path) if mask_path.suffix == '.npy' else np.array(Image.open(mask_path))
    if not np.issubdtype(mask.dtype, np.integer):
        mask = label(mask)

    props, neighbors = process_mask_regions(mask, radius)

    logger.info(f"Computing features for {len(props)} regions...")
    tasks = (
        delayed(compute_features)(r, neighbors[i], dark_map, img, (0, 0))
        for i, r in enumerate(props)
    )
    results = Parallel(n_jobs=n_jobs)(tasks)

    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved features to {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract nuclear features from segmented images.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input grayscale image.")
    parser.add_argument("--mask", type=Path, required=True, help="Path to segmentation mask (.npy or image).")
    parser.add_argument("--output", type=Path, default=Path("features.csv"), help="Output CSV file.")
    parser.add_argument("--neighbor_radius", type=float, default=50.0,
                        help="Radius to search for neighboring nuclei (pixels).")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs (CPU cores).")
    args = parser.parse_args()

    start = time.time()
    extract_features(
        image_path=args.image,
        mask_path=args.mask,
        output_csv=args.output,
        radius=args.neighbor_radius,
        n_jobs=args.jobs
    )
    elapsed = time.time() - start
    logger.info(f"Total processing time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
