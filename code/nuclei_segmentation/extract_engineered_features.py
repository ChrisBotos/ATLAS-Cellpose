#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: extract_engineered_features.py.
Description:
    Extract comprehensive nuclear features (morphological, intensity, texture, neighborhood,
    spatial context) from segmented DAPI-stained images using modern libraries and CLI.

Dependencies:
    • Python >= 3.9.
    • numpy, pandas, scipy, scikit-image, scikit-learn, Pillow, typer.

Usage:
    python extract_engineered_features.py --image <path/to/image> --mask <path/to/mask> [--output <path/to/output.csv>] [--neighbor_radius <radius>] [--jobs <n_jobs>]
    python extract_engineered_features.py \
  --image ../../results/an_example_result/preprocessed/cropped.tif \
  --mask  ../../results/an_example_result/masks/segmentation_masks.npy \
  --output ../../results/an_example_result/features.csv \
  --neighbor-radius 75.0 \
  --jobs 6

Positional Arguments:
    None.

Optional Arguments:
    --image             Path to input grayscale image (TIFF).
    --mask              Path to segmentation mask (NumPy .npy or image file).
    --output            CSV file to write features (default: features.csv).
    --neighbor_radius   Radius for neighbor search in pixels (default: 50.0).
    --jobs              Number of parallel workers (default: CPU count).

Inputs:
    • Grayscale DAPI image of kidney tissue section.
    • Labeled segmentation mask of nuclei.

Outputs:
    • CSV file with one row per nucleus and columns for each feature.

Key Features:
    • Modern CLI with Typer for validation and help.
    • Parallel processing via concurrent.futures.
    • Full feature suite: morphological (area, perimeter, axes, aspect ratio, circularity,
      solidity, eccentricity, Feret diameter, roughness, bounding box, fractal dimension),
      intensity (mean, std, median, skewness, kurtosis, entropy), LBP texture (bins 0–10),
      neighborhood (k-NN: mean/std area, eccentricity mean, orientation alignment,
      nearest distance, cluster density/elongation/polarization/area ratio),
      spatial context (centroid coords, distance to center/edge), distance to dark/sparse regions.

Notes:
    • Fractal dimension via box-counting on binary mask.
    • Sparse zones defined as connected low-intensity regions > 64×64 px.
"""

from pathlib import Path
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import entropy, skew, kurtosis
from scipy.spatial import cKDTree
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops, label
from sklearn.decomposition import PCA
import typer

app = typer.Typer()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_dark_distance_map(gray: np.ndarray, threshold: int = 50) -> np.ndarray:
    """
    Compute distance from each pixel to nearest dark region (intensity < threshold).
    """
    mask_dark = gray < threshold
    return ndimage.distance_transform_edt(~mask_dark)


def compute_sparse_distance_map(
    gray: np.ndarray,
    mask: np.ndarray,
    intensity_threshold: int = 5,
    min_size: int = 64 * 64,
) -> np.ndarray:
    """
    Compute distance to nearest sparse zone (large low-intensity background regions).
    """
    sparse = (gray < intensity_threshold) & (mask == 0)
    labeled_sparse, num = ndimage.label(sparse)
    keep = np.zeros_like(sparse)
    for idx in range(1, num + 1):
        comp = labeled_sparse == idx
        if comp.sum() >= min_size:
            keep |= comp
    return ndimage.distance_transform_edt(~keep)


def fractal_dimension(binary_mask: np.ndarray) -> float:
    """
    Estimate fractal dimension via box-counting method, robust to arbitrary shapes.
    Returns NaN if shape too small for estimation.
    """
    max_dim = min(binary_mask.shape)
    if max_dim < 2:
        return np.nan
    sizes = 2 ** np.arange(int(np.log2(max_dim)), 1, -1)
    if sizes.size < 2:
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
    return float(-coeff[0])


def compute_region_features(
    region: Any,
    neighbor_data: Dict[str, Any],
    dark_map: np.ndarray,
    sparse_map: np.ndarray,
    gray: np.ndarray,
    shape: tuple,
) -> Dict[str, Any]:
    """
    Extract features for a single nucleus region.
    """
    cy, cx = region.centroid
    area = region.area
    perimeter = region.perimeter or np.nan
    major = region.major_axis_length
    minor = region.minor_axis_length
    aspect_ratio = major / minor if minor else np.nan
    circularity = (4 * np.pi * area / perimeter**2) if perimeter else np.nan
    solidity = region.solidity
    eccentricity = region.eccentricity
    feret = getattr(region, 'feret_diameter_max', np.nan)
    roughness = perimeter / np.sqrt(area) if area else np.nan
    minr, minc, maxr, maxc = region.bbox
    bbox_w = maxc - minc
    bbox_h = maxr - minr
    frac_dim = fractal_dimension(region.image)

    patch = gray[minr:maxr, minc:maxc]
    mask_patch = region.image
    vals = patch[mask_patch]

    mean_int = float(vals.mean())
    std_int = float(vals.std())
    median_int = float(np.median(vals))
    skew_int = float(skew(vals)) if vals.size else np.nan
    kurt_int = float(kurtosis(vals)) if vals.size else np.nan
    hist = np.histogram(vals, bins=32)[0]
    tex_ent = float(entropy(hist + 1))

    # Local Binary Patterns: capture micro-texture of chromatin structure—
    # reflects euchromatin/heterochromatin granularity and edge features.
    lbp = local_binary_pattern(patch, P=8, R=1, method='uniform')
    lbp_vals = lbp[mask_patch]
    lbp_hist, _ = np.histogram(lbp_vals, bins=np.arange(12), density=True)

    centroids = neighbor_data['centroids']
    areas = neighbor_data['areas']
    eccs = neighbor_data['eccs']
    orients = neighbor_data['orients']
    dists = [np.hypot(cx - x, cy - y) for y, x in centroids]
    dist_min = float(min(dists)) if dists else 0.0
    cluster_density = len(dists) / (np.pi * neighbor_data['radius']**2)
    if centroids:
        coords = np.array([[x, y] for y, x in centroids])
        pca = PCA(n_components=2).fit(coords)
        eig = pca.explained_variance_
        elong = float(eig[0] / eig[1]) if eig[1] else 0.0
        cosines = [np.cos(region.orientation - o) for o in orients]
        pol = float(np.mean(cosines))
    else:
        elong = pol = 0.0

    neigh_mean_area = float(np.mean(areas)) if areas else 0.0

    h, w = shape
    dist_center = float(np.hypot(cx - w/2, cy - h/2))
    dist_edge = float(min(cx, cy, w-cx, h-cy))
    iy, ix = int(round(cy)), int(round(cx))
    dist_dark = float(dark_map[iy, ix])
    dist_sparse = float(sparse_map[iy, ix])

    feats = {
        'Label': region.label,
        'Centroid_X': cx,
        'Centroid_Y': cy,
        'Area': area,
        'Perimeter': perimeter,
        'Major_Axis_Length': major,
        'Minor_Axis_Length': minor,
        'Aspect_Ratio': aspect_ratio,
        'Circularity': circularity,
        'Eccentricity': eccentricity,
        'Solidity': solidity,
        'Feret_Diameter': feret,
        'Roughness_Index': roughness,
        'Bounding_Box_Width': bbox_w,
        'Bounding_Box_Height': bbox_h,
        'Fractal_Dimension': frac_dim,
        'Intensity_Mean': mean_int,
        'Intensity_Std': std_int,
        'Intensity_Median': median_int,
        'Intensity_Skewness': skew_int,
        'Intensity_Kurtosis': kurt_int,
        'Texture_Entropy': tex_ent,
        'Distance_to_Image_Center': dist_center,
        'Distance_to_Image_Edge': dist_edge,
        'Distance_to_Dark_Region': dist_dark,
        'Distance_to_Sparse_Zone': dist_sparse,
        'Neighborhood_Mean_Area': neigh_mean_area,
        'Neighborhood_Std_Area': float(np.std(areas)) if areas else 0.0,
        'Neighborhood_Eccentricity_Mean': float(np.mean(eccs)) if eccs else 0.0,
        'Orientation_Alignment_Std': float(np.std([region.orientation - o for o in orients])),
        'Distance_to_Nearest_Nucleus': dist_min,
        'Cluster_Density_Index': cluster_density,
        'Cluster_Elongation': elong,
        'Cluster_Polarization_Score': pol,
        'Cluster_Area_Ratio': area / neigh_mean_area if neigh_mean_area else 0.0,
    }
    for i in range(11):
        feats[f'LBP_Bin_{i}'] = float(lbp_hist[i] if i < len(lbp_hist) else 0.0)

    return feats


def build_neighbors_list(
    props: List[Any],
    tree: cKDTree,
    radius: float,
) -> List[Dict[str, Any]]:
    """
    For each region, collect neighbor centroids, areas, eccentricities, orientations.
    """
    data = []
    for idx, r in enumerate(props):
        ids = [i for i in tree.query_ball_point(r.centroid, radius) if i != idx]
        data.append({
            'centroids': [props[i].centroid for i in ids],
            'areas': [props[i].area for i in ids],
            'eccs': [props[i].eccentricity for i in ids],
            'orients': [props[i].orientation for i in ids],
            'radius': radius,
        })
    return data


def process_image(
    image_path: Path,
    mask_path: Path,
    output_csv: Path,
    neighbor_radius: float,
    jobs: int,
) -> pd.DataFrame:
    """
    Load image and mask, compute context maps, extract features in parallel, write CSV.
    """
    gray = np.array(Image.open(image_path).convert('L'))
    mask_arr = (
        np.load(mask_path) if mask_path.suffix == '.npy'
        else np.array(Image.open(mask_path))
    )
    mask_arr = label(mask_arr)

    dark_map = compute_dark_distance_map(gray)
    sparse_map = compute_sparse_distance_map(gray, mask_arr)

    props = regionprops(mask_arr)
    centroids = [r.centroid for r in props]
    tree = cKDTree(centroids)
    neighbors = build_neighbors_list(props, tree, neighbor_radius)

    workers = jobs if jobs > 0 else multiprocessing.cpu_count()
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [
            exe.submit(
                compute_region_features,
                props[i], neighbors[i], dark_map, sparse_map, gray, gray.shape,
            ) for i in range(len(props))
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Extracted {len(df)} nuclei features to {output_csv}")
    return df


@app.command()
def extract(
    image: Path = typer.Option(..., exists=True, help="Input TIFF image."),
    mask: Path = typer.Option(..., exists=True, help="Segmentation mask (.npy or image)."),
    output: Path = typer.Option(Path('features.csv'), help="Output CSV path."),
    neighbor_radius: float = typer.Option(50.0, help="Neighbor search radius."),
    jobs: int = typer.Option(-1, help="Number of parallel workers."),
) -> None:
    """
    Command: extract features from nuclei and save to CSV.
    """
    process_image(image, mask, output, neighbor_radius, jobs)


if __name__ == '__main__':
    app()
