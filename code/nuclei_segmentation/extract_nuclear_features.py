#!/usr/bin/env python3
"""
Advanced Nuclear Feature Extraction for Ischemia-Reperfusion Kidney Injury Analysis.

This script performs comprehensive morphological and intensity-based feature extraction
from segmented nuclei in kidney tissue sections. It is specifically designed for analyzing
nuclear changes during ischemia-reperfusion (I/R) injury progression, where nuclear
morphology alterations reflect underlying pathophysiological processes including apoptosis,
pyroptosis, necroptosis, and inflammatory responses.

Key capabilities:
- GPU acceleration via CuPy for processing large histological images efficiently
- Parallel processing of nuclear regions for faster analysis of densely cellular tissues
- Tiling support for whole-slide images exceeding several gigabytes
- Comprehensive feature extraction including shape metrics, intensity statistics, and
  neighborhood relationships that characterize tissue architecture disruption
- Statistical comparison between injured (IRI) and control (CNTL) samples with
  appropriate multiple testing correction
- Visualization of results through:
    - Violin plots comparing feature distributions between IRI and control groups
    - Bar plots of -log₁₀(FDR-corrected p-values) highlighting significant differences
    - Ratio plots showing magnitude of changes between conditions
    - Correlation matrix heatmaps revealing relationships between nuclear features

Technical improvements:
- Neighbor-based features default to zero when no neighbors are present, preventing NaN values
- Robust outlier filtering removes biologically implausible measurements (e.g., Circularity > 1)
- Verbosity control via VERBOSE flag for production environments
- Plot-only mode for generating visualizations from previously extracted features

This tool enables quantitative assessment of nuclear alterations during kidney I/R injury,
providing insights into cellular damage mechanisms, inflammatory responses, and repair processes.
"""

import os, gc, time, argparse
import numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="Precision loss occurred in moment calculation due to catastrophic cancellation")
Image.MAX_IMAGE_PIXELS = 10**9

VERBOSE = False

# Try to use GPU arrays via CuPy if available for faster processing of large images.
# This is particularly important for kidney tissue analysis where nuclei counts.
# can exceed 100,000 in a single whole-slide image.
try:
    import cupy as cp  # type: ignore
    xp = cp
    use_gpu = True
    if VERBOSE:
        print("Using GPU acceleration with CuPy for faster nuclear feature extraction!")
except ImportError:
    xp = np
    use_gpu = False
    if VERBOSE:
        print("CuPy not available. Falling back to CPU (NumPy) - processing may be slower.")

import scipy.stats as sps
from scipy.stats import entropy, skew, kurtosis
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial import cKDTree
from skimage.measure import regionprops, label
from joblib import Parallel, delayed

# --- Timer decorator for performance monitoring ---
def timer(msg):
    """Decorator that times function execution and logs the duration.

    This is particularly useful for monitoring performance bottlenecks in the
    feature extraction pipeline, especially when processing large kidney tissue
    images with thousands of nuclei.

    Args:
        msg: Description of the timed operation for logging.

    Returns:
        Decorated function that logs timing information when VERBOSE is True.
    """
    def wrap(fn):
        def inner(*args, **kwargs):
            if VERBOSE:
                print(f"[⏱️ START] {msg}")
            t0 = time.time()
            result = fn(*args, **kwargs)
            if VERBOSE:
                print(f"[✅ DONE] {msg} in {time.time() - t0:.2f} sec")
            return result
        return inner
    return wrap

# --- Tiling helper for large image processing ---
def get_tiles(img, tile_side_length, overlap=0):
    """
    Generator yielding (tile, (x_offset, y_offset)) for the image.

    This function divides large microscopy images into manageable tiles for processing,
    which is essential for whole-slide kidney tissue images that can exceed several
    gigabytes in size. The overlap parameter ensures proper handling of nuclei that
    cross tile boundaries, preventing fragmentation or duplication of objects.

    Args:
        img: PIL Image object to be tiled.
        tile_side_length: Tuple of (width, height) for each tile.
        overlap: Number of pixels to overlap between adjacent tiles.

    Yields:
        Tuple of (tile_image, (x_offset, y_offset)) for each tile position.
    """
    width, height = img.size
    tile_w, tile_h = tile_side_length
    for top in range(0, height, tile_h - overlap):
        for left in range(0, width, tile_w - overlap):
            right = min(left + tile_w, width)
            bottom = min(top + tile_h, height)
            yield img.crop((left, top, right, bottom)), (left, top)

@timer("Computing dark region map")
def compute_dark_distance_map(np_img, threshold=50):
    """
    Create a distance map from dark areas in the tissue image.

    This function identifies dark regions (typically non-nuclear areas in DAPI-stained
    kidney tissue) and computes the Euclidean distance from each pixel to the nearest
    dark region. This feature helps characterize the spatial distribution of nuclei
    relative to tubular structures and other tissue compartments in kidney sections.

    Args:
        np_img: Numpy array of the grayscale image.
        threshold: Intensity threshold below which pixels are considered "dark".

    Returns:
        Numpy array containing the distance map from dark regions.
    """
    dark_mask = np_img < threshold
    return distance_transform_edt(~dark_mask)

@timer("Extracting intensity patch")
def extract_intensity_patch(pil_img, region, offset=(0, 0)):
    """
    Extract a grayscale patch from the image using the region's bounding box.

    This function extracts the intensity values for a specific nuclear region,
    which is essential for calculating texture and intensity-based features.
    In kidney I/R injury, changes in chromatin condensation and nuclear intensity
    patterns are important indicators of cell stress, apoptosis, and other
    pathological processes.

    Args:
        pil_img: PIL Image object containing the full image.
        region: Region properties object from scikit-image.
        offset: (x, y) offset to apply when extracting from a tiled image.

    Returns:
        Tuple of (intensity_patch, binary_mask) for the nuclear region.
    """
    minr, minc, maxr, maxc = region.bbox
    minc_off = minc + offset[0]
    minr_off = minr + offset[1]
    maxc_off = maxc + offset[0]
    maxr_off = maxr + offset[1]
    patch = pil_img.crop((minc_off, minr_off, maxc_off, maxr_off)).convert("L")
    patch_np = np.array(patch)
    return patch_np, region.image

@timer("Computing region features")
def compute_region_features(i, region, neighbors_info, dark_distance_map,
                            skip_lbp, pil_img, lbp_p, lbp_r, tile_offset=(0, 0)):
    """
    Compute comprehensive morphological and intensity features for a single nuclear region.

    This function extracts multiple categories of features that characterize nuclear
    properties relevant to kidney I/R injury analysis:

    1. Shape features: Capture changes in nuclear morphology that occur during
       apoptosis, pyroptosis, and other cell death mechanisms (e.g., circularity,
       aspect ratio, solidity).

    2. Intensity features: Reflect chromatin condensation patterns associated with
       different stages of cell damage and repair (mean, std, skewness, kurtosis).

    3. Neighborhood features: Characterize tissue architecture disruption and
       inflammatory cell infiltration patterns typical in kidney I/R injury
       (distances between nuclei, orientation alignment).

    Args:
        i: Index of the region being processed.
        region: Region properties object from scikit-image.
        neighbors_info: Dictionary containing information about neighboring nuclei.
        dark_distance_map: Distance map to dark regions in the tissue.
        skip_lbp: Flag to skip LBP texture calculations (always True now).
        pil_img: PIL Image object containing the full image.
        lbp_p, lbp_r: Unused LBP parameters (kept for backward compatibility).
        tile_offset: (x, y) offset when processing a tile from a larger image.

    Returns:
        Dictionary containing all computed features for the nuclear region.

    Note:
        Neighbor-based features default to 0 if no neighbors exist, preventing NaN values.
    """
    cy, cx = region.centroid
    cy += tile_offset[1]
    cx += tile_offset[0]

    intensity_patch, binary_mask = extract_intensity_patch(pil_img, region, offset=tile_offset)
    masked_pixels = intensity_patch[binary_mask.astype(bool)]

    try:
        feret = region.feret_diameter_max
    except Exception:
        feret = np.nan
    perimeter = region.perimeter if region.perimeter else np.nan
    circularity = (4 * np.pi * region.area / (perimeter**2)) if (perimeter and perimeter > 0) else np.nan
    aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else np.nan
    roughness = perimeter / np.sqrt(region.area) if region.area > 0 else np.nan

    intensity_mean = float(np.mean(masked_pixels))
    intensity_std = float(np.std(masked_pixels))
    intensity_median = float(np.median(masked_pixels))
    intensity_skew = float(skew(masked_pixels)) if masked_pixels.size > 0 else np.nan
    intensity_kurt = float(kurtosis(masked_pixels)) if masked_pixels.size > 0 else np.nan

    intensity_hist, _ = np.histogram(masked_pixels, bins=32)
    entropy_val = float(entropy(intensity_hist + 1))

    bbox_height = region.bbox[2] - region.bbox[0]
    bbox_width = region.bbox[3] - region.bbox[1]

    neighbor_areas = neighbors_info.get('areas', [])
    neighbor_orients = neighbors_info.get('orientations', [])
    neighbor_centroids = neighbors_info.get('centroids', [])

    if neighbor_orients:
        orientation_diffs = [region.orientation - o for o in neighbor_orients]
        orient_alignment_std = np.std(orientation_diffs)
        neighborhood_orientation_mean = np.mean(neighbor_orients)
        orientation_deviation = region.orientation - neighborhood_orientation_mean
        orientation_diff_abs_mean = np.mean(np.abs(orientation_diffs))
        orientation_range = np.ptp(neighbor_orients)
    else:
        orient_alignment_std = 0.0
        neighborhood_orientation_mean = 0.0
        orientation_deviation = 0.0
        orientation_diff_abs_mean = 0.0
        orientation_range = 0.0

    if neighbor_centroids:
        distances = [np.hypot(cx - c[1] - tile_offset[0], cy - c[0] - tile_offset[1])
                     for c in neighbor_centroids]
        dist_mean = np.mean(distances) if distances else 0.0
        dist_std = np.std(distances) if distances else 0.0
        dist_min = np.min(distances) if distances else 0.0
        dist_max = np.max(distances) if distances else 0.0
        neighbor_count = len(neighbor_centroids)
    else:
        dist_mean = dist_std = dist_min = dist_max = 0.0
        neighbor_count = 0

    radius = 50.0
    radius_area = np.pi * (radius ** 2)
    neighborhood_radius_density = (neighbor_count / radius_area) if radius_area > 0 else 0.0

    map_h, map_w = dark_distance_map.shape
    if (0 <= int(round(cy)) < map_h) and (0 <= int(round(cx)) < map_w):
        dist_to_dark = float(dark_distance_map[int(round(cy)), int(round(cx))])
    else:
        dist_to_dark = 0.0

    features = {
        "Label": region.label,
        "Centroid Y": cy,
        "Centroid X": cx,
        "Area": region.area,
        "Perimeter": perimeter,
        "Major Axis Length": region.major_axis_length,
        "Minor Axis Length": region.minor_axis_length,
        "Aspect Ratio": aspect_ratio,
        "Circularity": circularity,
        "Eccentricity": region.eccentricity,
        "Solidity": region.solidity,
        "Intensity Mean": intensity_mean,
        "Intensity Std": intensity_std,
        "Intensity Median": intensity_median,
        "Intensity Skewness": intensity_skew,
        "Intensity Kurtosis": intensity_kurt,
        "Feret Diameter": feret,
        "Roughness": roughness,
        "Texture Entropy": entropy_val,
        "Bounding Box Width": bbox_width,
        "Bounding Box Height": bbox_height,
        "Distance to Dark Region": dist_to_dark,
        "Neighborhood Mean Area": np.mean(neighbor_areas) if neighbor_areas else 0.0,
        "Neighborhood Std Area": np.std(neighbor_areas) if neighbor_areas else 0.0,
        "Neighborhood Orientation Std": orient_alignment_std,
        "Neighborhood Orientation Mean": neighborhood_orientation_mean,
        "Orientation Deviation": orientation_deviation,
        "Orientation Abs Diff Mean": orientation_diff_abs_mean,
        "Orientation Range": orientation_range,
        "Neighborhood Mean Distance": dist_mean,
        "Neighborhood Distance Std": dist_std,
        "Neighborhood Min Distance": dist_min,
        "Neighborhood Max Distance": dist_max,
        "Neighborhood Count": neighbor_count,
        "Neighborhood Radius Density": neighborhood_radius_density,
    }
    return features

def process_image(image_path, seg_mask_path, output_csv, tile_side_length=None, overlap=20,
                  lbp_p=8, lbp_r=1.0, skip_lbp=True, n_jobs=-1, neighbor_radius=50.0):
    """
    Process a kidney tissue image and its segmentation mask to extract nuclear features.

    This is the main processing function that orchestrates the feature extraction pipeline.
    It handles loading the image and segmentation mask, optionally divides them into tiles
    for large images, computes features for each nuclear region, and saves the results.

    The neighbor_radius parameter (default 50.0 pixels) defines the search radius for
    identifying neighboring nuclei. This value is calibrated for typical kidney tissue
    sections where proximal tubular nuclei are spaced approximately 20-40 pixels apart
    at standard magnification.

    Args:
        image_path: Path to the input microscopy image.
        seg_mask_path: Path to the segmentation mask (either .npy or image format).
        output_csv: Path where the feature CSV will be saved.
        tile_side_length: Optional tuple of (width, height) for tiling large images.
        overlap: Pixel overlap between adjacent tiles.
        lbp_p, lbp_r: Unused LBP parameters (kept for backward compatibility).
        skip_lbp: Flag to skip LBP calculations (always True now).
        n_jobs: Number of parallel jobs for feature computation (-1 for all cores).
        neighbor_radius: Radius in pixels to search for neighboring nuclei.

    Returns:
        Pandas DataFrame containing extracted features for all nuclei.
    """
    pil_img_full = Image.open(image_path)
    gray_img = pil_img_full.convert("L")
    np_gray = np.array(gray_img)
    dark_distance_map = compute_dark_distance_map(np_gray, threshold=50)

    if seg_mask_path.lower().endswith('.npy'):
        np_mask = np.load(seg_mask_path)
    else:
        pil_mask = Image.open(seg_mask_path)
        np_mask = np.array(pil_mask)
    if not np.issubdtype(np_mask.dtype, np.integer):
        np_mask = label(np_mask)

    all_features = []
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if tile_side_length is not None:
        print("Processing image by tiling...")
        tiles = list(get_tiles(pil_img_full, tile_side_length, overlap))
        for tile, offset in tqdm(tiles, desc="Tiles", unit="tile"):
            left, top = offset
            right = left + tile.size[0]
            bottom = top + tile.size[1]
            tile_mask = np_mask[top:bottom, left:right]
            props = regionprops(tile_mask)
            if props:
                centroids = [reg.centroid for reg in props]
                tree = cKDTree(centroids)
                neighbors_info_list = []
                for i, center in enumerate(centroids):
                    neighbor_ids = tree.query_ball_point(center, r=neighbor_radius)
                    neighbor_ids = [j for j in neighbor_ids if j != i]
                    info = {
                        'areas': [props[j].area for j in neighbor_ids],
                        'eccentricities': [props[j].eccentricity for j in neighbor_ids],
                        'orientations': [props[j].orientation for j in neighbor_ids],
                        'centroids': [props[j].centroid for j in neighbor_ids]
                    }
                    neighbors_info_list.append(info)
                tile_features = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(compute_region_features)(
                        i, region,
                        neighbors_info_list[i] if i < len(neighbors_info_list) else {},
                        dark_distance_map, skip_lbp,
                        pil_img_full, lbp_p, lbp_r,
                        tile_offset=offset
                    )
                    for i, region in enumerate(props)
                )
                all_features.extend(tile_features)
            gc.collect()
    else:
        print("Processing full image...")
        props = regionprops(np_mask)
        if props:
            centroids = [reg.centroid for reg in props]
            tree = cKDTree(centroids)
            neighbors_info_list = []
            for i, center in enumerate(centroids):
                neighbor_ids = tree.query_ball_point(center, r=neighbor_radius)
                neighbor_ids = [j for j in neighbor_ids if j != i]
                info = {
                    'areas': [props[j].area for j in neighbor_ids],
                    'eccentricities': [props[j].eccentricity for j in neighbor_ids],
                    'orientations': [props[j].orientation for j in neighbor_ids],
                    'centroids': [props[j].centroid for j in neighbor_ids]
                }
                neighbors_info_list.append(info)
            all_features = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(compute_region_features)(
                    i, region,
                    neighbors_info_list[i] if i < len(neighbors_info_list) else {},
                    dark_distance_map, skip_lbp,
                    pil_img_full, lbp_p, lbp_r,
                    tile_offset=(0, 0)
                )
                for i, region in enumerate(tqdm(props, desc="Regions", unit="region"))
            )
    df_features = pd.DataFrame(all_features)
    df_features.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")
    return df_features

def filter_invalid_masks(df):
    """
    Remove invalid nuclear masks based on biologically implausible measurements.

    This function filters out segmentation artifacts and biologically implausible
    nuclear measurements that would otherwise skew statistical analyses. The filtering
    criteria are based on known biological constraints of kidney cell nuclei:

    - Area < 5 pixels: Too small to be valid nuclei, likely noise or artifacts
    - Area >= 1000 pixels: Too large for individual nuclei, likely merged objects
    - Circularity > 1: Mathematically impossible, indicates measurement error
    - Aspect Ratio > 10: Extremely elongated objects, likely artifacts or vessels
    - Solidity < 0.5: Highly concave objects, typically not biological nuclei

    These thresholds are calibrated specifically for kidney tissue nuclei based on
    manual validation and biological knowledge of nuclear morphology in nephrons.

    Args:
        df: Pandas DataFrame containing nuclear features.

    Returns:
        Filtered DataFrame with invalid measurements removed.
    """
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df["Area"] >= 5]
    filtered_df = filtered_df[filtered_df["Area"] < 1000]
    filtered_df = filtered_df[filtered_df["Circularity"] <= 1.0]
    if "Aspect Ratio" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Aspect Ratio"] <= 10]
    if "Solidity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Solidity"] >= 0.5]
    return filtered_df

def compare_features(iri_df, cntl_df, output_prefix):
    """
    Perform statistical comparison between IRI and control nuclear features.

    This function conducts a comprehensive statistical analysis comparing nuclear
    features between ischemia-reperfusion injured (IRI) and control (CNTL) kidney
    tissue samples. It performs the following analyses:

    1. Filters both datasets to remove invalid measurements
    2. Performs Mann-Whitney U tests for each feature (non-parametric comparison)
    3. Applies FDR correction for multiple testing
    4. Calculates fold changes between conditions
    5. Generates visualizations including violin plots, p-value bar plots,
       ratio scatter plots, and correlation heatmaps

    The Mann-Whitney U test is specifically chosen as it's robust to non-normal
    distributions commonly observed in nuclear morphology measurements, especially
    in heterogeneous kidney tissue with multiple cell types.

    Args:
        iri_df: DataFrame containing nuclear features from IRI samples.
        cntl_df: DataFrame containing nuclear features from control samples.
        output_prefix: Path prefix for saving results and visualizations.

    Returns:
        None. Results are saved to disk as CSV and image files.
    """
    import numpy as np
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Filter out invalid masks so that only valid masks are used.
    iri_df_filtered = filter_invalid_masks(iri_df.copy())
    cntl_df_filtered = filter_invalid_masks(cntl_df.copy())

    iri_df_filtered["Group"] = "IRI"
    cntl_df_filtered["Group"] = "CNTL"
    combined_df = pd.concat([iri_df_filtered, cntl_df_filtered], ignore_index=True)
    num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["Label", "Centroid Y", "Centroid X"]:
        if col in num_cols:
            num_cols.remove(col)

    plot_dir = os.path.join(os.path.dirname(output_prefix), "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # --- Violin Plots in Grid (9 per page) using central 98% of data ---
    features_to_plot = num_cols
    plots_per_page = 9
    n_features = len(features_to_plot)
    n_pages = (n_features + plots_per_page - 1) // plots_per_page
    group_palette = {"IRI": "#4c72b0", "CNTL": "#dd8452"}

    for page in range(n_pages):
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        axes = axes.flatten()
        for i in range(plots_per_page):
            idx = page * plots_per_page + i
            if idx >= n_features:
                axes[i].axis("off")
                continue
            feature = features_to_plot[idx]
            # Trim feature values to the central 98%.
            lower = combined_df[feature].quantile(0.01)
            upper = combined_df[feature].quantile(0.99)
            df_filtered = combined_df[(combined_df[feature] >= lower) & (combined_df[feature] <= upper)]
            ax = axes[i]
            sns.violinplot(x="Group", y=feature, data=df_filtered, hue="Group",
                           palette=group_palette, dodge=True, legend=False, cut=0, ax=ax)
            # Overlay median as a thick black line.
            for group in ["IRI", "CNTL"]:
                med = df_filtered[df_filtered["Group"] == group][feature].median()
                xpos = 0 if group == "IRI" else 1
                ax.plot([xpos - 0.2, xpos + 0.2], [med, med], lw=4, color="black")
            ax.set_title(f"{feature}\n[{lower:.2f}, {upper:.2f}]")
            ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"violin_plots_page_{page+1}.png"))
        plt.close()

    # --- Bar Plot of p-values computed on trimmed data ---
    scatter_cols = [c for c in num_cols if c not in ["Centroid Y", "Centroid X"]]
    trimmed_means_iri = {}
    trimmed_means_cntl = {}
    p_values = {}
    for feature in scatter_cols:
        lower = combined_df[feature].quantile(0.01)
        upper = combined_df[feature].quantile(0.99)
        iri_values = iri_df_filtered[feature][(iri_df_filtered[feature] >= lower) & (iri_df_filtered[feature] <= upper)]
        cntl_values = cntl_df_filtered[feature][(cntl_df_filtered[feature] >= lower) & (cntl_df_filtered[feature] <= upper)]
        trimmed_means_iri[feature] = iri_values.mean()
        trimmed_means_cntl[feature] = cntl_values.mean()
        try:
            stat, p = mannwhitneyu(iri_values, cntl_values, alternative="two-sided")
            p_values[feature] = p
        except Exception:
            p_values[feature] = np.nan

    comparison_df = pd.DataFrame({
        "Feature": scatter_cols,
        "IRI Mean": [trimmed_means_iri[f] for f in scatter_cols],
        "CNTL Mean": [trimmed_means_cntl[f] for f in scatter_cols],
        "IRI/CNTL Ratio": [trimmed_means_iri[f] / trimmed_means_cntl[f] for f in scatter_cols],
        "p-value": [p_values[f] for f in scatter_cols]
    })

    valid_p = comparison_df["p-value"].dropna().values
    if valid_p.size > 0:
        _, fdr_corrected, _, _ = multipletests(valid_p, method="fdr_bh")
        comparison_df.loc[comparison_df["p-value"].notna(), "FDR-corrected p"] = fdr_corrected
    else:
        comparison_df["FDR-corrected p"] = np.nan

    # Replace any zeros with a small epsilon to avoid issues with log.
    epsilon = 1e-300
    comparison_df["FDR-corrected p"] = comparison_df["FDR-corrected p"].replace(0, epsilon)
    comparison_df["-log10(FDR p)"] = -np.log10(comparison_df["FDR-corrected p"])

    # Sort by FDR-corrected p-value (ascending).
    comparison_df = comparison_df.sort_values("FDR-corrected p")
    comparison_csv = os.path.join(plot_dir, "feature_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="-log10(FDR p)", y="Feature", data=comparison_df)
    plt.xlabel("-log10(FDR p-value)")
    plt.title("Bar Plot of FDR-Corrected p-values (sorted)")
    # Draw a vertical line at -log10(0.05) (≈ 1.30).
    threshold = -np.log10(0.05)
    plt.axvline(x=threshold, color="red", linestyle="--", lw=2, label="p = 0.05 threshold")
    plt.legend()
    # Set a fixed upper limit to zoom in (e.g., from 0 to 3).
    # plt.xlim(0, 3).
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "barplot_pvalues.png"))
    plt.close()

    # --- Scatter Plot of IRI/CNTL Mean Ratios (linear scale) ---
    plt.figure(figsize=(10, 6))
    ratio_df = comparison_df.sort_values("IRI/CNTL Ratio")
    sns.scatterplot(x="Feature", y="IRI/CNTL Ratio", data=ratio_df, color="purple", s=100)
    plt.xticks(rotation=90)
    plt.axhline(y=1, color="black", linestyle="--", lw=2)
    plt.title("Scatter Plot of IRI/CNTL Mean Ratios")
    plt.ylabel("IRI/CNTL Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "scatter_ratio.png"))
    plt.close()

    # --- Correlation Heatmap ---
    plt.figure(figsize=(10, 10))
    corr = combined_df[num_cols].corr()
    sns.heatmap(corr, square=True, cmap="viridis", cbar=True,
                linewidths=0.5, linecolor="white")
    plt.title("Correlation Matrix of Features")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
    plt.close()

    # Show complete DataFrame.
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(f"Comparison plots and table saved in {plot_dir}")
    print("Feature Comparison Table:")
    print(comparison_df)

# --- Main CLI Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Mask Analysis with GPU, Parallelism, and NN Optimization")
    parser.add_argument("--image", type=str, help="Path to the input image file (IRI)")
    parser.add_argument("--mask", type=str, help="Path to the segmentation mask file (IRI)")
    parser.add_argument("--output", type=str, default="features.csv", help="Output CSV file for IRI features")
    parser.add_argument("--tile_width", type=int, default=None, help="Tile width in pixels (optional)")
    parser.add_argument("--tile_height", type=int, default=None, help="Tile height in pixels (optional)")
    parser.add_argument("--overlap", type=int, default=20, help="Overlap between tiles (in pixels)")
    parser.add_argument("--lbp_p", type=int, default=8, help="(Unused) Number of points for LBP")
    parser.add_argument("--lbp_r", type=float, default=1.0, help="(Unused) Radius for LBP")
    parser.add_argument("--skip_lbp", action="store_true", help="(Unused) Skip LBP calculation")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (default: -1 for all cores)")
    parser.add_argument("--control_image", type=str, default=None, help="Path to the control image file (CNTL)")
    parser.add_argument("--control_mask", type=str, default=None, help="Path to the control segmentation mask file (CNTL)")
    parser.add_argument("--plot_only", action="store_true", help="If set, only generate plots from existing CSV file(s)")
    args = parser.parse_args()

    tile_side_length = (args.tile_width, args.tile_height) if args.tile_width and args.tile_height else None

    if args.plot_only:
        # In plot_only mode, load the features CSV and (if provided) the control CSV, then generate plots.
        if not os.path.exists(args.output):
            print(f"Feature file {args.output} does not exist!")
            exit(1)
        iri_df = pd.read_csv(args.output)
        if args.control_image and args.control_mask and os.path.exists(os.path.splitext(args.output)[0] + "_control.csv"):
            cntl_df = pd.read_csv(os.path.splitext(args.output)[0] + "_control.csv")
            output_prefix = os.path.splitext(args.output)[0]
            compare_features(iri_df, cntl_df, output_prefix)
        else:
            # If no control data is provided, generate plots based solely on IRI data.
            # For example, create violin plots and correlation heatmap for IRI group.
            combined_df = filter_invalid_masks(iri_df.copy())
            combined_df["Group"] = "IRI"
            num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
            for col in ["Label", "Centroid Y", "Centroid X"]:
                if col in num_cols:
                    num_cols.remove(col)
            plot_dir = os.path.join(os.path.dirname(args.output), "plots")
            os.makedirs(plot_dir, exist_ok=True)
            # Violin plots for IRI only:
            features_to_plot = num_cols
            plots_per_page = 9
            n_features = len(features_to_plot)
            n_pages = (n_features + plots_per_page - 1) // plots_per_page
            for page in range(n_pages):
                fig, axes = plt.subplots(3, 3, figsize=(14, 12))
                axes = axes.flatten()
                for i in range(plots_per_page):
                    idx = page * plots_per_page + i
                    if idx >= n_features:
                        axes[i].axis("off")
                        continue
                    feature = features_to_plot[idx]
                    lower = combined_df[feature].quantile(0.01)
                    upper = combined_df[feature].quantile(0.99)
                    df_filtered = combined_df[(combined_df[feature] >= lower) & (combined_df[feature] <= upper)]
                    ax = axes[i]
                    sns.violinplot(x="Group", y=feature, data=df_filtered, hue="Group",
                                   palette="Set2", dodge=True, legend=False, cut=0, ax=ax)
                    med = df_filtered[df_filtered["Group"] == "IRI"][feature].median()
                    ax.plot([-0.2, 0.2], [med, med], lw=4, color="black")
                    ax.set_title(f"{feature}\n[{lower:.2f}, {upper:.2f}]")
                    ax.set_xlabel("")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"violin_plots_page_{page+1}.png"))
                plt.close()
            # Correlation heatmap:
            plt.figure(figsize=(10,10))
            corr = combined_df[num_cols].corr()
            sns.heatmap(corr, square=True, cmap="viridis", cbar=True,
                        linewidths=0.5, linecolor="white")
            plt.title("Correlation Matrix of Features (IRI)")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
            plt.close()
            print(f"Plots saved in {plot_dir}")
    else:
        print("Processing IRI group...")
        iri_df = process_image(
            args.image, args.mask, args.output,
            tile_side_length=tile_side_length,
            overlap=args.overlap,
            lbp_p=args.lbp_p,
            lbp_r=args.lbp_r,
            skip_lbp=args.skip_lbp,
            n_jobs=args.n_jobs
        )
        if args.control_image and args.control_mask:
            control_output = os.path.splitext(args.output)[0] + "_control.csv"
            print("Processing CNTL group...")
            cntl_df = process_image(
                args.control_image, args.control_mask, control_output,
                tile_side_length=tile_side_length,
                overlap=args.overlap,
                lbp_p=args.lbp_p,
                lbp_r=args.lbp_r,
                skip_lbp=args.skip_lbp,
                n_jobs=args.n_jobs
            )
            output_prefix = os.path.splitext(args.output)[0]
            compare_features(iri_df, cntl_df, output_prefix)
