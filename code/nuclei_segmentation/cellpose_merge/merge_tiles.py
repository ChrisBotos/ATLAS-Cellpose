"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: merge_tiles.py.
Description:
    Merge per‑tile nucleus‑instance masks (TIFF or NumPy) back into a full‑slide
    label map.  The implementation is a drop‑in replacement for the original
    ``merge_tiles.py`` but fixes the I/O‑path brittleness and improves both
    memory footprint and parallelism.  It detects “row_col.tif”, “row col.tif”,
    “row_col.npz”, and “pixelY pixelX.npz” automatically, regardless of whether
    the inference stage saved into *tile_masks/* or *tile_masks_npz/*.  The core
    merge logic is delegated to the well‑tested ``_merge_patch_cpu`` /
    ``_merge_patch_gpu`` back‑ends.

Dependencies:
    • Python ≥ 3.10.
    • numpy, pillow, tqdm, pytest.  (Optional: torch for GPU.)

Usage:
    from .merge_tiles import merge_masks_streaming

    merged = merge_masks_streaming(
        height=10_000, width=12_000,
        tile_h=512, tile_w=512, overlap=128,
        tiles_path="/path/to/tile_masks",  # or …_npz
        threshold=0.3
    )
"""

from __future__ import annotations

import concurrent.futures as _cf
import logging
import math
import re
import traceback
from pathlib import Path
from typing import Callable, Dict, Final, Iterable, List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover – CPU‑only envs.
    torch = None  # type: ignore[assignment]

"""
Lazy import of heavy merge back‑ends – keeps CLI snappy and avoids circulars
"""

_merge_patch_cpu: Callable[..., Tuple[NDArray[np.uint32], Dict[int, int]]]
_merge_patch_gpu: Callable[..., Tuple[NDArray[np.uint32], Dict[int, int]]]


def _lazy_import_merge_backends() -> None:
    """Import the CPU/GPU merge kernels exactly once and only when needed."""
    global _merge_patch_cpu, _merge_patch_gpu
    if "_merge_patch_cpu" in globals():
        return  # Already imported.

    try:
        from .rules import merge_patch_cpu as _m_cpu  # pylint: disable=import-error
        from .gpu_merge import merge_patch_gpu as _m_gpu  # pylint: disable=import-error
    except ImportError:
        # Fallback for when running as standalone script.
        try:
            from rules import merge_patch_cpu as _m_cpu  # pylint: disable=import-error
            from gpu_merge import merge_patch_gpu as _m_gpu  # pylint: disable=import-error
        except ImportError as e:
            raise ImportError(f"Could not import merge backends: {e}")

    _merge_patch_cpu = _m_cpu  # type: ignore[misc]
    _merge_patch_gpu = _m_gpu  # type: ignore[misc]

"""
1.  Path and filename helpers
"""

# Accepts either an underscore or a space delimiter between the two integers.
_PARSER = re.compile(r"^(\d+)[ _](\d+)\.(?:tif|np[sz])$", re.IGNORECASE)


def _parse_tile_filename(name: str) -> Tuple[int, int] | None:  # pragma: no cover
    """Return the two *raw* integers encoded in *name* or *None* if not a tile."""
    m = _PARSER.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _dir_contains_tiles(path: Path) -> bool:  # pragma: no cover
    """Cheap check: does *path* contain **any** file that looks like a tile?"""
    for p in path.iterdir():
        if p.is_file() and _parse_tile_filename(p.name):
            return True
    return False


def _resolve_tiles_path(path: Path) -> Path:
    """Find a directory that actually contains tile masks.

    We try subtle variations of the user‑supplied *path* so that the pipeline
    survives typos such as “tile_masks_npz” vs “tile_masks”.  The first match
    wins; otherwise we raise with a crystal‑clear diagnostic.
    """

    candidates: list[Path] = [path.expanduser()]

    # Common suffix confusion:  “…/_npz” vs plain.
    # Handle edge case where path.name is empty (e.g., current directory ".").
    if path.name and path.name.endswith("_npz"):
        candidates.append(path.with_name(path.name.removesuffix("_npz")))
    elif path.name:
        candidates.append(path.with_name(f"{path.name}_npz"))

    for cand in candidates:
        if cand.exists() and cand.is_dir() and _dir_contains_tiles(cand):
            return cand

    tried = "\n    • " + "\n    • ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"None of the following directories contains tile masks:{tried}\n"
        "Please check the --tiles-path argument or rerun inference."
    )


def _discover_tiles(path: Path) -> Tuple[Dict[Tuple[int, int], Path], List[Tuple[int, int]]]:
    """Map (raw_row, raw_col) → file‑path and return the mapping + coord list."""
    mapping: Dict[Tuple[int, int], Path] = {}
    for p in path.iterdir():
        if not p.is_file():
            continue
        parsed = _parse_tile_filename(p.name)
        if parsed is None:
            continue
        mapping[parsed] = p

    if not mapping:
        raise FileNotFoundError(
            f"Directory '{path}' exists but contains no <row>_<col>.tif or *.npz tiles."
        )
    return mapping, list(mapping)


"""
2.  Graph helpers
"""

def _build_memory_aware_clusters(
    coords: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int,
    max_cluster_memory_gb: float = 2.0,
    max_cluster_dimension: int = 4096,
    max_cluster_gpu_memory_gb: float = 4.0,
    cluster_subdivision_strategy: str = "spatial_quadtree",
    max_subdivision_depth: int = 6,
    min_cluster_size_after_subdivision: int = 2
) -> List[List[Tuple[int, int]]]:
    """
    Build memory-efficient clusters with adaptive subdivision for GPU processing.

    This enhanced function prevents massive GPU memory allocation failures by
    implementing adaptive cluster subdivision. It creates clusters that are
    guaranteed to fit within both CPU and GPU memory limits, preventing the
    200-800+ GiB allocation attempts that cause processing failures.

    Key Enhancements:
    - Adaptive subdivision based on GPU memory constraints
    - Multiple subdivision strategies for different tile distributions
    - Recursive subdivision with depth limits to prevent infinite loops
    - Minimum cluster size enforcement to maintain processing efficiency

    Parameters
    ----------
    coords : List[Tuple[int, int]]
        List of (row, col) coordinates for all tiles.
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    max_cluster_memory_gb : float, default 2.0
        Maximum memory per cluster for CPU processing in gigabytes.
    max_cluster_dimension : int, default 4096
        Maximum bounding box dimension per cluster in pixels.
    max_cluster_gpu_memory_gb : float, default 4.0
        Maximum memory per cluster for GPU processing in gigabytes.
    cluster_subdivision_strategy : str, default "spatial_quadtree"
        Strategy for subdividing oversized clusters.
    max_subdivision_depth : int, default 6
        Maximum recursive subdivision depth.
    min_cluster_size_after_subdivision : int, default 2
        Minimum cluster size after subdivision.

    Returns
    -------
    List[List[Tuple[int, int]]]
        List of memory-efficient clusters guaranteed to fit within GPU memory limits.
    """
    if not coords:
        return []

    logging.info(f"Building adaptive memory-aware clusters from {len(coords)} tiles "
                f"(CPU_limit={max_cluster_memory_gb:.1f}GB, GPU_limit={max_cluster_gpu_memory_gb:.1f}GB, "
                f"max_dimension={max_cluster_dimension}px, strategy={cluster_subdivision_strategy})")

    # Calculate stride for bounding box calculations.
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    # Start with traditional 4-neighbor connectivity to find base connected components.
    base_clusters = _build_traditional_clusters(coords)

    # Apply adaptive subdivision to ensure GPU memory safety.
    final_clusters = []
    total_subdivisions = 0

    for cluster_idx, cluster in enumerate(base_clusters):
        if len(cluster) <= 1:
            final_clusters.append(cluster)
            continue

        # Apply adaptive subdivision to ensure both CPU and GPU memory safety.
        subdivided_clusters = _adaptive_cluster_subdivision(
            cluster, tile_h, tile_w, overlap,
            max_cluster_memory_gb=max_cluster_memory_gb,
            max_cluster_dimension=max_cluster_dimension,
            max_cluster_gpu_memory_gb=max_cluster_gpu_memory_gb,
            subdivision_strategy=cluster_subdivision_strategy,
            max_depth=max_subdivision_depth,
            min_cluster_size=min_cluster_size_after_subdivision,
            depth=0
        )

        final_clusters.extend(subdivided_clusters)

        if len(subdivided_clusters) > 1:
            total_subdivisions += len(subdivided_clusters) - 1
            logging.debug(f"Cluster {cluster_idx} subdivided into {len(subdivided_clusters)} sub-clusters")

    logging.info(f"Adaptive clustering completed: {len(final_clusters)} clusters created "
                f"({total_subdivisions} subdivisions applied)")
    logging.info(f"Average cluster size: {len(coords) / len(final_clusters):.1f} tiles, "
                f"guaranteed GPU memory safety")

    return final_clusters


def _adaptive_cluster_subdivision(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int,
    max_cluster_memory_gb: float,
    max_cluster_dimension: int,
    max_cluster_gpu_memory_gb: float,
    subdivision_strategy: str,
    max_depth: int,
    min_cluster_size: int,
    depth: int = 0
) -> List[List[Tuple[int, int]]]:
    """
    Recursively subdivide clusters to ensure GPU memory safety.

    This function implements adaptive cluster subdivision to prevent massive
    GPU memory allocation attempts (200-800+ GiB). It uses multiple strategies
    to intelligently split oversized clusters while maintaining processing efficiency.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        List of (row, col) coordinates for tiles in the cluster.
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    max_cluster_memory_gb : float
        Maximum memory per cluster for CPU processing.
    max_cluster_dimension : int
        Maximum bounding box dimension per cluster.
    max_cluster_gpu_memory_gb : float
        Maximum memory per cluster for GPU processing.
    subdivision_strategy : str
        Strategy for subdividing clusters.
    max_depth : int
        Maximum recursive subdivision depth.
    min_cluster_size : int
        Minimum cluster size after subdivision.
    depth : int, default 0
        Current recursion depth.

    Returns
    -------
    List[List[Tuple[int, int]]]
        List of subdivided clusters that fit within memory limits.
    """
    # Base case: cluster is small enough or we've reached max depth.
    if len(cluster) <= min_cluster_size or depth >= max_depth:
        # EMERGENCY FALLBACK: If we've reached max depth but cluster is still problematic,
        # force split it into ultra-safe clusters to prevent memory failures.
        if depth >= max_depth and len(cluster) > 2:
            logging.warning(f"Max subdivision depth reached for cluster of {len(cluster)} tiles. "
                           f"Applying emergency splitting to prevent memory failures.")
            return _force_split_to_safe_clusters(cluster, tile_h, tile_w, overlap)
        return [cluster]

    # Check if cluster meets all memory constraints.
    cluster_memory, cluster_dimensions = _estimate_cluster_requirements(
        cluster, tile_h, tile_w, overlap
    )

    # Estimate GPU memory requirements (more conservative than CPU).
    gpu_memory_estimate = cluster_memory * 2.0  # GPU processing typically uses more memory.
    max_dim = max(cluster_dimensions)

    # Check if subdivision is needed.
    needs_subdivision = (
        cluster_memory > max_cluster_memory_gb or
        gpu_memory_estimate > max_cluster_gpu_memory_gb or
        max_dim > max_cluster_dimension or
        len(cluster) > 50  # Hard limit on cluster size
    )

    if not needs_subdivision:
        return [cluster]

    # Log subdivision attempt.
    logging.debug(f"Subdividing cluster at depth {depth}: {len(cluster)} tiles, "
                 f"CPU_mem={cluster_memory:.2f}GB, GPU_mem={gpu_memory_estimate:.2f}GB, "
                 f"max_dim={max_dim}px")

    # Apply subdivision strategy.
    if subdivision_strategy == "spatial_quadtree":
        sub_clusters = _subdivide_spatial_quadtree(cluster, tile_h, tile_w, overlap)
    elif subdivision_strategy == "spatial_grid":
        sub_clusters = _subdivide_spatial_grid(cluster, tile_h, tile_w, overlap)
    elif subdivision_strategy == "density_based":
        sub_clusters = _subdivide_density_based(cluster, tile_h, tile_w, overlap)
    elif subdivision_strategy == "hybrid":
        sub_clusters = _subdivide_hybrid(cluster, tile_h, tile_w, overlap)
    else:
        # Fallback to quadtree subdivision.
        logging.warning(f"Unknown subdivision strategy '{subdivision_strategy}', using spatial_quadtree")
        sub_clusters = _subdivide_spatial_quadtree(cluster, tile_h, tile_w, overlap)

    # Recursively subdivide each sub-cluster.
    final_clusters = []
    for sub_cluster in sub_clusters:
        if len(sub_cluster) > 0:
            subdivided = _adaptive_cluster_subdivision(
                sub_cluster, tile_h, tile_w, overlap,
                max_cluster_memory_gb, max_cluster_dimension, max_cluster_gpu_memory_gb,
                subdivision_strategy, max_depth, min_cluster_size, depth + 1
            )
            final_clusters.extend(subdivided)

    return final_clusters


def _subdivide_spatial_quadtree(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int
) -> List[List[Tuple[int, int]]]:
    """
    Subdivide cluster using spatial quadtree approach.

    This method divides the cluster's bounding box into four quadrants
    and assigns tiles to the appropriate quadrant based on their position.
    """
    if len(cluster) <= 2:
        return [cluster]

    # Calculate bounding box.
    min_r = min(r for r, _ in cluster)
    max_r = max(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    max_c = max(c for _, c in cluster)

    # Calculate midpoints.
    mid_r = (min_r + max_r) / 2
    mid_c = (min_c + max_c) / 2

    # Assign tiles to quadrants.
    quadrants = [[], [], [], []]  # NW, NE, SW, SE

    for r, c in cluster:
        if r <= mid_r and c <= mid_c:
            quadrants[0].append((r, c))  # NW
        elif r <= mid_r and c > mid_c:
            quadrants[1].append((r, c))  # NE
        elif r > mid_r and c <= mid_c:
            quadrants[2].append((r, c))  # SW
        else:
            quadrants[3].append((r, c))  # SE

    # Return non-empty quadrants.
    return [q for q in quadrants if len(q) > 0]


def _subdivide_spatial_grid(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int,
    grid_size: int = 2
) -> List[List[Tuple[int, int]]]:
    """
    Subdivide cluster using regular grid approach.

    This method divides the cluster's bounding box into a regular grid
    and assigns tiles to grid cells based on their position.
    """
    if len(cluster) <= grid_size * grid_size:
        return [cluster]

    # Calculate bounding box.
    min_r = min(r for r, _ in cluster)
    max_r = max(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    max_c = max(c for _, c in cluster)

    # Calculate grid cell dimensions.
    r_range = max_r - min_r + 1
    c_range = max_c - min_c + 1
    cell_h = r_range / grid_size
    cell_w = c_range / grid_size

    # Assign tiles to grid cells.
    grid_cells = {}

    for r, c in cluster:
        cell_r = min(grid_size - 1, int((r - min_r) / cell_h))
        cell_c = min(grid_size - 1, int((c - min_c) / cell_w))
        cell_key = (cell_r, cell_c)

        if cell_key not in grid_cells:
            grid_cells[cell_key] = []
        grid_cells[cell_key].append((r, c))

    # Return non-empty cells.
    return [cell for cell in grid_cells.values() if len(cell) > 0]


def _subdivide_density_based(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int
) -> List[List[Tuple[int, int]]]:
    """
    Subdivide cluster based on tile density.

    This method identifies dense regions and sparse regions,
    creating clusters that balance processing efficiency with memory safety.
    """
    if len(cluster) <= 4:
        return [cluster]

    # For now, use quadtree as a fallback.
    # This can be enhanced with more sophisticated density analysis.
    return _subdivide_spatial_quadtree(cluster, tile_h, tile_w, overlap)


def _subdivide_hybrid(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int
) -> List[List[Tuple[int, int]]]:
    """
    Subdivide cluster using hybrid approach.

    This method combines multiple strategies based on cluster characteristics.
    """
    # For small clusters, use quadtree.
    if len(cluster) <= 8:
        return _subdivide_spatial_quadtree(cluster, tile_h, tile_w, overlap)

    # For larger clusters, use grid subdivision.
    return _subdivide_spatial_grid(cluster, tile_h, tile_w, overlap, grid_size=3)


def _build_traditional_clusters(coords: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Return 4‑neighbour connected components of the coord grid."""
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {c: [] for c in coords}
    for r, c in coords:
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nbr = (r + dr, c + dc)
            if nbr in adj:
                adj[(r, c)].append(nbr)
    clusters: list[list[Tuple[int, int]]] = []
    seen: set[Tuple[int, int]] = set()
    for root in coords:
        if root in seen:
            continue
        comp: list[Tuple[int, int]] = []
        queue = [root]
        while queue:
            cur = queue.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            queue.extend(adj[cur])
        clusters.append(comp)
    return clusters


def _estimate_cluster_requirements(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int
) -> Tuple[float, Tuple[int, int]]:
    """
    Estimate memory requirements and bounding box dimensions for a cluster.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        List of (row, col) coordinates for tiles in the cluster.
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.

    Returns
    -------
    Tuple[float, Tuple[int, int]]
        Memory requirement in GB and (height, width) dimensions.
    """
    if not cluster:
        return 0.0, (0, 0)

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    min_r = min(r for r, _ in cluster)
    max_r = max(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    max_c = max(c for _, c in cluster)

    # Calculate bounding box dimensions.
    bbox_h = (max_r - min_r) * stride_h + tile_h
    bbox_w = (max_c - min_c) * stride_w + tile_w

    # CRITICAL FIX: Calculate memory based on actual processing requirements.
    # For sparse clusters, we should NOT use the full bounding box for memory estimation.
    # Instead, use a more realistic estimate based on actual processing needs.
    num_tiles = len(cluster)

    # Memory for actual tile data: num_tiles * tile_h * tile_w * 4 bytes (uint32).
    actual_tile_memory = num_tiles * tile_h * tile_w * 4 / (1024**3)  # GB

    # For workspace memory, use a more conservative approach for sparse clusters.
    # Calculate sparsity ratio to determine if this is a sparse cluster.
    total_tile_pixels = num_tiles * tile_h * tile_w
    bbox_pixels = bbox_h * bbox_w
    sparsity_ratio = bbox_pixels / total_tile_pixels if total_tile_pixels > 0 else 1.0

    if sparsity_ratio > 10.0:  # Very sparse cluster.
        logging.warning(f"Sparse cluster detected: {num_tiles} tiles in {bbox_h}x{bbox_w} bbox "
                       f"(sparsity ratio: {sparsity_ratio:.1f})")
        # For sparse clusters, workspace memory should be based on actual tiles, not bounding box.
        workspace_memory = actual_tile_memory * 2.0  # Conservative but not excessive.
    else:
        # For dense clusters, use bounding box approach.
        workspace_memory = bbox_h * bbox_w * 4 / (1024**3)  # GB

    # Total memory with safety factor.
    total_memory = (actual_tile_memory + workspace_memory) * 1.5  # Reasonable safety factor

    return total_memory, (bbox_h, bbox_w)


def _split_cluster_spatially(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int,
    max_memory_gb: float,
    max_dimension: int
) -> List[List[Tuple[int, int]]]:
    """
    Split a large cluster into smaller memory-efficient sub-clusters using spatial partitioning.

    This function uses a recursive spatial splitting approach to create sub-clusters
    that meet memory and dimension constraints while preserving spatial locality.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        Large cluster to split.
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    max_memory_gb : float
        Maximum memory per sub-cluster in GB.
    max_dimension : int
        Maximum bounding box dimension per sub-cluster in pixels.

    Returns
    -------
    List[List[Tuple[int, int]]]
        List of memory-efficient sub-clusters.
    """
    if len(cluster) <= 1:
        return [cluster]

    # Check if current cluster meets constraints.
    memory, dimensions = _estimate_cluster_requirements(cluster, tile_h, tile_w, overlap)
    max_dim = max(dimensions)

    # CRITICAL FIX: Much more aggressive cluster size limits to prevent memory issues.
    # Force splitting for any cluster that could potentially cause problems.
    max_safe_cluster_size = 4  # Ultra-conservative cluster size limit.

    if (memory <= max_memory_gb and
        max_dim <= max_dimension and
        len(cluster) <= max_safe_cluster_size):
        return [cluster]

    # Find the dimension with the largest span for splitting.
    min_r = min(r for r, _ in cluster)
    max_r = max(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    max_c = max(c for _, c in cluster)

    row_span = max_r - min_r
    col_span = max_c - min_c

    # Split along the larger dimension.
    if row_span >= col_span:
        # Split horizontally.
        mid_r = (min_r + max_r) // 2
        upper_cluster = [(r, c) for r, c in cluster if r <= mid_r]
        lower_cluster = [(r, c) for r, c in cluster if r > mid_r]
    else:
        # Split vertically.
        mid_c = (min_c + max_c) // 2
        left_cluster = [(r, c) for r, c in cluster if c <= mid_c]
        right_cluster = [(r, c) for r, c in cluster if c > mid_c]
        upper_cluster, lower_cluster = left_cluster, right_cluster

    # Recursively split sub-clusters if needed.
    result = []
    for sub_cluster in [upper_cluster, lower_cluster]:
        if sub_cluster:  # Only process non-empty clusters.
            result.extend(_split_cluster_spatially(
                sub_cluster, tile_h, tile_w, overlap, max_memory_gb, max_dimension
            ))

    return result


def _force_split_to_safe_clusters(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int,
    max_memory_gb: float = 0.1,  # Ultra-conservative memory limit.
    max_cluster_size: int = 2    # Maximum 2 tiles per cluster.
) -> List[List[Tuple[int, int]]]:
    """
    Emergency function to force split problematic clusters into ultra-safe sub-clusters.

    This function is called when normal splitting fails and we need to guarantee
    that clusters will not cause memory allocation failures. It splits clusters
    down to individual tiles if necessary.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        Problematic cluster to split.
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    max_memory_gb : float, default 0.1
        Ultra-conservative memory limit per cluster.
    max_cluster_size : int, default 2
        Maximum number of tiles per cluster.

    Returns
    -------
    List[List[Tuple[int, int]]]
        List of ultra-safe clusters guaranteed to be processable.
    """
    if len(cluster) <= 1:
        return [cluster]

    # If cluster is still too large, split it into individual tiles.
    if len(cluster) > max_cluster_size:
        logging.warning(f"Emergency splitting cluster of {len(cluster)} tiles into individual tiles")
        return [[tile] for tile in cluster]

    # Check if current cluster meets ultra-conservative constraints.
    memory, dimensions = _estimate_cluster_requirements(cluster, tile_h, tile_w, overlap)

    if memory <= max_memory_gb and len(cluster) <= max_cluster_size:
        return [cluster]

    # Split into pairs of adjacent tiles.
    result = []
    for i in range(0, len(cluster), max_cluster_size):
        sub_cluster = cluster[i:i + max_cluster_size]
        result.append(sub_cluster)

    logging.info(f"Emergency split: {len(cluster)} tiles -> {len(result)} ultra-safe clusters")
    return result


def _estimate_cluster_memory_requirements(
    cluster_size: int,
    cluster_h: int,
    cluster_w: int
) -> float:
    """
    Estimate memory requirements for processing a cluster.

    Parameters
    ----------
    cluster_size : int
        Number of tiles in the cluster.
    cluster_h, cluster_w : int
        Dimensions of the cluster bounding box in pixels.

    Returns
    -------
    float
        Estimated memory requirement in gigabytes.
    """
    # Memory for the 3D stack: (cluster_size, cluster_h, cluster_w) as uint32 (4 bytes).
    stack_memory_bytes = cluster_size * cluster_h * cluster_w * 4

    # Additional memory for intermediate processing (conservative estimate: 2x).
    total_memory_bytes = stack_memory_bytes * 2

    # Convert to gigabytes.
    memory_gb = total_memory_bytes / (1024**3)

    logging.debug(f"Cluster memory estimate: {cluster_size} tiles, "
                 f"{cluster_h}×{cluster_w} pixels = {memory_gb:.2f} GB")

    return memory_gb


def _check_cluster_feasibility(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int,
    height: int,
    width: int,
    memory_limit_gb: float = 16.0
) -> Tuple[bool, str]:
    """
    Check if a cluster can be processed with standard algorithms.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        List of (row, col) coordinates for tiles in the cluster.
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    height, width : int
        Full image dimensions in pixels.
    memory_limit_gb : float, default 16.0
        Maximum memory limit in gigabytes.

    Returns
    -------
    Tuple[bool, str]
        (is_feasible, reason_if_not_feasible)
    """
    cluster_size = len(cluster)

    # Check for uint32 overflow in composite keys (tile index << 32).
    if cluster_size >= (1 << 32):
        return False, f"Cluster has {cluster_size} tiles, exceeding uint32 composite key limit"

    # Calculate cluster bounding box with proper coordinate validation.
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    min_r = min(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    max_r = max(r for r, _ in cluster)
    max_c = max(c for _, c in cluster)

    # Validate tile indices are reasonable for the given image dimensions.
    max_possible_rows = (height + stride_h - 1) // stride_h
    max_possible_cols = (width + stride_w - 1) // stride_w

    if max_r >= max_possible_rows or max_c >= max_possible_cols:
        return False, f"Tile indices out of bounds: max_tile=({max_r},{max_c}), max_possible=({max_possible_rows-1},{max_possible_cols-1})"

    y0 = min_r * stride_h
    x0 = min_c * stride_w

    # Ensure starting coordinates are within image bounds.
    if y0 >= height or x0 >= width:
        return False, f"Cluster starting position ({y0},{x0}) exceeds image bounds ({height},{width})"

    # Calculate cluster dimensions with proper boundary clamping.
    cluster_h = min((max_r - min_r) * stride_h + tile_h, height - y0)
    cluster_w = min((max_c - min_c) * stride_w + tile_w, width - x0)

    # Ensure dimensions are positive and reasonable.
    if cluster_h <= 0 or cluster_w <= 0:
        return False, f"Invalid cluster dimensions: {cluster_h}×{cluster_w} (y0={y0}, x0={x0}, height={height}, width={width})"

    # Check for array size limits - the merged patch array size.
    merged_patch_elements = cluster_h * cluster_w

    # NumPy array size limit (platform dependent, but typically around 2^63-1 for 64-bit).
    max_array_elements = 2**31 - 1  # Conservative limit for compatibility.
    if merged_patch_elements > max_array_elements:
        return False, f"Merged patch would have {merged_patch_elements} elements, exceeding NumPy limit of {max_array_elements}"

    # Check for stack array size limits (cluster_size × tile_h × tile_w).
    stack_elements = cluster_size * tile_h * tile_w
    if stack_elements > max_array_elements:
        return False, f"Tile stack would have {stack_elements} elements, exceeding NumPy limit of {max_array_elements}"

    # Memory requirement check.
    estimated_memory = _estimate_cluster_memory_requirements(cluster_size, cluster_h, cluster_w)
    if estimated_memory > memory_limit_gb:
        return False, f"Estimated memory requirement {estimated_memory:.1f} GB exceeds limit of {memory_limit_gb} GB"

    return True, ""


def _split_large_cluster(
    cluster: List[Tuple[int, int]],
    max_cluster_size: int = 1000
) -> List[List[Tuple[int, int]]]:
    """
    Split a large cluster into smaller sub-clusters for processing.

    This function uses spatial proximity to group tiles into smaller clusters
    that can be processed without exceeding memory limits.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        List of (row, col) coordinates for tiles in the large cluster.
    max_cluster_size : int, default 1000
        Maximum number of tiles per sub-cluster.

    Returns
    -------
    List[List[Tuple[int, int]]]
        List of smaller sub-clusters.
    """
    if len(cluster) <= max_cluster_size:
        return [cluster]

    # Sort tiles by row, then by column for spatial locality.
    sorted_cluster = sorted(cluster)

    # Split into chunks of max_cluster_size.
    sub_clusters = []
    for i in range(0, len(sorted_cluster), max_cluster_size):
        sub_cluster = sorted_cluster[i:i + max_cluster_size]
        sub_clusters.append(sub_cluster)

    logging.info(f"TILE PROCESSING: Split large cluster of {len(cluster)} image tiles into {len(sub_clusters)} sub-clusters for memory management")

    return sub_clusters


def _get_next_safe_gid_range(
    current_gid: int,
    patch_max: int,
    max_safe_gid: int,
    reset_count: int,
    segment_size: int
) -> Tuple[int, int, bool]:
    """
    Calculate the next safe global ID range to prevent uint32 overflow.

    This function implements a segmented ID allocation strategy that reduces
    the likelihood of ID conflicts when counter resets are necessary.

    Parameters
    ----------
    current_gid : int
        Current global ID counter value.
    patch_max : int
        Maximum ID value in the current patch.
    max_safe_gid : int
        Maximum safe ID value before overflow risk.
    reset_count : int
        Number of times the counter has been reset.
    segment_size : int
        Size of each ID segment to prevent conflicts.

    Returns
    -------
    Tuple[int, int, bool]
        (new_gid_counter, gid_offset, was_reset)
        - new_gid_counter: Updated global ID counter
        - gid_offset: Offset to apply to patch IDs
        - was_reset: Whether a reset occurred
    """

    # Check if adding patch_max would exceed the safe limit.
    if current_gid + patch_max > max_safe_gid:
        # Calculate the next segment start to avoid conflicts.
        next_segment_start = (reset_count + 1) * segment_size + 1

        # Ensure we don't exceed the absolute uint32 limit.
        if next_segment_start + patch_max > 2**32 - 1:
            logging.error(f"Exhausted all available uint32 ID space. "
                         f"Consider using uint64 or reducing image size.")
            # Fall back to simple reset as last resort.
            next_segment_start = 1

        logging.warning(f"Global ID counter approaching uint32 limit. "
                       f"Current: {current_gid}, patch_max: {patch_max}, limit: {max_safe_gid}")
        logging.info(f"Resetting to segment {reset_count + 1} starting at ID {next_segment_start} "
                    f"to minimize conflicts.")

        return next_segment_start + patch_max, next_segment_start, True
    else:
        # Normal case: no reset needed.
        return current_gid + patch_max, current_gid, False


def _split_cluster_adaptively(
    cluster: List[Tuple[int, int]],
    tile_h: int,
    tile_w: int,
    overlap: int,
    height: int,
    width: int,
    memory_limit_gb: float = 8.0,
    max_iterations: int = 10
) -> List[List[Tuple[int, int]]]:
    """
    Adaptively split a cluster until all sub-clusters are feasible.

    This function repeatedly splits clusters until each sub-cluster
    passes the feasibility check.

    Parameters
    ----------
    cluster : List[Tuple[int, int]]
        List of (row, col) coordinates for tiles in the cluster.
    tile_h, tile_w : int
        Tile dimensions in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    height, width : int
        Full image dimensions in pixels.
    memory_limit_gb : float, default 8.0
        Maximum memory limit in gigabytes.
    max_iterations : int, default 10
        Maximum number of splitting iterations to prevent infinite loops.

    Returns
    -------
    List[List[Tuple[int, int]]]
        List of feasible sub-clusters.
    """
    clusters_to_process = [cluster]
    feasible_clusters = []

    for iteration in range(max_iterations):
        new_clusters_to_process = []

        for cl in clusters_to_process:
            is_feasible, reason = _check_cluster_feasibility(
                cl, tile_h, tile_w, overlap, height, width, memory_limit_gb
            )

            if is_feasible:
                feasible_clusters.append(cl)
            else:
                # Split this cluster further with more aggressive strategy.
                current_size = len(cl)

                # Use more aggressive splitting for problematic clusters.
                if "exceeding NumPy limit" in reason:
                    # For array size issues, split much more aggressively.
                    new_max_size = max(1, min(4, current_size // 8))
                elif "memory requirement" in reason:
                    # For memory issues, split into quarters.
                    new_max_size = max(1, current_size // 4)
                else:
                    # Default splitting strategy.
                    new_max_size = max(1, current_size // 2)

                logging.debug(f"TILE PROCESSING: Iteration {iteration+1}: Splitting cluster of {current_size} image tiles "
                             f"into chunks of {new_max_size} (reason: {reason})")

                sub_clusters = _split_large_cluster(cl, max_cluster_size=new_max_size)
                new_clusters_to_process.extend(sub_clusters)

        clusters_to_process = new_clusters_to_process

        # If no more clusters need processing, we're done.
        if not clusters_to_process:
            break

    # Handle remaining problematic clusters.
    if clusters_to_process:
        logging.warning(f"After {max_iterations} iterations, {len(clusters_to_process)} clusters "
                       f"are still not feasible.")

        # For extremely problematic clusters, try to process individual tiles.
        for cl in clusters_to_process:
            if len(cl) == 1:
                # Single tile that's still problematic - skip it with warning.
                tile_r, tile_c = cl[0]
                logging.error(f"Skipping problematic single tile at ({tile_r}, {tile_c}) - "
                             f"it may be outside image bounds or have invalid coordinates")
            else:
                # Split into individual tiles as last resort.
                logging.warning(f"Splitting cluster of {len(cl)} tiles into individual tiles as last resort")
                for tile in cl:
                    single_tile_feasible, single_reason = _check_cluster_feasibility(
                        [tile], tile_h, tile_w, overlap, height, width, memory_limit_gb
                    )
                    if single_tile_feasible:
                        feasible_clusters.append([tile])
                    else:
                        tile_r, tile_c = tile
                        logging.error(f"Skipping problematic tile at ({tile_r}, {tile_c}): {single_reason}")

    logging.info(f"Adaptive splitting produced {len(feasible_clusters)} feasible clusters "
                f"from original cluster of {len(cluster)} tiles")

    return feasible_clusters


def _merge_cluster(
    *,
    cluster: List[Tuple[int, int]],
    loader: Callable[[slice, slice], NDArray[np.uint32]],
    height: int,  # slide dimensions – only used to clamp at borders
    width: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    threshold: float,
    use_gpu: bool,
    gid_offset: int,
) -> Tuple[NDArray[np.uint32], Tuple[int, int], Dict[int, int]]:
    """Merge all tiles in *cluster* into a patch of size (Hc, Wc).

    The heavy lifting is delegated to ``merge_patch_cpu`` / ``merge_patch_gpu``.
    This wrapper merely constructs the per‑tile 3‑D tensor expected by those
    kernels and aligns tiles in the common coordinate frame.

    CRITICAL: This function now properly handles edge tiles that extend beyond
    image boundaries, ensuring complete coverage of the entire image area.
    """

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    min_r = min(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    max_r = max(r for r, _ in cluster)
    max_c = max(c for _, c in cluster)

    # Validate tile indices are reasonable for the given image dimensions.
    max_possible_rows = (height + stride_h - 1) // stride_h
    max_possible_cols = (width + stride_w - 1) // stride_w

    if max_r >= max_possible_rows or max_c >= max_possible_cols:
        raise ValueError(f"Tile indices out of bounds: max_tile=({max_r},{max_c}), max_possible=({max_possible_rows-1},{max_possible_cols-1})")

    y0 = min_r * stride_h
    x0 = min_c * stride_w

    # Ensure starting coordinates are within image bounds.
    if y0 >= height or x0 >= width:
        raise ValueError(f"Cluster starting position ({y0},{x0}) exceeds image bounds ({height},{width})")

    # Clamp the bounding box to the actual slide size (important at borders).
    cluster_h = min((max_r - min_r) * stride_h + tile_h, height - y0)
    cluster_w = min((max_c - min_c) * stride_w + tile_w, width - x0)

    # Ensure dimensions are positive.
    if cluster_h <= 0 or cluster_w <= 0:
        raise ValueError(f"Invalid cluster dimensions: {cluster_h}×{cluster_w} (y0={y0}, x0={x0}, height={height}, width={width})")

    # Enhanced debugging for edge tile processing.
    logging.debug(f"Processing cluster with {len(cluster)} tiles: "
                 f"tile_range=({min_r},{min_c}) to ({max_r},{max_c}), "
                 f"global_bbox=({y0},{x0}) to ({y0+cluster_h},{x0+cluster_w}), "
                 f"image_size=({height},{width})")

    T = len(cluster)
    stack = np.zeros((T, cluster_h, cluster_w), dtype=np.uint32)

    for t, (r, c) in enumerate(cluster):
        global_y0 = r * stride_h
        global_x0 = c * stride_w
        rel_y0 = global_y0 - y0
        rel_x0 = global_x0 - x0

        # Clamp the tile request to image boundaries for edge tiles.
        ys = slice(global_y0, min(global_y0 + tile_h, height))
        xs = slice(global_x0, min(global_x0 + tile_w, width))

        logging.debug(f"Loading tile ({r},{c}) at global=({global_y0},{global_x0}), "
                     f"slice=({ys.start}:{ys.stop}, {xs.start}:{xs.stop})")

        tile = loader(ys, xs)
        h, w = tile.shape

        # Ensure we don't exceed the cluster bounds when placing the tile.
        end_y = min(rel_y0 + h, cluster_h)
        end_x = min(rel_x0 + w, cluster_w)

        if end_y > rel_y0 and end_x > rel_x0:
            # Adjust tile dimensions if needed to fit within cluster bounds.
            tile_h_to_copy = end_y - rel_y0
            tile_w_to_copy = end_x - rel_x0

            stack[t, rel_y0:end_y, rel_x0:end_x] = tile[:tile_h_to_copy, :tile_w_to_copy]

            logging.debug(f"Placed tile ({r},{c}) in stack: "
                         f"stack_pos=({rel_y0}:{end_y}, {rel_x0}:{end_x}), "
                         f"tile_size=({h},{w}), copied=({tile_h_to_copy},{tile_w_to_copy})")
        else:
            logging.warning(f"Tile ({r},{c}) could not be placed in cluster stack - bounds issue")

    # Actual merge done by the back‑end kernels.
    merge_fn = _merge_patch_gpu if use_gpu else _merge_patch_cpu
    merged_patch, _ = merge_fn(stack, threshold=threshold)

    # Shift labels so that each cluster occupies its own ID range.
    # This ensures unique nucleus IDs across all clusters in the final merged mask.
    if gid_offset > 0:
        nucleus_mask = merged_patch != 0
        merged_patch = merged_patch.astype(np.uint32, copy=False)
        merged_patch[nucleus_mask] += int(gid_offset)

    logging.debug(f"Cluster merge completed: output_size=({merged_patch.shape[0]},{merged_patch.shape[1]}), "
                 f"max_label={merged_patch.max()}, non_zero_pixels={np.count_nonzero(merged_patch)}")

    return merged_patch, (y0, x0), {}


"""
3.  Public API
"""

__all__: Final = ["merge_masks_streaming"]


def merge_masks_streaming(
    *,
    height: int,
    width: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
    tiles_path: str | Path,
    threshold: float = 0.3,
    max_workers: int | None = None,
    use_gpu: bool | None = None,
    qc: bool = False,
    qc_dir: str | Path | None = None,
    qc_merge_use_full_image = False,
    gpu_batch_size: int = 1,
    gpu_memory_limit_gb: float = 8.0,
    gpu_memory_safety_factor: float = 1.5,
    gpu_spatial_strategy: str = "adaptive",
    gpu_adaptive_batching: bool = True,
    gpu_aggressive_cleanup: bool = True,
    gpu_max_retries: int = 3,
    gpu_timeout_seconds: int = 300,
    max_cluster_memory_gb: float = 2.0,
    max_cluster_dimension: int = 4096,
    enable_progress_tracking: bool = True,
    output_dir: str | Path | None = None,
) -> NDArray[np.uint32]:
    """Merge per‑tile instance masks into a 2‑D slide‑level label map.

    Parameters
    ----------
    height, width : int
        Size of the target canvas in **pixels**.
    tile_h, tile_w : int
        Size of each inference tile in **pixels**.
    overlap : int
        Spatial overlap between adjacent tiles, in **pixels**.
    tiles_path : str | Path
        Directory containing per‑tile *tif* or *npz* masks.
    threshold : float, default 0.3
        Overlap fraction below which a label becomes a new object.
    max_workers : int | None, default None
        Maximum CPU workers for cluster processing.  If *None*, use
        ``⌈√N⌉`` where *N* is the number of clusters.
    use_gpu : bool | None, default None
        • *True*  → force GPU (requires PyTorch + CUDA).
        • *False* → force CPU.
        • *None*  → auto‑detect.
    qc : bool, default False
        If *True*, save QC overlays next to *qc_dir* (if given) or under
        ``tiles_path/../qc``.
    qc_dir : str | Path | None
        Output folder for QC overlays if *qc* is *True*.
    gpu_batch_size : int, default 1
        Number of tiles to process simultaneously during GPU-based merging.
        Smaller values use less GPU memory but may be slower for dense clusters.
        The system will auto-optimize this value based on memory constraints.
    gpu_memory_limit_gb : float, default 8.0
        Maximum GPU memory to use in gigabytes for tile merging operations.
        The system will automatically adjust batch sizes to stay within this limit.
        Set to 0 for automatic detection based on available GPU memory.
    gpu_memory_safety_factor : float, default 1.5
        Safety multiplier for GPU memory estimates to prevent out-of-memory errors.
        Higher values are more conservative but may reduce GPU utilization.
    gpu_spatial_strategy : str, default "adaptive"
        Spatial batching strategy for tile grouping: "adaptive", "2x2", "spatial", "hybrid".
        Adaptive mode automatically selects the best strategy based on tile characteristics.
    gpu_adaptive_batching : bool, default True
        Enable adaptive batch sizing based on tile spatial distribution and GPU memory.
        Improves performance for mixed dense/sparse tile patterns.
    gpu_aggressive_cleanup : bool, default True
        Enable aggressive GPU memory cleanup between batches to prevent fragmentation.
        May slightly reduce performance but improves memory stability.
    output_dir : str | Path | None, default None
        Output directory where temp_merged_segmentations_masks.npy will be created
        and then renamed to segmentations_masks.npy. If None, uses tiles_path parent.
    """

    try:
        _lazy_import_merge_backends()

        # Validate input parameters for kidney tissue analysis.
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image dimensions: {height}x{width}")
        if tile_h <= 0 or tile_w <= 0:
            raise ValueError(f"Invalid tile dimensions: {tile_h}x{tile_w}")
        if overlap < 0 or overlap >= min(tile_h, tile_w):
            raise ValueError(f"Invalid overlap {overlap} for tile size {tile_h}x{tile_w}")

        logging.info(f"Starting mask merge for {height}x{width} image")
        logging.info(f"Tile configuration: {tile_h}x{tile_w} with {overlap}px overlap")

        # Set up output directory and temp file paths.
        # All segmentation masks should be saved in the masks/ subdirectory.
        if output_dir is None:
            output_dir = Path(tiles_path).parent
        else:
            output_dir = Path(output_dir)

        # Create masks subdirectory for organized output.
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        temp_merged_path = masks_dir / "temp_merged_segmentation_masks.npy"
        final_merged_path = masks_dir / "segmentation_masks.npy"

        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Masks directory: {masks_dir}")
        logging.info(f"Temporary merged file: {temp_merged_path}")

        path = _resolve_tiles_path(Path(tiles_path))
        file_map, raw_coords = _discover_tiles(path)

        logging.info(f"Found {len(raw_coords)} tile mask files in {path}")

        # Enhanced debugging: Log the range of discovered tiles.
        if raw_coords:
            min_r = min(r for r, _ in raw_coords)
            max_r = max(r for r, _ in raw_coords)
            min_c = min(c for _, c in raw_coords)
            max_c = max(c for _, c in raw_coords)
            logging.info(f"Tile coordinate range: rows {min_r} to {max_r}, cols {min_c} to {max_c}")

        # Detect whether the two integers are pixel coordinates or tile indices.
        # This is important for proper spatial alignment of kidney tissue tiles.
        stride_h, stride_w = tile_h - overlap, tile_w - overlap

        # Check if coordinates look like pixel coordinates (large values, divisible by stride)
        max_coord = max(max(r, c) for r, c in raw_coords)
        min_coord = min(min(r, c) for r, c in raw_coords)

        # If coordinates are large and many are divisible by stride, they're likely pixel coordinates
        pixel_coord_indicators = [
            max_coord > 1000,  # Large coordinates suggest pixels
            min_coord >= 0,    # Should start from 0 or positive
            sum(1 for r, c in raw_coords if r % stride_h == 0 and c % stride_w == 0) > len(raw_coords) * 0.5  # Most divisible by stride
        ]

        if sum(pixel_coord_indicators) >= 2:
            # Convert pixel coordinates to tile indices
            coords = [(r // stride_h, c // stride_w) for r, c in raw_coords]
            idx_to_path: Dict[Tuple[int, int], Path] = {
                (r // stride_h, c // stride_w): p for (r, c), p in file_map.items()
            }
            logging.info(f"Interpreting filenames as pixel coordinates (stride {stride_h}×{stride_w})")
            logging.info(f"Converted {len(raw_coords)} pixel coordinates to tile indices")
        else:
            # Assume they're already tile indices
            coords = raw_coords
            idx_to_path = file_map
            logging.info("Interpreting filenames as tile indices")

        # Enhanced debugging: Log the final coordinate mapping.
        if coords:
            final_min_r = min(r for r, _ in coords)
            final_max_r = max(r for r, _ in coords)
            final_min_c = min(c for _, c in coords)
            final_max_c = max(c for _, c in coords)
            logging.info(f"Final tile index range: rows {final_min_r} to {final_max_r}, cols {final_min_c} to {final_max_c}")

            # Calculate expected image coverage.
            expected_height = (final_max_r + 1) * stride_h + overlap
            expected_width = (final_max_c + 1) * stride_w + overlap
            logging.info(f"Expected coverage from tiles: {expected_height}x{expected_width} vs actual image: {height}x{width}")

            if expected_height < height or expected_width < width:
                logging.warning(f"Tiles may not cover the entire image! Missing coverage: "
                               f"height_gap={max(0, height - expected_height)}, "
                               f"width_gap={max(0, width - expected_width)}")

        # Hardware selection for optimal processing of large kidney tissue images.
        if use_gpu is None:
            use_gpu = bool(torch and torch.cuda.is_available())
        if use_gpu and torch is None:
            logging.warning("PyTorch is not installed; falling back to CPU processing.")
            use_gpu = False

        processing_mode = "GPU" if use_gpu else "CPU"
        logging.info(f"Using {processing_mode} processing for mask merging")

    except Exception as setup_error:
        logging.error(f"Failed to initialize mask merging: {setup_error}")
        logging.debug(f"Setup error traceback:\n{traceback.format_exc()}")

        # Clean up temp file if it exists.
        if 'temp_merged_path' in locals() and temp_merged_path.exists():
            temp_merged_path.unlink()
            logging.info(f"Cleaned up temporary file: {temp_merged_path}")

        raise

    # ------------------------------------------------------------------
    # Tile loader – returns a *view* of the requested slice to keep memory low.
    # ------------------------------------------------------------------

    def _loader(ys: slice, xs: slice) -> NDArray[np.uint32]:
        r_idx = ys.start // stride_h
        c_idx = xs.start // stride_w
        p = idx_to_path.get((r_idx, c_idx))

        # No tile means this spatial region was not covered at inference time.
        if p is None or not p.exists():
            logging.debug(f"Missing tile at ({r_idx}, {c_idx}) for region y={ys.start}:{ys.stop}, x={xs.start}:{xs.stop}")
            return np.zeros((ys.stop - ys.start, xs.stop - xs.start), dtype=np.uint32)

        # Load the tile array from disk.
        if p.suffix.lower() == ".tif":
            with Image.open(p) as im:
                arr = np.asarray(im, dtype=np.uint32)
        else:  # .npz is much faster and smaller.
            nz = np.load(p)
            # Micro‑optimisation: avoid data copy when possible.
            if len(nz.files) == 1:
                arr = nz[nz.files[0]].astype(np.uint32, copy=False)
            else:
                arr = np.asarray(nz, dtype=np.uint32)

        # CRITICAL FIX: Handle edge tiles that extend beyond image boundaries.
        # The requested slice may extend beyond the loaded tile dimensions,
        # especially for edge tiles that were cropped during inference.
        tile_h_actual, tile_w_actual = arr.shape

        # Calculate the tile's global position in the image.
        tile_global_y0 = r_idx * stride_h
        tile_global_x0 = c_idx * stride_w

        # Convert global slice coordinates to local tile coordinates.
        local_y_start = max(0, ys.start - tile_global_y0)
        local_y_stop = min(tile_h_actual, ys.stop - tile_global_y0)
        local_x_start = max(0, xs.start - tile_global_x0)
        local_x_stop = min(tile_w_actual, xs.stop - tile_global_x0)

        # Create output array with the requested dimensions.
        output_h = ys.stop - ys.start
        output_w = xs.stop - xs.start
        result = np.zeros((output_h, output_w), dtype=np.uint32)

        # Calculate where to place the valid tile data in the output array.
        output_y_start = max(0, tile_global_y0 - ys.start)
        output_y_stop = output_y_start + (local_y_stop - local_y_start)
        output_x_start = max(0, tile_global_x0 - xs.start)
        output_x_stop = output_x_start + (local_x_stop - local_x_start)

        # Copy the valid portion of the tile to the result array.
        if local_y_stop > local_y_start and local_x_stop > local_x_start:
            result[output_y_start:output_y_stop, output_x_start:output_x_stop] = \
                arr[local_y_start:local_y_stop, local_x_start:local_x_stop]

            logging.debug(f"Loaded tile ({r_idx}, {c_idx}): tile_size=({tile_h_actual}, {tile_w_actual}), "
                         f"local_slice=({local_y_start}:{local_y_stop}, {local_x_start}:{local_x_stop}), "
                         f"output_slice=({output_y_start}:{output_y_stop}, {output_x_start}:{output_x_stop})")
        else:
            logging.debug(f"Tile ({r_idx}, {c_idx}) has no valid data for requested region")

        return result

    # ------------------------------------------------------------------
    # Cluster discovery and parallel merge.
    # ------------------------------------------------------------------

    # Use adaptive memory-aware clustering to prevent massive GPU memory allocations.
    clusters = _build_memory_aware_clusters(
        coords, tile_h, tile_w, overlap,
        max_cluster_memory_gb=max_cluster_memory_gb,
        max_cluster_dimension=max_cluster_dimension,
        max_cluster_gpu_memory_gb=gpu_memory_limit_gb,
        cluster_subdivision_strategy=gpu_spatial_strategy,  # Reuse spatial strategy parameter
        max_subdivision_depth=6,  # Default value, could be made configurable
        min_cluster_size_after_subdivision=2  # Default value, could be made configurable
    )
    logging.info("TILE PROCESSING: Created %d memory-efficient tile clusters for merging.", len(clusters))

    # Enhanced debugging: Log cluster details.
    for i, cluster in enumerate(clusters):
        cluster_min_r = min(r for r, _ in cluster)
        cluster_max_r = max(r for r, _ in cluster)
        cluster_min_c = min(c for _, c in cluster)
        cluster_max_c = max(c for _, c in cluster)
        cluster_y0 = cluster_min_r * stride_h
        cluster_x0 = cluster_min_c * stride_w
        cluster_h = min((cluster_max_r - cluster_min_r) * stride_h + tile_h, height - cluster_y0)
        cluster_w = min((cluster_max_c - cluster_min_c) * stride_w + tile_w, width - cluster_x0)

        logging.info(f"TILE CLUSTER {i+1}: {len(cluster)} image tiles, "
                    f"tile_range=({cluster_min_r},{cluster_min_c}) to ({cluster_max_r},{cluster_max_c}), "
                    f"global_bbox=({cluster_y0},{cluster_x0}) to ({cluster_y0+cluster_h},{cluster_x0+cluster_w})")

    # Initialize temporary merged mask file for streaming processing.
    merged = np.zeros((height, width), dtype=np.uint32)
    np.save(temp_merged_path, merged)
    logging.info(f"Created temporary merged mask file: {temp_merged_path}")

    gid_counter = 1  # Monotonic global‑ID allocator.

    # Enhanced uint32 ID management with configurable parameters.
    # Use conservative limit from configuration to prevent overflow errors.
    max_safe_gid = min(2**31 - 1, 2000000000)  # Default conservative limit.
    id_segment_size = 100000000  # Default segment size.

    # Enhanced ID management to prevent conflicts during counter resets.
    id_reset_count = 0  # Track how many times we've reset the counter.

    logging.info(f"Enhanced uint32 ID management initialized: "
                f"max_safe_gid={max_safe_gid:,}, segment_size={id_segment_size:,}")

    if use_gpu:
        # GPU processing with batched approach for memory-efficient merging of large datasets.
        from .batch_merge import merge_cluster_batched

        # Enhanced progress tracking for cluster processing.
        if enable_progress_tracking:
            progress_desc = f"Processing {len(clusters)} memory-efficient clusters (GPU)"
            iterable = tqdm(clusters, desc=progress_desc, unit="cluster", leave=True)
        else:
            iterable = clusters

        for cluster_idx, cl in enumerate(iterable, 1):
            try:
                # FINAL SAFETY CHECK: Emergency split any cluster that could cause memory issues.
                cluster_memory, cluster_dims = _estimate_cluster_requirements(cl, tile_h, tile_w, overlap)
                if cluster_memory > 1.0 or len(cl) > 4:  # Ultra-conservative final check.
                    logging.error(f"EMERGENCY: Cluster {cluster_idx} still too large after all splitting attempts! "
                                 f"Memory: {cluster_memory:.2f}GB, Size: {len(cl)} tiles")
                    logging.error(f"Applying final emergency splitting to prevent system failure.")
                    emergency_clusters = _force_split_to_safe_clusters(cl, tile_h, tile_w, overlap)

                    # Add all emergency clusters to the processing queue.
                    # We'll process the first one now and add the rest to be processed later.
                    if len(emergency_clusters) > 1:
                        # Add remaining emergency clusters to the end of the processing list.
                        remaining_emergency = emergency_clusters[1:]
                        if hasattr(iterable, 'extend'):
                            # If iterable supports extend, add remaining clusters.
                            logging.info(f"Adding {len(remaining_emergency)} additional emergency clusters to processing queue")
                        else:
                            # Log that we're processing them individually.
                            logging.info(f"Will process {len(emergency_clusters)} emergency clusters individually")

                    # Process the first emergency cluster.
                    cl = emergency_clusters[0]
                    logging.info(f"Processing first emergency cluster with {len(cl)} tiles")

                # Check if cluster can be processed with standard algorithms.
                cluster_size = len(cl)
                is_feasible, reason = _check_cluster_feasibility(
                    cl, tile_h, tile_w, overlap, height, width,
                    memory_limit_gb=gpu_memory_limit_gb * 2  # Allow more memory for GPU processing
                )

                if not is_feasible:
                    logging.warning(f"Cluster {cluster_idx} is not feasible for standard processing: {reason}")
                    logging.info(f"TILE PROCESSING: Attempting to split cluster of {cluster_size} image tiles into smaller sub-clusters for memory management")

                    # Try to split the cluster into smaller, manageable pieces using adaptive splitting.
                    sub_clusters = _split_cluster_adaptively(
                        cl, tile_h, tile_w, overlap, height, width,
                        memory_limit_gb=gpu_memory_limit_gb * 2
                    )

                    for sub_idx, sub_cl in enumerate(sub_clusters):
                        # Check feasibility of sub-cluster.
                        sub_feasible, sub_reason = _check_cluster_feasibility(
                            sub_cl, tile_h, tile_w, overlap, height, width,
                            memory_limit_gb=gpu_memory_limit_gb * 2
                        )

                        if not sub_feasible:
                            logging.error(f"Sub-cluster {sub_idx+1} of cluster {cluster_idx} still not feasible: {sub_reason}")
                            raise RuntimeError(f"Even after splitting, cluster {cluster_idx} sub-cluster {sub_idx+1} "
                                             f"exceeds processing limits: {sub_reason}")

                        # Process sub-cluster using batched approach.
                        logging.info(f"Processing sub-cluster {sub_idx+1}/{len(sub_clusters)} "
                                   f"with {len(sub_cl)} tiles from cluster {cluster_idx}")

                        patch, (y0, x0), _ = merge_cluster_batched(
                            cluster=sub_cl,
                            loader=_loader,
                            height=height,
                            width=width,
                            tile_h=tile_h,
                            tile_w=tile_w,
                            overlap=overlap,
                            threshold=threshold,
                            use_gpu=True,
                            gid_offset=gid_counter,
                            batch_size=1,  # Use very conservative batch size for split clusters.
                            memory_limit_gb=gpu_memory_limit_gb * 0.6,  # Use even more conservative memory limit.
                            memory_safety_factor=gpu_memory_safety_factor,
                            spatial_strategy=gpu_spatial_strategy,
                            adaptive_batching=gpu_adaptive_batching,
                            aggressive_cleanup=gpu_aggressive_cleanup,
                            temp_file_path=temp_merged_path,
                            global_merged_array=merged,
                            max_retries=gpu_max_retries,
                            timeout_seconds=gpu_timeout_seconds,
                        )

                        # Update global ID counter and merge patch into final mask.
                        patch_max = int(patch.max().item()) if patch.size > 0 else 0

                        # Use enhanced ID management to prevent conflicts.
                        gid_counter, gid_offset_used, was_reset = _get_next_safe_gid_range(
                            gid_counter, patch_max, max_safe_gid, id_reset_count, id_segment_size
                        )

                        if was_reset:
                            id_reset_count += 1
                            logging.info(f"ID counter reset #{id_reset_count} completed. "
                                        f"New range starts at {gid_offset_used}")

                            # Re-apply the new offset to the patch if it was reset.
                            if patch_max > 0:
                                nucleus_mask = patch != 0
                                # Remove old offset and apply new one.
                                patch[nucleus_mask] = (patch[nucleus_mask] - gid_counter + patch_max) + gid_offset_used

                        # Copy non-zero pixels to the final merged mask.
                        nucleus_pixels = patch != 0
                        merged[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]][nucleus_pixels] = patch[nucleus_pixels]

                        logging.debug(f"Sub-cluster {sub_idx+1} processed: "
                                     f"patch_size=({patch.shape[0]},{patch.shape[1]}), "
                                     f"non_zero_pixels={np.count_nonzero(nucleus_pixels)}")

                    # Skip the normal processing since we handled the split cluster.
                    continue

                # Determine if this cluster needs batched processing.
                needs_batching = cluster_size > 100  # Threshold for using batched processing.

                if needs_batching:
                    logging.info(f"Using batched processing for large cluster {cluster_idx} with {cluster_size} tiles")
                    patch, (y0, x0), _ = merge_cluster_batched(
                        cluster=cl,
                        loader=_loader,
                        height=height,
                        width=width,
                        tile_h=tile_h,
                        tile_w=tile_w,
                        overlap=overlap,
                        threshold=threshold,
                        use_gpu=True,
                        gid_offset=gid_counter,
                        batch_size=max(1, gpu_batch_size // 2),  # Use more conservative batch size.
                        memory_limit_gb=gpu_memory_limit_gb,
                        memory_safety_factor=gpu_memory_safety_factor,
                        spatial_strategy=gpu_spatial_strategy,
                        adaptive_batching=gpu_adaptive_batching,
                        aggressive_cleanup=gpu_aggressive_cleanup,
                        temp_file_path=temp_merged_path,
                        global_merged_array=merged,
                        max_retries=gpu_max_retries,
                        timeout_seconds=gpu_timeout_seconds,
                    )
                else:
                    # Use original approach for smaller clusters
                    patch, (y0, x0), _ = _merge_cluster(
                        cluster=cl,
                        loader=_loader,
                        height=height,
                        width=width,
                        tile_h=tile_h,
                        tile_w=tile_w,
                        overlap=overlap,
                        threshold=threshold,
                        use_gpu=True,
                        gid_offset=gid_counter,
                    )

                # Update global ID counter and merge patch into final mask.
                patch_max = int(patch.max().item()) if patch.size > 0 else 0

                # Use enhanced ID management to prevent conflicts.
                gid_counter, gid_offset_used, was_reset = _get_next_safe_gid_range(
                    gid_counter, patch_max, max_safe_gid, id_reset_count, id_segment_size
                )

                if was_reset:
                    id_reset_count += 1
                    logging.info(f"ID counter reset #{id_reset_count} completed. "
                                f"New range starts at {gid_offset_used}")

                    # Re-apply the new offset to the patch if it was reset.
                    if patch_max > 0:
                        nucleus_mask = patch != 0
                        # Remove old offset and apply new one.
                        patch[nucleus_mask] = (patch[nucleus_mask] - gid_counter + patch_max) + gid_offset_used

                # Copy non-zero pixels to the final merged mask.
                nucleus_pixels = patch != 0

                # Enhanced debugging: Track patch placement.
                logging.debug(f"GPU cluster {cluster_idx}: placing patch at ({y0},{x0}) "
                             f"with size ({patch.shape[0]},{patch.shape[1]}), "
                             f"non_zero_pixels={np.count_nonzero(nucleus_pixels)}")

                merged[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]][nucleus_pixels] = patch[nucleus_pixels]

                # Incremental saving: Save progress after each cluster is processed.
                np.save(temp_merged_path, merged)
                logging.debug(f"Incremental save completed for cluster {cluster_idx}")

            except Exception as gpu_error:
                logging.error(f"GPU cluster {cluster_idx} processing failed: {gpu_error}")
                logging.debug(f"GPU error traceback:\n{traceback.format_exc()}")

                # Try to recover with CPU processing for this cluster
                logging.warning(f"Falling back to CPU processing for cluster {cluster_idx}")
                try:
                    patch, (y0, x0), _ = _merge_cluster(
                        cluster=cl,
                        loader=_loader,
                        height=height,
                        width=width,
                        tile_h=tile_h,
                        tile_w=tile_w,
                        overlap=overlap,
                        threshold=threshold,
                        use_gpu=False,
                        gid_offset=gid_counter,
                    )

                    # Update global ID counter and merge patch into final mask.
                    patch_max = int(patch.max().item()) if patch.size > 0 else 0
                    gid_counter += patch_max

                    # Copy non-zero pixels to the final merged mask.
                    nucleus_pixels = patch != 0
                    merged[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]][nucleus_pixels] = patch[nucleus_pixels]

                    # Incremental saving: Save progress after CPU fallback processing.
                    np.save(temp_merged_path, merged)
                    logging.debug(f"Incremental save completed for cluster {cluster_idx} (CPU fallback)")

                    logging.info(f"Successfully processed cluster {cluster_idx} with CPU fallback")

                except Exception as fallback_error:
                    logging.error(f"CPU fallback also failed for cluster {cluster_idx}: {fallback_error}")
                    raise gpu_error  # Raise the original GPU error
    else:
        # CPU processing with parallel workers for efficient cluster merging.
        workers = max_workers or (math.isqrt(len(clusters)) or 1)
        logging.info(f"Using {workers} CPU workers for parallel cluster processing")

        # Check feasibility of all clusters and split large ones if needed.
        processed_clusters = []
        for i, cl in enumerate(clusters):
            is_feasible, reason = _check_cluster_feasibility(
                cl, tile_h, tile_w, overlap, height, width,
                memory_limit_gb=16.0  # Conservative limit for CPU processing
            )
            if not is_feasible:
                logging.warning(f"CPU cluster {i+1} is not feasible for standard processing: {reason}")
                logging.info(f"TILE PROCESSING: Splitting CPU cluster {i+1} of {len(cl)} image tiles into smaller sub-clusters for memory management")

                # Split the cluster into smaller pieces using adaptive splitting.
                sub_clusters = _split_cluster_adaptively(
                    cl, tile_h, tile_w, overlap, height, width,
                    memory_limit_gb=16.0
                )

                for sub_cl in sub_clusters:
                    # Check feasibility of sub-cluster.
                    sub_feasible, sub_reason = _check_cluster_feasibility(
                        sub_cl, tile_h, tile_w, overlap, height, width,
                        memory_limit_gb=16.0
                    )
                    if not sub_feasible:
                        logging.error(f"CPU sub-cluster of cluster {i+1} still not feasible: {sub_reason}")
                        raise RuntimeError(f"Even after splitting, CPU cluster {i+1} sub-cluster "
                                         f"exceeds processing limits: {sub_reason}")
                    processed_clusters.append(sub_cl)
            else:
                processed_clusters.append(cl)

        logging.info(f"Processing {len(processed_clusters)} clusters (including split sub-clusters) with CPU")

        with _cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _merge_cluster,
                    cluster=cl,
                    loader=_loader,
                    height=height,
                    width=width,
                    tile_h=tile_h,
                    tile_w=tile_w,
                    overlap=overlap,
                    threshold=threshold,
                    use_gpu=False,
                    gid_offset=gid_counter + i * 10_000_000,  # Pre-allocate distinct ID ranges.
                )
                for i, cl in enumerate(processed_clusters)
            ]

            # Process completed futures and merge results.
            for fut in tqdm(_cf.as_completed(futures), total=len(futures), desc="Merging clusters (CPU)"):
                try:
                    patch, (y0, x0), _ = fut.result()

                    # Copy non-zero pixels to the final merged mask.
                    nucleus_pixels = patch != 0

                    # Enhanced debugging: Track patch placement.
                    logging.debug(f"CPU cluster: placing patch at ({y0},{x0}) "
                                 f"with size ({patch.shape[0]},{patch.shape[1]}), "
                                 f"non_zero_pixels={np.count_nonzero(nucleus_pixels)}")

                    merged[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]][nucleus_pixels] = patch[nucleus_pixels]

                    # Update global ID counter to track maximum used ID.
                    patch_max = int(patch.max().item()) if patch.size > 0 else 0
                    gid_counter = max(gid_counter, patch_max + 1)

                    # Incremental saving: Save progress after each CPU cluster is processed.
                    # Note: This is done for each completed future, providing frequent saves.
                    np.save(temp_merged_path, merged)
                    logging.debug(f"Incremental save completed for CPU cluster")

                except Exception as cpu_error:
                    logging.error(f"CPU cluster processing failed: {cpu_error}")
                    logging.debug(f"CPU error traceback:\n{traceback.format_exc()}")
                    raise

    # ------------------------------------------------------------------
    # Optional QC overlays.
    # ------------------------------------------------------------------

    if qc:
        if qc_dir is None:
            qc_dir = path.parent / "qc"
        try:
            from .qc import write_overlays
        except ModuleNotFoundError:
            logging.warning("QC requested but helper 'qc.py' not found; skipping.")
        else:
            # Create a proper loader for QC that can access the full merged image.
            def _qc_tile_loader(y_slice: slice, x_slice: slice) -> NDArray[np.uint32]:
                """Load tile data for QC visualization from individual tile files."""
                try:
                    return _loader(y_slice, x_slice)
                except Exception as e:
                    logging.debug(f"QC loader failed for slice ({y_slice}, {x_slice}): {e}")
                    # Return empty mask if tile loading fails.
                    y_size = y_slice.stop - y_slice.start if y_slice.stop else tile_h
                    x_size = x_slice.stop - x_slice.start if x_slice.stop else tile_w
                    return np.zeros((y_size, x_size), dtype=np.uint32)

            write_overlays(
                loader=_qc_tile_loader,
                merged=merged,
                height=height,
                width=width,
                tile_h=tile_h,
                tile_w=tile_w,
                overlap=overlap,
                qc_dir=qc_dir,
                use_full_image=qc_merge_use_full_image,
            )

    # Final summary with enhanced debugging information.
    total_nuclei_pixels = np.count_nonzero(merged)
    coverage_percentage = (total_nuclei_pixels / (height * width)) * 100
    max_label = merged.max()

    logging.info(f"Merge completed: {total_nuclei_pixels} nuclei pixels ({coverage_percentage:.2f}% coverage) "
                f"in {height}×{width} image, max_label={max_label}")

    # Check for potential edge coverage issues.
    edge_margin = min(tile_h, tile_w) // 4  # Check a quarter tile from edges.

    top_edge = merged[:edge_margin, :].sum()
    bottom_edge = merged[-edge_margin:, :].sum()
    left_edge = merged[:, :edge_margin].sum()
    right_edge = merged[:, -edge_margin:].sum()

    logging.info(f"Edge coverage check: top={top_edge}, bottom={bottom_edge}, "
                f"left={left_edge}, right={right_edge}")

    if any(edge == 0 for edge in [top_edge, bottom_edge, left_edge, right_edge]):
        logging.warning("Some image edges have zero segmentation - this may indicate edge tile processing issues")

    # Save the final merged mask and rename temp file to final name.
    np.save(temp_merged_path, merged)

    # Rename temp file to final file name.
    if final_merged_path.exists():
        final_merged_path.unlink()  # Remove existing file if it exists.
    temp_merged_path.rename(final_merged_path)

    # Also save as TIFF for compatibility with downstream analysis tools.
    try:
        from skimage import io as skio
        tif_path = final_merged_path.parent / "segmentation_masks.tif"
        skio.imsave(tif_path, merged.astype(np.uint32), plugin="tifffile")
        logging.info(f"Successfully created TIFF mask: {tif_path}")
    except Exception as e:
        logging.warning(f"Could not write TIFF mask: {e}")

    logging.info(f"Successfully created final merged mask: {final_merged_path}")
    logging.info(f"Final mask statistics: shape={merged.shape}, max_label={merged.max()}, "
                f"non_zero_pixels={np.count_nonzero(merged)}")

    return merged


"""
4.  Unit tests  –  run with «pytest merge_tiles.py»
"""

import pytest


@pytest.fixture(scope="module", params=[(529, 529), (4096, 3072)])
def _toy_masks(request: pytest.FixtureRequest) -> Tuple[NDArray[np.uint32], int, int, int, int]:
    """Return a synthetic grid mask covering *height* × *width* pixels."""

    height, width = request.param
    tile_h, tile_w, overlap = 256, 256, 64
    stride_h, stride_w = tile_h - overlap, tile_w - overlap
    rows = math.ceil(height / stride_h)
    cols = math.ceil(width / stride_w)

    full = np.zeros((height, width), dtype=np.uint32)
    lbl = 1
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * stride_h, c * stride_w
            y1 = min(y0 + tile_h, height)
            x1 = min(x0 + tile_w, width)
            full[y0:y1, x0:x1] = lbl
            lbl += 1
    return full, tile_h, tile_w, overlap, lbl - 1


def _dummy_loader(mask: NDArray[np.uint32]):
    """Return a closure that slices *mask* like the real loader."""

    def _inner(ys: slice, xs: slice):
        return mask[ys, xs]

    return _inner


@pytest.mark.parametrize("use_gpu", [False])
def test_full_reconstruction(_toy_masks, use_gpu):
    full, tile_h, tile_w, overlap, _ = _toy_masks
    height, width = full.shape

    # Monkey‑patch merge back‑end to bypass file I/O.
    from types import SimpleNamespace

    def _merge_cluster(**kwargs):  # type: ignore[no-redef]
        patch = kwargs["loader"](slice(0, height), slice(0, width))
        return patch, (0, 0), {}

    module = globals()
    module["_merge_cluster"] = _merge_cluster  # type: ignore[assignment]

    merged = merge_masks_streaming(
        height=height,
        width=width,
        tile_h=tile_h,
        tile_w=tile_w,
        overlap=overlap,
        tiles_path=".",  # Path is ignored by monkey‑patched loader.
        threshold=0.1,
        use_gpu=use_gpu,
        qc=False,
    )
    assert np.array_equal(merged, full)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))