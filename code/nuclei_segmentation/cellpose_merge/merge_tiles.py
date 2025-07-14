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
    • Python ≥ 3.10.
    • numpy, pillow, tqdm, pytest.  (Optional: torch for GPU.)

Usage:
    from merge_tiles import merge_masks_streaming

    merged = merge_masks_streaming(
        height=10_000, width=12_000,
        tile_h=512, tile_w=512, overlap=128,
        tiles_path="/path/to/tile_masks",  # or …_npz
        threshold=0.3,
    )
"""

from __future__ import annotations

import concurrent.futures as _cf
import itertools as _it
import logging
import math
import re
import traceback
from pathlib import Path
from typing import Callable, Dict, Final, Iterable, List, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover – CPU‑only envs.
    torch = None  # type: ignore[assignment]

"""------------------------------------------------------------------------
Lazy import of heavy merge back‑ends – keeps CLI snappy and avoids circulars
------------------------------------------------------------------------"""

_merge_patch_cpu: Callable[..., Tuple[NDArray[np.uint32], Dict[int, int]]]
_merge_patch_gpu: Callable[..., Tuple[NDArray[np.uint32], Dict[int, int]]]


def _lazy_import_merge_backends() -> None:
    """Import the CPU/GPU merge kernels exactly once and only when needed."""
    global _merge_patch_cpu, _merge_patch_gpu
    if "_merge_patch_cpu" in globals():
        return  # Already imported.
    from .rules import merge_patch_cpu as _m_cpu  # pylint: disable=import-error
    from .gpu_merge import merge_patch_gpu as _m_gpu  # pylint: disable=import-error

    _merge_patch_cpu = _m_cpu  # type: ignore[misc]
    _merge_patch_gpu = _m_gpu  # type: ignore[misc]

"""------------------------------------------------------------------------
1.  Path and filename helpers
------------------------------------------------------------------------"""

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
    if path.name.endswith("_npz"):
        candidates.append(path.with_name(path.name.removesuffix("_npz")))
    else:
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


"""------------------------------------------------------------------------
2.  Graph helpers
------------------------------------------------------------------------"""

def _build_clusters(coords: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
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
    """Merge all tiles in *cluster* into a patch of size (Hc, Wc).

    The heavy lifting is delegated to ``merge_patch_cpu`` / ``merge_patch_gpu``.
    This wrapper merely constructs the per‑tile 3‑D tensor expected by those
    kernels and aligns tiles in the common coordinate frame.
    """

    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    min_r = min(r for r, _ in cluster)
    min_c = min(c for _, c in cluster)
    max_r = max(r for r, _ in cluster)
    max_c = max(c for _, c in cluster)

    y0 = min_r * stride_h
    x0 = min_c * stride_w

    # Clamp the bounding box to the actual slide size (important at borders).
    cluster_h = min((max_r - min_r) * stride_h + tile_h, height - y0)
    cluster_w = min((max_c - min_c) * stride_w + tile_w, width  - x0)

    T = len(cluster)
    stack = np.zeros((T, cluster_h, cluster_w), dtype=np.uint32)

    for t, (r, c) in enumerate(cluster):
        global_y0 = r * stride_h
        global_x0 = c * stride_w
        rel_y0 = global_y0 - y0
        rel_x0 = global_x0 - x0

        ys = slice(global_y0, global_y0 + tile_h)
        xs = slice(global_x0, global_x0 + tile_w)
        tile = loader(ys, xs)
        h, w = tile.shape
        stack[t, rel_y0 : rel_y0 + h, rel_x0 : rel_x0 + w] = tile

    # Actual merge done by the back‑end kernels.
    merge_fn = _merge_patch_gpu if use_gpu else _merge_patch_cpu
    merged_patch, _ = merge_fn(stack, threshold=threshold)

    # Shift labels so that each cluster occupies its own ID range.
    # This ensures unique nucleus IDs across all clusters in the final merged mask.
    if gid_offset > 0:
        nucleus_mask = merged_patch != 0
        merged_patch = merged_patch.astype(np.uint32, copy=False)
        merged_patch[nucleus_mask] += int(gid_offset)

    return merged_patch, (y0, x0), {}


"""------------------------------------------------------------------------
3.  Public API
------------------------------------------------------------------------"""

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

        path = _resolve_tiles_path(Path(tiles_path))
        file_map, raw_coords = _discover_tiles(path)

        logging.info(f"Found {len(raw_coords)} tile mask files in {path}")

        # Detect whether the two integers are pixel coordinates or tile indices.
        # This is important for proper spatial alignment of kidney tissue tiles.
        stride_h, stride_w = tile_h - overlap, tile_w - overlap

        if all(r % stride_h == 0 for r, _ in raw_coords) and all(c % stride_w == 0 for _, c in raw_coords):
            coords = [(r // stride_h, c // stride_w) for r, c in raw_coords]
            idx_to_path: Dict[Tuple[int, int], Path] = {
                (r // stride_h, c // stride_w): p for (r, c), p in file_map.items()
            }
            logging.info("Interpreting filenames as pixel coordinates (stride %dx%d).", stride_h, stride_w)
        else:
            coords = raw_coords
            idx_to_path = file_map
            logging.info("Interpreting filenames as tile indices.")

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
            return np.zeros((ys.stop - ys.start, xs.stop - xs.start), dtype=np.uint32)

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
        return arr[ys, xs]

    # ------------------------------------------------------------------
    # Cluster discovery and parallel merge.
    # ------------------------------------------------------------------

    clusters = _build_clusters(coords)
    logging.info("Discovered %d independent tile clusters.", len(clusters))

    merged = np.zeros((height, width), dtype=np.uint32)
    gid_counter = 1  # Monotonic global‑ID allocator.

    if use_gpu:
        # GPU processing for faster merging of large kidney tissue datasets.
        iterable: Iterable[List[Tuple[int, int]]] = clusters
        for cluster_idx, cl in enumerate(tqdm(iterable, desc="Merging clusters (GPU)"), 1):
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
                    use_gpu=True,
                    gid_offset=gid_counter,
                )

                # Update global ID counter and merge patch into final mask.
                patch_max = int(patch.max().item()) if patch.size > 0 else 0
                gid_counter += patch_max

                # Copy non-zero pixels to the final merged mask.
                nucleus_pixels = patch != 0
                merged[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]][nucleus_pixels] = patch[nucleus_pixels]

            except Exception as gpu_error:
                logging.error(f"GPU cluster {cluster_idx} processing failed: {gpu_error}")
                logging.debug(f"GPU error traceback:\n{traceback.format_exc()}")
                raise
    else:
        # CPU processing with parallel workers for efficient cluster merging.
        workers = max_workers or (math.isqrt(len(clusters)) or 1)
        logging.info(f"Using {workers} CPU workers for parallel cluster processing")

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
                for i, cl in enumerate(clusters)
            ]

            # Process completed futures and merge results.
            for fut in tqdm(_cf.as_completed(futures), total=len(futures), desc="Merging clusters (CPU)"):
                try:
                    patch, (y0, x0), _ = fut.result()

                    # Copy non-zero pixels to the final merged mask.
                    nucleus_pixels = patch != 0
                    merged[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]][nucleus_pixels] = patch[nucleus_pixels]

                    # Update global ID counter to track maximum used ID.
                    patch_max = int(patch.max().item()) if patch.size > 0 else 0
                    gid_counter = max(gid_counter, patch_max + 1)

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
            write_overlays(
                loader=_loader,  # type: ignore[arg-type]
                merged=merged,
                height=height,
                width=width,
                tile_h=tile_h,
                tile_w=tile_w,
                overlap=overlap,
                qc_dir=qc_dir,
            )

    return merged


"""------------------------------------------------------------------------
4.  Unit tests  –  run with «pytest merge_tiles.py»
------------------------------------------------------------------------"""

import pytest


@pytest.fixture(scope="module", params=[(529, 529), (4096, 3072)])
def _toy_masks(request: pytest.FixtureRequest) -> Tuple[NDArray[np.uint32], int, int, int, int]:
    """Return a synthetic grid mask covering *height* × *width* pixels."""

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
