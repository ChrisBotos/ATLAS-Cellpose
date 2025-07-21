import os
import shutil
import configparser
from datetime import datetime
from pathlib import Path
import torch

def setup_project_structure():
    """
    Create standard project directories if missing.
    Returns:
        dict: Paths to base, data, results, logs, etc.
    """
    base_dir = Path(__file__).resolve().parents[3]
    paths = {
        "base": base_dir,
        "code": base_dir / "code",
        "nuclei": base_dir / "code" / "nuclei_segmentation",
        "configs": base_dir / "configs",
        "data": base_dir / "data",
        "results": base_dir / "results",
        "logs": base_dir / "logs",
        "tests": base_dir / "tests",
        "debug": base_dir / "debug",
    }
    for path in paths.values():
        path.mkdir(exist_ok=True, parents=True)
    return paths


def load_config(config_path=None):
    """
    Load INI config, copy to timestamped results folder, and return parsed settings.

    Args:
        config_path (str or Path, optional): Path to original INI config file.

    Returns:
        tuple: (settings dict, CELLPOSE_PARAMS dict, PROJECT_DIRS dict).
    """
    dirs = setup_project_structure()

    config_path = config_path or (dirs["configs"] / "nuclei_segmentation_config.ini")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"[CONFIG ERROR] Missing config: {config_path}")

    base_config = configparser.ConfigParser()
    base_config.read(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = base_config.get("general", "output_dir", fallback="iri_results")
    output_dir = dirs["results"] / f"{timestamp}_{name}"
    output_dir.mkdir(exist_ok=True)

    copied_config_path = output_dir / "config_used.ini"
    shutil.copy2(config_path, copied_config_path)

    config = configparser.ConfigParser()
    config.read(copied_config_path)

    # Main settings.
    settings = {
        "output_dir": output_dir,
        "image_path": resolve_path(config.get("general", "image_path"), dirs["data"]),
        "upscale_factor": config.getint("general", "upscale_factor", fallback=1),
        "crop_image": config.getboolean("general", "crop_image", fallback=False),
        "enhance_contrast": config.getboolean("general", "enhance_contrast", fallback=False),
        "generate_overlay": config.getboolean("general", "generate_overlay", fallback=True),
        "crop_box": get_tuple(config, "general", "crop_box", default=(0, 1, 0, 1)),
        "debug_mode": config.getboolean("debug", "debug_mode", fallback=False),

        "enhance_dim": config.getboolean("gamma_correction", "enhance_dim", fallback=False),
        "min_gamma": config.getfloat("gamma_correction", "min_gamma", fallback=1.9),
        "max_gamma": config.getfloat("gamma_correction", "max_gamma", fallback=2.2),


        "use_edge_detection": config.getboolean("edge_detection", "use_edge_detection", fallback=False),
        "apply_watershed": config.getboolean("watershed", "apply_watershed", fallback=False),
        "clahe_cliplimit": config.getfloat("clahe", "cliplimit", fallback=5.0),
        "clahe_tile_grid_size": get_tuple(config, "clahe", "tile_grid_size", default=(16, 16), cast=int),
        "canny_threshold1": config.getint("edge_detection", "canny_threshold1", fallback=50),
        "canny_threshold2": config.getint("edge_detection", "canny_threshold2", fallback=150),
        "area_threshold_for_watershed": config.getint("watershed", "area_threshold", fallback=1000),
        "local_maxima_footprint": get_tuple(config, "watershed", "local_maxima_footprint", default=(3, 3), cast=int),

        "use_tiling": config.getboolean("tiling", "use_tiling", fallback=True),
        "merge_overlap_threshold": config.getfloat("tiling", "merge_overlap_threshold", fallback=0.3),
        "tile_side_length": config.getint("tiling", "tile_side_length", fallback=1024),
        "tile_overlap": config.getfloat("tiling", "tile_overlap", fallback=0.1),
        "small_overlay_size": config.getint("overlay", "small_overlay_size", fallback=1024),
        "qc_overlays": config.getboolean("tiling", "qc_overlays", fallback=True),
        "qc_downsample_factor": config.getint("tiling", "qc_downsample_factor", fallback=4),
        "qc_merge_use_full_image": config.getboolean("tiling", "qc_merge_use_full_image", fallback=True),
        "memmap_dtype": config.get("tiling", "memmap_dtype", fallback="uint32"),

        # GPU batched processing parameters for handling large images with thousands of tiles.
        "gpu_batch_size": config.getint("tiling", "gpu_batch_size", fallback=1),
        "gpu_memory_limit_gb": config.getfloat("tiling", "gpu_memory_limit_gb", fallback=8.0),

        # Enhanced GPU memory management parameters for optimized tile merging.
        "gpu_memory_safety_factor": config.getfloat("tiling", "gpu_memory_safety_factor", fallback=1.5),
        "gpu_spatial_strategy": config.get("tiling", "gpu_spatial_strategy", fallback="adaptive"),
        "gpu_adaptive_batching": config.getboolean("tiling", "gpu_adaptive_batching", fallback=True),
        "gpu_aggressive_cleanup": config.getboolean("tiling", "gpu_aggressive_cleanup", fallback=True),

        # Timeout and retry parameters to prevent infinite loops.
        "gpu_max_retries": config.getint("tiling", "gpu_max_retries", fallback=3),
        "gpu_timeout_seconds": config.getint("tiling", "gpu_timeout_seconds", fallback=300),

        # Memory-aware clustering parameters to prevent problematic array allocations.
        "max_cluster_memory_gb": config.getfloat("tiling", "max_cluster_memory_gb", fallback=2.0),
        "max_cluster_dimension": config.getint("tiling", "max_cluster_dimension", fallback=4096),
        "enable_progress_tracking": config.getboolean("tiling", "enable_progress_tracking", fallback=True),

        # Adaptive cluster subdivision parameters to prevent massive GPU memory allocations.
        "max_cluster_gpu_memory_gb": config.getfloat("tiling", "max_cluster_gpu_memory_gb", fallback=4.0),
        "cluster_subdivision_strategy": config.get("tiling", "cluster_subdivision_strategy", fallback="spatial_quadtree"),
        "max_subdivision_depth": config.getint("tiling", "max_subdivision_depth", fallback=6),
        "min_cluster_size_after_subdivision": config.getint("tiling", "min_cluster_size_after_subdivision", fallback=2),

        # uint32 ID management parameters to prevent overflow errors.
        "uint32_id_management": config.get("tiling", "uint32_id_management", fallback="hybrid"),
        "uint32_conservative_limit": config.getint("tiling", "uint32_conservative_limit", fallback=2000000000),
        "uint32_segment_size": config.getint("tiling", "uint32_segment_size", fallback=100000000),

        "use_previous_results": config.getboolean("using_previous_results", "use_previous_results", fallback=False),
        "previous_results_dir": resolve_path(config.get("using_previous_results", "previous_results_dir", fallback=""), dirs["results"]),
        "skip_and_copy_preprocessing": config.getboolean("using_previous_results", "skip_and_copy_preprocessing", fallback=False),
        "skip_and_copy_segmentation": config.getboolean("using_previous_results", "skip_and_copy_segmentation", fallback=False),
        "skip_and_copy_merging": config.getboolean("using_previous_results", "skip_and_copy_merging", fallback=False),
        "skip_and_copy_postprocessing": config.getboolean("using_previous_results", "skip_and_copy_postprocessing", fallback=False),
        "skip_and_copy_visualization": config.getboolean("using_previous_results", "skip_and_copy_visualization", fallback=False),
    }

    # Cellpose-specific settings optimized for adaptive diameter detection.
    CELLPOSE_PARAMS = {
        "model_type": config.get("cellpose", "model_type", fallback="nuclei"),
        "gpu": config.getboolean("cellpose", "gpu", fallback=True) and torch.cuda.is_available(),
        "diameter": config.getint("cellpose", "diameter", fallback=0),  # 0 = auto-detection.
        "flow_threshold": config.getfloat("cellpose", "flow_threshold", fallback=0.4),
        "cellprob_threshold": config.getfloat("cellpose", "cellprob_threshold", fallback=-9.0),
        "resample": config.getboolean("cellpose", "resample", fallback=True),  # Required for diameter=0.
        "stitch_threshold": config.getfloat("cellpose", "stitch_threshold", fallback=0.4),
        "channels": get_tuple(config, "cellpose", "channels", default=(0, 0), cast=int),
        "batch_size": choose_batch_size(settings.get("tile_side_length")**2),

        # Parallel processing parameters for improved performance.
        "enable_parallel_processing": config.getboolean("cellpose", "enable_parallel_processing", fallback=True),
        "parallel_batch_size": config.getint("cellpose", "parallel_batch_size", fallback=4),
        "parallel_max_workers": config.getint("cellpose", "parallel_max_workers", fallback=2),
        "parallel_memory_limit_gb": config.getfloat("cellpose", "parallel_memory_limit_gb", fallback=6.0),
        "parallel_timeout_seconds": config.getint("cellpose", "parallel_timeout_seconds", fallback=300),
    }

    return settings, CELLPOSE_PARAMS, dirs


def resolve_path(path_str, data_dir):
    """
    Resolve paths with cross-platform support for WSL and Windows environments.

    This function handles Windows paths (C:/) when running in WSL by converting
    them to the appropriate /mnt/c/ format, ensuring compatibility for kidney
    tissue analysis workflows across different development environments.
    """
    import os

    path_str = str(path_str).strip()

    # Handle Windows paths when running in WSL.
    if os.name == 'posix' and path_str.startswith(('C:/', 'D:/', 'E:/', 'F:/')):
        # Convert Windows path to WSL mount point.
        drive_letter = path_str[0].lower()
        wsl_path = f"/mnt/{drive_letter}" + path_str[2:].replace('\\', '/')
        path = Path(wsl_path)
    else:
        path = Path(path_str)

    # Check if path is absolute and exists.
    if path.is_absolute() and path.exists():
        return path

    # Try relative to data directory.
    if not path.is_absolute():
        full = data_dir / path
        if full.exists():
            return full

    # Final fallback - resolve as-is.
    return path.resolve()


def get_tuple(config, section, option, default, cast=float):
    """
    Parse a comma-separated config option into a tuple.
    """
    try:
        return tuple(cast(v.strip()) for v in config.get(section, option).split(","))
    except Exception:
        return default


def choose_batch_size(tile_pixels, bytes_per_pixel=1, target_mem=150_000_000):
    """
    Estimate batch size from available GPU memory.
    """
    if not torch.cuda.is_available():
        return 1
    props = torch.cuda.get_device_properties(0)
    usable = props.total_memory // 2
    per_tile = tile_pixels * bytes_per_pixel
    return max(1, usable // per_tile)
