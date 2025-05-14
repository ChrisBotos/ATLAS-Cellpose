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
        tuple: (SETTINGS dict, CELLPOSE_PARAMS dict, PROJECT_DIRS dict, Path to copied config).
    """
    dirs = setup_project_structure()

    config_path = config_path or (dirs["configs"] / "nuclei_segmentation_config.ini")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"[CONFIG ERROR] Missing config: {config_path}")

    # Load the original config first just to extract output_dir name.
    base_config = configparser.ConfigParser()
    base_config.read(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = base_config.get("General", "output_dir", fallback="iri_results")
    output_dir = dirs["results"] / f"{timestamp}_{name}"
    output_dir.mkdir(exist_ok=True)

    # Copy config to output dir and reload from there.
    copied_config_path = output_dir / "config_used.ini"
    shutil.copy2(config_path, copied_config_path)

    config = configparser.ConfigParser()
    config.read(copied_config_path)

    SETTINGS = {
        "OUTPUT_DIR": output_dir,
        "IMAGE_PATH": resolve_path(config.get("General", "image_path"), dirs["data"]),
        "UPSCALE_FACTOR": config.getint("General", "upscale_factor", fallback=1),
        "CROP_IMAGE": config.getboolean("General", "crop_image", fallback=False),
        "ENHANCE_CONTRAST": config.getboolean("General", "enhance_contrast", fallback=False),
        "GENERATE_OVERLAY": config.getboolean("General", "generate_overlay", fallback=True),
        "USE_EDGE_DETECTION": config.getboolean("EdgeDetection", "use_edge_detection", fallback=False),
        "APPLY_WATERSHED": config.getboolean("Watershed", "apply_watershed", fallback=False),
        "USE_TILING": config.getboolean("Tiling", "use_tiling", fallback=True),
        "MERGE_OVERLAP_THRESHOLD": config.getfloat("Tiling", "merge_overlap_threshold", fallback=0.3),
        "DEBUG_MODE": config.getboolean("Debug", "debug_mode", fallback=False),
        "CROP_BBOX": get_tuple(config, "General", "crop_bbox", default=(0, 1, 0, 1)),
        "CLAHE_CLIPLIMIT": config.getfloat("CLAHE", "cliplimit", fallback=2.0),
        "CLAHE_TILE_GRID_SIZE": get_tuple(config, "CLAHE", "tile_grid_size", default=(8, 8), cast=int),
        "enhance_dim": config.getboolean("Gamma_Correction", "enhance_dim", fallback=False),
        "MIN_GAMMA": config.getfloat("Gamma_Correction", "min_gamma", fallback=1.9),
        "MAX_GAMMA": config.getfloat("Gamma_Correction", "max_gamma", fallback=2.2),
        "CANNY_THRESHOLD1": config.getint("EdgeDetection", "canny_threshold1", fallback=50),
        "CANNY_THRESHOLD2": config.getint("EdgeDetection", "canny_threshold2", fallback=150),
        "AREA_THRESHOLD_FOR_WATERSHED": config.getint("Watershed", "area_threshold", fallback=1000),
        "LOCAL_MAXIMA_FOOTPRINT": get_tuple(config, "Watershed", "local_maxima_footprint", default=(3, 3), cast=int),
        "tile_side_length": config.getint("Tiling", "tile_side_length", fallback=1024),
        "TILE_OVERLAP": config.getfloat("Tiling", "tile_overlap", fallback=0.1),
        "SMALL_OVERLAY_SIZE": config.getint("Overlay", "small_overlay_size", fallback=1024),
    }

    CELLPOSE_PARAMS = {
        "model_type": config.get("Cellpose", "model_type", fallback="nuclei"),
        "gpu": config.getboolean("Cellpose", "gpu", fallback=True) and torch.cuda.is_available(),
        "diameter": config.getint("Cellpose", "diameter", fallback=0),
        "flow_threshold": config.getfloat("Cellpose", "flow_threshold", fallback=0.4),
        "cellprob_threshold": config.getfloat("Cellpose", "cellprob_threshold", fallback=0.0),
        "resample": config.getboolean("Cellpose", "resample", fallback=True),
        "stitch_threshold": config.getfloat("Cellpose", "stitch_threshold", fallback=0.4),
        "channels": get_tuple(config, "Cellpose", "channels", default=(0, 0), cast=int),
        "batch_size": 1,
    }

    return SETTINGS, CELLPOSE_PARAMS, dirs


def resolve_path(path_str, data_dir):
    """Ensure absolute image path, fallback to data dir if needed."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    full = data_dir / path
    return full if full.exists() else path.resolve()


def get_tuple(config, section, option, default, cast=float):
    try:
        return tuple(cast(v.strip()) for v in config.get(section, option).split(","))
    except Exception:
        return default


def choose_batch_size(tile_pixels, bytes_per_pixel=1, target_mem=150_000_000):
    """Estimate batch size from available GPU memory."""
    if not torch.cuda.is_available():
        return 1
    props = torch.cuda.get_device_properties(0)
    usable = props.total_memory // 2
    per_tile = tile_pixels * bytes_per_pixel
    return max(1, usable // per_tile)
