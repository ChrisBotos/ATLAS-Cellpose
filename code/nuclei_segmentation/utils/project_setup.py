"""
Project Setup Utilities for Kidney I/R Injury Spatial Multiomics Analysis.

This module provides functions for setting up the project structure and loading
configuration settings for the nuclei segmentation pipeline. It ensures that all
necessary directories exist and that configuration parameters are properly loaded
with appropriate defaults.

The module handles:
1. Creating the project directory structure
2. Loading and validating configuration files
3. Backing up configuration files for reproducibility
4. Setting default parameters when not specified in the config
"""

# Standard library imports.
import os
import sys
import configparser
import shutil
import traceback
from datetime import datetime
from pathlib import Path


def setup_project_structure():
    """
    Create the project directory structure if it doesn't exist.

    Creates all necessary directories for the pipeline including configs, data,
    results, logs, and debug directories. This ensures the pipeline can run without
    file system errors even on a fresh installation.

    Returns:
        dict: Dictionary containing paths to all project directories.
    """
    # Define base directories.
    script_path = Path(__file__).absolute()

    # Check if we're running from within the code/nuclei_segmentation/utils directory
    if script_path.parent.name == "utils" and script_path.parent.parent.name == "nuclei_segmentation":
        # Running as a module from the correct location
        base_dir = script_path.parent.parent.parent.parent  # Go up to project root
        code_dir = base_dir / "code"
        nuclei_dir = code_dir / "nuclei_segmentation"
    else:
        # Running the script directly from another location
        base_dir = script_path.parent.parent.parent  # Assume we're in utils dir
        code_dir = base_dir / "code"
        nuclei_dir = script_path.parent.parent

    dirs = {
        "base": base_dir,
        "code": code_dir,
        "nuclei": nuclei_dir,
        "configs": base_dir / "configs",
        "data": base_dir / "data",
        "results": base_dir / "results",
        "logs": base_dir / "logs",
        "tests": base_dir / "tests",
        "utils": nuclei_dir / "utils",
        "debug": base_dir / "debug"
    }

    # Create directories if they don't exist.
    for dir_path in dirs.values():
        dir_path.mkdir(exist_ok=True)

    return dirs


def load_config(config_path=None):
    """
    Load configuration from the specified INI file.

    Parses the configuration file to extract all settings for the segmentation pipeline.
    Provides sensible defaults for optional parameters and performs validation.
    Makes a copy of the configuration file in the results directory for future reference.

    Args:
        config_path: Path to the configuration INI file. If None, uses the default config.

    Returns:
        tuple: (SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS) dictionaries containing all
               configuration parameters needed for the pipeline.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If there's an error parsing the configuration file.
    """
    # Get current timestamp for output directory naming.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get project directories.
    PROJECT_DIRS = setup_project_structure()

    # =========================================================================
    # Find and validate the configuration file.
    # =========================================================================
    if config_path is None:
        # Use the PROJECT_DIRS to get the absolute path to the default config file.
        config_path = PROJECT_DIRS["configs"] / "nuclei_segmentation_config.ini"

        if not config_path.exists():
            # If config file is not found, raise an error with helpful message.
            raise FileNotFoundError(
                f"Configuration file not found at expected location: {config_path}"
            )
    elif not os.path.exists(config_path):
        # If a custom config path is provided but doesn't exist, raise an error.
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Store the actual config path used for later reference.
    actual_config_path = Path(config_path)
    print(f"DEBUG: Using configuration file: {actual_config_path}")

    # =========================================================================
    # Parse the configuration file.
    # =========================================================================
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        print(f"DEBUG: Successfully parsed configuration file with {len(config.sections())} sections.")
    except Exception as e:
        raise ValueError(f"Error parsing configuration file: {e}")

    # =========================================================================
    # Create output directory and make a copy of the config file.
    # =========================================================================
    # Define output directory path with timestamp for uniqueness.
    output_dir = PROJECT_DIRS["results"] / f"{config.get('General', 'output_dir')}_{timestamp}"

    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
    print(f"DEBUG: Created output directory: {output_dir}")

    # Make a copy of the configuration file in the results directory for future reference.
    config_backup_path = output_dir / "config_used.ini"
    try:
        shutil.copy2(actual_config_path, config_backup_path)
        print(f"DEBUG: Configuration file backed up to: {config_backup_path}")
    except Exception as e:
        print(f"WARNING: Could not backup configuration file: {e}")

    # =========================================================================
    # Parse general settings from config with default values for optional parameters.
    # =========================================================================
    SETTINGS = {
        "IMAGE_PATH": config.get("General", "image_path"),
        "OUTPUT_DIR": output_dir,
        "UPSCALE_FACTOR": config.getint("General", "upscale_factor", fallback=1),
        "CROP_IMAGE": config.getboolean("General", "crop_image", fallback=False),
        "ENHANCE_CONTRAST": config.getboolean("General", "enhance_contrast", fallback=False),
        "ENHANCE_DIM": config.getboolean("General", "enhance_dim", fallback=False),
        "GENERATE_OVERLAY": config.getboolean("General", "generate_overlay", fallback=True),
        "USE_EDGE_DETECTION": config.getboolean("EdgeDetection", "use_edge_detection", fallback=False),
        "APPLY_WATERSHED": config.getboolean("Watershed", "apply_watershed", fallback=False),
        "USE_TILING": config.getboolean("Tiling", "use_tiling", fallback=True),
        "MERGE_OVERLAP_THRESHOLD": config.getfloat("Tiling", "merge_overlap_threshold", fallback=0.3),
        "DEBUG_MODE": config.getboolean("Debug", "debug_mode", fallback=False) if "Debug" in config.sections() else False,
    }

    # =========================================================================
    # Parse optional settings with try/except blocks to provide defaults.
    # =========================================================================
    # Cropping parameters.
    try:
        SETTINGS["CROP_BBOX"] = tuple(map(float, config.get("General", "crop_bbox").split(',')))
    except (configparser.NoOptionError, ValueError):
        SETTINGS["CROP_BBOX"] = (0, 1, 0, 1)  # Default: full image.

    # CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters.
    try:
        SETTINGS["CLAHE_CLIPLIMIT"] = config.getfloat("CLAHE", "cliplimit", fallback=2.0)
        SETTINGS["CLAHE_TILE_GRID_SIZE"] = tuple(map(int, config.get("CLAHE", "tile_grid_size", fallback="8,8").split(',')))
    except (configparser.NoSectionError, ValueError):
        SETTINGS["CLAHE_CLIPLIMIT"] = 2.0
        SETTINGS["CLAHE_TILE_GRID_SIZE"] = (8, 8)

    # Edge detection parameters.
    try:
        SETTINGS["CANNY_THRESHOLD1"] = config.getint("EdgeDetection", "canny_threshold1", fallback=50)
        SETTINGS["CANNY_THRESHOLD2"] = config.getint("EdgeDetection", "canny_threshold2", fallback=150)
    except configparser.NoSectionError:
        SETTINGS["CANNY_THRESHOLD1"] = 50
        SETTINGS["CANNY_THRESHOLD2"] = 150

    # Watershed parameters for splitting merged nuclei.
    try:
        SETTINGS["AREA_THRESHOLD_FOR_WATERSHED"] = config.getint("Watershed", "area_threshold", fallback=1000)
        SETTINGS["LOCAL_MAXIMA_FOOTPRINT"] = tuple(map(int, config.get("Watershed", "local_maxima_footprint", fallback="3,3").split(',')))
    except configparser.NoSectionError:
        SETTINGS["AREA_THRESHOLD_FOR_WATERSHED"] = 1000
        SETTINGS["LOCAL_MAXIMA_FOOTPRINT"] = (3, 3)

    # Tiling parameters for processing large images.
    try:
        SETTINGS["tile_side_length"] = config.getint("Tiling", "tile_side_length", fallback=1024)
        SETTINGS["TILE_OVERLAP"] = config.getfloat("Tiling", "tile_overlap", fallback=0.1)
        SETTINGS["MERGE_OVERLAP_THRESHOLD"] = config.getfloat("Tiling", "merge_overlap_threshold", fallback=0.3)
    except configparser.NoSectionError:
        SETTINGS["tile_side_length"] = 1024
        SETTINGS["TILE_OVERLAP"] = 0.1
        SETTINGS["MERGE_OVERLAP_THRESHOLD"] = 0.3

    # Overlay visualization parameters.
    try:
        SETTINGS["SMALL_OVERLAY_SIZE"] = config.getint("Overlay", "small_overlay_size", fallback=1024)
    except configparser.NoSectionError:
        SETTINGS["SMALL_OVERLAY_SIZE"] = 1024

    # =========================================================================
    # Ensure image path is absolute for consistent file access.
    # =========================================================================
    if not os.path.isabs(SETTINGS["IMAGE_PATH"]):
        # Convert to Path object for consistent handling.
        img_path = Path(SETTINGS["IMAGE_PATH"])

        # Check if the path exists directly.
        if img_path.exists():
            SETTINGS["IMAGE_PATH"] = img_path.absolute()
        else:
            # Try to find it in the data directory.
            data_path = PROJECT_DIRS["data"] / img_path
            if data_path.exists():
                SETTINGS["IMAGE_PATH"] = data_path
            else:
                # Keep the original path but make it absolute for error reporting.
                SETTINGS["IMAGE_PATH"] = data_path

    # =========================================================================
    # Parse Cellpose parameters with appropriate defaults.
    # =========================================================================
    import torch
    
    CELLPOSE_PARAMS = {
        "model_type": config.get("Cellpose", "model_type", fallback="nuclei"),
        "gpu": config.getboolean("Cellpose", "gpu", fallback=True) and torch.cuda.is_available(),
        "diameter": config.getint("Cellpose", "diameter", fallback=0),
        "flow_threshold": config.getfloat("Cellpose", "flow_threshold", fallback=0.4),
        "cellprob_threshold": config.getfloat("Cellpose", "cellprob_threshold", fallback=0.0),
        "resample": config.getboolean("Cellpose", "resample", fallback=True),
        "stitch_threshold": config.getfloat("Cellpose", "stitch_threshold", fallback=0.4),
        "batch_size": 1  # Placeholder, will be updated dynamically based on GPU memory.
    }

    # Parse channels with error handling.
    try:
        CELLPOSE_PARAMS["channels"] = tuple(map(int, config.get("Cellpose", "channels", fallback="0,0").split(',')))
    except ValueError:
        CELLPOSE_PARAMS["channels"] = (0, 0)  # Default: grayscale.

    # Print summary of loaded configuration.
    print(f"DEBUG: Configuration loaded successfully with {len(SETTINGS)} settings and {len(CELLPOSE_PARAMS)} Cellpose parameters.")

    return SETTINGS, CELLPOSE_PARAMS, PROJECT_DIRS


def choose_batch_size(tile_pixels, bytes_per_pixel=1, target_mem_per_batch=150_000_000):
    """
    Calculate optimal batch size for GPU processing based on available memory.

    Dynamically determines how many image tiles can be processed in parallel
    based on the GPU memory available and the size of each tile.

    Args:
        tile_pixels: Number of pixels per tile (tile_side_length²).
        bytes_per_pixel: Memory usage per pixel (1 for uint8, ~4 for float32).
        target_mem_per_batch: Target GPU memory usage per batch in bytes.

    Returns:
        int: Optimal batch size for GPU processing.
    """
    import torch
    
    if not torch.cuda.is_available():
        return 1

    # Get GPU properties.
    props = torch.cuda.get_device_properties(0)
    total_mem = props.total_memory  # in bytes.

    # Use half of available memory to be safe.
    usable = total_mem // 2

    # Calculate memory needed per tile.
    bytes_per_patch = tile_pixels * bytes_per_pixel

    # Calculate how many tiles fit into target memory.
    max_batch = max(1, usable // (bytes_per_patch * (usable // target_mem_per_batch)))
    return int(max_batch)
