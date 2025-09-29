"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: config_loader.py
Description:
    Configuration loader for engineered feature extraction and clustering analysis.
    Provides centralized configuration management for nuclear morphological feature analysis,
    clustering, and visualization parameters specific to kidney I/R injury research.

Dependencies:
    • Python >= 3.10.
    • configparser (standard library).
    • pathlib (standard library).

Usage:
    from config_loader import load_feature_extraction_config
    config = load_feature_extraction_config()
    
    # Or with custom config file
    config = load_feature_extraction_config('custom_config.ini')

Key Features:
    • Centralized configuration management for feature extraction pipeline.
    • Automatic fallback to default values for missing parameters.
    • Type validation and conversion for all configuration parameters.
    • Support for custom configuration files and parameter overrides.
    • Integration with existing project configuration patterns.

Notes:
    • All parameter names follow lowercase conventions for consistency.
    • Configuration files use INI format with detailed comments.
    • Provides sensible defaults optimized for kidney I/R injury analysis.
    • Supports both absolute and relative paths for configuration files.
"""
import traceback
import configparser
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_tuple(config: configparser.ConfigParser, section: str, option: str, 
              default: Tuple, cast: type = float) -> Tuple:
    """
    Parse tuple values from configuration file.
    
    Args:
        config: ConfigParser instance.
        section: Configuration section name.
        option: Configuration option name.
        default: Default tuple value.
        cast: Type to cast tuple elements to.
        
    Returns:
        Parsed tuple with specified type casting.
        
    This function handles comma-separated values in configuration files
    and converts them to tuples with proper type casting.
    """
    try:
        value_str = config.get(section, option, fallback=None)
        if value_str is None:
            return default
        
        # Parse comma-separated values.
        values = [cast(v.strip()) for v in value_str.split(',')]
        return tuple(values)
    except Exception as e:
        logger.warning(f"Failed to parse tuple {section}.{option}: {e}, using default")
        return default


def get_list(config: configparser.ConfigParser, section: str, option: str, 
             default: List, cast: type = str) -> List:
    """
    Parse list values from configuration file.
    
    Args:
        config: ConfigParser instance.
        section: Configuration section name.
        option: Configuration option name.
        default: Default list value.
        cast: Type to cast list elements to.
        
    Returns:
        Parsed list with specified type casting.
        
    This function handles comma-separated values in configuration files
    and converts them to lists with proper type casting.
    """
    try:
        value_str = config.get(section, option, fallback=None)
        if value_str is None or not value_str.strip():
            return default
        
        # Parse comma-separated values, filtering empty strings.
        values = [cast(v.strip()) for v in value_str.split(',') if v.strip()]
        return values
    except Exception as e:
        logger.warning(f"Failed to parse list {section}.{option}: {e}, using default")
        return default


def load_feature_extraction_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load engineered feature extraction configuration from INI file.
    
    Args:
        config_path: Optional path to configuration file. If None, uses default location.
        
    Returns:
        Dictionary containing all configuration parameters with proper type conversion.
        
    This function loads the comprehensive configuration for nuclear feature extraction,
    clustering analysis, and visualization, providing sensible defaults for all parameters.
    """
    # Determine configuration file path.
    if config_path is None:
        # Use default configuration file in project configs directory.
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / 'configs' / 'engineered_feature_extraction_config.ini'
    else:
        config_path = Path(config_path)
    
    # Initialize configuration parser.
    config = configparser.ConfigParser()
    
    # Load configuration file if it exists.
    if config_path.exists():
        try:
            config.read(config_path)
            logger.info(f"Loaded feature extraction config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration parameters")
    else:
        logger.info(f"Config file not found at {config_path}, using defaults")
    
    # Build simplified configuration dictionary with only used parameters.
    settings = {
        # ─── General Settings ─────────────────────────────────────────────────
        "enable_feature_extraction": config.getboolean("general", "enable_feature_extraction", fallback=True),
        "enable_clustering": config.getboolean("general", "enable_clustering", fallback=True),
        "enable_visualizations": config.getboolean("general", "enable_visualizations", fallback=True),

        # ─── Feature Extraction Parameters ────────────────────────────────────
        "neighborhood_radius": config.getfloat("feature_extraction", "neighborhood_radius", fallback=20.0),
        "extract_texture_features": config.getboolean("feature_extraction", "extract_texture_features", fallback=False),

        # ─── Input/Output Paths ───────────────────────────────────────────────
        "extraction_image_path": config.get("feature_extraction", "extraction_image_path", fallback=""),
        "extraction_mask_path": config.get("feature_extraction", "extraction_mask_path", fallback=""),
        "extraction_output_dir": config.get("feature_extraction", "extraction_output_dir", fallback="results/simple_features"),

        # ─── Clustering Input/Output Paths ───────────────────────────────────
        "features_csv_path": config.get("clustering", "features_csv_path", fallback=""),
        "image_path": config.get("clustering", "image_path", fallback=""),
        "mask_path": config.get("clustering", "mask_path", fallback=""),
        "clustering_output_dir": config.get("clustering", "clustering_output_dir", fallback=""),

        # ─── Clustering Algorithm Parameters ──────────────────────────────────
        "default_clusters": config.getint("clustering", "default_clusters", fallback=8),
        "auto_k_method": config.get("clustering", "auto_k_method", fallback="None"),
        "max_clusters_test": config.getint("clustering", "max_clusters_test", fallback=25),
        "clustering_seed": config.getint("clustering", "clustering_seed", fallback=42),
        "clustering_batch_size": config.getint("clustering", "clustering_batch_size", fallback=5000),

        # ─── Visualization Parameters ─────────────────────────────────────────
        "generate_cluster_overlay": config.getboolean("clustering", "generate_cluster_overlay", fallback=True),
        "generate_pca_plot": config.getboolean("clustering", "generate_pca_plot", fallback=True),
        "generate_feature_importance": config.getboolean("clustering", "generate_feature_importance", fallback=True),
        "pca_sample_size": config.getint("clustering", "pca_sample_size", fallback=5000),
        "overlay_crop_region": get_tuple(config, "clustering", "overlay_crop_region", default=(0.1, 0.9, 0.1, 0.9), cast=float),
        "overlay_downsample_factor": config.getint("clustering", "overlay_downsample_factor", fallback=1),
        "overlay_tile_size": config.getint("clustering", "overlay_tile_size", fallback=1024),
        "overlay_workers": config.get("clustering", "overlay_workers", fallback="auto"),
        "overlay_alpha": config.getfloat("clustering", "overlay_alpha", fallback=0.85),
        "overlay_gpu": config.getboolean("clustering", "overlay_gpu", fallback=True),
        "overlay_memory_limit_mb": config.getint("clustering", "overlay_memory_limit_mb", fallback=8192),

        # ─── Color Configuration ──────────────────────────────────────────────
        "color_background": config.get("clustering", "color_background", fallback="dark"),
        "color_alpha": config.getint("clustering", "color_alpha", fallback=250),
        "color_saturation": config.getfloat("clustering", "color_saturation", fallback=0.98),
        "color_contrast_ratio": config.getfloat("clustering", "color_contrast_ratio", fallback=4.5),
        "color_hue_start": config.getfloat("clustering", "color_hue_start", fallback=0.0),
        "custom_colors": get_list(config, "clustering", "custom_colors", default=["#FF0000", "#00FF00", "#0080FF", "#FF00FF", "#FF8C00"], cast=str),

        # ─── Output Parameters ────────────────────────────────────────────────
        "save_cluster_statistics": config.getboolean("clustering", "save_cluster_statistics", fallback=True),
        "save_clustering_model": config.getboolean("clustering", "save_clustering_model", fallback=True),
    }
    
    # Validate critical parameters.
    if settings["default_clusters"] < 2:
        logger.warning("default_clusters must be >= 2, setting to 2")
        settings["default_clusters"] = 2

    if settings["max_clusters_test"] < settings["default_clusters"]:
        logger.warning("max_clusters_test must be >= default_clusters, adjusting")
        settings["max_clusters_test"] = max(settings["default_clusters"], 25)

    if not (0.0 <= settings["color_saturation"] <= 1.0):
        logger.warning("color_saturation must be 0.0-1.0, setting to 0.98")
        settings["color_saturation"] = 0.98

    if not (0 <= settings["color_alpha"] <= 255):
        logger.warning("color_alpha must be 0-255, setting to 250")
        settings["color_alpha"] = 250

        # Also enable legacy category flags for backward compatibility.
        settings["shape_features"] = True
        settings["size_features"] = True
        settings["neighborhood_features"] = True
        settings["texture_features"] = True

    logger.info(f"Loaded {len(settings)} configuration parameters")

    return settings


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters for consistency and correctness.
    
    Args:
        config: Configuration dictionary to validate.
        
    Returns:
        True if configuration is valid, False otherwise.
        
    This function performs comprehensive validation of configuration parameters
    to ensure they are within valid ranges and consistent with each other.
    """
    try:
        # Validate clustering parameters.
        if config["auto_k_method"] not in ["none", "silhouette", "dbi"]:
            logger.error(f"Invalid auto_k_method: {config['auto_k_method']}")
            return False
        
        if config["default_clusters"] < 2 or config["default_clusters"] > 100:
            logger.error(f"default_clusters out of range: {config['default_clusters']}")
            return False
        
        if config["max_clusters_test"] < config["default_clusters"]:
            logger.error("max_clusters_test must be >= default_clusters")
            return False
        
        # Validate visualization parameters.
        if config["figure_format"] not in ["png", "pdf", "svg", "tiff"]:
            logger.error(f"Invalid figure_format: {config['figure_format']}")
            return False
        
        if config["figure_dpi"] < 50 or config["figure_dpi"] > 1200:
            logger.error(f"figure_dpi out of range: {config['figure_dpi']}")
            return False
        
        # Validate performance parameters.
        if config["max_memory_gb"] < 1.0 or config["max_memory_gb"] > 1000.0:
            logger.error(f"max_memory_gb out of range: {config['max_memory_gb']}")
            return False
        
        if config["log_level"] not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            logger.error(f"Invalid log_level: {config['log_level']}")
            return False
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def get_enabled_features(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Get lists of enabled features by category from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary with lists of enabled features by category.

    This function analyzes the individual feature flags and returns organized
    lists of which features are enabled for extraction.
    """
    enabled_features = {
        'shape': [],
        'size': [],
        'neighborhood': [],
        'texture': []
    }

    # Shape features.
    shape_feature_map = {
        'extract_circularity': 'circularity',
        'extract_eccentricity': 'eccentricity',
        'extract_solidity': 'solidity',
        'extract_aspect_ratio': 'aspect_ratio',
        'extract_compactness': 'compactness',
        'extract_elongation': 'elongation',
        'extract_roundness': 'roundness',
        'extract_form_factor': 'form_factor',
        'extract_convex_area_ratio': 'convex_area_ratio',
        'extract_convexity': 'convexity',
        'extract_fractal_dimension': 'fractal_dimension'
    }

    # Size features.
    size_feature_map = {
        'extract_area': 'area',
        'extract_perimeter': 'perimeter',
        'extract_equivalent_diameter': 'equivalent_diameter',
        'extract_major_axis_length': 'major_axis_length',
        'extract_minor_axis_length': 'minor_axis_length',
        'extract_bounding_box_width': 'bounding_box_width',
        'extract_bounding_box_height': 'bounding_box_height',
        'extract_bounding_box_area': 'bounding_box_area',
        'extract_feret_diameter_max': 'feret_diameter_max',
        'extract_feret_diameter_min': 'feret_diameter_min'
    }

    # Neighborhood features.
    neighborhood_feature_map = {
        'extract_nearest_neighbor_distance': 'nearest_neighbor_distance',
        'extract_neighborhood_density': 'neighborhood_density',
        'extract_boundary_proximity': 'boundary_proximity',
        'extract_cluster_elongation': 'cluster_elongation',
        'extract_cluster_polarization': 'cluster_polarization',
        'extract_spatial_autocorrelation': 'spatial_autocorrelation',
        'extract_tissue_organization_index': 'tissue_organization_index',
        'extract_local_clustering_coefficient': 'local_clustering_coefficient'
    }

    # Texture features.
    texture_feature_map = {
        'extract_intensity_mean': 'intensity_mean',
        'extract_intensity_std': 'intensity_std',
        'extract_intensity_median': 'intensity_median',
        'extract_intensity_skewness': 'intensity_skewness',
        'extract_intensity_kurtosis': 'intensity_kurtosis',
        'extract_texture_entropy': 'texture_entropy',
        'extract_gradient_magnitude_mean': 'gradient_magnitude_mean',
        'extract_gradient_magnitude_std': 'gradient_magnitude_std',
        'extract_glcm_contrast': 'glcm_contrast',
        'extract_glcm_dissimilarity': 'glcm_dissimilarity',
        'extract_glcm_homogeneity': 'glcm_homogeneity',
        'extract_glcm_energy': 'glcm_energy'
    }

    # Check which features are enabled.
    for config_key, feature_name in shape_feature_map.items():
        if config.get(config_key, False):
            enabled_features['shape'].append(feature_name)

    for config_key, feature_name in size_feature_map.items():
        if config.get(config_key, False):
            enabled_features['size'].append(feature_name)

    for config_key, feature_name in neighborhood_feature_map.items():
        if config.get(config_key, False):
            enabled_features['neighborhood'].append(feature_name)

    for config_key, feature_name in texture_feature_map.items():
        if config.get(config_key, False):
            enabled_features['texture'].append(feature_name)

    return enabled_features


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration parameters for feature extraction and clustering.

    Returns:
        Dictionary with default configuration parameters.

    This function provides a complete set of default parameters that can be used
    when no configuration file is available or as a fallback for missing parameters.
    """
    return load_feature_extraction_config(config_path=None)


if __name__ == "__main__":
    # Test configuration loading.
    print("Testing feature extraction configuration loader...")
    
    try:
        # Load default configuration.
        config = load_feature_extraction_config()
        print(f"✓ Loaded {len(config)} configuration parameters")
        
        # Validate configuration.
        is_valid = validate_config(config)
        print(f"✓ Configuration validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Display key parameters.
        print("\nKey Configuration Parameters:")
        key_params = [
            "enable_clustering", "default_clusters", "auto_k_method",
            "clustering_batch_size", "color_background", "figure_dpi"
        ]
        
        for param in key_params:
            if param in config:
                print(f"  {param}: {config[param]}")
        
        print("\n✓ Configuration loader test completed successfully")
        
    except Exception as e:
        print(f"✗ Configuration loader test failed: {e}")
        traceback.print_exc()
