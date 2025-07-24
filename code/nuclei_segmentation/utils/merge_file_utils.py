"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: merge_file_utils.py.
Description:
    File handling utilities for tile mask merging operations in tissue analysis.
    This module provides robust file discovery, path resolution, and filename parsing
    capabilities for various tile mask formats (TIFF, NPZ) used in spatial multiomics
    segmentation workflows.

    Key Features:
    - Automatic detection of tile naming conventions (row_col, pixelY pixelX).
    - Robust path resolution with fallback strategies for common directory variations.
    - Support for both TIFF and NPZ tile formats.
    - Enhanced error handling with detailed diagnostic messages.

Dependencies:
    • Python ≥ 3.10.
    • pathlib, re for file operations and pattern matching.

Usage:
    from utils.merge_file_utils import resolve_tiles_path, discover_tiles
    
    path = resolve_tiles_path(Path("tile_masks"))
    file_map, coords = discover_tiles(path)

Arguments:
    None (utility functions only).

Inputs:
    - Tile mask directories containing TIFF or NPZ files.
    - Various naming conventions for tile files.

Outputs:
    - Resolved paths to tile directories.
    - Mappings from tile coordinates to file paths.
    - Lists of discovered tile coordinates.

Key Features:
    - Robust file discovery with multiple naming convention support.
    - Intelligent path resolution with fallback strategies.
    - Comprehensive error handling for missing or malformed files.

Notes:
    - Supports both underscore and space delimited tile coordinates.
    - Handles edge cases like missing directories or empty tile sets.
    - Provides detailed diagnostic information for troubleshooting.
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import logging

# Accepts either an underscore or a space delimiter between the two integers.
_PARSER = re.compile(r"^(\d+)[ _](\d+)\.(?:tif|np[sz])$", re.IGNORECASE)


def parse_tile_filename(name: str) -> Optional[Tuple[int, int]]:
    """
    Parse tile filename to extract coordinate information.
    
    This function extracts row and column indices from tile filenames that follow
    standard naming conventions used in spatial segmentation workflows.
    
    Parameters
    ----------
    name : str
        Filename to parse (e.g., "10_15.tif", "100 200.npz").
        
    Returns
    -------
    Optional[Tuple[int, int]]
        Tuple of (row, col) coordinates if parsing succeeds, None otherwise.
        
    Examples
    --------
    >>> parse_tile_filename("10_15.tif")
    (10, 15)
    >>> parse_tile_filename("100 200.npz")
    (100, 200)
    >>> parse_tile_filename("invalid.tif")
    None
    """
    try:
        match = _PARSER.match(name)
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            logging.debug(f"Parsed tile filename '{name}' -> ({row}, {col})")
            return (row, col)
        else:
            logging.debug(f"Failed to parse tile filename: '{name}'")
            return None
    except Exception as e:
        logging.warning(f"Error parsing tile filename '{name}': {e}")
        return None


def dir_contains_tiles(path: Path) -> bool:
    """
    Check if directory contains any files that look like tile masks.
    
    This function performs a quick scan to determine if a directory contains
    files matching expected tile naming patterns, useful for path validation.
    
    Parameters
    ----------
    path : Path
        Directory path to check.
        
    Returns
    -------
    bool
        True if directory contains at least one tile-like file, False otherwise.
    """
    try:
        if not path.exists() or not path.is_dir():
            return False
            
        for file_path in path.iterdir():
            if file_path.is_file() and parse_tile_filename(file_path.name):
                logging.debug(f"Found tile file in {path}: {file_path.name}")
                return True
                
        logging.debug(f"No tile files found in directory: {path}")
        return False
        
    except Exception as e:
        logging.warning(f"Error checking directory {path} for tiles: {e}")
        return False


def resolve_tiles_path(path: Path) -> Path:
    """
    Find a directory that actually contains tile masks with intelligent fallback.
    
    This function tries multiple variations of the user-supplied path to handle
    common naming inconsistencies (e.g., "tile_masks" vs "tile_masks_npz").
    It provides robust path resolution for kidney tissue analysis workflows.
    
    Parameters
    ----------
    path : Path
        Initial path to tile directory (may not exist or be empty).
        
    Returns
    -------
    Path
        Resolved path to directory containing tile masks.
        
    Raises
    ------
    FileNotFoundError
        If no valid tile directory can be found after trying all candidates.
        
    Examples
    --------
    >>> resolve_tiles_path(Path("tile_masks_npz"))
    Path("tile_masks")  # If tile_masks exists and contains tiles
    """
    try:
        candidates: List[Path] = [path.expanduser()]
        
        # Common suffix confusion: "…/_npz" vs plain.
        # Handle edge case where path.name is empty (e.g., current directory ".").
        if path.name and path.name.endswith("_npz"):
            candidates.append(path.with_name(path.name.removesuffix("_npz")))
        elif path.name:
            candidates.append(path.with_name(f"{path.name}_npz"))
        
        # Additional common variations for kidney tissue analysis.
        if path.name:
            base_name = path.name.replace("_npz", "").replace("_tif", "")
            candidates.extend([
                path.with_name(f"{base_name}_tif"),
                path.with_name(f"{base_name}_masks"),
                path.with_name(f"{base_name}_segmentation"),
            ])
        
        logging.info(f"Searching for tile directory among {len(candidates)} candidates")
        
        for cand in candidates:
            if cand.exists() and cand.is_dir() and dir_contains_tiles(cand):
                logging.info(f"Found valid tile directory: {cand}")
                return cand
        
        # Provide detailed diagnostic information.
        tried = "\n    • " + "\n    • ".join(str(c) for c in candidates)
        error_msg = (
            f"None of the following directories contains tile masks:{tried}\n"
            f"Please check the --tiles-path argument or rerun inference.\n"
            f"Expected files matching pattern: <row>_<col>.tif or <row>_<col>.npz"
        )
        
        logging.error(f"Tile directory resolution failed: {error_msg}")
        raise FileNotFoundError(error_msg)
        
    except Exception as e:
        logging.error(f"Error resolving tiles path {path}: {e}")
        logging.debug(f"Path resolution error traceback:\n{traceback.format_exc()}")
        raise


def discover_tiles(path: Path) -> Tuple[Dict[Tuple[int, int], Path], List[Tuple[int, int]]]:
    """
    Map (raw_row, raw_col) coordinates to file paths and return coordinate list.
    
    This function scans a directory for tile mask files and creates a mapping
    from tile coordinates to file paths. It supports both TIFF and NPZ formats
    commonly used in spatial multiomics analysis.
    
    Parameters
    ----------
    path : Path
        Directory containing tile mask files.
        
    Returns
    -------
    Tuple[Dict[Tuple[int, int], Path], List[Tuple[int, int]]]
        - Dictionary mapping (row, col) coordinates to file paths.
        - List of all discovered tile coordinates.
        
    Raises
    ------
    FileNotFoundError
        If directory exists but contains no valid tile files.
        
    Examples
    --------
    >>> file_map, coords = discover_tiles(Path("tile_masks"))
    >>> print(f"Found {len(coords)} tiles")
    >>> print(f"First tile: {coords[0]} -> {file_map[coords[0]]}")
    """
    try:
        mapping: Dict[Tuple[int, int], Path] = {}
        
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Tile directory does not exist: {path}")
        
        logging.info(f"Discovering tiles in directory: {path}")
        
        for file_path in path.iterdir():
            if not file_path.is_file():
                continue
                
            parsed = parse_tile_filename(file_path.name)
            if parsed is None:
                logging.debug(f"Skipping non-tile file: {file_path.name}")
                continue
                
            mapping[parsed] = file_path
            logging.debug(f"Discovered tile: {parsed} -> {file_path.name}")
        
        if not mapping:
            error_msg = (
                f"Directory '{path}' exists but contains no <row>_<col>.tif or *.npz tiles.\n"
                f"Expected naming patterns: '10_15.tif', '100_200.npz', '10 15.tif', etc."
            )
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        coords = list(mapping.keys())
        logging.info(f"Successfully discovered {len(coords)} tile files")
        
        # Log coordinate range for debugging.
        if coords:
            min_r = min(r for r, _ in coords)
            max_r = max(r for r, _ in coords)
            min_c = min(c for _, c in coords)
            max_c = max(c for _, c in coords)
            logging.info(f"Tile coordinate range: rows {min_r}-{max_r}, cols {min_c}-{max_c}")
        
        return mapping, coords
        
    except Exception as e:
        logging.error(f"Error discovering tiles in {path}: {e}")
        logging.debug(f"Tile discovery error traceback:\n{traceback.format_exc()}")
        raise
