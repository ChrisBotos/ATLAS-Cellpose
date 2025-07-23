#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: results_locator.py.
Description:
    Utility functions for locating and validating segmentation results directories.
    Designed to make server job scripts more robust and reliable.

Dependencies:
    • Python >= 3.7.
    • pathlib for cross-platform path handling.

Usage:
    from utils.results_locator import find_latest_results, validate_results_directory

Key Features:
    • Robust results directory discovery with multiple fallback strategies.
    • Comprehensive validation of required output files.
    • Support for both symlink and text file-based latest results tracking.
    • Detailed error reporting for debugging server job issues.

Notes:
    • This module is designed specifically for server job script integration.
    • It provides multiple strategies for finding results to handle various scenarios.
"""

import traceback
import re
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime


def find_latest_results(base_dir: Path, pattern: str = None) -> Optional[Path]:
    """
    Find the most recent results directory using multiple strategies.
    
    Args:
        base_dir (Path): Base results directory to search in.
        pattern (str, optional): Regex pattern to match directory names.
        
    Returns:
        Optional[Path]: Path to the latest results directory, or None if not found.
    """
    try:
        # Strategy 1: Check for 'latest' symlink.
        latest_link = base_dir / "latest"
        if latest_link.exists() and latest_link.is_symlink():
            target = latest_link.resolve()
            if target.exists() and target.is_dir():
                print(f"Found latest results via symlink: {target}")
                return target
        
        # Strategy 2: Check for 'latest.txt' file.
        latest_txt = base_dir / "latest.txt"
        if latest_txt.exists():
            try:
                with open(latest_txt, 'r') as f:
                    dir_name = f.read().strip()
                target = base_dir / dir_name
                if target.exists() and target.is_dir():
                    print(f"Found latest results via latest.txt: {target}")
                    return target
            except Exception as e:
                print(f"Warning: Could not read latest.txt: {e}")
        
        # Strategy 3: Find most recent timestamped directory.
        if not pattern:
            # Default pattern for timestamped directories.
            pattern = r"^\d{8}_\d{6}_.*"
        
        matching_dirs = []
        for item in base_dir.iterdir():
            if item.is_dir() and re.match(pattern, item.name):
                matching_dirs.append(item)
        
        if matching_dirs:
            # Sort by modification time (most recent first).
            latest = max(matching_dirs, key=lambda p: p.stat().st_mtime)
            print(f"Found latest results via timestamp search: {latest}")
            return latest
        
        print(f"No results directories found in {base_dir}")
        return None
        
    except Exception as e:
        print(f"Error finding latest results: {e}")
        print(traceback.format_exc())
        return None


def validate_results_directory(results_dir: Path, required_files: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate that a results directory contains all required output files.
    
    Args:
        results_dir (Path): Path to the results directory to validate.
        required_files (List[str], optional): List of required file paths relative to results_dir.
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_missing_files).
    """
    if required_files is None:
        required_files = [
            "masks/segmentation_masks.npy",
            "masks/segmentation_masks.tif",
            "settings_snapshot.json"
        ]
    
    missing_files = []
    
    try:
        if not results_dir.exists():
            return False, [f"Results directory does not exist: {results_dir}"]
        
        if not results_dir.is_dir():
            return False, [f"Results path is not a directory: {results_dir}"]
        
        for required_file in required_files:
            file_path = results_dir / required_file
            if not file_path.exists():
                missing_files.append(required_file)
        
        is_valid = len(missing_files) == 0
        return is_valid, missing_files
        
    except Exception as e:
        return False, [f"Error validating results directory: {e}"]


def find_segmentation_mask(results_dir: Path) -> Optional[Path]:
    """
    Find the segmentation mask file in a results directory.
    
    Args:
        results_dir (Path): Path to the results directory.
        
    Returns:
        Optional[Path]: Path to the segmentation mask file, or None if not found.
    """
    try:
        # Primary location: masks/segmentation_masks.npy.
        primary_path = results_dir / "masks" / "segmentation_masks.npy"
        if primary_path.exists():
            return primary_path
        
        # Fallback: search for any .npy file containing "segmentation" in masks directory.
        masks_dir = results_dir / "masks"
        if masks_dir.exists():
            for npy_file in masks_dir.glob("*segmentation*.npy"):
                print(f"Found fallback segmentation mask: {npy_file}")
                return npy_file
        
        # Last resort: search entire results directory.
        for npy_file in results_dir.rglob("*segmentation*.npy"):
            print(f"Found segmentation mask in subdirectory: {npy_file}")
            return npy_file
        
        return None
        
    except Exception as e:
        print(f"Error finding segmentation mask: {e}")
        return None


def create_results_summary(results_dir: Path) -> dict:
    """
    Create a summary of results directory contents for debugging.
    
    Args:
        results_dir (Path): Path to the results directory.
        
    Returns:
        dict: Summary information about the results directory.
    """
    summary = {
        "directory": str(results_dir),
        "exists": results_dir.exists() if results_dir else False,
        "is_directory": results_dir.is_dir() if results_dir and results_dir.exists() else False,
        "files": [],
        "subdirectories": [],
        "mask_files": [],
        "total_size_mb": 0
    }
    
    try:
        if not results_dir or not results_dir.exists():
            return summary
        
        total_size = 0
        for item in results_dir.rglob("*"):
            if item.is_file():
                size = item.stat().st_size
                total_size += size
                summary["files"].append({
                    "path": str(item.relative_to(results_dir)),
                    "size_mb": round(size / (1024 * 1024), 2)
                })
                
                # Track mask files specifically.
                if item.suffix == ".npy" and "mask" in item.name.lower():
                    summary["mask_files"].append(str(item.relative_to(results_dir)))
            elif item.is_dir() and item != results_dir:
                summary["subdirectories"].append(str(item.relative_to(results_dir)))
        
        summary["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
    except Exception as e:
        summary["error"] = str(e)
    
    return summary


if __name__ == "__main__":
    """Command-line interface for testing results location functionality."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python results_locator.py <results_base_directory> [pattern]")
        sys.exit(1)
    
    base_dir = Path(sys.argv[1])
    pattern = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Searching for latest results in: {base_dir}")
    if pattern:
        print(f"Using pattern: {pattern}")
    
    latest = find_latest_results(base_dir, pattern)
    if latest:
        print(f"\nLatest results directory: {latest}")
        
        # Validate the results.
        is_valid, missing = validate_results_directory(latest)
        print(f"Results validation: {'PASS' if is_valid else 'FAIL'}")
        if missing:
            print("Missing files:")
            for file in missing:
                print(f"  - {file}")
        
        # Find segmentation mask.
        mask_path = find_segmentation_mask(latest)
        if mask_path:
            print(f"Segmentation mask: {mask_path}")
        else:
            print("Segmentation mask: NOT FOUND")
        
        # Create summary.
        summary = create_results_summary(latest)
        print(f"\nResults summary:")
        print(f"  Total files: {len(summary['files'])}")
        print(f"  Total size: {summary['total_size_mb']} MB")
        print(f"  Mask files: {len(summary['mask_files'])}")
        
    else:
        print("No results directory found.")
        sys.exit(1)
