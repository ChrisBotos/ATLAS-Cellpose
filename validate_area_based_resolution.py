"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: validate_area_based_resolution.py.
Description:
    Validation script to check if area-based conflict resolution is working
    on real segmentation data. Analyzes overlap data and individual nucleus masks
    to verify that smaller nuclei are winning conflicts with larger nuclei.

Dependencies:
    • Python ≥ 3.10.
    • numpy for array operations.
    • pathlib for file handling.

Usage:
    python validate_area_based_resolution.py

Key Features:
    • Loads overlap data from recent segmentation runs.
    • Analyzes individual nucleus masks for true overlaps.
    • Validates area-based conflict resolution results.
    • Reports statistics on overlapping nuclei and conflict resolution.

Notes:
    • Looks for the most recent results directory.
    • Requires overlap_data.npz and individual nucleus masks to exist.
"""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import glob

# Configure logging.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def find_latest_results_dir() -> Path:
    """Find the most recent results directory."""
    results_base = Path("C:/Projects/I-R-Injury-Spatial-Multiomics-Analysis/results")

    if not results_base.exists():
        results_base = Path("results")

    if not results_base.exists():
        raise FileNotFoundError(f"Results directory not found: {results_base}")

    # Find all result directories (exclude symlinks and special files).
    result_dirs = []
    for item in results_base.iterdir():
        if item.is_dir() and not item.is_symlink() and not item.name.startswith('.'):
            result_dirs.append(item)

    if not result_dirs:
        raise FileNotFoundError(f"No results directories found in: {results_base}")

    # Sort by modification time and get the latest.
    latest_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
    logger.info(f"Using latest results directory: {latest_dir}")

    return latest_dir

def analyze_overlap_data(results_dir: Path) -> Dict:
    """Analyze overlap data from the results directory."""
    overlap_file = results_dir / "masks" / "overlap_data.npz"
    
    if not overlap_file.exists():
        logger.warning(f"Overlap data file not found: {overlap_file}")
        return {}
    
    try:
        data = np.load(overlap_file, allow_pickle=True)
        overlap_info = data['overlap_info'].item()
        merged_mask = data['merged_mask']
        
        logger.info(f"Loaded overlap data:")
        logger.info(f"  Overlapping pixels: {overlap_info.get('overlapping_pixels', 0)}")
        logger.info(f"  Max overlap depth: {overlap_info.get('max_overlap_depth', 0)}")
        logger.info(f"  Overlapping pairs: {len(overlap_info.get('overlapping_pairs', set()))}")
        
        return {
            'overlap_info': overlap_info,
            'merged_mask': merged_mask,
            'overlap_file': overlap_file
        }
        
    except Exception as e:
        logger.error(f"Failed to load overlap data: {e}")
        return {}

def analyze_individual_masks(results_dir: Path) -> Dict:
    """Analyze individual nucleus masks for true overlaps."""
    masks_dir = results_dir / "masks" / "individual_nucleus_masks"
    
    if not masks_dir.exists():
        logger.warning(f"Individual masks directory not found: {masks_dir}")
        return {}
    
    mask_files = list(masks_dir.glob("nucleus_*.npy"))
    
    if not mask_files:
        logger.warning(f"No individual mask files found in: {masks_dir}")
        return {}
    
    logger.info(f"Found {len(mask_files)} individual nucleus mask files")
    
    # Load all masks and calculate areas.
    nucleus_masks = {}
    nucleus_areas = {}
    
    for mask_file in mask_files:
        try:
            nucleus_id = int(mask_file.stem.split('_')[1])
            mask = np.load(mask_file).astype(bool)
            nucleus_masks[nucleus_id] = mask
            nucleus_areas[nucleus_id] = np.sum(mask)
        except Exception as e:
            logger.warning(f"Failed to load mask {mask_file}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(nucleus_masks)} nucleus masks")
    
    # Analyze overlaps between masks.
    overlapping_pairs = []
    total_overlap_pixels = 0
    
    nucleus_ids = sorted(nucleus_masks.keys())
    for i, id1 in enumerate(nucleus_ids):
        for id2 in nucleus_ids[i+1:]:
            overlap_mask = nucleus_masks[id1] & nucleus_masks[id2]
            overlap_count = np.sum(overlap_mask)
            
            if overlap_count > 0:
                overlapping_pairs.append({
                    'id1': id1,
                    'id2': id2,
                    'overlap_pixels': overlap_count,
                    'area1': nucleus_areas[id1],
                    'area2': nucleus_areas[id2],
                    'smaller_id': id1 if nucleus_areas[id1] < nucleus_areas[id2] else id2
                })
                total_overlap_pixels += overlap_count
    
    return {
        'nucleus_masks': nucleus_masks,
        'nucleus_areas': nucleus_areas,
        'overlapping_pairs': overlapping_pairs,
        'total_overlap_pixels': total_overlap_pixels,
        'masks_dir': masks_dir
    }

def validate_conflict_resolution(overlap_data: Dict, individual_data: Dict) -> Dict:
    """Validate that area-based conflict resolution is working correctly."""
    if not overlap_data or not individual_data:
        logger.warning("Missing data for conflict resolution validation")
        return {}
    
    merged_mask = overlap_data['merged_mask']
    overlapping_pairs = individual_data['overlapping_pairs']
    nucleus_areas = individual_data['nucleus_areas']
    
    if not overlapping_pairs:
        logger.info("No overlapping pairs found - no conflicts to validate")
        return {'conflicts_resolved_correctly': 0, 'total_conflicts': 0}
    
    logger.info(f"Validating conflict resolution for {len(overlapping_pairs)} overlapping pairs...")
    
    conflicts_resolved_correctly = 0
    total_conflicts = len(overlapping_pairs)
    
    for pair in overlapping_pairs:
        id1, id2 = pair['id1'], pair['id2']
        smaller_id = pair['smaller_id']
        overlap_pixels = pair['overlap_pixels']
        
        # Sample a few overlapping pixels to check which nucleus won.
        mask1 = individual_data['nucleus_masks'][id1]
        mask2 = individual_data['nucleus_masks'][id2]
        overlap_mask = mask1 & mask2
        
        # Get coordinates of overlapping pixels.
        overlap_coords = np.where(overlap_mask)
        
        if len(overlap_coords[0]) > 0:
            # Sample up to 10 overlapping pixels.
            sample_size = min(10, len(overlap_coords[0]))
            sample_indices = np.random.choice(len(overlap_coords[0]), sample_size, replace=False)
            
            winners = []
            for idx in sample_indices:
                y, x = overlap_coords[0][idx], overlap_coords[1][idx]
                winner_id = merged_mask[y, x]
                winners.append(winner_id)
            
            # Check if the smaller nucleus won in the majority of cases.
            smaller_wins = sum(1 for w in winners if w == smaller_id)
            if smaller_wins > sample_size // 2:  # Majority rule.
                conflicts_resolved_correctly += 1
                logger.debug(f"✅ Conflict {id1} vs {id2}: Smaller nucleus {smaller_id} won ({smaller_wins}/{sample_size} samples)")
            else:
                logger.debug(f"❌ Conflict {id1} vs {id2}: Smaller nucleus {smaller_id} lost ({smaller_wins}/{sample_size} samples)")
    
    resolution_accuracy = (conflicts_resolved_correctly / total_conflicts) * 100 if total_conflicts > 0 else 0
    
    return {
        'conflicts_resolved_correctly': conflicts_resolved_correctly,
        'total_conflicts': total_conflicts,
        'resolution_accuracy': resolution_accuracy
    }

def main():
    """Main validation function."""
    logger.info("Starting area-based conflict resolution validation...")
    
    try:
        # Find the latest results directory.
        results_dir = find_latest_results_dir()
        
        # Analyze overlap data.
        logger.info("Analyzing overlap data...")
        overlap_data = analyze_overlap_data(results_dir)
        
        # Analyze individual nucleus masks.
        logger.info("Analyzing individual nucleus masks...")
        individual_data = analyze_individual_masks(results_dir)
        
        # Validate conflict resolution.
        logger.info("Validating conflict resolution...")
        validation_results = validate_conflict_resolution(overlap_data, individual_data)
        
        # Report results.
        logger.info("VALIDATION RESULTS:")
        logger.info(f"  Results directory: {results_dir}")
        
        if overlap_data:
            logger.info(f"  Overlap data loaded: ✅")
            logger.info(f"  Overlapping pixels: {overlap_data['overlap_info'].get('overlapping_pixels', 0)}")
        else:
            logger.info(f"  Overlap data loaded: ❌")
        
        if individual_data:
            logger.info(f"  Individual masks loaded: ✅ ({len(individual_data.get('nucleus_masks', {}))} nuclei)")
            logger.info(f"  True overlapping pairs: {len(individual_data.get('overlapping_pairs', []))}")
            logger.info(f"  Total overlap pixels: {individual_data.get('total_overlap_pixels', 0)}")
        else:
            logger.info(f"  Individual masks loaded: ❌")
        
        if validation_results:
            logger.info(f"  Conflict resolution validation:")
            logger.info(f"    Correctly resolved: {validation_results['conflicts_resolved_correctly']}/{validation_results['total_conflicts']}")
            logger.info(f"    Accuracy: {validation_results['resolution_accuracy']:.1f}%")
            
            if validation_results['resolution_accuracy'] > 80:
                logger.info("  🎉 Area-based conflict resolution is working correctly!")
            elif validation_results['total_conflicts'] == 0:
                logger.info("  ⚠️  No conflicts found to validate")
            else:
                logger.info("  ❌ Area-based conflict resolution needs improvement")
        else:
            logger.info(f"  Conflict resolution validation: ❌ (insufficient data)")
        
        # Summary.
        if overlap_data and individual_data and validation_results.get('resolution_accuracy', 0) > 80:
            logger.info("\n✅ VALIDATION PASSED: Area-based conflict resolution is working!")
        elif not overlap_data or not individual_data:
            logger.info("\n⚠️  VALIDATION INCOMPLETE: Missing required data files")
        else:
            logger.info("\n❌ VALIDATION FAILED: Area-based conflict resolution not working correctly")
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")

if __name__ == "__main__":
    main()
