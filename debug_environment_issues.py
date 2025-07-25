#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: debug_environment_issues.py.
Description:
    Comprehensive diagnostic script to investigate environment issues causing
    segmentation failures, particularly with vertical tiles and parallel processing
    timeouts. This script tests PyTorch/Cellpose3 compatibility and identifies
    environment conflicts.

Dependencies:
    • Python >= 3.10.
    • torch, cellpose, numpy for environment testing.
    • psutil for system monitoring.

Usage:
    python debug_environment_issues.py

Arguments:
    None (runs comprehensive diagnostics automatically).

Inputs:
    • System environment and package versions.
    • Available hardware resources.

Outputs:
    • Detailed diagnostic report in logs/environment_diagnostic.log.
    • Recommendations for fixing environment issues.
    • Test results for PyTorch/Cellpose3 compatibility.

Key Features:
    • Comprehensive environment analysis.
    • PyTorch CUDA compatibility testing.
    • Cellpose3 vs Cellpose4 comparison.
    • Memory and threading diagnostics.
    • Parallel processing stress testing.

Notes:
    • Run this script when experiencing segmentation timeouts or failures.
    • Pay special attention to PyTorch/Cellpose version conflicts.
    • Use results to optimize parallel processing configuration.
"""

import traceback
import sys
import os
import logging
import time
import gc
from pathlib import Path
from datetime import datetime
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import multiprocessing as mp

import numpy as np

# Set up logging.
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"environment_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_basic_imports():
    """Test basic package imports and versions."""
    logger.info("=" * 60)
    logger.info("TESTING BASIC PACKAGE IMPORTS")
    logger.info("=" * 60)
    
    # Test Python version.
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    
    # Test critical imports.
    packages_to_test = [
        'numpy', 'torch', 'cellpose', 'PIL', 'scipy', 'skimage',
        'matplotlib', 'tqdm', 'psutil', 'concurrent.futures'
    ]
    
    import_results = {}
    
    for package in packages_to_test:
        try:
            if package == 'torch':
                import torch
                logger.info(f"✓ PyTorch version: {torch.__version__}")
                logger.info(f"  CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"  CUDA version: {torch.version.cuda}")
                    logger.info(f"  GPU count: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                import_results[package] = True
                
            elif package == 'cellpose':
                import cellpose
                from cellpose import models
                logger.info(f"✓ Cellpose version: {cellpose.__version__}")
                
                # Test both Cellpose3 and Cellpose4 APIs.
                try:
                    # Try Cellpose3 API.
                    model3 = models.Cellpose(model_type='nuclei', gpu=False)
                    logger.info("✓ Cellpose3 API available")
                except AttributeError:
                    logger.warning("✗ Cellpose3 API not available")
                
                try:
                    # Try Cellpose4 API.
                    model4 = models.CellposeModel(model_type='nuclei', gpu=False)
                    logger.info("✓ Cellpose4 API available")
                except AttributeError:
                    logger.warning("✗ Cellpose4 API not available")
                    
                import_results[package] = True
                
            elif package == 'psutil':
                import psutil
                logger.info(f"✓ psutil version: {psutil.__version__}")
                logger.info(f"  CPU count: {psutil.cpu_count()}")
                logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
                import_results[package] = True
                
            else:
                exec(f"import {package}")
                logger.info(f"✓ {package} imported successfully")
                import_results[package] = True
                
        except ImportError as e:
            logger.error(f"✗ Failed to import {package}: {e}")
            import_results[package] = False
        except Exception as e:
            logger.error(f"✗ Error testing {package}: {e}")
            import_results[package] = False
    
    return import_results


def test_pytorch_cellpose_compatibility():
    """Test PyTorch and Cellpose compatibility."""
    logger.info("=" * 60)
    logger.info("TESTING PYTORCH/CELLPOSE COMPATIBILITY")
    logger.info("=" * 60)
    
    try:
        import torch
        from cellpose import models
        
        # Test CPU model creation.
        logger.info("Testing CPU model creation...")
        try:
            model_cpu = models.Cellpose(model_type='nuclei', gpu=False)
            logger.info("✓ CPU Cellpose model created successfully")
        except Exception as e:
            logger.error(f"✗ CPU model creation failed: {e}")
            return False
        
        # Test GPU model creation if available.
        if torch.cuda.is_available():
            logger.info("Testing GPU model creation...")
            try:
                model_gpu = models.Cellpose(model_type='nuclei', gpu=True)
                logger.info("✓ GPU Cellpose model created successfully")
            except Exception as e:
                logger.error(f"✗ GPU model creation failed: {e}")
                logger.info("This might indicate PyTorch/Cellpose version conflicts")
        
        # Test basic segmentation on dummy data.
        logger.info("Testing basic segmentation...")
        dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        try:
            masks, flows, styles, diams = model_cpu.eval(dummy_image, diameter=None, channels=[0,0])
            logger.info(f"✓ Basic segmentation successful, detected {len(np.unique(masks))-1} objects")
        except Exception as e:
            logger.error(f"✗ Basic segmentation failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ PyTorch/Cellpose compatibility test failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False


def test_parallel_processing():
    """Test parallel processing with threading and multiprocessing."""
    logger.info("=" * 60)
    logger.info("TESTING PARALLEL PROCESSING")
    logger.info("=" * 60)
    
    def dummy_task(task_id, duration=1):
        """Dummy task that simulates Cellpose processing."""
        time.sleep(duration)
        return f"Task {task_id} completed"
    
    # Test threading.
    logger.info("Testing ThreadPoolExecutor...")
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(dummy_task, i, 0.5) for i in range(8)]
            results = [f.result(timeout=10) for f in futures]
        logger.info(f"✓ Threading test successful: {len(results)} tasks completed")
    except Exception as e:
        logger.error(f"✗ Threading test failed: {e}")
    
    # Test multiprocessing.
    logger.info("Testing multiprocessing...")
    try:
        with mp.Pool(processes=4) as pool:
            results = pool.starmap(dummy_task, [(i, 0.5) for i in range(8)])
        logger.info(f"✓ Multiprocessing test successful: {len(results)} tasks completed")
    except Exception as e:
        logger.error(f"✗ Multiprocessing test failed: {e}")


def test_memory_usage():
    """Test memory usage patterns."""
    logger.info("=" * 60)
    logger.info("TESTING MEMORY USAGE PATTERNS")
    logger.info("=" * 60)
    
    try:
        import psutil
        
        # Get initial memory.
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Create large arrays to simulate tile processing.
        arrays = []
        for i in range(10):
            # Simulate 512x512 tiles.
            array = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
            arrays.append(array)
            
            current_memory = process.memory_info().rss / (1024**2)
            logger.info(f"After array {i+1}: {current_memory:.1f} MB (+{current_memory-initial_memory:.1f} MB)")
        
        # Test garbage collection.
        del arrays
        gc.collect()
        
        final_memory = process.memory_info().rss / (1024**2)
        logger.info(f"After cleanup: {final_memory:.1f} MB")
        
        if final_memory - initial_memory > 100:
            logger.warning("Memory usage increased significantly after cleanup - possible memory leak")
        else:
            logger.info("✓ Memory cleanup successful")
            
    except Exception as e:
        logger.error(f"✗ Memory usage test failed: {e}")


def main():
    """Run comprehensive environment diagnostics."""
    logger.info("Starting comprehensive environment diagnostics...")
    logger.info(f"Log file: {log_file}")
    
    try:
        # Test basic imports.
        import_results = test_basic_imports()
        
        # Test PyTorch/Cellpose compatibility.
        compatibility_ok = test_pytorch_cellpose_compatibility()
        
        # Test parallel processing.
        test_parallel_processing()
        
        # Test memory usage.
        test_memory_usage()
        
        # Summary.
        logger.info("=" * 60)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("=" * 60)
        
        failed_imports = [pkg for pkg, success in import_results.items() if not success]
        if failed_imports:
            logger.error(f"Failed imports: {failed_imports}")
        else:
            logger.info("✓ All package imports successful")
        
        if compatibility_ok:
            logger.info("✓ PyTorch/Cellpose compatibility OK")
        else:
            logger.error("✗ PyTorch/Cellpose compatibility issues detected")
        
        # Recommendations.
        logger.info("=" * 60)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 60)
        
        if not compatibility_ok:
            logger.info("1. Check PyTorch and Cellpose version compatibility")
            logger.info("2. Consider reinstalling packages in clean environment")
            logger.info("3. Test with different Cellpose versions (3 vs 4)")
        
        if failed_imports:
            logger.info("4. Install missing packages")
            logger.info("5. Check conda environment activation")
        
        logger.info("6. Consider reducing parallel batch size if timeouts occur")
        logger.info("7. Monitor memory usage during processing")
        
        logger.info("Diagnostics completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
