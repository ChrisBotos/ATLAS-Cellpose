#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: check_gpu_support.py.
Description:
    Comprehensive GPU support verification script for the I/R injury spatial multiomics
    analysis pipeline. This script checks PyTorch CUDA support, CuPy availability,
    and Cellpose GPU compatibility to ensure optimal performance for large kidney
    tissue image processing.

Dependencies:
    • Python >= 3.10.
    • torch >= 2.0.0 (with CUDA support).
    • cupy >= 12.0.0 (for GPU-accelerated array operations).
    • cellpose >= 3.0.0 (for GPU-accelerated segmentation).

Usage:
    python check_gpu_support.py

Key Features:
    • Verifies CUDA runtime availability and version.
    • Tests PyTorch GPU tensor operations.
    • Validates CuPy GPU array processing.
    • Checks Cellpose GPU model loading.
    • Reports GPU memory availability.

Notes:
    • This script provides detailed diagnostics for troubleshooting GPU issues.
    • All GPU checks include fallback error handling for CPU-only environments.
"""

import sys
import traceback
import logging

def setup_logging():
    """Set up logging for GPU diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_cuda_runtime():
    """Check CUDA runtime availability and version."""
    logger = logging.getLogger(__name__)
    logger.info("Checking CUDA runtime availability...")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        logger.info(f"PyTorch CUDA available: {cuda_available}")
        
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logger.info(f"CUDA version: {cuda_version}")
            logger.info(f"GPU device count: {device_count}")
            logger.info(f"Current GPU device: {current_device}")
            logger.info(f"GPU device name: {device_name}")
            
            # Check GPU memory.
            memory_allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(current_device) / (1024**3)
            memory_total = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            logger.info(f"GPU memory - Allocated: {memory_allocated:.2f} GB")
            logger.info(f"GPU memory - Reserved: {memory_reserved:.2f} GB")
            logger.info(f"GPU memory - Total: {memory_total:.2f} GB")
            
            return True, {
                'cuda_version': cuda_version,
                'device_count': device_count,
                'device_name': device_name,
                'memory_total_gb': memory_total
            }
        else:
            logger.warning("CUDA is not available in PyTorch")
            return False, {}
            
    except ImportError as e:
        logger.error(f"PyTorch not available: {e}")
        return False, {}
    except Exception as e:
        logger.error(f"CUDA check failed: {e}")
        logger.debug(f"CUDA check traceback:\n{traceback.format_exc()}")
        return False, {}

def check_pytorch_gpu():
    """Test PyTorch GPU tensor operations."""
    logger = logging.getLogger(__name__)
    logger.info("Testing PyTorch GPU tensor operations...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - skipping PyTorch GPU test")
            return False
        
        # Create test tensors on GPU.
        device = torch.device('cuda')
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Perform GPU computation.
        c = torch.matmul(a, b)
        result = c.sum().item()
        
        logger.info(f"PyTorch GPU computation successful - result: {result:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"PyTorch GPU test failed: {e}")
        logger.debug(f"PyTorch GPU traceback:\n{traceback.format_exc()}")
        return False

def check_cupy_support():
    """Check CuPy availability and GPU array operations."""
    logger = logging.getLogger(__name__)
    logger.info("Checking CuPy GPU support...")
    
    try:
        import cupy as cp
        
        # Check CuPy version and device info.
        cupy_version = cp.__version__
        device_count = cp.cuda.runtime.getDeviceCount()
        
        logger.info(f"CuPy version: {cupy_version}")
        logger.info(f"CuPy device count: {device_count}")
        
        if device_count > 0:
            # Test CuPy GPU operations.
            device = cp.cuda.Device()
            free_memory, total_memory = device.mem_info
            
            logger.info(f"CuPy GPU memory - Free: {free_memory / (1024**3):.2f} GB")
            logger.info(f"CuPy GPU memory - Total: {total_memory / (1024**3):.2f} GB")
            
            # Perform test computation.
            a = cp.random.randn(1000, 1000)
            b = cp.random.randn(1000, 1000)
            c = cp.dot(a, b)
            result = float(cp.sum(c))
            
            logger.info(f"CuPy GPU computation successful - result: {result:.2f}")
            return True
        else:
            logger.warning("No CUDA devices found by CuPy")
            return False
            
    except ImportError as e:
        logger.error(f"CuPy not available: {e}")
        return False
    except Exception as e:
        logger.error(f"CuPy test failed: {e}")
        logger.debug(f"CuPy traceback:\n{traceback.format_exc()}")
        return False

def check_cellpose_gpu():
    """Check Cellpose GPU model loading capability."""
    logger = logging.getLogger(__name__)
    logger.info("Checking Cellpose GPU support...")
    
    try:
        from cellpose import models
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - Cellpose will use CPU")
            return False
        
        # Try to create a GPU-enabled Cellpose model.
        logger.info("Loading Cellpose model with GPU support...")
        try:
            # Try the newer API first.
            model = models.CellposeModel(gpu=True, model_type='nuclei')
        except AttributeError:
            try:
                # Try the older API.
                model = models.Cellpose(gpu=True, model_type='nuclei')
            except AttributeError:
                # Try direct model loading.
                model = models.CellposeModel(gpu=True)

        if hasattr(model, 'gpu') and model.gpu:
            logger.info("Cellpose GPU model loaded successfully")
            logger.info(f"Cellpose using device: {getattr(model, 'device', 'unknown')}")
            return True
        elif hasattr(model, 'device') and 'cuda' in str(model.device):
            logger.info("Cellpose GPU model loaded successfully")
            logger.info(f"Cellpose using device: {model.device}")
            return True
        else:
            logger.warning("Cellpose model loaded but GPU not enabled")
            return False
            
    except ImportError as e:
        logger.error(f"Cellpose not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Cellpose GPU test failed: {e}")
        logger.debug(f"Cellpose GPU traceback:\n{traceback.format_exc()}")
        return False

def check_environment_config():
    """Check environment configuration for GPU support."""
    logger = logging.getLogger(__name__)
    logger.info("Checking environment configuration...")
    
    import os
    
    # Check CUDA environment variables.
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    
    logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    logger.info(f"CUDA_HOME: {cuda_home}")
    logger.info(f"CUDA_PATH: {cuda_path}")
    
    # Check Python version.
    python_version = sys.version
    logger.info(f"Python version: {python_version}")
    
    return True

def main():
    """Run comprehensive GPU support checks."""
    logger = setup_logging()
    logger.info("Starting comprehensive GPU support verification")
    logger.info("=" * 60)
    
    # Track test results.
    results = {}
    
    # Test 1: CUDA Runtime.
    logger.info("TEST 1: CUDA Runtime Check")
    logger.info("-" * 30)
    cuda_available, cuda_info = check_cuda_runtime()
    results['cuda_runtime'] = cuda_available
    
    # Test 2: PyTorch GPU.
    logger.info("\nTEST 2: PyTorch GPU Operations")
    logger.info("-" * 30)
    results['pytorch_gpu'] = check_pytorch_gpu()
    
    # Test 3: CuPy Support.
    logger.info("\nTEST 3: CuPy GPU Support")
    logger.info("-" * 30)
    results['cupy_support'] = check_cupy_support()
    
    # Test 4: Cellpose GPU.
    logger.info("\nTEST 4: Cellpose GPU Support")
    logger.info("-" * 30)
    results['cellpose_gpu'] = check_cellpose_gpu()
    
    # Test 5: Environment Config.
    logger.info("\nTEST 5: Environment Configuration")
    logger.info("-" * 30)
    results['environment'] = check_environment_config()
    
    # Summary.
    logger.info("\n" + "=" * 60)
    logger.info("GPU SUPPORT SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name.upper().replace('_', ' ')}: {status}")
    
    # Overall assessment.
    gpu_ready = results['cuda_runtime'] and results['pytorch_gpu']
    merge_ready = results['cupy_support']
    segmentation_ready = results['cellpose_gpu']
    
    logger.info("\nOVERALL ASSESSMENT:")
    logger.info(f"GPU Environment Ready: {'YES' if gpu_ready else 'NO'}")
    logger.info(f"GPU Segmentation Ready: {'YES' if segmentation_ready else 'NO'}")
    logger.info(f"GPU Merging Ready: {'YES' if merge_ready else 'NO'}")
    
    if gpu_ready and merge_ready and segmentation_ready:
        logger.info("\n✓ FULL GPU SUPPORT AVAILABLE")
        logger.info("Both segmentation and merging can use GPU acceleration")
        return 0
    elif gpu_ready:
        logger.warning("\n⚠ PARTIAL GPU SUPPORT")
        if not segmentation_ready:
            logger.warning("Cellpose segmentation will use CPU")
        if not merge_ready:
            logger.warning("Tile merging will use CPU")
        return 1
    else:
        logger.error("\n✗ NO GPU SUPPORT")
        logger.error("Both segmentation and merging will use CPU")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
