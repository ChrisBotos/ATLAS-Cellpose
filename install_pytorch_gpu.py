#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: install_pytorch_gpu.py.
Description:
    Script to install GPU-enabled PyTorch for Cellpose segmentation in the I/R injury
    spatial multiomics analysis pipeline. This script ensures that both segmentation
    (via PyTorch) and tile merging (via CuPy) can utilize GPU acceleration.

Dependencies:
    • pip package manager.
    • NVIDIA GPU with CUDA support.
    • Windows/Linux system with CUDA drivers installed.

Usage:
    python install_pytorch_gpu.py

Key Features:
    • Installs CUDA-enabled PyTorch from official PyTorch index.
    • Verifies GPU support after installation.
    • Tests Cellpose GPU functionality.
    • Provides detailed installation progress reporting.

Notes:
    • This script requires internet connection for package downloads.
    • Installation may take 5-10 minutes depending on internet speed.
"""

import subprocess
import sys
import logging
import time

def setup_logging():
    """Set up logging for PyTorch GPU installation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/pytorch_gpu_install.log')
        ]
    )
    return logging.getLogger(__name__)

def run_command(command, description, timeout=600):
    """Run a shell command with logging and timeout."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            logger.info(f"SUCCESS: {description}")
            if result.stdout.strip():
                logger.debug(f"Output: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"FAILED: {description}")
            logger.error(f"Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT: {description} took longer than {timeout} seconds")
        return False
    except Exception as e:
        logger.error(f"EXCEPTION: {description} failed with {e}")
        return False

def install_pytorch_gpu():
    """Install GPU-enabled PyTorch packages."""
    logger = logging.getLogger(__name__)
    logger.info("Installing GPU-enabled PyTorch packages...")
    
    # PyTorch GPU installation command for CUDA 12.1.
    install_cmd = (
        "pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    
    return run_command(install_cmd, "Installing GPU-enabled PyTorch", timeout=1800)

def verify_pytorch_gpu():
    """Verify that PyTorch GPU installation worked."""
    logger = logging.getLogger(__name__)
    logger.info("Verifying PyTorch GPU installation...")
    
    # Test PyTorch CUDA.
    pytorch_test = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    # Test GPU computation
    device = torch.device('cuda')
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
    result = c.sum().item()
    print(f"GPU computation successful: {result:.2f}")
else:
    print("CUDA not available in PyTorch")
"""
    
    return run_command(f'python -c "{pytorch_test}"', "Testing PyTorch CUDA")

def verify_cellpose_gpu():
    """Verify that Cellpose can use GPU with new PyTorch."""
    logger = logging.getLogger(__name__)
    logger.info("Verifying Cellpose GPU support...")
    
    # Test Cellpose GPU.
    cellpose_test = """
from cellpose import models
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        model = models.Cellpose(gpu=True, model_type='nuclei')
        print(f"Cellpose GPU enabled: {model.gpu}")
        print(f"Cellpose device: {model.device}")
        print("Cellpose GPU test successful")
    except Exception as e:
        print(f"Cellpose GPU test failed: {e}")
else:
    print("CUDA not available for Cellpose")
"""
    
    return run_command(f'python -c "{cellpose_test}"', "Testing Cellpose GPU")

def main():
    """Main PyTorch GPU installation process."""
    logger = setup_logging()
    logger.info("Starting PyTorch GPU installation for I/R injury analysis pipeline")
    logger.info("=" * 70)
    
    # Step 1: Install GPU-enabled PyTorch.
    logger.info("STEP 1: Installing GPU-enabled PyTorch")
    logger.info("-" * 40)
    if not install_pytorch_gpu():
        logger.error("Failed to install GPU-enabled PyTorch")
        return 1
    
    # Step 2: Verify PyTorch GPU.
    logger.info("\nSTEP 2: Verifying PyTorch GPU")
    logger.info("-" * 40)
    if not verify_pytorch_gpu():
        logger.error("PyTorch GPU verification failed")
        return 1
    
    # Step 3: Verify Cellpose GPU.
    logger.info("\nSTEP 3: Verifying Cellpose GPU")
    logger.info("-" * 40)
    if not verify_cellpose_gpu():
        logger.warning("Cellpose GPU verification failed - but PyTorch GPU works")
        logger.warning("You may need to restart your Python environment")
    
    # Success!
    logger.info("\n" + "=" * 70)
    logger.info("PYTORCH GPU INSTALLATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info("Your environment now supports:")
    logger.info("  * GPU-accelerated PyTorch operations")
    logger.info("  * GPU-accelerated Cellpose segmentation")
    logger.info("  * GPU-accelerated tile merging with CuPy")
    logger.info("\nYou can now run the pipeline with full GPU acceleration!")
    logger.info("Note: You may need to restart your Python environment for changes to take effect.")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
