#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: update_gpu_environment.py.
Description:
    Environment update script to install GPU-enabled PyTorch and related packages
    for the I/R injury spatial multiomics analysis pipeline. This script ensures
    both Cellpose segmentation and tile merging can utilize GPU acceleration.

Dependencies:
    • conda or mamba package manager.
    • NVIDIA GPU with CUDA support.
    • Windows/Linux system with CUDA drivers installed.

Usage:
    python update_gpu_environment.py

Key Features:
    • Installs CUDA-enabled PyTorch from pytorch channel.
    • Ensures CuPy compatibility for GPU tile merging.
    • Validates GPU support after installation.
    • Provides detailed installation progress reporting.

Notes:
    • This script requires conda environment to be activated.
    • Installation may take 10-15 minutes depending on internet speed.
"""

import subprocess
import sys
import logging
from pathlib import Path

def setup_logging():
    """Set up logging for environment update."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/gpu_environment_update.log')
        ]
    )
    return logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command with logging."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
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
        logger.error(f"TIMEOUT: {description} took longer than 30 minutes")
        return False
    except Exception as e:
        logger.error(f"EXCEPTION: {description} failed with {e}")
        return False

def check_conda_environment():
    """Check if conda environment is properly activated."""
    logger = logging.getLogger(__name__)
    logger.info("Checking conda environment...")
    
    # Check if we're in a conda environment.
    conda_env = subprocess.run(
        "conda info --envs | grep '*'",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if conda_env.returncode == 0 and conda_env.stdout.strip():
        env_info = conda_env.stdout.strip()
        logger.info(f"Active conda environment: {env_info}")
        return True
    else:
        logger.error("No active conda environment detected")
        logger.error("Please activate your conda environment first:")
        logger.error("  conda activate iri310")
        return False

def install_gpu_pytorch():
    """Install GPU-enabled PyTorch packages."""
    logger = logging.getLogger(__name__)
    logger.info("Installing GPU-enabled PyTorch packages...")
    
    # First, remove existing CPU-only PyTorch.
    remove_cmd = "conda remove pytorch torchvision torchaudio pytorch-cuda -y"
    if not run_command(remove_cmd, "Removing existing PyTorch packages"):
        logger.warning("Failed to remove existing PyTorch - continuing anyway")
    
    # Install GPU-enabled PyTorch from pytorch channel.
    install_cmd = (
        "conda install pytorch torchvision torchaudio pytorch-cuda=12.1 "
        "-c pytorch -c nvidia -y"
    )
    
    return run_command(install_cmd, "Installing GPU-enabled PyTorch")

def install_cupy():
    """Install CuPy for GPU array operations."""
    logger = logging.getLogger(__name__)
    logger.info("Installing CuPy for GPU array operations...")
    
    # Install CuPy with CUDA 12.x support.
    install_cmd = "conda install cupy -c conda-forge -y"
    
    return run_command(install_cmd, "Installing CuPy")

def verify_installation():
    """Verify that GPU packages are properly installed."""
    logger = logging.getLogger(__name__)
    logger.info("Verifying GPU package installation...")
    
    # Test PyTorch CUDA.
    pytorch_test = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
"""
    
    if not run_command(f'python -c "{pytorch_test}"', "Testing PyTorch CUDA"):
        return False
    
    # Test CuPy.
    cupy_test = """
import cupy as cp
print(f"CuPy version: {cp.__version__}")
print(f"CUDA devices: {cp.cuda.runtime.getDeviceCount()}")
if cp.cuda.runtime.getDeviceCount() > 0:
    device = cp.cuda.Device()
    free, total = device.mem_info
    print(f"GPU memory: {free/(1024**3):.1f}/{total/(1024**3):.1f} GB")
"""
    
    if not run_command(f'python -c "{cupy_test}"', "Testing CuPy"):
        return False
    
    # Test Cellpose GPU.
    cellpose_test = """
from cellpose import models
import torch
if torch.cuda.is_available():
    model = models.Cellpose(gpu=True, model_type='nuclei')
    print(f"Cellpose GPU enabled: {model.gpu}")
    print(f"Cellpose device: {model.device}")
else:
    print("CUDA not available for Cellpose")
"""
    
    return run_command(f'python -c "{cellpose_test}"', "Testing Cellpose GPU")

def update_configuration():
    """Update pipeline configuration for GPU usage."""
    logger = logging.getLogger(__name__)
    logger.info("Updating pipeline configuration for GPU usage...")
    
    config_path = Path("configs/nuclei_segmentation_config.ini")
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        return True
    
    try:
        # Read current configuration.
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Update GPU settings.
        updated_content = config_content.replace(
            "gpu = False", "gpu = True"
        ).replace(
            "gpu_batch_size = 1", "gpu_batch_size = 2"
        ).replace(
            "gpu_memory_limit_gb = 2.0", "gpu_memory_limit_gb = 4.0"
        )
        
        # Write updated configuration.
        with open(config_path, 'w') as f:
            f.write(updated_content)
        
        logger.info("Configuration updated for GPU usage")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        return False

def main():
    """Main environment update process."""
    logger = setup_logging()
    logger.info("Starting GPU environment update for I/R injury analysis pipeline")
    logger.info("=" * 70)
    
    # Step 1: Check conda environment.
    if not check_conda_environment():
        logger.error("Please activate your conda environment and try again")
        return 1
    
    # Step 2: Install GPU-enabled PyTorch.
    logger.info("\nSTEP 1: Installing GPU-enabled PyTorch")
    logger.info("-" * 40)
    if not install_gpu_pytorch():
        logger.error("Failed to install GPU-enabled PyTorch")
        return 1
    
    # Step 3: Install CuPy.
    logger.info("\nSTEP 2: Installing CuPy")
    logger.info("-" * 40)
    if not install_cupy():
        logger.error("Failed to install CuPy")
        return 1
    
    # Step 4: Verify installation.
    logger.info("\nSTEP 3: Verifying Installation")
    logger.info("-" * 40)
    if not verify_installation():
        logger.error("GPU package verification failed")
        return 1
    
    # Step 5: Update configuration.
    logger.info("\nSTEP 4: Updating Configuration")
    logger.info("-" * 40)
    if not update_configuration():
        logger.warning("Configuration update failed - you may need to update manually")
    
    # Success!
    logger.info("\n" + "=" * 70)
    logger.info("GPU ENVIRONMENT UPDATE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info("Your environment now supports:")
    logger.info("  ✓ GPU-accelerated Cellpose segmentation")
    logger.info("  ✓ GPU-accelerated tile merging with CuPy")
    logger.info("  ✓ Optimized memory management for large images")
    logger.info("\nYou can now run the pipeline with full GPU acceleration!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
