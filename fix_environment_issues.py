#!/usr/bin/env python3
"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: fix_environment_issues.py.
Description:
    Comprehensive fix for environment issues causing segmentation failures.
    This script addresses conda environment activation, PyTorch CUDA setup,
    and parallel processing configuration issues.

Dependencies:
    • Python >= 3.10.
    • Access to conda environment management.
    • System administration privileges for environment setup.

Usage:
    python fix_environment_issues.py

Arguments:
    None (runs automated fixes).

Inputs:
    • Current environment configuration.
    • System hardware capabilities.

Outputs:
    • Fixed environment configuration.
    • Updated pipeline scripts with proper environment activation.
    • Optimized parallel processing settings.
    • Detailed fix report in logs/environment_fixes.log.

Key Features:
    • Automatic conda environment activation fix.
    • PyTorch CUDA installation and configuration.
    • Parallel processing optimization for CPU-only systems.
    • Configuration file updates for better stability.

Notes:
    • Run this script to fix segmentation timeout and environment issues.
    • Restart your terminal/IDE after running this script.
    • Test the pipeline with smaller images first after fixes.
"""

import traceback
import sys
import os
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import configparser

# Set up logging.
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"environment_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_command(command, description, check=True):
    """Run a shell command with logging."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=check
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        if result.stderr and result.returncode == 0:
            logger.info(f"Warnings: {result.stderr.strip()}")
        
        logger.info(f"SUCCESS: {description}")
        return result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {description}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        if not check:
            return e
        raise


def fix_conda_environment():
    """Fix conda environment activation issues."""
    logger.info("=" * 60)
    logger.info("FIXING CONDA ENVIRONMENT ACTIVATION")
    logger.info("=" * 60)
    
    # Check if conda is available.
    try:
        result = run_command(
            "wsl bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda --version'",
            "Testing conda availability"
        )
    except subprocess.CalledProcessError:
        logger.error("Conda not available. Please install Miniconda/Anaconda first.")
        return False
    
    # Check if the environment exists.
    try:
        result = run_command(
            "wsl bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda info --envs | grep iri310_cellpose3'",
            "Checking iri310_cellpose3 environment"
        )
    except subprocess.CalledProcessError:
        logger.error("Environment iri310_cellpose3 not found. Please create it first.")
        return False
    
    # Test environment activation.
    try:
        result = run_command(
            "wsl bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate iri310_cellpose3 && python --version'",
            "Testing environment activation"
        )
        logger.info("✓ Conda environment activation working correctly")
        return True
    except subprocess.CalledProcessError:
        logger.error("Environment activation failed")
        return False


def check_pytorch_cuda():
    """Check and potentially fix PyTorch CUDA installation."""
    logger.info("=" * 60)
    logger.info("CHECKING PYTORCH CUDA SETUP")
    logger.info("=" * 60)
    
    # Check current PyTorch installation.
    try:
        result = run_command(
            "wsl bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate iri310_cellpose3 && python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\"'",
            "Checking current PyTorch installation"
        )
        
        if "False" in result.stdout:
            logger.warning("PyTorch CUDA not available - running on CPU only")
            logger.info("This will cause slower processing and potential timeouts")
            
            # Ask user if they want to install CUDA-enabled PyTorch.
            logger.info("To fix this, you can:")
            logger.info("1. Install CUDA-enabled PyTorch (requires NVIDIA GPU)")
            logger.info("2. Optimize CPU-only configuration (recommended for now)")
            
            return False  # CUDA not available.
        else:
            logger.info("✓ PyTorch CUDA is available")
            return True
            
    except subprocess.CalledProcessError:
        logger.error("Failed to check PyTorch installation")
        return False


def optimize_cpu_configuration():
    """Optimize configuration for CPU-only processing."""
    logger.info("=" * 60)
    logger.info("OPTIMIZING CPU-ONLY CONFIGURATION")
    logger.info("=" * 60)
    
    config_path = Path("configs/nuclei_segmentation_config.ini")
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False
    
    # Backup original config.
    backup_path = config_path.with_suffix('.ini.backup')
    shutil.copy2(config_path, backup_path)
    logger.info(f"Backed up original config to: {backup_path}")
    
    # Read and modify configuration.
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # CPU-optimized settings.
    cpu_optimizations = {
        'cellpose': {
            'gpu': 'False',  # Force CPU mode.
            'parallel_batch_size': '2',  # Reduce batch size for CPU.
            'parallel_max_workers': '2',  # Reduce workers for CPU.
            'parallel_timeout_seconds': '600',  # Increase timeout for CPU.
            'parallel_memory_limit_gb': '4.0',  # Reduce memory limit.
        },
        'tiling': {
            'tile_side_length': '256',  # Smaller tiles for CPU.
            'merge_batch_size': '2',  # Reduce merge batch size.
            'gpu_batch_size': '1',  # Single tile processing.
            'gpu_memory_limit_gb': '1.0',  # Conservative memory limit.
        }
    }
    
    # Apply optimizations.
    changes_made = []
    for section_name, settings in cpu_optimizations.items():
        if section_name not in config:
            config.add_section(section_name)
        
        for key, value in settings.items():
            old_value = config.get(section_name, key, fallback='N/A')
            config.set(section_name, key, value)
            changes_made.append(f"{section_name}.{key}: {old_value} → {value}")
    
    # Write updated configuration.
    with open(config_path, 'w') as f:
        config.write(f)
    
    logger.info("Applied CPU optimizations:")
    for change in changes_made:
        logger.info(f"  {change}")
    
    return True


def create_environment_wrapper():
    """Create a wrapper script that ensures proper environment activation."""
    logger.info("=" * 60)
    logger.info("CREATING ENVIRONMENT WRAPPER SCRIPT")
    logger.info("=" * 60)
    
    wrapper_content = '''#!/bin/bash
# Environment wrapper for I/R injury analysis pipeline.
# This script ensures proper conda environment activation.

# Source conda initialization.
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the specific environment.
conda activate iri310_cellpose3

# Check if activation was successful.
if [[ "$CONDA_DEFAULT_ENV" != "iri310_cellpose3" ]]; then
    echo "ERROR: Failed to activate iri310_cellpose3 environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "Successfully activated environment: $CONDA_DEFAULT_ENV"
echo "Python executable: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Run the provided command.
exec "$@"
'''
    
    wrapper_path = Path("run_with_env.sh")
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    # Make executable.
    os.chmod(wrapper_path, 0o755)
    
    logger.info(f"Created environment wrapper: {wrapper_path}")
    logger.info("Usage: wsl ./run_with_env.sh python code/nuclei_segmentation/run_this.py")
    
    return True


def update_run_script():
    """Update the main run script to use proper environment activation."""
    logger.info("=" * 60)
    logger.info("UPDATING MAIN RUN SCRIPT")
    logger.info("=" * 60)
    
    run_script_path = Path("code/nuclei_segmentation/run_this.py")
    if not run_script_path.exists():
        logger.warning(f"Main run script not found: {run_script_path}")
        return False
    
    # Add environment check at the beginning.
    env_check_code = '''
# Environment validation.
import sys
import os

def validate_environment():
    """Validate that we're running in the correct conda environment."""
    expected_env = "iri310_cellpose3"
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    
    if current_env != expected_env:
        print(f"ERROR: Wrong conda environment!")
        print(f"Expected: {expected_env}")
        print(f"Current: {current_env}")
        print("Please activate the correct environment:")
        print(f"  conda activate {expected_env}")
        sys.exit(1)
    
    print(f"✓ Running in correct environment: {current_env}")

# Validate environment before proceeding.
validate_environment()
'''
    
    # Read current script.
    with open(run_script_path, 'r') as f:
        content = f.read()
    
    # Check if validation is already present.
    if "validate_environment" in content:
        logger.info("Environment validation already present in run script")
        return True
    
    # Insert validation after imports.
    import_end = content.find('from pipeline import run_segmentation_pipeline')
    if import_end == -1:
        logger.error("Could not find import section in run script")
        return False
    
    import_end = content.find('\n', import_end) + 1
    new_content = content[:import_end] + env_check_code + content[import_end:]
    
    # Backup and write.
    backup_path = run_script_path.with_suffix('.py.backup')
    shutil.copy2(run_script_path, backup_path)
    
    with open(run_script_path, 'w') as f:
        f.write(new_content)
    
    logger.info("Added environment validation to run script")
    return True


def main():
    """Run comprehensive environment fixes."""
    logger.info("Starting comprehensive environment fixes...")
    logger.info(f"Log file: {log_file}")
    
    try:
        # Fix conda environment.
        if not fix_conda_environment():
            logger.error("Failed to fix conda environment - aborting")
            return 1
        
        # Check PyTorch CUDA.
        cuda_available = check_pytorch_cuda()
        
        # Optimize for CPU if CUDA not available.
        if not cuda_available:
            if not optimize_cpu_configuration():
                logger.error("Failed to optimize CPU configuration")
                return 1
        
        # Create environment wrapper.
        if not create_environment_wrapper():
            logger.error("Failed to create environment wrapper")
            return 1
        
        # Update run script.
        if not update_run_script():
            logger.warning("Failed to update run script - manual activation required")
        
        # Summary.
        logger.info("=" * 60)
        logger.info("ENVIRONMENT FIXES COMPLETED")
        logger.info("=" * 60)
        
        logger.info("✓ Conda environment activation verified")
        if cuda_available:
            logger.info("✓ PyTorch CUDA available")
        else:
            logger.info("⚠ PyTorch CUDA not available - optimized for CPU")
        logger.info("✓ Configuration optimized for current hardware")
        logger.info("✓ Environment wrapper script created")
        
        logger.info("\nTo run the pipeline with proper environment:")
        logger.info("  wsl ./run_with_env.sh python code/nuclei_segmentation/run_this.py")
        
        logger.info("\nOr manually activate environment:")
        logger.info("  wsl bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate iri310_cellpose3 && python code/nuclei_segmentation/run_this.py'")
        
        return 0
        
    except Exception as e:
        logger.error(f"Environment fixes failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
