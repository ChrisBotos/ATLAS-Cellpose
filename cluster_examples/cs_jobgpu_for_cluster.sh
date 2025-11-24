#!/bin/bash
#
# ATLAS-Cellpose GPU Job Script for SLURM Clusters.
# Author: Christos Botos.
# Affiliation: Leiden University Medical Center.
#
# Description:
#   Example SLURM batch script for running ATLAS-Cellpose segmentation with GPU acceleration.
#   This script demonstrates proper environment setup, module loading, and conda activation
#   in HPC cluster environments.
#
# IMPORTANT: Customize the following parameters for your cluster:
#   - mail-user: Your email address.
#   - partition: Your cluster's GPU partition name.
#   - mem: Memory allocation based on your image size.
#   - time: Maximum runtime based on expected processing time.
#   - CUDA module: Adjust version to match your cluster's available modules.
#
# Usage:
#   1. Copy this script and customize SLURM parameters.
#   2. Update the pipeline parameters in run_segmentation_instance.sh call.
#   3. Submit job: sbatch cs_jobgpu_for_cluster.sh
#

#SBATCH --job-name=atlas_cellpose_gpu
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=40:00:00
#SBATCH --partition=highmemgpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "=========================================="
echo "ATLAS-Cellpose GPU Segmentation Job"
echo "=========================================="
echo "Start time:   $(date)"
echo "Hostname:     $HOSTNAME"
echo "Job ID:       $SLURM_JOB_ID"
echo "Job Name:     $SLURM_JOB_NAME"
echo "Node:         $SLURMD_NODENAME"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Memory:       $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Clean environment modules for safety.
module purge

# Load CUDA module (adjust version for your cluster).
# Check available modules with: module avail cuda
module load library/cuda/11.8.0/gcc.8.5.0

# Verify CUDA availability.
echo "CUDA version:"
nvcc --version || echo "Warning: nvcc not found in PATH"
echo ""

# Initialize conda in batch script.
eval "$(conda shell.bash hook)"

# Activate the Cellpose3 environment.
conda activate venv310_cellpose3

# Verify environment activation.
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Navigate to project directory.
# Assumes this script is in cluster_examples/ subdirectory.
cd "$(dirname "$0")/.." || { echo "Error: Could not navigate to project directory"; exit 1; }
echo "Working directory: $(pwd)"
echo ""

echo "=========================================="
echo "Starting ATLAS-Cellpose Pipeline"
echo "=========================================="

# Run the segmentation pipeline with GPU acceleration.
# Customize parameters as needed for your analysis.
./run_segmentation_instance.sh gpu True

exit_code=$?

echo ""
echo "=========================================="
echo "Job Completion Summary"
echo "=========================================="
echo "End time:     $(date)"
echo "Exit code:    $exit_code"
echo "=========================================="

exit $exit_code