#!/bin/bash
#
# ATLAS-Cellpose CPU Job Script for SLURM Clusters.
# Author: Christos Botos.
# Affiliation: Leiden University Medical Center.
#
# Description:
#   Example SLURM batch script for running ATLAS-Cellpose segmentation without GPU (CPU-only mode).
#   This script is suitable for clusters without GPU availability or for very large images that
#   require extensive memory resources.
#
# IMPORTANT: Customize the following parameters for your cluster:
#   - mail-user: Your email address.
#   - mem: Memory allocation based on your image size (400GB is for very large images).
#   - cpus-per-task: Number of CPU cores (adjust based on availability).
#   - time: Maximum runtime based on expected processing time.
#
# Usage:
#   1. Copy this script and customize SLURM parameters.
#   2. Update the pipeline parameters in run_segmentation_instance.sh call.
#   3. Submit job: sbatch cs_jobcpu_for_cluster.sh
#

#SBATCH --job-name=atlas_cellpose_cpu
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=400G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=220:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

echo "=========================================="
echo "ATLAS-Cellpose CPU Segmentation Job"
echo "=========================================="
echo "Start time:   $(date)"
echo "Hostname:     $HOSTNAME"
echo "Job ID:       $SLURM_JOB_ID"
echo "Job Name:     $SLURM_JOB_NAME"
echo "Node:         $SLURMD_NODENAME"
echo "Nodes used:   $SLURM_JOB_NUM_NODES"
echo "Tasks:        $SLURM_NTASKS"
echo "CPUs/task:    $SLURM_CPUS_PER_TASK"
echo "Memory:       $SLURM_MEM_PER_NODE MB"
echo "Account:      ${SLURM_JOB_ACCOUNT:-not_set}"
echo "Submit host:  $SLURM_SUBMIT_HOST"
echo "Working dir:  $(pwd)"
echo "=========================================="

# Clean environment modules for safety.
module purge

# Initialize conda in batch script.
eval "$(conda shell.bash hook)"

# Activate the Cellpose3 environment.
conda activate venv310_cellpose3

# Verify environment activation.
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

echo "=========================================="
echo "Starting ATLAS-Cellpose Pipeline (CPU Mode)"
echo "=========================================="

# Run the segmentation pipeline in CPU-only mode.
# Customize parameters as needed for your analysis.
cd ../
./run_segmentation_instance.sh gpu False

exit_code=$?

echo ""
echo "=========================================="
echo "Job Completion Summary"
echo "=========================================="
echo "End time:     $(date)"
echo "Exit code:    $exit_code"
echo "=========================================="

exit $exit_code
