#!/bin/bash
#SBATCH --job-name=ss_bIRI2_segmentation_and_cpu_merging
#SBATCH --mail-user="hcty03@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --time=24:00:00
#SBATCH --partition=highmemgpu
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15GB
#SBATCH --gres=gpu:1

set -euo pipefail

###############################################################################
# 0 ┃ Configuration and Path Setup                                            .
###############################################################################

# Repository and project configuration.
REPO_NAME="ATLAS-Cellpose"
JOB_NAME="segmentation_only_run"                  # Custom job name for predictable results.

# Validate that we're in the expected directory structure.
if [[ ! -d "${REPO_NAME}" ]]; then
    echo "[FATAL] Repository ${REPO_NAME} not found in current directory: $(pwd)" >&2
    echo "Expected directory structure:" >&2
    echo "  $(pwd)/${REPO_NAME}/" >&2
    exit 1
fi

# Set absolute paths for reliability.
SEGMENTATION_REPO=$(realpath "${REPO_NAME}")
SEGMENTATION_CODE="${SEGMENTATION_REPO}/code/nuclei_segmentation"

echo "=== Job Configuration ==="
echo "Segmentation repo: ${SEGMENTATION_REPO}"
echo "Segmentation code: ${SEGMENTATION_CODE}"
echo "Job name: ${JOB_NAME}"

###############################################################################
# 1 ┃ Environment Setup                                                       .
###############################################################################

source venv311/bin/activate        # note: *relative* path, no leading /

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-0}

###############################################################################
# 2 ┃ Environment Validation                                                  .
###############################################################################

echo "=== Environment Validation ==="
python - <<'PY'
import torch
print("torch :", torch.__version__)
print("cuda  :", torch.version.cuda)
print("avail :", torch.cuda.is_available())
print("GPUs  :", torch.cuda.device_count())
PY

# Validate segmentation code directory.
if [[ ! -d "${SEGMENTATION_CODE}" ]]; then
    echo "[FATAL] Segmentation code directory not found: ${SEGMENTATION_CODE}" >&2
    exit 1
fi

if [[ ! -f "${SEGMENTATION_CODE}/run_this.py" ]]; then
    echo "[FATAL] Main segmentation script not found: ${SEGMENTATION_CODE}/run_this.py" >&2
    exit 1
fi

echo "Segmentation code validation: PASS"

###############################################################################
# 3 ┃ Nuclei Segmentation Pipeline                                            .
###############################################################################

# Change to segmentation code directory.
cd "${SEGMENTATION_CODE}"
echo "Current working directory is $(pwd)"

# Set job name as environment variable for the Python script.
export SEGMENTATION_JOB_NAME="${JOB_NAME}"

# Run segmentation with custom job name.
python3 -c "
import os
import sys
sys.path.insert(0, '.')
from utils.project_setup import load_config
from utils.logging_utils import setup_logging
from utils.debug_utils import setup_debug
from pipeline import run_segmentation_pipeline

# Load config with custom job name.
job_name = os.environ.get('SEGMENTATION_JOB_NAME', 'segmentation_run')
settings, CELLPOSE_PARAMS, PROJECT_DIRS = load_config(job_name=job_name)

# Setup logging and debug.
debug_mode = settings.get('debug_mode', False)
logger = setup_logging(settings['output_dir'], debug_mode)
snap = setup_debug(settings)

logger.info(f'==== Server Job: {job_name} ====')
logger.info(f'Results directory: {settings[\"output_dir\"]}')

# Run pipeline.
exit_code = run_segmentation_pipeline(settings, CELLPOSE_PARAMS, PROJECT_DIRS, logger, snap)
sys.exit(exit_code)
"

SEGMENTATION_EXIT_CODE=$?
echo "Program finished with exit code ${SEGMENTATION_EXIT_CODE} at: $(date)"

if [[ ${SEGMENTATION_EXIT_CODE} -ne 0 ]]; then
    echo "[FATAL] Segmentation failed with exit code ${SEGMENTATION_EXIT_CODE}" >&2
    exit ${SEGMENTATION_EXIT_CODE}
fi

###############################################################################
# 4 ┃ Results Summary                                                         .
###############################################################################

echo "=== Results Summary ==="
echo "Job name: ${JOB_NAME}"

# Use the results locator to show where results were saved.
python3 -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
from utils.results_locator import find_latest_results, create_results_summary

# Find the latest results directory.
results_base = Path('../../results')
latest_results = find_latest_results(results_base)

if latest_results:
    print(f'Results saved to: {latest_results}')
    summary = create_results_summary(latest_results)
    print(f'Total files: {len(summary[\"files\"])}')
    print(f'Total size: {summary[\"total_size_mb\"]} MB')
    print(f'Mask files: {len(summary[\"mask_files\"])}')
else:
    print('WARNING: Could not locate results directory')
"

echo "✓ Segmentation job completed successfully."
