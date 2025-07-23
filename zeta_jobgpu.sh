#!/usr/bin/env bash
#SBATCH --job-name=ss_bIRI2_full_pipeline
#SBATCH --partition=highmemgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --chdir=/exports/humgen/cbotos            # Common parent for both repos.

set -euo pipefail

###############################################################################
# 0 ┃ Configuration and Path Setup                                            .
###############################################################################

# Repository and project configuration.
REPO_NAME="IR_cell_segmentation"
VIT_REPO_NAME="iri_vit"
JOB_NAME="server_gpu_run2"                         # Custom job name for predictable results.

# Validate that we're in the expected directory structure.
if [[ ! -d "${REPO_NAME}" ]]; then
    echo "[FATAL] Repository ${REPO_NAME} not found in current directory: $(pwd)" >&2
    echo "Expected directory structure:" >&2
    echo "  $(pwd)/${REPO_NAME}/" >&2
    echo "  $(pwd)/${VIT_REPO_NAME}/" >&2
    exit 1
fi

# Set absolute paths for reliability.
SEGMENTATION_REPO=$(realpath "${REPO_NAME}")
VIT_REPO=$(realpath "${VIT_REPO_NAME}")
SEGMENTATION_CODE="${SEGMENTATION_REPO}/code/nuclei_segmentation"

echo "=== Job Configuration ==="
echo "Segmentation repo: ${SEGMENTATION_REPO}"
echo "ViT repo: ${VIT_REPO}"
echo "Segmentation code: ${SEGMENTATION_CODE}"
echo "Job name: ${JOB_NAME}"

###############################################################################
# 1 ┃ Module stack & virtual-environment.                                     .
###############################################################################
module purge
module load library/cuda/12.2.2/gcc.8.5.0        # CUDA 12.2 tool-chain with GCC 8.5.

export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

source venv311/bin/activate                       # One shared venv for both projects.

###############################################################################
# 2 ┃ Quick environment sanity checks.                                        .
###############################################################################
echo "=== Environment Validation ==="
nvidia-smi || echo "No NVIDIA driver visible"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python - <<'PY'
import torch, os, sys
print("torch :", torch.__version__)
print("cuda  :", torch.version.cuda)
print("avail :", torch.cuda.is_available())
print("GPUs  :", torch.cuda.device_count())
print("python:", sys.version.replace("\n", " "))
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
# 3 ┃ Stage A – nuclei segmentation (Cellpose 4 pipeline).                    .
###############################################################################
printf '\n=== Stage A: Running Cellpose-based nuclei segmentation ===\n'

# Change to segmentation code directory.
cd "${SEGMENTATION_CODE}"
echo "Working directory: $(pwd)"

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
job_name = os.environ.get('SEGMENTATION_JOB_NAME', 'server_run')
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
echo "Cell segmentation finished with exit code ${SEGMENTATION_EXIT_CODE} at: $(date)"

if [[ ${SEGMENTATION_EXIT_CODE} -ne 0 ]]; then
    echo "[FATAL] Segmentation failed with exit code ${SEGMENTATION_EXIT_CODE}" >&2
    exit ${SEGMENTATION_EXIT_CODE}
fi

###############################################################################
# 4 ┃ Stage B – copy segmentation mask into iri_vit.                          .
###############################################################################
printf '\n=== Stage B: Copying segmentation mask to iri_vit ===\n'

# Use the results locator utility to find the segmentation mask reliably.
MASK_SRC=$(python3 -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
from utils.results_locator import find_latest_results, find_segmentation_mask

# Find the latest results directory.
results_base = Path('../../results')
latest_results = find_latest_results(results_base)

if not latest_results:
    print('ERROR: No results directory found', file=sys.stderr)
    sys.exit(1)

# Find the segmentation mask.
mask_path = find_segmentation_mask(latest_results)
if not mask_path:
    print('ERROR: No segmentation mask found', file=sys.stderr)
    sys.exit(1)

print(str(mask_path))
")

MASK_SEARCH_EXIT_CODE=$?
if [[ ${MASK_SEARCH_EXIT_CODE} -ne 0 ]]; then
    echo "[FATAL] Failed to locate segmentation mask" >&2
    exit 127
fi

if [[ -z "${MASK_SRC}" ]]; then
    echo "[FATAL] segmentation_masks.npy not found – aborting." >&2
    exit 127
fi

echo "Found segmentation mask at: ${MASK_SRC}"

# Validate that the mask file exists and is readable.
if [[ ! -f "${MASK_SRC}" ]]; then
    echo "[FATAL] Mask file does not exist: ${MASK_SRC}" >&2
    exit 127
fi

if [[ ! -r "${MASK_SRC}" ]]; then
    echo "[FATAL] Mask file is not readable: ${MASK_SRC}" >&2
    exit 127
fi

# Validate ViT repository exists.
if [[ ! -d "${VIT_REPO}" ]]; then
    echo "[FATAL] ViT repository not found: ${VIT_REPO}" >&2
    exit 127
fi

# Destination is the root of iri_vit so that pipeline.sh finds it via RAW_MASKS.
MASK_DEST="${VIT_REPO}/segmentation_masks_whole.npy"
echo "Copying mask to: ${MASK_DEST}"

cp -v "${MASK_SRC}" "${MASK_DEST}"

# Verify the copy was successful.
if [[ ! -f "${MASK_DEST}" ]]; then
    echo "[FATAL] Failed to copy mask file to destination" >&2
    exit 127
fi

echo "Mask copy completed successfully"

###############################################################################
# 5 ┃ Stage C – ViT-based clustering pipeline.                                .
###############################################################################
printf '\n=== Stage C: Running ViT clustering pipeline ===\n'

# Change to ViT repository directory.
cd "${VIT_REPO}"
echo "Working directory: $(pwd)"

# Validate that the pipeline script exists.
if [[ ! -f "./pipeline.sh" ]]; then
    echo "[FATAL] ViT pipeline script not found: $(pwd)/pipeline.sh" >&2
    exit 127
fi

# Make sure the pipeline script is executable.
chmod +x ./pipeline.sh

# Run the ViT pipeline.
./pipeline.sh --workers 8 --batch_size 90000 --no_compile
VIT_EXIT_CODE=$?

echo "ViT pipeline finished with exit code ${VIT_EXIT_CODE} at: $(date)"

if [[ ${VIT_EXIT_CODE} -ne 0 ]]; then
    echo "[FATAL] ViT pipeline failed with exit code ${VIT_EXIT_CODE}" >&2
    exit ${VIT_EXIT_CODE}
fi

###############################################################################
# 6 ┃ Final Summary and Cleanup                                               .
###############################################################################
printf '\n=== Job Summary ===\n'
echo "Segmentation job name: ${JOB_NAME}"
echo "Segmentation mask source: ${MASK_SRC}"
echo "ViT mask destination: ${MASK_DEST}"
echo "All stages completed successfully at: $(date)"

printf '\n✓ Combined segmentation → ViT job completed successfully.\n'
