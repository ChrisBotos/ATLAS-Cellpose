# Server Job Script Improvements

## Overview

This document describes the comprehensive improvements made to the server job scripts (`zeta_jobgpu.sh` and `cs_jobgpu.sh`) to make them more robust, reliable, and user-friendly for running the nuclei segmentation pipeline on server environments.

## Problems Addressed

### 1. **Path Dependencies**
- **Before**: Scripts assumed hardcoded directory structure (`IRI_cell_segmentation/code/nuclei_segmentation`)
- **After**: Dynamic path resolution based on actual repository structure (`code/nuclei_segmentation`)

### 2. **Results Directory Naming**
- **Before**: Unpredictable timestamped directories made it difficult for scripts to locate results
- **After**: Predictable job naming with automatic latest results tracking

### 3. **Error Handling**
- **Before**: Limited validation and error reporting
- **After**: Comprehensive validation at each step with detailed error messages

### 4. **Path Management**
- **Before**: Inconsistent working directory changes and relative path usage
- **After**: Absolute path management with proper validation

## Key Improvements

### 1. **Enhanced Results Locator (`utils/results_locator.py`)**

New utility module that provides robust results directory discovery:

```python
from utils.results_locator import find_latest_results, find_segmentation_mask

# Find the most recent results directory
latest_results = find_latest_results(results_base_dir)

# Locate segmentation mask reliably
mask_path = find_segmentation_mask(latest_results)
```

**Features:**
- Multiple discovery strategies (symlinks, text files, timestamp search)
- Comprehensive validation of results directories
- Detailed error reporting and debugging information
- Cross-platform compatibility

### 2. **Improved Configuration Management**

Enhanced `load_config()` function with job name support:

```python
# Load config with custom job name for server runs
settings, cellpose_params, dirs = load_config(job_name="server_gpu_run")
```

**Features:**
- Custom job names for predictable results directory naming
- Automatic creation of "latest" results tracking
- Environment variable support for job scripts

### 3. **Robust Job Scripts**

Both `zeta_jobgpu.sh` and `cs_jobgpu.sh` now include:

#### **Configuration Section**
```bash
# Repository and project configuration
REPO_NAME="I-R-Injury-Spatial-Multiomics-Analysis"
JOB_NAME="server_gpu_run"

# Validate directory structure
if [[ ! -d "${REPO_NAME}" ]]; then
    echo "[FATAL] Repository ${REPO_NAME} not found" >&2
    exit 1
fi
```

#### **Path Validation**
```bash
# Set absolute paths for reliability
SEGMENTATION_REPO=$(realpath "${REPO_NAME}")
SEGMENTATION_CODE="${SEGMENTATION_REPO}/code/nuclei_segmentation"

# Validate paths exist
if [[ ! -f "${SEGMENTATION_CODE}/run_this.py" ]]; then
    echo "[FATAL] Main segmentation script not found" >&2
    exit 1
fi
```

#### **Robust Results Location**
```bash
# Use results locator utility to find segmentation mask
MASK_SRC=$(python3 -c "
from utils.results_locator import find_latest_results, find_segmentation_mask
latest_results = find_latest_results(Path('../../results'))
mask_path = find_segmentation_mask(latest_results)
print(str(mask_path))
")
```

### 4. **Environment Variable Integration**

Job scripts now set environment variables that are picked up by the Python pipeline:

```bash
export SEGMENTATION_JOB_NAME="${JOB_NAME}"
```

The `run_this.py` script automatically uses this for configuration:

```python
job_name = os.environ.get('SEGMENTATION_JOB_NAME')
settings, CELLPOSE_PARAMS, PROJECT_DIRS = load_config(job_name=job_name)
```

## Usage Examples

### 1. **Running the Full Pipeline (zeta_jobgpu.sh)**

```bash
# Submit job to SLURM
sbatch zeta_jobgpu.sh

# The script will:
# 1. Validate environment and paths
# 2. Run nuclei segmentation with job name "server_gpu_run"
# 3. Locate results automatically using results locator
# 4. Copy segmentation mask to ViT repository
# 5. Run ViT clustering pipeline
```

### 2. **Running Segmentation Only (cs_jobgpu.sh)**

```bash
# Submit segmentation-only job
sbatch cs_jobgpu.sh

# The script will:
# 1. Validate environment and paths
# 2. Run nuclei segmentation with job name "segmentation_only_run"
# 3. Provide detailed results summary
```

### 3. **Testing the Improvements**

```bash
# Run the test suite to validate improvements
python test_server_job_improvements.py
```

## Directory Structure

The improved scripts expect this repository structure:

```
/exports/humgen/cbotos/
├── I-R-Injury-Spatial-Multiomics-Analysis/
│   ├── code/
│   │   └── nuclei_segmentation/
│   │       ├── run_this.py
│   │       └── utils/
│   │           └── results_locator.py
│   ├── results/
│   │   ├── latest -> 20250723_123456_server_gpu_run/
│   │   └── 20250723_123456_server_gpu_run/
│   │       └── masks/
│   │           └── segmentation_masks.npy
│   └── configs/
└── iri_vit/
    └── pipeline.sh
```

## Results Directory Naming

Results directories now follow a predictable pattern:

- **Format**: `YYYYMMDD_HHMMSS_<job_name>`
- **Examples**:
  - `20250723_143022_server_gpu_run`
  - `20250723_143155_segmentation_only_run`

## Latest Results Tracking

The system creates automatic tracking of the latest results:

1. **Symlink** (preferred): `results/latest -> 20250723_143022_server_gpu_run`
2. **Text file** (fallback): `results/latest.txt` containing directory name

## Error Handling

Comprehensive error handling at each stage:

- **Path validation**: Ensures all required directories and files exist
- **Exit code checking**: Validates success of each pipeline stage
- **File verification**: Confirms output files are created and accessible
- **Detailed logging**: Provides clear error messages for debugging

## Backward Compatibility

The improvements maintain backward compatibility:

- Existing configuration files work unchanged
- Default behavior preserved when no job name is specified
- All existing functionality remains available

## Testing

Use the provided test script to validate the improvements:

```bash
python test_server_job_improvements.py
```

The test suite validates:
- Results locator functionality
- Configuration with job names
- Environment variable support
- Path validation
- Error handling

## Benefits

1. **Reliability**: Robust path management and error handling
2. **Predictability**: Consistent results directory naming
3. **Debuggability**: Detailed logging and error messages
4. **Flexibility**: Support for different job types and configurations
5. **Maintainability**: Clean, well-documented code structure

## Future Enhancements

Potential future improvements:
- Configuration file validation
- Automatic cleanup of old results
- Integration with job monitoring systems
- Support for different cluster environments
- Advanced error recovery mechanisms
