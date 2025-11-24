# ATLAS-Cellpose HPC Cluster Examples

This directory contains example SLURM batch scripts for running ATLAS-Cellpose on high-performance computing (HPC) clusters.

## Overview

These scripts demonstrate how to submit ATLAS-Cellpose segmentation jobs to SLURM-managed HPC clusters. They handle environment setup, module loading, and proper conda activation in non-interactive batch environments.

## Available Scripts

### 1. GPU Job Script (`cs_jobgpu_for_cluster.sh`)

**Purpose**: Run ATLAS-Cellpose segmentation with GPU acceleration on HPC clusters.

**Key Features**:
- GPU acceleration with CUDA 11.8.
- Optimized for moderate memory requirements (15GB RAM).
- Automatic conda environment activation.
- Comprehensive job logging.

**Resource Requirements**:
- **Partition**: `highmemgpu` (adjust for your cluster).
- **GPU**: 1 GPU (NVIDIA with CUDA support).
- **Memory**: 15GB RAM.
- **CPUs**: 8 cores.
- **Time**: 40 hours maximum.

### 2. CPU Job Script (`cs_jobcpu_for_cluster.sh`)

**Purpose**: Run ATLAS-Cellpose segmentation without GPU (CPU-only mode).

**Key Features**:
- High-memory CPU processing for large images.
- Extensive SLURM job information logging.
- Suitable for clusters without GPU availability.

**Resource Requirements**:
- **Memory**: 400GB RAM (adjust based on image size).
- **CPUs**: 64 cores.
- **Time**: 220 hours maximum.

## Before Using These Scripts

### 1. Customize SLURM Parameters

**IMPORTANT**: These scripts contain example parameters that **must be customized** for your specific HPC cluster:

```bash
# Update these parameters in both scripts:
#SBATCH --mail-user=YOUR_EMAIL@example.com     # Your email address.
#SBATCH --partition=YOUR_PARTITION             # Your cluster's partition name.
#SBATCH --mem=YOUR_MEMORY                      # Adjust based on your image size.
#SBATCH --time=YOUR_TIME_LIMIT                 # Adjust based on expected runtime.
```

### 2. Verify Module Availability

Check if CUDA modules are available on your cluster:

```bash
# List available CUDA modules.
module avail cuda

# Load the appropriate CUDA version.
module load library/cuda/11.8.0/gcc.8.5.0  # Adjust version as needed.
```

If your cluster doesn't use environment modules, remove the `module` commands and ensure CUDA is available in your PATH.

### 3. Verify Conda Installation

Ensure conda is installed and accessible:

```bash
# Check conda installation.
which conda

# Verify environment exists.
conda env list | grep venv310_cellpose3
```

### 4. Adjust Working Directory

The scripts assume they are located in a `cluster_examples/` subdirectory within the ATLAS-Cellpose repository:

```bash
# Current structure assumed:
# /path/to/ATLAS-Cellpose/
# ├── cluster_examples/
# │   ├── cs_jobgpu_for_cluster.sh
# │   └── cs_jobcpu_for_cluster.sh
# ├── run_segmentation_instance.sh
# └── configs/
```

If your directory structure differs, update the `cd ../` line in both scripts.

## Usage Instructions

### Step 1: Copy and Customize Scripts

```bash
# Copy example scripts to your working directory.
cp cs_jobgpu_for_cluster.sh my_gpu_job.sh
cp cs_jobcpu_for_cluster.sh my_cpu_job.sh

# Edit scripts with your parameters.
nano my_gpu_job.sh  # Update email, partition, resources, etc.
```

### Step 2: Configure Pipeline Parameters

Edit the main configuration file or use command-line parameters:

```bash
# Option 1: Edit configuration file directly.
nano ../configs/nuclei_segmentation_config.ini

# Option 2: Pass parameters via run_segmentation_instance.sh.
# Modify the script call in your SLURM script:
./run_segmentation_instance.sh \
    job_name "cluster_run_1" \
    image_path "data/kidney_sample.tif" \
    gpu True \
    cellprob_threshold -12
```

### Step 3: Submit Job to SLURM

```bash
# Submit GPU job.
sbatch my_gpu_job.sh

# Submit CPU job.
sbatch my_cpu_job.sh

# Check job status.
squeue -u $USER

# Monitor job output.
tail -f atlas_cellpose_seg_JOBID.out
```

### Step 4: Monitor Job Progress

```bash
# Check job status.
squeue -u $USER

# View real-time output.
tail -f atlas_cellpose_seg_*.out

# Check for errors.
tail -f atlas_cellpose_seg_*.err

# Cancel job if needed.
scancel JOBID
```

## Customization Examples

### Example 1: Parameter Sweep on Cluster

Run multiple jobs with different parameters:

```bash
# Create multiple job scripts with different parameters.
for threshold in -9 -12 -14; do
    cat > job_threshold_${threshold}.sh <<EOF
#!/bin/bash
#SBATCH --job-name=atlas_thresh_${threshold}
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --partition=highmemgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --time=40:00:00

module load library/cuda/11.8.0/gcc.8.5.0
eval "\$(conda shell.bash hook)"
conda activate venv310_cellpose3

cd /path/to/ATLAS-Cellpose

./run_segmentation_instance.sh \\
    job_name "threshold_${threshold}" \\
    cellprob_threshold ${threshold} \\
    gpu True
EOF
    sbatch job_threshold_${threshold}.sh
done
```

### Example 2: Batch Processing Multiple Images

Process multiple images in parallel:

```bash
# Create job array for multiple images.
cat > batch_process.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=atlas_batch
#SBATCH --array=1-10                    # Process 10 images.
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --partition=highmemgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --time=40:00:00

module load library/cuda/11.8.0/gcc.8.5.0
eval "$(conda shell.bash hook)"
conda activate venv310_cellpose3

cd /path/to/ATLAS-Cellpose

# Get image path from array index.
IMAGE_LIST=(data/image1.tif data/image2.tif data/image3.tif ...)
IMAGE=${IMAGE_LIST[$SLURM_ARRAY_TASK_ID-1]}

./run_segmentation_instance.sh \
    job_name "batch_${SLURM_ARRAY_TASK_ID}" \
    image_path "$IMAGE" \
    gpu True
EOF

sbatch batch_process.sh
```

### Example 3: CPU-Only Processing for Large Images

For very large images that don't fit in GPU memory:

```bash
#!/bin/bash
#SBATCH --job-name=atlas_large_cpu
#SBATCH --mem=500G                      # Large memory for gigapixel images.
#SBATCH --cpus-per-task=128             # Maximum CPU cores.
#SBATCH --time=300:00:00                # Extended time limit.

eval "$(conda shell.bash hook)"
conda activate venv310_cellpose3

cd /path/to/ATLAS-Cellpose

./run_segmentation_instance.sh \
    job_name "large_image_cpu" \
    image_path "data/gigapixel_image.tif" \
    gpu False \
    tile_side_length 512 \
    use_tiling True
```

## Troubleshooting

### Issue: Conda activation fails

**Solution**: Ensure conda is properly initialized in your `.bashrc`:

```bash
# Add to ~/.bashrc if missing.
eval "$(conda shell.bash hook)"
```

### Issue: CUDA module not found

**Solution**: Check available modules and load correct version:

```bash
module avail cuda
module load cuda/YOUR_VERSION
```

### Issue: Out of memory errors

**Solution**: Adjust memory allocation or tile size:

```bash
# Increase memory in SLURM script.
#SBATCH --mem=32G

# Or reduce tile size in pipeline.
./run_segmentation_instance.sh tile_side_length 256
```

### Issue: Job timeout

**Solution**: Increase time limit or optimize parameters:

```bash
# Increase time limit.
#SBATCH --time=100:00:00

# Or optimize processing.
./run_segmentation_instance.sh \
    use_tiling True \
    tile_side_length 512
```

## Best Practices

1. **Test locally first**: Run pipeline on small crop before submitting cluster jobs.
2. **Start with short time limits**: Use conservative time estimates and increase if needed.
3. **Monitor resource usage**: Check `sacct` to optimize future job submissions.
4. **Use job arrays**: Process multiple images efficiently with SLURM job arrays.
5. **Save intermediate results**: Enable QC overlays to validate segmentation quality.
6. **Document parameters**: Keep detailed records of successful parameter combinations.

## Resource Estimation Guidelines

### GPU Jobs

| Image Size | Memory | CPUs | Time | GPU |
|------------|--------|------|------|-----|
| Small (< 2K × 2K) | 8GB | 4 | 1h | 1 |
| Medium (2K-4K × 2K-4K) | 15GB | 8 | 10h | 1 |
| Large (> 4K × 4K) | 32GB | 16 | 40h | 1 |

### CPU Jobs

| Image Size | Memory | CPUs | Time |
|------------|--------|------|------|
| Small (< 2K × 2K) | 32GB | 16 | 5h |
| Medium (2K-4K × 2K-4K) | 128GB | 32 | 50h |
| Large (> 4K × 4K) | 400GB | 64 | 200h |

**Note**: These are conservative estimates. Actual requirements depend on nucleus density, tiling parameters, and cluster performance.

## Additional Resources

- **Main Documentation**: See `../README.md` for complete pipeline documentation.
- **Configuration Guide**: See `../configs/nuclei_segmentation_config.ini` for parameter descriptions.
- **SLURM Documentation**: Consult your cluster's documentation for specific SLURM parameters.

## Contact

For questions about cluster deployment:
- **Email**: botoschristos@gmail.com
- **GitHub**: github.com/ChrisBotos/ATLAS-Cellpose

---

**Note**: These scripts are provided as examples and must be customized for your specific HPC environment. Always consult your cluster's documentation and system administrators for optimal resource allocation and job submission practices.

