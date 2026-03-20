#!/bin/bash
# Author: Christos Botos.
# Affiliation: Leiden University Medical Center.
# GitHub: github.com/ChrisBotos.
# PyPi: pypi.org/user/ChrisBotos.
# Email: botoschristos@gmail.com.
# LinkedIn: www.linkedin.com/in/christos-botos-2369hcty3396.
#
# Project Starting Date: Winter 2025.
#
# Description:
# This script runs multiple instances of the nuclei segmentation pipeline in parallel with potentially
# different parameters. It copies the required configuration file to a unique temporary directory,
# updates the configuration based on parameter-value pairs provided as command-line arguments,
# and launches the segmentation pipeline using the temporary configuration file. The pipeline process PID is
# saved so that if the script is interrupted, the pipeline can be terminated gracefully.
# All output (informational messages and errors) is logged to a log file in the "logs"
# directory located in the project root. Initially, the log file name is constructed using
# the job name provided via the command line; however, once the configuration file is updated, the script
# extracts the job name from the updated configuration and renames the log file accordingly.
# If a log file with the final name already exists, it is overwritten.
#
# Example Usage:
#   ./run_segmentation_instance.sh job_name test_run gpu True diameter 30 flow_threshold 0.9
#
# Contact:
# Feel free to contact me for any questions.

# ------------------------------------------------------------------------------
# Setup Logging.
# ------------------------------------------------------------------------------
# Determine the directory of this script and the project root.
script_dir="$(cd "$(dirname "$0")" && pwd)"
ATLAS_DIR="$(cd "$script_dir/.." && pwd)"

# Set the log directory to be the "logs" folder in the project root.
LOG_DIR="${ATLAS_DIR}/logs/run_segmentation_instance"

# Create the log directory if it does not exist.
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR" || { echo "Error: Could not create log directory $LOG_DIR." >&2; exit 1; }
fi

# Extract the job name from the command-line arguments.
# It is expected that one pair of arguments is: job_name <job_name>.
job_name="default"
for (( i=1; i<=$#; i++ )); do
    if [ "${!i}" == "job_name" ]; then
        next_index=$((i+1))
        job_name="${!next_index}"
        break
    fi
done

# Get the base name of this shell script.
script_basename=$(basename "$0")
# Construct the initial log file path using the job name and script base name.
LOG_FILE="${LOG_DIR}/${job_name}_${script_basename%.sh}.log"
# Overwrite any existing log file.
: > "$LOG_FILE"

# Function to log informational messages to the log file only.
log_info() {
    local message="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') INFO: $message" >> "$LOG_FILE"
}

# Function to log error messages to the log file only.
log_error() {
    local message="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: $message" >> "$LOG_FILE" 2>&1
}

log_info "Logging started for job '$job_name' in script '$script_basename'."

# ------------------------------------------------------------------------------
# Function: update_config.
# This function updates a configuration parameter in the .ini file within a specified directory.
# It creates a backup of the original configuration file if one does not already exist.
# ------------------------------------------------------------------------------
update_config() {
    local parameter="$1"
    local value="$2"
    local config_file_path="$3"
    local found=false

    # Check if the parameter exists in the configuration file.
    if grep -q "^$parameter" "$config_file_path"; then
        # Create a backup file if it does not already exist.
        local backup_file_path="${config_file_path}.bak"
        if [ ! -f "$backup_file_path" ]; then
            cp "$config_file_path" "$backup_file_path" || { log_error "Could not create backup for $config_file_path."; exit 1; }
        fi

        # Update the parameter in the configuration file using sed.
        # Use | as delimiter instead of / to handle file paths with slashes.
        sed -i "s|^\($parameter *= *\).*|\1$value|" "$config_file_path" || { log_error "Failed to update $parameter in $config_file_path."; exit 1; }
        log_info "Updated $parameter to $value in $config_file_path."
        found=true
    fi

    # If the parameter was not found, log an error and exit.
    if [ "$found" = false ]; then
        log_error "Parameter $parameter not found in configuration file $config_file_path."
        exit 1
    fi
}

# ------------------------------------------------------------------------------
# Create a unique temporary directory inside configs for the segmentation run.
# ------------------------------------------------------------------------------
temp_dir="${ATLAS_DIR}/configs/temp_$(date +%s%N)_$$"
mkdir -p "$temp_dir" || { log_error "Could not create temporary directory $temp_dir."; exit 1; }
log_info "Created temporary configuration directory: $temp_dir."

# ------------------------------------------------------------------------------
# Function: cleanup.
# This function cleans up by terminating the segmentation process (if it is still running)
# and removing the temporary configuration directory.
# ------------------------------------------------------------------------------
cleanup() {
    # If the segmentation process ID is set and the process exists, terminate it.
    if [[ -n "$segmentation_pid" ]]; then
        if ps -p "$segmentation_pid" > /dev/null 2>&1; then
            log_info "Killing segmentation process with PID $segmentation_pid."
            kill -TERM "$segmentation_pid" 2>/dev/null
        fi
    fi

    # Remove the temporary configuration directory.
    rm -rf "$temp_dir"
    log_info "Removed temporary directory $temp_dir."
}

# Set a trap to ensure cleanup is executed upon exit or interruption.
trap cleanup EXIT SIGINT SIGTERM

# ------------------------------------------------------------------------------
# Copy the required configuration file into the temporary directory.
# ------------------------------------------------------------------------------
CONFIG_FILE="${ATLAS_DIR}/configs/nuclei_segmentation_config.ini"

if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$temp_dir/nuclei_segmentation_config.ini" || { log_error "Failed to copy $CONFIG_FILE to $temp_dir."; exit 1; }
    log_info "Copied $CONFIG_FILE to $temp_dir."
else
    log_error "Configuration file $CONFIG_FILE does not exist."
    exit 1
fi

TEMP_CONFIG_FILE="$temp_dir/nuclei_segmentation_config.ini"

# ------------------------------------------------------------------------------
# Process command-line parameters to update configuration file.
# This loop updates the configuration file in the temporary directory based on parameter-value pairs.
# ------------------------------------------------------------------------------
while [ $# -gt 1 ]; do
    parameter="$1"
    value="$2"
    update_config "$parameter" "$value" "$TEMP_CONFIG_FILE"
    shift 2
done

# ------------------------------------------------------------------------------
# Update log file name based on job name from the updated configuration.
# ------------------------------------------------------------------------------
job_name_from_config=$(grep "^job_name" "$TEMP_CONFIG_FILE" | awk -F' *= *' '{print $2}')
if [ -n "$job_name_from_config" ]; then
    new_LOG_FILE="${LOG_DIR}/${job_name_from_config}_${script_basename%.sh}.log"
    # Only rename if the new log file path is different from the current one.
    if [ "$LOG_FILE" != "$new_LOG_FILE" ]; then
        mv "$LOG_FILE" "$new_LOG_FILE"
        LOG_FILE="$new_LOG_FILE"
        log_info "Log file renamed to reflect job name: $LOG_FILE."
    else
        log_info "Log file name already matches job name: $LOG_FILE."
    fi
fi

# ------------------------------------------------------------------------------
# Modify run_this.py to use the temporary configuration file.
# ------------------------------------------------------------------------------
# We need to temporarily modify the load_config call to use our temp config.
# Create a temporary wrapper script that sets the config path.
TEMP_RUNNER="${temp_dir}/run_segmentation_temp.py"
cat > "$TEMP_RUNNER" << 'EOF'
#!/usr/bin/env python3
"""
Temporary wrapper script for running segmentation with custom config path.
"""
import sys
import os
from pathlib import Path

# Add project root src directory to path.
# The temp script is in configs/temp_*/run_segmentation_temp.py.
# So we need to go up two levels to get to project root.
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import after path is set.
from atlas_cellpose.utils.project_setup import load_config
from atlas_cellpose.utils.logging_utils import setup_logging
from atlas_cellpose.utils.debug_utils import setup_debug
from atlas_cellpose.pipeline import run_segmentation_pipeline
import traceback
import json

def main():
    """
    Initializes logging, config, debugging, and launches the pipeline.
    """
    try:
        # Get the temp config path from environment variable.
        temp_config_path = os.environ.get("TEMP_CONFIG_PATH")
        if not temp_config_path:
            raise ValueError("TEMP_CONFIG_PATH environment variable not set.")

        # Load config from the temporary location.
        settings, CELLPOSE_PARAMS, PROJECT_DIRS = load_config(temp_config_path)

        # Snapshot parsed settings to disk.
        snapshot_path = Path(settings["output_dir"]) / "settings_snapshot.json"
        serializable = {}
        for key, value in settings.items():
            if isinstance(value, Path):
                serializable[key] = str(value)
            else:
                serializable[key] = value

        with open(snapshot_path, mode="w", encoding="utf-8") as fp:
            json.dump(serializable, fp, indent=2)

        # Determine debug mode explicitly.
        debug_mode = settings.get("debug_mode")
        if debug_mode is None:
            debug_mode = False

        # Set up logging and debug.
        logger = setup_logging(settings["output_dir"], debug_mode)
        snap = setup_debug(settings)

        logger.info("==== Kidney I/R Nuclei Segmentation Pipeline Started ====")

        # Run the pipeline.
        exit_code = run_segmentation_pipeline(
            settings,
            CELLPOSE_PARAMS,
            PROJECT_DIRS,
            logger,
            snap
        )

        return exit_code

    except Exception as e:
        print(f"[FATAL ERROR] {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

log_info "Created temporary runner script: $TEMP_RUNNER."

# ------------------------------------------------------------------------------
# Run the segmentation using the temporary configuration file.
# ------------------------------------------------------------------------------
# Export the temp config path for the wrapper script.
export TEMP_CONFIG_PATH="$TEMP_CONFIG_FILE"

# Activate conda environment and run.
source "$(conda info --base)/etc/profile.d/conda.sh" || { log_error "Could not source conda."; exit 1; }
conda activate venv310_cellpose3 || { log_error "Could not activate conda environment venv310_cellpose3."; exit 1; }

log_info "Running segmentation pipeline with temporary config: $TEMP_CONFIG_FILE."

# Run the temporary wrapper script.
python3 "$TEMP_RUNNER" &

# Save the process ID of the pipeline.
segmentation_pid=$!

log_info "Segmentation pipeline started with PID $segmentation_pid."

# Wait for the pipeline process to finish.
wait $segmentation_pid

exit_code=$?
log_info "Segmentation pipeline finished with exit code $exit_code."

# The trap will ensure that cleanup is executed when the script finishes or is interrupted.
