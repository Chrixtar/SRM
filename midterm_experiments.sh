#!/bin/bash

# --- Usage Instructions ---
#
# This script runs a series of experiments for the Spatial Reasoning Models (SRM)
# project based on the configurations specified below.
#
# 1. Make the script executable:
#    chmod +x midterm_experiments.sh
#
# 2. Run the script:
#    ./midterm_experiments.sh [num_samples]
#
# Optional Argument:
#   [num_samples]: Override the default number of samples (500) for each experiment.
#                  Useful for quick testing (e.g., ./midterm_experiments.sh 10).
#
# CUDA Out of Memory Errors:
#   If you encounter CUDA Out of Memory (OOM) errors, even with low sample counts,
#   try setting the following environment variable before running the script:
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   This can help PyTorch manage GPU memory more efficiently.
#
# Output:
#   - Progress and results are logged to a file named 'midterm_experiments_YYYY-MM-DD-HH-MM-SS.log'
#     in the same directory where the script is run.
#   - The script will execute 'bash test.sh' for each experimental configuration.
#     Ensure 'test.sh' and the necessary config files/checkpoints are accessible.
#
# --- Configuration ---

# Default number of samples (can be overridden by command line argument)
NUM_SAMPLES=${1:-500}
# Create a log file with a timestamp
LOG_FILE="midterm_experiments_$(date +%Y-%m-%d-%H-%M-%S).log"
EXPERIMENT_CONFIG="ms1000_28"
EXPERIMENT_ID="paper"

# Clear log file or create it if it doesn't exist
> "$LOG_FILE"

echo "Starting midterm experiments at $(date)" | tee -a "$LOG_FILE"
echo "Using NUM_SAMPLES=$NUM_SAMPLES" | tee -a "$LOG_FILE"

# Function to run a single experiment and log details
run_experiment() {
    local difficulty=$1
    local sampler_base=$2
    local test_config=$3
    local top_k=$4
    local max_steps=$5

    # Construct Hydra overrides target key based on sampler type and corrected path
    local sampler_name="${sampler_base}"
    if [ "$sampler_base" == "seq_adaptive" ]; then
        sampler_name="seq_adaptive000"
    fi
    local sampler_override_key="test.${difficulty}.samplers.${sampler_name}"

    # Define base overrides targeting the correct nested structure (e.g., test.hard.num_samples)
    local num_samples_override="test.${difficulty}.num_samples=$NUM_SAMPLES"
    local top_k_override="++${sampler_override_key}.top_k=$top_k"
    local max_steps_override="++${sampler_override_key}.max_steps=$max_steps"

    # Construct the full command with individual overrides
    # Ensure overrides are passed as separate arguments without extra quotes
    local filtered_overrides=("$num_samples_override" "$top_k_override" "$max_steps_override")
    local cmd="bash test.sh $EXPERIMENT_CONFIG $EXPERIMENT_ID $test_config ${filtered_overrides[@]}"

    echo "--------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Running experiment: Difficulty=$difficulty, Sampler=$sampler_base, TestConfig=$test_config, TopK=$top_k, MaxSteps=$max_steps" | tee -a "$LOG_FILE"
    # Log the command
    echo "Command: $cmd" | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"
    local start_time=$(date +%s)

    # Execute command and append stdout/stderr to log file
    # Pass overrides as separate arguments.
    bash test.sh $EXPERIMENT_CONFIG $EXPERIMENT_ID $test_config ${filtered_overrides[@]} >> "$LOG_FILE" 2>&1
    local exit_code=$?

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    echo "Duration: ${duration} seconds" | tee -a "$LOG_FILE"
    if [ $exit_code -ne 0 ]; then
        echo "Error: Experiment failed with exit code $exit_code" | tee -a "$LOG_FILE"
    else
        echo "Experiment completed successfully." | tee -a "$LOG_FILE"
    fi
    echo "--------------------------------------------------" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# --- Experiments --- 

difficulties=("easy" "medium" "hard")

# # TEMPORARY SIMPLE TEST CASE TO CHECK OVERRIDES
# run_experiment "easy" "sim_adaptive" "ms_easy_sim_adaptive" 1 81
# # end script
# exit 0

# Sim Adaptive
for diff in "${difficulties[@]}"; do
    test_config_name="ms_${diff}_sim_adaptive"
    for top_k in 1 3 9; do
        for max_steps in 162 324; do
            run_experiment "$diff" "sim_adaptive" "$test_config_name" "$top_k" "$max_steps"
        done
    done
done

# Seq Adaptive
for diff in "${difficulties[@]}"; do
    test_config_name="ms_${diff}_seq_adaptive000"
    for max_steps in 162 324; do
        # For seq_adaptive, top_k is always 1
        run_experiment "$diff" "seq_adaptive" "$test_config_name" 1 "$max_steps"
    done
done

echo "Finished all midterm experiments at $(date)" | tee -a "$LOG_FILE" 