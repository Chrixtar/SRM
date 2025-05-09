#!/bin/bash

# --- Usage Instructions ---
#
# This script runs a series of experiments for the Spatial Reasoning Models (SRM)
# project based on the configurations specified below.
#
# 1. Make the script executable:
#    chmod +x final_experiments.sh
#
# 2. Run the script:
#    ./final_experiments.sh [num_samples]
#
# Optional Argument:
#   [num_samples]: Override the default number of samples (500) for each experiment.
#                  Useful for quick testing (e.g., ./final_experiments.sh 10).
#
# CUDA Out of Memory Errors:
#   If you encounter CUDA Out of Memory (OOM) errors, even with low sample counts,
#   try setting the following environment variable before running the script:
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   This can help PyTorch manage GPU memory more efficiently.
#
# Output:
#   - Progress and results are logged to a file named 'final_experiments_YYYY-MM-DD-HH-MM-SS.log'
#     in the same directory where the script is run.
#   - The script will execute 'bash test.sh' for each experimental configuration.
#     Ensure 'test.sh' and the necessary config files/checkpoints are accessible.
#
# --- Configuration ---

# Default number of samples (can be overridden by command line argument)
NUM_SAMPLES=${1:-500}
# Create a log file with a timestamp
LOG_FILE="final_experiments_$(date +%Y-%m-%d-%H-%M-%S).log"
EXPERIMENT_CONFIG="ms1000_28"
EXPERIMENT_ID="paper"

# --- Parameter Lists ---
difficulties=("easy" "medium" "hard")
uncertainty_powers=(1.0 3.0 9.0)
top_ks=(1 3 9)
max_steps_list=(162 324)

# Clear log file or create it if it doesn't exist
> "$LOG_FILE"

echo "Starting final experiments at $(date)" | tee -a "$LOG_FILE"
echo "Using NUM_SAMPLES=$NUM_SAMPLES" | tee -a "$LOG_FILE"

# Function to run a single experiment and log details
run_experiment() {
    local difficulty=$1
    local sampler_base=$2
    local test_config=$3
    local top_k=$4           # 4th arg: top_k (used by sim_adaptive, seq_adaptive)
    local max_steps=$5
    local uncertainty_power=$6 # 6th arg: uncertainty_power (used by sim_weighted_adaptive)

    # Construct Hydra overrides target key based on sampler type and corrected path
    local sampler_name="${sampler_base}"
    if [ "$sampler_base" == "seq_adaptive" ]; then
        sampler_name="seq_adaptive000"
    fi
    local sampler_override_key="test.${difficulty}.samplers.${sampler_name}"

    # Define base overrides targeting the correct nested structure
    local num_samples_override="test.${difficulty}.num_samples=$NUM_SAMPLES"
    local max_steps_override="++${sampler_override_key}.max_steps=$max_steps"
    local specific_param_override=""
    local specific_param_name=""
    local specific_param_value=""

    # Determine the specific parameter override based on the sampler type
    if [ "$sampler_base" == "sim_weighted_adaptive" ]; then
        specific_param_name="uncertainty_power"
        specific_param_value="$uncertainty_power"
        specific_param_override="++${sampler_override_key}.${specific_param_name}=${specific_param_value}"
    elif [ "$sampler_base" == "sim_adaptive" ] || [ "$sampler_base" == "seq_adaptive" ]; then
        specific_param_name="top_k"
        specific_param_value="$top_k"
        specific_param_override="++${sampler_override_key}.${specific_param_name}=${specific_param_value}"
    else
        echo "Error: Unknown sampler base '$sampler_base'" | tee -a "$LOG_FILE"
        return 1
    fi

    # Construct the full command with individual overrides
    # Combine all overrides into an array
    local filtered_overrides=("$num_samples_override" "$specific_param_override" "$max_steps_override")
    local cmd="bash test.sh $EXPERIMENT_CONFIG $EXPERIMENT_ID $test_config ${filtered_overrides[@]}"

    echo "--------------------------------------------------" | tee -a "$LOG_FILE"
    # Update log message to show the correct parameter name and value
    echo "Running experiment: Difficulty=$difficulty, Sampler=$sampler_base, TestConfig=$test_config, ${specific_param_name}=${specific_param_value}, MaxSteps=$max_steps" | tee -a "$LOG_FILE"
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

# difficulties=("easy" "medium" "hard") # Moved to top

# # TEMPORARY SIMPLE TEST CASE TO CHECK OVERRIDES
# run_experiment "easy" "sim_adaptive" "ms_easy_sim_adaptive" 1 81
# # end script
# exit 0

# Sim Adaptive Weighted
for diff in "${difficulties[@]}"; do
    test_config_name="ms_${diff}_sim_weighted_adaptive"
    for uncertainty_power in "${uncertainty_powers[@]}"; do
        for max_steps in "${max_steps_list[@]}"; do
            # Pass uncertainty_power as 6th arg, use placeholder '_' for top_k (4th arg)
            run_experiment "$diff" "sim_weighted_adaptive" "$test_config_name" "_" "$max_steps" "$uncertainty_power"
        done
    done
done 

# Sim Adaptive
for diff in "${difficulties[@]}"; do
    test_config_name="ms_${diff}_sim_adaptive"
    for top_k in "${top_ks[@]}"; do
        for max_steps in "${max_steps_list[@]}"; do
            # Pass top_k as 4th arg, use placeholder '_' for uncertainty_power (6th arg)
            run_experiment "$diff" "sim_adaptive" "$test_config_name" "$top_k" "$max_steps" "_"
        done
    done
done

# Seq Adaptive
for diff in "${difficulties[@]}"; do
    test_config_name="ms_${diff}_seq_adaptive000"
    for max_steps in "${max_steps_list[@]}"; do
        # For seq_adaptive, top_k is always 1
        # Pass top_k (1) as 4th arg, use placeholder '_' for uncertainty_power (6th arg)
        run_experiment "$diff" "seq_adaptive" "$test_config_name" 1 "$max_steps" "_"
    done
done

echo "Finished all final experiments at $(date)" | tee -a "$LOG_FILE" 