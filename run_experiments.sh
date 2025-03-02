#!/bin/bash

# Enhanced script to run experiments on two datasets with parallel GPU utilization
# Uses a proper job queue system to run 4 jobs in parallel (one per GPU)

# Base directories and settings
BASE_LOG_DIR="ext_exps"
BASE_ITERATIONS=3000

# Define datasets
declare -a DATASETS=(
    "jumpingjacks:/media/staging1/dhwang/D-Nerf/data/jumpingjacks/:output/baselines/jumpingjack"
    "bouncingballs:/media/staging1/dhwang/D-Nerf/data/bouncingballs/:output/bouncingballsbaseline"
)

# Create experiments directory
mkdir -p "${BASE_LOG_DIR}"

# Set up log directory for job management
JOB_LOGS="${BASE_LOG_DIR}/job_logs"
mkdir -p "${JOB_LOGS}"

# Array to store all experiment configurations
declare -a ALL_EXPERIMENTS=()

# Function to add an experiment to the queue
add_experiment() {
    local dataset_name="$1"
    local data_path="$2"
    local output_dir="$3"
    local exp_name="$4"
    shift 4
    local args="$@"
    
    local full_exp_name="${dataset_name}_${exp_name}"
    
    # Create the full command
    local cmd="python train_extrapolate_ode_render.py -s ${data_path} -m ${output_dir} --eval --is_blender --iterations ${BASE_ITERATIONS} --log_directory ${BASE_LOG_DIR}/${full_exp_name} ${args} > ${JOB_LOGS}/${full_exp_name}.log 2>&1"
    
    # Add to the array
    ALL_EXPERIMENTS+=("$cmd")
}

# Function to run jobs in parallel across 4 GPUs
run_parallel_jobs() {
    local total_jobs=${#ALL_EXPERIMENTS[@]}
    local running_jobs=0
    local completed_jobs=0
    local max_concurrent_jobs=4
    
    echo "Starting parallel execution of $total_jobs jobs across $max_concurrent_jobs GPUs"
    
    # Array to track running job PIDs and their assigned GPUs
    declare -A job_pids
    declare -A job_gpus
    declare -a available_gpus=(0 1 2 3)
    
    # Process all jobs in the queue
    while [ $completed_jobs -lt $total_jobs ]; do
        # Start new jobs if we have available GPUs and pending jobs
        while [ ${#available_gpus[@]} -gt 0 ] && [ $((completed_jobs + running_jobs)) -lt $total_jobs ]; do
            # Get the next available GPU
            gpu=${available_gpus[0]}
            available_gpus=("${available_gpus[@]:1}") # Remove the first element
            
            # Get next job
            job_idx=$((completed_jobs + running_jobs))
            cmd="${ALL_EXPERIMENTS[$job_idx]}"
            
            # Set the GPU and start the job
            echo "Starting job $((job_idx+1))/$total_jobs on GPU $gpu"
            export CUDA_VISIBLE_DEVICES=$gpu
            
            # Run the command but ensure it continues even if the command fails
            # by wrapping it in a subshell that always returns success
            (eval "$cmd" || echo "WARNING: Job $((job_idx+1)) failed with exit code $? - continuing with other jobs" >> ${JOB_LOGS}/failed_jobs.log) &
            pid=$!
            
            # Track the job
            job_pids[$pid]=$pid
            job_gpus[$pid]=$gpu
            ((running_jobs++))
        done
        
        # Check for completed jobs
        for pid in "${!job_pids[@]}"; do
            if ! ps -p $pid > /dev/null; then
                # Job completed, free up the GPU
                gpu=${job_gpus[$pid]}
                available_gpus+=($gpu)
                
                # Update counters
                unset job_pids[$pid]
                unset job_gpus[$pid]
                ((running_jobs--))
                ((completed_jobs++))
                
                echo "Job completed. Progress: $completed_jobs/$total_jobs jobs done. GPU $gpu now available."
            fi
        done
        
        # Don't consume all CPU checking job status
        sleep 5
    done
    
    # Check for any failed jobs
    if [ -f "${JOB_LOGS}/failed_jobs.log" ]; then
        echo "WARNING: Some jobs encountered errors. See ${JOB_LOGS}/failed_jobs.log for details."
        echo "All other jobs completed successfully."
    else
        echo "All $total_jobs jobs completed successfully!"
    fi
}

# Generate all experiment combinations
for dataset_config in "${DATASETS[@]}"; do
    IFS=':' read -r dataset_name data_path output_dir <<< "$dataset_config"
    
    echo "Generating experiments for dataset: ${dataset_name}"
    
    # 1. MODEL ARCHITECTURE EXPERIMENTS
    # Small model - Variational
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "model_size_small_var" \
        --nhead 2 --num_encoder_layers 2 --num_decoder_layers 2 \
        --ode_nhidden 16 --encoder_nhidden 32 --decoder_nhidden 32 \
        --latent_dim 32 --var_inf --noise_std 0.01

    # Small model - Deterministic
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "model_size_small_det" \
        --nhead 2 --num_encoder_layers 2 --num_decoder_layers 2 \
        --ode_nhidden 16 --encoder_nhidden 32 --decoder_nhidden 32 \
        --latent_dim 32

    # Medium model - Variational
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "model_size_medium_var" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 --var_inf --noise_std 0.01

    # Medium model - Deterministic
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "model_size_medium_det" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32

    # Large model - Variational
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "model_size_large_var" \
        --nhead 8 --num_encoder_layers 4 --num_decoder_layers 4 \
        --ode_nhidden 64 --encoder_nhidden 128 --decoder_nhidden 128 \
        --latent_dim 32 --var_inf --noise_std 0.01

    # Large model - Deterministic
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "model_size_large_det" \
        --nhead 8 --num_encoder_layers 4 --num_decoder_layers 4 \
        --ode_nhidden 64 --encoder_nhidden 128 --decoder_nhidden 128 \
        --latent_dim 32

    # 2. LATENT DIMENSION EXPERIMENTS
    for latent_dim in 8 16 64 128; do
        # Variational version
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "latent_dim_${latent_dim}_var" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim ${latent_dim} --var_inf --noise_std 0.01
        
        # Deterministic version
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "latent_dim_${latent_dim}_det" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim ${latent_dim}
    done

    # 3. MODEL VARIANT EXPERIMENTS
    # Transformer - Variational
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "model_transformer_var" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 --var_inf --noise_std 0.01

    # Transformer - Deterministic
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "model_transformer_det" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32

    # Latent ODE-RNN - Variational
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "model_latent_ode_rnn_var" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 --var_inf --noise_std 0.01 --use_latent_ode_rnn

    # Latent ODE-RNN - Deterministic
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "model_latent_ode_rnn_det" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 --use_latent_ode_rnn

    # 4. NUMERICAL PRECISION EXPERIMENTS
    for tol_level in "high" "medium" "low"; do
        if [[ "$tol_level" == "high" ]]; then
            rtol="1e-1"
            atol="1e-1"
        elif [[ "$tol_level" == "medium" ]]; then
            rtol="1e-3"
            atol="1e-3"
        else
            rtol="1e-5"
            atol="1e-5"
        fi
        
        # Variational
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "tolerance_${tol_level}_var" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 --var_inf --noise_std 0.01 \
            --rtol ${rtol} --atol ${atol}
        
        # Deterministic
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "tolerance_${tol_level}_det" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 \
            --rtol ${rtol} --atol ${atol}
    done

    # 5. TRAINING DYNAMICS EXPERIMENTS
    for batch_size in 512 2048 8192 20000; do
        # Variational
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "batch_size_${batch_size}_var" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 --var_inf --noise_std 0.01 \
            --batch_size ${batch_size} --val_batch_size ${batch_size}
        
        # Deterministic
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "batch_size_${batch_size}_det" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 \
            --batch_size ${batch_size} --val_batch_size ${batch_size}
    done

    # Learning rate experiments
    for lr in 1e-2 1e-3 1e-4; do
        # Variational
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "learning_rate_${lr}_var" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 --var_inf --noise_std 0.01 \
            --learning_rate ${lr}
        
        # Deterministic
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "learning_rate_${lr}_det" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 \
            --learning_rate ${lr}
    done

    # 6. WINDOW AND OBSERVATION LENGTH EXPERIMENTS
    for win_obs in "10:5" "20:10" "40:20" "60:40"; do
        IFS=':' read -r window_length obs_length <<< "$win_obs"
        
        # Variational
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "window_obs_${window_length}_${obs_length}_var" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 --var_inf --noise_std 0.01 \
            --window_length ${window_length} --obs_length ${obs_length}
        
        # Deterministic
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "window_obs_${window_length}_${obs_length}_det" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 \
            --window_length ${window_length} --obs_length ${obs_length}
    done

    # 7. LOSS WEIGHTING EXPERIMENTS
    # ODE weights
    for ode_weight in 1 1e-3 1e-5; do
        # Variational
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "ode_weight_${ode_weight}_var" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 --var_inf --noise_std 0.01 \
            --ode_weight ${ode_weight}
        
        # Deterministic
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "ode_weight_${ode_weight}_det" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 \
            --ode_weight ${ode_weight}
    done

    # Render weights
    for render_weight in 1 10 100; do
        # Variational
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "render_weight_${render_weight}_var" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 --var_inf --noise_std 0.01 \
            --render_weight ${render_weight}
        
        # Deterministic
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "render_weight_${render_weight}_det" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 \
            --render_weight ${render_weight}
    done

    # 8. REGULARIZATION WEIGHT EXPERIMENTS
    for reg_weight in 1 1e-3 1e-5; do
        # Variational
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "reg_weight_${reg_weight}_var" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 --var_inf --noise_std 0.01 \
            --reg_weight ${reg_weight}
        
        # Deterministic
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "reg_weight_${reg_weight}_det" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 \
            --reg_weight ${reg_weight}
    done

    # 9. NOISE STANDARD DEVIATION EXPERIMENTS (variational only)
    for noise_std in 0.1 0.01 0.001; do
        add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "noise_std_${noise_std}" \
            --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
            --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
            --latent_dim 32 --var_inf \
            --noise_std ${noise_std}
    done
    
    # 10. ACTIVATION FUNCTION EXPERIMENTS
    # Variational with tanh
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "activation_tanh_var" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 --var_inf --noise_std 0.01 \
        --use_tanh
    
    # Deterministic with tanh
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "activation_tanh_det" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 \
        --use_tanh
    
    # 11. INTERESTING COMBINATIONS
    # Large model with large latent dim - Variational
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "combo_large_model_large_latent_var" \
        --nhead 8 --num_encoder_layers 4 --num_decoder_layers 4 \
        --ode_nhidden 64 --encoder_nhidden 128 --decoder_nhidden 128 \
        --latent_dim 128 --var_inf --noise_std 0.01
    
    # Large model with large latent dim - Deterministic
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "combo_large_model_large_latent_det" \
        --nhead 8 --num_encoder_layers 4 --num_decoder_layers 4 \
        --ode_nhidden 64 --encoder_nhidden 128 --decoder_nhidden 128 \
        --latent_dim 128
    
    # Medium model with low tolerance - Variational
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "combo_medium_model_low_tol_var" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 --var_inf --noise_std 0.01 \
        --rtol 1e-5 --atol 1e-5
    
    # Medium model with low tolerance - Deterministic
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "combo_medium_model_low_tol_det" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 \
        --rtol 1e-5 --atol 1e-5
    
    # Tanh with other settings
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "combo_tanh_large_model_var" \
        --nhead 8 --num_encoder_layers 4 --num_decoder_layers 4 \
        --ode_nhidden 64 --encoder_nhidden 128 --decoder_nhidden 128 \
        --latent_dim 64 --var_inf --noise_std 0.01 \
        --use_tanh
    
    # Tanh with ODE-RNN
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "combo_tanh_ode_rnn_det" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 \
        --use_tanh --use_latent_ode_rnn
    
    # High learning rate with large batch size
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "combo_high_lr_large_batch_var" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 --var_inf --noise_std 0.01 \
        --learning_rate 1e-2 --batch_size 20000 --val_batch_size 20000
    
    # Large window/obs with latent ODE-RNN
    add_experiment "${dataset_name}" "${data_path}" "${output_dir}" "combo_large_window_ode_rnn_var" \
        --nhead 4 --num_encoder_layers 3 --num_decoder_layers 3 \
        --ode_nhidden 32 --encoder_nhidden 64 --decoder_nhidden 64 \
        --latent_dim 32 --var_inf --noise_std 0.01 \
        --window_length 60 --obs_length 40 --use_latent_ode_rnn
done

# Print summary of experiments
echo "Generated ${#ALL_EXPERIMENTS[@]} experiment configurations."
echo "Ready to run experiments in parallel across 4 GPUs."
echo "Output logs will be stored in: ${JOB_LOGS}"
echo

# Run all jobs in parallel
run_parallel_jobs

echo "All experiments completed successfully!"