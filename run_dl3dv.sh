#!/bin/bash

# Standalone script to run simple_trainer_deblur.py
# Dedicated to running simple_trainer_deblur.py across all scenes

# Base paths
BASE_DIR=""
DATA_DIR=""
RESULTS_DIR=""

# Script path
SCRIPT_TRAINER="$BASE_DIR/simple_deblur_difix.py"

# Set GPU device
export CUDA_VISIBLE_DEVICES=3

# Create results directory
mkdir -p "$RESULTS_DIR"


declare -a DL3DV_SCENES=(
    "0001"
    "0002"
    "0003"
    "0004"
    "0005"
)

# Colored output functions
print_header() {
    echo -e "\n\033[1;34m========================================\033[0m"
    echo -e "\033[1;34m$1\033[0m"
    echo -e "\033[1;34m========================================\033[0m\n"
}

print_info() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Function to run a single experiment
run_trainer_experiment() {
    local scene_name=$1
    local data_path=$2
    
    local result_dir="$RESULTS_DIR/3views200/${scene_name}"
    
    
    # Create result directory
    mkdir -p "$result_dir"
    #
    # Run training
    cd "$BASE_DIR"
    DATA_FACTOR=4
    SCALE_FACTOR=0.25
    TRAJ_TYPE="bezier"
     #--virtual-view-st--virtual-view-start-step 2000 \
     #   --virtual-view-interval 250 art-step 1500 \
      #  --virtual-view-interval 200 \
      #--max-steps 7000 \
    python "$SCRIPT_TRAINER" default \
        --data-dir "$data_path" \
        --result-dir "$result_dir" \
        --disable-viewer \
        --virtual-view-start-step 1500 \
        --virtual-view-interval 200 \
        --max-steps 7000 \
        --camera-optimizer.mode $TRAJ_TYPE \
        --train-indices 5 15 25 \
        2>&1 | tee "${result_dir}_training.log"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_info "✅ Experiment completed: $scene_name with simple_trainer_deblur.py"
    else
        print_error "❌ Experiment failed: $scene_name with simple_trainer_deblur.py (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Main experiment loop
main() {
    print_header "Start simple_trainer_deblur.py experiments for all scenes (GPU 3)"
    
    local total_experiments=0
    local successful_experiments=0
    local failed_experiments=0
    
    # Process bad-nerf-gtK-colmap-nvs scene
    print_header "Process bad-nerf-gtK-colmap-nvs scene"

    
    for scene in "${DL3DV_SCENES[@]}"; do
        local data_path="$DATA_DIR/$scene/3views"
        
        if [ ! -d "$data_path" ]; then
            print_warning "Skipping non-existent scene: $data_path"
            continue
        fi
        
        total_experiments=$((total_experiments + 1))
        
        if run_trainer_experiment "$scene" "$data_path"; then
            successful_experiments=$((successful_experiments + 1))
        else
            failed_experiments=$((failed_experiments + 1))
        fi
        
        # Wait a bit to free GPU memory
        print_info "⏸️  Waiting for GPU memory to be released..."
        sleep 10
    done
    
    # Final statistics
    print_header "simple_trainer_deblur.py experiment summary"
    print_info "Total experiments: $total_experiments"
    print_info "Successful experiments: $successful_experiments"
    print_info "Failed experiments: $failed_experiments"
    
    if [ $total_experiments -gt 0 ]; then
        local success_rate=$(echo "scale=2; $successful_experiments * 100 / $total_experiments" | bc 2>/dev/null || echo "N/A")
        print_info "Success rate: ${success_rate}%"
    else
        print_info "Success rate: 0%"
    fi
    
    # Generate experiment report
    local report_file="$RESULTS_DIR/trainer_experiment_report_$(date +%Y%m%d_%H%M%S).txt"
    {
        echo "simple_trainer_deblur.py experiment report - $(date)"
        echo "============================================="
        echo "GPU: 3"
        echo "Script: simple_trainer_deblur.py"
        echo "Total experiments: $total_experiments"
        echo "Successful experiments: $successful_experiments"
        echo "Failed experiments: $failed_experiments"
        if [ $total_experiments -gt 0 ]; then
            echo "Success rate: $(echo "scale=2; $successful_experiments * 100 / $total_experiments" | bc 2>/dev/null || echo "N/A")%"
        else
            echo "Success rate: 0%"
        fi
        echo ""
        echo "Experiment scene list:"
        echo "-------------"
        
        for scene in "${BAD_NERF_SCENES[@]}" "${REAL_BLUR_SCENES[@]}" "${DL3DV_SCENES[@]}"; do
            echo "- $scene"
        done
        
    } > "$report_file"
    
    print_info "Experiment report saved to: $report_file"
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check if trainer script exists
    if [ ! -f "$SCRIPT_TRAINER" ]; then
        print_error "Script not found: $SCRIPT_TRAINER"
        exit 1
    fi
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        print_error "Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    # Check GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not available, cannot check GPU status"
    else
        print_info "GPU 3 status:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | head -1
    fi
    
    print_info "Dependency check complete"
}

# Script entry
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --check        Only check dependencies, do not run experiments"
    echo ""
    echo "Notes:"
    echo "  This script runs simple_trainer_deblur.py across all scenes (using GPU 3)"
    echo "  Total runs: $(( ${#BAD_NERF_SCENES[@]} + ${#REAL_BLUR_SCENES[@]} + ${#DL3DV_SCENES[@]} ))"
    exit 0
fi

if [ "$1" = "--check" ]; then
    check_dependencies
    exit 0
fi

# Run main program
check_dependencies
main
