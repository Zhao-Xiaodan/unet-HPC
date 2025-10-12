#!/bin/bash
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Validate_Training_Fixes
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg

################################################################################
# Phase 1: Quick Validation of Training Fixes
################################################################################
#
# This script tests all 5 solutions to the FP16/NaN training failure:
#
# Solution 1: Disable mixed precision (FP16 → FP32)
# Solution 2: Increase smoothing constants (1e-6 → 1e-3)
# Solution 3: Add NaN detection callback
# Solution 4: Add gradient clipping
# Solution 5: Improve loss function stability
#
# Expected runtime: 30-60 minutes
# Expected memory: 8-12 GB GPU, 32 GB RAM
#
# SUCCESS CRITERIA:
# ✓ No NaN or inf in loss values
# ✓ Loss decreases over epochs
# ✓ Validation Jaccard > 0.25 by epoch 20
# ✓ Smooth convergence curves
#
################################################################################

echo "=============================================================================="
echo "Phase 1: Quick Validation of Training Fixes"
echo "=============================================================================="
echo ""
echo "Job started: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Hostname: $(hostname)"
echo "Working directory: $PBS_O_WORKDIR"
echo ""

# Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "Current directory: $(pwd)"
echo ""

################################################################################
# GPU Configuration
################################################################################

export CUDA_VISIBLE_DEVICES=0

# IMPORTANT: NO mixed precision
# We're testing FP32 to verify it fixes the NaN issue
export TF_ENABLE_ONEDNN_OPTS=1
export TF_CPP_MIN_LOG_LEVEL=1

# GPU memory settings (conservative for single model training)
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private

# Memory settings for FP32 training
# More memory needed than FP16, but ensures stability
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC_SIZE_BYTES=536870912  # 512MB prealloc
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_MAX_ALLOCATION_SIZE_BYTES=8589934592  # 8GB max

echo "=============================================================================="
echo "GPU Configuration"
echo "=============================================================================="
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "TensorFlow GPU memory: Growing (not pre-allocated)"
echo "Precision: FP32 (mixed precision DISABLED)"
echo ""
nvidia-smi
echo ""

################################################################################
# Environment Setup
################################################################################

echo "=============================================================================="
echo "Loading Environment"
echo "=============================================================================="

# Load Singularity module
module load singularity

# Use modern TensorFlow container with GPU support
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

echo "Singularity image: $image"
echo ""

################################################################################
# Pre-flight Checks
################################################################################

echo "=============================================================================="
echo "Pre-flight Checks"
echo "=============================================================================="

# Check dataset
if [ -d "dataset_shrunk_masks" ]; then
    echo "✓ Dataset directory found: dataset_shrunk_masks"
    num_images=$(find dataset_shrunk_masks/images -type f | wc -l)
    num_masks=$(find dataset_shrunk_masks/masks -type f | wc -l)
    echo "  Images: $num_images files"
    echo "  Masks: $num_masks files"
else
    echo "✗ ERROR: Dataset directory not found!"
    echo "  Expected: dataset_shrunk_masks/"
    exit 1
fi

# Check required Python files
echo ""
echo "Checking required files..."
required_files=(
    "validate_training_fixes.py"
    "loss_functions_fixed.py"
    "model_architectures.py"
)

all_found=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ ERROR: $file not found!"
        all_found=false
    fi
done

if [ "$all_found" = false ]; then
    echo ""
    echo "✗ Missing required files. Cannot proceed."
    exit 1
fi

echo ""

################################################################################
# Test Loss Functions First
################################################################################

echo "=============================================================================="
echo "Testing Loss Functions (Pre-check)"
echo "=============================================================================="
echo "This verifies the loss functions are numerically stable before training..."
echo ""

singularity exec $image python3 loss_functions_fixed.py

loss_test_status=$?

if [ $loss_test_status -eq 0 ]; then
    echo ""
    echo "✓ Loss function test PASSED"
    echo ""
else
    echo ""
    echo "✗ Loss function test FAILED (exit code: $loss_test_status)"
    echo "Cannot proceed with training if loss functions are unstable."
    exit 1
fi

################################################################################
# Run Validation Training
################################################################################

echo "=============================================================================="
echo "Starting Validation Training"
echo "=============================================================================="
echo ""
echo "Configuration:"
echo "  Architecture: U-Net"
echo "  Batch size: 4"
echo "  Loss function: Combined (Dice + Focal)"
echo "  Epochs: 20"
echo "  Precision: FP32 (NO mixed precision)"
echo ""
echo "Expected output:"
echo "  ✓ No NaN or inf in loss values"
echo "  ✓ Loss: ~0.5 → ~0.3 (decreasing)"
echo "  ✓ Validation Jaccard: ~0.15 → ~0.30+ (increasing)"
echo ""
echo "Estimated time: 30-60 minutes"
echo "=============================================================================="
echo ""

# Run validation script
singularity exec $image python3 validate_training_fixes.py

training_status=$?

echo ""
echo "=============================================================================="
echo "Training Complete"
echo "=============================================================================="
echo ""

################################################################################
# Post-Training Analysis
################################################################################

if [ $training_status -eq 0 ]; then
    echo "✓ Training script completed successfully (exit code: 0)"
    echo ""

    # Find the output directory
    output_dir=$(find . -maxdepth 1 -type d -name "validation_fixes_*" | sort -r | head -1)

    if [ -n "$output_dir" ]; then
        echo "Output directory: $output_dir"
        echo ""

        # Check if validation summary exists
        if [ -f "$output_dir/validation_summary.json" ]; then
            echo "--- Validation Summary ---"
            cat "$output_dir/validation_summary.json"
            echo ""

            # Extract validation status
            validation_passed=$(grep -o '"validation_passed": [a-z]*' "$output_dir/validation_summary.json" | cut -d' ' -f2)

            if [ "$validation_passed" = "true" ]; then
                echo "=============================================================================="
                echo "✓✓✓ VALIDATION PASSED ✓✓✓"
                echo "=============================================================================="
                echo ""
                echo "All fixes are working correctly!"
                echo "Training is numerically stable with FP32."
                echo ""
                echo "Next steps:"
                echo "  1. Review training history: $output_dir/training_history.csv"
                echo "  2. Apply these fixes to full hyperparameter search"
                echo "  3. Run complete search with 30 configurations"
                echo ""
            else
                echo "=============================================================================="
                echo "⚠⚠⚠ VALIDATION NEEDS IMPROVEMENT ⚠⚠⚠"
                echo "=============================================================================="
                echo ""
                echo "Training is stable (no NaN), but performance could be better."
                echo "Review the results and consider adjustments."
                echo ""
            fi
        else
            echo "⚠ Warning: validation_summary.json not found"
            echo "Training may have completed but results are unclear."
        fi

        # Show training history preview
        if [ -f "$output_dir/training_history.csv" ]; then
            echo ""
            echo "--- Training History (First 5 epochs) ---"
            head -6 "$output_dir/training_history.csv" | column -t -s,
            echo ""
            echo "--- Training History (Last 5 epochs) ---"
            tail -5 "$output_dir/training_history.csv" | column -t -s,
            echo ""
        fi

        # Directory contents
        echo "--- Output Files ---"
        ls -lh "$output_dir/"
        echo ""

    else
        echo "⚠ Warning: Could not find output directory"
        echo "Training may have failed or been interrupted."
    fi

else
    echo "✗ Training script failed (exit code: $training_status)"
    echo ""
    echo "Possible causes:"
    echo "  - NaN detected during training (fixes didn't work)"
    echo "  - Out of memory (try reducing batch size)"
    echo "  - Missing dependencies"
    echo "  - Dataset issues"
    echo ""
    echo "Check the log output above for error messages."
fi

################################################################################
# GPU Status After Training
################################################################################

echo "=============================================================================="
echo "GPU Status After Training"
echo "=============================================================================="
nvidia-smi
echo ""

################################################################################
# Final Summary
################################################################################

echo "=============================================================================="
echo "Job Summary"
echo "=============================================================================="
echo "Job ID: $PBS_JOBID"
echo "Started: $(date)"
echo "Exit code: $training_status"
echo ""

if [ $training_status -eq 0 ]; then
    echo "Status: ✓ Completed successfully"
else
    echo "Status: ✗ Failed (exit code: $training_status)"
fi

echo "=============================================================================="
echo ""

exit $training_status
