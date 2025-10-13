#!/bin/bash
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Test_Focal_Tversky
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg

################################################################################
# Phase 1B Test 1: Focal Tversky Loss
################################################################################
#
# Tests if focal_tversky loss handles class imbalance better than combined loss
#
# Expected improvements over combined loss:
# - Best Val Jaccard: 13.8% → 20-30% (45-115% better)
# - Final Val Jaccard: 3.0% → 10-20% (233-567% better)
# - Degradation: 78% → <50% (better stability)
# - Overfitting gap: 10.5× → <5× (less overfitting)
#
# Expected runtime: 30-60 minutes
#
################################################################################

echo "=============================================================================="
echo "Phase 1B Test 1: Focal Tversky Loss"
echo "=============================================================================="
echo ""
echo "Job started: $(date)"
echo "Job ID: $PBS_JOBID"
echo ""

cd /home/svu/phyzxi/scratch/unet-HPC
echo "Working directory: $(pwd)"
echo ""

################################################################################
# GPU Configuration
################################################################################

export CUDA_VISIBLE_DEVICES=0
export TF_ENABLE_ONEDNN_OPTS=1
export TF_CPP_MIN_LOG_LEVEL=1
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC_SIZE_BYTES=536870912
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_MAX_ALLOCATION_SIZE_BYTES=8589934592

echo "GPU Configuration: FP32, memory growing"
nvidia-smi
echo ""

################################################################################
# Load Environment
################################################################################

module load singularity
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

echo "Singularity image: $image"
echo ""

################################################################################
# Pre-flight Checks
################################################################################

echo "=============================================================================="
echo "Pre-flight Checks"
echo "=============================================================================="

if [ -f "validate_focal_tversky.py" ]; then
    echo "  ✓ validate_focal_tversky.py"
else
    echo "  ✗ ERROR: validate_focal_tversky.py not found!"
    exit 1
fi

if [ -f "loss_functions_fixed.py" ]; then
    echo "  ✓ loss_functions_fixed.py"
else
    echo "  ✗ ERROR: loss_functions_fixed.py not found!"
    exit 1
fi

if [ -d "dataset_shrunk_masks" ]; then
    echo "  ✓ dataset_shrunk_masks"
else
    echo "  ✗ ERROR: dataset_shrunk_masks not found!"
    exit 1
fi

echo ""

################################################################################
# Run Test
################################################################################

echo "=============================================================================="
echo "Running Focal Tversky Test"
echo "=============================================================================="
echo ""
echo "This tests if focal_tversky loss handles class imbalance better."
echo ""
echo "Why focal_tversky?"
echo "  - Designed for severe class imbalance (92% background)"
echo "  - Penalizes false negatives more (α=0.7, β=0.3)"
echo "  - Focuses on hard examples (γ=1.33)"
echo "  - Better for small object detection (microbeads)"
echo ""
echo "Expected improvements:"
echo "  ✓ Val Jaccard: 3% → 10-20%"
echo "  ✓ Degradation: 78% → <50%"
echo "  ✓ Overfitting: 10.5× → <5×"
echo ""
echo "=============================================================================="
echo ""

singularity exec $image python3 validate_focal_tversky.py

test_status=$?

echo ""
echo "=============================================================================="
echo "Test Complete"
echo "=============================================================================="
echo ""

################################################################################
# Post-Test Analysis
################################################################################

if [ $test_status -eq 0 ]; then
    echo "✓ Test script completed successfully (exit code: 0)"
    echo ""

    output_dir=$(find . -maxdepth 1 -type d -name "validation_focal_tversky_*" | sort -r | head -1)

    if [ -n "$output_dir" ]; then
        echo "Output directory: $output_dir"
        echo ""

        if [ -f "$output_dir/test_summary.json" ]; then
            echo "--- Test Summary ---"
            cat "$output_dir/test_summary.json"
            echo ""

            test_passed=$(grep -o '"test_passed": [a-z]*' "$output_dir/test_summary.json" | cut -d' ' -f2)

            if [ "$test_passed" = "true" ]; then
                echo "=============================================================================="
                echo "✓✓✓ FOCAL TVERSKY TEST PASSED ✓✓✓"
                echo "=============================================================================="
                echo ""
                echo "Focal Tversky loss handles class imbalance better than combined loss!"
                echo ""
                echo "Next steps:"
                echo "  1. Use focal_tversky loss in full hyperparameter search"
                echo "  2. Consider combining with stronger regularization"
                echo "  3. Proceed to Phase 2 with this loss function"
                echo ""
            else
                echo "=============================================================================="
                echo "⚠⚠⚠ FOCAL TVERSKY TEST: PARTIAL ⚠⚠⚠"
                echo "=============================================================================="
                echo ""
                echo "Focal Tversky shows some improvement but not enough."
                echo "Consider: Stronger regularization or smaller model."
                echo ""
            fi
        fi

        if [ -f "$output_dir/training_history.csv" ]; then
            echo ""
            echo "--- Training History (First 5 epochs) ---"
            head -6 "$output_dir/training_history.csv" | column -t -s,
            echo ""
            echo "--- Training History (Last 5 epochs) ---"
            tail -5 "$output_dir/training_history.csv" | column -t -s,
            echo ""
        fi

        echo "--- Output Files ---"
        ls -lh "$output_dir/"
        echo ""

    fi

else
    echo "✗ Test script failed (exit code: $test_status)"
    echo ""
fi

################################################################################
# GPU Status After Test
################################################################################

echo "=============================================================================="
echo "GPU Status After Test"
echo "=============================================================================="
nvidia-smi
echo ""

################################################################################
# Summary
################################################################################

echo "=============================================================================="
echo "Job Summary"
echo "=============================================================================="
echo "Job ID: $PBS_JOBID"
echo "Completed: $(date)"
echo "Exit code: $test_status"
echo ""

if [ $test_status -eq 0 ]; then
    echo "Status: ✓ Completed successfully"
else
    echo "Status: ✗ Failed (exit code: $test_status)"
fi

echo "=============================================================================="
echo ""

exit $test_status
