#!/bin/bash
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -k oed
#PBS -N SmallModelTest
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg

################################################################################
# Small Model Validation - Priority 1 Test
################################################################################
#
# PURPOSE:
#   Test if reducing model complexity improves validation performance
#
# HYPOTHESIS:
#   Current model (31M params) is too complex for dataset (83 images)
#   Reducing to filters=16 (~2M params) should improve generalization
#
# KEY CHANGES FROM PHASE 1:
#   - filters: 64 → 16 (reduces params from 31M to ~2M)
#   - dropout: 0.3 → 0.5 (stronger regularization)
#   - loss: combined (same as Phase 1 for fair comparison)
#
# EXPECTED OUTCOME:
#   - Best Val Jaccard: 20-30% (vs Phase 1's 13.8%)
#   - Overfitting gap: 3-5× (vs Phase 1's 10.5×)
#   - Best epoch > 1 (vs Phase 1's epoch 1)
#   - Stable training without immediate collapse
#
# ESTIMATED TIME: 2 hours
#
################################################################################

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    SMALL MODEL VALIDATION - PRIORITY 1                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Hypothesis: Reducing model complexity will improve validation performance"
echo "Approach: filters=64→16 (31M→2M params), dropout=0.3→0.5"
echo ""
echo "Starting at: $(date)"
echo ""

################################################################################
# Environment Setup
################################################################################

echo "─────────────────────────────────────────────────────────────────────────────"
echo "1. Environment Setup"
echo "─────────────────────────────────────────────────────────────────────────────"

# CRITICAL: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC || exit 1
echo "✓ Working directory: $(pwd)"

# GPU Configuration - FP32 (NO mixed precision)
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

# Memory management for FP32
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC_SIZE_BYTES=536870912  # 512MB
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_MAX_ALLOCATION_SIZE_BYTES=4294967296  # 4GB (smaller model needs less)

echo "✓ GPU Configuration:"
echo "  - Precision: FP32 (mixed precision disabled)"
echo "  - GPU Memory: 4GB max (reduced for smaller model)"
echo "  - CUDA device: 0"

################################################################################
# Load Modules
################################################################################

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "2. Loading Modules"
echo "─────────────────────────────────────────────────────────────────────────────"

module load singularity
echo "✓ Singularity loaded"

# TensorFlow container
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

if [ ! -f "$image" ]; then
    echo "❌ ERROR: TensorFlow container not found: $image"
    exit 1
fi
echo "✓ TensorFlow container: $(basename $image)"

################################################################################
# Verify Files
################################################################################

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "3. Verifying Required Files"
echo "─────────────────────────────────────────────────────────────────────────────"

# Check script
if [ ! -f "validate_small_model.py" ]; then
    echo "❌ ERROR: validate_small_model.py not found"
    exit 1
fi
echo "✓ validate_small_model.py"

# Check model definitions
if [ ! -f "models_224_225_226.py" ]; then
    echo "❌ ERROR: models_224_225_226.py not found"
    exit 1
fi
echo "✓ models_224_225_226.py"

# Check loss functions
if [ ! -f "loss_functions_fixed.py" ]; then
    echo "❌ ERROR: loss_functions_fixed.py not found"
    exit 1
fi
echo "✓ loss_functions_fixed.py"

# Check dataset
if [ ! -d "dataset_full_stack/images" ] || [ ! -d "dataset_full_stack/masks" ]; then
    echo "❌ ERROR: Dataset directories not found"
    exit 1
fi

img_count=$(ls dataset_full_stack/images/*.tif 2>/dev/null | wc -l)
mask_count=$(ls dataset_full_stack/masks/*.tif 2>/dev/null | wc -l)

echo "✓ Dataset:"
echo "  - Images: $img_count"
echo "  - Masks: $mask_count"

if [ "$img_count" -eq 0 ] || [ "$mask_count" -eq 0 ]; then
    echo "❌ ERROR: No dataset files found"
    exit 1
fi

################################################################################
# Test Configuration
################################################################################

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "4. Test Configuration"
echo "─────────────────────────────────────────────────────────────────────────────"

echo "Model Configuration:"
echo "  - Architecture:  U-Net"
echo "  - Filters:       16 (KEY: Reduced from 64)"
echo "  - Dropout:       0.5 (INCREASED from 0.3)"
echo "  - Batch size:    4"
echo "  - Loss function: combined (BCE + Dice)"
echo "  - Epochs:        20"
echo ""
echo "Expected Model Size:"
echo "  - Parameters:    ~2M (vs Phase 1's 31M)"
echo "  - Reduction:     93% fewer parameters"
echo "  - Params/image:  ~24,000 (vs Phase 1's 378,117)"
echo ""
echo "Success Criteria:"
echo "  ✓ No NaN/Inf detected"
echo "  ✓ Best Val Jaccard ≥ 15%"
echo "  ✓ Overfitting gap ≤ 7.0×"
echo "  ✓ Degradation ≤ 50%"
echo "  ✓ Best epoch ≥ 2 (improvement beyond first epoch)"

################################################################################
# Pre-Training System Check
################################################################################

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "5. Pre-Training System Check"
echo "─────────────────────────────────────────────────────────────────────────────"

# GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "⚠ GPU info unavailable"

# Disk space
echo "Disk space:"
df -h . | tail -1

################################################################################
# Run Validation
################################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                          STARTING VALIDATION                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

start_time=$(date +%s)

singularity exec --nv $image python3 validate_small_model.py

exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))
duration_min=$((duration / 60))
duration_sec=$((duration % 60))

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                          VALIDATION COMPLETE                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Exit code: $exit_code"
echo "Duration: ${duration_min}m ${duration_sec}s"
echo "Finished at: $(date)"

################################################################################
# Post-Run Analysis
################################################################################

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "6. Post-Run Analysis"
echo "─────────────────────────────────────────────────────────────────────────────"

# Find latest results directory
latest_dir=$(ls -dt validation_small_model_* 2>/dev/null | head -1)

if [ -z "$latest_dir" ]; then
    echo "⚠ No results directory found"
    exit $exit_code
fi

echo "Results directory: $latest_dir"
echo ""

# Check for key files
if [ -f "$latest_dir/test_summary.json" ]; then
    echo "✓ test_summary.json created"

    # Extract key metrics using Python
    singularity exec $image python3 -c "
import json
import sys

try:
    with open('$latest_dir/test_summary.json', 'r') as f:
        data = json.load(f)

    print('Key Results:')
    print(f'  Best Val Jaccard:    {data[\"best_val_jacard\"]*100:.1f}%')
    print(f'  Final Val Jaccard:   {data[\"final_val_jacard\"]*100:.1f}%')
    print(f'  Overfitting Gap:     {data[\"overfitting_gap\"]:.1f}×')
    print(f'  Degradation:         {data[\"degradation\"]*100:.0f}%')
    print(f'  Best Epoch:          {data[\"best_epoch\"]+1}')
    print(f'  NaN Detected:        {\"YES\" if data[\"nan_detected\"] else \"NO\"}')
    print(f'  Test Passed:         {\"YES\" if data[\"test_passed\"] else \"NO\"}')
    print()
    print('Comparison with Phase 1:')
    comp = data['comparison']
    print(f'  Best Val Jaccard:    {comp[\"phase1_best_jacard\"]*100:.1f}% → {comp[\"this_test_best_jacard\"]*100:.1f}% ({comp[\"improvement_vs_phase1\"]*100:+.0f}%)')
    print(f'  Overfitting Gap:     {comp[\"phase1_gap\"]:.1f}× → {comp[\"this_test_gap\"]:.1f}× ({comp[\"gap_reduction\"]:+.1f}×)')
    print()
    print('Criteria Met: {}/{}'.format(data['criteria_met'], data['total_criteria']))
except Exception as e:
    print(f'Error reading summary: {e}', file=sys.stderr)
    sys.exit(1)
"
else
    echo "⚠ test_summary.json not found"
fi

if [ -f "$latest_dir/training_history.csv" ]; then
    echo ""
    echo "✓ training_history.csv created"
    line_count=$(wc -l < "$latest_dir/training_history.csv")
    echo "  Training epochs: $((line_count - 1))"
fi

if [ -f "$latest_dir/best_model.keras" ]; then
    echo ""
    echo "✓ best_model.keras saved"
    model_size=$(du -h "$latest_dir/best_model.keras" | cut -f1)
    echo "  Model size: $model_size"
fi

################################################################################
# Recommendations
################################################################################

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "7. Next Steps Recommendation"
echo "─────────────────────────────────────────────────────────────────────────────"

if [ $exit_code -eq 0 ]; then
    echo "✅ Test passed! Small model shows improvement."
    echo ""
    echo "Next Priority:"
    echo "  • Implement 5-fold cross-validation for robust evaluation"
    echo "  • Fine-tune small model hyperparameters (filters=24,32, dropout=0.4,0.6)"
else
    echo "❌ Test did not meet success criteria"
    echo ""
    echo "Next Priority:"
    echo "  • Implement 5-fold cross-validation immediately"
    echo "  • Analyze train/val distribution mismatch"
    echo "  • Consider even smaller model (filters=8) if overfitting persists"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                        SMALL MODEL TEST COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"

exit $exit_code
