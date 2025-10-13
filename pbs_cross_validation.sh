#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oed
#PBS -N CrossValidation
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg

################################################################################
# 5-Fold Cross-Validation - PRIORITY 1 CRITICAL TEST
################################################################################
#
# PURPOSE:
#   Get reliable performance estimates using cross-validation
#
# PROBLEM:
#   All 3 tests (Phase 1, Focal Tversky, Small Model) peaked at epoch 1
#   indicating validation set is not representative
#
# SOLUTION:
#   5-fold stratified CV provides:
#   - 5 different train/val splits
#   - Average performance (more reliable)
#   - Confidence intervals
#   - Detection of split bias
#
# CONFIGURATION:
#   BASELINE from Phase 1 (filters=64, dropout=0.3)
#   for fair comparison and proven capacity
#
# EXPECTED INSIGHTS:
#   - If CV avg > 13.8% → Phase 1 split was biased
#   - If CV avg ~ 13.8% → Task is genuinely hard
#   - Fold variance reveals stability
#   - Best epoch pattern reveals if problem persists
#
# ESTIMATED TIME: 10-12 hours (2 hours per fold)
#
################################################################################

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    5-FOLD CROSS-VALIDATION - PRIORITY 1                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Purpose: Get reliable performance estimates using cross-validation"
echo "Config:  BASELINE (filters=64, dropout=0.3)"
echo ""
echo "Starting at: $(date)"
echo ""

################################################################################
# Environment Setup
################################################################################

echo "─────────────────────────────────────────────────────────────────────────────"
echo "1. Environment Setup"
echo "─────────────────────────────────────────────────────────────────────────────"

# Change to working directory
cd /home/svu/phyzxi/scratch/unet-HPC || exit 1
echo "✓ Working directory: $(pwd)"

# GPU Configuration - FP32
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

# Memory management for FP32
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC_SIZE_BYTES=536870912  # 512MB
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_MAX_ALLOCATION_SIZE_BYTES=8589934592  # 8GB

echo "✓ GPU Configuration:"
echo "  - Precision: FP32"
echo "  - GPU Memory: 8GB max"
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
if [ ! -f "validate_cross_validation.py" ]; then
    echo "❌ ERROR: validate_cross_validation.py not found"
    exit 1
fi
echo "✓ validate_cross_validation.py"

# Check model definitions
if [ ! -f "model_architectures.py" ]; then
    echo "❌ ERROR: model_architectures.py not found"
    exit 1
fi
echo "✓ model_architectures.py"

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

echo "Cross-Validation Setup:"
echo "  - Number of folds:  5"
echo "  - Stratification:   By density (quartiles)"
echo "  - Random seed:      42 (reproducible)"
echo ""
echo "Model Configuration (BASELINE):"
echo "  - Architecture:     U-Net"
echo "  - Filters:          64 (31M params)"
echo "  - Dropout:          0.3"
echo "  - Batch size:       4"
echo "  - Loss function:    combined (Dice + Focal)"
echo "  - Epochs per fold:  20 (with early stopping)"
echo ""
echo "Expected Per Fold:"
echo "  - Training samples: ~79 images"
echo "  - Val samples:      ~20 images"
echo "  - Training time:    ~2 hours"
echo ""
echo "Total Expected Time: ~10 hours"

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
# Run Cross-Validation
################################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                      STARTING CROSS-VALIDATION                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

start_time=$(date +%s)

singularity exec --nv $image python3 validate_cross_validation.py

exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))
duration_hours=$((duration / 3600))
duration_min=$(((duration % 3600) / 60))

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                      CROSS-VALIDATION COMPLETE                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Exit code: $exit_code"
echo "Duration: ${duration_hours}h ${duration_min}m"
echo "Finished at: $(date)"

################################################################################
# Post-Run Analysis
################################################################################

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "6. Post-Run Analysis"
echo "─────────────────────────────────────────────────────────────────────────────"

# Find latest results directory
latest_dir=$(ls -dt validation_cv_* 2>/dev/null | head -1)

if [ -z "$latest_dir" ]; then
    echo "⚠ No results directory found"
    exit $exit_code
fi

echo "Results directory: $latest_dir"
echo ""

# Check for summary file
if [ -f "$latest_dir/cv_summary.json" ]; then
    echo "✓ cv_summary.json created"

    # Extract key metrics using Python
    singularity exec $image python3 -c "
import json
import sys

try:
    with open('$latest_dir/cv_summary.json', 'r') as f:
        data = json.load(f)

    stats = data['statistics']
    comparison = data['comparison']
    diagnosis = data['diagnosis']

    print('Cross-Validation Results:')
    print(f'  Best Val Jaccard:    {stats[\"best_val_jacard\"][\"mean\"]*100:.1f}% ± {stats[\"best_val_jacard\"][\"std\"]*100:.1f}%')
    print(f'  Range:               {stats[\"best_val_jacard\"][\"min\"]*100:.1f}% - {stats[\"best_val_jacard\"][\"max\"]*100:.1f}%')
    print(f'  Overfitting Gap:     {stats[\"overfitting_gap\"][\"mean\"]:.1f}× ± {stats[\"overfitting_gap\"][\"std\"]:.1f}×')
    print()
    print('Comparison with Phase 1:')
    print(f'  Phase 1 Best:        {comparison[\"phase1_best_jacard\"]*100:.1f}%')
    print(f'  CV Mean:             {comparison[\"cv_mean_best_jacard\"]*100:.1f}%')
    print(f'  Improvement:         {comparison[\"improvement\"]*100:+.1f}%')
    print()
    print('Diagnosis:')
    print(f'  Split bias detected: {\"YES\" if diagnosis[\"split_bias\"] else \"NO\"}')
    print(f'  Epoch 1 problem:     {\"YES\" if diagnosis[\"epoch_1_problem_persists\"] else \"NO\"}')
    print(f'  High variance:       {\"YES\" if diagnosis[\"high_variance\"] else \"NO\"}')
    print(f'  Coef. of variation:  {diagnosis[\"coefficient_of_variation\"]:.1%}')
except Exception as e:
    print(f'Error reading summary: {e}', file=sys.stderr)
    sys.exit(1)
"
else
    echo "⚠ cv_summary.json not found"
fi

# Check individual fold results
echo ""
echo "Individual Fold Results:"
for fold_dir in "$latest_dir"/fold_*; do
    if [ -f "$fold_dir/results.json" ]; then
        fold_num=$(basename "$fold_dir" | sed 's/fold_//')
        singularity exec $image python3 -c "
import json
with open('$fold_dir/results.json') as f:
    r = json.load(f)
print(f'  Fold $fold_num: {r[\"best_val_jacard\"]*100:.1f}% (epoch {r[\"best_epoch\"]+1})')
"
    fi
done

################################################################################
# Recommendations
################################################################################

echo ""
echo "─────────────────────────────────────────────────────────────────────────────"
echo "7. Next Steps Recommendation"
echo "─────────────────────────────────────────────────────────────────────────────"

if [ $exit_code -eq 0 ]; then
    echo "✅ Cross-validation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Review cv_summary.json for detailed statistics"
    echo "  2. If CV mean > Phase 1 → Phase 1 split was biased"
    echo "  3. If CV mean ~ Phase 1 → Task is genuinely hard"
    echo "  4. Check if 'epoch 1 problem' persists across folds"
    echo "  5. Based on diagnosis, proceed with:"
    echo "     - Train/val distribution analysis (if problem persists)"
    echo "     - Optimal model size search (if no split bias)"
else
    echo "❌ Cross-validation encountered errors"
    echo ""
    echo "Check the logs above for error messages"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                      CROSS-VALIDATION COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"

exit $exit_code
