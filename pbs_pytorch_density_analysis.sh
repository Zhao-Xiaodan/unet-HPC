#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -k oed
#PBS -N PyTorch_Density_Analysis
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=64gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# PyTorch Model Prediction and Density Analysis
# ==============================================================================
#
# This script performs:
# 1. Prediction using best models from PyTorch comparison experiments
# 2. Density analysis with 4-panel tile visualizations
#
# Input:
#   - Trained models from experiment directory
#   - Test images from ./test_images/
#
# Output:
#   - Predictions for each architecture (UNet, Attention UNet, Attention ResUNet)
#   - Density boxplots (full range and low dilution range)
#   - 4-panel representative tiles (5 rows × 4 columns per dilution factor)
#
# Expected runtime: ~2-3 hours
# ==============================================================================

# Configuration - MODIFY THESE VARIABLES
# ==============================================================================

# Experiment directory containing trained models
# Change this to the experiment you want to analyze
# IMPORTANT: Ensure the experiment directory contains model checkpoints at:
#   <experiment_dir>/<architecture>/checkpoints/<model_name>/best_model.pth
# Example: pytorch_comparison_no_aug_20251021_121918/unet/checkpoints/unet_n_filters32_dropout0.1_learning_rate0.001/best_model.pth
EXPERIMENT_DIR="pytorch_comparison_no_aug_20251021_121918"

# NOTE: If checkpoints don't exist, you need to retrain using the attention_only scripts:
#   qsub pbs_train_pytorch_comparison_no_aug_attention_only.sh
#   qsub pbs_train_pytorch_comparison_with_aug_attention_only.sh
#   qsub pbs_train_pytorch_comparison_adaptive_loss_attention_only.sh

# Output directory (will be created with timestamp)
OUTPUT_BASE="pytorch_density_analysis"

# Test images directory
TEST_IMAGES_DIR="./test_images"

# ==============================================================================
# Setup
# ==============================================================================

# Load required modules
module load singularity

# Define singularity container (PyTorch)
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

# Change to working directory
cd $PBS_O_WORKDIR

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

# Print job information
echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Working Directory: $PWD"
echo "========================================"
echo ""

echo "Configuration:"
echo "  Experiment: $EXPERIMENT_DIR"
echo "  Test Images: $TEST_IMAGES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Check GPU availability
echo "GPU Information:"
singularity exec --nv $image nvidia-smi
echo ""

# Print Python and PyTorch versions
echo "Python version:"
singularity exec --nv $image python --version
echo ""

echo "PyTorch version:"
singularity exec --nv $image python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# ==============================================================================
# Verify Checkpoints Exist
# ==============================================================================

echo "========================================"
echo "VERIFYING MODEL CHECKPOINTS"
echo "========================================"
echo ""

# Check if experiment directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "ERROR: Experiment directory not found: $EXPERIMENT_DIR"
    exit 1
fi

# Check for checkpoint files
CHECKPOINT_COUNT=$(find "$EXPERIMENT_DIR" -name "best_model.pth" | wc -l)

if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
    echo "ERROR: No model checkpoints found in $EXPERIMENT_DIR"
    echo ""
    echo "Expected structure:"
    echo "  $EXPERIMENT_DIR/unet/checkpoints/<model_name>/best_model.pth"
    echo "  $EXPERIMENT_DIR/attention_unet/checkpoints/<model_name>/best_model.pth"
    echo "  $EXPERIMENT_DIR/attention_resunet/checkpoints/<model_name>/best_model.pth"
    echo ""
    echo "Please retrain models using attention_only scripts:"
    echo "  qsub pbs_train_pytorch_comparison_no_aug_attention_only.sh"
    echo "  qsub pbs_train_pytorch_comparison_with_aug_attention_only.sh"
    echo "  qsub pbs_train_pytorch_comparison_adaptive_loss_attention_only.sh"
    echo ""
    exit 1
fi

echo "✓ Found $CHECKPOINT_COUNT model checkpoint(s)"
find "$EXPERIMENT_DIR" -name "best_model.pth" | sed 's/^/  - /'
echo ""

# ==============================================================================
# Step 1: Generate Predictions
# ==============================================================================

echo "========================================"
echo "STEP 1: GENERATING PREDICTIONS"
echo "========================================"
echo ""
echo "Note: Best models will be cached to ./best_models_PyTorch/"
echo "      First run: Searches experiment dir and copies best models"
echo "      Future runs: Loads directly from cache (faster)"
echo ""

PREDICTIONS_DIR="${OUTPUT_DIR}/predictions"

singularity exec --nv $image python predict_pytorch_comparison.py \
    --experiment "$EXPERIMENT_DIR" \
    --output "$PREDICTIONS_DIR" \
    --test_images "$TEST_IMAGES_DIR" \
    --batch_size 8

PREDICT_STATUS=$?

if [ $PREDICT_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Prediction failed with exit code $PREDICT_STATUS"
    echo "Aborting job."
    exit $PREDICT_STATUS
fi

echo ""
echo "✓ Predictions complete"
echo ""

# ==============================================================================
# Step 2: Density Analysis
# ==============================================================================

echo "========================================"
echo "STEP 2: DENSITY ANALYSIS"
echo "========================================"
echo ""

ANALYSIS_DIR="${OUTPUT_DIR}/analysis"

singularity exec --nv $image python density_analysis_pytorch_comparison.py \
    --predictions "$PREDICTIONS_DIR" \
    --test_images "$TEST_IMAGES_DIR" \
    --output "$ANALYSIS_DIR"

ANALYSIS_STATUS=$?

if [ $ANALYSIS_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Density analysis failed with exit code $ANALYSIS_STATUS"
    exit $ANALYSIS_STATUS
fi

echo ""
echo "✓ Density analysis complete"
echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo "========================================"
echo "JOB COMPLETE"
echo "========================================"
echo "End Time: $(date)"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  Predictions:"
echo "    - $PREDICTIONS_DIR/unet/"
echo "    - $PREDICTIONS_DIR/attention_unet/"
echo "    - $PREDICTIONS_DIR/attention_resunet/"
echo "    - $PREDICTIONS_DIR/prediction_metadata.json"
echo ""
echo "  Analysis:"
echo "    - $ANALYSIS_DIR/density_results_tile_level.csv"
echo "    - $ANALYSIS_DIR/density_results_image_summary.csv"
echo "    - $ANALYSIS_DIR/density_boxplot_full_range__threshold_0.5.png"
echo "    - $ANALYSIS_DIR/density_boxplot_low_dilution_range__threshold_0.5.png"
echo "    - $ANALYSIS_DIR/representative_tiles_4panel/"
echo "    - $ANALYSIS_DIR/EXPERIMENT_INFO.json"
echo ""
echo "========================================"

exit 0
