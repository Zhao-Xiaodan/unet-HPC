#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Attention_Hyperparam
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

################################################################################
# Attention Models Hyperparameter Search
################################################################################
#
# Purpose: Train Attention UNet and Attention ResUNet with hyperparameter tuning
#
# Features:
#   - NO Lambda layers (uses RepeatElements custom layer)
#   - Saves BOTH best and final models
#   - Hyperparameter grid search
#   - Proper BinaryFocalLoss serialization
#
# Models: Attention UNet, Attention ResUNet
# Training: Hyperparameter search (filters, dropout, learning rate)
#
# Expected Runtime: 24-48 hours (multiple hyperparameter combinations)
#
# Date: October 16, 2025
################################################################################

echo "========================================================================"
echo "ATTENTION MODELS HYPERPARAMETER SEARCH"
echo "========================================================================"
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $PBS_O_WORKDIR"
echo "========================================================================"
echo ""

# Navigate to working directory
cd /home/svu/phyzxi/scratch/unet-HPC

# =======================================================================
# ENVIRONMENT SETUP
# =======================================================================

echo "=== ENVIRONMENT SETUP ==="

export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

module load singularity

image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

if [ ! -f "$image" ]; then
    echo "ERROR: TensorFlow container not found at $image"
    exit 1
fi

echo "✓ TensorFlow Container: $image"
echo "==========================="
echo ""

# Verify required files exist
echo "Verifying required files..."

TRAIN_DIR="./dataset_new_shrunk/train"
VAL_DIR="./dataset_new_shrunk/val"
SCRIPT="./train_attention_models_hyperparam.py"

if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: Training directory not found: $TRAIN_DIR"
    exit 1
fi

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: Validation directory not found: $VAL_DIR"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Training script not found: $SCRIPT"
    exit 1
fi

# Check for required Python modules
if [ ! -f "./models_fixed.py" ]; then
    echo "ERROR: models_fixed.py not found"
    exit 1
fi

if [ ! -f "./loss_functions_fixed.py" ]; then
    echo "ERROR: loss_functions_fixed.py not found"
    exit 1
fi

if [ ! -f "./data_generator.py" ]; then
    echo "ERROR: data_generator.py not found"
    exit 1
fi

echo "✓ Training directory: $TRAIN_DIR"
echo "✓ Validation directory: $VAL_DIR"
echo "✓ Training script: $SCRIPT"
echo "✓ All required modules present"
echo ""

# Count dataset samples
TRAIN_IMAGES=$(find "$TRAIN_DIR/images" -name "*.tif" -o -name "*.png" 2>/dev/null | wc -l)
VAL_IMAGES=$(find "$VAL_DIR/images" -name "*.tif" -o -name "*.png" 2>/dev/null | wc -l)

echo "Dataset:"
echo "  Training images: $TRAIN_IMAGES"
echo "  Validation images: $VAL_IMAGES"
echo ""

# Run training
echo "========================================================================"
echo "RUNNING HYPERPARAMETER SEARCH"
echo "========================================================================"
echo "Command: singularity exec --nv \"$image\" python3 $SCRIPT"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
singularity exec --nv "$image" python3 "$SCRIPT" 2>&1 | tee "train_attention_hyperparam_console_${TIMESTAMP}.log"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "TRAINING COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Find and display output directory
if [ $EXIT_CODE -eq 0 ]; then
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "attention_hyperparam_*" | sort | tail -1)

    if [ -n "$OUTPUT_DIR" ]; then
        echo ""
        echo "Output directory: $OUTPUT_DIR"
        echo ""

        # Display results summary
        if [ -f "$OUTPUT_DIR/all_results.csv" ]; then
            echo "Results Summary:"
            head -20 "$OUTPUT_DIR/all_results.csv"
            echo ""
        fi

        # Count saved models
        MODEL_COUNT=$(find "$OUTPUT_DIR/models" -name "*.keras" 2>/dev/null | wc -l)
        CHECKPOINT_COUNT=$(find "$OUTPUT_DIR/checkpoints" -name "*.keras" 2>/dev/null | wc -l)

        echo "Models saved:"
        echo "  Final models: $MODEL_COUNT"
        echo "  Best checkpoints: $CHECKPOINT_COUNT"
        echo ""

        # Display directory structure
        echo "Output structure:"
        ls -lh "$OUTPUT_DIR"
        echo ""
    fi

    echo "✓ Hyperparameter search completed successfully!"
else
    echo ""
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo "Check the log file for details."
fi

echo "========================================================================"

exit $EXIT_CODE
