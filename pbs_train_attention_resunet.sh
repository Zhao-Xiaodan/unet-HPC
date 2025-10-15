#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N AttentionResUNet_Hyperparam
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

################################################################################
# Attention ResUNet Hyperparameter Search
################################################################################
#
# Purpose: Train Attention ResUNet with hyperparameter tuning
#
# Features:
#   - NO Lambda layers (uses RepeatElements custom layer)
#   - Saves BOTH best and final models
#   - Hyperparameter grid search
#   - Proper BinaryFocalLoss serialization
#
# Model: Attention ResUNet (with residual blocks + attention gates)
# Training: Hyperparameter search (filters, dropout, learning rate)
#
# Expected Runtime: 24-48 hours
#
# Date: October 16, 2025
################################################################################

echo "========================================================================"
echo "ATTENTION RESUNET HYPERPARAMETER SEARCH"
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
echo "============================"
echo ""

# Verify required files exist
echo "Verifying required files..."

IMAGES_DIR="./dataset_shrunk_masks/images"
MASKS_DIR="./dataset_shrunk_masks/masks"
SCRIPT="./train_attention_resunet_hyperparam.py"

if [ ! -d "$IMAGES_DIR" ]; then
    echo "ERROR: Images directory not found: $IMAGES_DIR"
    exit 1
fi

if [ ! -d "$MASKS_DIR" ]; then
    echo "ERROR: Masks directory not found: $MASKS_DIR"
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

echo "✓ Images directory: $IMAGES_DIR"
echo "✓ Masks directory: $MASKS_DIR"
echo "✓ Training script: $SCRIPT"
echo "✓ All required modules present"
echo ""

# Count dataset samples
TOTAL_IMAGES=$(find "$IMAGES_DIR" -name "*.tif" -o -name "*.png" 2>/dev/null | wc -l)
TOTAL_MASKS=$(find "$MASKS_DIR" -name "*.tif" -o -name "*.png" 2>/dev/null | wc -l)

echo "Dataset:"
echo "  Total images: $TOTAL_IMAGES"
echo "  Total masks: $TOTAL_MASKS"
echo "  (Script will split 80% train / 20% validation)"
echo ""

# Run training
echo "========================================================================"
echo "RUNNING HYPERPARAMETER SEARCH"
echo "========================================================================"
echo "Command: singularity exec --nv \"$image\" python3 $SCRIPT"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
singularity exec --nv "$image" python3 "$SCRIPT" 2>&1 | tee "train_attention_resunet_hyperparam_console_${TIMESTAMP}.log"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "TRAINING COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Find and display output directory
if [ $EXIT_CODE -eq 0 ]; then
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "attention_resunet_hyperparam_*" | sort | tail -1)

    if [ -n "$OUTPUT_DIR" ]; then
        echo ""
        echo "Output directory: $OUTPUT_DIR"
        echo ""

        # Display results summary
        if [ -f "$OUTPUT_DIR/attention_resunet_results.csv" ]; then
            echo "Results Summary:"
            head -20 "$OUTPUT_DIR/attention_resunet_results.csv"
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
