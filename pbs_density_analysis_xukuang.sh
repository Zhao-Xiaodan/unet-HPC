#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Density_MultiModel
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

################################################################################
# Multi-Model Density Analysis Using Xukuang Models
################################################################################
#
# Purpose: Perform density analysis on test images using ALL THREE models
#          from xukuang_params_shrunk_20251015_071224
#
# Models: UNet, Attention UNet, Attention ResUNet
# Training: LR=0.005, 200 epochs, BinaryFocalLoss, 512×512 RGB
#
# NEW FEATURES:
#   - Tile-level density values (n=28 per image) for ALL models
#   - Multi-model box plots (individual + comparison)
#   - 4-panel tile comparisons (Original, UNet, Attn UNet, Attn ResUNet)
#   - CORRECTED dilution ordering (10x - 10240x)
#
# Date: October 15, 2025
# Updated: Multi-model comparison
################################################################################

echo "========================================================================"
echo "MULTI-MODEL DENSITY ANALYSIS - XUKUANG MODELS"
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

MODEL_DIR="./xukuang_params_shrunk_20251015_071224"
TEST_DIR="./test_images"
SCRIPT="./density_analysis_xukuang.py"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

if [ ! -d "$TEST_DIR" ]; then
    echo "ERROR: Test images directory not found: $TEST_DIR"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Analysis script not found: $SCRIPT"
    exit 1
fi

# Check for model files
MODEL_COUNT=$(find "$MODEL_DIR" -name "*.keras" -o -name "*.h5" | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "ERROR: No model files (.keras or .h5) found in $MODEL_DIR"
    exit 1
fi

echo "✓ Model directory: $MODEL_DIR"
echo "✓ Test images directory: $TEST_DIR"
echo "✓ Found $MODEL_COUNT model file(s)"
echo ""

# Count test images
TEST_COUNT=$(find "$TEST_DIR" -name "*.tif" -o -name "*.tiff" | wc -l)
echo "Found $TEST_COUNT test images"
echo ""

# List test images
echo "Test images:"
ls -1 "$TEST_DIR"/*.tif "$TEST_DIR"/*.tiff 2>/dev/null | while read img; do
    echo "  - $(basename "$img")"
done
echo ""

# Run density analysis
echo "========================================================================"
echo "RUNNING DENSITY ANALYSIS"
echo "========================================================================"
echo "Command: singularity exec --nv \"$image\" python3 $SCRIPT"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
singularity exec --nv "$image" python3 "$SCRIPT" 2>&1 | tee "density_analysis_xukuang_console_${TIMESTAMP}.log"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "DENSITY ANALYSIS COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Find and display output directory
if [ $EXIT_CODE -eq 0 ]; then
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "density_analysis_xukuang_multimodel_*" | sort | tail -1)

    if [ -n "$OUTPUT_DIR" ]; then
        echo ""
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        echo "Generated files:"
        ls -lh "$OUTPUT_DIR"
        echo ""

        if [ -d "$OUTPUT_DIR/representative_tiles_4panel" ]; then
            echo "4-panel tile comparisons:"
            ls -1 "$OUTPUT_DIR/representative_tiles_4panel"
        fi

        # Display summary if CSV exists
        if [ -f "$OUTPUT_DIR/density_results_tile_level.csv" ]; then
            echo ""
            echo "Tile-Level Density Results Summary (first 20 rows):"
            head -20 "$OUTPUT_DIR/density_results_tile_level.csv"
        fi

        if [ -f "$OUTPUT_DIR/density_results_image_summary.csv" ]; then
            echo ""
            echo "Image-Level Density Summary:"
            head -20 "$OUTPUT_DIR/density_results_image_summary.csv"
        fi
    fi

    echo ""
    echo "✓ Analysis completed successfully!"
else
    echo ""
    echo "✗ Analysis failed with exit code $EXIT_CODE"
    echo "Check the log file for details."
fi

echo "========================================================================"

exit $EXIT_CODE
