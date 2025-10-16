#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Density_UNet_Only
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

################################################################################
# UNet-Only Density Analysis - Best Model from Hyperparameter Search
################################################################################
#
# Purpose: Perform density analysis on test images using the BEST UNet model
#          from hyperparameter search (unet_hyperparam_20251015_224125)
#
# Updates previous analysis (density_analysis_xukuang_20251015_142119) with:
#   1. ✅ Tile-level density tracking (all 28 tiles per image saved)
#   2. ✅ Updated boxplot style (log-scale 1/dilution x-axis)
#   3. ✅ Two boxplot ranges:
#      - Full range: 1/10240 to 1/10
#      - Low dilution range: 1/10240 to 1/80
#   4. ✅ Multiple density calculation methods (5 total):
#      - Threshold 0.2, 0.5, 0.8 (simple thresholding)
#      - CLAHE+Otsu on predicted mask (denoised)
#      - CLAHE+Otsu on original image (baseline)
#   5. ✅ 3-panel tile visualizations (5 representative tiles per image):
#      - Panel 1: Original tile
#      - Panel 2: Inverted predicted mask (white beads, threshold 0.5)
#      - Panel 3: Inverted CLAHE+Otsu (white beads, denoised)
#
# Model Selection:
#   - Automatically selects best UNet model based on validation IoU
#   - From: unet_hyperparam_20251015_224125/checkpoints/
#   - Training: 100 epochs, BinaryFocalLoss, 512×512 RGB
#
# Date: October 16, 2025
################################################################################

echo "========================================================================"
echo "UNET-ONLY DENSITY ANALYSIS - BEST MODEL"
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

MODEL_DIR="./unet_hyperparam_20251015_224125"
TEST_DIR="./test_images"
SCRIPT="./density_analysis_unet_only.py"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

if [ ! -d "$MODEL_DIR/checkpoints" ]; then
    echo "ERROR: Checkpoints directory not found: $MODEL_DIR/checkpoints"
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
MODEL_COUNT=$(find "$MODEL_DIR/checkpoints" -name "best_model.keras" | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "ERROR: No best_model.keras files found in $MODEL_DIR/checkpoints"
    echo "       Make sure UNet hyperparameter search has completed!"
    exit 1
fi

echo "✓ Model directory: $MODEL_DIR"
echo "✓ Checkpoints directory: $MODEL_DIR/checkpoints"
echo "✓ Test images directory: $TEST_DIR"
echo "✓ Found $MODEL_COUNT UNet model(s)"
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
singularity exec --nv "$image" python3 "$SCRIPT" 2>&1 | tee "density_analysis_unet_only_console_${TIMESTAMP}.log"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "DENSITY ANALYSIS COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Find and display output directory
if [ $EXIT_CODE -eq 0 ]; then
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "density_analysis_unet_only_*" | sort | tail -1)

    if [ -n "$OUTPUT_DIR" ]; then
        echo ""
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        echo "Generated files:"
        ls -lh "$OUTPUT_DIR"
        echo ""

        # Display summary if CSV exists
        if [ -f "$OUTPUT_DIR/density_results_tile_level.csv" ]; then
            echo ""
            echo "Tile-Level Density Results Summary (first 20 rows):"
            head -20 "$OUTPUT_DIR/density_results_tile_level.csv"
        fi

        if [ -f "$OUTPUT_DIR/density_results_image_summary.csv" ]; then
            echo ""
            echo "Image-Level Density Summary:"
            cat "$OUTPUT_DIR/density_results_image_summary.csv"
        fi

        if [ -f "$OUTPUT_DIR/EXPERIMENT_INFO.json" ]; then
            echo ""
            echo "Experiment Info:"
            cat "$OUTPUT_DIR/EXPERIMENT_INFO.json"
        fi

        if [ -d "$OUTPUT_DIR/representative_tiles_3panel" ]; then
            echo ""
            echo "3-Panel Tile Visualizations:"
            ls -1 "$OUTPUT_DIR/representative_tiles_3panel"
            TILE_COUNT=$(ls -1 "$OUTPUT_DIR/representative_tiles_3panel" | wc -l)
            echo "Total: $TILE_COUNT image tile sets"
        fi
    fi

    echo ""
    echo "✓ Analysis completed successfully!"
    echo ""
    echo "Generated visualizations (10 boxplots + tile visualizations):"
    echo "  Boxplots (5 methods × 2 dilution ranges = 10 total):"
    echo "    1. Threshold 0.2: density_boxplot_*_threshold_0.2.png"
    echo "    2. Threshold 0.5: density_boxplot_*_threshold_0.5.png"
    echo "    3. Threshold 0.8: density_boxplot_*_threshold_0.8.png"
    echo "    4. CLAHE+Otsu on pred: density_boxplot_*_claheotsu_on_pred.png"
    echo "    5. CLAHE+Otsu on orig: density_boxplot_*_claheotsu_on_original.png"
    echo "  Tile visualizations:"
    echo "    - representative_tiles_3panel/ (5 tiles per image, 3 panels each)"
    echo "      * Panel 1: Original tile"
    echo "      * Panel 2: Inverted predicted mask (white beads, threshold 0.5)"
    echo "      * Panel 3: Inverted CLAHE+Otsu (white beads, denoised)"
else
    echo ""
    echo "✗ Analysis failed with exit code $EXIT_CODE"
    echo "Check the log file for details."
fi

echo "========================================================================"

exit $EXIT_CODE
