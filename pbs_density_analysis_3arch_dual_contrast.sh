#!/bin/bash
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Density_3Arch_Dual
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

################################################################################
# 3-Architecture Density Analysis with Dual Contrast and Edge Fix
################################################################################
#
# Purpose: Analyze 3 UNet architectures with dual visualization and edge fix
#
# KEY FEATURES:
#   1. ✅ 3 architectures: UNet, Attention UNet, Attention ResUNet
#   2. ✅ 4-panel visualization (Original + 3 predictions)
#   3. ✅ DUAL contrast sets (Fixed vmin=0,vmax=1 + Auto vmin/vmax from data)
#   4. ✅ FIXES CLAHE edge artifacts with border padding/cropping
#   5. ✅ Auto-contrast shows actual min/max ranges in titles
#   6. ✅ FIXED tile indexing bug (tiles now match correctly between architectures)
#   7. ✅ Boxplots for threshold 0.5 (full range + low dilution range)
#
# Date: November 1, 2025
################################################################################

echo "========================================================================"
echo "3-ARCHITECTURE DENSITY ANALYSIS - DUAL CONTRAST + EDGE FIX"
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

# Verify required files
echo "Verifying required files..."

SCRIPT="./density_analysis_3arch_dual_contrast.py"
TEST_DIR="./test_images"

MODEL_DIR_UNET="./unet_hyperparam_20251015_224125"
MODEL_DIR_ATTN="./attention_unet_hyperparam_20251015_230149"
MODEL_DIR_RES="./attention_resunet_hyperparam_20251015_235542"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Analysis script not found: $SCRIPT"
    exit 1
fi

if [ ! -d "$TEST_DIR" ]; then
    echo "ERROR: Test images directory not found: $TEST_DIR"
    exit 1
fi

echo "✓ Script: $SCRIPT"
echo "✓ Test images: $TEST_DIR"

# Check model directories
for MODEL_DIR in "$MODEL_DIR_UNET" "$MODEL_DIR_ATTN" "$MODEL_DIR_RES"; do
    if [ -d "$MODEL_DIR" ]; then
        echo "✓ Model directory: $MODEL_DIR"
    else
        echo "⚠ Warning: Model directory not found: $MODEL_DIR"
    fi
done
echo ""

# Count test images
TEST_COUNT=$(find "$TEST_DIR" -name "*.tif" -o -name "*.tiff" | wc -l)
echo "Found $TEST_COUNT test images"
echo ""

echo "========================================================================"
echo "KEY IMPROVEMENTS IN THIS ANALYSIS"
echo "========================================================================"
echo "1. FIXES CLAHE edge artifacts:"
echo "   - Pads borders before CLAHE processing"
echo "   - Crops after Otsu thresholding"
echo "   - Result: NO MORE BLACK/WHITE FRAMES!"
echo ""
echo "2. 4-panel architecture comparison:"
echo "   - Panel 1: Original image"
echo "   - Panel 2: UNet prediction (inverted)"
echo "   - Panel 3: Attention UNet prediction (inverted)"
echo "   - Panel 4: Attention ResUNet prediction (inverted)"
echo ""
echo "3. DUAL visualization sets:"
echo "   - SET 1: Fixed contrast (vmin=0, vmax=1)"
echo "     └─> For comparison with previous results"
echo "   - SET 2: Auto contrast (vmin/vmax from actual data)"
echo "     └─> Shows actual min/max range in title"
echo "     └─> Proves auto-contrast is working!"
echo ""
echo "4. FIXED tile indexing bug:"
echo "   - Previous version accessed tiles by wrong list index"
echo "   - Now matches tiles by (image_name, tile_idx)"
echo "   - Result: Tiles CORRECTLY CORRELATED between architectures!"
echo ""
echo "5. Boxplots for threshold 0.5:"
echo "   - Full dilution range (1/10240 to 1/10)"
echo "   - Low dilution range (1/10240 to 1/80)"
echo "   - Grouped comparison of all 3 architectures"
echo ""
echo "6. Same tiles in both sets (fair comparison)"
echo "========================================================================"
echo ""

# Run analysis
echo "========================================================================"
echo "RUNNING 3-ARCHITECTURE ANALYSIS"
echo "========================================================================"
echo "Command: singularity exec --nv \"$image\" python3 $SCRIPT"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
singularity exec --nv "$image" python3 "$SCRIPT" 2>&1 | tee "density_3arch_dual_contrast_console_${TIMESTAMP}.log"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "ANALYSIS COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Display results
if [ $EXIT_CODE -eq 0 ]; then
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "density_analysis_3arch_dual_contrast_*" | sort | tail -1)

    if [ -n "$OUTPUT_DIR" ]; then
        echo ""
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        echo "Generated files:"
        ls -lh "$OUTPUT_DIR"
        echo ""

        if [ -f "$OUTPUT_DIR/EXPERIMENT_INFO.json" ]; then
            echo "Experiment Info:"
            cat "$OUTPUT_DIR/EXPERIMENT_INFO.json"
            echo ""
        fi

        if [ -d "$OUTPUT_DIR/representative_tiles_4panel_fixed_contrast" ]; then
            echo "4-Panel Tiles (FIXED CONTRAST):"
            TILE_COUNT=$(ls -1 "$OUTPUT_DIR/representative_tiles_4panel_fixed_contrast" | wc -l)
            echo "  Total: $TILE_COUNT tile sets"
            ls -1 "$OUTPUT_DIR/representative_tiles_4panel_fixed_contrast" | head -3
            echo "  ..."
        fi

        if [ -d "$OUTPUT_DIR/representative_tiles_4panel_auto_contrast" ]; then
            echo ""
            echo "4-Panel Tiles (AUTO CONTRAST):"
            TILE_COUNT=$(ls -1 "$OUTPUT_DIR/representative_tiles_4panel_auto_contrast" | wc -l)
            echo "  Total: $TILE_COUNT tile sets"
            ls -1 "$OUTPUT_DIR/representative_tiles_4panel_auto_contrast" | head -3
            echo "  ..."
        fi
    fi

    echo ""
    echo "✓ Analysis completed successfully!"
    echo ""
    echo "WHAT TO CHECK:"
    echo "  1. Edge artifacts: Should be GONE from all tiles!"
    echo "  2. Auto-contrast: Check titles show actual [min, max] ranges"
    echo "  3. Architecture comparison: 4 panels side-by-side"
    echo "  4. Dual sets: Compare fixed vs auto contrast versions"
else
    echo ""
    echo "✗ Analysis failed with exit code $EXIT_CODE"
    echo "Check the log file for details."
fi

echo "========================================================================"

exit $EXIT_CODE
