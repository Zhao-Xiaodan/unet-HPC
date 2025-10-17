#!/bin/bash
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Density_Arch_Comparison
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

################################################################################
# Architecture Comparison Density Analysis
################################################################################
#
# Purpose: Compare UNet, Attention UNet, and Attention ResUNet on test images
#
# Output:
#   1. Combined boxplots (3 architectures side-by-side for each density method)
#   2. 4-panel tile visualizations (Original + 3 architecture predictions)
#   3. Comparative statistics CSV
#
# Models used:
#   - UNet: Best from unet_hyperparam_20251013_034824
#   - Attention UNet: Best from attention_unet_hyperparam_20251015_230149
#   - Attention ResUNet: Best from attention_resunet_hyperparam_20251015_235542
#
# Date: October 17, 2025
################################################################################

echo "========================================================================"
echo "ARCHITECTURE COMPARISON DENSITY ANALYSIS"
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

UNET_DIR="./unet_hyperparam_20251013_034824"
ATTN_UNET_DIR="./attention_unet_hyperparam_20251015_230149"
ATTN_RESUNET_DIR="./attention_resunet_hyperparam_20251015_235542"
TEST_DIR="./test_images"
SCRIPT="./density_analysis_architecture_comparison.py"

if [ ! -d "$UNET_DIR" ]; then
    echo "ERROR: UNet model directory not found: $UNET_DIR"
    exit 1
fi

if [ ! -d "$UNET_DIR/checkpoints" ]; then
    echo "ERROR: UNet checkpoints directory not found: $UNET_DIR/checkpoints"
    exit 1
fi

if [ ! -d "$ATTN_UNET_DIR" ]; then
    echo "ERROR: Attention UNet model directory not found: $ATTN_UNET_DIR"
    exit 1
fi

if [ ! -d "$ATTN_UNET_DIR/checkpoints" ]; then
    echo "ERROR: Attention UNet checkpoints directory not found: $ATTN_UNET_DIR/checkpoints"
    exit 1
fi

if [ ! -d "$ATTN_RESUNET_DIR" ]; then
    echo "ERROR: Attention ResUNet model directory not found: $ATTN_RESUNET_DIR"
    exit 1
fi

if [ ! -d "$ATTN_RESUNET_DIR/checkpoints" ]; then
    echo "ERROR: Attention ResUNet checkpoints directory not found: $ATTN_RESUNET_DIR/checkpoints"
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

# Check for model files in each architecture
UNET_COUNT=$(find "$UNET_DIR/checkpoints" -name "best_model.keras" | wc -l)
ATTN_UNET_COUNT=$(find "$ATTN_UNET_DIR/checkpoints" -name "best_model.keras" | wc -l)
ATTN_RESUNET_COUNT=$(find "$ATTN_RESUNET_DIR/checkpoints" -name "best_model.keras" | wc -l)

if [ "$UNET_COUNT" -eq 0 ]; then
    echo "ERROR: No best_model.keras files found in $UNET_DIR/checkpoints"
    exit 1
fi

if [ "$ATTN_UNET_COUNT" -eq 0 ]; then
    echo "ERROR: No best_model.keras files found in $ATTN_UNET_DIR/checkpoints"
    exit 1
fi

if [ "$ATTN_RESUNET_COUNT" -eq 0 ]; then
    echo "ERROR: No best_model.keras files found in $ATTN_RESUNET_DIR/checkpoints"
    exit 1
fi

echo "✓ UNet model directory: $UNET_DIR"
echo "✓ UNet checkpoints: $UNET_DIR/checkpoints ($UNET_COUNT models)"
echo "✓ Attention UNet model directory: $ATTN_UNET_DIR"
echo "✓ Attention UNet checkpoints: $ATTN_UNET_DIR/checkpoints ($ATTN_UNET_COUNT models)"
echo "✓ Attention ResUNet model directory: $ATTN_RESUNET_DIR"
echo "✓ Attention ResUNet checkpoints: $ATTN_RESUNET_DIR/checkpoints ($ATTN_RESUNET_COUNT models)"
echo "✓ Test images directory: $TEST_DIR"
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

# Run architecture comparison analysis
echo "========================================================================"
echo "RUNNING ARCHITECTURE COMPARISON DENSITY ANALYSIS"
echo "========================================================================"
echo "Command: singularity exec --nv \"$image\" python3 $SCRIPT"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
singularity exec --nv "$image" python3 "$SCRIPT" 2>&1 | tee "density_analysis_architecture_comparison_console_${TIMESTAMP}.log"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "ARCHITECTURE COMPARISON DENSITY ANALYSIS COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Find and display output directory
if [ $EXIT_CODE -eq 0 ]; then
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "density_analysis_architecture_comparison_*" | sort | tail -1)

    if [ -n "$OUTPUT_DIR" ]; then
        echo ""
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        echo "Generated files:"
        ls -lh "$OUTPUT_DIR"
        echo ""

        # Display summary if CSV exists
        if [ -f "$OUTPUT_DIR/architecture_comparison_image_summary.csv" ]; then
            echo ""
            echo "Architecture Comparison Summary (first 20 rows):"
            head -20 "$OUTPUT_DIR/architecture_comparison_image_summary.csv"
        fi

        if [ -f "$OUTPUT_DIR/EXPERIMENT_INFO.json" ]; then
            echo ""
            echo "Experiment Info:"
            cat "$OUTPUT_DIR/EXPERIMENT_INFO.json"
        fi

        if [ -d "$OUTPUT_DIR/representative_tiles_4panel" ]; then
            echo ""
            echo "4-Panel Tile Visualizations:"
            ls -1 "$OUTPUT_DIR/representative_tiles_4panel"
            TILE_COUNT=$(ls -1 "$OUTPUT_DIR/representative_tiles_4panel" | wc -l)
            echo "Total: $TILE_COUNT image tile sets"
        fi
    fi

    echo ""
    echo "✓ Architecture comparison analysis completed successfully!"
    echo ""
    echo "Generated visualizations:"
    echo "  Combined boxplots (6 density methods, 3 architectures per plot):"
    echo "    1. density_boxplot_comparison_threshold_0p2.png"
    echo "    2. density_boxplot_comparison_threshold_0p5.png"
    echo "    3. density_boxplot_comparison_threshold_0p8.png"
    echo "    4. density_boxplot_comparison_threshold_0p95.png"
    echo "    5. density_boxplot_comparison_claheotsu_on_pred.png"
    echo "    6. density_boxplot_comparison_claheotsu_on_original.png"
    echo "  4-panel tile visualizations:"
    echo "    - representative_tiles_4panel/ (5 tiles per image, 4 panels each)"
    echo "      * Panel 1: Original tile"
    echo "      * Panel 2: UNet prediction (white beads)"
    echo "      * Panel 3: Attention UNet prediction (white beads)"
    echo "      * Panel 4: Attention ResUNet prediction (white beads)"
else
    echo ""
    echo "✗ Architecture comparison analysis failed with exit code $EXIT_CODE"
    echo "Check the log file for details."
fi

echo "========================================================================"

exit $EXIT_CODE
