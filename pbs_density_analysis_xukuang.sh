#!/bin/bash
#PBS -N Density_Analysis_Xukuang
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -q gpu
#PBS -j oe
#PBS -o density_analysis_xukuang.log

################################################################################
# Density Analysis Using Xukuang UNet Model
################################################################################
#
# Purpose: Perform density analysis on test images using the best UNet model
#          from xukuang_params_shrunk_20251015_071224
#
# Model: UNet (Val IoU: 0.6789 at epoch 140)
# Training: LR=0.005, 200 epochs, BinaryFocalLoss, 512×512 RGB
#
# Outputs:
#   - Box plot with CORRECTED dilution ordering (10x - 10240x)
#   - Representative tile visualizations
#   - CSV with density results
#   - EXPERIMENT_INFO.json
#
# Date: October 15, 2025
################################################################################

echo "========================================================================"
echo "DENSITY ANALYSIS - XUKUANG UNET MODEL"
echo "========================================================================"
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $PBS_O_WORKDIR"
echo "========================================================================"
echo ""

# Navigate to working directory
cd $PBS_O_WORKDIR || exit 1

# Load modules
echo "Loading modules..."
module load anaconda/2023a
module load cuda/11.8.0
echo "✓ Modules loaded"
echo ""

# Activate conda environment
echo "Activating conda environment: unetCNN"
source activate unetCNN || { echo "ERROR: Failed to activate environment"; exit 1; }
echo "✓ Environment activated"
echo ""

# Verify Python and packages
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "Keras version: $(python -c 'import tensorflow.keras as keras; print(keras.__version__)')"
echo ""

# GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

# Set environment variables
export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_VISIBLE_DEVICES=0

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
echo "Command: python $SCRIPT"
echo ""

python "$SCRIPT"

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "DENSITY ANALYSIS COMPLETED"
echo "========================================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

# Find and display output directory
if [ $EXIT_CODE -eq 0 ]; then
    OUTPUT_DIR=$(find . -maxdepth 1 -type d -name "density_analysis_xukuang_*" | sort | tail -1)

    if [ -n "$OUTPUT_DIR" ]; then
        echo ""
        echo "Output directory: $OUTPUT_DIR"
        echo ""
        echo "Generated files:"
        ls -lh "$OUTPUT_DIR"
        echo ""

        if [ -d "$OUTPUT_DIR/representative_tiles" ]; then
            echo "Representative tiles:"
            ls -1 "$OUTPUT_DIR/representative_tiles"
        fi

        # Display summary if CSV exists
        if [ -f "$OUTPUT_DIR/density_results.csv" ]; then
            echo ""
            echo "Density Results Summary:"
            head -20 "$OUTPUT_DIR/density_results.csv"
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
