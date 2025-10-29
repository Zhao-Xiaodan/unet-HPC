#!/bin/bash
#PBS -N UNet_Viz_Distill
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -j oe
#PBS -o UNet_Viz_Distill.o

# ============================================================================
# U-Net Feature Visualization with Distill 2017 Enhancements - HPC Execution
# ============================================================================
#
# FIXED VERSION: Uses Singularity container (like other working scripts)
#
# This script runs enhanced feature visualization with:
# 1. Fourier preconditioning (Distill's key innovation)
# 2. Enhanced transformation robustness (rotation, scale, larger jitter)
# 3. Explicit diversity term option
# 4. Neuron interaction visualizations
#
# Based on: "Feature Visualization" - Olah et al., Distill 2017
# https://distill.pub/2017/feature-visualization/
#
# ============================================================================

# Load Singularity module (like other working scripts)
module load singularity

# PyTorch Singularity container (same as pbs_crp_complete_with_encoders.sh)
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

# Change to working directory
cd $PBS_O_WORKDIR

# Print environment info
echo "=================================================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Singularity image: $image"
echo "=================================================================="
echo ""

# Check GPU availability
nvidia-smi
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model path (using the standard location from other scripts)
MODEL_PATH="./best_models_PyTorch/unet/best_model.pth"

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "=================================================================="
    echo "ERROR: Model not found at: $MODEL_PATH"
    echo "=================================================================="
    echo ""
    echo "Available model directories:"
    ls -ld ./best_models_PyTorch/* 2>/dev/null || echo "  No best_models_PyTorch directory found"
    echo ""
    echo "Please update MODEL_PATH in this script to point to your trained model."
    echo "=================================================================="
    exit 1
fi

# Output directory name
OUTPUT_DIR="unet_viz_distill"

# Layers to visualize
LAYERS="encoder_1_conv2 encoder_3_conv2 decoder_1_conv2 bottleneck_conv2"

# Number of channels per layer
CHANNELS_PER_LAYER=12

# Number of diverse examples per channel (Distill diversity)
DIVERSE_PER_CHANNEL=3

# Optimization iterations
ITERATIONS=500

# Use Fourier preconditioning (Distill's key innovation)
USE_FOURIER="--use_fourier"

# Method comparison (Fourier vs standard) - OPTIONAL, increases runtime ~2x
# COMPARE_METHODS="--compare_methods"
COMPARE_METHODS=""

# ============================================================================
# EXECUTION
# ============================================================================

echo "=================================================================="
echo "Distill 2017 Enhanced Feature Visualization"
echo "=================================================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Layers: $LAYERS"
echo "Channels per layer: $CHANNELS_PER_LAYER"
echo "Diverse examples: $DIVERSE_PER_CHANNEL"
echo "Iterations: $ITERATIONS"
echo "Fourier preconditioning: ENABLED"
echo "Enhanced transforms: ENABLED (jitter ±16px, rotation ±10°, scale 0.95-1.05×)"
echo "=================================================================="
echo ""

# Run visualization using Singularity
singularity exec --nv $image python unet_feature_viz_distill.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --layers $LAYERS \
    --channels_per_layer $CHANNELS_PER_LAYER \
    --diverse_per_channel $DIVERSE_PER_CHANNEL \
    --iterations $ITERATIONS \
    $USE_FOURIER \
    $COMPARE_METHODS

# Check exit status
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================================================="
    echo "✅ Visualization completed successfully!"
    echo "=================================================================="
else
    echo ""
    echo "=================================================================="
    echo "❌ Visualization failed with error code $EXIT_CODE"
    echo "=================================================================="
    exit $EXIT_CODE
fi

# ============================================================================
# POST-PROCESSING
# ============================================================================

echo ""
echo "Compressing results..."
OUTPUT_LATEST=$(ls -td ${OUTPUT_DIR}_* 2>/dev/null | head -1)

if [ -n "$OUTPUT_LATEST" ]; then
    # Create compressed archive
    tar -czf "${OUTPUT_LATEST}.tar.gz" "$OUTPUT_LATEST"
    echo "Created archive: ${OUTPUT_LATEST}.tar.gz"

    # Count files
    N_FILES=$(find "$OUTPUT_LATEST" -type f | wc -l)
    echo "Total files generated: $N_FILES"

    # Disk usage
    DU=$(du -sh "$OUTPUT_LATEST" | cut -f1)
    echo "Disk usage: $DU"

    # Show some example outputs
    echo ""
    echo "Example visualizations created:"
    ls "$OUTPUT_LATEST"/*_overview.png 2>/dev/null | head -3

    if [ -d "$OUTPUT_LATEST/method_comparison" ]; then
        echo ""
        echo "Method comparison images:"
        ls "$OUTPUT_LATEST"/method_comparison/*.png 2>/dev/null | head -3
    fi
fi

echo ""
echo "=================================================================="
echo "Job finished: $(date)"
echo "=================================================================="
