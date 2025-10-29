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

# Load required modules
module purge
module load anaconda3/2023.09-0-gcc/12.3.0-bvbszyk

# Activate conda environment
source activate unetCNN

# Change to working directory
cd $PBS_O_WORKDIR

# Print environment info
echo "=================================================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "=================================================================="
echo ""

# Check GPU availability
nvidia-smi
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model path (update this to your trained model)
MODEL_PATH="/path/to/your/trained/unet_model.pth"  # UPDATE THIS

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

# Method comparison (Fourier vs standard)
COMPARE_METHODS="--compare_methods"

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

# Run visualization
python unet_feature_viz_distill.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --layers $LAYERS \
    --channels_per_layer $CHANNELS_PER_LAYER \
    --diverse_per_channel $DIVERSE_PER_CHANNEL \
    --iterations $ITERATIONS \
    $USE_FOURIER \
    $COMPARE_METHODS

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================================="
    echo "✅ Visualization completed successfully!"
    echo "=================================================================="
else
    echo ""
    echo "=================================================================="
    echo "❌ Visualization failed with error code $?"
    echo "=================================================================="
    exit 1
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
fi

echo ""
echo "=================================================================="
echo "Job finished: $(date)"
echo "=================================================================="
