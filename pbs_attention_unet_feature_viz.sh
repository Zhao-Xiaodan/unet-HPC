#!/usr/bin/bash
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -k oed
#PBS -N AttUNet_Feature_Viz
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# Attention U-Net Feature Visualization
# ==============================================================================
#
# Implements optimization-based feature visualization techniques from:
# "Feature Visualization" - Olah et al., Distill 2017
#
# Generates:
# - Synthetic images that maximally activate specific channels
# - Diverse visualizations showing different patterns per channel
# - Attention gate feature visualizations (unique to Attention U-Net)
# - DeepDream amplification of network-detected patterns
#
# Expected runtime: ~2-3 hours
# ==============================================================================

# Load modules
module load singularity

# Singularity container
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

# Change to working directory
cd $PBS_O_WORKDIR

echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "========================================"
echo ""

# Configuration
MODEL_PATH="./best_models_PyTorch/attention_unet/best_model.pth"
OUTPUT_DIR="./attention_unet_feature_viz"
TEST_IMAGES_DIR="./test_images"
N_FILTERS=32
DROPOUT=0.1

# Visualization parameters
LAYERS="encoder_1_conv2 encoder_2_conv2 encoder_3_conv2 bottleneck_conv2 attention_gate_1 attention_gate_3 decoder_3_conv2 decoder_1_conv2"
CHANNELS_PER_LAYER=12
DIVERSE_PER_CHANNEL=3
ITERATIONS=500
IMAGE_SIZE=512

echo "Configuration:"
echo "  Model: Attention U-Net"
echo "  Model path: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Layers: $LAYERS"
echo "  Channels per layer: $CHANNELS_PER_LAYER"
echo "  Diverse examples: $DIVERSE_PER_CHANNEL"
echo "  Iterations: $ITERATIONS"
echo "========================================"
echo ""
echo "KEY FEATURE:"
echo "  This script visualizes attention gates!"
echo "  Compare attention-modulated vs standard features"
echo "========================================"
echo ""

# Check GPU
echo "GPU Information:"
singularity exec --nv $image nvidia-smi -L
echo ""

# ==============================================================================
# Run Feature Visualization
# ==============================================================================

echo "========================================"
echo "RUNNING FEATURE VISUALIZATION"
echo "========================================"
echo ""
echo "This will generate:"
echo "  - Synthetic images maximizing channel activations"
echo "  - Grid visualizations for each layer"
echo "  - Activation optimization histories"
echo "  - Attention gate visualizations (NEW!)"
echo "  - DeepDream enhancements (optional)"
echo ""

singularity exec --nv $image python attention_unet_feature_visualization.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_filters $N_FILTERS \
    --dropout $DROPOUT \
    --layers $LAYERS \
    --channels_per_layer $CHANNELS_PER_LAYER \
    --diverse_per_channel $DIVERSE_PER_CHANNEL \
    --iterations $ITERATIONS \
    --image_size $IMAGE_SIZE \
    --deepdream \
    --test_images_dir "$TEST_IMAGES_DIR"

EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Feature visualization failed with exit code $EXIT_STATUS"
    exit $EXIT_STATUS
fi

echo ""
echo "✓ Feature visualization complete"
echo ""

# ==============================================================================
# Summary
# ==============================================================================

# Find most recent output directory
LATEST_DIR=$(ls -td ${OUTPUT_DIR}_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "ERROR: Could not find output directory"
    exit 1
fi

echo "========================================"
echo "JOB COMPLETE"
echo "========================================"
echo "End Time: $(date)"
echo ""
echo "Output directory: $LATEST_DIR"
echo ""
echo "Generated files:"
echo "  - Grid visualizations: <layer>_diverse_visualizations.png"
echo "  - Individual channels: <layer>/ch<###>_div<#>.png"
echo "  - Optimization history: <layer>/ch<###>_div<#>_history.png"
echo "  - DeepDream results: deepdream/*.png"
echo "  - Metadata: metadata.json"
echo ""
echo "To view results:"
echo "  1. Copy to your local machine:"
echo "     scp -r $LATEST_DIR <local_path>"
echo "  2. Open PNG files to see visualizations"
echo "  3. Check grid images for layer-level overview"
echo ""
echo "What the visualizations show:"
echo "  - Early layers (encoder_1): Edge detectors, simple patterns"
echo "  - Middle layers (encoder_3): Texture patterns, object parts"
echo "  - Bottleneck: High-level semantic features"
echo "  - Attention gates: What patterns attention focuses on (UNIQUE!)"
echo "  - Decoder layers: Reconstruction patterns, spatial features"
echo ""
echo "Diverse examples per channel reveal:"
echo "  - Multiple patterns that activate the same channel"
echo "  - Invariances and transformations learned"
echo "  - Different aspects of what the channel detects"
echo ""
echo "ATTENTION U-NET SPECIFIC INSIGHTS:"
echo "  - Compare attention_gate_X with encoder_X layers"
echo "  - See how attention modulates feature representations"
echo "  - Identify which patterns attention suppresses vs enhances"
echo ""
echo "========================================"

exit 0
