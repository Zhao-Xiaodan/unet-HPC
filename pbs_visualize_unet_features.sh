#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N UNet_Visualization
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# U-Net Feature Visualization - All Test Images
# ==============================================================================
#
# This script visualizes the internal workings of trained U-Net models by:
# 1. Extracting center tiles (row 3, col 4) from ALL test images
# 2. Visualizing feature maps at BOTH conv layers in each encoder/decoder block
# 3. Reconstructing input from layer activations (feature inversion)
#
# Input:
#   - Trained models from ./best_models_PyTorch/
#   - All test images in ./test_images/ (8 images)
#
# Output (per image):
#   - 3-panel figure: original + preprocessed + prediction
#   - Feature maps: 18 images (9 blocks × 2 conv layers, showing 32 channels each)
#   - Feature inversions: 9 images (all conv2 layers from encoder to decoder)
#
# Expected runtime: ~4-6 hours for 8 images
#   - Feature maps: ~10 min per image
#   - Feature inversions: ~45-60 min per image (9 layers × 5-7 min each)
# ==============================================================================

# Configuration - MODIFY THESE VARIABLES IF NEEDED
# ==============================================================================

# Test images directory (will process ALL .tif images)
TEST_IMAGES_DIR="./test_images"

# Tile position for extraction (1-indexed)
# Row 3, Column 4 = center-ish tile for most images
TILE_ROW=3
TILE_COL=4

# Model cache directory (created by pbs_pytorch_density_analysis.sh)
MODEL_CACHE="./best_models_PyTorch"

# Output directory (will be created with timestamp)
OUTPUT_BASE="unet_visualization"

# ==============================================================================
# Setup
# ==============================================================================

# Load required modules
module load singularity

# Define singularity container (PyTorch)
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

# Change to working directory
cd $PBS_O_WORKDIR

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

# Print job information
echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Working Directory: $PWD"
echo "========================================"
echo ""

echo "Configuration:"
echo "  Test Images Dir: $TEST_IMAGES_DIR"
echo "  Tile Position: row $TILE_ROW, column $TILE_COL"
echo "  Model Cache: $MODEL_CACHE"
echo "  Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Check GPU availability
echo "GPU Information:"
singularity exec --nv $image nvidia-smi
echo ""

# Print Python and PyTorch versions
echo "Python version:"
singularity exec --nv $image python --version
echo ""

echo "PyTorch version:"
singularity exec --nv $image python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# ==============================================================================
# Verify Prerequisites
# ==============================================================================

echo "========================================"
echo "VERIFYING PREREQUISITES"
echo "========================================"
echo ""

# Check if test images directory exists
if [ ! -d "$TEST_IMAGES_DIR" ]; then
    echo "ERROR: Test images directory not found: $TEST_IMAGES_DIR"
    exit 1
fi

# Count test images
NUM_IMAGES=$(ls -1 "$TEST_IMAGES_DIR"/*.tif 2>/dev/null | wc -l)
if [ "$NUM_IMAGES" -eq 0 ]; then
    echo "ERROR: No .tif images found in $TEST_IMAGES_DIR"
    exit 1
fi
echo "✓ Found $NUM_IMAGES test images in $TEST_IMAGES_DIR"

# Check if model cache exists
if [ ! -d "$MODEL_CACHE" ]; then
    echo "ERROR: Model cache directory not found: $MODEL_CACHE"
    echo ""
    echo "Please run pbs_pytorch_density_analysis.sh first to cache best models:"
    echo "  qsub pbs_pytorch_density_analysis.sh"
    echo ""
    exit 1
fi

# Check for cached models
MODEL_COUNT=$(find "$MODEL_CACHE" -name "best_model.pth" | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "ERROR: No cached models found in $MODEL_CACHE"
    echo ""
    echo "Please run pbs_pytorch_density_analysis.sh first to cache best models:"
    echo "  qsub pbs_pytorch_density_analysis.sh"
    echo ""
    exit 1
fi
echo "✓ Found $MODEL_COUNT cached model(s)"
echo ""

# ==============================================================================
# Run Visualization
# ==============================================================================

echo "========================================"
echo "RUNNING U-NET FEATURE VISUALIZATION"
echo "========================================"
echo ""
echo "Processing $NUM_IMAGES test images..."
echo "Each image will generate:"
echo "  - 1 × 3-panel figure (original + preprocessed + prediction)"
echo "  - 18 × feature maps (encoder_1-4, bottleneck, decoder_1-4, conv1 & conv2)"
echo "  - 9 × feature inversions (all conv2 layers)"
echo ""
echo "Estimated time: 4-6 hours total (~30-45 min per image)"
echo "  - Feature maps: ~10 min per image"
echo "  - Feature inversions: ~35-45 min per image"
echo ""
echo "Note: Feature inversions involve optimization and take significant time"
echo "========================================"
echo ""

singularity exec --nv $image python visualize_unet_features.py \
    --model_cache "$MODEL_CACHE" \
    --test_images "$TEST_IMAGES_DIR" \
    --output "$OUTPUT_DIR" \
    --tile_row $TILE_ROW \
    --tile_col $TILE_COL

# Capture exit status
EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Visualization failed with exit code $EXIT_STATUS"
    exit $EXIT_STATUS
fi

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "========================================"
echo "VISUALIZATION COMPLETE"
echo "========================================"
echo "End Time: $(date)"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated structure:"
echo ""
echo "$OUTPUT_DIR/"
echo "  ├── <image_1>/"
echo "  │   ├── <image_1>_3panel.png"
echo "  │   ├── feature_maps/"
echo "  │   │   ├── feature_map_encoder_1_conv1.png"
echo "  │   │   ├── feature_map_encoder_1_conv2.png"
echo "  │   │   ├── feature_map_encoder_2_conv1.png"
echo "  │   │   ├── feature_map_encoder_2_conv2.png"
echo "  │   │   ├── feature_map_encoder_3_conv1.png"
echo "  │   │   ├── feature_map_encoder_3_conv2.png"
echo "  │   │   ├── feature_map_encoder_4_conv1.png"
echo "  │   │   ├── feature_map_encoder_4_conv2.png"
echo "  │   │   ├── feature_map_bottleneck_conv1.png"
echo "  │   │   ├── feature_map_bottleneck_conv2.png"
echo "  │   │   ├── feature_map_decoder_4_conv1.png"
echo "  │   │   ├── feature_map_decoder_4_conv2.png"
echo "  │   │   ├── feature_map_decoder_3_conv1.png"
echo "  │   │   ├── feature_map_decoder_3_conv2.png"
echo "  │   │   ├── feature_map_decoder_2_conv1.png"
echo "  │   │   ├── feature_map_decoder_2_conv2.png"
echo "  │   │   ├── feature_map_decoder_1_conv1.png"
echo "  │   │   └── feature_map_decoder_1_conv2.png"
echo "  │   ├── feature_inversions/"
echo "  │   │   ├── feature_inversion_encoder_1_conv2.png"
echo "  │   │   ├── feature_inversion_encoder_2_conv2.png"
echo "  │   │   ├── feature_inversion_encoder_3_conv2.png"
echo "  │   │   ├── feature_inversion_encoder_4_conv2.png"
echo "  │   │   ├── feature_inversion_bottleneck_conv2.png"
echo "  │   │   ├── feature_inversion_decoder_4_conv2.png"
echo "  │   │   ├── feature_inversion_decoder_3_conv2.png"
echo "  │   │   ├── feature_inversion_decoder_2_conv2.png"
echo "  │   │   └── feature_inversion_decoder_1_conv2.png"
echo "  │   └── tile_metadata.json"
echo "  ├── <image_2>/"
echo "  │   └── ..."
echo "  ├── ... (8 images total)"
echo "  └── visualization_metadata.json"
echo ""
echo "Total per image:"
echo "  - 1 × 3-panel figure"
echo "  - 18 × feature maps (32 channels each)"
echo "  - 9 × feature inversions"
echo ""
echo "Total files: ~$((NUM_IMAGES * 28)) visualization images"
echo ""
echo "========================================"
echo ""
echo "To view results, sync from HPC to your local machine:"
echo "  rsync -avz <user>@hpc:$(pwd)/$OUTPUT_DIR /local/path/"
echo ""
echo "========================================"

exit 0
