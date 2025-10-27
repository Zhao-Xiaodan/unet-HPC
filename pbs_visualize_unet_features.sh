#!/bin/bash
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -k oed
#PBS -N UNet_Visualization
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# U-Net Feature Visualization
# ==============================================================================
#
# This script visualizes the internal workings of trained U-Net models by:
# 1. Extracting a representative 512x512 tile from test images
# 2. Visualizing feature maps at each encoder and decoder layer
# 3. Reconstructing input from layer activations (feature inversion)
#
# Input:
#   - Trained models from ./best_models_PyTorch/
#   - Test image (512x512 tile will be extracted)
#
# Output:
#   - Input tile (original and preprocessed)
#   - Model prediction
#   - Feature maps for 9 layers (encoder_1 to decoder_1)
#   - Feature inversions for 6 key layers
#
# Expected runtime: ~30-60 minutes (feature inversions take time)
# ==============================================================================

# Configuration - MODIFY THESE VARIABLES
# ==============================================================================

# Test image to visualize (will extract 512x512 tile)
# Choose a representative image from your test set
TEST_IMAGE="./test_images/1280x_2025-05-16_00-59-00_002.tif"

# Tile position (top-left corner of 512x512 tile to extract)
# Default: (0, 0) extracts top-left corner
# Adjust these to select different regions of the image
TILE_X=0
TILE_Y=0

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
echo "  Test Image: $TEST_IMAGE"
echo "  Tile Position: ($TILE_X, $TILE_Y)"
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

# Check if test image exists
if [ ! -f "$TEST_IMAGE" ]; then
    echo "ERROR: Test image not found: $TEST_IMAGE"
    exit 1
fi
echo "✓ Test image found: $TEST_IMAGE"

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
echo "Note: Feature inversions involve optimization and may take 30-60 minutes"
echo ""

singularity exec --nv $image python visualize_unet_features.py \
    --model_cache "$MODEL_CACHE" \
    --test_image "$TEST_IMAGE" \
    --output "$OUTPUT_DIR" \
    --tile_x $TILE_X \
    --tile_y $TILE_Y

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
echo "Generated files:"
echo "  Input:"
echo "    - $OUTPUT_DIR/input_tile_original.png"
echo "    - $OUTPUT_DIR/input_tile_preprocessed.png"
echo "    - $OUTPUT_DIR/prediction.png"
echo ""
echo "  Feature Maps (9 layers):"
echo "    - $OUTPUT_DIR/feature_maps/feature_map_encoder_1.png"
echo "    - $OUTPUT_DIR/feature_maps/feature_map_encoder_2.png"
echo "    - $OUTPUT_DIR/feature_maps/feature_map_encoder_3.png"
echo "    - $OUTPUT_DIR/feature_maps/feature_map_encoder_4.png"
echo "    - $OUTPUT_DIR/feature_maps/feature_map_bottleneck.png"
echo "    - $OUTPUT_DIR/feature_maps/feature_map_decoder_4.png"
echo "    - $OUTPUT_DIR/feature_maps/feature_map_decoder_3.png"
echo "    - $OUTPUT_DIR/feature_maps/feature_map_decoder_2.png"
echo "    - $OUTPUT_DIR/feature_maps/feature_map_decoder_1.png"
echo ""
echo "  Feature Inversions (6 layers):"
echo "    - $OUTPUT_DIR/feature_inversions/feature_inversion_encoder_1.png"
echo "    - $OUTPUT_DIR/feature_inversions/feature_inversion_encoder_2.png"
echo "    - $OUTPUT_DIR/feature_inversions/feature_inversion_encoder_3.png"
echo "    - $OUTPUT_DIR/feature_inversions/feature_inversion_encoder_4.png"
echo "    - $OUTPUT_DIR/feature_inversions/feature_inversion_bottleneck.png"
echo "    - $OUTPUT_DIR/feature_inversions/feature_inversion_decoder_1.png"
echo ""
echo "  Metadata:"
echo "    - $OUTPUT_DIR/visualization_metadata.json"
echo ""
echo "========================================"
echo ""
echo "To view results, sync from HPC to your local machine:"
echo "  rsync -avz <user>@hpc:<path>/$OUTPUT_DIR /local/path/"
echo ""
echo "========================================"

exit 0
