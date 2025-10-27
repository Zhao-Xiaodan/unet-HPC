#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N UNet_Viz_Advanced
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# U-Net Feature Visualization - ADVANCED VERSION
# ==============================================================================
#
# Enhanced visualization with:
# 1. Feature inversions at CORRECT spatial resolutions (512→256→128→64→32)
# 2. UMAP clustering to identify representative feature maps
#
# Features:
# - Inverts from dimensions matching each layer architecture
# - Clusters similar feature maps using UMAP + KMeans
# - Shows only representative feature maps per cluster (reduces redundancy)
# - Generates UMAP scatter plots showing feature map clusters
#
# Input:
#   - Trained models from ./best_models_PyTorch/
#   - All test images in ./test_images/ (8 images)
#
# Output (per image):
#   - 3-panel figure: original + preprocessed + prediction
#   - UMAP cluster plots: showing how feature maps group
#   - Representative feature maps: ~8 per layer (one per cluster)
#   - Feature inversions: 9 images at correct dimensions
#
# Expected runtime: ~4-6 hours for 8 images
# ==============================================================================

# Configuration
# ==============================================================================

# Test images directory
TEST_IMAGES_DIR="./test_images"

# Tile position (row 3, column 4 = center-ish)
TILE_ROW=3
TILE_COL=4

# Model cache directory
MODEL_CACHE="./best_models_PyTorch"

# Output directory
OUTPUT_BASE="unet_visualization_advanced"

# Number of UMAP clusters per layer (default: 8)
# This determines how many representative feature maps to show
N_CLUSTERS=8

# ==============================================================================
# Setup
# ==============================================================================

module load singularity
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

cd $PBS_O_WORKDIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_BASE}_${TIMESTAMP}"

echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Job Name: $PBS_JOBNAME"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Working Directory: $PWD"
echo "========================================"
echo ""

echo "Configuration:"
echo "  Test Images: $TEST_IMAGES_DIR"
echo "  Tile Position: row $TILE_ROW, col $TILE_COL"
echo "  Model Cache: $MODEL_CACHE"
echo "  UMAP Clusters: $N_CLUSTERS"
echo "  Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# GPU info
echo "GPU Information:"
singularity exec --nv $image nvidia-smi
echo ""

# Python/PyTorch versions
echo "Software versions:"
singularity exec --nv $image python --version
singularity exec --nv $image python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo ""

# Check UMAP availability
echo "Checking UMAP availability:"
singularity exec --nv $image python -c "try:
    import umap
    print('✓ UMAP available')
except ImportError:
    print('✗ UMAP not available (will skip clustering)')
    print('  Install with: pip install umap-learn')"
echo ""

# ==============================================================================
# Verify Prerequisites
# ==============================================================================

echo "========================================"
echo "VERIFYING PREREQUISITES"
echo "========================================"
echo ""

if [ ! -d "$TEST_IMAGES_DIR" ]; then
    echo "ERROR: Test images directory not found: $TEST_IMAGES_DIR"
    exit 1
fi

NUM_IMAGES=$(ls -1 "$TEST_IMAGES_DIR"/*.tif 2>/dev/null | wc -l)
if [ "$NUM_IMAGES" -eq 0 ]; then
    echo "ERROR: No .tif images in $TEST_IMAGES_DIR"
    exit 1
fi
echo "✓ Found $NUM_IMAGES test images"

if [ ! -d "$MODEL_CACHE" ]; then
    echo "ERROR: Model cache not found: $MODEL_CACHE"
    echo "Run pbs_pytorch_density_analysis.sh first"
    exit 1
fi

MODEL_COUNT=$(find "$MODEL_CACHE" -name "best_model.pth" | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "ERROR: No cached models in $MODEL_CACHE"
    exit 1
fi
echo "✓ Found $MODEL_COUNT cached model(s)"
echo ""

# ==============================================================================
# Run Visualization
# ==============================================================================

echo "========================================"
echo "RUNNING ADVANCED U-NET VISUALIZATION"
echo "========================================"
echo ""
echo "Processing $NUM_IMAGES test images..."
echo ""
echo "Enhanced features:"
echo "  1. Feature inversions at correct dimensions"
echo "     - encoder_1: 512×512"
echo "     - encoder_2: 256×256 (after pool1)"
echo "     - encoder_3: 128×128 (after pool2)"
echo "     - encoder_4: 64×64 (after pool3)"
echo "     - bottleneck: 32×32 (after pool4)"
echo "     - decoder layers: 64→128→256→512"
echo ""
echo "  2. UMAP clustering of feature maps"
echo "     - Reduces ~32-512 feature maps to $N_CLUSTERS representatives"
echo "     - Shows UMAP scatter plots with cluster assignments"
echo "     - Plots only representative feature maps per cluster"
echo ""
echo "Output per image:"
echo "  - 1 × 3-panel figure"
echo "  - 9 × UMAP cluster scatter plots"
echo "  - 9 × representative feature maps (~$N_CLUSTERS channels each)"
echo "  - 9 × feature inversions (at correct dimensions)"
echo ""
echo "Estimated time: 4-6 hours total (~30-45 min per image)"
echo "========================================"
echo ""

singularity exec --nv $image python visualize_unet_features_advanced.py \
    --model_cache "$MODEL_CACHE" \
    --test_images "$TEST_IMAGES_DIR" \
    --output "$OUTPUT_DIR" \
    --tile_row $TILE_ROW \
    --tile_col $TILE_COL \
    --n_clusters $N_CLUSTERS

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
echo "  │   │   ├── umap_clusters/"
echo "  │   │   │   ├── umap_clusters_encoder_1_conv2.png"
echo "  │   │   │   ├── umap_clusters_encoder_2_conv2.png"
echo "  │   │   │   ├── ... (9 UMAP plots)"
echo "  │   │   │   └── umap_clusters_decoder_1_conv2.png"
echo "  │   │   └── representative_feature_maps/"
echo "  │   │       ├── feature_map_encoder_1_conv2.png (~$N_CLUSTERS reps)"
echo "  │   │       ├── feature_map_encoder_2_conv2.png"
echo "  │   │       ├── ... (9 layers)"
echo "  │   │       └── feature_map_decoder_1_conv2.png"
echo "  │   ├── feature_inversions/"
echo "  │   │   ├── feature_inversion_encoder_1_conv2.png (from 512×512)"
echo "  │   │   ├── feature_inversion_encoder_2_conv2.png (from 256×256)"
echo "  │   │   ├── feature_inversion_encoder_3_conv2.png (from 128×128)"
echo "  │   │   ├── feature_inversion_encoder_4_conv2.png (from 64×64)"
echo "  │   │   ├── feature_inversion_bottleneck_conv2.png (from 32×32)"
echo "  │   │   ├── feature_inversion_decoder_4_conv2.png (from 64×64)"
echo "  │   │   ├── feature_inversion_decoder_3_conv2.png (from 128×128)"
echo "  │   │   ├── feature_inversion_decoder_2_conv2.png (from 256×256)"
echo "  │   │   └── feature_inversion_decoder_1_conv2.png (from 512×512)"
echo "  │   └── tile_metadata.json"
echo "  ├── <image_2>/"
echo "  │   └── ..."
echo "  ├── ... (8 images total)"
echo "  └── visualization_metadata.json"
echo ""
echo "Key improvements over basic version:"
echo "  ✓ Feature inversions match architecture dimensions"
echo "  ✓ UMAP clustering reduces feature map redundancy"
echo "  ✓ Only ~$N_CLUSTERS representative features per layer (vs 32-512)"
echo "  ✓ Cluster visualization shows feature map groupings"
echo ""
echo "Total files per image: ~27 visualizations"
echo "  - 1 3-panel figure"
echo "  - 9 UMAP cluster plots"
echo "  - 9 representative feature maps"
echo "  - 9 feature inversions"
echo ""
echo "========================================"
echo ""
echo "To view results, sync from HPC:"
echo "  rsync -avz <user>@hpc:$(pwd)/$OUTPUT_DIR /local/path/"
echo ""
echo "========================================"

exit 0
