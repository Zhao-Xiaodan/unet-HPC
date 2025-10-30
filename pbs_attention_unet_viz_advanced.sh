#!/usr/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N AttUNet_Viz_Adv
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# Attention U-Net Advanced Visualization
# ==============================================================================
#
# Enhanced visualization with:
# 1. Feature inversions at correct spatial resolutions (512→256→128→64→32)
# 2. PCA clustering to identify representative feature maps
# 3. Attention gate visualizations
# 4. Full feature map exports with PCA analysis
#
# Generates for each test image:
# - 3-panel figure (input, preprocessed, prediction)
# - Feature inversions for all layers (9 conv layers)
# - PCA cluster scatter plots (13 layers including 4 attention gates)
# - Representative feature maps (reduced from 32-512 channels to ~8 per layer)
#
# Expected runtime: ~6-8 hours (depends on number of test images)
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
TEST_IMAGES_DIR="./test_images"
OUTPUT_DIR="./attention_unet_visualization_advanced"
N_FILTERS=32
DROPOUT=0.1
N_CLUSTERS=8
TILE_ROW=3
TILE_COL=4

# Optional: Filter to process only specific images (e.g., "320x")
# Leave empty to process all images in TEST_IMAGES_DIR
FILTER_IMAGE="320x"

echo "Configuration:"
echo "  Model: Attention U-Net"
echo "  Model path: $MODEL_PATH"
echo "  Test images: $TEST_IMAGES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  n_filters: $N_FILTERS"
echo "  dropout: $DROPOUT"
echo "  PCA clusters: $N_CLUSTERS"
echo "  Tile position: row $TILE_ROW, col $TILE_COL"
if [ -n "$FILTER_IMAGE" ]; then
    echo "  Image filter: $FILTER_IMAGE"
fi
echo ""

echo "Features:"
echo "  ✓ Feature inversions at correct dimensions"
echo "  ✓ PCA clustering for representative feature maps"
echo "  ✓ Attention gate visualizations"
echo "  ✓ Full feature map exports"
echo ""

echo "Output structure per image:"
echo "  - <image_name>/"
echo "    ├── <image_name>_3panel.png"
echo "    ├── feature_inversions/"
echo "    │   ├── feature_inversion_encoder_1_conv2.png"
echo "    │   ├── feature_inversion_encoder_2_conv2.png"
echo "    │   └── ... (9 total)"
echo "    ├── feature_maps/"
echo "    │   ├── pca_clusters/"
echo "    │   │   ├── pca_clusters_encoder_1_conv2.png"
echo "    │   │   ├── pca_clusters_attention_gate_1.png  ← NEW!"
echo "    │   │   └── ... (13 total)"
echo "    │   └── representative_feature_maps_pca/"
echo "    │       ├── feature_map_encoder_1_conv2_pca.png"
echo "    │       ├── feature_map_attention_gate_1_pca.png  ← NEW!"
echo "    │       └── ... (13 total)"
echo "    └── tile_metadata.json"
echo ""

echo "========================================"
echo "Starting advanced visualization..."
echo "========================================"
echo ""

# Run advanced visualization
FILTER_ARG=""
if [ -n "$FILTER_IMAGE" ]; then
    FILTER_ARG="--filter_image $FILTER_IMAGE"
fi

singularity exec --nv $image python visualize_attention_unet_advanced.py \
    --model_path $MODEL_PATH \
    --test_images_dir $TEST_IMAGES_DIR \
    --output_dir $OUTPUT_DIR \
    --n_filters $N_FILTERS \
    --dropout $DROPOUT \
    --n_clusters $N_CLUSTERS \
    --tile_row $TILE_ROW \
    --tile_col $TILE_COL \
    $FILTER_ARG

echo ""
echo "========================================"
echo "Visualization complete!"
echo "End Time: $(date)"
echo "========================================"
echo ""

# Show output directory structure
echo "Output directory structure:"
ls -la $OUTPUT_DIR/
echo ""

echo "Sample outputs (first image):"
FIRST_IMAGE=$(ls -1 $OUTPUT_DIR/ | grep -v ".json" | head -1)
if [ -n "$FIRST_IMAGE" ]; then
    echo ""
    echo "Contents of $OUTPUT_DIR/$FIRST_IMAGE/:"
    ls -la "$OUTPUT_DIR/$FIRST_IMAGE/"
    echo ""
    echo "Feature inversions:"
    ls -1 "$OUTPUT_DIR/$FIRST_IMAGE/feature_inversions/" | head -5
    echo "  ... (total: $(ls -1 $OUTPUT_DIR/$FIRST_IMAGE/feature_inversions/ | wc -l) files)"
    echo ""
    echo "PCA clusters:"
    ls -1 "$OUTPUT_DIR/$FIRST_IMAGE/feature_maps/pca_clusters/" | head -5
    echo "  ... (total: $(ls -1 $OUTPUT_DIR/$FIRST_IMAGE/feature_maps/pca_clusters/ | wc -l) files)"
    echo ""
    echo "Representative feature maps (PCA):"
    ls -1 "$OUTPUT_DIR/$FIRST_IMAGE/feature_maps/representative_feature_maps_pca/" | head -5
    echo "  ... (total: $(ls -1 $OUTPUT_DIR/$FIRST_IMAGE/feature_maps/representative_feature_maps_pca/ | wc -l) files)"
fi

echo ""
echo "========================================"
echo "✓ Job complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Download results to local machine:"
echo "   scp -r <username>@hopper:~/scratch/unet-HPC/$OUTPUT_DIR ./"
echo ""
echo "2. Compare with standard U-Net results:"
echo "   diff -r unet_visualization_advanced_20251028_091857/ $OUTPUT_DIR/"
echo ""
echo "3. Analyze attention gate patterns:"
echo "   # Check attention_gate_* visualizations"
echo "   ls $OUTPUT_DIR/*/feature_maps/pca_clusters/pca_clusters_attention_gate_*.png"
echo ""
