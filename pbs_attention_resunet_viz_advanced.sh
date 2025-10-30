#!/usr/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N AttResUNet_Viz_Adv
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# Attention ResU-Net Advanced Visualization
# ==============================================================================
#
# Enhanced visualization for Attention ResU-Net with:
# 1. Feature inversions at correct spatial resolutions (512→256→128→64→32)
# 2. PCA clustering to identify representative feature maps
# 3. Attention gate visualizations
# 4. Residual block feature analysis
# 5. Full feature map exports with PCA analysis
#
# Generates for each test image:
# - 3-panel figure (input, preprocessed, prediction)
# - Feature inversions for all residual blocks (9 resconv layers)
# - PCA cluster scatter plots (13 layers including 4 attention gates)
# - Representative feature maps (reduced from 64-1024 channels to ~8 per layer)
#
# Note: Attention ResU-Net uses n_filters=64 (4× more capacity than standard U-Net)
#
# Expected runtime: ~8-10 hours (depends on number of test images)
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
MODEL_PATH="./best_models_PyTorch/attention_resunet/best_model.pth"
TEST_IMAGES_DIR="./test_images"
OUTPUT_DIR="./attention_resunet_visualization_advanced"
N_FILTERS=64  # Trained model uses 64 base filters (not 32)
DROPOUT=0.1
N_CLUSTERS=8
TILE_ROW=3
TILE_COL=4

# Optional: Filter to process only specific images (e.g., "320x")
# Leave empty to process all images in TEST_IMAGES_DIR
FILTER_IMAGE="320x"

echo "Configuration:"
echo "  Model: Attention ResU-Net"
echo "  Model path: $MODEL_PATH"
echo "  Test images: $TEST_IMAGES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  n_filters: $N_FILTERS (4× more than standard U-Net)"
echo "  dropout: $DROPOUT"
echo "  PCA clusters: $N_CLUSTERS"
echo "  Tile position: row $TILE_ROW, col $TILE_COL"
if [ -n "$FILTER_IMAGE" ]; then
    echo "  Image filter: $FILTER_IMAGE"
fi
echo ""

echo "Model characteristics:"
echo "  ✓ Residual connections (skip connections within blocks)"
echo "  ✓ Attention gates (focus on salient features)"
echo "  ✓ n_filters=64 → ~7.7M parameters (vs ~1.9M for n_filters=32)"
echo "  ✓ Best Val IoU: ~0.63"
echo ""

echo "Features:"
echo "  ✓ Feature inversions at correct dimensions"
echo "  ✓ PCA clustering for representative feature maps"
echo "  ✓ Attention gate visualizations"
echo "  ✓ Residual block feature analysis"
echo "  ✓ Full feature map exports"
echo ""

echo "Output structure per image:"
echo "  - <image_name>/"
echo "    ├── <image_name>_3panel.png"
echo "    ├── feature_inversions/"
echo "    │   ├── feature_inversion_encoder_1_resconv.png  ← Residual block"
echo "    │   ├── feature_inversion_encoder_2_resconv.png"
echo "    │   └── ... (9 total)"
echo "    ├── feature_maps/"
echo "    │   ├── pca_clusters/"
echo "    │   │   ├── pca_clusters_encoder_1_resconv.png"
echo "    │   │   ├── pca_clusters_attention_gate_1.png  ← Attention"
echo "    │   │   └── ... (13 total)"
echo "    │   └── representative_feature_maps_pca/"
echo "    │       ├── feature_map_encoder_1_resconv_pca.png"
echo "    │       ├── feature_map_attention_gate_1_pca.png"
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

singularity exec --nv $image python visualize_attention_resunet_advanced.py \
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
    echo "Feature inversions (residual blocks):"
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
echo "2. Compare with Attention U-Net results:"
echo "   # ResConv blocks vs standard ConvBlocks"
echo "   diff -r attention_unet_visualization_advanced/ $OUTPUT_DIR/"
echo ""
echo "3. Compare with standard U-Net results:"
echo "   # Residual connections + attention vs neither"
echo "   diff -r unet_visualization_advanced_20251028_091857/ $OUTPUT_DIR/"
echo ""
echo "4. Analyze residual block features:"
echo "   # Check encoder_*_resconv visualizations"
echo "   # Look for evidence of residual connections"
echo "   ls $OUTPUT_DIR/*/feature_maps/pca_clusters/pca_clusters_encoder_*_resconv.png"
echo ""
echo "5. Compare attention gate patterns across models:"
echo "   # Attention U-Net vs Attention ResU-Net"
echo "   # Do attention gates behave differently with residual connections?"
echo ""
