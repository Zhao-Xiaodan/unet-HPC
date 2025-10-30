#!/usr/bin/bash
#PBS -N EdgeDet_Viz_Train
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -j oe
#PBS -P personal-phyzxi
#PBS -o EdgeDet_Viz_Train.o
#PBS -q normal

################################################################################
# Advanced Visualization for Trainable Gabor Edge Detector Model
#
# This script generates comprehensive visualizations for the trainable layer 1
# Gabor-initialized U-Net model:
#   - Feature inversions at correct dimensions (9 layers)
#   - PCA clustering analysis (9 layers)
#   - Representative feature maps
#   - Layer 1 Gabor filter comparison (initial vs final - shows adaptation!)
#
# Expected runtime: ~2 hours for 1 image (320x)
################################################################################

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
MODEL_PATH="./edge_detector_experiment/20251030_122737/unet_trainable_layer1/best_model.pth"
INITIAL_WEIGHTS="./edge_detector_experiment/20251030_122737/unet_trainable_layer1/layer1_weights_epoch000.pth"
TEST_IMAGES_DIR="./test_images"
OUTPUT_DIR="./edge_detector_viz_advanced_trainable_layer1"
VARIANT="trainable_layer1"
N_FILTERS=32
DROPOUT=0.2
N_CLUSTERS=8
TILE_ROW=3
TILE_COL=4
FILTER_IMAGE="320x"  # Process only 320x image

echo "Configuration:"
echo "  Variant: Trainable Layer 1 (Gabor filters can adapt)"
echo "  Model: $MODEL_PATH"
echo "  Initial weights: $INITIAL_WEIGHTS"
echo "  Test images: $TEST_IMAGES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  n_filters: $N_FILTERS"
echo "  dropout: $DROPOUT"
echo "  PCA clusters: $N_CLUSTERS"
echo "  Image filter: $FILTER_IMAGE"
echo ""

echo "Expected outputs:"
echo "  ✓ Feature inversions (9 layers: 512x512 → 32x32)"
echo "  ✓ PCA cluster scatter plots (9 layers)"
echo "  ✓ Representative feature maps (9 layers)"
echo "  ✓ Layer 1 Gabor comparison (initial → final, shows Gabor adaptation!)"
echo ""

echo "Key question:"
echo "  Did Gabor filters adapt toward textures (like baseline)?"
echo "  Or did they stay edge-like (suggesting edges are optimal)?"
echo ""

echo "========================================"
echo "Starting advanced visualization..."
echo "========================================"
echo ""

# Run advanced visualization
singularity exec --nv $image python visualize_edge_detector_advanced.py \
    --model_path $MODEL_PATH \
    --variant $VARIANT \
    --test_images_dir $TEST_IMAGES_DIR \
    --output_dir $OUTPUT_DIR \
    --n_filters $N_FILTERS \
    --dropout $DROPOUT \
    --n_clusters $N_CLUSTERS \
    --tile_row $TILE_ROW \
    --tile_col $TILE_COL \
    --filter_image $FILTER_IMAGE \
    --initial_weights $INITIAL_WEIGHTS

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

if [ -d "$OUTPUT_DIR/320x_2025-05-15_02-05-00" ]; then
    echo "Contents of $OUTPUT_DIR/320x_2025-05-15_02-05-00/:"
    ls -laR $OUTPUT_DIR/320x_2025-05-15_02-05-00/
    echo ""

    echo "Feature inversions:"
    ls -lh $OUTPUT_DIR/320x_2025-05-15_02-05-00/feature_inversions/ 2>/dev/null | wc -l | xargs echo "  ... (total:" | tr '\n' ' ' && echo "files)"
    echo ""

    echo "PCA clusters:"
    ls -lh $OUTPUT_DIR/320x_2025-05-15_02-05-00/feature_maps/pca_clusters/ 2>/dev/null | wc -l | xargs echo "  ... (total:" | tr '\n' ' ' && echo "files)"
    echo ""

    echo "Representative feature maps:"
    ls -lh $OUTPUT_DIR/320x_2025-05-15_02-05-00/feature_maps/representative_feature_maps_pca/ 2>/dev/null | wc -l | xargs echo "  ... (total:" | tr '\n' ' ' && echo "files)"
    echo ""

    echo "Layer 1 comparison (KEY RESULT!):"
    ls -lh $OUTPUT_DIR/320x_2025-05-15_02-05-00/layer1_comparison/ 2>/dev/null | wc -l | xargs echo "  ... (total:" | tr '\n' ' ' && echo "files)"
    echo ""

    echo "Layer 1 adaptation metrics:"
    cat $OUTPUT_DIR/320x_2025-05-15_02-05-00/layer1_comparison/layer1_metrics.json 2>/dev/null || echo "  (metrics not yet generated)"
fi

echo "========================================"
echo "✓ Job complete!"
echo "========================================"
echo ""

echo "Next steps:"
echo "1. Download results:"
echo "   scp -r <username>@hopper:~/scratch/unet-HPC/$OUTPUT_DIR ./"
echo ""
echo "2. Compare with frozen layer 1:"
echo "   # Check Layer 1 adaptation"
echo "   diff -y \\"
echo "     edge_detector_viz_advanced_frozen_layer1/*/layer1_comparison/layer1_metrics.json \\"
echo "     $OUTPUT_DIR/*/layer1_comparison/layer1_metrics.json"
echo ""
echo "3. Analyze Gabor → texture adaptation:"
echo "   # If cosine_similarity < 0.8: Gabor filters significantly changed"
echo "   # If cosine_similarity > 0.95: Gabor filters stayed mostly edge-like"
echo ""
echo "4. Compare feature inversions with baseline:"
echo "   # Do trainable layer 1 features resemble baseline or stay edge-like?"
echo "   diff -r unet_visualization_advanced_20251028_091857/ $OUTPUT_DIR/"
