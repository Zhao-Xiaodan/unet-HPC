#!/usr/bin/bash
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -k oed
#PBS -N CRP_Complete_Encoders
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# Complete CRP Analysis with Encoder Paths and Feature Maps
# ==============================================================================
#
# This script computes COMPLETE CRP analysis including:
# - Decoder path: decoder_1 → decoder_2 → decoder_3 → decoder_4 → bottleneck
# - Encoder path: bottleneck → encoder_4 → encoder_3 → encoder_2 → encoder_1
# - Skip connections: decoder ← encoder (4 lateral connections)
# - Feature map extraction for all layers and channels
#
# Total: 12 connections per image (decoder + encoder + skip)
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
MODEL_PATH="./best_models_PyTorch/unet/best_model.pth"
TEST_IMAGES_DIR="./test_images"
OUTPUT_DIR="./unet_crp_complete"
N_FILTERS=32
DROPOUT=0.2
TILE_ROW=3
TILE_COL=4

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test Images: $TEST_IMAGES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Tile Position: Row $TILE_ROW, Col $TILE_COL"
echo "  Analysis Type: Complete (decoder + encoder + skip)"
echo "========================================"
echo ""

# Check GPU
echo "GPU Information:"
singularity exec --nv $image nvidia-smi -L
echo ""

# ==============================================================================
# Run Complete CRP Analysis
# ==============================================================================

echo "========================================"
echo "RUNNING COMPLETE CRP ANALYSIS"
echo "========================================"
echo ""
echo "This analysis includes:"
echo "  - 4 decoder path connections"
echo "  - 4 encoder path connections"
echo "  - 4 skip connections"
echo "  - Feature maps for all layers"
echo ""

singularity exec --nv $image python unet_crp_complete_with_encoders.py \
    --model_path "$MODEL_PATH" \
    --test_images_dir "$TEST_IMAGES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_filters $N_FILTERS \
    --dropout $DROPOUT \
    --tile_row $TILE_ROW \
    --tile_col $TILE_COL

EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Analysis failed with exit code $EXIT_STATUS"
    exit $EXIT_STATUS
fi

echo ""
echo "✓ Complete CRP analysis finished"
echo ""

# ==============================================================================
# Generate Enhanced Interactive Visualization
# ==============================================================================

echo "========================================"
echo "GENERATING ENHANCED VISUALIZATION"
echo "========================================"
echo ""

# Find most recent output directory
LATEST_DIR=$(ls -td ${OUTPUT_DIR}_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "ERROR: Could not find output directory"
    exit 1
fi

echo "Using CRP data from: $LATEST_DIR"
echo ""

singularity exec $image python generate_enhanced_interactive_visualization.py \
    --crp_data_dir "$LATEST_DIR" \
    --output "$LATEST_DIR/crp_enhanced_visualization.html"

EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Visualization generation failed with exit code $EXIT_STATUS"
    exit $EXIT_STATUS
fi

echo ""
echo "✓ Enhanced interactive visualization generated"
echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo "========================================"
echo "JOB COMPLETE"
echo "========================================"
echo "End Time: $(date)"
echo ""
echo "Output directory: $LATEST_DIR"
echo ""
echo "Generated files:"
echo "  - metadata.json: Complete analysis metadata"
echo "  - crp_enhanced_visualization.html: Interactive web interface"
echo "  - <image_name>/<layer>_from_<layer>.npy: Relevance matrices (12 per image)"
echo "  - <image_name>/feature_maps/<layer>/ch###.png: Feature map visualizations"
echo ""
echo "Analysis includes:"
echo "  ✓ Decoder path (decoder_1 → bottleneck)"
echo "  ✓ Encoder path (bottleneck → encoder_1)"
echo "  ✓ Skip connections (decoder ← encoder)"
echo "  ✓ Feature maps for all layers"
echo ""
echo "To view the interactive visualization:"
echo "  1. Copy to your local machine:"
echo "     scp -r $LATEST_DIR <local_path>"
echo "  2. Open crp_enhanced_visualization.html in a web browser"
echo "  3. Interact with the graph to explore multi-hop channel dependencies"
echo ""
echo "Enhanced features:"
echo "  ✓ Multi-hop path tracing (decoder → encoder)"
echo "  ✓ Dynamic top-K slider (1-10)"
echo "  ✓ Feature map preview on node hover/click"
echo "  ✓ Skip connection toggle"
echo "  ✓ Image switching"
echo "  ✓ Zoom and pan"
echo "  ✓ Path depth grouping in info panel"
echo ""
echo "========================================"

exit 0
