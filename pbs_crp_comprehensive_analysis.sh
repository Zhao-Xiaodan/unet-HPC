#!/usr/bin/bash
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -k oed
#PBS -N CRP_Comprehensive
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# Comprehensive CRP Analysis - All Test Images
# ==============================================================================
#
# This script computes COMPLETE relevance matrices (not just top-K) for all
# test images. This provides full channel-to-channel connectivity data for
# interactive visualization.
#
# Expected runtime: ~2-3 hours (depends on number of test images)
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
OUTPUT_DIR="./unet_crp_comprehensive"
N_FILTERS=32
DROPOUT=0.2
TILE_ROW=3
TILE_COL=4

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test Images: $TEST_IMAGES_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Tile Position: Row $TILE_ROW, Col $TILE_COL"
echo "========================================"
echo ""

# Check GPU
echo "GPU Information:"
singularity exec --nv $image nvidia-smi -L
echo ""

# ==============================================================================
# Run Comprehensive CRP Analysis
# ==============================================================================

echo "========================================"
echo "RUNNING COMPREHENSIVE CRP ANALYSIS"
echo "========================================"
echo ""

singularity exec --nv $image python unet_crp_comprehensive_analysis.py \
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
echo "✓ Comprehensive CRP analysis complete"
echo ""

# ==============================================================================
# Generate Interactive Visualization
# ==============================================================================

echo "========================================"
echo "GENERATING INTERACTIVE VISUALIZATION"
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

singularity exec $image python generate_interactive_crp_visualization.py \
    --crp_data_dir "$LATEST_DIR" \
    --output "$LATEST_DIR/crp_interactive_visualization.html"

EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Visualization generation failed with exit code $EXIT_STATUS"
    exit $EXIT_STATUS
fi

echo ""
echo "✓ Interactive visualization generated"
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
echo "  - metadata.json: Analysis metadata"
echo "  - crp_interactive_visualization.html: Interactive web interface"
echo "  - <image_name>/<layer>_from_<layer>.npy: Relevance matrices"
echo ""
echo "To view the interactive visualization:"
echo "  1. Copy to your local machine:"
echo "     scp -r $LATEST_DIR <local_path>"
echo "  2. Open crp_interactive_visualization.html in a web browser"
echo "  3. Interact with the graph to explore channel dependencies"
echo ""
echo "Features:"
echo "  - Hover/click nodes to highlight top-K paths"
echo "  - Adjust K dynamically with slider (1-10)"
echo "  - Switch between different test images"
echo "  - Zoom and pan the graph"
echo "  - View relevance values on edges"
echo ""
echo "========================================"

exit 0
