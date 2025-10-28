#!/usr/bin/bash
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -k oed
#PBS -N UNet_CRP_Analysis
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8:ngpus=1:mem=32gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# U-Net Concept Relevance Propagation (CRP) Analysis
# ==============================================================================
#
# This script performs hierarchical concept analysis using CRP:
# 1. Traces concept composition from decoder_1_conv2 Ch4 (Cluster 6)
# 2. Identifies top 2 contributing channels from decoder_2, decoder_3, etc.
# 3. Generates hierarchical concept composition graph
# 4. Creates conditional heatmaps and masked reference samples
#
# Based on: "Concept Relevance Propagation" (Nature Machine Intelligence, 2023)
#
# Expected runtime: ~30 minutes
# ==============================================================================

# Configuration
# ==============================================================================

# Model path
MODEL_PATH="./best_models_PyTorch/unet/best_model.pth"

# Test image (from visualization analysis)
TEST_IMAGE="./test_images/320x_2025-05-15_02-05-00.tif"

# Starting point for CRP analysis (from PCA cluster analysis)
START_LAYER="decoder_1_conv2"
START_CHANNEL=4  # Ch4 in Cluster 6

# Analysis parameters
TOP_K=2  # Number of top contributing channels per layer
N_FILTERS=32
DROPOUT=0.2

# Output directory
OUTPUT_DIR="./unet_crp_analysis"

# ==============================================================================
# Setup
# ==============================================================================

# Load required modules
module load singularity

# Define singularity container (PyTorch)
image=/app1/common/singularity-img/hopper/pytorch/pytorch_2.4.0a0-cuda_12.5.0_ngc_24.06.sif

# Change to working directory
cd $PBS_O_WORKDIR

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
echo "  Model: $MODEL_PATH"
echo "  Test Image: $TEST_IMAGE"
echo "  Start: $START_LAYER Ch$START_CHANNEL"
echo "  Top-K: $TOP_K"
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
# Verify Files Exist
# ==============================================================================

echo "========================================"
echo "VERIFYING FILES"
echo "========================================"
echo ""

# Check if model exists, if not try to find it from source experiment
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠ Model not found at: $MODEL_PATH"
    echo "  Searching for model in source experiments..."

    # Try to find model info
    if [ -f "./best_models_PyTorch/unet/model_info.json" ]; then
        echo "  Found model_info.json, extracting source experiment..."
        SOURCE_EXP=$(singularity exec $image python -c "import json; d=json.load(open('./best_models_PyTorch/unet/model_info.json')); print(d['source_experiment'])" 2>/dev/null)
        MODEL_NAME=$(singularity exec $image python -c "import json; d=json.load(open('./best_models_PyTorch/unet/model_info.json')); print(d['model_name'])" 2>/dev/null)

        if [ -n "$SOURCE_EXP" ] && [ -n "$MODEL_NAME" ]; then
            echo "  Source: $SOURCE_EXP"
            echo "  Model: $MODEL_NAME"

            # Look for checkpoint in source experiment
            SOURCE_MODEL="./${SOURCE_EXP}/unet/checkpoints/${MODEL_NAME}/best_model.pth"
            if [ -f "$SOURCE_MODEL" ]; then
                echo "  ✓ Found checkpoint: $SOURCE_MODEL"
                MODEL_PATH="$SOURCE_MODEL"
            else
                echo "  ✗ Checkpoint not found: $SOURCE_MODEL"
            fi
        fi
    fi

    # If still not found, list available models
    if [ ! -f "$MODEL_PATH" ]; then
        echo ""
        echo "ERROR: Could not locate model checkpoint"
        echo ""
        echo "Available model directories:"
        ls -la ./best_models_PyTorch/*/ 2>/dev/null || echo "  No models cached"
        echo ""
        echo "Available experiment directories:"
        ls -ld ./pytorch_comparison_*/ 2>/dev/null | tail -5 || echo "  No experiments found"
        echo ""
        echo "Note: Run visualization analysis first to cache best models"
        exit 1
    fi
fi

if [ ! -f "$TEST_IMAGE" ]; then
    echo "ERROR: Test image not found: $TEST_IMAGE"
    echo ""
    echo "Available test images:"
    ls -la ./test_images/
    exit 1
fi

echo "✓ Model found: $MODEL_PATH"
echo "✓ Test image found: $TEST_IMAGE"
echo ""

# ==============================================================================
# Run CRP Analysis
# ==============================================================================

echo "========================================"
echo "RUNNING CRP ANALYSIS"
echo "========================================"
echo ""
echo "Tracing hierarchical concept composition:"
echo "  $START_LAYER Ch$START_CHANNEL"
echo "  ↓"
echo "  decoder_2_conv2 (top $TOP_K channels)"
echo "  ↓"
echo "  decoder_3_conv2 (top $TOP_K channels)"
echo "  ↓"
echo "  decoder_4_conv2 (top $TOP_K channels)"
echo "  ↓"
echo "  bottleneck_conv2 (top $TOP_K channels)"
echo ""

singularity exec --nv $image python unet_crp_hierarchical_concepts.py \
    --model_path "$MODEL_PATH" \
    --test_image "$TEST_IMAGE" \
    --start_layer "$START_LAYER" \
    --start_channel $START_CHANNEL \
    --top_k $TOP_K \
    --output_dir "$OUTPUT_DIR" \
    --n_filters $N_FILTERS \
    --dropout $DROPOUT

EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: CRP analysis failed with exit code $EXIT_STATUS"
    exit $EXIT_STATUS
fi

echo ""
echo "✓ CRP analysis complete"
echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo "========================================"
echo "JOB COMPLETE"
echo "========================================"
echo "End Time: $(date)"
echo ""

# Find most recent output directory
LATEST_DIR=$(ls -td ${OUTPUT_DIR}_* 2>/dev/null | head -1)

if [ -n "$LATEST_DIR" ]; then
    echo "Output directory: $LATEST_DIR"
    echo ""
    echo "Generated files:"
    ls -lh "$LATEST_DIR"/
    echo ""
    echo "Key results:"
    echo "  - input_tile.png: Input image tile"
    echo "  - hierarchy.json: Hierarchical concept composition data"
    echo "  - hierarchical_concept_graph.png: Concept composition graph"
    echo ""

    # Display hierarchy if JSON exists
    if [ -f "$LATEST_DIR/hierarchy.json" ]; then
        echo "Concept Hierarchy Summary:"
        echo "========================================"
        singularity exec $image python -c "
import json
import sys
with open('$LATEST_DIR/hierarchy.json') as f:
    hierarchy = json.load(f)
for layer in sorted(hierarchy.keys(), key=lambda x: int(x.split('_')[1]) if 'decoder' in x else 999):
    info = hierarchy[layer]
    print(f'\n{layer}:')
    print(f'  Channels: {info[\"channels\"]}')
    if 'contributes_from' in info:
        cf = info['contributes_from']
        print(f'  ← {cf[\"layer\"]}:')
        for ch, rel in zip(cf['channels'], cf['relevance']):
            print(f'      Ch{ch}: {rel:.4f}')
"
    fi
else
    echo "⚠ Output directory not found"
fi

echo ""
echo "========================================"

exit 0
