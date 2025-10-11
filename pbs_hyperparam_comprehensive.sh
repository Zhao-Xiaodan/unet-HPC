#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Hyperparam_Comprehensive
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg

# ==============================================================================
# Comprehensive Hyperparameter Search for Microbead Segmentation
# ==============================================================================
#
# Tests combinations of:
# - Architectures: U-Net, ResU-Net, Attention ResU-Net
# - Batch Sizes: 8, 16, 32 (memory-optimized for 512×512)
# - Loss Functions: Focal, Combined, Focal Tversky, Combined Tversky
# - Dataset: dataset_shrunk_masks (512×512)
#
# Fixed parameters (based on previous analysis):
# - Learning Rate: 5e-5 (lower for stability with variable batch sizes)
# - Dropout: 0.3 (worked well previously)
# - Early Stopping Patience: 30 epochs (increased for convergence)
#
# Estimated runtime:
# - Random search (30 configs): 7-15 hours
# - Grid search (36 configs): 9-18 hours
# ==============================================================================

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

# TensorFlow environment variables
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

# Memory optimization for large batch sizes
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Load required modules
module load singularity

# Use the modern TensorFlow container
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

echo "=============================================================================="
echo "Comprehensive Hyperparameter Search - Microbead Segmentation"
echo "=============================================================================="
echo ""
echo "Job started: $(date)"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Check dataset
echo "Checking dataset directory..."
if [ -d "dataset_shrunk_masks/images" ] && [ -d "dataset_shrunk_masks/masks" ]; then
    echo "✓ Dataset found: dataset_shrunk_masks"
    echo "  Images: $(ls dataset_shrunk_masks/images | wc -l) files"
    echo "  Masks:  $(ls dataset_shrunk_masks/masks | wc -l) files"
else
    echo "✗ ERROR: dataset_shrunk_masks directory not found!"
    echo "Please ensure dataset_shrunk_masks/images and dataset_shrunk_masks/masks exist"
    exit 1
fi
echo ""

# Check required files
echo "Checking required Python files..."
for file in hyperparam_search_comprehensive.py model_architectures.py loss_functions.py; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ ERROR: $file not found!"
        exit 1
    fi
done
echo ""

echo "=============================================================================="
echo "Search Configuration"
echo "=============================================================================="
echo "Search Type: Random (30 combinations)"
echo ""
echo "Search Space:"
echo "  Architectures: U-Net, ResU-Net, Attention ResU-Net"
echo "  Batch Sizes: 8, 16, 32"
echo "  Loss Functions: focal, combined, focal_tversky, combined_tversky"
echo "  Dropout: 0.3 (fixed)"
echo ""
echo "Fixed Parameters (from analysis):"
echo "  Learning Rate: 5e-5"
echo "  Early Stopping Patience: 30 epochs"
echo "  Image Size: 512×512"
echo "  Validation Split: 15%"
echo "=============================================================================="
echo ""

# Run hyperparameter search with random sampling (default: 30 combinations)
# For full grid search, use: --search-type grid
echo "Starting hyperparameter search..."
echo ""

singularity exec --nv $image python hyperparam_search_comprehensive.py \
    --search-type random \
    --n-random 30

EXIT_CODE=$?

echo ""
echo "=============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Search completed successfully!"
    echo ""

    # Find the latest results directory
    LATEST_DIR=$(ls -td hyperparam_comprehensive_* 2>/dev/null | head -1)

    if [ -n "$LATEST_DIR" ]; then
        echo "Results saved to: $LATEST_DIR"
        echo ""

        if [ -f "$LATEST_DIR/best_hyperparameters.json" ]; then
            echo "Best configuration:"
            cat "$LATEST_DIR/best_hyperparameters.json"
            echo ""
        fi

        if [ -f "$LATEST_DIR/search_results_final.csv" ]; then
            echo "Top 5 results:"
            head -6 "$LATEST_DIR/search_results_final.csv" | column -t -s,
        fi
    fi
else
    echo "Search failed with exit code: $EXIT_CODE"
    echo "Check the error output above for details."
fi
echo "=============================================================================="
echo "Job finished: $(date)"
echo ""

exit $EXIT_CODE
