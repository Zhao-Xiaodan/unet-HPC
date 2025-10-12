#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -k oed
#PBS -N Predict_Density_Analysis
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg

###############################################################################
# PBS Script for Prediction and Density Analysis
###############################################################################
# Purpose: Run inference on test images using trained U-Net, ResU-Net, and
#          Attention ResU-Net models, then calculate particle density using
#          CLAHE+OTSU method and generate comparison plots.
#
# Models: Best models from hyperparameter search (BS=8, combined_tversky)
#         - U-Net
#         - ResU-Net (best performer, 0.307 peak Jaccard)
#         - Attention ResU-Net
#
# Analysis:
#         - Predict on 512x512 tiles from test images
#         - Calculate density using CLAHE+OTSU (reference method)
#         - Calculate density from predicted masks
#         - Generate boxplots comparing all methods
#         - Export summary statistics
###############################################################################

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

echo "========================================================================"
echo "PREDICTION AND DENSITY ANALYSIS"
echo "========================================================================"
echo "Start time: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "========================================================================"

# TensorFlow environment variables (optimized for inference)
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

# GPU memory settings (conservative for inference)
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

# Memory optimization for inference
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC_SIZE_BYTES=268435456  # 256MB preallocate
export TF_CUDA_MALLOC_ASYNC_SUPPORTED_MAX_ALLOCATION_SIZE_BYTES=4294967296  # 4GB max

echo ""
echo "Environment Configuration:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  TF_GPU_ALLOCATOR: $TF_GPU_ALLOCATOR"
echo "  TF_FORCE_GPU_ALLOW_GROWTH: $TF_FORCE_GPU_ALLOW_GROWTH"
echo "========================================================================"

# Load required modules
echo ""
echo "Loading modules..."
module load singularity

# Check if singularity loaded
if ! command -v singularity &> /dev/null; then
    echo "ERROR: Singularity not loaded"
    exit 1
fi

echo "✓ Singularity loaded: $(singularity --version)"

# TensorFlow container
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

if [ ! -f "$image" ]; then
    echo "ERROR: TensorFlow container not found: $image"
    exit 1
fi

echo "✓ TensorFlow container: $image"
echo "========================================================================"

# Verify input files exist
echo ""
echo "Verifying input files..."

# Check models directory
if [ ! -d "./hyperparam_comprehensive_20251012_005054" ]; then
    echo "ERROR: Models directory not found: ./hyperparam_comprehensive_20251012_005054"
    exit 1
fi

# Check test images directory
if [ ! -d "./test_images" ]; then
    echo "ERROR: Test images directory not found: ./test_images"
    exit 1
fi

# Count test images
num_test_images=$(find ./test_images -type f \( -name "*.tif" -o -name "*.tiff" -o -name "*.png" \) | wc -l)
echo "✓ Test images directory: ./test_images"
echo "  Found $num_test_images test images"

# Check required Python scripts
required_files=(
    "predict_with_density_analysis.py"
    "model_architectures.py"
    "loss_functions.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "./$file" ]; then
        echo "ERROR: Required file not found: ./$file"
        exit 1
    fi
    echo "✓ Found: $file"
done

echo "========================================================================"

# Run prediction and density analysis
echo ""
echo "Starting prediction and density analysis..."
echo "Time: $(date)"
echo ""

singularity exec --nv $image python predict_with_density_analysis.py

exit_code=$?

echo ""
echo "========================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✓ PREDICTION AND ANALYSIS COMPLETED SUCCESSFULLY"
else
    echo "✗ PREDICTION AND ANALYSIS FAILED (exit code: $exit_code)"
fi
echo "End time: $(date)"
echo "========================================================================"

# Display output summary
echo ""
echo "Output Summary:"
echo "----------------"

# Find the latest prediction_analysis directory
latest_output=$(ls -dt prediction_analysis_* 2>/dev/null | head -1)

if [ -n "$latest_output" ] && [ -d "$latest_output" ]; then
    echo "Output directory: $latest_output"
    echo ""
    echo "Directory structure:"
    tree -L 2 "$latest_output" 2>/dev/null || find "$latest_output" -maxdepth 2 -type d

    echo ""
    echo "File counts:"
    echo "  Predicted masks (U-Net): $(find "$latest_output/predicted_masks/unet" -type f 2>/dev/null | wc -l)"
    echo "  Predicted masks (ResU-Net): $(find "$latest_output/predicted_masks/resunet" -type f 2>/dev/null | wc -l)"
    echo "  Predicted masks (Attention ResU-Net): $(find "$latest_output/predicted_masks/attention_resunet" -type f 2>/dev/null | wc -l)"
    echo "  Density boxplots: $(find "$latest_output/boxplots" -name "*.png" 2>/dev/null | wc -l)"
    echo "  Summary files: $(find "$latest_output/summary" -type f 2>/dev/null | wc -l)"

    # Display summary statistics if available
    if [ -f "$latest_output/summary/density_analysis_summary.csv" ]; then
        echo ""
        echo "Summary statistics preview:"
        head -20 "$latest_output/summary/density_analysis_summary.csv"
    fi
else
    echo "⚠ No output directory found"
fi

echo ""
echo "========================================================================"
echo "Job completed: $(date)"
echo "========================================================================"

exit $exit_code
