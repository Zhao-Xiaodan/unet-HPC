#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_model=a100
#PBS -l walltime=06:00:00
#PBS -q gpu
#PBS -N mitochondria_original_broken
#PBS -o mitochondria_original_broken.o
#PBS -e mitochondria_original_broken.e

# Job description: Test original broken Jaccard implementation with full dataset

# Load required modules
module load python/3.11.5
module load cuda/12.3.0
module load cudnn/8.9.7.29-12.3

# Set working directory
cd $PBS_O_WORKDIR

# Print job information
echo "=========================================="
echo "🚨 ORIGINAL BROKEN IMPLEMENTATION TEST"
echo "=========================================="
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $PWD"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

# Check GPU availability
nvidia-smi
echo ""

# Setup Python environment
echo "Setting up environment..."
source /home/users/nus/e1351829/miniconda3/etc/profile.d/conda.sh
conda activate unetCNN

# Print environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Available packages:"
pip list | grep -E "(tensorflow|keras|numpy|pandas|opencv|pillow)"
echo ""

# Verify dataset
echo "🔍 Verifying dataset..."
if [ ! -d "dataset_full_stack" ]; then
    echo "❌ ERROR: dataset_full_stack directory not found!"
    exit 1
fi

echo "Dataset structure:"
ls -la dataset_full_stack/
echo "Images count: $(find dataset_full_stack/images -name "*.tif" | wc -l)"
echo "Masks count: $(find dataset_full_stack/masks -name "*.tif" | wc -l)"
echo ""

# Verify focal loss installation
echo "🔍 Checking focal_loss package..."
python -c "from focal_loss import BinaryFocalLoss; print('✅ focal_loss package available')" || {
    echo "⚠️  Installing focal_loss..."
    pip install focal-loss
}
echo ""

# Run the original broken implementation
echo "🚀 Starting original broken implementation training..."
echo "⚠️  WARNING: This uses the BROKEN Jaccard coefficient implementation!"
echo "📊 Dataset: Full dataset_full_stack (1,980 patches)"
echo "🎯 Hyperparameters: LR=1e-2, Batch=8, Epochs=50"
echo "🏗️  Architectures: UNet, Attention UNet, Attention ResUNet"
echo ""

# Set memory growth for GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run the training
python 224_225_226_mito_segm_using_various_unet_models_original.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"

    # Find the output directory
    output_dir=$(find . -maxdepth 1 -name "mitochondria_segmentation_original_*" -type d | head -1)

    if [ -d "$output_dir" ]; then
        echo "📁 Results saved to: $output_dir"
        echo ""
        echo "📋 Generated files:"
        ls -la "$output_dir"
        echo ""

        # Display summary results if available
        if [ -f "$output_dir/original_results_summary.json" ]; then
            echo "📊 Results Summary:"
            cat "$output_dir/original_results_summary.json"
        fi

        # Display key metrics from history files
        echo ""
        echo "🎯 Training History Summary:"
        if [ -f "$output_dir/unet_history_df_original.csv" ]; then
            echo "UNet validation Jaccard (last 5 epochs):"
            tail -5 "$output_dir/unet_history_df_original.csv" | cut -d',' -f5
        fi

        if [ -f "$output_dir/att_unet_history_df_original.csv" ]; then
            echo "Attention UNet validation Jaccard (last 5 epochs):"
            tail -5 "$output_dir/att_unet_history_df_original.csv" | cut -d',' -f5
        fi

        if [ -f "$output_dir/custom_code_att_res_unet_history_df_original.csv" ]; then
            echo "Attention ResUNet validation Jaccard (last 5 epochs):"
            tail -5 "$output_dir/custom_code_att_res_unet_history_df_original.csv" | cut -d',' -f5
        fi
    else
        echo "⚠️  Warning: Output directory not found!"
    fi
else
    echo "❌ Training failed with exit code: $?"
    echo "Check error logs for details."
fi

echo ""
echo "=========================================="
echo "🚨 ORIGINAL BROKEN IMPLEMENTATION COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo "Job finished."