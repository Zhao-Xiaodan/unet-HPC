#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N PyTorch_WithAug_Comparison
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# PyTorch Training: Fair Comparison with Keras (WITH Augmentation)
# ==============================================================================
#
# This script trains 3 architectures (UNet, Attention UNet, Attention ResUNet)
# with the SAME preprocessing as train.py and using BinaryFocalLoss (like Keras)
# but WITH augmentation to test if augmentation improves performance
#
# Key features:
# - Same preprocessing: grayscale, percentile normalization
# - Same loss: BinaryFocalLoss (gamma=2, alpha=0.25)
# - WITH augmentation (synthetic background artifacts - same as train.py)
# - Grid search: 27 configs per architecture (81 total)
#
# Expected runtime: ~18-24 hours for all 81 models
# ==============================================================================

# Load required modules
module load singularity

# Define singularity container
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

# Check GPU availability
echo "GPU Information:"
singularity exec --nv $image nvidia-smi
echo ""

# Print Python and PyTorch versions
echo "Python version:"
singularity exec --nv $image python --version
echo ""

echo "PyTorch version:"
singularity exec --nv $image python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run training
echo "========================================"
echo "Starting PyTorch Training (WITH Augmentation)"
echo "Training 3 architectures × 27 configs = 81 models"
echo "Augmentation: 40% none, 30% old-style, 30% new-style with fading"
echo "========================================"
echo ""

singularity exec --nv $image python train_pytorch_comparison_with_aug.py

# Capture exit status
EXIT_STATUS=$?

# Print completion information
echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "========================================"

exit $EXIT_STATUS
