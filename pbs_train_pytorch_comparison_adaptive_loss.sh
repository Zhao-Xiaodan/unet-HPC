#!/bin/bash
#PBS -N PyTorch_AdaptiveLoss_Comparison
#PBS -l select=1:ncpus=8:mem=240gb:ngpus=1:gpu_model=a100,walltime=24:00:00
#PBS -j oe
#PBS -o ./pytorch_comparison_adaptive_loss_${PBS_JOBID}.log
#PBS -m abe
#PBS -M xiaodan@clemson.edu

# ==============================================================================
# PyTorch Training: Fair Comparison Using AdaptiveBGDiceLoss
# ==============================================================================
#
# This script trains 3 architectures (UNet, Attention UNet, Attention ResUNet)
# with the SAME complete setup as train.py:
# - Same preprocessing: grayscale, percentile normalization
# - Same augmentation: synthetic background artifacts
# - Same loss: AdaptiveBGDiceLoss (focal + TV + Tversky + bg-adaptive)
# - Grid search: 27 configs per architecture (81 total)
#
# This enables comparison of:
# 1. Architecture performance with advanced loss function
# 2. Impact of advanced loss vs simple BinaryFocalLoss
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
echo "Starting PyTorch Training (AdaptiveBGDiceLoss)"
echo "Training 3 architectures × 27 configs = 81 models"
echo "Loss: Focal + TV + Tversky + BG-Adaptive"
echo "Augmentation: 40% none, 30% old-style, 30% new-style with fading"
echo "========================================"
echo ""

singularity exec --nv $image python train_pytorch_comparison_adaptive_loss.py

# Capture exit status
EXIT_STATUS=$?

# Print completion information
echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "========================================"

exit $EXIT_STATUS
