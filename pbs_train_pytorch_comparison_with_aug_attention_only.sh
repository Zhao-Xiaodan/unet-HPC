#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed
#PBS -N PyTorch_WithAug_AttentionOnly
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe

# ==============================================================================
# PyTorch Training: Attention Architectures Only (With Augmentation)
# ==============================================================================
#
# This script trains ONLY Attention-based architectures:
# - Attention UNet
# - Attention ResUNet
#
# UNet training is SKIPPED (already completed in previous job)
#
# Settings:
# - WITH augmentation (40% none, 30% old-style, 30% new-style with fading)
# - BinaryFocalLoss
# - Same preprocessing as train.py (grayscale, percentile normalization)
# - Grid search: 27 configs per architecture (54 total)
#
# Expected runtime: ~12-16 hours for 54 models
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
echo "Starting PyTorch Training (Attention Architectures Only)"
echo "Training 2 architectures × 27 configs = 54 models"
echo "Loss: BinaryFocalLoss"
echo "Augmentation: 40% none, 30% old-style, 30% new-style with fading"
echo "Skipping: UNet (already trained)"
echo "========================================"
echo ""

# Create a wrapper script to modify CONFIG and run training
cat > run_attention_only_with_aug.py << 'WRAPPER_EOF'
import sys
import train_pytorch_comparison_with_aug as training_script

# Override CONFIG to skip UNet
training_script.CONFIG['architectures'] = ['attention_unet', 'attention_resunet']

# Run main
if __name__ == "__main__":
    training_script.main()
WRAPPER_EOF

singularity exec --nv $image python run_attention_only_with_aug.py

# Capture exit status
EXIT_STATUS=$?

# Clean up wrapper script
rm -f run_attention_only_with_aug.py

# Print completion information
echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "========================================"

exit $EXIT_STATUS
