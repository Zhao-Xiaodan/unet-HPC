# Claude Code Project Configuration

## Environment Setup

### Python Environment
**IMPORTANT**: Always activate the conda environment before running Python scripts:
```bash
conda activate unetCNN
```

This environment contains the required packages for:
- pandas
- matplotlib
- numpy
- tensorflow
- keras
- opencv (cv2)
- PIL
- sklearn
- focal_loss

### Project Structure
This is a mitochondria segmentation project comparing U-Net architectures:
- Standard U-Net
- Attention U-Net
- Attention Residual U-Net

### Key Files
- `224_225_226_mito_segm_using_various_unet_models.py` - Main training script
- `224_225_226_models.py` - Model definitions
- `pbs_unet.sh` - HPC job submission script
- `mitochondria_segmentation_20250925_133928/` - Training results directory

### Analysis Scripts
- `analyze_unet_comparison.py` - Performance analysis and visualization

### PBS document guidence
follow workable pbs_unet.sh as an example, like the following but not limited to 

#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -k oed
#PBS -N project name
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=240gb
#PBS -M phyzxi@nus.edu.sg

# IMPORTANT: Change to working directory FIRST
cd /home/svu/phyzxi/scratch/unet-HPC

# TensorFlow environment variables
export TF_CPP_MIN_LOG_LEVEL=1
export TF_ENABLE_ONEDNN_OPTS=1
export CUDA_VISIBLE_DEVICES=0

# Load required modules
module load singularity

# Use the modern TensorFlow container
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif


## Notes
- Dataset should be in `dataset_full_stack/images/` and `dataset_full_stack/masks/` directories
- Models use Binary Focal Loss with Jaccard coefficient metrics
- Training results include .hdf5 model files and .csv history files