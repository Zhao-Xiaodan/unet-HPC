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

## Notes
- Dataset should be in `dataset/images/` and `dataset/masks/` directories
- Models use Binary Focal Loss with Jaccard coefficient metrics
- Training results include .hdf5 model files and .csv history files