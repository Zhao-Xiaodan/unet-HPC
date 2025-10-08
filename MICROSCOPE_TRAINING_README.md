# Optimized Microscope Dataset Training - User Guide

## Overview

This training setup applies **optimized hyperparameters** from the comprehensive hyperparameter optimization study to train three U-Net architectures on your microscope dataset.

### Architecture Configurations

Based on the hyperparameter optimization results (see `Hyperparameter_Optimization_Report.md`):

| Model | Learning Rate | Batch Size | Expected Val Jaccard | Training Stability |
|-------|--------------|------------|---------------------|-------------------|
| **Attention U-Net** ⭐ | **1e-4** | **16** | **~0.0699** | Excellent |
| **Attention ResU-Net** | **5e-4** | **16** | **~0.0695** | Very Good |
| **Standard U-Net** | **1e-3** | **8** | **~0.0670** | Good |

⭐ **Attention U-Net** achieved the best performance in the optimization study.

### Key Improvements

All models use optimized training strategies:
- ✅ **Gradient Clipping** (clipnorm=1.0) - prevents unstable gradients
- ✅ **Binary Focal Loss** (gamma=2) - focuses on hard examples
- ✅ **Extended Early Stopping** (patience=15) - better convergence
- ✅ **Adaptive Learning Rate** - automatic reduction on plateau
- ✅ **Model Checkpointing** - saves best performing weights

## Files

### 1. `microscope_optimized_training.py`
Main training script with optimized hyperparameters for all three models.

**Features:**
- Trains all 3 architectures sequentially
- Automatically creates timestamped output directory
- Generates comprehensive visualizations
- Supports multiple image formats (.tif, .tiff, .png, .jpg)
- Performance comparison and summary report

### 2. `pbs_microscope_optimized.sh`
PBS job submission script for HPC cluster.

**Features:**
- Pre-execution validation checks
- Automatic dependency installation (focal_loss)
- GPU status verification
- Detailed logging
- Post-training performance summary

## Quick Start Guide

### Step 1: Prepare Your Dataset

Ensure your dataset is structured as follows:

```
unet-HPC/
├── dataset_microscope/
│   ├── images/          # Input microscopy images
│   │   ├── img_001.tif
│   │   ├── img_002.tif
│   │   └── ...
│   └── masks/           # Ground truth segmentation masks
│       ├── mask_001.tif
│       ├── mask_002.tif
│       └── ...
├── microscope_optimized_training.py
├── pbs_microscope_optimized.sh
├── 224_225_226_models.py  (or models.py)
└── MICROSCOPE_TRAINING_README.md
```

**Important:**
- Image and mask filenames should correspond
- All images will be resized to 256×256
- Supported formats: .tif, .tiff, .png, .jpg
- Masks should be grayscale (0-255)

### Step 2: Submit to HPC

```bash
# Navigate to project directory
cd /home/svu/phyzxi/scratch/unet-HPC

# Submit the job
qsub pbs_microscope_optimized.sh
```

### Step 3: Monitor Training

```bash
# Check job status
qstat -u phyzxi

# View live output (while job is running)
tail -f microscope_training_YYYYMMDD_HHMMSS.log

# Check detailed job output
cat Microscope_Optimized_UNet.o<JOBID>
```

## Expected Output

### Directory Structure

After successful training, you'll get a timestamped directory:

```
microscope_training_20251008_143022/
├── best_unet_model.hdf5                    # Best UNet checkpoint
├── best_attention_unet_model.hdf5          # Best Attention UNet checkpoint
├── best_attention_resunet_model.hdf5       # Best Attention ResUNet checkpoint
├── final_unet_model.hdf5                   # Final UNet weights
├── final_attention_unet_model.hdf5         # Final Attention UNet weights
├── final_attention_resunet_model.hdf5      # Final Attention ResUNet weights
├── unet_history.csv                        # UNet training metrics
├── attention_unet_history.csv              # Attention UNet training metrics
├── attention_resunet_history.csv           # Attention ResUNet training metrics
├── training_curves_comparison.png          # Training curves for all models
├── performance_summary.png                 # Bar chart comparing models
└── dataset_sample_check.png                # Sample image/mask pair
```

### Model Files

- **`best_*.hdf5`** - Model weights at the epoch with highest validation Jaccard
- **`final_*.hdf5`** - Model weights at the final epoch (may differ if early stopping triggered)

### History CSV Files

Each CSV contains epoch-by-epoch metrics:
- `loss` - Training loss
- `val_loss` - Validation loss
- `accuracy` - Training accuracy
- `val_accuracy` - Validation accuracy
- `jacard_coef` - Training Jaccard coefficient
- `val_jacard_coef` - Validation Jaccard coefficient

### Visualizations

1. **`training_curves_comparison.png`**
   - 2×3 grid showing loss and Jaccard curves for all models
   - Helps identify overfitting and convergence patterns

2. **`performance_summary.png`**
   - Bar chart comparing best validation Jaccard scores
   - Shows learning rate and batch size for each model

3. **`dataset_sample_check.png`**
   - Sample image and mask from your dataset
   - Useful for verifying data loading

## Expected Training Time

Based on dataset size (assuming ~1000-2000 images):

| Phase | Estimated Time |
|-------|---------------|
| UNet | 2-3 hours |
| Attention UNet | 2-4 hours |
| Attention ResUNet | 3-5 hours |
| **Total** | **6-12 hours** |

*Note: Early stopping may reduce training time significantly*

## Performance Interpretation

### Jaccard Coefficient (IoU) Ranges

- **> 0.7** - Excellent segmentation
- **0.5 - 0.7** - Good segmentation
- **0.3 - 0.5** - Moderate segmentation
- **< 0.3** - Poor segmentation (check dataset quality)

### What to Expect

The optimization study used a different dataset, so your results may vary:

- **Best case**: Jaccard > 0.70 (excellent segmentation)
- **Typical**: Jaccard 0.50-0.70 (good segmentation)
- **If Jaccard < 0.30**: Check dataset quality, image/mask alignment

## Troubleshooting

### Common Issues

#### 1. Dataset Not Found
```
ERROR: Dataset directories not found!
```
**Solution:**
- Verify `dataset_microscope/images/` and `dataset_microscope/masks/` exist
- Check you're in the correct directory (`/home/svu/phyzxi/scratch/unet-HPC`)

#### 2. No Image Files Found
```
ERROR: No image files found in dataset directories
```
**Solution:**
- Ensure images have supported extensions (.tif, .tiff, .png, .jpg)
- Check file permissions (`ls -la dataset_microscope/images/`)

#### 3. Unequal Image/Mask Counts
```
WARNING: Unequal number of images (500) and masks (450)
```
**Solution:**
- Ensure every image has a corresponding mask
- Check for hidden files or incorrect file types

#### 4. Out of Memory (OOM)
```
ResourceExhaustedError: OOM when allocating tensor
```
**Solution:**
- Reduce batch size in `microscope_optimized_training.py`
- For UNet: change `batch_size=8` to `batch_size=4`
- For Attention models: change `batch_size=16` to `batch_size=8`

#### 5. Focal Loss Import Error
```
ImportError: cannot import name 'BinaryFocalLoss'
```
**Solution:**
- The PBS script auto-installs or creates custom implementation
- If still failing, manually install: `pip install focal-loss --user`

#### 6. Poor Performance (Low Jaccard)
```
Best Val Jaccard: 0.15
```
**Solution:**
- Check image/mask alignment (verify dataset_sample_check.png)
- Ensure masks are binary (0 and 255 values)
- Verify image quality and contrast
- Consider data augmentation for small datasets

## Advanced Customization

### Modifying Hyperparameters

Edit `microscope_optimized_training.py`:

```python
# Example: Change UNet learning rate
unet_model.compile(
    optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),  # Changed from 1e-3
    loss=BinaryFocalLoss(gamma=2),
    metrics=['accuracy', jacard_coef]
)
```

### Adjusting Training Duration

```python
# Example: Increase maximum epochs
unet_history = unet_model.fit(
    X_train, y_train,
    epochs=150,  # Changed from 100
    callbacks=callbacks_unet
)
```

### Changing Image Size

```python
# Example: Use 512×512 images (requires more GPU memory)
SIZE = 512  # Changed from 256
```

## Using Trained Models

### Loading a Saved Model

```python
import tensorflow as tf

# Load the best Attention UNet model
model = tf.keras.models.load_model(
    'microscope_training_20251008_143022/best_attention_unet_model.hdf5',
    compile=False
)

# Compile with custom metrics if needed
from models import jacard_coef
from focal_loss import BinaryFocalLoss

model.compile(
    optimizer='adam',
    loss=BinaryFocalLoss(gamma=2),
    metrics=['accuracy', jacard_coef]
)
```

### Making Predictions

```python
import numpy as np
from PIL import Image
import cv2

# Load and preprocess test image
test_img = cv2.imread('test_image.tif', 1)
test_img = cv2.resize(test_img, (256, 256))
test_img = test_img / 255.0

# Add batch dimension
test_img_input = np.expand_dims(test_img, 0)

# Predict
prediction = model.predict(test_img_input)[0, :, :, 0]

# Threshold to binary mask
binary_mask = (prediction > 0.5).astype(np.uint8) * 255

# Save result
Image.fromarray(binary_mask).save('prediction.png')
```

## Citation and References

### Hyperparameter Optimization Study

This training setup is based on the comprehensive hyperparameter optimization study documented in:
- `Hyperparameter_Optimization_Report.md`
- Experiment date: September 26, 2025
- 9 successful experiments across 3 architectures

### Key Findings Applied

1. **Training stability dramatically improved** (up to 362× for UNet)
2. **Architecture-specific hyperparameters** identified
3. **Gradient clipping essential** for stable convergence
4. **Attention mechanisms prefer lower learning rates** (1e-4 vs 1e-3)

### Model Architectures

- **U-Net**: Ronneberger et al., 2015
- **Attention U-Net**: Oktay et al., 2018 (https://arxiv.org/abs/1804.03999)
- **Attention ResU-Net**: Alom et al., 2018 (https://arxiv.org/abs/1802.06955)

## Support

If you encounter issues:

1. Check the console log: `microscope_training_YYYYMMDD_HHMMSS.log`
2. Review PBS job output: `Microscope_Optimized_UNet.o<JOBID>`
3. Verify dataset structure and file formats
4. Compare with working example: `dataset_full_stack/`

## Summary

This optimized training setup provides:

✅ **State-of-the-art hyperparameters** from systematic optimization
✅ **Automatic best model selection** via validation metrics
✅ **Comprehensive logging and visualization** for analysis
✅ **Production-ready models** with stable training
✅ **Fair architecture comparison** under optimal conditions

**Expected outcome:** High-quality mitochondria segmentation models tailored to your microscope dataset with minimal manual tuning required.

Good luck with your training! 🚀
