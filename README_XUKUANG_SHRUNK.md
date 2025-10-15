# Training Shrunk Masks with Xukuang's Parameters

## Overview

This training setup uses **Xukuang's parameters from `bead_seg.ipynb`** to train 3 U-Net architectures on the `dataset_shrunk_masks` dataset.

The training replicates the exact hyperparameters, loss function, and training configuration used in the original notebook.

## Models

Three U-Net architectures will be trained:

1. **Standard U-Net**
2. **Attention U-Net**
3. **Attention Residual U-Net**

## Xukuang's Training Parameters

The following parameters are taken directly from `bead_seg.ipynb`:

| Parameter | Value | Source |
|-----------|-------|--------|
| **Learning Rate** | `5e-3` | `lr = 5e-3` |
| **Epochs** | `200` | `epochs = 200` |
| **Batch Size** | `4` | `batch_size = 4` |
| **Image Size** | `512×512` | `SIZE = 512` |
| **Loss Function** | `BinaryFocalLoss(gamma=2)` | `BinaryFocalLoss(gamma=2)` |
| **Optimizer** | `Adam` | `Adam(lr = lr)` |
| **Test Split** | `20%` | `test_size = 0.2` |
| **Random State** | `0` | `random_state = 0` |
| **Shuffle** | `False` | `shuffle=False` |

## Dataset

**Source:** `dataset_shrunk_masks/`

```
dataset_shrunk_masks/
├── images/  (98 images)
└── masks/   (98 masks)
```

**Preprocessing:**
- Images resized to 512×512
- Normalization: `/255.0`
- RGB images (3 channels)
- Binary masks (1 channel)

**Split:**
- Training: 78 samples (80%)
- Testing: 20 samples (20%)

## Files

- **`train_shrunk_xukuang_parameters.py`** - Main Python training script
- **`pbs_train_shrunk_xukuang_parameters.sh`** - PBS submission script
- **`README_XUKUANG_SHRUNK.md`** - This file

## Requirements

### On HPC

1. **Dataset** must exist:
   ```bash
   dataset_shrunk_masks/
   ├── images/*.png
   └── masks/*.png
   ```

2. **Dependencies:**
   - TensorFlow 2.16.1
   - models.py (UNet, Attention_UNet, Attention_ResUNet definitions)
   - focal_loss.py (BinaryFocalLoss implementation)
   - Standard libs: numpy, pandas, matplotlib, opencv, PIL, sklearn

## Usage

### On HPC

```bash
cd /home/svu/phyzxi/scratch/unet-HPC

# Verify dataset exists
ls dataset_shrunk_masks/images/ | wc -l
ls dataset_shrunk_masks/masks/ | wc -l

# Verify required scripts exist
ls train_shrunk_xukuang_parameters.py
ls models.py
ls focal_loss.py

# Submit job
qsub pbs_train_shrunk_xukuang_parameters.sh

# Monitor job
qstat -u phyzxi

# Check output (after completion)
ls xukuang_params_shrunk_*/
```

## Expected Output

The script will create a timestamped directory:

```
xukuang_params_shrunk_YYYYMMDD_HHMMSS/
├── EXPERIMENT_INFO.json                          # Training metadata
├── TRAINING_SUMMARY.json                         # Final metrics summary
│
├── unet_xukuang_params_shrunk.keras              # U-Net model
├── unet_history.csv                              # U-Net training history
│
├── attention_unet_xukuang_params_shrunk.keras    # Attention U-Net model
├── attention_unet_history.csv                    # Attention U-Net history
│
├── attention_resunet_xukuang_params_shrunk.keras # Attention ResUNet model
├── attention_resunet_history.csv                 # Attention ResUNet history
│
├── train_shrunk_xukuang_parameters.py            # Source script (archived)
├── pbs_train_shrunk_xukuang_parameters.sh        # PBS script (archived)
├── Xukuang_Shrunk.o######                        # PBS output log
└── train_shrunk_console_*.log                    # Python console log
```

## Training Process

### Phase 1: Dataset Loading
- Load 98 images from `dataset_shrunk_masks/images/`
- Load 98 masks from `dataset_shrunk_masks/masks/`
- Resize all to 512×512
- Normalize to [0, 1]

### Phase 2: Train-Test Split
- Split into 78 training + 20 testing samples
- Random state = 0 for reproducibility
- No shuffling during training (matches notebook)

### Phase 3: Train U-Net
- 200 epochs
- Batch size: 4
- Loss: Binary Focal Loss (gamma=2)
- Optimizer: Adam (lr=5e-3)
- Metrics: accuracy, jaccard_coef

### Phase 4: Train Attention U-Net
- Same parameters as U-Net
- More complex architecture (attention gates)

### Phase 5: Train Attention ResUNet
- Same parameters as U-Net
- Most complex architecture (residual + attention)

## Expected Runtime

- **U-Net:** ~4-6 hours
- **Attention U-Net:** ~4-6 hours
- **Attention ResUNet:** ~4-6 hours
- **Total:** ~12-18 hours (depends on GPU)

**Walltime Requested:** 24 hours (conservative)

## Monitoring Progress

### Check Job Status
```bash
qstat -u phyzxi
```

### View Real-Time Log
```bash
tail -f train_shrunk_console_*.log
```

### View PBS Output
```bash
tail -f Xukuang_Shrunk.o*
```

## Post-Training Analysis

### 1. View Training Summary
```bash
cat xukuang_params_shrunk_*/TRAINING_SUMMARY.json
```

Example output:
```json
{
    "execution_times": {
        "unet": "5:23:15",
        "attention_unet": "5:45:30",
        "attention_resunet": "6:12:45"
    },
    "final_validation_metrics": {
        "unet": {
            "loss": 0.0234,
            "accuracy": 0.9876,
            "jaccard": 0.7654
        },
        ...
    }
}
```

### 2. Plot Training Curves

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load history
unet_history = pd.read_csv('xukuang_params_shrunk_*/unet_history.csv')

# Plot Jaccard coefficient
plt.plot(unet_history['jacard_coef'], label='Train')
plt.plot(unet_history['val_jacard_coef'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Jaccard Coefficient')
plt.legend()
plt.show()
```

### 3. Compare Models

Load all three histories and plot side-by-side comparisons:

```python
models = ['unet', 'attention_unet', 'attention_resunet']
for model in models:
    df = pd.read_csv(f'xukuang_params_shrunk_*/{model}_history.csv')
    plt.plot(df['val_jacard_coef'], label=model)
plt.legend()
plt.show()
```

## Comparison with Original Notebook

### Similarities
- ✓ Same learning rate (5e-3)
- ✓ Same epochs (200)
- ✓ Same batch size (4)
- ✓ Same image size (512×512)
- ✓ Same loss function (BinaryFocalLoss, gamma=2)
- ✓ Same optimizer (Adam)
- ✓ Same test split (20%)
- ✓ Same random state (0)
- ✓ Same shuffle setting (False)

### Differences
| Aspect | Original (bead_seg.ipynb) | This Training |
|--------|---------------------------|---------------|
| Dataset | bead_data/ | dataset_shrunk_masks/ |
| # Images | Unknown | 98 images |
| Domain | Bead segmentation | Mitochondria segmentation |
| Environment | Jupyter Notebook | HPC (PBS) |

## Troubleshooting

### Dataset Not Found

**Error:** `dataset_shrunk_masks directory not found`

**Solution:**
```bash
# Check if dataset exists
ls dataset_shrunk_masks/

# If missing, check correct path
pwd
# Should be: /home/svu/phyzxi/scratch/unet-HPC
```

### Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'focal_loss'`

**Solution:**
```bash
# Check if focal_loss.py exists
ls focal_loss.py

# Check if models.py exists
ls models.py
```

### GPU Out of Memory

**Error:** `ResourceExhaustedError: OOM when allocating tensor`

**Solution:**

In `train_shrunk_xukuang_parameters.py`, reduce batch size:
```python
BATCH_SIZE = 2  # Reduce from 4
```

### Low Jaccard Scores

**Symptom:** Final Jaccard < 0.1

**Possible Causes:**
1. Dataset quality issues (misaligned images/masks)
2. Image preprocessing mismatch
3. Training instability

**Diagnostic Steps:**
```python
# Check dataset alignment
from PIL import Image
import numpy as np

img = np.array(Image.open('dataset_shrunk_masks/images/example.png'))
mask = np.array(Image.open('dataset_shrunk_masks/masks/example.png'))

print(f"Image shape: {img.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Image range: [{img.min()}, {img.max()}]")
print(f"Mask range: [{mask.min()}, {mask.max()}]")
```

## Next Steps After Training

1. **Evaluate on Test Set:**
   - Load trained models
   - Predict on X_test
   - Calculate metrics (IoU, Dice, etc.)

2. **Visual Inspection:**
   - Plot predictions vs. ground truth
   - Identify failure cases

3. **Compare with Other Methods:**
   - Compare with hyperparameter search results
   - Compare with original 256×256 models
   - Compare with 512×512 grayscale models

4. **Fine-Tuning (if needed):**
   - Adjust learning rate
   - Try different loss functions
   - Experiment with data augmentation

## Citation

If using these results, reference:
- Original notebook: `bead_seg.ipynb` (Dr. Sreenivas Bhattiprolu)
- This training: `xukuang_params_shrunk_YYYYMMDD_HHMMSS/EXPERIMENT_INFO.json`

---

**Created:** October 15, 2025
**Author:** Claude Code
**Purpose:** Train shrunk masks dataset using Xukuang's parameters from bead_seg.ipynb
