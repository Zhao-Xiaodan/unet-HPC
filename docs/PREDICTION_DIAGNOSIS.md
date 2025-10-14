# Prediction Issues Diagnosis and Solutions

## Analysis Date: 2025-10-12
## Results Directory: `prediction_analysis_20251012_074415`

---

## 🔍 Issues Identified

### Issue 1: ResU-Net Predicting 100% White (Complete Failure)

**Evidence:**
```
ResU-Net mean density: 1.0000 ± 0.0000 (100% foreground across ALL images)
```

**Problem:** Model is predicting every single pixel as particle (foreground), resulting in completely white masks.

**Root Cause:**
- Model weights may not have loaded correctly
- Model architecture mismatch between training and inference
- Output activation may be stuck or incorrectly configured
- Model checkpoint may be corrupted

---

### Issue 2: U-Net Under-Segmenting Severely

**Evidence:**
```
U-Net mean density: 0.0021 ± 0.0011 (0.21% foreground)
CLAHE+OTSU reference: 0.3432 ± 0.1926 (34.32% foreground)
```

**Problem:** Model is predicting almost no particles, with only 0.6% of the expected density.

**Comparison:**
- 10x dilution: U-Net = 0.08%, CLAHE = 64.8% (805× underestimation)
- 80x dilution: U-Net = 0.38%, CLAHE = 48.2% (127× underestimation)

---

### Issue 3: Attention ResU-Net Also Under-Segmenting

**Evidence:**
```
Attention ResU-Net mean density: 0.0041 ± 0.0036 (0.41% foreground)
```

**Problem:** Better than U-Net but still predicting only 1.2% of expected density.

---

## 📊 Expected vs Actual Densities

| Dilution | CLAHE+OTSU (Expected) | U-Net | ResU-Net | Attention ResU-Net |
|----------|-----------------------|-------|----------|-------------------|
| 10x      | 64.8%                 | 0.08% | 100.0%   | 1.42%             |
| 20x      | 55.1%                 | 0.29% | 100.0%   | 0.41%             |
| 80x      | 48.2%                 | 0.38% | 100.0%   | 0.31%             |
| 160x     | 33.3%                 | 0.22% | 100.0%   | 0.31%             |
| 320x     | 19.1%                 | 0.13% | 100.0%   | 0.34%             |

**Conclusion:** All three deep learning models are producing incorrect predictions.

---

## 🔬 Root Cause Analysis

### Hypothesis 1: Model Checkpoint Files Missing or Corrupted

**Check:**
```bash
ls -lh hyperparam_comprehensive_20251012_005054/*.hdf5
```

**Expected:** Model files should exist (>100MB each for these architectures)

**If missing:** Models were never saved during training (checkpoint callback may have failed)

---

### Hypothesis 2: Wrong Models Loaded

**Investigation needed:**
```python
# Check which models were actually loaded
# Look at prediction script output for model file names
```

**Potential issue:** Script may have loaded models from wrong training run or with different hyperparameters than expected.

---

### Hypothesis 3: Input Normalization Mismatch

**Training:** Images normalized to [0, 1] range
**Inference:** Images may have been normalized differently

**Evidence:** The script uses:
```python
img_normalized = img.astype(np.float32) / 255.0
```

This should be correct, but need to verify training script used same normalization.

---

### Hypothesis 4: Threshold Issue

**Current threshold:** 0.5 (hardcoded)
```python
pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255
```

**Problem:** If models output continuous values with different ranges, fixed threshold may be inappropriate.

---

### Hypothesis 5: Model Output Activation Issue

**Expected:** Sigmoid activation for binary segmentation (outputs [0, 1])

**Possible issue:**
- ResU-Net outputting values > 0.5 for all pixels (always above threshold)
- U-Net/Attention ResU-Net outputting values < 0.5 for all pixels (always below threshold)

---

## ✅ Solutions

### Solution 1: Verify Model Files Exist

```bash
cd hyperparam_comprehensive_20251012_005054

# Check for .hdf5 or .h5 files
find . -name "*.hdf5" -o -name "*.h5" -exec ls -lh {} \;

# Expected output:
# model_unet_bs8_dr0.3_combined_tversky.hdf5 (~300MB)
# model_resunet_bs8_dr0.3_combined_tversky.hdf5 (~350MB)
# model_attention_resunet_bs8_dr0.3_focal_tversky.hdf5 (~370MB)
```

**If files don't exist:**
→ Need to retrain models with ModelCheckpoint callback enabled

---

### Solution 2: Check Model Predictions Before Thresholding

Create diagnostic script to inspect raw model outputs:

```python
# diagnose_model_outputs.py
import numpy as np
import cv2
from tensorflow import keras
from model_architectures import get_model
from loss_functions import jacard_coef, dice_coef, get_loss_function

# Load test image
img = cv2.imread('test_images/10x_2025-05-15_02-05-00.tif', cv2.IMREAD_GRAYSCALE)
img_norm = img.astype(np.float32) / 255.0

# Extract one tile
tile = img_norm[0:512, 0:512]
tile_input = tile[np.newaxis, :, :, np.newaxis]

# Load each model and check raw outputs
custom_objects = {
    'jacard_coef': jacard_coef,
    'dice_coef': dice_coef,
    'combined_tversky_focal_loss': get_loss_function('combined_tversky')
}

for arch in ['unet', 'resunet', 'attention_resunet']:
    model_path = f'hyperparam_comprehensive_20251012_005054/model_{arch}_bs8_dr0.3_*.hdf5'

    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)

        # Get raw prediction
        pred = model.predict(tile_input, verbose=0)

        print(f"\n{arch.upper()}")
        print(f"  Output shape: {pred.shape}")
        print(f"  Output range: [{pred.min():.6f}, {pred.max():.6f}]")
        print(f"  Output mean: {pred.mean():.6f}")
        print(f"  Output std: {pred.std():.6f}")
        print(f"  % pixels > 0.5: {(pred > 0.5).sum() / pred.size * 100:.2f}%")
        print(f"  % pixels > 0.3: {(pred > 0.3).sum() / pred.size * 100:.2f}%")
        print(f"  % pixels > 0.7: {(pred > 0.7).sum() / pred.size * 100:.2f}%")

    except Exception as e:
        print(f"{arch}: Error - {e}")
```

**Run this to diagnose the actual model outputs.**

---

### Solution 3: Adaptive Thresholding

If fixed threshold 0.5 is inappropriate, use Otsu's method on predictions:

```python
def predict_with_adaptive_threshold(model, tile):
    """Predict with adaptive thresholding"""
    # Get raw prediction
    pred = model.predict(tile_input, verbose=0)
    pred_array = (pred[0, :, :, 0] * 255).astype(np.uint8)

    # Apply Otsu threshold
    threshold, binary = cv2.threshold(pred_array, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(f"  Otsu threshold: {threshold/255:.3f}")
    return binary, threshold/255
```

---

### Solution 4: Retrain Models with Proper Callbacks

If models are missing or corrupted, retrain:

```python
# train_best_models.py
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

callbacks = [
    ModelCheckpoint(
        'model_best_{arch}.hdf5',
        monitor='val_jacard_coef',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False  # Save complete model
    ),
    EarlyStopping(
        monitor='val_jacard_coef',
        patience=12,  # Reduced from 30
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
]
```

---

### Solution 5: Use Training Script's Best Models

The hyperparameter search may have different models than expected. Check training log:

```bash
# Find training log
ls hyperparam_comprehensive_20251012_005054/*.log

# Check which models actually performed best
grep -A 5 "best_val_jacard" hyperparam_comprehensive_20251012_005054/*.log
```

---

## 🚀 Immediate Action Items

### 1. Check if Model Files Exist (Priority: CRITICAL)

```bash
cd /Users/xiaodan/unetCNN/unet-HPC
find hyperparam_comprehensive_20251012_005054 -name "*.hdf5" -ls
```

### 2. Create and Run Diagnostic Script (Priority: HIGH)

Save the diagnostic script above and run:
```bash
python diagnose_model_outputs.py
```

This will reveal:
- Whether models are loading correctly
- What range of values they're outputting
- Whether 0.5 threshold is appropriate

### 3. If Models Missing: Re-run Training (Priority: HIGH)

```bash
# Edit hyperparam_search_comprehensive.py to ensure ModelCheckpoint saves models
# Then re-run training
qsub pbs_hyperparam_comprehensive.sh
```

### 4. If Models Exist: Fix Prediction Script (Priority: MEDIUM)

Options:
a) Use adaptive thresholding instead of fixed 0.5
b) Normalize model outputs before thresholding
c) Use percentile-based thresholding

---

## 📈 Expected Outcomes After Fixes

After applying fixes, expected density ranges:

| Dilution | CLAHE+OTSU | Expected DL Model Range |
|----------|------------|------------------------|
| 10x      | 64.8%      | 50-70% (reasonable)     |
| 20x      | 55.1%      | 40-60%                  |
| 80x      | 48.2%      | 35-55%                  |
| 160x     | 33.3%      | 25-40%                  |
| 320x     | 19.1%      | 15-25%                  |

**Correlation with CLAHE+OTSU:** Should be r > 0.8 if models are working correctly.

---

## 📝 Files to Create

1. **`diagnose_model_outputs.py`** - Check raw model predictions
2. **`fix_prediction_threshold.py`** - Updated prediction script with adaptive thresholding
3. **`verify_model_checkpoints.sh`** - Script to verify all model files exist and are valid

---

## 🔗 References

- Original hyperparameter search: `hyperparam_comprehensive_20251012_005054/`
- Search results: `hyperparam_comprehensive_20251012_005054/search_results_final.csv`
- Prediction script: `predict_with_density_analysis.py`
- Analysis results: `density_analysis_dilution_factors/`

---

**Status:** Issues identified, diagnostic steps provided
**Next Steps:** Run diagnostic script to pinpoint exact cause
**ETA for fix:** 1-2 hours once root cause confirmed
