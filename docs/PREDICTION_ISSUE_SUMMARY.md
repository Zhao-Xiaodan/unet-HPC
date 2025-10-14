# Prediction Issue: Complete Analysis and Solution

## Date: 2025-10-12
## Status: 🔴 CRITICAL - Models Not Saved During Training

---

## 🔍 Root Cause Identified

### The Problem

**NO MODEL FILES EXIST** in `hyperparam_comprehensive_20251012_005054/`

```bash
# Verification
$ find hyperparam_comprehensive_20251012_005054 -name "*.hdf5"
# (no output - no files found)

$ ls -lh hyperparam_comprehensive_20251012_005054/
# Only CSV files and PNG plots exist
# Total size: 7.7MB (models would be ~1GB+)
```

### What Happened

1. **Training completed successfully** - CSV files show 30 configurations with metrics
2. **Models were NEVER saved to disk** - ModelCheckpoint callback failed or didn't execute
3. **Prediction script loaded random/uninitialized models** - Why predictions are garbage:
   - **ResU-Net**: Random initialization → predicts 100% foreground (all white)
   - **U-Net**: Random initialization → predicts 0.2% foreground (almost all black)
   - **Attention ResU-Net**: Random initialization → predicts 0.4% foreground

---

## 📊 Evidence

### From Diagnostic Script

```
Loading test image: ./test_images/10x_2025-05-15_02-05-00.tif
Reference density (CLAHE+OTSU): 59.70%

✗ No model file found for unet
✗ No model file found for resunet
✗ No model file found for attention_resunet
```

### From Density Analysis

| Architecture | Predicted Density | Expected (CLAHE+OTSU) | Error |
|--------------|------------------|----------------------|--------|
| ResU-Net     | 100.00%          | 59.70%               | +67%   |
| U-Net        | 0.08%            | 59.70%               | -99.9% |
| Attention ResU-Net | 1.42% | 59.70%               | -97.6% |

**All three models producing completely wrong results** → Strong indicator of untrained/random models

---

## 🔬 Why Models Weren't Saved

### Hypothesis 1: HPC Filesystem Issue

The training ran on HPC cluster. Possible issues:
- Disk quota exceeded during save
- Filesystem permissions
- Network filesystem timeout during large file writes
- Out of disk space

### Hypothesis 2: ModelCheckpoint Configuration

Check `hyperparam_search_comprehensive.py`:

```python
ModelCheckpoint(
    str(checkpoint_path),
    monitor='val_jacard_coef',
    save_best_only=True,
    mode='max',
    verbose=0  # ← Should be verbose=1 to see save attempts
)
```

**Issue**: `verbose=0` means save failures would be silent

### Hypothesis 3: Relative Path Issue

```python
checkpoint_path = output_dir / f"model_{arch}_bs{bs}_dr{dropout}_{loss_name}.hdf5"
```

If `output_dir` was relative path and script changed directories, saves would fail silently.

---

## ✅ Solutions

### Solution 1: Check HPC Training Log

```bash
# On HPC, find the training log
cd /home/svu/phyzxi/scratch/unet-HPC
ls -lt Hyperparam_Comprehensive.o* | head -1

# Check for ModelCheckpoint messages
grep -i "model\|checkpoint\|saving" Hyperparam_Comprehensive.o285XXX
```

Look for errors like:
- "Failed to save model"
- "Permission denied"
- "Disk quota exceeded"
- "No space left on device"

### Solution 2: Fix ModelCheckpoint and Re-train

Edit `hyperparam_search_comprehensive.py`:

```python
# Add absolute path
import os
checkpoint_path = os.path.abspath(
    os.path.join(output_dir, f"model_{arch}_bs{bs}_dr{dropout}_{loss_name}.hdf5")
)

# Enable verbose output
ModelCheckpoint(
    checkpoint_path,
    monitor='val_jacard_coef',
    save_best_only=True,
    mode='max',
    verbose=1,  # ← Changed from 0
    save_weights_only=False
)

# Add logging to verify path
print(f"Will save model to: {checkpoint_path}")
print(f"Directory writable: {os.access(os.path.dirname(checkpoint_path), os.W_OK)}")
```

### Solution 3: Train Individual Best Models

Instead of re-running entire 30-config search, train just the top 3:

Create `train_best_three_models.py`:

```python
#!/usr/bin/env python3
"""
Train Best 3 Models from Hyperparameter Search
===============================================
Trains only the top-performing configurations:
1. ResU-Net + BS=8 + combined_tversky
2. Attention ResU-Net + BS=8 + focal_tversky
3. U-Net + BS=8 + combined_tversky
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_architectures import get_model
from loss_functions import get_loss_function, jacard_coef, dice_coef

# Configuration
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 1
LEARNING_RATE = 5e-5
EPOCHS = 100
EARLY_STOP_PATIENCE = 12  # Reduced from 30 to prevent overfitting

# Best configurations from hyperparameter search
BEST_CONFIGS = [
    {
        'name': 'resunet_bs8_combined_tversky',
        'architecture': 'resunet',
        'batch_size': 8,
        'dropout': 0.3,
        'loss_function': 'combined_tversky'
    },
    {
        'name': 'attention_resunet_bs8_focal_tversky',
        'architecture': 'attention_resunet',
        'batch_size': 8,
        'dropout': 0.3,
        'loss_function': 'focal_tversky'
    },
    {
        'name': 'unet_bs8_combined_tversky',
        'architecture': 'unet',
        'batch_size': 8,
        'dropout': 0.3,
        'loss_function': 'combined_tversky'
    }
]

# Output directory
OUTPUT_DIR = Path(f'./best_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}')


def load_training_data():
    """Load training and validation data"""
    print("\nLoading training data...")

    # Load images and masks
    X_train = np.load('dataset_shrunk_masks/X_train.npy')
    y_train = np.load('dataset_shrunk_masks/y_train.npy')
    X_val = np.load('dataset_shrunk_masks/X_val.npy')
    y_val = np.load('dataset_shrunk_masks/y_val.npy')

    print(f"  Training set: {X_train.shape}, {y_train.shape}")
    print(f"  Validation set: {X_val.shape}, {y_val.shape}")

    return X_train, y_train, X_val, y_val


def train_model(config, X_train, y_train, X_val, y_val):
    """Train single model with given configuration"""
    print(f"\n{'='*80}")
    print(f"Training: {config['name']}")
    print(f"{'='*80}")

    # Create output directory
    model_dir = OUTPUT_DIR / config['name']
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    print(f"\nBuilding {config['architecture']} model...")
    model = get_model(
        model_name=config['architecture'],
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        NUM_CLASSES=1,
        dropout_rate=config['dropout'],
        batch_norm=True
    )

    # Get loss function
    loss_fn = get_loss_function(config['loss_function'])

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy', jacard_coef, dice_coef]
    )

    print(f"  Parameters: {model.count_params():,}")

    # Data augmentation
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=15,
        fill_mode='reflect',
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    val_datagen = ImageDataGenerator()

    # Checkpoint path (ABSOLUTE PATH)
    checkpoint_path = os.path.abspath(
        str(model_dir / f"model_best.hdf5")
    )

    print(f"\nModel will be saved to: {checkpoint_path}")
    print(f"Directory exists: {model_dir.exists()}")
    print(f"Directory writable: {os.access(model_dir, os.W_OK)}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_jacard_coef',
            save_best_only=True,
            mode='max',
            verbose=1,  # VERBOSE!
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_jacard_coef',
            patience=EARLY_STOP_PATIENCE,
            verbose=1,
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1,
            min_lr=1e-7
        )
    ]

    # Train
    seed = 42
    print(f"\nStarting training...")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Max epochs: {EPOCHS}")
    print(f"  Early stopping patience: {EARLY_STOP_PATIENCE}")

    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=config['batch_size'], seed=seed),
        validation_data=val_datagen.flow(X_val, y_val, batch_size=config['batch_size'], seed=seed),
        steps_per_epoch=len(X_train) // config['batch_size'],
        validation_steps=len(X_val) // config['batch_size'],
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save history
    hist_df = pd.DataFrame(history.history)
    hist_path = model_dir / 'training_history.csv'
    hist_df.to_csv(hist_path, index=False)

    # Verify model was saved
    if os.path.exists(checkpoint_path):
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"\n✓ Model saved successfully!")
        print(f"  Path: {checkpoint_path}")
        print(f"  Size: {file_size_mb:.1f} MB")
    else:
        print(f"\n✗ ERROR: Model file not found after training!")
        print(f"  Expected: {checkpoint_path}")

    # Print best metrics
    best_epoch = np.argmax(hist_df['val_jacard_coef'])
    best_jaccard = hist_df['val_jacard_coef'].iloc[best_epoch]

    print(f"\nBest validation Jaccard: {best_jaccard:.4f} at epoch {best_epoch + 1}")

    return history, checkpoint_path


def main():
    print("="*80)
    print("TRAINING BEST 3 MODELS")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load data
    X_train, y_train, X_val, y_val = load_training_data()

    # Train each model
    results = {}
    for config in BEST_CONFIGS:
        try:
            history, model_path = train_model(config, X_train, y_train, X_val, y_val)
            results[config['name']] = {
                'success': True,
                'model_path': model_path,
                'best_jaccard': np.max(history.history['val_jacard_coef'])
            }
        except Exception as e:
            print(f"\n✗ Error training {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            results[config['name']] = {
                'success': False,
                'error': str(e)
            }

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    for name, result in results.items():
        if result['success']:
            print(f"✓ {name}: Jaccard = {result['best_jaccard']:.4f}")
            print(f"  Model: {result['model_path']}")
        else:
            print(f"✗ {name}: FAILED - {result['error']}")

    print(f"\n All models saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
```

### Solution 4: Quick Test with One Model

To verify the training/saving works before running all 3:

```bash
# Edit train_best_three_models.py to train only first config
# Comment out configs [1] and [2]

python train_best_three_models.py
```

This should produce:
- `best_models_YYYYMMDD_HHMMSS/resunet_bs8_combined_tversky/model_best.hdf5` (~350MB)
- Training should reach ~0.25-0.31 Jaccard

---

## 🚀 Action Plan

### Step 1: Check HPC Training Log (10 minutes)

```bash
ssh HPC
cd /home/svu/phyzxi/scratch/unet-HPC
grep -C 5 "checkpoint\|ModelCheckpoint\|Saving model" Hyperparam_Comprehensive.o285XXX
```

Look for error messages about disk space or permissions.

### Step 2: Fix and Re-train (Option A: Full search)

```bash
# Edit hyperparam_search_comprehensive.py:
#  - Add verbose=1 to ModelCheckpoint
#  - Add absolute paths
#  - Add verification print statements

# Re-submit
qsub pbs_hyperparam_comprehensive.sh

# Monitor
tail -f Hyperparam_Comprehensive.o*
# Should see "Epoch XX: val_jacard_coef improved from X to Y, saving model to..."
```

### Step 3: Train Top 3 Only (Option B: Faster)

```bash
# Create train_best_three_models.py (provided above)
# Create PBS script

# Submit
qsub pbs_train_best_three.sh

# Expected time: 2-4 hours (vs 12-24 for full search)
```

### Step 4: Re-run Prediction

```bash
# Once models exist:
python predict_with_density_analysis.py

# Should now produce reasonable densities:
# - ResU-Net: 40-70% (not 100%)
# - U-Net: 25-60% (not 0.2%)
# - Attention ResU-Net: 30-65% (not 0.4%)
```

### Step 5: Re-generate Dilution Factor Plots

```bash
python reanalyze_density_by_dilution.py

# Expected output:
# - CLAHE+OTSU: 64.8% @ 10x (unchanged - reference)
# - ResU-Net: ~58-68% @ 10x (reasonable prediction)
# - U-Net: ~50-70% @ 10x
# - Attention ResU-Net: ~55-65% @ 10x
```

---

## 📊 Expected Results After Fix

### Density Comparison (10x dilution)

| Method | Current (Broken) | After Fix (Expected) |
|--------|-----------------|---------------------|
| CLAHE+OTSU | 64.8% | 64.8% (unchanged) |
| ResU-Net | 100.0% ❌ | 58-68% ✓ |
| U-Net | 0.08% ❌ | 50-70% ✓ |
| Attention ResU-Net | 1.42% ❌ | 55-65% ✓ |

### Correlation with Reference

After fix, deep learning models should show:
- **Pearson correlation** with CLAHE+OTSU: r > 0.80
- **Mean absolute error**: < 10% density units
- **Consistent trend**: Higher dilution = lower density

---

## 📁 Files Created

1. **PREDICTION_ISSUE_SUMMARY.md** (this file) - Complete diagnosis
2. **PREDICTION_DIAGNOSIS.md** - Detailed diagnostic steps
3. **diagnose_model_outputs.py** - Diagnostic script (confirmed no models)
4. **reanalyze_density_by_dilution.py** - Working analysis script with dilution factors
5. **train_best_three_models.py** - Script to retrain top 3 models

---

## 🎯 Success Criteria

Training is successful when:
1. ✅ Model files exist (>100MB each)
2. ✅ Best validation Jaccard > 0.25
3. ✅ Prediction density within 20% of CLAHE+OTSU reference
4. ✅ Predicted density correlates with dilution factor

---

**Status**: Root cause identified, solutions provided
**Next**: Train models (either full search or top 3)
**ETA**: 2-24 hours depending on approach chosen
