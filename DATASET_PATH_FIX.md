# Dataset Path Fix - Attention Hyperparameter Training

## Problem (Job 288571)

```
ERROR: Training directory not found: ./dataset_new_shrunk/train
```

## Root Cause

The training script `train_attention_models_hyperparam.py` was looking for a pre-split dataset with separate `train/` and `val/` directories:
- `./dataset_new_shrunk/train/images/`
- `./dataset_new_shrunk/train/masks/`
- `./dataset_new_shrunk/val/images/`
- `./dataset_new_shrunk/val/masks/`

However, the actual dataset is located at:
- `./dataset_shrunk_masks/images/` (98 images)
- `./dataset_shrunk_masks/masks/` (98 masks)

This matches the dataset used in the original Xukuang training (`train_shrunk_xukuang_parameters.py`).

## Solution

### 1. Updated Configuration (train_attention_models_hyperparam.py)

**Before:**
```python
CONFIG = {
    'train_images': './dataset_new_shrunk/train/images/',
    'train_masks': './dataset_new_shrunk/train/masks/',
    'val_images': './dataset_new_shrunk/val/images/',
    'val_masks': './dataset_new_shrunk/val/masks/',
    # ...
}
```

**After:**
```python
CONFIG = {
    'images_dir': './dataset_shrunk_masks/images/',
    'masks_dir': './dataset_shrunk_masks/masks/',
    'train_val_split': 0.8,  # 80% train, 20% validation
    # ...
}
```

### 2. Rewrote Data Loading Function

**Removed dependency on `DataGenerator`** (which didn't exist) and switched to direct array loading like the original Xukuang script.

**New `load_data()` function:**
- Loads all images and masks from single directory
- Normalizes images to [0, 1]
- Normalizes masks to binary [0, 1]
- Performs 80/20 train/val split using `sklearn.train_test_split`
- Returns: `X_train, X_val, y_train, y_val` as numpy arrays

**Key changes:**
```python
# Load images directly into memory (like train_shrunk_xukuang_parameters.py)
for image_name in image_files:
    image = cv2.imread(image_path, 1)  # RGB
    image = Image.fromarray(image).resize((512, 512))
    image_dataset.append(np.array(image))

    mask = cv2.imread(mask_path, 0)  # Grayscale
    mask = Image.fromarray(mask).resize((512, 512))
    mask_dataset.append(np.array(mask))

# Normalize and split
image_dataset = np.array(image_dataset) / 255.0
mask_dataset = np.array(mask_dataset) / 255.0
mask_dataset = np.expand_dims(mask_dataset, axis=-1)

X_train, X_val, y_train, y_val = train_test_split(
    image_dataset, mask_dataset,
    test_size=0.2, random_state=42
)
```

### 3. Updated Training Function

Changed from generator-based to array-based training:

**Before:**
```python
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=config['epochs'],
    callbacks=callbacks,
    verbose=1
)
```

**After:**
```python
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=config['epochs'],
    batch_size=config['batch_size'],
    callbacks=callbacks,
    verbose=1
)
```

### 4. Updated PBS Script Verification

**pbs_train_attention_hyperparam.sh:**

**Before:**
```bash
TRAIN_DIR="./dataset_new_shrunk/train"
VAL_DIR="./dataset_new_shrunk/val"
```

**After:**
```bash
IMAGES_DIR="./dataset_shrunk_masks/images"
MASKS_DIR="./dataset_shrunk_masks/masks"
```

Also removed check for non-existent `data_generator.py`.

## Dataset Info

- **Total images:** 98 PNG files
- **Total masks:** 98 PNG files
- **Train split:** 78 images (80%)
- **Val split:** 20 images (20%)
- **Image size:** Resized to 512×512 during loading
- **Normalization:** [0, 255] → [0, 1]
- **Random seed:** 42 (for reproducibility)

## Files Modified

1. `train_attention_models_hyperparam.py`
   - Updated CONFIG section
   - Removed DataGenerator import
   - Added cv2, PIL, sklearn imports
   - Rewrote `load_data()` function
   - Updated `train_model()` signature
   - Updated `run_hyperparam_search()` signature
   - Updated `main()` function calls

2. `pbs_train_attention_hyperparam.sh`
   - Updated dataset path verification
   - Removed data_generator.py check
   - Updated output messages

## Verification

```bash
# Syntax check
python3 -m py_compile train_attention_models_hyperparam.py
# ✓ Passed

# Dataset check
ls dataset_shrunk_masks/images/*.png | wc -l
# 98

ls dataset_shrunk_masks/masks/*.png | wc -l
# 98
```

## Ready for Submission

The script is now ready for HPC submission:

```bash
qsub pbs_train_attention_hyperparam.sh
```

## Expected Behavior

1. Load 98 image-mask pairs from `dataset_shrunk_masks/`
2. Split: 78 train, 20 validation
3. Train 2 architectures × 18 hyperparameter combinations = **36 models**
4. Save best and final models for each combination
5. Generate results CSV with all metrics

---

**Fixed:** October 16, 2025
**Status:** ✅ Ready for submission
