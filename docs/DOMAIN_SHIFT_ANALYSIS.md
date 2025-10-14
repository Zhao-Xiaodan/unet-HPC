# Domain Shift Analysis: Mitochondria vs. Microbeads

## Critical Problem Identified

**Your hypothesis is correct:** The hyperparameters optimized for **sparse mitochondria segmentation** (2-3 objects per image) are fundamentally incompatible with **dense microbead detection** (dozens of objects per image).

---

## Domain Comparison

### Original Training Dataset (Mitochondria - EM Images)

| Characteristic | Value | Impact on Training |
|----------------|-------|-------------------|
| **Objects per image** | 2-3 mitochondria | Sparse segmentation |
| **Object coverage** | 5-15% of image | Low positive class ratio |
| **Object morphology** | Irregular, branching | Complex boundaries |
| **Background** | Uniform, dark | Easy to distinguish |
| **Image modality** | Electron microscopy | High contrast |

**Hyperparameter optimization results:**
- Learning rates: 1e-4 to 1e-3
- Batch sizes: 8-16
- Binary Focal Loss (γ=2)
- Expected Jaccard: 0.067-0.070

---

### Your Dataset (Microbeads - Light Microscopy)

| Characteristic | Estimated Value | Impact on Training |
|----------------|----------------|-------------------|
| **Objects per image** | **Dozens (20-100+)** | **Dense segmentation** |
| **Object coverage** | **30-60% of image** | **High positive class ratio** |
| **Object morphology** | **Circular, uniform** | **Simple boundaries** |
| **Background** | Variable brightness | May have artifacts |
| **Image modality** | Light microscopy | **Different contrast characteristics** |

**Current training result:**
- Validation Jaccard: 0.11-0.14 (peak)
- Final Jaccard: ~0.0 (collapse)
- **Status: Complete failure**

---

## Why Hyperparameters Failed

### Issue 1: Class Imbalance Mismatch

**Mitochondria dataset:**
- Positive pixels: ~10%
- Negative pixels: ~90%
- **Focal Loss (γ=2)** designed to focus on hard negatives

**Microbead dataset:**
- Positive pixels: ~40-50% (estimated)
- Negative pixels: ~50-60%
- **Focal Loss (γ=2) is WRONG** - penalizes confident predictions too much

**Effect:**
```python
# With 50% positive class and γ=2:
# Model predicting confident 0.9 → Focal loss heavily down-weights it
# Model learns to predict uncertain 0.5 → Avoids penalty
# Result: Model never commits to strong predictions → Jaccard collapses
```

### Issue 2: Learning Rate Mismatch

**Mitochondria (sparse objects):**
- Gradients are weak (few positive pixels)
- Higher LR (1e-3) needed to make progress
- Optimization study found: LR=1e-3 for UNet, 1e-4 for Attention models

**Microbeads (dense objects):**
- Gradients are STRONG (many positive pixels)
- High LR causes instability and overfitting
- **Need: LR=1e-4 or even 1e-5**

**Your result:**
- UNet with LR=1e-3 peaked at epoch 4, then collapsed
- Attention models with LR=1e-4, 5e-4 peaked at epoch 1, then collapsed
- **Pattern: Initial learning, then catastrophic overfitting**

### Issue 3: Batch Size Mismatch

**Mitochondria (sparse):**
- Small batch (8-16) provides diverse gradient signals
- Each image has different sparse patterns

**Microbeads (dense, uniform):**
- Small batch leads to overfitting (too similar within batch)
- **Need: Larger batch size (16-32) for better generalization**

### Issue 4: Partial Mask Amplification

With **dozens of microbeads per image**:
- If only 50% are labeled → Model sees 50% as "negative class"
- With sparse mitochondria: Unlabeled = background (tolerable)
- With dense beads: Unlabeled beads = **active confusion signal**
- **Result: Model learns contradictory patterns → collapse**

---

## Recommended Solutions

### Solution 1: Custom Hyperparameter Optimization for Microbeads

Create a new optimization study specifically for your microbead dataset:

**Suggested hyperparameter ranges:**

```python
# Learning rates (MUCH LOWER than mitochondria)
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]

# Batch sizes (LARGER than mitochondria)
batch_sizes = [16, 32, 64]

# Loss function alternatives
loss_functions = [
    'binary_crossentropy',           # Standard, no focal bias
    BinaryFocalLoss(gamma=1),        # Mild focusing
    DiceLoss(),                      # Area-based, good for dense objects
    TverskyLoss(alpha=0.3, beta=0.7) # Reduce false positives
]

# Dropout rates (prevent overfitting)
dropout_rates = [0.1, 0.2, 0.3]
```

**Why these changes:**
- **Lower LR (1e-5)**: Strong gradients from dense objects need gentle updates
- **Larger batch (32-64)**: Smooths gradient variance from similar bead patterns
- **Binary cross-entropy**: No focal bias for balanced classes
- **Dice loss**: Optimizes overlap directly (good for dense segmentation)
- **Dropout**: Prevents memorization of specific bead configurations

### Solution 2: Fix Partial Mask Problem FIRST

**This is CRITICAL before any retraining:**

```bash
# Step 1: Use current predictions to identify unlabeled beads
python predict_microscope.py --input_dir dataset_microscope/images --output_dir mask_completion

# Step 2: Manually review and complete masks
# Compare predictions vs. ground truth
# Add missing bead labels to ground truth masks

# Step 3: Verify completeness
python verify_mask_completeness.py
```

**Verification script to create:**

```python
# verify_mask_completeness.py
import cv2
import numpy as np
from pathlib import Path

for img_path in Path('dataset_microscope/images').glob('*.tif'):
    img = cv2.imread(str(img_path))
    mask_path = Path('dataset_microscope/masks') / img_path.name
    mask = cv2.imread(str(mask_path), 0)

    # Count objects in image (via simple thresholding)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_img = cv2.connectedComponents(thresh)[0] - 1

    # Count objects in mask
    n_mask = cv2.connectedComponents(mask)[0] - 1

    coverage_ratio = n_mask / max(n_img, 1)

    if coverage_ratio < 0.8:
        print(f"⚠️  {img_path.name}: {n_mask}/{n_img} objects labeled ({coverage_ratio*100:.1f}%)")
    else:
        print(f"✓  {img_path.name}: {n_mask}/{n_img} objects labeled ({coverage_ratio*100:.1f}%)")
```

### Solution 3: Data Augmentation Strategy

For dense, uniform objects like microbeads:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Aggressive augmentation to prevent overfitting
datagen = ImageDataGenerator(
    rotation_range=180,           # Full rotation (beads are circular)
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],  # Handle lighting variation
    fill_mode='reflect'
)
```

**Why this helps:**
- Uniform bead shapes → rotation/flip doesn't change problem
- Prevents memorizing specific spatial positions
- Increases effective dataset size 10-20×

### Solution 4: Modified Loss Function

**Replace Binary Focal Loss with Dice Loss:**

```python
def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss for dense object segmentation
    Better than focal loss for balanced classes
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice

# Or combined loss
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice
```

**Why Dice is better here:**
- Optimizes Jaccard-like metric directly
- Not biased by class imbalance
- Works well for dense, overlapping objects

### Solution 5: Architecture Modifications

**Add regularization to prevent overfitting:**

```python
from tensorflow.keras import regularizers

# In model definitions, add:
Conv2D(filters, kernel_size,
       kernel_regularizer=regularizers.l2(1e-4),  # L2 weight decay
       activation='relu')

# Add dropout in decoder path
Dropout(0.3)  # After each decoder block
```

**Why this helps:**
- Dense objects → more parameters needed → more overfitting risk
- L2 regularization: Keeps weights small
- Dropout: Forces robustness to feature removal

---

## Recommended Training Strategy

### Phase 1: Dataset Preparation (ESSENTIAL)

1. **Complete partial masks** using prediction-assisted labeling
2. **Verify mask coverage** reaches >90% of visible beads
3. **Stratified train/val split** by bead density:
   ```python
   # Group by bead count
   bead_counts = [count_beads(mask) for mask in masks]
   bins = [0, 20, 50, 100, np.inf]
   strata = np.digitize(bead_counts, bins)

   X_train, X_val, y_train, y_val = train_test_split(
       images, masks, test_size=0.15,
       stratify=strata, random_state=42
   )
   ```

### Phase 2: Quick Baseline Test

**Test if the problem is hyperparameters or data:**

```python
# Simple baseline with standard settings
model = UNet(input_shape)
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Conservative LR
    loss='binary_crossentropy',          # No focal bias
    metrics=['accuracy', jacard_coef]
)

# Short training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,  # Larger batch
    epochs=20,       # Short test
    callbacks=[EarlyStopping(patience=5)]
)

# Check: Does val_jaccard improve past 0.3?
# If YES → hyperparameter issue
# If NO → dataset/architecture issue
```

### Phase 3: Systematic Optimization

**Create microbead-specific hyperparameter search:**

```python
# Grid search configuration
config = {
    'architecture': ['UNet', 'Attention_UNet'],
    'learning_rate': [5e-5, 1e-4, 5e-4],
    'batch_size': [16, 32],
    'loss': ['binary_crossentropy', 'dice', 'combined'],
    'dropout': [0.2, 0.3],
    'augmentation': [True]
}

# Expected better results:
# Target: Val Jaccard > 0.50
# Good: Val Jaccard > 0.70
# Excellent: Val Jaccard > 0.85
```

### Phase 4: Training Configuration

**Updated training script for microbeads:**

```python
# Microbead-optimized configuration
IMG_SIZE = 256
BATCH_SIZE = 32          # Increased from 8-16
LEARNING_RATE = 1e-4     # Lower than mitochondria
EPOCHS = 100
DROPOUT = 0.3            # Added regularization

model = UNet(input_shape, dropout_rate=DROPOUT)
model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
    ),
    loss=combined_loss,   # BCE + Dice instead of Focal
    metrics=['accuracy', jacard_coef, dice_coef]
)

callbacks = [
    EarlyStopping(
        monitor='val_jacard_coef',
        patience=20,  # Increased patience
        mode='max',
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        'best_microbead_model.hdf5',
        monitor='val_jacard_coef',
        mode='max',
        save_best_only=True
    )
]

# Use data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=callbacks,
    steps_per_epoch=len(X_train) // BATCH_SIZE
)
```

---

## Expected Performance Improvements

### Current Results (Mitochondria Hyperparameters on Microbeads)

| Metric | Value | Status |
|--------|-------|--------|
| Best Val Jaccard | 0.11-0.14 | ❌ Very poor |
| Final Val Jaccard | ~0.0 | ❌ Collapsed |
| Training stability | Good (low σ) | ⚠️ Misleading |
| Epoch to peak | 1-4 | ❌ Immediate overfitting |

### Expected Results (Microbead-Optimized Hyperparameters)

| Metric | Expected Value | Likelihood |
|--------|---------------|------------|
| Best Val Jaccard | 0.50-0.70 | High (with complete masks) |
| Best Val Jaccard | 0.70-0.85 | Medium (with augmentation) |
| Final Val Jaccard | Within 10% of best | High (with proper regularization) |
| Training stability | Stable convergence | High |
| Epoch to peak | 30-60 | High (slower, more stable learning) |

---

## Diagnostic Comparison

### What Poor Predictions Tell Us

**If predictions show:**

1. **Many small false positives scattered everywhere:**
   - Issue: Model memorizing training noise
   - Solution: Increase dropout, larger batch size

2. **Missing many obvious beads:**
   - Issue: Partial mask problem OR threshold too high
   - Solution: Complete masks, try threshold=0.3

3. **Predicting all background:**
   - Issue: Loss function driving to trivial solution
   - Solution: Change from Focal to Dice/BCE

4. **Blurry boundaries:**
   - Issue: Tile overlap too low OR model uncertainty
   - Solution: Increase overlap to 64px, add boundary loss term

5. **Correctly detecting beads but poor boundaries:**
   - Issue: Model is learning but needs refinement
   - Solution: Train longer, add deep supervision

---

## Implementation Priority

### Priority 1: CRITICAL (Do First)

1. ✅ **Complete partial masks** - Cannot train properly otherwise
2. ✅ **Verify mask quality** - Check >90% bead coverage
3. ✅ **Quick baseline test** - Use BCE loss, LR=1e-4, BS=32 for 20 epochs

### Priority 2: HIGH (Essential for Good Results)

4. ✅ **Change loss function** - Dice or Combined instead of Focal
5. ✅ **Adjust learning rate** - Lower to 1e-4 or 5e-5
6. ✅ **Increase batch size** - 32 or 64 instead of 8-16
7. ✅ **Add dropout regularization** - 0.2-0.3 in decoder

### Priority 3: MEDIUM (Significant Improvements)

8. ✅ **Data augmentation** - Rotation, flip, brightness
9. ✅ **Stratified validation split** - By bead density
10. ✅ **Increase patience** - 15-20 epochs instead of 10

### Priority 4: LOW (Fine-tuning)

11. ✅ **Hyperparameter grid search** - Systematic optimization
12. ✅ **Architecture modifications** - Deep supervision, attention variants
13. ✅ **Ensemble methods** - Combine multiple models

---

## Next Steps

### Step 1: Diagnostic Check (Today)

```bash
# Run quick analysis on your dataset
python analyze_microbead_dataset.py
```

I'll create this script in the next response.

### Step 2: Baseline Test (1-2 hours)

```bash
# Test with microbead-appropriate settings
python train_microbead_baseline.py
```

Expected: Val Jaccard > 0.30 indicates hyperparameters are the issue
If still < 0.20: Dataset quality is the issue

### Step 3: Full Retraining (4-8 hours)

```bash
# Complete retraining with optimized settings
qsub pbs_microbead_optimized.sh
```

Expected: Val Jaccard > 0.60

---

## Summary

**Your intuition is 100% correct:** The hyperparameters optimized for sparse mitochondria (2-3 per image) are fundamentally wrong for dense microbeads (dozens per image).

**Key differences:**
- **Class balance:** 10% → 40-50% positive pixels
- **Object density:** 2-3 → 20-100+ objects
- **Gradient strength:** Weak → Strong
- **Optimal LR:** 1e-3 → 1e-4 or lower
- **Optimal batch:** 8-16 → 32-64
- **Optimal loss:** Focal(γ=2) → Dice or BCE

**Most critical issue:** Likely **partial masks** amplified by dense objects causing training collapse.

**Immediate action:** Create diagnostic scripts to analyze your dataset characteristics before retraining.

Would you like me to create:
1. Dataset analysis script
2. Baseline test script
3. Optimized training script for microbeads
4. PBS script for HPC retraining
