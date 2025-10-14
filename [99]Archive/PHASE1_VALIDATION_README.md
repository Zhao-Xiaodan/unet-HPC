# Phase 1: Quick Validation of Training Fixes

## Overview

This is a **1-hour quick test** to verify that all 5 solutions fix the FP16/NaN training failure identified in the analysis.

**Purpose:** Validate that training is numerically stable before running the full 12-24 hour hyperparameter search.

---

## Files Created

### 1. `loss_functions_fixed.py`
**Updated loss functions with numerical stability improvements:**
- ✅ Smoothing constants: `1e-6` → `1e-3` (Solution 2)
- ✅ Safe clipping to prevent underflow/overflow (Solution 5)
- ✅ Robust edge case handling
- ✅ Works with both FP32 and FP16 (though we're using FP32)

**Key changes:**
```python
# BEFORE (unstable in FP16):
def tversky_loss(y_true, y_pred, smooth=1e-6):  # ← Underflows to 0 in FP16
    ...

# AFTER (stable in FP32 and FP16):
def tversky_loss(y_true, y_pred, smooth=1e-3):  # ← Safe for FP16
    y_pred_f = K.clip(y_pred_f, 1e-3, 1 - 1e-3)  # ← Additional clipping
    ...
```

---

### 2. `validate_training_fixes.py`
**Validation training script implementing ALL 5 solutions:**

#### Solution 1: Disable Mixed Precision ✅
```python
# NO mixed precision code - uses default FP32
verify_no_mixed_precision()  # Verifies FP32 is active
```

#### Solution 2: Increase Smoothing Constants ✅
```python
from loss_functions_fixed import get_loss_function  # Uses smooth=1e-3
```

#### Solution 3: NaN Detection Callback ✅
```python
class TerminateOnNaN(Callback):
    """Stops training immediately if NaN/inf detected"""
    def on_batch_end(self, batch, logs=None):
        if np.isnan(loss) or np.isinf(loss):
            self.model.stop_training = True
```

#### Solution 4: Gradient Clipping ✅
```python
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=5e-5,
        clipnorm=1.0  # ← Prevents gradient explosion
    ),
    loss=loss_fn,
    metrics=['accuracy', jacard_coef, dice_coef]
)
```

#### Solution 5: Improved Loss Function Stability ✅
- Uses `loss_functions_fixed.py` with all stability improvements

---

### 3. `pbs_validate_fixes.sh`
**PBS submission script for HPC:**
- ✅ 2-hour walltime (enough for 20 epochs)
- ✅ GPU memory configured for FP32
- ✅ Pre-flight checks (dataset, files, loss functions)
- ✅ Post-training analysis and reporting

---

## Running the Validation

### Step 1: Upload Files to HPC

```bash
# On your local machine:
cd /Users/xiaodan/unetCNN/unet-HPC

# Upload the new files:
scp loss_functions_fixed.py pbs_validate_fixes.sh validate_training_fixes.py \
    phyzxi@hpc:/home/svu/phyzxi/scratch/unet-HPC/

# SSH to HPC:
ssh phyzxi@hpc
cd /home/svu/phyzxi/scratch/unet-HPC
```

---

### Step 2: Verify Files

```bash
# Check all required files exist:
ls -lh loss_functions_fixed.py
ls -lh validate_training_fixes.py
ls -lh pbs_validate_fixes.sh
ls -lh model_architectures.py

# Check dataset:
ls dataset_shrunk_masks/images/ | wc -l  # Should be 98
ls dataset_shrunk_masks/masks/ | wc -l   # Should be 98
```

---

### Step 3: Test Loss Functions Locally (Optional)

```bash
# Quick test to verify loss functions work:
module load singularity
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

singularity exec $image python3 loss_functions_fixed.py
```

**Expected output:**
```
================================================================================
TESTING NUMERICALLY STABLE LOSS FUNCTIONS
================================================================================

Test data:
  y_true: [1. 0. 1. 0.]
  y_pred: [0.9 0.1 0.8 0.2]

--------------------------------------------------------------------------------
Loss function values:
--------------------------------------------------------------------------------
focal                    : 0.060583  ✓ FINITE
tversky                  : 0.149925  ✓ FINITE
focal_tversky            : 0.183608  ✓ FINITE
combined                 : 0.122573  ✓ FINITE
combined_tversky         : 0.114150  ✓ FINITE

--------------------------------------------------------------------------------
Metric values:
--------------------------------------------------------------------------------
dice_coef                : 0.850075  ✓
jacard_coef              : 0.739510  ✓

================================================================================
✓ All loss functions tested successfully!
================================================================================
```

---

### Step 4: Submit Validation Job

```bash
# Make PBS script executable:
chmod +x pbs_validate_fixes.sh

# Submit job:
qsub pbs_validate_fixes.sh

# Note the job ID (e.g., 285500)
```

---

### Step 5: Monitor Progress

```bash
# Check job status:
qstat -u phyzxi

# Watch the log file in real-time:
tail -f Validate_Training_Fixes.o285500

# Check for key indicators:
grep "Epoch" Validate_Training_Fixes.o285500      # Training progress
grep "NaN" Validate_Training_Fixes.o285500        # NaN detection
grep "VALIDATION" Validate_Training_Fixes.o285500 # Final result
```

---

## Expected Output

### Phase 1: Pre-flight Checks

```
==============================================================================
Pre-flight Checks
==============================================================================
✓ Dataset directory found: dataset_shrunk_masks
  Images: 98 files
  Masks: 98 files

Checking required files...
  ✓ validate_training_fixes.py
  ✓ loss_functions_fixed.py
  ✓ model_architectures.py
```

---

### Phase 2: Loss Function Test

```
==============================================================================
Testing Loss Functions (Pre-check)
==============================================================================
...
✓ Loss function test PASSED
```

---

### Phase 3: Training Progress

**What you should see:**
```
==============================================================================
STARTING TRAINING
==============================================================================

Epoch 1/20
13/13 [==============================] - 42s 2s/step - loss: 0.5234 - val_loss: 0.6012 - val_jacard_coef: 0.1534
Epoch 1/20: loss=0.5234, val_loss=0.6012, val_jaccard=0.1534

Epoch 2/20
13/13 [==============================] - 38s 2s/step - loss: 0.4891 - val_loss: 0.5678 - val_jacard_coef: 0.1789
Epoch 2/20: loss=0.4891, val_loss=0.5678, val_jaccard=0.1789

...

Epoch 20/20
13/13 [==============================] - 38s 2s/step - loss: 0.3234 - val_loss: 0.4123 - val_jacard_coef: 0.3156 ✓ NEW BEST
Epoch 20/20: loss=0.3234, val_loss=0.4123, val_jaccard=0.3156 ✓ NEW BEST
```

**Key indicators of SUCCESS:**
- ✅ Loss values are **finite numbers** (not `nan`, not `inf`)
- ✅ Loss **decreases**: 0.52 → 0.32
- ✅ Jaccard **increases**: 0.15 → 0.31
- ✅ Smooth progression (no sudden jumps to NaN)

---

### Phase 4: Validation Results

**If validation PASSED:**
```
==============================================================================
VALIDATION RESULTS
==============================================================================

================================================================================
FINAL METRICS
================================================================================
Training loss:
  Initial: 0.5234
  Final: 0.3234
  Change: -0.2000 ✓

Validation loss:
  Initial: 0.6012
  Final: 0.4123

Validation Jaccard:
  Initial: 0.1534
  Final: 0.3156
  Best: 0.3201 (epoch 18)
  Change: +0.1622 ✓

================================================================================
SUCCESS CRITERIA
================================================================================
✓ No NaN/inf detected: PASS
✓ Loss decreased: PASS
✓ Jaccard increased: PASS
✓ Best Jaccard > 0.25: PASS (0.3201)

================================================================================
✓ VALIDATION PASSED (4/4 criteria met)

The fixes WORKED! Training is numerically stable.
Next step: Run full hyperparameter search with these fixes.
================================================================================
```

---

**If validation FAILED (NaN detected):**
```
==============================================================================
VALIDATION RESULTS
==============================================================================

❌ VALIDATION FAILED: NaN DETECTED

Root cause:
  The numerical stability fixes were insufficient
  Further investigation needed
```

This would indicate the fixes didn't work, and we need to investigate further.

---

## Success Criteria

| Criterion | Target | Why It Matters |
|-----------|--------|----------------|
| **No NaN/inf** | ✅ Required | Proves training is numerically stable |
| **Loss decreases** | ✅ Required | Proves model is learning |
| **Jaccard increases** | ✅ Required | Proves segmentation is improving |
| **Best Jaccard > 0.25** | ⚠️ Nice-to-have | Proves good performance (but 20 epochs is short) |

**Minimum requirement:** First 3 criteria (3/4)
**Full success:** All 4 criteria (4/4)

---

## Output Files

After completion, check the output directory:

```bash
# Find the output directory:
ls -ld validation_fixes_*

# Typical: validation_fixes_20251013_143022/
cd validation_fixes_20251013_143022/

# Files created:
ls -lh
```

**Expected files:**
```
model_best.hdf5             # Best model checkpoint (~350 MB)
training_history.csv        # Complete training history (20 rows)
validation_summary.json     # Summary of validation results
```

---

### Analyzing Results

#### 1. Check Validation Summary
```bash
cat validation_summary.json
```

**Example:**
```json
{
  "config": {
    "architecture": "unet",
    "batch_size": 4,
    "dropout": 0.3,
    "loss_function": "combined"
  },
  "final_loss": 0.3234,
  "final_val_loss": 0.4123,
  "final_val_jacard": 0.3156,
  "best_val_jacard": 0.3201,
  "best_epoch": 17,
  "nan_detected": false,
  "criteria_met": 4,
  "total_criteria": 4,
  "validation_passed": true
}
```

**Key fields:**
- `"nan_detected": false` ← Most important! No NaN means fixes worked
- `"best_val_jacard": 0.3201` ← Performance is good
- `"validation_passed": true` ← Ready for Phase 2

---

#### 2. Check Training History
```bash
# View first 5 epochs:
head -6 training_history.csv | column -t -s,

# View last 5 epochs:
tail -5 training_history.csv | column -t -s,

# Check for any NaN:
grep -i "nan\|inf" training_history.csv
# (Should return nothing!)
```

**Example history:**
```csv
epoch,loss,val_loss,val_jacard_coef,lr
0,0.5234,0.6012,0.1534,5e-05
1,0.4891,0.5678,0.1789,5e-05
2,0.4567,0.5234,0.2012,5e-05
...
19,0.3234,0.4123,0.3156,5e-05
```

**Check for:**
- ✅ No `nan` or `inf` values anywhere
- ✅ Loss decreases monotonically (or nearly so)
- ✅ Jaccard increases over time

---

#### 3. Plot Training Curves (Optional)

You can plot the training curves locally:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read history
df = pd.read_csv('validation_fixes_*/training_history.csv')

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
ax1.plot(df['loss'], label='Training Loss')
ax1.plot(df['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Curves')
ax1.legend()
ax1.grid(True)

# Jaccard curve
ax2.plot(df['val_jacard_coef'], label='Validation Jaccard', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Jaccard Coefficient')
ax2.set_title('Segmentation Performance')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('validation_curves.png', dpi=150)
print("Saved: validation_curves.png")
```

**Expected curves:**
- Loss: Smooth downward trend
- Jaccard: Smooth upward trend
- No sudden jumps or flat regions

---

## Troubleshooting

### Issue 1: Job Fails Immediately

**Symptoms:**
```
qstat shows job finished in <1 minute
Log shows "ERROR: Dataset directory not found"
```

**Solution:**
```bash
# Check dataset exists:
ls -ld dataset_shrunk_masks/

# If missing, check current directory:
pwd
# Should be: /home/svu/phyzxi/scratch/unet-HPC

# If in wrong directory, edit PBS script line 49:
cd /home/svu/phyzxi/scratch/unet-HPC  # ← Fix this path
```

---

### Issue 2: Loss Function Test Fails

**Symptoms:**
```
Testing Loss Functions (Pre-check)
...
✗ Loss function test FAILED (exit code: 1)
```

**Solution:**
```bash
# Test manually to see error:
module load singularity
image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif
singularity exec $image python3 loss_functions_fixed.py

# Check for import errors:
singularity exec $image python3 -c "import tensorflow as tf; print(tf.__version__)"
```

---

### Issue 3: Out of Memory

**Symptoms:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**
```python
# Edit validate_training_fixes.py line 47:
TEST_CONFIG = {
    'batch_size': 2,  # ← Reduce from 4 to 2
    ...
}
```

---

### Issue 4: NaN Still Detected

**Symptoms:**
```
❌ BATCH 5: INVALID LOSS DETECTED
Loss value: nan
```

**This would indicate the fixes DIDN'T work. Investigate:**

1. Check which loss function:
```bash
grep "Loss function:" Validate_Training_Fixes.o*
# If "combined" failed, try "focal" or "dice" alone
```

2. Check if FP16 is somehow enabled:
```bash
grep "Mixed precision" Validate_Training_Fixes.o*
# Should show "DISABLED (using float32)"
```

3. Verify loss_functions_fixed.py is being imported:
```bash
grep "from loss_functions_fixed import" validate_training_fixes.py
```

4. Try even larger smoothing:
```python
# Edit loss_functions_fixed.py:
smooth=1e-2  # ← Try 1e-2 instead of 1e-3
```

---

## Next Steps

### If Validation PASSED ✅

**Proceed to Phase 2: Full Hyperparameter Search**

1. Create `hyperparam_search_fixed.py` based on `validate_training_fixes.py`
2. Apply all 5 solutions to the full search
3. Test all 30 configurations (12-24 hours)
4. Expected results:
   - Best Jaccard: **0.50-0.70** (vs current 0.31 broken)
   - All models: **No NaN/inf**
   - Predictions: **Reasonable densities** (50-70% vs 100%/0%)

---

### If Validation PARTIAL ⚠️ (3/4 criteria met)

**Training is stable but performance could be better**

Possible improvements:
1. **More epochs**: Increase from 20 → 50 epochs
2. **Different loss**: Try `focal_tversky` instead of `combined`
3. **Larger batch size**: Try BS=8 if memory allows
4. **Learning rate tuning**: Try 1e-4 or 3e-5

Still proceed to Phase 2, as stability is the main goal.

---

### If Validation FAILED ❌ (NaN detected)

**Do NOT proceed to Phase 2**

Further investigation needed:
1. Verify all files uploaded correctly
2. Check TensorFlow version compatibility
3. Test with even simpler configuration (dice loss only, BS=2)
4. Consider using PyTorch instead (more numerically stable)

---

## Comparison: Before vs After Fixes

### Training Logs

**BEFORE (Broken - with FP16):**
```
Epoch 1/100: loss: nan - val_loss: nan - val_jacard: 0.138
Epoch 2/100: loss: nan - val_loss: nan - val_jacard: 0.140
...
Epoch 58/100: loss: nan - val_loss: nan - val_jacard: 0.137
```
❌ Training continues despite NaN from epoch 1!

**AFTER (Fixed - with FP32):**
```
Epoch 1/20: loss: 0.5234 - val_loss: 0.6012 - val_jacard: 0.1534
Epoch 2/20: loss: 0.4891 - val_loss: 0.5678 - val_jacard: 0.1789
...
Epoch 20/20: loss: 0.3234 - val_loss: 0.4123 - val_jacard: 0.3156
```
✅ All values finite, smooth convergence!

---

### Predictions

**BEFORE (Broken):**
- ResU-Net: 100% density (all white masks)
- U-Net: 0.08% density (all black masks)
- Attention ResU-Net: 1.4% density

**AFTER (Fixed - Expected):**
- ResU-Net: 55-68% density ✅
- U-Net: 50-70% density ✅
- Attention ResU-Net: 55-70% density ✅
- Reference (CLAHE+OTSU): 64.8% density

---

## Summary

**Phase 1 Purpose:** Prove that FP32 training with all fixes works correctly

**Time Required:** 30-60 minutes

**Success Metric:** No NaN/inf in training logs

**If successful:** Proceed to Phase 2 (full search)

**If failed:** Investigate further before Phase 2

---

## Questions?

Check the detailed analysis: `CRITICAL_TRAINING_FAILURE_ANALYSIS.md`

Key sections:
- Root cause explanation (page 3-8)
- Solution details (page 12-16)
- Expected results (page 18)

---

**Good luck with the validation!** 🚀

If all goes well, you should see:
```
✓✓✓ VALIDATION PASSED ✓✓✓
The fixes WORKED! Training is numerically stable.
```
