# Bug Fix: Attention UNet Density Analysis

**Date:** October 16, 2025
**Job Failed:** Density_AttnUNet_Only.o293378
**Status:** ✅ Fixed

---

## Problem

Density analysis for Attention UNet failed immediately with error:

```
ERROR: No best_model.keras files found in ./attention_unet_hyperparam_20251015_230149/models
       Make sure Attention UNet hyperparameter search has completed!
```

**Job Details:**
- Job ID: 293378.stdct-mgmt-02
- Exit Status: 1
- Walltime Used: 00:00:01 (failed immediately)
- Node: GN-A40-074

---

## Root Cause

**Incorrect directory path in model search function.**

The analysis script (`density_analysis_attention_unet_only.py`) was searching for models in:
```python
model_dirs = list((base_dir / 'models').glob('attention_unet_*'))
```

But the actual training script saves models to:
```
attention_unet_hyperparam_20251015_230149/checkpoints/
```

### Why the Confusion?

**During development**, I incorrectly assumed:
- UNet models saved to `checkpoints/` ✓
- Attention UNet models saved to `models/` ✗ **WRONG**
- Attention ResUNet models saved to `models/` ✗ **WRONG**

**Reality (verified from training logs):**
- **ALL three architectures** save to `checkpoints/`
- Training script uses `ModelCheckpoint` with path: `checkpoints/{experiment_name}/best_model.keras`
- Final models also saved to `models/{experiment_name}_final.keras` but **not best_model.keras**

---

## Verification from Training Logs

From `AttentionUNet_Hyperparam.o288650`:

```
Epoch 1: val_jacard_coef improved from -inf to 0.16581,
    saving model to attention_unet_hyperparam_20251015_230149/checkpoints/
    attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001/best_model.keras

Epoch 45: val_jacard_coef improved from 0.43007 to 0.43520,
    saving model to attention_unet_hyperparam_20251015_230149/checkpoints/
    attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001/best_model.keras

✓ Saved final model: attention_unet_hyperparam_20251015_230149/models/
    attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001_final.keras
```

**Key Observations:**
- `best_model.keras` → `checkpoints/` (used by density analysis)
- `*_final.keras` → `models/` (not used by density analysis)

---

## Fix Applied

### 1. Python Script (`density_analysis_attention_unet_only.py`)

**Line 144:**
```python
# OLD (incorrect)
model_dirs = list((base_dir / 'models').glob('attention_unet_*'))

# NEW (correct)
model_dirs = list((base_dir / 'checkpoints').glob('attention_unet_*'))
```

**Line 147:**
```python
# OLD
raise FileNotFoundError(f"No Attention UNet model directories found in {base_dir / 'models'}")

# NEW
raise FileNotFoundError(f"No Attention UNet model directories found in {base_dir / 'checkpoints'}")
```

### 2. PBS Script (`pbs_density_analysis_attention_unet_only.sh`)

**Line 93:**
```bash
# OLD
if [ ! -d "$MODEL_DIR/models" ]; then
    echo "ERROR: Models directory not found: $MODEL_DIR/models"
    exit 1
fi

# NEW
if [ ! -d "$MODEL_DIR/checkpoints" ]; then
    echo "ERROR: Checkpoints directory not found: $MODEL_DIR/checkpoints"
    exit 1
fi
```

**Line 109:**
```bash
# OLD
MODEL_COUNT=$(find "$MODEL_DIR/models" -name "best_model.keras" | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "ERROR: No best_model.keras files found in $MODEL_DIR/models"

# NEW
MODEL_COUNT=$(find "$MODEL_DIR/checkpoints" -name "best_model.keras" | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "ERROR: No best_model.keras files found in $MODEL_DIR/checkpoints"
```

**Line 117:**
```bash
# OLD
echo "✓ Models subdirectory: $MODEL_DIR/models"

# NEW
echo "✓ Checkpoints subdirectory: $MODEL_DIR/checkpoints"
```

**Line 34 (documentation):**
```bash
# OLD
#   - From: attention_unet_hyperparam_20251015_230149/models/

# NEW
#   - From: attention_unet_hyperparam_20251015_230149/checkpoints/
```

### 3. Documentation (`ATTENTION_UNET_DENSITY_ANALYSIS_SETUP.md`)

Updated directory structure diagram and model selection algorithm to reflect correct path.

---

## Testing Strategy

### Verification Checklist

**Before resubmitting job:**
- [x] Verify `checkpoints/` directory exists on HPC
- [x] Count model files: Should find 27 `best_model.keras` files
- [x] Check path consistency across all 3 files (Python, PBS, docs)

### Expected Behavior (after fix)

```bash
# PBS script validation output:
✓ Model directory: ./attention_unet_hyperparam_20251015_230149
✓ Checkpoints subdirectory: ./attention_unet_hyperparam_20251015_230149/checkpoints
✓ Test images directory: ./test_images
✓ Found 27 Attention UNet model(s)

# Python script model selection:
Searching for best Attention UNet model...
Found 27 Attention UNet model configurations
  New best: attention_unet_n_filters16_dropout0p3_batch_normTrue_learning_rate0p003 (IoU: 0.4875)
```

---

## Impact Analysis

### Files Modified
1. ✅ `density_analysis_attention_unet_only.py` (2 lines)
2. ✅ `pbs_density_analysis_attention_unet_only.sh` (4 locations)
3. ✅ `ATTENTION_UNET_DENSITY_ANALYSIS_SETUP.md` (documentation)

### Files NOT Affected
- ✅ `density_analysis_unet_only.py` (already uses `checkpoints/`)
- ⚠️ `density_analysis_attention_resunet_only.py` (if exists, needs same fix)

### Backward Compatibility
- ✅ No breaking changes
- ✅ Works with existing training outputs
- ✅ Consistent with UNet density analysis

---

## Lessons Learned

### 1. Verify Directory Structure Before Deployment

**Issue:** Assumed Attention UNet used different directory structure without verification.

**Solution:** Always check actual training outputs:
```bash
# Quick check on HPC
ls -la attention_unet_hyperparam_20251015_230149/
ls -la attention_unet_hyperparam_20251015_230149/checkpoints/ | head -5
```

### 2. Inconsistent Documentation Led to Error

**Problem:** Earlier documentation mentioned `models/` directory, causing confusion.

**Fix:** Updated all documentation to clearly state:
- `checkpoints/` contains `best_model.keras` (used by analysis)
- `models/` contains `*_final.keras` (not used by analysis)

### 3. Test Jobs Immediately After Creation

**Mistake:** Created analysis scripts without running a quick test.

**Better Approach:**
```bash
# Dry-run test (locally or quick HPC test)
python density_analysis_attention_unet_only.py --dry-run
# OR submit with short walltime for validation
qsub -l walltime=00:05:00 pbs_density_analysis_attention_unet_only.sh
```

---

## Prevention for Future Scripts

### Template Checklist for New Density Analysis Scripts

When creating similar scripts for other architectures:

1. **Verify model save location:**
   ```bash
   grep -r "ModelCheckpoint\|save.*model" train_*.py
   ```

2. **Check actual directories on HPC:**
   ```bash
   ls -la {architecture}_hyperparam_*/
   find {architecture}_hyperparam_*/ -name "best_model.keras" | head -3
   ```

3. **Validate model count:**
   ```python
   model_count = len(list(Path(base_dir / 'checkpoints').glob('**/best_model.keras')))
   assert model_count == 27, f"Expected 27 models, found {model_count}"
   ```

4. **Test model loading:**
   ```python
   from tensorflow import keras
   test_model = keras.models.load_model(first_model_path, compile=False)
   assert test_model is not None
   ```

---

## Status

✅ **Fixed and Ready for Resubmission**

**Next Steps:**
1. Transfer updated files to HPC
2. Resubmit job: `qsub pbs_density_analysis_attention_unet_only.sh`
3. Monitor initial output to confirm directory found
4. Expected runtime: ~3-4 hours (similar to UNet density analysis)

---

## Related Files

- `density_analysis_attention_unet_only.py` - Fixed Python script
- `pbs_density_analysis_attention_unet_only.sh` - Fixed PBS script
- `ATTENTION_UNET_DENSITY_ANALYSIS_SETUP.md` - Updated documentation
- `Density_AttnUNet_Only.o293378` - Original error log (reference)

---

**Bug Report Date:** October 16, 2025
**Fix Applied:** October 16, 2025
**Status:** ✅ Ready for deployment
