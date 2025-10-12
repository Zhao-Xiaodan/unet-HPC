# Bug Fix: Prediction Script Import Error

## Error Report

**Log File:** `Predict_Density_Analysis.o285422`
**Error Type:** `ImportError`
**Date:** 2025-10-12
**Status:** ✓ FIXED

---

## Error Details

### Error Message

```
Traceback (most recent call last):
  File "/scratch/phyzxi/unet-HPC/predict_with_density_analysis.py", line 36, in <module>
    from model_architectures import build_unet, build_resunet, build_attention_resunet
ImportError: cannot import name 'build_unet' from 'model_architectures'
```

### Root Cause

**Mismatch between import statement and actual function names in `model_architectures.py`**

The prediction script attempted to import:
```python
from model_architectures import build_unet, build_resunet, build_attention_resunet
```

But `model_architectures.py` actually provides:
```python
# Actual structure in model_architectures.py
MODEL_ARCHITECTURES = {
    'unet': UNet,
    'resunet': ResUNet,
    'attention_resunet': AttentionResUNet,
}

def get_model(model_name, input_shape, NUM_CLASSES=1, dropout_rate=0.3, batch_norm=True):
    """Factory function to create models by name"""
    ...
```

### Why This Happened

The hyperparameter search script (`hyperparam_search_comprehensive.py`) uses `get_model()` to create models dynamically, but the prediction script was written assuming individual builder functions existed.

---

## Fix Applied

### Change 1: Import Statement

**Before:**
```python
from model_architectures import build_unet, build_resunet, build_attention_resunet
from loss_functions import get_loss_function, jacard_coef
```

**After:**
```python
from model_architectures import get_model, UNet, ResUNet, AttentionResUNet
from loss_functions import get_loss_function, jacard_coef
```

### Change 2: Model Reconstruction (for weight-only loading)

**Before:**
```python
# Rebuild architecture
img_height = CONFIG['img_height']
img_width = CONFIG['img_width']
img_channels = CONFIG['img_channels']

if architecture == 'unet':
    model = build_unet(img_height, img_width, img_channels)
elif architecture == 'resunet':
    model = build_resunet(img_height, img_width, img_channels)
elif architecture == 'attention_resunet':
    model = build_attention_resunet(img_height, img_width, img_channels)
else:
    raise ValueError(f"Unknown architecture: {architecture}")
```

**After:**
```python
# Rebuild architecture using get_model()
img_height = CONFIG['img_height']
img_width = CONFIG['img_width']
img_channels = CONFIG['img_channels']
input_shape = (img_height, img_width, img_channels)

model = get_model(
    model_name=architecture,
    input_shape=input_shape,
    NUM_CLASSES=1,
    dropout_rate=0.3,
    batch_norm=True
)
```

---

## Verification

### Syntax Check

```bash
python -m py_compile predict_with_density_analysis.py
# ✓ No errors
```

### Expected Behavior After Fix

1. **Import succeeds** - `get_model()` and architecture functions correctly imported
2. **Model loading works** - Can load saved .hdf5 files with custom objects
3. **Fallback works** - If loading fails, rebuilds architecture and loads weights only

---

## Files Modified

1. **`predict_with_density_analysis.py`** (2 changes)
   - Line 36: Fixed imports
   - Lines 189-201: Fixed model reconstruction logic

---

## Testing Recommendation

```bash
# Re-submit job
qsub pbs_predict_density.sh

# Monitor
qstat -u $USER
tail -f Predict_Density_Analysis.o*

# Expected output should show:
# "Loading trained models..."
# "Searching for unet model..."
# "Found unet model: model_unet_bs8_dr0.3_*.hdf5"
# "✓ unet model loaded successfully"
# ... (similar for other architectures)
```

---

## Compatibility

The fixed script is now compatible with the actual `model_architectures.py` structure used throughout the project, ensuring consistency with:

- `hyperparam_search_comprehensive.py`
- `224_225_226_mito_segm_using_various_unet_models.py`
- All other scripts using `get_model()` factory function

---

## Lessons Learned

**Best Practice:** When creating new scripts that import from existing modules, always verify the actual function/class names in those modules rather than assuming based on convention or documentation.

**Project Pattern:** This codebase uses a **factory pattern** for model creation:
```python
# Correct usage
model = get_model('unet', input_shape=(512, 512, 1), dropout_rate=0.3)

# Incorrect (doesn't exist)
model = build_unet(512, 512, 1)
```

---

**Fix Date:** 2025-10-12
**Fixed By:** Claude Code (debugging Predict_Density_Analysis.o285422)
**Status:** Ready for re-testing
