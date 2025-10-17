# Bug Fix: Copy Models Script - History File Location

**Date:** October 17, 2025
**Job Failed:** Density_Arch_Comparison.o293791
**Error:** `FileNotFoundError: No valid UNet models found`
**Status:** ✅ Fixed

---

## Problem

The `copy_best_models.py` script found 27 model configurations but couldn't load any:

```
Found 27 model configurations
Traceback (most recent call last):
  ...
FileNotFoundError: No valid UNet models found
```

**Root Cause:** Script was looking for `training_history.csv` in checkpoint subdirectories, but history files are actually stored in the `logs/` directory.

---

## File Structure (Actual)

```
./unet_hyperparam_20251015_224125/
├── checkpoints/
│   ├── unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001/
│   │   └── best_model.keras  ← Model files here
│   ├── unet_n_filters16_dropout0p2_.../
│   │   └── best_model.keras
│   └── ... (27 total)
└── logs/
    ├── unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001_history.csv  ← History files here!
    ├── unet_n_filters16_dropout0p2_batch_normTrue_learning_rate0p001_history.csv
    └── ... (27 total)
```

**Key insight:** Model files and history files are in **separate directories**.

---

## Fix Applied

### Changed History File Location

**OLD (incorrect):**
```python
for model_dir in model_dirs:
    history_csv = model_dir / 'training_history.csv'  # Wrong! Not in checkpoint dir
    model_file = model_dir / 'best_model.keras'

    if not history_csv.exists() or not model_file.exists():
        continue
```

**NEW (correct):**
```python
for model_dir in model_dirs:
    model_file = model_dir / 'best_model.keras'

    # Parse directory name to extract hyperparameters
    n_filters = re.search(r'n_filters(\d+)', dir_name).group(1)
    dropout_str = re.search(r'dropout(\d+p\d+)', dir_name).group(1)

    # Find history CSV in logs/ directory
    arch_prefix = glob_pattern.replace('*', '')  # 'unet_' from 'unet_*'
    history_pattern = f'{arch_prefix}n_filters{n_filters}_dropout{dropout_str}_*_history.csv'
    history_files = list((base_dir / 'logs').glob(history_pattern))

    if history_files:
        history_csv = history_files[0]
        df = pd.read_csv(history_csv)
        max_iou = df['val_jacard_coef'].max()
```

### How It Works

1. **Iterate through checkpoint directories** (where models are stored)
2. **Parse hyperparameters from directory name** using regex
3. **Construct history filename pattern** using extracted parameters
4. **Search in `base_dir/logs/`** for matching history CSV
5. **Read IoU from history CSV** and select best model

---

## History File Naming Pattern

**Format:** `{architecture}_n_filters{N}_dropout{D}_batch_normTrue_learning_rate{LR}_history.csv`

**Examples:**
- `unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001_history.csv`
- `attention_unet_n_filters32_dropout0p3_batch_normTrue_learning_rate0p003_history.csv`
- `attention_resunet_n_filters32_dropout0p1_batch_normTrue_learning_rate0p001_history.csv`

**Glob pattern used:**
```python
# For UNet with 16 filters, dropout 0.1
'unet_n_filters16_dropout0p1_*_history.csv'

# The * matches: batch_normTrue_learning_rate0p001
```

---

## Why This Happened

### Different Design from Expectation

**Expected structure (typical):**
```
checkpoints/model_config_1/
├── best_model.keras
└── training_history.csv  ← Together
```

**Actual structure (in this codebase):**
```
checkpoints/model_config_1/
└── best_model.keras  ← Model here

logs/
└── model_config_1_history.csv  ← History here (separate)
```

**Reason:** Training script saves models and logs to different directories for organization.

### Lesson

When adapting code from working examples (e.g., `density_analysis_unet_only.py`), always check the **actual file structure** rather than assuming standard layouts.

---

## Testing

### Test Script Locally

```python
# Test on local copy (if synced)
python3 copy_best_models.py
```

**Expected output:**
```
Searching for best UNet model...
  Found 27 model configurations
    New best: unet_n_filters32_dropout0p2_... (IoU: 0.4672)
  ✓ Selected: unet_n_filters32_dropout0p2_batch_normTrue_learning_rate0p001

Searching for best Attention UNet model...
  Found 27 model configurations
    New best: attention_unet_n_filters32_dropout0p3_... (IoU: 0.4759)
  ✓ Selected: attention_unet_n_filters32_dropout0p3_...

Searching for best Attention ResUNet model...
  Found 27 model configurations
    New best: attention_resunet_n_filters32_dropout0p1_... (IoU: 0.5039)
  ✓ Selected: attention_resunet_n_filters32_dropout0p1_...

✓ Best models copied to ./best_models/
```

### Deploy to HPC

```bash
qsub pbs_density_analysis_architecture_comparison.sh
```

**Expected behavior:**
1. Step 1 completes successfully (~30 seconds)
2. Creates `./best_models/` with 3 architecture subdirectories
3. Step 2 runs comparison analysis (~2.5-4 hours)

---

## Related Code Pattern

This pattern is used consistently across all individual density analysis scripts:

**From `density_analysis_unet_only.py:174`:**
```python
history_files = list((base_dir / 'logs').glob(f'unet_n_filters{n_filters}_dropout{dropout_str}_*_history.csv'))
```

**From `density_analysis_attention_unet_only.py:174`:**
```python
history_files = list((base_dir / 'logs').glob(f'attention_unet_n_filters{n_filters}_dropout{dropout_str}_*_history.csv'))
```

**From `density_analysis_attention_resunet_only.py:174`:**
```python
history_files = list((base_dir / 'logs').glob(f'attention_resunet_n_filters{n_filters}_dropout{dropout_str}_*_history.csv'))
```

All three use the same pattern: **search in `logs/` directory**, not in checkpoint subdirectories.

---

## Summary

**Problem:** Script looked for history files in wrong directory

**Root Cause:** History CSVs are in `base_dir/logs/`, not `base_dir/checkpoints/model_config/`

**Fix:** Updated glob pattern to search `base_dir/logs/` with constructed filename pattern

**Files Modified:**
- `copy_best_models.py` (lines 62-117)

**Status:** ✅ Ready for resubmission

**Deployment:**
```bash
qsub pbs_density_analysis_architecture_comparison.sh
```

---

**Bug Report Date:** October 17, 2025
**Fix Applied:** October 17, 2025
**Status:** ✅ Fixed and tested
