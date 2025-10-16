# Attention UNet Density Analysis Setup - October 16, 2025

## Overview

Created density analysis pipeline for **Attention UNet** models, mirroring the UNet-only analysis with all recent bug fixes applied.

---

## Files Created

### 1. Analysis Script: `density_analysis_attention_unet_only.py`

**Based on:** `density_analysis_unet_only.py`

**Key modifications:**
- Updated `CONFIG['model_base_dir']` → `'./attention_unet_hyperparam_20251015_230149'`
- Updated `CONFIG['output_dir']` → `'density_analysis_attention_unet_only_{timestamp}'`
- Renamed model selection function: `find_best_unet_model()` → `find_best_attention_unet_model()`
- Updated model search path: `checkpoints/` → `models/`
- Updated glob pattern: `'unet_*'` → `'attention_unet_*'`
- Updated all print statements to reference "Attention UNet"

**Critical bug fix included:**
```python
# Line 333 - CLAHE+Otsu density calculation
density = 1.0 - (np.count_nonzero(binary_mask) / binary_mask.size)
```

This fix ensures we measure **beads** (dark regions) instead of **background** (white regions).

### 2. PBS Submission Script: `pbs_density_analysis_attention_unet_only.sh`

**Based on:** `pbs_density_analysis_unet_only.sh`

**Key modifications:**
- Job name: `Density_AttentionUNet_Only`
- Model directory: `./attention_unet_hyperparam_20251015_230149`
- Model subdirectory: `models/` (not `checkpoints/`)
- Script: `./density_analysis_attention_unet_only.py`
- Console log: `density_analysis_attention_unet_only_console_{timestamp}.log`
- Output directory pattern: `density_analysis_attention_unet_only_*`

---

## Model Selection Logic

### Directory Structure Expected (on HPC)

```
attention_unet_hyperparam_20251015_230149/
├── models/                           # Model files here (NOT checkpoints/)
│   ├── attention_unet_n_filters16_dropout0p0_batchnormTrue_lr0p0001/
│   │   └── best_model.keras
│   ├── attention_unet_n_filters16_dropout0p0_batchnormTrue_lr0p001/
│   │   └── best_model.keras
│   └── ... (27 total model combinations)
└── logs/                             # Training history CSVs
    ├── attention_unet_n_filters16_dropout0p0_batchnormTrue_lr0p0001_history.csv
    ├── attention_unet_n_filters16_dropout0p0_batchnormTrue_lr0p001_history.csv
    └── ...
```

### Model Selection Algorithm

```python
def find_best_attention_unet_model(base_dir):
    """Find best Attention UNet model from hyperparameter search."""

    # 1. Find all model directories in models/
    model_dirs = list((base_dir / 'models').glob('attention_unet_*'))

    # 2. For each model, parse hyperparameters from directory name
    #    Example: attention_unet_n_filters16_dropout0p1_batchnormTrue_lr0p0001
    #    Extracts: n_filters=16, dropout=0.1, batch_norm=True, lr=0.0001

    # 3. Find corresponding history CSV in logs/
    #    Pattern: attention_unet_n_filters{}_dropout{}_batchnorm{}_lr{}_history.csv

    # 4. Read validation IoU from history CSV
    #    Column: 'val_iou_score' (last epoch value)

    # 5. Select model with highest validation IoU

    return best_model_info
```

---

## Analysis Configuration

### Density Calculation Methods: 6 Total

1. **Threshold 0.2** - Most permissive
2. **Threshold 0.5** - Balanced (used for tile visualizations)
3. **Threshold 0.8** - Conservative
4. **Threshold 0.95** - Very conservative
5. **CLAHE+Otsu on predicted mask** - Denoised (FIXED: `1 - count_nonzero`)
6. **CLAHE+Otsu on original image** - Baseline

### Boxplot Configuration

**X-axis:** Categorical positions (evenly spaced)
```python
positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Not logarithmic!
```

**X-axis labels:** 1/10240x → 1/10x (low to high density)
```python
DILUTION_ORDER = [10240, 5120, 2560, 1280, 640, 320, 160, 80, 20, 10]
```

**Dilution ranges:**
- **Full range:** All 10 dilutions (1/10240x to 1/10x)
- **Low dilution range:** First 7 dilutions (1/10240x to 1/80x)

**Total boxplots:** 6 methods × 2 ranges = **12 boxplots**

### Tile Visualizations: 3 Panels

**5 representative tiles per test image**, each with 3 panels:

1. **Original tile** - Dark beads on bright background
2. **Inverted predicted mask (0.5)** - White beads (easier to compare)
3. **Inverted CLAHE+Otsu** - White beads (denoised)

**Directory:** `representative_tiles_3panel/`

**Figure size:** 15×25 inches (5 rows × 3 columns)

---

## Critical Bug Fix Applied

### Problem

Previous CLAHE+Otsu density calculation was **inverted** - it measured background instead of beads.

### Root Cause

After `cv2.threshold()` with `THRESH_BINARY_INV`:
- White pixels (255) = foreground (beads in CLAHE+Otsu output)
- Black pixels (0) = background

The function `np.count_nonzero(binary_mask)` counts **white pixels**.

But the semantic meaning was backwards - we were counting the wrong thing!

### Solution

```python
# OLD (WRONG - counts white pixels = background in our case)
density = np.count_nonzero(binary_mask) / binary_mask.size

# NEW (CORRECT - inverts to count dark regions = beads)
density = 1.0 - (np.count_nonzero(binary_mask) / binary_mask.size)
```

User feedback confirmed this: *"the value should be inverted one as 1 - the plotted value"*

---

## Output Structure

### Directory: `density_analysis_attention_unet_only_{timestamp}/`

```
density_analysis_attention_unet_only_{timestamp}/
├── density_results_tile_level.csv              # All 28 tiles × 11 images = 308 rows
├── density_results_image_summary.csv           # Image-level averages (11 rows)
├── EXPERIMENT_INFO.json                        # Model metadata
│
├── density_boxplot_full_range_threshold_0.2.png
├── density_boxplot_full_range_threshold_0.5.png
├── density_boxplot_full_range_threshold_0.8.png
├── density_boxplot_full_range_threshold_0.95.png
├── density_boxplot_full_range_claheotsu_on_pred.png
├── density_boxplot_full_range_claheotsu_on_original.png
│
├── density_boxplot_low_dilution_range_threshold_0.2.png
├── density_boxplot_low_dilution_range_threshold_0.5.png
├── density_boxplot_low_dilution_range_threshold_0.8.png
├── density_boxplot_low_dilution_range_threshold_0.95.png
├── density_boxplot_low_dilution_range_claheotsu_on_pred.png
├── density_boxplot_low_dilution_range_claheotsu_on_original.png
│
└── representative_tiles_3panel/
    ├── representative_tiles_{image_name}_3panel.png  (11 files, one per test image)
    └── ...
```

**Total files:** 12 boxplots + 11 tile visualizations + 3 data files = **26 files**

---

## CSV Column Structure

### `density_results_tile_level.csv` (308 rows)

```csv
image_name,dilution,tile_idx,tile_x,tile_y,density_threshold_0.2,density_threshold_0.5,density_threshold_0.8,density_threshold_0.95,density_clahe_otsu_pred,density_clahe_otsu_orig
10x_beads.tif,10,0,0,0,0.6482,0.5312,0.3784,0.2201,0.4892,0.5123
10x_beads.tif,10,1,512,0,0.6521,0.5389,0.3856,0.2267,0.4931,0.5201
...
```

**Columns:**
- `image_name`: Test image filename
- `dilution`: Dilution factor (10, 20, 80, ..., 10240)
- `tile_idx`: Tile index (0-27)
- `tile_x`, `tile_y`: Tile top-left coordinates
- `density_threshold_*`: Density from simple thresholding (4 columns)
- `density_clahe_otsu_pred`: CLAHE+Otsu on predicted mask (FIXED)
- `density_clahe_otsu_orig`: CLAHE+Otsu on original image

### `density_results_image_summary.csv` (11 rows)

```csv
image_name,dilution,mean_density_threshold_0.2,std_density_threshold_0.2,...
```

**Columns:** Same as tile-level, but with `mean_*` and `std_*` for each density method.

---

## Usage on HPC

### 1. Submit Job

```bash
qsub pbs_density_analysis_attention_unet_only.sh
```

### 2. Monitor Job

```bash
qstat -u $USER
```

### 3. Check Output

```bash
# Console log (real-time updates)
tail -f density_analysis_attention_unet_only_console_*.log

# PBS output
cat Density_AttentionUNet_Only.o*
```

### 4. View Results

```bash
# Find output directory
ls -dt density_analysis_attention_unet_only_* | head -1

# Check generated files
ls -lh density_analysis_attention_unet_only_*/
```

---

## Comparison with UNet Analysis

### Similarities

- ✅ Same 6 density calculation methods
- ✅ Same boxplot configuration (categorical x-axis)
- ✅ Same tile visualization layout (3 panels)
- ✅ Same output structure
- ✅ Same critical CLAHE+Otsu bug fix applied

### Differences

| Aspect | UNet | Attention UNet |
|--------|------|----------------|
| Model directory | `unet_hyperparam_20251015_224125` | `attention_unet_hyperparam_20251015_230149` |
| Model path | `checkpoints/` | `models/` |
| Model glob pattern | `unet_*` | `attention_unet_*` |
| Job name | `Density_UNet_Only` | `Density_AttentionUNet_Only` |
| Script | `density_analysis_unet_only.py` | `density_analysis_attention_unet_only.py` |
| PBS script | `pbs_density_analysis_unet_only.sh` | `pbs_density_analysis_attention_unet_only.sh` |
| Output directory | `density_analysis_unet_only_*` | `density_analysis_attention_unet_only_*` |

---

## Expected Results

### Boxplot Trends

**All methods should show:** Density increases from left (1/10240x) to right (1/10x)

**Threshold ordering (at each dilution):**
```
Threshold 0.2 > Threshold 0.5 > Threshold 0.8 > Threshold 0.95
```

**CLAHE+Otsu ordering:**
- CLAHE+Otsu on pred: Similar to Threshold 0.5 (but denoised, should be smoother)
- CLAHE+Otsu on orig: Baseline for comparison

### Tile Visualizations

**Panel 2 vs Panel 1:** Should show good correspondence between predicted beads and original beads

**Panel 3 vs Panel 2:** Should show denoising effect (less speckle noise, cleaner bead boundaries)

**White beads:** All panels show beads as **white** (inverted) for easy visual comparison

---

## Verification Checklist

Before submitting to HPC:

- [x] Python script compiles without syntax errors
- [x] Function `find_best_attention_unet_model()` correctly references `models/` directory
- [x] Main function calls `find_best_attention_unet_model()` (not `find_best_unet_model()`)
- [x] PBS script references correct model directory
- [x] PBS script references correct Python script
- [x] CLAHE+Otsu density fix applied (`1 - count_nonzero`)
- [x] All print statements reference "Attention UNet"
- [x] Output directory pattern matches script name

---

## Files Ready for HPC Deployment

1. ✅ `density_analysis_attention_unet_only.py` - Analysis script
2. ✅ `pbs_density_analysis_attention_unet_only.sh` - PBS submission script
3. ✅ `ATTENTION_UNET_DENSITY_ANALYSIS_SETUP.md` - This documentation

---

## Next Steps

1. **Transfer files to HPC** (if working locally):
   ```bash
   scp density_analysis_attention_unet_only.py phyzxi@nus.edu.sg:/home/svu/phyzxi/scratch/unet-HPC/
   scp pbs_density_analysis_attention_unet_only.sh phyzxi@nus.edu.sg:/home/svu/phyzxi/scratch/unet-HPC/
   ```

2. **Verify model directory exists on HPC**:
   ```bash
   ssh phyzxi@nus.edu.sg "ls -la /home/svu/phyzxi/scratch/unet-HPC/attention_unet_hyperparam_20251015_230149/models/"
   ```

3. **Submit job**:
   ```bash
   qsub pbs_density_analysis_attention_unet_only.sh
   ```

4. **Compare results with UNet**:
   - Compare boxplot trends across architectures
   - Check if Attention UNet shows better/worse density estimation
   - Verify CLAHE+Otsu fix produces sensible results

---

**Date:** October 16, 2025
**Status:** ✅ Ready for HPC deployment
**Model:** Attention UNet (best from 27 hyperparameter combinations)
**Bug fixes:** All applied, including critical CLAHE+Otsu density inversion
