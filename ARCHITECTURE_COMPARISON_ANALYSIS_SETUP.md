# Architecture Comparison Density Analysis - Setup and Usage

**Date:** October 17, 2025
**Status:** ✅ Ready for HPC deployment
**Purpose:** Compare UNet, Attention UNet, and Attention ResUNet performance on test images

---

## Overview

This analysis compares all three architectures side-by-side on the same test images using their respective best models from hyperparameter searches.

### Key Features

1. ✅ **Side-by-side architecture comparison** in combined boxplots
2. ✅ **4-panel tile visualizations** (Original + 3 architecture predictions)
3. ✅ **6 density methods** analyzed for each architecture
4. ✅ **Automatic best model selection** from each hyperparameter search
5. ✅ **Statistical comparison** with mean/median/std per architecture

---

## Files

### Python Script
**File:** `density_analysis_architecture_comparison.py`

**Key Functions:**
- `find_best_model()`: Selects best model from each architecture's hyperparameter search
- `load_model()`: Loads all 3 models with appropriate custom objects
- `predict_on_test_images_all_architectures()`: Runs prediction with all 3 models
- `create_combined_boxplot()`: Generates 3-architecture comparison boxplots
- `create_4panel_tile_visualizations()`: Creates Original + 3 predictions visualizations

**Model Directories Used:**
- UNet: `./unet_hyperparam_20251013_034824`
- Attention UNet: `./attention_unet_hyperparam_20251015_230149`
- Attention ResUNet: `./attention_resunet_hyperparam_20251015_235542`

### PBS Script
**File:** `pbs_density_analysis_architecture_comparison.sh`

**PBS Configuration:**
- Walltime: 6 hours (loading 3 models + 3× predictions)
- Resources: 1 GPU node (A40/A100), 36 CPUs, 240 GB RAM
- Job name: `Density_Arch_Comparison`
- Email notifications: Start, abort, end

---

## Output Structure

### Generated Files

**CSV Files:**
1. `architecture_comparison_tile_level.csv`
   - Tile-level densities for all architectures
   - Columns: image, dilution, tile_idx, architecture, density_*
   - Rows: 224 tiles × 3 architectures = 672 rows

2. `architecture_comparison_image_summary.csv`
   - Image-level statistics grouped by architecture
   - Columns: image, dilution, architecture, mean/median/std for each method
   - Rows: 8 images × 3 architectures = 24 rows

**Visualizations:**
1. **Combined Boxplots (6 files):**
   - `density_boxplot_comparison_threshold_0p2.png`
   - `density_boxplot_comparison_threshold_0p5.png`
   - `density_boxplot_comparison_threshold_0p8.png`
   - `density_boxplot_comparison_threshold_0p95.png`
   - `density_boxplot_comparison_claheotsu_on_pred.png`
   - `density_boxplot_comparison_claheotsu_on_original.png`

   Each boxplot shows:
   - X-axis: 8 dilution levels (10240x to 80x)
   - Y-axis: Density (log scale)
   - 3 box groups per dilution (one per architecture)
   - Color-coded: Blue (UNet), Red (Attention UNet), Green (Attention ResUNet)

2. **4-Panel Tile Visualizations:**
   - Directory: `representative_tiles_4panel/`
   - Format: `tiles_4panel_{dilution}_{image_name}.png`
   - 5 representative tiles per image × 8 images = 40 files

   Each 4-panel visualization shows:
   - **Panel 1:** Original tile (dark beads)
   - **Panel 2:** UNet prediction (inverted - white beads)
   - **Panel 3:** Attention UNet prediction (inverted - white beads)
   - **Panel 4:** Attention ResUNet prediction (inverted - white beads)

---

## Expected Best Models

Based on previous hyperparameter searches:

| Architecture | Expected Best Config | Expected Val IoU |
|--------------|---------------------|------------------|
| **UNet** | 32 filters, dropout=0.2, LR=0.001 | ~0.467 |
| **Attention UNet** | 32 filters, dropout=0.3, LR=0.003 | ~0.476 |
| **Attention ResUNet** | 32 filters, dropout=0.1, LR=0.001 | **~0.504** |

**Ranking:** Attention ResUNet > Attention UNet > UNet

---

## Batch Size Configuration

**Conservative batch_size=2** used for all architectures to accommodate the largest model (Attention ResUNet).

**Why batch_size=2:**
- Attention ResUNet requires more memory due to residual connections + attention
- Using same batch size for all ensures fair comparison
- Slightly slower but avoids OOM crashes

---

## Usage Instructions

### 1. Prerequisites

**On HPC, verify all model directories exist:**
```bash
ssh HPC
cd /home/svu/phyzxi/scratch/unet-HPC

ls -d unet_hyperparam_20251013_034824/checkpoints
ls -d attention_unet_hyperparam_20251015_230149/checkpoints
ls -d attention_resunet_hyperparam_20251015_235542/checkpoints
ls -d test_images
```

**All three hyperparameter searches must be completed.**

### 2. Submit Job

```bash
qsub pbs_density_analysis_architecture_comparison.sh
```

**Expected output:**
```
293XXX.stdct-mgmt-02
```

### 3. Monitor Job

**Check queue:**
```bash
qstat -u phyzxi
```

**Monitor console output (live):**
```bash
tail -f density_analysis_architecture_comparison_console_YYYYMMDD_HHMMSS.log
```

**Check job output (after completion):**
```bash
cat Density_Arch_Comparison.oJOBID
```

### 4. Expected Runtime

**Total Walltime:** ~4-5 hours

**Breakdown:**
- Model selection (3 architectures): ~1 minute
- Model loading (3 models): ~1-2 minutes
- Prediction on 8 images:
  - UNet: 224 tiles @ batch=2 → ~20 minutes
  - Attention UNet: 224 tiles @ batch=2 → ~25 minutes
  - Attention ResUNet: 224 tiles @ batch=2 → ~30 minutes
  - **Total prediction: ~75 minutes** (sequential)
- Density calculations (3 × 224 tiles): ~30 minutes
- Visualization generation:
  - 6 combined boxplots: ~10 minutes
  - 40 4-panel tile images: ~30 minutes
  - **Total visualization: ~40 minutes**

**Total: ~2.5 hours** (well within 6-hour walltime)

---

## Visualization Details

### Combined Boxplots

**Layout:**
- Each dilution level has 3 boxes side-by-side
- Box width: 0.25 (narrow to fit 3 architectures)
- Colors:
  - **Blue (#3498db):** UNet
  - **Red (#e74c3c):** Attention UNet
  - **Green (#2ecc71):** Attention ResUNet
- Y-axis: Log scale (handles wide density range)
- Legend: Top-left corner

**Interpretation:**
- **Higher boxes** = higher density predictions
- **Taller boxes** = more variability across tiles
- **Overlapping boxes** = similar performance
- **Separated boxes** = different sensitivity

### 4-Panel Tile Visualizations

**Format:**
- 5 rows (5 representative tiles per image)
- 4 columns (Original + 3 architectures)
- All predictions inverted (white beads on black background)

**Purpose:**
- Visual comparison of prediction quality
- Identify which architecture handles different dilutions better
- Spot systematic biases (over-prediction, under-prediction)

**What to Look For:**
- **Low dilution (10240x, 5120x):** Which architecture detects sparse beads best?
- **High dilution (80x, 160x):** Which architecture handles crowded beads?
- **Edge cases:** Background artifacts, clumping, out-of-focus regions

---

## Analysis Workflow

### Step 1: Load Best Models

Script automatically:
1. Searches each checkpoint directory for all trained models
2. Reads `training_history.csv` for each model
3. Selects model with highest `val_jacard_coef` (validation IoU)
4. Loads `.keras` model file with all custom objects

**Custom objects included:**
- Loss functions: `combined_dice_focal_loss`, `focal_loss`, `BinaryFocalLoss`
- Metrics: `jacard_coef`, `dice_coef`
- Custom layers: `RepeatElements` (for Attention UNet/ResUNet)

### Step 2: Predict on Test Images

For each test image:
1. Extract dilution from filename
2. Load image as RGB float32
3. Extract 28 non-overlapping 512×512 tiles
4. **Predict with all 3 architectures** (sequential)
5. Calculate 6 density methods per architecture per tile

**Total predictions:**
- 8 images × 28 tiles × 3 architectures = **672 predictions**

### Step 3: Calculate Densities

**6 methods applied per tile per architecture:**
1. Threshold 0.2
2. Threshold 0.5
3. Threshold 0.8
4. Threshold 0.95
5. CLAHE+Otsu on predicted mask (denoised)
6. CLAHE+Otsu on original image (baseline)

**Total density values:** 672 tiles × 6 methods = 4,032 values

### Step 4: Generate Visualizations

**Combined boxplots:**
- Group by dilution and architecture
- Calculate boxplot statistics (median, Q1, Q3, whiskers)
- Plot 3 architectures side-by-side for each dilution
- Repeat for all 6 density methods

**4-panel tiles:**
- Select 5 representative tiles per image (first 5)
- Create 5-row × 4-column grid
- Invert predictions for visualization
- Save as high-resolution PNG (300 DPI)

---

## Troubleshooting

### Job Fails: Model Directory Not Found

**Error:**
```
ERROR: UNet model directory not found: ./unet_hyperparam_20251013_034824
```

**Cause:** Model directory doesn't exist or wrong path

**Fix:**
1. Check directory name:
   ```bash
   ls -d unet_hyperparam_*
   ls -d attention_unet_hyperparam_*
   ls -d attention_resunet_hyperparam_*
   ```
2. Update `CONFIG` in Python script if directory names differ

### Job Fails: No Models Found

**Error:**
```
ERROR: No best_model.keras files found in .../checkpoints
```

**Cause:** Hyperparameter search incomplete or failed

**Fix:** Complete hyperparameter search for missing architecture

### GPU Out of Memory

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Cause:** batch_size=2 still too large (unlikely)

**Fix:** Reduce to batch_size=1 in CONFIG:
```python
'batch_size': 1,
```

**Impact:** ~2× slower prediction (~5 hours total)

### Different Number of Test Images

**Note:** Script expects 8 test images. If you have more or fewer:
- Script will still work (processes all .tif/.tiff files)
- Runtime scales linearly with image count
- Adjust walltime if needed: ~20 minutes per image

---

## Expected Results

### Performance Ranking (Based on Val IoU)

**Expected order:** Attention ResUNet > Attention UNet > UNet

**Validation IoU:**
- Attention ResUNet: ~0.504 ✓ (highest)
- Attention UNet: ~0.476
- UNet: ~0.467

### Density Trends

**Expected behavior across dilutions:**

| Dilution | Expected Density | Difficulty |
|----------|-----------------|------------|
| 10240x | Very low (~0.001) | Hardest (sparse beads) |
| 5120x | Low (~0.005) | Hard |
| 2560x | Low-medium (~0.01) | Medium |
| 1280x | Medium (~0.04) | Medium |
| 640x | Medium-high (~0.12) | Medium-easy |
| 320x | High (~0.23) | Easy |
| 160x | Very high (~0.56) | Easy |
| 80x | Extremely high (~0.79) | Easy |

**Architecture comparison:**
- **Attention ResUNet:** Expected to perform best across all dilutions
- **Attention UNet:** Should show improvement over UNet, especially at low dilutions
- **UNet:** Baseline performance, may struggle at very low dilutions

### Visual Assessment

**4-panel tiles - What to expect:**

**Low dilution (10240x, 5120x):**
- UNet: May miss some sparse beads (false negatives)
- Attention UNet: Better detection, fewer misses
- Attention ResUNet: Best detection, cleanest segmentation

**High dilution (80x, 160x):**
- All architectures should perform well
- Minor differences in boundary smoothness
- Attention ResUNet may have sharper edges

---

## Comparison with Individual Analysis

### Differences from Single-Architecture Scripts

**This comparison script:**
- ✅ Loads all 3 models simultaneously
- ✅ Runs all 3 predictions on same tiles
- ✅ Generates combined visualizations
- ✅ Provides direct side-by-side comparison

**Individual architecture scripts:**
- Only analyze one architecture
- Separate boxplots (not comparable)
- 3-panel tiles (Original + predicted + CLAHE)

**Use cases:**
- **Individual scripts:** Detailed analysis of single architecture
- **Comparison script:** Understand relative performance, choose best architecture

---

## Statistical Analysis (Post-Processing)

After the analysis completes, you can perform additional statistical tests:

### Paired t-test (Architecture Comparison)

```python
import pandas as pd
from scipy import stats

# Load results
df = pd.read_csv('architecture_comparison_tile_level.csv')

# Compare Attention ResUNet vs UNet on threshold 0.5
resunet = df[df['architecture'] == 'Attention_ResUNet']['density_threshold_0.5']
unet = df[df['architecture'] == 'UNet']['density_threshold_0.5']

t_stat, p_value = stats.ttest_rel(resunet, unet)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4e}")
```

### Correlation with Dilution

```python
# Check if density correlates with dilution
for arch in ['UNet', 'Attention_UNet', 'Attention_ResUNet']:
    df_arch = df[df['architecture'] == arch]
    corr = df_arch['dilution'].corr(df_arch['density_threshold_0.5'])
    print(f"{arch}: correlation = {corr:.4f}")
```

### Performance vs. Dilution Level

```python
# Compare architectures at each dilution
summary = df.groupby(['dilution', 'architecture'])['density_threshold_0.5'].mean().unstack()
print(summary)
```

---

## Next Steps

### 1. After Comparison Completes

**Review combined boxplots:**
- Identify which architecture performs best overall
- Check if performance advantage is consistent across dilutions
- Assess variability (box height) for robustness

**Review 4-panel tiles:**
- Visually confirm boxplot findings
- Identify systematic biases
- Check for architectural artifacts

### 2. Choose Best Architecture

Based on:
- **Validation IoU** (quantitative metric)
- **Density correlation** with expected dilution series
- **Visual quality** from tile visualizations
- **Robustness** (low variability across tiles)

**Expected winner:** Attention ResUNet (highest IoU, best across dilutions)

### 3. Use for Downstream Analysis

Once best architecture is identified:
- Use its predictions for scientific analysis
- Report architecture choice in methods section
- Include comparison boxplots in supplementary materials

---

## Publication-Ready Figures

The comparison boxplots are designed for publication:
- **Resolution:** 300 DPI (print quality)
- **Size:** 16×8 inches (suitable for full-width figure)
- **Font sizes:** 16pt title, 14pt labels, 12pt legend
- **Colors:** Colorblind-friendly palette (blue, red, green)
- **Format:** PNG with transparency support

**To convert to vector format for publication:**
```python
# In the script, change:
plt.savefig(output_path, dpi=300, bbox_inches='tight')
# To:
plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', format='pdf')
```

---

## Summary

**Purpose:** Compare UNet, Attention UNet, and Attention ResUNet

**Input:**
- Best models from 3 hyperparameter searches
- 8 test images across dilution series

**Output:**
- 6 combined boxplots (3 architectures per plot)
- 40 4-panel tile visualizations
- 2 CSV files with comparative statistics

**Runtime:** ~2.5-4 hours

**Expected Result:** Attention ResUNet shows best performance across all dilutions

**Deployment Command:**
```bash
qsub pbs_density_analysis_architecture_comparison.sh
```

---

**Document Date:** October 17, 2025
**Author:** Automated by Claude Code
**Version:** 1.0
**Status:** ✅ Production Ready
