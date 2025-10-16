# UNet-Only Density Analysis - Updated Version

## Overview

This analysis updates the previous density analysis (`density_analysis_xukuang_20251015_142119`) with improvements from `DENSITY_ANALYSIS_UPDATE_SUMMARY.md`, specifically for the **best UNet model** from the hyperparameter search.

**Why UNet-only?** The Attention UNet and Attention ResUNet models are currently being re-trained with the hyperparameter search, so they are not ready for multi-model comparison yet.

## Key Improvements

### 1. ✅ Tile-Level Density Tracking
**Previous:** Only image-level mean densities
**Now:** ALL individual tile densities saved (28 tiles per image)

**Output:**
- `density_results_tile_level.csv` - Every tile with its density value
- `density_results_image_summary.csv` - Aggregated statistics per image

**Example tile-level data:**
```csv
image,dilution,dilution_label,tile_idx,position_y,position_x,density
10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,0.4523
10x_2025-05-15_02-05-00.tif,10,10x,1,0,512,0.4601
10x_2025-05-15_02-05-00.tif,10,10x,2,0,1024,0.4389
...
```

### 2. ✅ Updated Boxplot Style
**Previous:** Linear x-axis, simple categorical labels
**Now:** Log-scale x-axis with 1/Dilution factor (matching reference image)

**Key styling changes:**
- **X-axis:** 1/Dilution Factor (1/10240, 1/5120, ..., 1/10)
- **X-scale:** Logarithmic
- **Y-axis:** Foreground Percentage
- **Y-scale:** Logarithmic (0.002 to 1.5)
- **Colors:** Light blue boxes (#5FA3D9) with orange median lines (#FF8C42)

### 3. ✅ Two Boxplot Ranges
**NEW:** Generate two separate boxplots for different analysis needs

**Full Range (1/10 to 1/10240):**
- Shows complete dilution series
- Useful for overall trends
- File: `density_boxplot_full_range.png`

**Low Dilution Range (1/80 to 1/10240):**
- Focuses on high dilution factors only
- Better resolution for low-density samples
- File: `density_boxplot_low_dilution_range.png`

### 4. ✅ 3-Panel Tile Visualizations
**NEW:** Generate representative tile visualizations for each test image

**Layout per image:**
- 5 representative tiles (spanning density range: min, Q1, median, Q3, max)
- 3 panels per tile:
  - **Panel 1:** Original tile
  - **Panel 2:** Predicted mask (black background, white foreground)
  - **Panel 3:** Inverted mask (white beads on black background)

**Why inverted mask?**
Since beads are **black** in the original images, the inverted mask (white beads on black) makes it much easier to visually compare predictions with the original.

**Output:** `representative_tiles_3panel/` directory with one PNG per test image

### 5. ✅ Best Model Selection
**Automatic:** Script finds the best UNet model from hyperparameter search based on validation IoU

**Search location:** `unet_hyperparam_20251015_224125/checkpoints/`

**Selection criteria:** Highest `val_jacard_coef` from training history

## Files

### Scripts
- **`density_analysis_unet_only.py`** - Main analysis script
- **`pbs_density_analysis_unet_only.sh`** - PBS submission script

### Previous Analysis (for reference)
- **`density_analysis_xukuang_20251015_142119/`** - Previous analysis directory
  - Contains tile visualizations that can be reused
  - Original boxplot (now superseded by new version)

## Usage

### 1. Submit to HPC

```bash
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_density_analysis_unet_only.sh
```

### 2. Monitor Progress

```bash
# Check job status
qstat -u $USER

# Watch log output
tail -f Density_UNet_Only.o<JOBID>
```

### 3. Check Results

```bash
# Find output directory
ls -ld density_analysis_unet_only_*

# View files
ls -lh density_analysis_unet_only_20251016_*/
```

## Expected Output

```
density_analysis_unet_only_20251016_HHMMSS/
├── density_results_tile_level.csv          # All tile densities
├── density_results_image_summary.csv       # Per-image statistics
├── density_boxplot_full_range.png          # Boxplot: 1/10 - 1/10240
├── density_boxplot_low_dilution_range.png  # Boxplot: 1/80 - 1/10240
├── EXPERIMENT_INFO.json                    # Metadata
└── representative_tiles_3panel/            # 3-panel tile visualizations
    ├── tiles_3panel_10x_<image_name>.png
    ├── tiles_3panel_20x_<image_name>.png
    ├── tiles_3panel_80x_<image_name>.png
    ├── tiles_3panel_160x_<image_name>.png
    ├── tiles_3panel_320x_<image_name>.png
    ├── tiles_3panel_640x_<image_name>.png
    ├── tiles_3panel_1280x_<image_name>.png
    ├── tiles_3panel_2560x_<image_name>.png
    ├── tiles_3panel_5120x_<image_name>.png
    └── tiles_3panel_10240x_<image_name>.png
```

### Output Files Description

#### `density_results_tile_level.csv`
Contains density for every tile:
- **Columns:** image, dilution, dilution_label, tile_idx, position_y, position_x, density
- **Rows:** n_images × 28 tiles = ~308 rows (for 11 images)
- **Usage:** Detailed tile-level analysis, variance studies

#### `density_results_image_summary.csv`
Contains aggregated statistics per image:
- **Columns:** image, dilution, dilution_label, n_tiles, mean_density, median_density, std_density, min_density, max_density
- **Rows:** n_images = 11 rows
- **Usage:** Quick overview, image-level comparisons

#### `density_boxplot_full_range.png`
Full dilution series (10x to 10240x):
- **Format:** PNG, 300 DPI
- **Size:** ~14×8 inches
- **Style:** Log-log scale, blue boxes, orange medians
- **Usage:** Overall trend visualization

#### `density_boxplot_low_dilution_range.png`
Low dilution subset (80x to 10240x):
- **Format:** PNG, 300 DPI
- **Size:** ~14×8 inches
- **Style:** Log-log scale, blue boxes, orange medians
- **Usage:** Focus on high-dilution samples

#### `EXPERIMENT_INFO.json`
Metadata about the analysis:
- Timestamp
- Best model info (hyperparameters, validation IoU)
- Configuration parameters
- Total images and tiles processed

#### `representative_tiles_3panel/`
3-panel tile visualizations (one PNG per test image):
- **Format:** PNG, 300 DPI, 15×25 inches (5 rows × 3 columns)
- **Rows:** 5 representative tiles (min, Q1, median, Q3, max density)
- **Columns:**
  - Column 1: Original tile
  - Column 2: Predicted mask
  - Column 3: Inverted mask (white beads on black)
- **Usage:** Visual quality assessment, easy comparison with black beads in originals

## Analysis Workflow

### 1. Model Selection
```
Script automatically:
1. Scans unet_hyperparam_20251015_224125/checkpoints/
2. Finds all unet_* subdirectories
3. Reads training history CSVs
4. Selects model with highest val_jacard_coef
5. Prints selected model info
```

### 2. Tile Extraction
```
For each test image:
1. Load image as RGB
2. Extract 28 non-overlapping 512×512 tiles
3. Normalize to [0, 1]
4. Predict with best UNet model
5. Calculate foreground density per tile
6. Save tile-level results
```

### 3. Visualization Generation
```
A. Two boxplots created:
1. Full range (1/10 to 1/10240)
   - 10 dilution levels
   - Log-log scale
   - Blue boxes, orange medians

2. Low dilution range (1/80 to 1/10240)
   - 8 dilution levels (excludes 10x and 20x)
   - Log-log scale
   - Blue boxes, orange medians

B. 3-panel tile visualizations:
For each test image:
1. Sort all 28 tiles by density
2. Select 5 representative tiles (min, Q1, median, Q3, max)
3. Create 5×3 grid:
   - Row: Each representative tile
   - Col 1: Original tile
   - Col 2: Predicted mask
   - Col 3: Inverted mask (white beads)
4. Save as tiles_3panel_{dilution}_{image}.png
```

## Comparison with Previous Analysis

| Feature | Previous (xukuang) | New (UNet-only) |
|---------|-------------------|-----------------|
| **Tile-level data** | ❌ No | ✅ Yes (28 per image) |
| **Boxplot style** | Linear scale | ✅ Log-log scale with 1/dilution |
| **Dilution ranges** | Single plot | ✅ Two plots (full + low dilution) |
| **Model selection** | Manual (xukuang params) | ✅ Automatic (best from hyperparam search) |
| **Tile visualizations** | 2-panel (original + prediction) | ✅ 3-panel (original + mask + inverted) |
| **Tiles per image** | 5 per dilution level | ✅ 5 per test image |
| **Inverted mask** | ❌ No | ✅ Yes (white beads for comparison) |
| **Colors** | Seaborn default | ✅ Blue boxes + orange medians |

## Statistical Analysis Opportunities

With tile-level data, you can now analyze:

### Variance Analysis
```python
import pandas as pd

df = pd.read_csv('density_results_tile_level.csv')

# Within-image variance
variance_by_image = df.groupby('image')['density'].std()

# Between-dilution variance
variance_by_dilution = df.groupby('dilution')['density'].std()

# Overall variance
total_variance = df['density'].std()
```

### Dilution-Density Relationship
```python
import numpy as np
from scipy.stats import pearsonr

# Calculate correlation
inv_dilution = 1.0 / df['dilution']
correlation, p_value = pearsonr(inv_dilution, df['density'])

print(f"Correlation (1/dilution vs density): {correlation:.4f} (p={p_value:.4e})")
```

### Tile Position Effects
```python
# Check if edge tiles differ from center tiles
df['is_edge'] = (df['position_y'] == 0) | (df['position_x'] == 0)

edge_mean = df[df['is_edge']]['density'].mean()
center_mean = df[~df['is_edge']]['density'].mean()

print(f"Edge tiles: {edge_mean:.4f}")
print(f"Center tiles: {center_mean:.4f}")
```

## Model Information

The best UNet model is automatically selected from the hyperparameter search grid:

**Hyperparameter Grid:**
- `n_filters`: [16, 32, 64]
- `dropout`: [0.1, 0.2, 0.3]
- `batch_norm`: [True]
- `learning_rate`: [0.001, 0.003, 0.005]

**Total combinations:** 27 models

**Selection metric:** Maximum validation IoU (Jaccard coefficient)

**Expected best model characteristics** (as of Oct 16, 2025):
- Best Val IoU: ~0.4627 (from combination 20/27)
- Training: 100 epochs
- Loss: BinaryFocalLoss(gamma=2, alpha=0.25)
- Image format: 512×512 RGB

## Reusing Previous Tile Visualizations

The previous analysis (`density_analysis_xukuang_20251015_142119`) generated tile visualizations that are still valid:

```
density_analysis_xukuang_20251015_142119/
└── representative_tiles/
    ├── tiles_10x.png
    ├── tiles_20x.png
    ├── tiles_80x.png
    ├── tiles_160x.png
    ├── tiles_320x.png
    ├── tiles_640x.png
    ├── tiles_1280x.png
    ├── tiles_2560x.png
    ├── tiles_5120x.png
    └── tiles_10240x.png
```

**Why reuse?**
- Same test images
- Same tile extraction method
- Saves computation time (~30-60 minutes)
- Focus on the new improvements (tile-level data + boxplots)

**When to regenerate:**
- If you want to compare predictions from different models
- If test images change
- If you need 4-panel comparisons (original + 3 models)

## Troubleshooting

### No model files found
```bash
# Check if UNet training completed
ls -la unet_hyperparam_20251015_224125/checkpoints/

# You should see directories like:
# unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001/
# unet_n_filters16_dropout0p2_batch_normTrue_learning_rate0p001/
# ...

# Each should contain best_model.keras
find unet_hyperparam_20251015_224125/checkpoints -name "best_model.keras"
```

### Model selection fails
```bash
# Check if training history CSVs exist
ls -la unet_hyperparam_20251015_224125/logs/*.csv

# Each model should have a corresponding history file
```

### Boxplot looks wrong
- Verify x-axis shows 1/10240 on left, 1/10 on right (full range)
- Verify x-axis shows 1/10240 on left, 1/80 on right (low dilution range)
- Check that both axes use log scale
- Ensure dilution order is correct in script

### Job fails with memory error
```bash
# Increase memory allocation in PBS script
#PBS -l select=1:ncpus=36:mpiprocs=1:ompthreads=36:ngpus=1:mem=360gb
```

## Future Enhancements

Once Attention UNet and Attention ResUNet complete training:

### 1. Multi-Model Comparison
```bash
# Use the full multi-model script
qsub pbs_density_analysis_xukuang.sh
```

This will generate:
- Side-by-side boxplots for all 3 models
- 4-panel tile comparisons (Original + UNet + Attention UNet + Attention ResUNet)
- Model agreement analysis

### 2. Statistical Comparison
```python
# Compare models statistically
from scipy.stats import ttest_rel

# Paired t-test (same tiles predicted by different models)
t_stat, p_value = ttest_rel(unet_densities, attention_unet_densities)
```

### 3. Best Model Selection Across Architectures
```python
# After all 81 models trained (27 per architecture × 3 architectures)
# Select the single best model overall
```

## Summary

**Updates applied:**
1. ✅ Tile-level density tracking (28 tiles per image)
2. ✅ Log-scale boxplot with 1/Dilution x-axis
3. ✅ Two boxplot ranges (full: 1/10-1/10240, low: 1/80-1/10240)
4. ✅ Automatic best model selection from hyperparameter search

**Reused from previous analysis:**
- Tile visualizations (representative_tiles/)
- Test images
- Overall pipeline structure

**Ready for HPC submission:**
```bash
qsub pbs_density_analysis_unet_only.sh
```

**Expected runtime:** 1-2 hours

**Output:** 2 improved boxplots + tile-level CSV data for detailed analysis

---

**Created:** October 16, 2025
**Author:** Claude Code
**Status:** ✅ Ready for deployment
