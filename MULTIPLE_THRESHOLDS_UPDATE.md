# Multiple Thresholds Analysis - October 16, 2025

## Overview

Updated the density analysis to test **three different threshold values** (0.2, 0.5, 0.8) for segmenting predicted masks, in addition to the existing CLAHE+Otsu methods. This allows comprehensive analysis of how threshold choice affects density estimates.

## Five Density Calculation Methods

### Simple Thresholding (3 methods)

**Method 1: Threshold 0.2 (Low threshold)**
```python
density = mean(prediction > 0.2)
```
- **Sensitivity:** High - captures more potential beads
- **Risk:** May include false positives and noise
- **Best for:** Ensuring no beads are missed (high recall)
- **Expected:** Higher density values than 0.5 and 0.8

**Method 2: Threshold 0.5 (Middle threshold)**
```python
density = mean(prediction > 0.5)
```
- **Sensitivity:** Moderate - balanced approach
- **Risk:** Moderate false positives and false negatives
- **Best for:** Standard segmentation (most commonly used)
- **Expected:** Middle range density values

**Method 3: Threshold 0.8 (High threshold)**
```python
density = mean(prediction > 0.8)
```
- **Sensitivity:** Low - only captures high-confidence beads
- **Risk:** May miss faint or partially visible beads
- **Best for:** High precision (low false positives)
- **Expected:** Lower density values than 0.2 and 0.5

### CLAHE+Otsu Methods (2 methods)

**Method 4: CLAHE+Otsu on Predicted Mask**
- Applies adaptive preprocessing to denoise predictions
- Automatic threshold selection via Otsu
- Reduces noise in low-density scenarios

**Method 5: CLAHE+Otsu on Original Image**
- Traditional computer vision baseline
- No ML model involved
- Shows improvement gained by using UNet

## Expected Relationships

### Between Thresholds
```
Density(0.2) > Density(0.5) > Density(0.8)
```
Lower thresholds include more pixels, resulting in higher density estimates.

### Example at 10x Dilution (High Bead Count)
```
Threshold 0.2:  ~0.65 (includes partial beads and noise)
Threshold 0.5:  ~0.53 (balanced segmentation)
Threshold 0.8:  ~0.38 (only bright, confident beads)
CLAHE+Otsu:     ~0.46 (adaptive denoising)
```

### Example at 10240x Dilution (Low Bead Count)
```
Threshold 0.2:  ~0.018 (captures faint beads + some noise)
Threshold 0.5:  ~0.010 (balanced)
Threshold 0.8:  ~0.005 (only very bright beads)
CLAHE+Otsu:     ~0.009 (denoised, similar to 0.5)
```

## Analysis Questions Answered

### 1. What is the optimal threshold for this dataset?

Compare the three threshold boxplots:
- If all three show similar trends → Model is confident in predictions
- If 0.2 and 0.5 are close → Predictions are generally binary (confident)
- If large spread between 0.2/0.5/0.8 → Predictions have gradients (less confident)

### 2. How sensitive are density estimates to threshold choice?

Calculate coefficient of variation:
```python
cv = std([density_0.2, density_0.5, density_0.8]) / mean([...])
```
- Low CV (<0.2) → Threshold choice matters less
- High CV (>0.5) → Threshold choice is critical

### 3. Does CLAHE+Otsu approximate a fixed threshold?

Compare CLAHE+Otsu boxplot to the three threshold boxplots:
- If CLAHE+Otsu ≈ Threshold 0.5 → Automatic threshold finds ~0.5
- If CLAHE+Otsu < Threshold 0.8 → Very conservative (high precision)
- If CLAHE+Otsu > Threshold 0.2 → May be too permissive

### 4. Which method gives the most stable estimates?

Compare variance across dilutions for each method:
- Low variance → Method is stable and reliable
- High variance → Method is sensitive to image quality

## Output Files

### CSV Data Files

**density_results_tile_level.csv**
```csv
image,dilution,dilution_label,tile_idx,position_y,position_x,density_threshold_0.2,density_threshold_0.5,density_threshold_0.8,density_clahe_otsu_pred,density_clahe_otsu_orig
10x.tif,10,10x,0,0,0,0.6523,0.5288,0.3891,0.4612,0.4501
...
```

**density_results_image_summary.csv**
```csv
image,dilution,n_tiles,mean_density_threshold_0.2,mean_density_threshold_0.5,mean_density_threshold_0.8,mean_density_clahe_otsu_pred,mean_density_clahe_otsu_orig,...
10x.tif,10,28,0.6523,0.5288,0.3891,0.4612,0.4501,...
...
```

### Boxplot Files (10 Total)

**Full Range (1/10 to 1/10240):**
1. `density_boxplot_full_range_threshold_0.2.png`
2. `density_boxplot_full_range_threshold_0.5.png`
3. `density_boxplot_full_range_threshold_0.8.png`
4. `density_boxplot_full_range_claheotsu_on_pred.png`
5. `density_boxplot_full_range_claheotsu_on_original.png`

**Low Dilution Range (1/80 to 1/10240):**
6. `density_boxplot_low_dilution_range_threshold_0.2.png`
7. `density_boxplot_low_dilution_range_threshold_0.5.png`
8. `density_boxplot_low_dilution_range_threshold_0.8.png`
9. `density_boxplot_low_dilution_range_claheotsu_on_pred.png`
10. `density_boxplot_low_dilution_range_claheotsu_on_original.png`

### Tile Visualizations

**representative_tiles_3panel/**
- Uses **threshold 0.5** for the predicted mask panel
- Shows inverted masks for easy comparison (white beads on black)
- 5 representative tiles per test image

## Comparative Analysis

### Side-by-Side Comparison

Open all 5 full-range boxplots and compare:

**Vertical Alignment:** All should show the same trend (decreasing density with increasing dilution)

**Relative Positions:**
```
Expected vertical ordering (from top to bottom):
1. Threshold 0.2 (highest densities)
2. Threshold 0.5
3. CLAHE+Otsu on pred (near 0.5 or between 0.5 and 0.8)
4. Threshold 0.8 (lowest densities)
5. CLAHE+Otsu on original (comparison baseline)
```

**Variance Comparison:**
- Check which method has the tightest boxes (lowest variance)
- Smaller boxes = more consistent predictions across tiles

### Statistical Comparison

```python
import pandas as pd
import numpy as np

df = pd.read_csv('density_results_tile_level.csv')

# Calculate mean densities per method
methods = ['density_threshold_0.2', 'density_threshold_0.5',
           'density_threshold_0.8', 'density_clahe_otsu_pred']

for method in methods:
    mean_density = df.groupby('dilution')[method].mean()
    print(f"\n{method}:")
    print(mean_density)

# Calculate correlation between methods
correlations = df[methods].corr()
print("\nCorrelations between methods:")
print(correlations)
```

## Code Changes Summary

### Configuration
```python
CONFIG = {
    'thresholds': [0.2, 0.5, 0.8],  # NEW: Multiple thresholds
    ...
}
```

### Prediction Loop
```python
# OLD: Single threshold
density_threshold = calculate_foreground_density(prediction, config['threshold'])

# NEW: Three thresholds
density_threshold_02 = calculate_foreground_density(prediction, 0.2)
density_threshold_05 = calculate_foreground_density(prediction, 0.5)
density_threshold_08 = calculate_foreground_density(prediction, 0.8)
```

### Tile-Level Results
```python
tile_results.append({
    'density_threshold_0.2': density_threshold_02,  # NEW
    'density_threshold_0.5': density_threshold_05,  # RENAMED from density_threshold
    'density_threshold_0.8': density_threshold_08,  # NEW
    'density_clahe_otsu_pred': density_clahe_otsu_pred,
    'density_clahe_otsu_orig': density_clahe_otsu_orig,
})
```

### Boxplot Generation
```python
# OLD: 2 boxplots per method × 3 methods = 6 boxplots
# NEW: 2 boxplots per method × 5 methods = 10 boxplots

for threshold in [0.2, 0.5, 0.8]:
    create_boxplot_full_range(..., density_column=f'density_threshold_{threshold}')
    create_boxplot_low_dilution_range(..., density_column=f'density_threshold_{threshold}')
```

## Usage

### Run Analysis on HPC
```bash
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_density_analysis_unet_only.sh
```

### Compare Thresholds Locally
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('density_results_tile_level.csv')

# Group by dilution
grouped = df.groupby('dilution')

# Plot mean densities for each threshold
thresholds = [0.2, 0.5, 0.8]
for t in thresholds:
    means = grouped[f'density_threshold_{t}'].mean()
    plt.plot(1/means.index, means.values, 'o-', label=f'Threshold {t}')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('1 / Dilution')
plt.ylabel('Mean Density')
plt.legend()
plt.title('Threshold Comparison')
plt.show()
```

## Recommendations

### For Publication
- **Use Threshold 0.5** as the primary method (most standard)
- **Include CLAHE+Otsu on pred** to show robustness to noise
- **Show Threshold 0.2 and 0.8** in supplementary materials to demonstrate sensitivity analysis

### For Analysis
- **Compare all 5 methods** to understand model behavior
- **Check correlation** between methods (high correlation = consistent predictions)
- **Examine low-density samples** (10240x, 5120x) where methods may diverge

### For Model Evaluation
- If thresholds 0.2/0.5/0.8 give very different results → Model needs improvement (predictions are ambiguous)
- If thresholds give similar results → Model is confident and well-calibrated

## Summary

**Update:** Added multiple threshold analysis (0.2, 0.5, 0.8)

**Total methods:** 5 (3 thresholds + 2 CLAHE+Otsu)

**Total boxplots:** 10 (5 methods × 2 dilution ranges)

**CSV columns:** 5 density columns per tile

**Purpose:** Comprehensive analysis of threshold sensitivity and optimal operating point

**Benefit:** Better understanding of model confidence and appropriate threshold selection

---

**Date:** October 16, 2025
**Status:** ✅ Ready for deployment
**Runtime:** ~1-2 hours (same as before, minimal overhead)
