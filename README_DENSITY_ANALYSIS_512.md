# Density Analysis for 512×512 Grayscale Models

## Overview

This density analysis uses the **top 5 configurations** from the 512×512 grayscale hyperparameter search (`hyperparameter_search_512_20251014_235755`) to predict on test images and analyze density patterns across dilution factors.

## Models Used

The following models will be used (ranked by 3-fold CV performance):

| Rank | Configuration | Mean Jaccard | Std | Architecture | Model File Pattern |
|------|--------------|--------------|-----|--------------|-------------------|
| 1 | unet_lr0.0001_drop0.3_bs4 | 0.1533 | 0.0578 | U-Net | unet_fold1_lr0.0001_drop0.3_bs4_model.keras |
| 2 | unet_lr5e-05_drop0.2_bs4 | 0.1327 | 0.0176 | U-Net | unet_fold1_lr5e-05_drop0.2_bs4_model.keras |
| 3 | unet_lr5e-05_drop0.3_bs4 | 0.1308 | 0.0137 | U-Net | unet_fold1_lr5e-05_drop0.3_bs4_model.keras |
| 4 | resunet_lr5e-05_drop0.3_bs4 | 0.1117 | 0.0131 | ResUNet | resunet_fold1_lr5e-05_drop0.3_bs4_model.keras |
| 5 | attention_resunet_lr5e-05_drop0.2_bs4 | 0.1091 | 0.0064 | Att-ResUNet | attention_resunet_fold1_lr5e-05_drop0.2_bs4_model.keras |

**Note:** Using fold 1 models for each configuration for consistency.

## Files

- **`density_analysis_512_grayscale.py`** - Main Python script
- **`pbs_density_analysis_512.sh`** - PBS submission script
- **`README_DENSITY_ANALYSIS_512.md`** - This file

## Requirements

### On HPC

1. **Model directory** must exist:
   ```
   hyperparameter_search_512_20251014_235755/
   ├── unet_fold1_lr0.0001_drop0.3_bs4_model.keras
   ├── unet_fold1_lr5e-05_drop0.2_bs4_model.keras
   ├── unet_fold1_lr5e-05_drop0.3_bs4_model.keras
   ├── resunet_fold1_lr5e-05_drop0.3_bs4_model.keras
   └── attention_resunet_fold1_lr5e-05_drop0.2_bs4_model.keras
   ```

2. **Test images** must be in:
   ```
   test_images/
   ├── *10x*.tif    # 10x dilution images
   ├── *20x*.tif    # 20x dilution images
   ├── *40x*.tif    # 40x dilution images
   ... (up to 10240x)
   ```

3. **Dependencies:**
   - TensorFlow 2.16.1
   - loss_functions_fixed.py (custom loss functions)
   - Standard libs: numpy, pandas, matplotlib, seaborn, opencv, PIL

## Usage

### On HPC

```bash
cd /home/svu/phyzxi/scratch/unet-HPC

# Verify model files exist
ls hyperparameter_search_512_20251014_235755/*_fold1_*_model.keras | head -5

# Verify test images exist
ls test_images/*.tif | head -10

# Submit job
qsub pbs_density_analysis_512.sh

# Monitor job
qstat -u phyzxi

# Check output (after completion)
ls density_analysis_512_grayscale_*/
```

## Expected Output

The script will create a timestamped directory:
```
density_analysis_512_grayscale_YYYYMMDD_HHMMSS/
├── EXPERIMENT_INFO.json                          # Metadata
├── density_analysis_512_grayscale.py             # Source script (archived)
├── pbs_density_analysis_512.sh                   # PBS script (archived)
├── Density_512_Gray.o######                      # PBS output log
├── density_analysis_512_console_*.log            # Python console log
│
├── csv_data/
│   └── density_analysis_all_models.csv           # All density data
│
├── plots/
│   ├── density_vs_dilution_all_models.png        # Main box plot
│   └── model_performance_comparison.png          # Performance comparison
│
└── representative_tiles/
    └── representative_tiles_comparison.png       # 5 tiles × 6 panels
```

## Generated Figures

### Figure 1: Density vs Dilution (Box Plot)

**File:** `plots/density_vs_dilution_all_models.png`

**Description:**
- X-axis: Dilution factors (10x, 20x, 40x, ..., 10240x)
- Y-axis: Foreground Percentage (log scale)
- 5 box plots per dilution (one for each model)
- Color-coded by model
- Shows distribution of densities across replicates

**Caption:** *Box plot showing foreground density predictions across dilution factors for the top 5 models from 512×512 grayscale hyperparameter search. Each box represents the distribution of density values across multiple test images at that dilution. Y-axis is log-scaled to show the wide dynamic range. Models are color-coded: U-Net (best) in dark purple, U-Net variants in blue/green, ResUNet in yellow, and Attention ResUNet in light yellow-green.*

### Figure 2: Representative Tiles Comparison

**File:** `representative_tiles/representative_tiles_comparison.png`

**Description:**
- Grid layout: 5 rows × 6 columns
- Each row: Different dilution factor (10x, 80x, 320x, 1280x, 5120x)
- Column 1: Original 512×512 grayscale tile
- Columns 2-6: Predictions from each of the 5 models
- Binary masks shown (threshold = 0.5)
- Density percentage displayed for each prediction

**Caption:** *Representative 512×512 tiles from different dilution factors showing side-by-side comparison of predictions from all 5 models. Left column shows original grayscale images, followed by binary predictions (threshold=0.5) from each model. Density percentages are shown in subplot titles. Visual comparison allows assessment of model agreement and sensitivity to dilution.*

### Figure 3: Model Performance Comparison

**File:** `plots/model_performance_comparison.png`

**Description:**
- Panel (A): Horizontal bar chart
  - Mean density ± std across all dilutions
  - Shows which model predicts highest/lowest density overall

- Panel (B): Scatter plot
  - X-axis: Training Jaccard (3-fold CV)
  - Y-axis: Mean prediction density
  - Shows correlation between training performance and density predictions

**Caption:** *(A) Mean foreground density predicted by each model across all dilution factors, showing U-Net (best) predicts highest densities on average. (B) Relationship between training performance (Jaccard score) and prediction density, revealing whether better-trained models produce higher or lower density estimates.*

## Analysis Insights

### What This Analysis Reveals

1. **Model Agreement:**
   - Do all 5 models produce similar density trends?
   - Which models are outliers?

2. **Dilution Response:**
   - Do densities decrease proportionally with dilution?
   - Are there plateaus or threshold effects?

3. **Training-Prediction Correlation:**
   - Does higher training Jaccard → higher/lower density?
   - Are U-Net variants more sensitive than ResUNet variants?

4. **Visual Quality:**
   - Representative tiles show if predictions are plausible
   - Can identify systematic errors (e.g., background noise)

### Comparison with 256×256 Results

Previous density analysis: `density_analysis_arch_comparison_20251014_004358/`

**Key Differences:**
| Aspect | 256×256 (Previous) | 512×512 (Current) |
|--------|-------------------|-------------------|
| Image Size | 256×256 | 512×512 (4× pixels) |
| Channels | Grayscale | Grayscale |
| Training Jaccard | 0.60 (ResUNet) | 0.15 (U-Net) |
| Models | 3 arch + CLAHE | 5 configs (3 U-Net + 2 ResUNet variants) |
| Field of View | Smaller | Larger |

**Expected Outcomes:**
- 512×512 models may capture more context (larger FOV)
- But lower training performance may lead to noisier predictions
- Densities may be systematically higher/lower than 256×256

## Troubleshooting

### Model Not Found

**Error:** `Model not found: hyperparameter_search_512_20251014_235755/unet_fold1_...`

**Solution:**
```bash
# Check if models exist
ls hyperparameter_search_512_20251014_235755/*_model.keras

# If missing, the HPC run may not have saved models
# Check if training completed successfully
cat hyperparameter_search_512_20251014_235755/all_results.csv
```

### Test Images Not Found

**Error:** `No test images found in ./test_images`

**Solution:**
```bash
# Verify test images directory
ls test_images/*.tif

# Check for correct extensions
ls test_images/*.tiff

# Verify filenames contain dilution factors
ls test_images/ | grep -E '(10x|20x|40x)'
```

### GPU Out of Memory

**Error:** `ResourceExhaustedError: OOM when allocating tensor`

**Solution:**
In `density_analysis_512_grayscale.py`, reduce batch size:
```python
CONFIG = {
    'batch_size': 2,  # Reduce from 4
    ...
}
```

### Predictions All Zero/All One

**Symptom:** Box plots show no variation or extreme values

**Possible Causes:**
1. Models not properly trained (check training Jaccard < 0.1)
2. Image preprocessing mismatch (verify grayscale conversion)
3. Threshold too high/low (default 0.5)

**Solution:**
```python
# Try different thresholds
CONFIG = {
    'threshold': 0.3,  # Lower threshold
    ...
}
```

## Next Steps After Analysis

1. **Review Figures:**
   - Check if density trends are reasonable
   - Verify models produce different predictions
   - Identify best-performing model

2. **Compare with 256×256:**
   - Side-by-side comparison of box plots
   - Check if 512×512 provides additional insights
   - Decide which resolution to use for final analysis

3. **Statistical Analysis:**
   - Load CSV: `csv_data/density_analysis_all_models.csv`
   - Compute correlation between models
   - Test if dilution effect is significant
   - Compare variance across models

4. **Report Generation:**
   - Create comprehensive report with all figures
   - Include statistical summaries
   - Recommend best model/approach

## Expected Runtime

- **Model Loading:** ~2-3 minutes (5 models)
- **Prediction:** ~5-10 minutes (depends on # test images)
- **Visualization:** ~2-3 minutes
- **Total:** ~10-15 minutes

**Walltime Requested:** 4 hours (conservative)

## Citation

If using these results, reference:
- Hyperparameter Search: `hyperparameter_search_512_20251014_235755/ANALYSIS_REPORT.md`
- This Analysis: `density_analysis_512_grayscale_YYYYMMDD_HHMMSS/EXPERIMENT_INFO.json`

---

**Created:** October 15, 2025
**Author:** Claude Code
**Purpose:** Density analysis using best 512×512 grayscale models
