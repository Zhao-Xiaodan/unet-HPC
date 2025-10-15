# Density Analysis Pipeline - Update Summary

## Date: October 15, 2025

## Updates Completed

### 1. ✅ Multi-Model Support
**Previous:** Single model (UNet only)
**Now:** All three models (UNet, Attention UNet, Attention ResUNet)

- Loads all three models from `xukuang_params_shrunk_20251015_071224/`
- Predicts with all models on every tile
- Saves results for each model separately

### 2. ✅ Tile-Level Density Tracking
**Previous:** Only image-level mean densities
**Now:** ALL individual tile densities saved

**Output Files:**
- `density_results_tile_level.csv` - Every tile (n=28 per image) × 3 models
- `density_results_image_summary.csv` - Aggregated statistics per image per model

**Example:**
```csv
image,dilution,dilution_label,tile_idx,position_y,position_x,model,density
10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,unet,0.4523
10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,attention_unet,0.4512
10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,attention_resunet,0.4534
```

### 3. ✅ Updated Boxplot Style
**Previous:** Linear x-axis, categorical dilution labels
**Now:** Log-scale x-axis with 1/Dilution factor (matching reference image)

**Key Changes:**
- **X-axis:** 1/Dilution Factor (1/10240, 1/5120, ..., 1/80)
- **X-scale:** Logarithmic
- **Y-axis:** Foreground Percentage (renamed from "density")
- **Y-scale:** Logarithmic (0.002 to 1.5)
- **Colors:** Light blue boxes with orange median lines
- **Style:** Clean matplotlib boxplot (not seaborn)

**Generated Plots:**
1. `density_boxplot_multimodel.png` - Three stacked subplots (one per model)
2. `density_boxplot_comparison.png` - Grouped comparison with color-coded models

### 4. ✅ 4-Panel Tile Visualizations
**Previous:** 2-panel (original + prediction)
**Now:** 4-panel (original + 3 model predictions)

**Layout:**
```
[Original] [UNet] [Attention UNet] [Attention ResUNet]
```

**Output:**
- Directory: `representative_tiles_4panel/`
- Files: `tiles_4panel_10x.png`, `tiles_4panel_20x.png`, etc.
- 5 representative tiles per dilution (spanning density range)
- Each prediction shows density value

### 5. ✅ Updated PBS Script
**Changes:**
- Job name: `Density_Xukuang` → `Density_MultiModel`
- Walltime: 4 hours → **8 hours** (for 3 models)
- Output directory pattern: `density_analysis_xukuang_multimodel_*`
- Updated logging for multi-model output

### 6. ✅ Comprehensive Documentation
Created `DENSITY_ANALYSIS_MULTIMODEL_README.md` with:
- Complete usage instructions
- File format descriptions
- Pipeline details
- Troubleshooting guide
- Analysis questions to answer

## Files Modified

### 1. `density_analysis_xukuang.py`
**Major Changes:**
- Added `load_all_models()` function to load all 3 models
- Updated `predict_on_test_images_multimodel()` to predict with all models
- Rewrote `create_multimodel_boxplot()` for log-scale 1/dilution style
- Rewrote `create_model_comparison_boxplot()` for grouped multi-model comparison
- Added `create_4panel_comparison()` for multi-model tile visualizations
- Updated CSV output to include tile-level data

**Key Code Sections:**
```python
# Line 268-282: Load all three models
def load_all_models(config):
    models = {}
    for model_name in config['models']:
        models[model_name] = load_model(config['model_dir'], model_name)
    return models

# Line 288-371: Predict with all models on every tile
def predict_on_test_images_multimodel(models, test_images_dir, config):
    # ... predicts with all models, saves tile-level results

# Line 377-467: Multi-model boxplot with log-scale 1/dilution
def create_multimodel_boxplot(df_tile_results, output_dir, config):
    # Uses matplotlib boxplot with log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Positions: [1/10240, 1/5120, ..., 1/80]

# Line 469-556: Grouped comparison boxplot
def create_model_comparison_boxplot(df_tile_results, output_dir, config):
    # Overlays all 3 models at each dilution point

# Line 528-600: 4-panel tile comparisons
def create_4panel_comparison(tile_data, output_dir, config):
    # Creates 4-column layout: Original + 3 predictions
```

### 2. `pbs_density_analysis_xukuang.sh`
**Changes:**
- Header: Updated to "Multi-Model Density Analysis"
- Walltime: `4:00:00` → `8:00:00`
- Job name: `Density_Xukuang` → `Density_MultiModel`
- Output directory detection: Updated pattern to `density_analysis_xukuang_multimodel_*`
- CSV file references: Updated to `density_results_tile_level.csv` and `density_results_image_summary.csv`

### 3. Documentation Files Created
- `DENSITY_ANALYSIS_MULTIMODEL_README.md` - Complete usage guide (500+ lines)
- `DENSITY_ANALYSIS_UPDATE_SUMMARY.md` - This file

## Key Technical Details

### Boxplot Styling (Matching Reference)

**X-Axis (1/Dilution):**
```python
positions = [1.0/d for d in DILUTION_ORDER]
# positions = [1/10240, 1/5120, 1/2560, ..., 1/80]

ax.set_xscale('log')
ax.set_xticks(positions)
ax.set_xticklabels(['1/10240', '1/5120', ..., '1/80'])
```

**Y-Axis (Foreground Percentage):**
```python
ax.set_yscale('log')
ax.set_ylim(0.002, 1.5)  # Log scale from 0.002 to 1.5
ax.set_ylabel('Foreground Percentage')
```

**Colors:**
```python
box_color = '#5FA3D9'  # Light blue boxes
median_color = '#FF8C42'  # Orange median lines

# For comparison plot (3 models):
model_colors = {
    'unet': '#5FA3D9',           # Blue
    'attention_unet': '#8BC34A',  # Green
    'attention_resunet': '#FF6B9D' # Pink
}
```

**Boxplot Properties:**
```python
bp = ax.boxplot(
    data_by_dilution,
    positions=positions,
    widths=[p*0.15 for p in positions],  # Width proportional to position
    patch_artist=True,
    boxprops=dict(facecolor=box_color, color='black', linewidth=1),
    medianprops=dict(color=median_color, linewidth=2),
    whiskerprops=dict(color='black', linewidth=1),
    capprops=dict(color='black', linewidth=1),
    flierprops=dict(marker='o', markerfacecolor='black', markersize=3)
)
```

### Tile-Level Data Structure

**Before:**
```csv
image,dilution,dilution_label,n_tiles,mean_density,median_density,std_density
10x_2025-05-15_02-05-00.tif,10,10x,28,0.4523,0.4512,0.0234
```

**After (Tile-Level):**
```csv
image,dilution,dilution_label,tile_idx,position_y,position_x,model,density
10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,unet,0.4523
10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,attention_unet,0.4512
10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,attention_resunet,0.4534
10x_2025-05-15_02-05-00.tif,10,10x,1,0,512,unet,0.4601
...
```

**Total Rows:** n_images × n_tiles_per_image × n_models
- Example: 11 images × 28 tiles × 3 models = 924 rows

## Usage

### Submit to HPC:
```bash
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_density_analysis_xukuang.sh
```

### Monitor:
```bash
qstat -u $USER
tail -f Density_MultiModel.o*
```

### Check Results:
```bash
ls -la density_analysis_xukuang_multimodel_*/
```

## Expected Output Directory

```
density_analysis_xukuang_multimodel_20251015_HHMMSS/
├── density_results_tile_level.csv        # Tile-level densities (all models)
├── density_results_image_summary.csv     # Image-level summaries
├── density_boxplot_multimodel.png        # 3 stacked boxplots
├── density_boxplot_comparison.png        # Grouped comparison
├── EXPERIMENT_INFO.json                  # Metadata
└── representative_tiles_4panel/
    ├── tiles_4panel_10x.png
    ├── tiles_4panel_20x.png
    ├── tiles_4panel_80x.png
    ├── tiles_4panel_160x.png
    ├── tiles_4panel_320x.png
    ├── tiles_4panel_640x.png
    ├── tiles_4panel_1280x.png
    ├── tiles_4panel_2560x.png
    ├── tiles_4panel_5120x.png
    └── tiles_4panel_10240x.png
```

## Verification Checklist

After job completes:

- [ ] **3 models loaded successfully** (UNet, Attention UNet, Attention ResUNet)
- [ ] **Tile-level CSV exists** with n_images × n_tiles × 3 rows
- [ ] **Image summary CSV exists** with n_images × 3 rows
- [ ] **Multimodel boxplot** shows 3 stacked subplots with log scales
- [ ] **Comparison boxplot** shows grouped models with color coding
- [ ] **10 4-panel PNGs** in representative_tiles_4panel/ directory
- [ ] **X-axis ordering:** 1/10240, 1/5120, ..., 1/80 (left to right)
- [ ] **Y-axis scale:** Log scale from 0.002 to 1.5
- [ ] **Boxplot style:** Blue boxes with orange medians

## Performance Comparison Analysis

With tile-level data, you can now analyze:

### 1. Model-Level Questions:
- Which model has highest/lowest mean density?
- Which model has most/least variance?
- Do attention mechanisms improve consistency?

### 2. Dilution-Level Questions:
- Is density inversely proportional to dilution?
- Which dilution shows most inter-model variation?
- Are high dilutions (10240x) more variable?

### 3. Tile-Level Questions:
- How much within-image variation exists?
- Are edge tiles different from center tiles?
- Which images have highest tile-to-tile variance?

### 4. Model Agreement:
- Correlation between model predictions
- Where do models disagree most?
- Visual inspection via 4-panel comparisons

## Statistical Analysis Ideas

With tile-level data:

### ANOVA:
```r
# Model effect
anova(lm(density ~ model, data=tile_data))

# Dilution effect
anova(lm(density ~ dilution, data=tile_data))

# Model × Dilution interaction
anova(lm(density ~ model * dilution, data=tile_data))
```

### Correlation:
```r
# Inter-model correlation
cor.test(unet_densities, attention_unet_densities)

# Density vs 1/Dilution
cor.test(1/dilution, density)
```

### Variance Partitioning:
```r
library(lme4)
model <- lmer(density ~ model + (1|image) + (1|tile_idx), data=tile_data)
```

## Next Steps

1. **Submit Job:**
   ```bash
   qsub pbs_density_analysis_xukuang.sh
   ```

2. **Wait for Completion:**
   - Expected runtime: 1-2 hours
   - Walltime limit: 8 hours

3. **Review Results:**
   - Check boxplots match reference style
   - Inspect 4-panel comparisons
   - Verify CSV data integrity

4. **Statistical Analysis:**
   - Load tile-level CSV into R/Python
   - Run ANOVA, correlation, variance analysis
   - Generate summary statistics table

5. **Report Generation:**
   - Include both boxplots
   - Select best 4-panel comparisons
   - Add model performance table
   - Discuss model differences

## Troubleshooting

### If job fails:
1. Check log file: `Density_MultiModel.o*`
2. Verify all 3 model files exist: `ls -la xukuang_params_shrunk_20251015_071224/*.keras`
3. Check memory usage (may need more than 240GB for 3 models)

### If boxplots look wrong:
1. Verify x-axis shows 1/10240 on left, 1/80 on right
2. Check log scale is enabled on both axes
3. Ensure dilution order is correct in DILUTION_ORDER list

### If 4-panel comparisons missing:
1. Check `representative_tiles_4panel/` directory exists
2. Verify tile data was saved during prediction
3. Check for errors in create_4panel_comparison() function

## Summary

**All requested features implemented:**
1. ✅ Tile-level density tracking (n=28 per image)
2. ✅ Multi-model prediction (UNet + Attention UNet + Attention ResUNet)
3. ✅ Log-scale boxplot with 1/Dilution x-axis (matching reference)
4. ✅ 4-panel tile comparisons
5. ✅ Comprehensive documentation

**Ready for HPC submission!**

---

**Created:** October 15, 2025
**Author:** Claude Code
**Status:** ✅ Complete and ready for deployment
