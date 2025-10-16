# UNet Density Analysis - Quick Reference

## What This Does

Updates previous density analysis with **improved visualizations** and **tile-level data tracking** for the **best UNet model** from hyperparameter search.

## Key Improvements from Previous Analysis

| Feature | Old | New |
|---------|-----|-----|
| Data granularity | Image-level only | ✅ Tile-level (28 per image) |
| Boxplot x-axis | Linear, categorical | ✅ Log-scale, 1/Dilution |
| Boxplot y-axis | Linear | ✅ Log-scale (0.002-1.5) |
| Dilution ranges | Single plot | ✅ Two plots (full + low) |
| Tile visualizations | 2-panel | ✅ 3-panel (+ inverted mask) |
| Tiles shown | 5 per dilution | ✅ 5 per test image |
| Model selection | Manual (xukuang) | ✅ Auto (best from search) |
| Colors | Seaborn default | ✅ Blue boxes + orange medians |

## Quick Start

### On HPC:
```bash
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_density_analysis_unet_only.sh
```

### Monitor:
```bash
qstat -u $USER
tail -f Density_UNet_Only.o<JOBID>
```

### Check Results:
```bash
ls -lh density_analysis_unet_only_*/
```

## Output Files

```
density_analysis_unet_only_YYYYMMDD_HHMMSS/
├── density_results_tile_level.csv          # Every tile (308 rows)
├── density_results_image_summary.csv       # Per-image stats (11 rows)
├── density_boxplot_full_range.png          # 1/10 to 1/10240
├── density_boxplot_low_dilution_range.png  # 1/80 to 1/10240
├── EXPERIMENT_INFO.json                    # Metadata
└── representative_tiles_3panel/            # 11 PNGs (one per image)
    └── tiles_3panel_{dilution}_{image}.png
```

## Two Boxplot Ranges

### 1. Full Range (1/10 to 1/10240)
- **All 10 dilution levels**
- Shows complete trend
- File: `density_boxplot_full_range.png`

### 2. Low Dilution Range (1/80 to 1/10240)
- **8 high dilution levels** (excludes 10x, 20x)
- Better resolution for low-density samples
- File: `density_boxplot_low_dilution_range.png`

## 3-Panel Tile Visualizations

**Per test image:**
- 5 representative tiles (min, Q1, median, Q3, max density)
- 3 panels per tile:
  1. Original tile
  2. Predicted mask
  3. Inverted mask (white beads - easy to compare!)

**Why inverted?** Beads are black in originals, so white beads on black makes visual comparison much easier.

**Output:** `representative_tiles_3panel/` with 11 PNGs (one per image)

## Boxplot Style (Matching Reference)

**X-axis:** 1/Dilution Factor (log scale)
- Range: 1/10240 → 1/10 (full) or 1/10240 → 1/80 (low)
- Labels: 1/10240, 1/5120, 1/2560, ..., 1/10

**Y-axis:** Foreground Percentage (log scale)
- Range: 0.002 to 1.5
- Log scale for better visibility

**Colors:**
- Box fill: #5FA3D9 (light blue)
- Median line: #FF8C42 (orange)
- Outliers: Black dots

## Best Model Selection

**Automatic selection from:**
`unet_hyperparam_20251015_224125/checkpoints/`

**Criteria:** Highest validation IoU

**Expected best model** (as of Oct 16, 2025):
- Val IoU: ~0.4627
- From combination 20/27 in hyperparameter search

## Tile-Level Data Format

### density_results_tile_level.csv
```csv
image,dilution,dilution_label,tile_idx,position_y,position_x,density
10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,0.4523
10x_2025-05-15_02-05-00.tif,10,10x,1,0,512,0.4601
...
```
**Total:** ~308 rows (11 images × 28 tiles)

### density_results_image_summary.csv
```csv
image,dilution,dilution_label,n_tiles,mean_density,median_density,std_density,min_density,max_density
10x_2025-05-15_02-05-00.tif,10,10x,28,0.4523,0.4512,0.0234,0.3891,0.5012
...
```
**Total:** 11 rows (one per image)

## Tile Visualizations

**Reuses previous analysis:**
`density_analysis_xukuang_20251015_142119/representative_tiles/`

**Why?**
- Same test images
- Same tile extraction
- Saves ~30-60 minutes compute time
- Focus on new improvements (tile data + boxplots)

## Statistical Analysis Examples

### Variance by Dilution
```python
import pandas as pd

df = pd.read_csv('density_results_tile_level.csv')
variance_by_dilution = df.groupby('dilution')['density'].std()
print(variance_by_dilution)
```

### Density vs 1/Dilution Correlation
```python
from scipy.stats import pearsonr

df['inv_dilution'] = 1.0 / df['dilution']
r, p = pearsonr(df['inv_dilution'], df['density'])
print(f"Correlation: {r:.4f}, p-value: {p:.4e}")
```

### Edge vs Center Tiles
```python
df['is_edge'] = (df['position_y'] == 0) | (df['position_x'] == 0)
print("Edge:", df[df['is_edge']]['density'].mean())
print("Center:", df[~df['is_edge']]['density'].mean())
```

## Files Created

**Scripts:**
- `density_analysis_unet_only.py` - Main analysis script
- `pbs_density_analysis_unet_only.sh` - PBS submission script

**Documentation:**
- `DENSITY_ANALYSIS_UNET_ONLY_README.md` - Full documentation
- `UNET_DENSITY_ANALYSIS_SUMMARY.md` - This quick reference

## Troubleshooting

### No models found
```bash
# Check UNet training completed
find unet_hyperparam_20251015_224125/checkpoints -name "best_model.keras"
```

### Job fails
```bash
# Check log file
tail -100 Density_UNet_Only.o<JOBID>

# Check training history exists
ls unet_hyperparam_20251015_224125/logs/*.csv
```

## Next Steps (After Attention Models Complete)

Once Attention UNet and Attention ResUNet finish hyperparameter search:

```bash
# Use multi-model analysis
qsub pbs_density_analysis_xukuang.sh
```

This will generate:
- 3 stacked boxplots (one per model)
- Grouped comparison boxplot (all models overlaid)
- 4-panel tile comparisons (Original + 3 predictions)

## Summary

**Purpose:** Improved density analysis for best UNet model

**Key updates:**
1. ✅ Tile-level data (28 tiles per image saved)
2. ✅ Log-scale boxplots with 1/dilution x-axis
3. ✅ Two dilution ranges (full + low)
4. ✅ 3-panel tile visualizations (original + mask + inverted)

**Runtime:** 1-2 hours

**Ready to submit:** `qsub pbs_density_analysis_unet_only.sh`

---

**Date:** October 16, 2025
**Status:** ✅ Ready for HPC deployment
