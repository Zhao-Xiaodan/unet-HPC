# Density Analysis Fixes - October 16, 2025

## Overview

This document summarizes the fixes applied to `density_analysis_unet_only.py` based on user feedback from the analysis run `density_analysis_unet_only_20251016_015254`.

## Three Critical Fixes Applied

### 1. ✅ Reduced Tile Visualizations from 5 Panels to 3 Panels

**User Request:**
> "representative_tiles_5panel change to 3 panel, only show original, Inverted Pred, Inverted CLAHE+Otsu"

**Changes Made:**
- **Function:** `create_representative_tile_visualizations()`
- **Grid layout:** Changed from 5×5 to 5×3 (5 representative tiles × 3 panels)
- **Figure size:** Reduced from (25, 25) to (15, 25)
- **Output directory:** Changed from `representative_tiles_5panel/` to `representative_tiles_3panel/`
- **Filename pattern:** Changed from `tiles_5panel_*.png` to `tiles_3panel_*.png`

**Panel Layout (Before - 5 panels):**
1. Original tile
2. Predicted mask
3. CLAHE+Otsu on predicted mask
4. Inverted predicted mask (white beads)
5. Inverted CLAHE+Otsu mask (white beads, denoised)

**Panel Layout (After - 3 panels):**
1. **Original tile** - Actual microscope image (dark beads on bright background)
2. **Inverted Pred** - Model prediction inverted to show white beads on black
3. **Inverted CLAHE+Otsu** - Denoised prediction inverted to show white beads on black

**Rationale:** The 3-panel layout focuses on the most important comparisons:
- Original vs. predictions (panel 1 vs. panels 2-3)
- Predicted vs. denoised (panel 2 vs. panel 3)
- White beads on black background make visual comparison easier

**Lines Modified:** 690-780, 927-931

---

### 2. ✅ Fixed Opposite Trend in CLAHE+Otsu on Original Images

**User Request:**
> "density_boxplot_full_range__clahe+otsu_on_original.png generate opposite trend which is wrong. Refer to previous density analysis in density_analysis_xukuang_20251015_142119, which give proper trend that low density 1/10240 give low density value"

**Problem:**
The boxplot for CLAHE+Otsu applied to original images showed the **opposite trend**:
- High dilution (1/10240) → **High density** (WRONG!)
- Low dilution (1/10) → **Low density** (WRONG!)

**Expected trend (from reference analysis):**
- High dilution (1/10240) → **Low density** (0.00977) ✓
- Low dilution (1/10) → **High density** (0.5288) ✓

**Root Cause:**
Original microscope images have **dark beads on bright background**, but the CLAHE+Otsu pipeline (with `THRESH_BINARY_INV`) was designed for **bright foreground on dark background**.

**Solution:**
Added image inversion step **before** CLAHE+Otsu processing in `calculate_density_clahe_otsu_on_original()`.

**Changes Made:**
```python
# NEW: Invert original image so beads become bright (like predicted masks)
tile_gray_inverted = 255 - tile_gray

# THEN: Rescale to full 0-255 range
tile_rescaled = rescale_image_full_range(tile_gray_inverted)

# FINALLY: Apply CLAHE+Otsu (now works correctly)
binary_mask = apply_clahe_otsu(tile_rescaled)
```

**Processing Pipeline (After Fix):**
```
Original image: Dark beads on bright background
    ↓
Invert (255 - pixel_value): Bright beads on dark background
    ↓
Rescale to 0-255: Full dynamic range
    ↓
CLAHE: Enhance local contrast
    ↓
Otsu + THRESH_BINARY_INV: Segment bright beads as foreground
    ↓
Result: Correct density (high dilution → low density)
```

**Lines Modified:** 333-368

**Documentation:** See `CLAHE_OTSU_ORIGINAL_FIX.md` for detailed explanation

---

### 3. ✅ Removed Fixed Y-Axis Scale from Low Dilution Range Boxplots

**User Request (Previous Feedback):**
> "do not use the fixed y-axis scale range. Manually choosing large y-axis scale make the density_boxplot_full_range__clahe+otsu_on_pred.png flattened"

**Problem:**
The fixed y-axis scale (`ax.set_ylim(0.002, 1.5)`) was flattening the data visualization, making it hard to see differences between dilution levels.

**Status Before This Fix:**
- **Full range boxplots (line 578):** Already commented out ✓
- **Low dilution range boxplots (line 664):** Still active ❌

**Changes Made:**
Commented out the fixed y-axis limit in `create_boxplot_low_dilution_range()`:

```python
# Before (line 664)
ax.set_ylim(0.002, 1.5)  # Slightly wider range for visibility

# After (line 664)
# ax.set_ylim(0.002, 1.5)  # Commented out: fixed scale flattens the data
```

**Result:**
Now **all 6 boxplots** (3 density methods × 2 dilution ranges) use automatic y-axis scaling:
- Full range plots: Auto scale ✓
- Low dilution plots: Auto scale ✓
- Better visualization of density differences
- Data-driven axis limits

**Lines Modified:** 664

---

## Summary of Changes

### Files Modified
- `density_analysis_unet_only.py` - Main analysis script

### Files Created
- `CLAHE_OTSU_ORIGINAL_FIX.md` - Detailed explanation of inversion fix
- `DENSITY_ANALYSIS_FIXES_OCT16.md` - This summary document

### Code Changes by Line Number

| Line Range | Change | Purpose |
|------------|--------|---------|
| 333-368 | Added inversion in `calculate_density_clahe_otsu_on_original()` | Fix opposite trend |
| 664 | Commented out `ax.set_ylim()` | Remove fixed y-axis scale |
| 690-702 | Updated docstring and directory name | 5-panel → 3-panel |
| 731-780 | Reduced panels in visualization function | Show only 3 most relevant panels |
| 927-931 | Updated summary output | Reflect 3-panel format |

### Impact on Output Files

**Before:**
```
density_analysis_unet_only_YYYYMMDD_HHMMSS/
├── representative_tiles_5panel/
│   └── tiles_5panel_*.png (5 rows × 5 columns)
└── density_boxplot_*.png (with fixed/inconsistent y-axis)
```

**After:**
```
density_analysis_unet_only_YYYYMMDD_HHMMSS/
├── representative_tiles_3panel/
│   └── tiles_3panel_*.png (5 rows × 3 columns)
└── density_boxplot_*.png (all with automatic y-axis)
```

### Expected Results

**Boxplots (CLAHE+Otsu on original):**
- ✅ High dilution (1/10240) → Low density (correct trend)
- ✅ Low dilution (1/10) → High density (correct trend)
- ✅ Matches predicted mask trends

**Tile Visualizations:**
- ✅ 3 focused panels instead of 5
- ✅ Easier visual comparison with white beads on black
- ✅ Smaller file sizes (15×25 instead of 25×25)

**Boxplot Y-Axis:**
- ✅ All 6 boxplots use automatic scaling
- ✅ Better visibility of data distribution
- ✅ No artificial flattening of trends

## Testing Recommendations

### 1. Verify Correct Trend
Compare the new CLAHE+Otsu on original boxplot with the reference:
```bash
# Reference (correct trend)
density_analysis_xukuang_20251015_142119/density_boxplot.png

# New output (should match trend)
density_analysis_unet_only_YYYYMMDD_HHMMSS/density_boxplot_full_range_claheotsu_on_original.png
```

Expected: Both should show decreasing density with increasing dilution

### 2. Check 3-Panel Visualizations
Verify that panels 2 and 3 look similar (both show white beads):
```bash
ls -lh density_analysis_unet_only_YYYYMMDD_HHMMSS/representative_tiles_3panel/
```

Expected: 11 PNG files, each ~1-2 MB, showing 3 clear panels

### 3. Inspect Y-Axis Scaling
Open all 6 boxplots and verify automatic y-axis scaling:
```bash
ls -1 density_analysis_unet_only_YYYYMMDD_HHMMSS/density_boxplot_*.png
```

Expected: Y-axis ranges should differ between plots, not all 0.002-1.5

## Next Steps

### Run Updated Analysis
```bash
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_density_analysis_unet_only.sh
```

### Monitor Progress
```bash
qstat -u $USER
tail -f Density_UNet_Only.o<JOBID>
```

### Verify Output
```bash
# Check output directory created
ls -ld density_analysis_unet_only_*

# Verify 3-panel tiles (should have 11 images)
ls -lh density_analysis_unet_only_*/representative_tiles_3panel/

# Check all 6 boxplots generated
ls -lh density_analysis_unet_only_*/density_boxplot_*.png
```

## Related Documentation

- **`RESCALE_FIX_CRITICAL.md`** - Explains why rescaling before CLAHE+Otsu is critical
- **`CLAHE_OTSU_UPDATE.md`** - Overview of 3-method density calculation approach
- **`CLAHE_OTSU_ORIGINAL_FIX.md`** - Detailed explanation of inversion fix (NEW)
- **`DENSITY_ANALYSIS_UNET_ONLY_README.md`** - Full usage documentation

---

**Date:** October 16, 2025
**Author:** Claude Code
**Status:** ✅ Ready for deployment
**Testing:** Awaiting HPC run to verify fixes
