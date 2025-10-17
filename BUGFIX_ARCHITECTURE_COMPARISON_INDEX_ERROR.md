# Bug Fix: Architecture Comparison IndexError

**Date:** October 17, 2025
**Job:** Density_Arch_Comparison.o293815
**Error:** `IndexError: list index out of range`
**Status:** ✅ Fixed

---

## Problem

The architecture comparison script crashed during density calculation with:

```python
IndexError: list index out of range
  File "density_analysis_architecture_comparison.py", line 365
    density_clahe_orig = tile_results[-3]['density_clahe_otsu_orig']
```

**Result:** Empty output directory, no visualizations generated.

---

## Root Cause

### Flawed Logic

The script tried to **reuse** the CLAHE+Otsu result from UNet by looking back in the `tile_results` list:

```python
for arch_name in ARCHITECTURES:
    if arch_name == 'UNet':
        density_clahe_orig, binary_orig = calculate_density_clahe_otsu_on_original(tile)
    else:
        density_clahe_orig = tile_results[-3]['density_clahe_otsu_orig']  # ❌ WRONG!
```

**Problem:** On the **first tile**, `tile_results` is empty, so `tile_results[-3]` raises `IndexError`.

**Even if it worked:** The index `-3` assumes exactly 3 architectures and that we're always looking at the right tile, which is fragile.

---

## Fix Applied

### Calculate Once Per Tile

Move the CLAHE+Otsu calculation **outside** the architecture loop:

```python
# OLD (broken)
for tile_idx, (tile, pos) in enumerate(tiles_with_pos):
    for arch_name in ARCHITECTURES:
        if arch_name == 'UNet':
            density_clahe_orig = calculate_density_clahe_otsu_on_original(tile)
        else:
            density_clahe_orig = tile_results[-3]['density_clahe_otsu_orig']  # ❌

# NEW (fixed)
for tile_idx, (tile, pos) in enumerate(tiles_with_pos):
    # Calculate once for this tile (same for all architectures)
    density_clahe_orig, binary_orig = calculate_density_clahe_otsu_on_original(tile)

    for arch_name in ARCHITECTURES:
        # Use density_clahe_orig for all architectures ✓
```

**Why this works:**
- CLAHE+Otsu on original image is **architecture-independent**
- Calculate once per tile, use for all 3 architectures
- No list indexing needed
- Simpler and clearer logic

---

## Performance Impact

**Before fix:**
- Calculated CLAHE+Otsu once per tile (for UNet only)
- Tried to reuse for other architectures (but crashed)

**After fix:**
- Still calculates CLAHE+Otsu once per tile
- **No performance difference** (same number of calculations)

---

## Testing

### Verify Fix Locally

```python
# Test the nested loop logic
tile_results = []
tiles_with_pos = [(np.random.rand(512, 512, 3), (0, 0))] * 5

for tile_idx, (tile, pos) in enumerate(tiles_with_pos):
    density_clahe_orig = 0.5  # Mock calculation

    for arch_name in ['UNet', 'Attention_UNet', 'Attention_ResUNet']:
        tile_results.append({
            'tile_idx': tile_idx,
            'architecture': arch_name,
            'density_clahe_otsu_orig': density_clahe_orig  # Same for all archs
        })

print(f"Total results: {len(tile_results)}")  # Expected: 5 tiles × 3 archs = 15
```

### Deploy to HPC

```bash
qsub pbs_density_analysis_architecture_comparison.sh
```

**Expected behavior:**
1. Step 1: Copies best models (~30s)
2. Step 2: Runs comparison successfully (~2.5-4 hours)
3. Generates outputs in `density_analysis_architecture_comparison_YYYYMMDD_HHMMSS/`

---

## Expected Output Directory

**Directory:** `./density_analysis_architecture_comparison_YYYYMMDD_HHMMSS/`

**Contents:**
```
density_analysis_architecture_comparison_20251017_HHMMSS/
├── architecture_comparison_tile_level.csv
├── architecture_comparison_image_summary.csv
├── EXPERIMENT_INFO.json
├── density_boxplot_comparison_threshold_0p2.png
├── density_boxplot_comparison_threshold_0p5.png
├── density_boxplot_comparison_threshold_0p8.png
├── density_boxplot_comparison_threshold_0p95.png
├── density_boxplot_comparison_claheotsu_on_pred.png
├── density_boxplot_comparison_claheotsu_on_original.png
└── representative_tiles_4panel/
    ├── tiles_4panel_10240x_10240x_2025-05-29_02-22-00_002.png
    ├── tiles_4panel_1280x_1280x_2025-05-16_00-59-00_002.png
    └── ... (40 total)
```

**NOT in `best_models/`** - that directory only contains the copied model files.

---

## Summary

**Problem:** IndexError when trying to reuse CLAHE+Otsu result via list indexing

**Root Cause:** Fragile logic that assumed list had entries and correct positioning

**Fix:** Calculate CLAHE+Otsu once per tile, outside architecture loop

**Impact:** No performance change, cleaner code, bug fixed

**Files Modified:**
- `density_analysis_architecture_comparison.py` (lines 343-379)

**Status:** ✅ Ready for resubmission

**Deployment:**
```bash
qsub pbs_density_analysis_architecture_comparison.sh
```

---

**Bug Report Date:** October 17, 2025
**Fix Applied:** October 17, 2025
**Estimated Runtime:** ~2.5-4 hours
**Output:** 6 combined boxplots + 40 4-panel tile visualizations
