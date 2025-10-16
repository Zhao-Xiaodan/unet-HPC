# Critical Fix: Rescale Before CLAHE+Otsu

## Problem Identified

The CLAHE+Otsu analysis was failing because of a **missing rescaling step**. The previous working implementation in `density_analysis_arch_comparison_20251014_004358` had a critical preprocessing step that was initially overlooked.

## Root Cause

### What Was Missing

**`rescale_image_full_range()` function:**
```python
def rescale_image_full_range(img):
    """Rescale image to full 0-255 range."""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)
```

### Why This is Critical

**CLAHE and Otsu thresholding work best on images with full dynamic range (0-255).**

**Without rescaling:**
- Predicted masks might have values in a narrow range (e.g., 0.1-0.4)
- When converted to uint8, this becomes (25-102) instead of (0-255)
- CLAHE has less contrast to enhance
- Otsu threshold finds suboptimal separation point
- **Result:** Poor segmentation, incorrect densities

**With rescaling:**
- Image is stretched to use full 0-255 range
- CLAHE has full dynamic range to work with
- Otsu finds optimal threshold for the data
- **Result:** Clean segmentation, accurate densities

## Fix Applied

### 1. Added Rescale Function
```python
def rescale_image_full_range(img):
    """
    Rescale image to full 0-255 range.
    This is CRITICAL for CLAHE+Otsu to work properly.
    """
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)
```

### 2. Updated CLAHE+Otsu on Predicted Mask

**Before (WRONG):**
```python
def calculate_density_with_clahe_otsu(prediction):
    pred_uint8 = (prediction.squeeze() * 255).astype(np.uint8)  # ❌ Direct scale
    binary_mask = apply_clahe_otsu(pred_uint8)
    ...
```

**After (CORRECT):**
```python
def calculate_density_with_clahe_otsu(prediction):
    pred_rescaled = rescale_image_full_range(prediction.squeeze())  # ✅ Rescale first!
    binary_mask = apply_clahe_otsu(pred_rescaled)
    ...
```

### 3. Updated CLAHE+Otsu on Original Image

**Before (WRONG):**
```python
def calculate_density_clahe_otsu_on_original(tile):
    tile_gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)  # ❌ Direct scale
    binary_mask = apply_clahe_otsu(tile_gray)
    ...
```

**After (CORRECT):**
```python
def calculate_density_clahe_otsu_on_original(tile):
    tile_gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    tile_rescaled = rescale_image_full_range(tile_gray)  # ✅ Rescale first!
    binary_mask = apply_clahe_otsu(tile_rescaled)
    ...
```

## Example: Why Rescaling Matters

### Scenario: Predicted mask with narrow range

**Original prediction values:**
```
min: 0.1, max: 0.4, range: 0.3
```

**Without rescaling (wrong):**
```python
pred_uint8 = prediction * 255
# Result: min=25, max=102, range=77
# Only using 30% of available dynamic range (77/255)
# CLAHE and Otsu work on limited range → poor results
```

**With rescaling (correct):**
```python
pred_rescaled = rescale_image_full_range(prediction)
# Step 1: Normalize to 0-1: (prediction - 0.1) / 0.3
# Step 2: Scale to 0-255: result * 255
# Result: min=0, max=255, range=255
# Using FULL dynamic range → excellent results
```

## Updated Tile Visualizations

To help diagnose issues like this in the future, tile visualizations now show **5 panels**:

1. **Original tile** - Raw microscope image
2. **Predicted mask** - Model output (0-1 range)
3. **CLAHE+Otsu on prediction** - Rescaled + denoised (0-255 binary)
4. **Inverted prediction** - White beads on black
5. **Inverted CLAHE+Otsu** - White beads, denoised

This allows visual comparison of:
- How much rescaling changes the result
- Whether CLAHE+Otsu is working properly
- Noise reduction effectiveness

## Verification

### Before Fix (density_analysis_unet_only_20251016_005738)
- CLAHE+Otsu densities likely incorrect
- Poor noise reduction
- Suboptimal thresholding

### After Fix (Next Run)
- CLAHE+Otsu densities should match expectations
- Effective noise reduction in low-density scenarios
- Optimal automatic thresholding

## Lesson Learned

**Always check the full pipeline of working reference code**, especially for:
1. Preprocessing steps (rescaling, normalization)
2. Data range assumptions (0-1 vs 0-255)
3. Image format conversions

**CLAHE+Otsu requires:**
1. uint8 format (0-255)
2. **Full dynamic range** (rescaled)
3. Consistent THRESH_BINARY_INV for black beads

## Files Modified

### density_analysis_unet_only.py
- ✅ Added `rescale_image_full_range()` function
- ✅ Updated `calculate_density_with_clahe_otsu()` to rescale before CLAHE+Otsu
- ✅ Updated `calculate_density_clahe_otsu_on_original()` to rescale before CLAHE+Otsu
- ✅ Updated both functions to return binary masks for visualization
- ✅ Updated tile visualizations to 5-panel layout
- ✅ Added binary masks to tile_data storage

## Summary

**Critical missing step:** `rescale_image_full_range()` before CLAHE+Otsu

**Why it matters:** CLAHE and Otsu need full 0-255 dynamic range to work optimally

**Fix:** Rescale all images to 0-255 BEFORE applying CLAHE+Otsu

**Verification:** 5-panel tile visualizations show the effect of each step

---

**Fixed:** October 16, 2025
**Based on:** density_analysis_arch_comparison_20251014_004358
**Status:** ✅ Ready for testing
