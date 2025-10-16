# Critical Fix: CLAHE+Otsu on Original Images - Inversion Required

## Problem Identified

The CLAHE+Otsu density analysis on **original microscope images** was showing an **opposite trend** (high dilution → high density instead of low density). This was fixed by adding an image inversion step before CLAHE+Otsu processing.

## Root Cause

### Difference Between Predicted Masks and Original Images

**Predicted masks (from UNet model):**
- Output probabilities (0-1 range)
- **High probability = Bead region (foreground)**
- Beads appear as **bright/white regions** in prediction
- When rescaled to 0-255, beads become bright pixels
- `THRESH_BINARY_INV` correctly segments bright beads as foreground

**Original microscope images:**
- Physical microscope captures
- **Dark beads on bright background** (typical brightfield microscopy)
- Beads appear as **dark/black regions** in original
- When rescaled to 0-255, beads remain dark pixels
- `THRESH_BINARY_INV` on dark beads gives **incorrect segmentation**

### Why the Opposite Trend Occurred

**Without inversion (WRONG):**
```
Original image: Dark beads (low intensity) on bright background (high intensity)
↓ Rescale to 0-255
Still dark beads on bright background
↓ CLAHE + Otsu with THRESH_BINARY_INV
Otsu finds threshold that separates dark from bright
THRESH_BINARY_INV inverts: makes BRIGHT regions foreground
Result: Background becomes foreground, beads become background (WRONG!)
```

**At 10240x (high dilution, sparse beads):**
- Mostly bright background → Becomes foreground after THRESH_BINARY_INV
- **High density** (INCORRECT - should be low!)

**At 10x (low dilution, many beads):**
- More dark beads → Becomes background after THRESH_BINARY_INV
- **Lower density** (INCORRECT - should be high!)

## Solution: Invert Before CLAHE+Otsu

**With inversion (CORRECT):**
```
Original image: Dark beads on bright background
↓ INVERT (255 - pixel_value)
Bright beads on dark background (like predicted masks!)
↓ Rescale to 0-255
Beads use full dynamic range, bright on dark
↓ CLAHE + Otsu with THRESH_BINARY_INV
Otsu finds threshold that separates bright beads from dark background
THRESH_BINARY_INV works correctly: bright regions are foreground
Result: Beads become foreground (CORRECT!)
```

**At 10240x (high dilution, sparse beads):**
- Few bright beads after inversion → Low foreground after CLAHE+Otsu
- **Low density** (CORRECT!)

**At 10x (low dilution, many beads):**
- Many bright beads after inversion → High foreground after CLAHE+Otsu
- **High density** (CORRECT!)

## Fix Applied

### Updated `calculate_density_clahe_otsu_on_original()`

```python
def calculate_density_clahe_otsu_on_original(tile):
    """
    Apply CLAHE+Otsu directly on original image tile (baseline method).

    IMPORTANT: For original microscope images, beads are DARK on BRIGHT background.
    We need to INVERT the image first so that THRESH_BINARY_INV works correctly.
    """
    # Convert to grayscale if RGB
    if len(tile.shape) == 3:
        tile_gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        tile_gray = (tile * 255).astype(np.uint8)

    # CRITICAL: Invert the grayscale image so beads become BRIGHT on DARK background
    tile_gray_inverted = 255 - tile_gray

    # CRITICAL: Rescale to full 0-255 range AFTER inversion
    tile_rescaled = rescale_image_full_range(tile_gray_inverted)

    # Apply CLAHE+Otsu
    binary_mask = apply_clahe_otsu(tile_rescaled, clipLimit=2.0, tileGridSize=(8, 8))

    # Calculate density
    density = np.count_nonzero(binary_mask) / binary_mask.size
    return density, binary_mask
```

### Key Changes

**Before (WRONG):**
```python
tile_gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
tile_rescaled = rescale_image_full_range(tile_gray)  # No inversion!
binary_mask = apply_clahe_otsu(tile_rescaled)
```

**After (CORRECT):**
```python
tile_gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
tile_gray_inverted = 255 - tile_gray  # ✅ INVERT FIRST
tile_rescaled = rescale_image_full_range(tile_gray_inverted)
binary_mask = apply_clahe_otsu(tile_rescaled)
```

## Why Predicted Masks Don't Need Inversion

**Predicted masks** are already in the correct format:
- Model outputs high probabilities for bead regions
- After rescaling, beads are **bright pixels** (high intensity)
- `THRESH_BINARY_INV` directly segments bright regions as foreground
- **No inversion needed**

## Expected Results

### Before Fix
- 10240x dilution: **High density** (opposite trend ❌)
- 10x dilution: **Low density** (opposite trend ❌)
- Boxplot shows inverted relationship

### After Fix
- 10240x dilution: **Low density** (correct trend ✅)
- 10x dilution: **High density** (correct trend ✅)
- Boxplot matches predicted mask trends

## Visual Verification

The 3-panel tile visualizations will help verify the fix:
1. **Panel 1: Original** - Shows dark beads on bright background
2. **Panel 2: Inverted Pred** - Shows bright beads (from model)
3. **Panel 3: Inverted CLAHE+Otsu** - Should show bright beads (after inversion + processing)

Panels 2 and 3 should look similar (both bright beads), confirming that original images are now processed correctly.

## Lesson Learned

**Always consider the input data format when applying image processing pipelines:**

1. **Predicted masks**: Probability maps where high values = foreground
2. **Original images**: Physical captures where foreground intensity depends on imaging modality

**For brightfield microscopy (dark beads on bright background):**
- Must invert before applying algorithms designed for bright foreground

**For fluorescence microscopy (bright signals on dark background):**
- No inversion needed, direct processing works

## Summary

**Problem:** CLAHE+Otsu on original images showed opposite trend (high dilution → high density)

**Root cause:** Original microscope images have **dark beads on bright background**, but CLAHE+Otsu pipeline expected **bright foreground on dark background**

**Solution:** **Invert original images (255 - pixel_value)** before CLAHE+Otsu processing

**Verification:** Compare boxplots before/after fix, check 3-panel tile visualizations

---

**Fixed:** October 16, 2025
**Files Modified:** `density_analysis_unet_only.py`
**Status:** ✅ Ready for testing
