# CLAHE+Otsu Denoising Update

## Overview

Added CLAHE+Otsu preprocessing to handle noise in low-density scenarios, based on the approach from `density_analysis_arch_comparison_20251014_004358`.

## Three Density Calculation Methods

The updated analysis now calculates density using **three different methods** for comparison:

###  1. Simple Threshold (0.5) - Original Method
```python
density = mean(prediction > 0.5)
```
**Pros:** Simple, direct interpretation
**Cons:** Sensitive to noise in low-density predictions

### 2. CLAHE+Otsu on Predicted Mask - Denoising
```python
pred_uint8 = (prediction * 255).astype(uint8)
clahe_img = CLAHE(pred_uint8)
binary_mask = Otsu_threshold(clahe_img, THRESH_BINARY_INV)
density = count_nonzero(binary_mask) / size
```
**Pros:** Reduces noise in predictions, better for low-density scenarios
**Cons:** Additional processing step
**Use case:** **Recommended for final density measurements**

### 3. CLAHE+Otsu on Original Image - Baseline
```python
original_gray = rgb2gray(original_tile)
clahe_img = CLAHE(original_gray)
binary_mask = Otsu_threshold(clahe_img, THRESH_BINARY_INV)
density = count_nonzero(binary_mask) / size
```
**Pros:** Traditional CV baseline, no ML model required
**Cons:** May not capture complex bead patterns
**Use case:** Comparison baseline to show ML model improvement

## Why CLAHE+Otsu?

### Problem: Noise in Low-Density Predictions
- At high dilutions (5120x, 10240x), beads are sparse
- Model predictions may have noise/artifacts
- Simple thresholding at 0.5 can be affected by noisy pixels

### Solution: CLAHE + Otsu Thresholding

**CLAHE (Contrast Limited Adaptive Histogram Equalization):**
- Enhances local contrast
- Parameters: `clipLimit=2.0`, `tileGridSize=(8,8)`
- Makes faint beads more visible

**Otsu Thresholding:**
- Automatically finds optimal threshold
- `THRESH_BINARY_INV`: Inverts because beads are dark (foreground=white)
- Reduces noise by finding natural intensity separation

## Implementation Details

### Applied to Predicted Masks (Method 2)

```python
def calculate_density_with_clahe_otsu(prediction, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    Calculate density from predicted mask after applying CLAHE+Otsu denoising.
    This reduces noise in low-density scenarios.
    """
    # Convert prediction (0-1 float) to uint8 (0-255)
    pred_uint8 = (prediction.squeeze() * 255).astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_img = clahe.apply(pred_uint8)

    # Apply Otsu thresholding (INV because beads are black)
    _, binary_mask = cv2.threshold(clahe_img, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate density
    density = np.count_nonzero(binary_mask) / binary_mask.size
    return density
```

### Applied to Original Images (Method 3)

```python
def calculate_density_clahe_otsu_on_original(tile):
    """
    Apply CLAHE+Otsu directly on original image tile (baseline method).
    This serves as a traditional CV baseline for comparison.
    """
    # Convert RGB to grayscale if needed
    if len(tile.shape) == 3:
        tile_gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        tile_gray = (tile * 255).astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(tile_gray)

    # Apply Otsu thresholding
    _, binary_mask = cv2.threshold(clahe_img, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate density
    density = np.count_nonzero(binary_mask) / binary_mask.size
    return density
```

## Output Files

### CSV Data

**density_results_tile_level.csv:**
```csv
image,dilution,dilution_label,tile_idx,position_y,position_x,density_threshold,density_clahe_otsu_pred,density_clahe_otsu_orig
10x_img.tif,10,10x,0,0,0,0.4523,0.4612,0.4501
10x_img.tif,10,10x,1,0,512,0.4601,0.4689,0.4578
...
```

**density_results_image_summary.csv:**
```csv
image,dilution,dilution_label,n_tiles,mean_density_threshold,median_density_threshold,std_density_threshold,mean_density_clahe_otsu_pred,median_density_clahe_otsu_pred,std_density_clahe_otsu_pred,mean_density_clahe_otsu_orig,median_density_clahe_otsu_orig,std_density_clahe_otsu_orig
10x_img.tif,10,10x,28,0.4523,0.4512,0.0234,0.4612,0.4599,0.0198,0.4501,0.4488,0.0221
...
```

### Boxplots (6 Total)

**Method 1: Threshold (0.5)**
- `density_boxplot_full_range_threshold_0.5.png` (1/10 - 1/10240)
- `density_boxplot_low_dilution_range_threshold_0.5.png` (1/80 - 1/10240)

**Method 2: CLAHE+Otsu on Predicted Mask** ⭐ **Recommended**
- `density_boxplot_full_range_claheotsu_on_pred.png` (1/10 - 1/10240)
- `density_boxplot_low_dilution_range_claheotsu_on_pred.png` (1/80 - 1/10240)

**Method 3: CLAHE+Otsu on Original Image (Baseline)**
- `density_boxplot_full_range_claheotsu_on_original.png` (1/10 - 1/10240)
- `density_boxplot_low_dilution_range_claheotsu_on_original.png` (1/80 - 1/10240)

### Tile Visualizations

**representative_tiles_3panel/**
- 5 representative tiles per test image
- 3 panels per tile:
  1. **Original:** Actual microscope image
  2. **Predicted:** Model output with **both** density values shown:
     - Threshold: X.XXXX
     - CLAHE+Otsu: X.XXXX
  3. **Inverted:** White beads on black (easy comparison)

## Expected Results

### Low-Density Scenarios (High Dilution)

**Before (Threshold 0.5):**
- May include noise pixels
- Higher variance
- Less stable measurements

**After (CLAHE+Otsu on Pred):**
- Reduced noise
- Lower variance
- More stable measurements
- Better separation of true beads from artifacts

### Comparison

You can now compare:
1. **ML Model vs Traditional CV:** Compare Method 2 vs Method 3
2. **Denoising Effect:** Compare Method 1 vs Method 2
3. **Density Trends:** All methods should show similar trends, but Method 2 should be cleanest

## Usage

The updated script automatically calculates all three densities for every tile. No configuration changes needed.

```bash
# Same command as before
qsub pbs_density_analysis_unet_only.sh
```

## Analysis Recommendations

### Primary Method
Use **CLAHE+Otsu on Predicted Mask** (Method 2) for:
- Final density measurements
- Statistical analysis
- Publication figures

### Comparison
Include **Threshold** (Method 1) to show:
- Effect of denoising
- Noise levels at different dilutions

### Baseline
Include **CLAHE+Otsu on Original** (Method 3) to show:
- ML model improvement over traditional CV
- Value of learned features

## Technical Notes

### CLAHE Parameters
```python
clipLimit = 2.0       # Limits contrast amplification
tileGridSize = (8, 8) # 8×8 pixel tiles for local processing
```

These values are from `density_analysis_arch_comparison_20251014_004358` and work well for bead detection.

### Otsu Thresholding
```python
cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
```
- `THRESH_OTSU`: Automatically finds optimal threshold
- `THRESH_BINARY_INV`: Inverts (because beads are dark)
- Result: White foreground (beads), black background

### Why Not Preprocess Training Data?
CLAHE+Otsu is **not** applied during training because:
1. Model was trained on raw images (no CLAHE+Otsu preprocessing)
2. Applying it during prediction would create train/test mismatch
3. Instead, we apply it **post-prediction** to denoise the output

This is why we have two CLAHE+Otsu methods:
- **On predicted mask:** Denoises ML output
- **On original:** Traditional CV baseline

## File Changes

### Modified Files
- `density_analysis_unet_only.py`:
  - Added `apply_clahe_otsu()` function
  - Added `calculate_density_with_clahe_otsu()` for predicted masks
  - Added `calculate_density_clahe_otsu_on_original()` for baseline
  - Updated `predict_on_test_images()` to calculate all 3 densities
  - Updated boxplot functions to accept `density_column` parameter
  - Updated tile visualizations to show both density values
  - Updated CSV output to include all 3 density methods

### New Output Files (per run)
- 6 boxplot PNGs (was 2)
- CSV with 3 density columns per tile (was 1)
- Summary CSV with statistics for all 3 methods

## Summary

**Problem:** Noise in low-density predictions affects accuracy

**Solution:** CLAHE+Otsu denoising on predicted masks

**Result:** 3 density calculation methods for comprehensive analysis

**Recommendation:** Use CLAHE+Otsu on predicted mask (Method 2) for final measurements

---

**Updated:** October 16, 2025
**Based on:** density_analysis_arch_comparison_20251014_004358
**Status:** ✅ Ready for deployment
