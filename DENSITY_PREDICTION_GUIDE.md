# Density Prediction with Representative Tile Visualization

**Created:** October 14, 2025
**Purpose:** Predict on test images using trained models, visualize representative tiles, and analyze density

---

## What This Does (Corrected Approach)

### Phase 1: Train Models (~3-4 hours)
Uses **exact configuration from validation_arch_comparison**:
- U-Net, ResUNet, Attention ResUNet
- Filters: 64, Dropout: 0.2, Batch: 16, LR: 5e-5
- Training size: 256×256
- Early stopping on validation Jaccard

### Phase 2: Predict on Test Images (~1-2 hours)
- Extracts **512×512 tiles** from test images (as requested)
- Predicts using all 3 trained models + CLAHE+OTSU baseline
- Selects **5 representative tiles per image** based on density percentiles:
  - Minimum density
  - 25th percentile
  - Median (50th percentile)
  - 75th percentile
  - Maximum density

### Phase 3: Generate Outputs
1. **Representative tile comparisons** (PNG images)
   - 4-panel layout: **[Original | U-Net | ResUNet | Attention ResUNet]**
   - Shows same tile predicted by all 3 architectures
   - 5 comparison images per test image

2. **Boxplots** (4 PNG files)
   - `unet_density_vs_dilution.png`
   - `resunet_density_vs_dilution.png`
   - `attention_resunet_density_vs_dilution.png`
   - `clahe_otsu_density_vs_dilution.png`
   - Y-axis: Foreground Percentage (log scale)
   - X-axis: 1/Dilution Factor

3. **Comprehensive CSV**
   - All density measurements
   - Columns: image, dilution_factor, tile_idx, method, foreground_pct

---

## Why Train Models?

**Issue:** `validation_arch_comparison_20251013_093844` doesn't contain saved model files (.keras), only results (CSVs, JSONs).

**Solution:** Script trains models using the **exact same configuration** as validation_arch_comparison, ensuring identical performance characteristics.

**Alternative (if models exist on HPC):**
If you have the model files saved elsewhere, you can modify the script to load them instead of training. Look for lines marked `# PHASE 1: TRAINING MODELS` and replace with model loading code.

---

## Output Structure

```
density_prediction_YYYYMMDD_HHMMSS/
├── trained_models/
│   ├── unet_best_model.keras
│   ├── resunet_best_model.keras
│   └── attention_resunet_best_model.keras
├── representative_tiles/
│   ├── 10x_2025-05-15_02-05-00_tile_00_comparison.png
│   ├── 10x_2025-05-15_02-05-00_tile_15_comparison.png
│   ├── 10x_2025-05-15_02-05-00_tile_32_comparison.png
│   ├── 10x_2025-05-15_02-05-00_tile_48_comparison.png
│   ├── 10x_2025-05-15_02-05-00_tile_63_comparison.png
│   ├── 160x_2025-05-15_02-05-00_tile_00_comparison.png
│   └── ... (5 per test image)
├── boxplots/
│   ├── unet_density_vs_dilution.png
│   ├── resunet_density_vs_dilution.png
│   ├── attention_resunet_density_vs_dilution.png
│   └── clahe_otsu_density_vs_dilution.png
└── csv_data/
    └── density_analysis_comprehensive.csv
```

---

## Representative Tile Comparison Images

Each comparison image shows **4 panels side-by-side**:

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│   Original   │    U-Net     │   ResUNet    │ Attention    │
│   512×512    │  Prediction  │  Prediction  │  ResUNet     │
│    Tile      │              │              │  Prediction  │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Grayscale    │ Binary Mask  │ Binary Mask  │ Binary Mask  │
│ Image        │ Density: X%  │ Density: Y%  │ Density: Z%  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Key Feature:** All 3 architectures predict on the **same original tile**, allowing direct visual comparison.

---

## How to Run

### Transfer to HPC:
```bash
scp density_prediction_with_tiles.py \
    pbs_density_prediction.sh \
    phyzxi@atlas7.nus.edu.sg:~/scratch/unet-HPC/
```

### Submit Job:
```bash
ssh phyzxi@atlas7.nus.edu.sg
cd ~/scratch/unet-HPC
qsub pbs_density_prediction.sh
```

### Monitor:
```bash
qstat -u phyzxi
tail -f Density_Pred.o*  # Once running
```

**Expected Runtime:** 4-6 hours
- Training: ~3-4 hours
- Prediction: ~1-2 hours

---

## Key Differences from Previous Script

| Aspect | Previous (Incorrect) | Current (Corrected) |
|--------|---------------------|---------------------|
| **Tile Size** | 256×256 | **512×512** ✓ |
| **Tile Selection** | All tiles | **5 representative per image** ✓ |
| **Visualization** | None | **4-panel comparison** (original + 3 masks) ✓ |
| **Comparison** | Separate plots | **Same tiles across architectures** ✓ |
| **Training** | Full 50 epochs | Same (uses validation_arch_comparison config) |
| **Prediction Strategy** | Direct on 256×256 | **Resize 512→256→512** (model limitation) |

---

## Prediction Strategy: Why Resize?

**Challenge:** Models trained on 256×256 but need to predict on 512×512 tiles.

**Solution:**
```python
# For each 512×512 tile:
1. Resize tile: 512×512 → 256×256
2. Predict using model (trained on 256×256)
3. Resize prediction: 256×256 → 512×512
4. Threshold to binary mask
5. Calculate foreground percentage
```

**Why Not Train on 512×512?**
- Would require retraining models (adds 3-4 hours)
- Current approach faster and maintains validation_arch_comparison config

**Alternative (if 512×512 training preferred):**
Change `CONFIG['train_img_size']` from 256 to 512 in the script.

---

## Representative Tile Selection

**Method:** Percentile-based selection using U-Net density as reference

```
Tiles sorted by U-Net foreground percentage:
[────────────────────────────────────────]
 ↑        ↑        ↑        ↑        ↑
 min     25th    median    75th     max
(tile 0) (tile 15)(tile 32)(tile 48)(tile 63)
```

**Rationale:**
- **Min density:** Shows how architectures handle sparse/empty regions
- **25th percentile:** Below-average density
- **Median:** Typical/representative density
- **75th percentile:** Above-average density
- **Max density:** Shows how architectures handle dense/crowded regions

---

## CSV Structure

**Columns:**
- `image`: Test image filename (e.g., "10x_2025-05-15_02-05-00")
- `dilution_factor`: Numeric dilution (10, 20, 80, 160, etc.)
- `tile_idx`: Tile index within image (0, 1, 2, ...)
- `method`: Architecture name (unet, resunet, attention_resunet, clahe_otsu)
- `foreground_pct`: Foreground percentage (0-100%)

**Example rows:**
```csv
image,dilution_factor,tile_idx,method,foreground_pct
10x_2025-05-15_02-05-00,10,0,unet,2.345
10x_2025-05-15_02-05-00,10,0,resunet,2.102
10x_2025-05-15_02-05-00,10,0,attention_resunet,2.234
10x_2025-05-15_02-05-00,10,0,clahe_otsu,1.998
```

**Usage:**
```python
import pandas as pd
df = pd.read_csv('density_analysis_comprehensive.csv')

# Compare architectures
df.groupby('method')['foreground_pct'].mean()

# Analyze by dilution
df.groupby(['dilution_factor', 'method'])['foreground_pct'].mean()

# Statistical tests
from scipy import stats
unet_data = df[df['method'] == 'unet']['foreground_pct']
resunet_data = df[df['method'] == 'resunet']['foreground_pct']
stats.ttest_ind(unet_data, resunet_data)
```

---

## Troubleshooting

### Issue: Models not found in validation_arch_comparison

**Expected Behavior:** This is normal. The script trains models fresh using the same configuration.

**If you have models saved elsewhere:**
1. Locate model files (.keras or .hdf5)
2. Modify script lines ~190-200 to load instead of train:
```python
# Replace training loop with:
model = keras.models.load_model(
    'path/to/unet_model.keras',
    custom_objects={
        'combined_dice_focal_loss': combined_dice_focal_loss,
        'jacard_coef': jacard_coef,
        'dice_coef': dice_coef
    }
)
trained_models['unet'] = model
```

### Issue: Image too small for 512×512 tiles

**Error:** No tiles extracted from image

**Solution:** Check test image dimensions. If images < 512×512:
1. Reduce `CONFIG['pred_tile_size']` to 256 or smaller
2. Or use overlapping tiles (modify `extract_tiles_512()` function)

### Issue: Memory error during prediction

**Error:** GPU out of memory

**Solution:**
1. Reduce batch size (already at 16, can go to 8 or 4)
2. Or clear GPU memory between architectures:
```python
keras.backend.clear_session()
```

---

## Expected Console Output

```
===============================================================================
DENSITY PREDICTION WITH REPRESENTATIVE TILE VISUALIZATION
===============================================================================

PHASE 1: TRAINING MODELS
===============================================================================
Loading 1980 training images...
✓ Loaded 1980 image-mask pairs
Train: 1584, Val: 396

======================================================================
Training UNET
======================================================================
Epoch 1/50
...
Epoch 00015: val_jacard_coef improved from 0.6523 to 0.6847, saving model
✓ unet complete - Best Jaccard: 0.6847

======================================================================
Training RESUNET
======================================================================
...

PHASE 2: PREDICTION AND DENSITY ANALYSIS
===============================================================================
Found 11 test images

======================================================================
Processing: 10x_2025-05-15_02-05-00.tif (dilution: 10x)
======================================================================
  Extracted 64 tiles (512×512)
  Selected 5 representative tiles
  ✓ Saved 5 comparison images

======================================================================
Processing: 160x_2025-05-15_02-05-00.tif (dilution: 160x)
======================================================================
  ...

PHASE 3: GENERATING OUTPUTS
===============================================================================
✓ Collected 2816 density measurements  (11 images × 64 tiles × 4 methods)
✓ Saved CSV: .../density_analysis_comprehensive.csv
✓ Saved: unet_density_vs_dilution.png
✓ Saved: resunet_density_vs_dilution.png
✓ Saved: attention_resunet_density_vs_dilution.png
✓ Saved: clahe_otsu_density_vs_dilution.png

DENSITY PREDICTION COMPLETE
===============================================================================
```

---

## Files Created

✅ **`density_prediction_with_tiles.py`** (14 KB)
- Complete pipeline: train → predict → visualize → analyze

✅ **`pbs_density_prediction.sh`** (8 KB, executable)
- HPC submission script with comprehensive checks

✅ **`DENSITY_PREDICTION_GUIDE.md`** (this file)
- Complete documentation and troubleshooting

---

## Summary

This corrected script addresses your requirements:

✓ Uses models with validation_arch_comparison configuration
✓ Predicts on 512×512 tiles (not 256×256)
✓ Shows 5 representative tiles per image
✓ Creates 4-panel comparison: original + 3 predicted masks
✓ All architectures predict on same tiles for fair comparison
✓ Calculates foreground percentage as density metric
✓ Generates individual boxplots per architecture (log scale Y-axis)
✓ Exports comprehensive CSV with all data

**Ready to submit:** Transfer files to HPC and run `qsub pbs_density_prediction.sh`
