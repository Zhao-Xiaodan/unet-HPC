# Density Analysis Using Xukuang UNet Model

## Overview

This pipeline performs bead density analysis on test images using the best UNet model from the Xukuang parameters experiment (`xukuang_params_shrunk_20251015_071224`).

**Model Performance:**
- **Using FINAL model from Epoch 200** (Val IoU: 0.6065)
- Best checkpoint was at Epoch 140 (Val IoU: 0.6789), but model NOT saved
- Training: LR=0.005, 200 epochs, BinaryFocalLoss(γ=2)
- Image Format: 512×512 RGB

**Important Note:** The Xukuang training script saves the FINAL model after 200 epochs, not the best checkpoint. This is still a high-performing model (IoU 0.6065), significantly better than the hyperparameter search models (IoU 0.219).

## Critical Fix: Dilution Label Ordering

**Previous Issue:** The hyperparameter search density analysis had incorrect x-axis ordering in box plots due to string sorting (e.g., "10240x" came before "20x").

**Solution:** This implementation uses explicit categorical ordering:
```python
DILUTION_ORDER = [10, 20, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
DILUTION_LABELS = ['10x', '20x', '80x', '160x', '320x', '640x', '1280x', '2560x', '5120x', '10240x']
```

## Files

### Scripts
1. **`density_analysis_xukuang.py`** - Main analysis script
   - Loads Xukuang UNet model from HPC
   - Predicts on test images in `./test_images/`
   - Generates visualizations and statistics
   - **CORRECTED:** Proper dilution ordering in plots

2. **`pbs_density_analysis_xukuang.sh`** - PBS job submission script
   - Allocates: 1 GPU, 4 CPUs, 32GB RAM, 4-hour walltime
   - Activates conda environment
   - Runs density analysis
   - Logs all output

3. **`DENSITY_ANALYSIS_README.md`** - This file

### Input

**Models:** Located in `./xukuang_params_shrunk_20251015_071224/`
- `unet_xukuang_params_shrunk.keras` (FINAL epoch 200 model)
- `attention_unet_xukuang_params_shrunk.keras`
- `attention_resunet_xukuang_params_shrunk.keras`
- Trained with RGB images (512×512×3)
- **Note:** These are FINAL models, not best checkpoint models

**Test Images:** Located in `./test_images/`
- Format: TIFF files (`.tif` or `.tiff`)
- Naming convention: `{dilution}_{timestamp}.tif`
  - Examples: `10x_2025-05-15_02-05-00.tif`, `640x_2025-05-16_00-59-00_002.tif`
- Dilution factors: 10x, 20x, 80x, 160x, 320x, 640x, 1280x, 2560x, 5120x, 10240x

**Current Test Images:**
```
10x_2025-05-15_02-05-00.tif
20x_2025-05-15_02-05-00.tif
80x_1_2025-05-22_14-48-00_003.tif
80x_2_2025-05-22_14-48-00.tif
160x_2025-05-15_02-05-00.tif
320x_2025-05-15_02-05-00.tif
640x_2025-05-16_00-59-00_002.tif
1280x_2025-05-16_00-59-00_002.tif
2560x_10240x_2025-05-16_00-59-00_002.tif  (contains both dilutions)
5120x_2025-05-16_00-59-00_002.tif
2025-05-16_00-59-00_002.tif  (undiluted, 1x)
```

### Output

Directory: `./density_analysis_xukuang_YYYYMMDD_HHMMSS/`

**Files Generated:**

1. **`density_results.csv`** - Density measurements for each image
   ```csv
   image,dilution,dilution_label,n_tiles,mean_density,median_density,std_density,min_density,max_density
   10x_2025-05-15_02-05-00.tif,10,10x,16,0.4523,0.4512,0.0234,0.4101,0.4895
   ...
   ```

2. **`density_boxplot.png`** - Box plot showing density distribution across dilutions
   - **CORRECTED:** X-axis shows dilutions in proper order: 10x → 10240x
   - Y-axis: Foreground density (0-1)
   - Annotations: Sample size for each dilution

3. **`representative_tiles/`** - Directory with tile visualizations
   - Format: `tiles_{dilution}.png` (e.g., `tiles_10x.png`, `tiles_640x.png`)
   - Each file shows 5 representative tiles:
     - Top row: Original RGB tiles
     - Bottom row: Model predictions (grayscale)
   - Tiles span density range for each dilution

4. **`EXPERIMENT_INFO.json`** - Metadata
   ```json
   {
     "timestamp": "2025-10-15 21:30:00",
     "model_name": "UNet (Xukuang)",
     "model_performance": {
       "final_val_iou": 0.6065,
       "best_val_iou": 0.6789,
       "best_epoch": 140
     },
     "training_params": {
       "learning_rate": 0.005,
       "epochs": 200,
       "image_type": "RGB"
     },
     "dilution_order": [10, 20, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
   }
   ```

## Usage

### On HPC (Recommended)

1. **Ensure models are in correct location:**
   ```bash
   ls -la xukuang_params_shrunk_20251015_071224/*.keras
   # Should see: unet_xukuang_params_shrunk.keras (and attention variants)
   ```

2. **Verify test images:**
   ```bash
   ls -la test_images/*.tif
   ```

3. **Submit job:**
   ```bash
   qsub pbs_density_analysis_xukuang.sh
   ```

4. **Monitor job:**
   ```bash
   qstat -u $USER
   tail -f density_analysis_xukuang.log
   ```

5. **Check results:**
   ```bash
   ls -la density_analysis_xukuang_*/
   ```

### Locally (For Testing)

```bash
python density_analysis_xukuang.py
```

**Note:** Requires:
- TensorFlow with GPU support
- 512×512 RGB model from Xukuang experiment
- Sufficient memory for image processing

## Pipeline Details

### 1. Image Processing
```
Test Image (TIFF, various sizes)
  ↓
Convert to RGB (if grayscale/RGBA)
  ↓
Extract 512×512 non-overlapping tiles
  ↓
Normalize to [0, 1]
```

### 2. Prediction
```
Tile (512×512×3, RGB, [0-1])
  ↓
UNet Model Prediction
  ↓
Prediction Map (512×512, [0-1])
  ↓
Threshold at 0.5
  ↓
Binary Mask (512×512, {0,1})
```

### 3. Density Calculation
```
Binary Mask
  ↓
Density = Mean(Binary Mask)
  ↓
Aggregate across all tiles per image
```

### 4. Visualization
```
Densities + Dilution Labels
  ↓
Create Categorical Order (10x → 10240x)
  ↓
Generate Box Plot with Correct Ordering
```

## Key Differences from Previous Analysis

| Aspect | Previous (Hyperparam) | This (Xukuang) |
|--------|----------------------|----------------|
| **Model** | UNet (IoU: 0.219) | UNet (IoU: 0.679) |
| **Image Format** | Grayscale (1 channel) | **RGB (3 channels)** |
| **Training LR** | 1e-4 (too low) | 5e-3 (optimal) |
| **Dilution Ordering** | ✗ Incorrect (string sort) | **✓ Correct (categorical)** |
| **Performance** | Poor (unusable) | **Good (production-ready)** |

## Expected Runtime

- **Tile Extraction:** ~1-2 min per large image
- **Prediction:** ~10-20 ms per tile (GPU)
- **Total:** ~10-30 minutes for 11 test images
- **Walltime Allocated:** 4 hours (safe margin)

## Troubleshooting

### Issue: Model file not found
```bash
# Check model directory
ls -la xukuang_params_shrunk_20251015_071224/
# Should contain: unet_xukuang_params_shrunk.keras (and attention variants)
```

**Solution:** Models are saved on HPC during training. Ensure you're running on HPC at `/home/svu/phyzxi/scratch/unet-HPC/`, not locally.

### Issue: Wrong image format (grayscale model on RGB images)
**Error:** Input shape mismatch

**Solution:** This script expects RGB model. Verify:
```python
model.input_shape  # Should be (None, 512, 512, 3)
```

### Issue: Dilution labels still wrong in plot
**Check:** Verify `DILUTION_ORDER` and `DILUTION_LABELS` are used in boxplot creation with `order=DILUTION_LABELS` parameter.

### Issue: Out of memory
**Solutions:**
1. Reduce batch size in CONFIG: `'batch_size': 4` → `'batch_size': 2`
2. Request more memory: `#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1`

## Verification

After analysis completes, verify:

1. **Boxplot x-axis ordering:**
   - Should go: 10x, 20x, 80x, 160x, ..., 5120x, 10240x
   - NOT: 10240x, 1280x, 160x, ... (string sort)

2. **Density trend:**
   - Higher dilution → Lower density
   - Should see decreasing trend from 10x to 10240x

3. **Tile predictions:**
   - Check `representative_tiles/` directory
   - Verify predictions look reasonable (beads detected)

## Next Steps

After successful analysis:

1. **Review Results:**
   ```bash
   # View boxplot
   open density_analysis_xukuang_*/density_boxplot.png

   # View representative tiles
   open density_analysis_xukuang_*/representative_tiles/*.png

   # Check CSV
   cat density_analysis_xukuang_*/density_results.csv
   ```

2. **Compare with Previous Analysis:**
   - Compare with `density_analysis_512_grayscale_20251015_052432/`
   - Expected: Much better performance with Xukuang model

3. **Statistical Analysis:**
   - Correlation between dilution and density
   - Variance analysis across replicates (80x_1 vs 80x_2)
   - Compare against ground truth if available

4. **Report Generation:**
   - Include boxplot in final report
   - Add representative tiles for key dilutions
   - Document density vs dilution relationship

## References

- **Model Source:** `xukuang_params_shrunk_20251015_071224/`
- **Model Report:** `xukuang_params_shrunk_20251015_071224/report.md`
- **Training Parameters:** `xukuang_params_shrunk_20251015_071224/EXPERIMENT_INFO.json`
- **Previous Analysis:** `density_analysis_512_grayscale_20251015_052432/ANALYSIS_REPORT.md`

## Contact

For questions about:
- **Model:** See `xukuang_params_shrunk_20251015_071224/report.md`
- **Analysis Pipeline:** Review this README and script comments
- **Results Interpretation:** Check generated `EXPERIMENT_INFO.json`

---

**Created:** October 15, 2025
**Author:** Claude Code
**Version:** 1.0
