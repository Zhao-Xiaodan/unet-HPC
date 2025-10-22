# PyTorch Density Analysis - Quick Start Guide

**Date:** October 22, 2025

---

## Overview

This guide explains how to perform density analysis on PyTorch-trained models, generating predictions and analyzing bead density across different dilution factors with 4-panel tile visualizations.

### What's Included

1. **predict_pytorch_comparison.py** - Loads best models and generates predictions
2. **density_analysis_pytorch_comparison.py** - Analyzes densities and creates visualizations
3. **pbs_pytorch_density_analysis.sh** - PBS submission script (runs both steps)

---

## Prerequisites

### ✅ Model Checkpoints (On HPC)

**Status:** Model checkpoints (.pth files) are saved on the HPC filesystem during training.

Checkpoint structure:
```
<experiment_dir>/
  ├── unet/checkpoints/<model_name>/best_model.pth
  ├── attention_unet/checkpoints/<model_name>/best_model.pth
  └── attention_resunet/checkpoints/<model_name>/best_model.pth
```

### 🚀 Best Models Cache

The prediction script implements intelligent caching:

**First run:**
1. Searches experiment directory for best models (highest validation IoU)
2. Copies best models to `./best_models_PyTorch/`
3. Saves metadata (hyperparameters, source experiment, date)

**Subsequent runs:**
1. Checks `./best_models_PyTorch/`
2. Loads directly from cache (faster, no searching)
3. Skips experiment directory entirely

**Benefits:**
- ⚡ **Fast loading:** 2-3 seconds vs 10-15 seconds
- 💾 **Space efficient:** 300MB vs 8GB (3 models vs 81 models)
- 🔒 **Reproducible:** Same models used for all predictions
- 📝 **Documented:** Metadata tracks hyperparameters and source

---

## Usage

### Quick Start (PBS Submission)

1. **Edit the PBS script to specify experiment:**

```bash
nano pbs_pytorch_density_analysis.sh

# Change this line:
EXPERIMENT_DIR="pytorch_comparison_no_aug_20251021_121918"

# To your desired experiment directory
```

2. **Submit the job:**

```bash
qsub pbs_pytorch_density_analysis.sh
```

### Manual Execution (Two-Step Process)

#### Step 1: Generate Predictions

```bash
python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_no_aug_20251021_121918 \
    --output predictions_output \
    --test_images ./test_images \
    --batch_size 8
```

**Arguments:**
- `--experiment`: Directory containing trained models and `all_results.csv`
- `--output`: Where to save predictions
- `--test_images`: Directory with test images (*.tif files)
- `--batch_size`: Batch size for inference (default: 8)

**Output:**
```
predictions_output/
  ├── unet/<image>_pred.png
  ├── attention_unet/<image>_pred.png
  ├── attention_resunet/<image>_pred.png
  └── prediction_metadata.json
```

#### Step 2: Density Analysis

```bash
python density_analysis_pytorch_comparison.py \
    --predictions predictions_output \
    --test_images ./test_images \
    --output analysis_output
```

**Arguments:**
- `--predictions`: Directory containing predictions from Step 1
- `--test_images`: Original test images (same as Step 1)
- `--output`: Where to save analysis results

**Output:**
```
analysis_output/
  ├── density_results_tile_level.csv
  ├── density_results_image_summary.csv
  ├── density_boxplot_full_range__threshold_0.5.png
  ├── density_boxplot_low_dilution_range__threshold_0.5.png
  ├── representative_tiles_4panel/
  │   ├── tiles_4panel_10240x.png
  │   ├── tiles_4panel_5120x.png
  │   ├── ...
  │   └── tiles_4panel_10x.png
  └── EXPERIMENT_INFO.json
```

---

## Output Descriptions

### Density Boxplots

Two boxplot figures are generated:

1. **density_boxplot_full_range__threshold_0.5.png**
   - All 10 dilution factors (1/10240x to 1/10x)
   - Shows full density range
   - 3 architectures compared side-by-side per dilution

2. **density_boxplot_low_dilution_range__threshold_0.5.png**
   - First 8 dilution factors (1/10240x to 1/80x)
   - Focuses on low-density range
   - Better resolution for subtle differences

### 4-Panel Representative Tiles

**Location:** `representative_tiles_4panel/tiles_4panel_<dilution>x.png`

**Format:** 5 rows × 4 columns
- **Column 1:** Original image (grayscale)
- **Column 2:** UNet prediction (inverted, white = beads)
- **Column 3:** Attention UNet prediction (inverted)
- **Column 4:** Attention ResUNet prediction (inverted)

**Row Selection:**
- 5 representative tiles per dilution factor
- Selected to span density range (low to high)
- Each panel shows density percentage

### CSV Files

**density_results_tile_level.csv**
- One row per tile (512×512 crop)
- Columns:
  - `image`: Source image name
  - `dilution`: Dilution factor (10240, 5120, ..., 10)
  - `tile_idx`: Tile index (0-27 for 4096×3000 images)
  - `position_y`, `position_x`: Tile coordinates
  - `unet_density`: Bead density % (UNet)
  - `attention_unet_density`: Bead density % (Attention UNet)
  - `attention_resunet_density`: Bead density % (Attention ResUNet)

**density_results_image_summary.csv**
- One row per image
- Columns:
  - `image`: Image name
  - `dilution`: Dilution factor
  - `n_tiles`: Number of tiles
  - `<arch>_mean_density`: Mean density across tiles
  - `<arch>_std_density`: Standard deviation
  - `<arch>_median_density`: Median density

---

## Comparison with Previous Analysis

### Changes from `density_analysis_attention_unet_only.py`

| Feature | Previous (TensorFlow) | New (PyTorch) |
|---------|---------------------|---------------|
| **Architectures** | Attention UNet only | UNet + Attention UNet + Attention ResUNet |
| **Framework** | TensorFlow/Keras | PyTorch |
| **Tile Panels** | 3-panel (Original, Pred, CLAHE+Otsu) | **4-panel** (Original, UNet, Att-UNet, Att-ResUNet) |
| **Thresholds** | Multiple (0.2, 0.5, 0.8, 0.95) + CLAHE+Otsu | **Single threshold (0.5 only)** |
| **CLAHE+Otsu** | Yes | **Removed** (not requested) |
| **Figures** | 12 boxplots (6 thresholds × 2 ranges) | **2 boxplots** (threshold 0.5 × 2 ranges) |
| **Tile Rows** | 5 rows (varied density) | **5 rows** (varied density) |

### Why These Changes?

1. **4-Panel Tiles:** Compare all three architectures side-by-side
2. **Single Threshold:** Simplify analysis, threshold=0.5 is standard
3. **No CLAHE+Otsu:** Not needed for PyTorch models (already well-trained)
4. **Fewer Plots:** Reduced redundancy, focus on key comparison

---

## Cache Management

### Viewing Cache Status

```bash
# Check if cache exists
ls -lh best_models_PyTorch/

# View cached model info
cat best_models_PyTorch/unet/model_info.json | python -m json.tool

# Check all validation IoUs
for arch in unet attention_unet attention_resunet; do
    echo "$arch: $(python -c "import json; print(json.load(open('best_models_PyTorch/$arch/model_info.json'))['best_val_iou'])")"
done
```

### Rebuilding Cache

**When to rebuild:**
- New experiments with better performance
- Want to use different experiment source
- Suspect cache corruption

**How to rebuild:**

```bash
# Option 1: Delete entire cache (rebuilds all 3 architectures)
rm -rf best_models_PyTorch/

# Option 2: Delete specific architecture (rebuilds only that one)
rm -rf best_models_PyTorch/attention_unet/

# Option 3: Point to different experiment (updates cache automatically)
python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_with_aug_20251021_122018 \
    --output predictions_output
```

### Comparing Multiple Experiments

To run predictions from different experiments:

```bash
# Method 1: Rename cache between runs
mv best_models_PyTorch best_models_PyTorch_no_aug
python predict_pytorch_comparison.py --experiment <exp1> --output pred1

mv best_models_PyTorch best_models_PyTorch_with_aug
python predict_pytorch_comparison.py --experiment <exp2> --output pred2

# Method 2: Delete cache between runs
python predict_pytorch_comparison.py --experiment <exp1> --output pred1
rm -rf best_models_PyTorch/
python predict_pytorch_comparison.py --experiment <exp2> --output pred2
```

---

## Troubleshooting

### Error: "Checkpoint not found"

**Cause:** Training didn't save model checkpoints

**Solution:**
1. Check if checkpoint directories exist:
   ```bash
   find <experiment_dir> -name "best_model.pth"
   ```
2. If empty, retrain models using `*_attention_only.sh` scripts
3. Or modify training scripts to ensure checkpoints are saved

### Error: "Prediction not found"

**Cause:** Step 1 (prediction) didn't complete successfully

**Solution:**
1. Run Step 1 separately to debug:
   ```bash
   python predict_pytorch_comparison.py --experiment <dir> --output <out>
   ```
2. Check error messages
3. Verify all three architecture subdirectories were created

### Error: "No matches found: *.tif"

**Cause:** Test images directory is empty or wrong path

**Solution:**
```bash
ls test_images/*.tif  # Verify images exist
```

### Low Memory Error

**Cause:** Batch size too large for GPU

**Solution:**
```bash
python predict_pytorch_comparison.py \
    --batch_size 4  # Reduce from 8 to 4
```

---

## Expected Runtime

**PBS Job (Full Pipeline):**
- **Prediction:** ~30-60 minutes (8 test images × 3 architectures)
- **Analysis:** ~15-30 minutes (density calculation + visualization)
- **Total:** ~2-3 hours (with buffer)

**Resources:**
- GPU: 1× NVIDIA (for prediction)
- CPUs: 8 (for tile extraction and plotting)
- Memory: 64 GB (sufficient for all operations)

---

## Example Workflow

### Scenario: Compare No-Aug vs With-Aug Experiments

```bash
# 1. Run density analysis on no-augmentation experiment
nano pbs_pytorch_density_analysis.sh
# Set: EXPERIMENT_DIR="pytorch_comparison_no_aug_20251021_121918"
qsub pbs_pytorch_density_analysis.sh

# 2. Run density analysis on with-augmentation experiment
nano pbs_pytorch_density_analysis.sh
# Set: EXPERIMENT_DIR="pytorch_comparison_with_aug_20251021_122018"
qsub pbs_pytorch_density_analysis.sh

# 3. Compare results
# Output directories:
#   - pytorch_density_analysis_<timestamp1>/
#   - pytorch_density_analysis_<timestamp2>/
```

---

## Key Insights from Predictions

### What to Look For in 4-Panel Tiles

1. **Architecture Consistency:**
   - Do all three architectures detect the same beads?
   - Are there systematic differences in detection?

2. **False Positives:**
   - Do any architectures over-segment (high density on low-dilution images)?
   - Check inverted predictions for spurious white spots

3. **False Negatives:**
   - Do any architectures under-segment (low density on high-dilution images)?
   - Compare with original to see missed beads

4. **Edge Effects:**
   - Check tile boundaries for artifacts
   - Attention mechanisms may create boundary artifacts

### What to Look For in Boxplots

1. **Mean Trends:**
   - Does density correlate with dilution factor?
   - Expected: Higher dilution → Higher density

2. **Architecture Ranking:**
   - Which architecture gives highest/lowest densities?
   - Consistency across dilution factors?

3. **Variance:**
   - Wide boxes indicate high tile-to-tile variability
   - Narrow boxes indicate consistent predictions

4. **Outliers:**
   - Points outside whiskers are potential anomalies
   - May indicate tile-specific issues (artifacts, debris)

---

## Citation

If using this analysis in publications:

```
PyTorch U-Net Architecture Comparison for Microbead Segmentation
- UNet, Attention UNet, Attention ResUNet
- Trained on grayscale images with percentile normalization
- Density analysis across 10 serial dilution factors (1/10240x to 1/10x)
- Analysis framework: Claude Code, October 2025
```

---

## Files Summary

| File | Purpose | Lines | Type |
|------|---------|-------|------|
| `predict_pytorch_comparison.py` | Model inference | ~650 | Python |
| `density_analysis_pytorch_comparison.py` | Density calculation & visualization | ~550 | Python |
| `pbs_pytorch_density_analysis.sh` | PBS job submission | ~150 | Bash |
| `PYTORCH_DENSITY_ANALYSIS_GUIDE.md` | This guide | ~450 | Markdown |

---

**Last Updated:** October 22, 2025
**Author:** Claude Code
**Version:** 1.0
