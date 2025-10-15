# Multi-Model Density Analysis Using Xukuang Models

## Overview

This pipeline performs comprehensive bead density analysis on test images using **ALL THREE models** from the Xukuang parameters experiment (`xukuang_params_shrunk_20251015_071224`):

1. **UNet** (Final Val IoU: 0.6065, Best: 0.6789)
2. **Attention UNet**
3. **Attention ResUNet**

**Key Features:**
- ✅ **Tile-level density tracking** - Save ALL individual tile densities (n=28 per image)
- ✅ **Multi-model comparison** - Compare all three architectures side-by-side
- ✅ **4-panel tile visualizations** - Original + 3 model predictions
- ✅ **Correct dilution ordering** - Categorical ordering (10x → 10240x)

## Training Details

| Parameter | Value |
|-----------|-------|
| **Learning Rate** | 0.005 |
| **Epochs** | 200 |
| **Loss Function** | BinaryFocalLoss(γ=2, α=0.25) |
| **Image Format** | 512×512 RGB (3 channels) |
| **Best UNet IoU** | 0.6789 (epoch 140) |
| **Final UNet IoU** | 0.6065 (epoch 200) |

**Important Note:** The saved models are from FINAL epoch (200), not best checkpoint (140).

## Files

### Scripts
1. **`density_analysis_xukuang.py`** - Main multi-model analysis script
   - Loads all three models
   - Predicts on test images with each model
   - Saves tile-level density values
   - Generates comprehensive visualizations
   - **CORRECTED:** Proper dilution ordering in plots

2. **`pbs_density_analysis_xukuang.sh`** - PBS job submission script
   - Allocates: 1 GPU, 36 CPUs, 240GB RAM, **8-hour walltime** (increased for 3 models)
   - Uses TensorFlow 2.16.1 Singularity container
   - Runs density analysis and logs all output

3. **`DENSITY_ANALYSIS_MULTIMODEL_README.md`** - This file

### Input

**Models:** Located in `./xukuang_params_shrunk_20251015_071224/`
- `unet_xukuang_params_shrunk.keras`
- `attention_unet_xukuang_params_shrunk.keras`
- `attention_resunet_xukuang_params_shrunk.keras`
- Trained with RGB images (512×512×3)
- **Note:** These are FINAL models (epoch 200), not best checkpoint (140)

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

Directory: `./density_analysis_xukuang_multimodel_YYYYMMDD_HHMMSS/`

**Files Generated:**

#### 1. **`density_results_tile_level.csv`** - ALL tile-level densities
   ```csv
   image,dilution,dilution_label,tile_idx,position_y,position_x,model,density
   10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,unet,0.4523
   10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,attention_unet,0.4512
   10x_2025-05-15_02-05-00.tif,10,10x,0,0,0,attention_resunet,0.4534
   10x_2025-05-15_02-05-00.tif,10,10x,1,0,512,unet,0.4601
   ...
   ```
   - **Every tile** (n=28 per image) has a row for **each model**
   - Total rows: n_images × n_tiles × n_models

#### 2. **`density_results_image_summary.csv`** - Image-level summaries
   ```csv
   image,dilution,dilution_label,model,n_tiles,mean_density,median_density,std_density,min_density,max_density
   10x_2025-05-15_02-05-00.tif,10,10x,unet,28,0.4523,0.4512,0.0234,0.4101,0.4895
   10x_2025-05-15_02-05-00.tif,10,10x,attention_unet,28,0.4489,0.4478,0.0241,0.4078,0.4862
   10x_2025-05-15_02-05-00.tif,10,10x,attention_resunet,28,0.4556,0.4545,0.0228,0.4123,0.4912
   ...
   ```
   - Aggregated statistics per image per model

#### 3. **`density_boxplot_multimodel.png`** - Individual model boxplots
   - **Three stacked subplots** (one per model)
   - Each shows tile-level density distribution across dilutions
   - **CORRECTED:** X-axis shows dilutions in proper order: 10x → 20x → ... → 10240x
   - Y-axis: Foreground density (0-1)
   - Annotations: Sample size (n) for each dilution

#### 4. **`density_boxplot_comparison.png`** - Side-by-side model comparison
   - **Single plot** with all three models grouped by dilution
   - Allows direct visual comparison of model performance
   - Color-coded by model
   - Shows tile-level distributions

#### 5. **`representative_tiles_4panel/`** - Directory with 4-panel comparisons
   - Format: `tiles_4panel_{dilution}.png` (e.g., `tiles_4panel_10x.png`)
   - Each file shows 5 representative tiles (spanning density range)
   - **4 columns per tile:**
     - Column 1: **Original RGB tile**
     - Column 2: **UNet prediction** (grayscale)
     - Column 3: **Attention UNet prediction** (grayscale)
     - Column 4: **Attention ResUNet prediction** (grayscale)
   - Each prediction shows density value as subtitle

#### 6. **`EXPERIMENT_INFO.json`** - Metadata
   ```json
   {
     "timestamp": "2025-10-15 21:30:00",
     "models": ["unet", "attention_unet", "attention_resunet"],
     "model_dir": "./xukuang_params_shrunk_20251015_071224",
     "n_images": 11,
     "n_tiles_total": 308,
     "dilution_order": [10, 20, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
   }
   ```

## Usage

### On HPC (Recommended)

1. **Ensure models are in correct location:**
   ```bash
   ls -la xukuang_params_shrunk_20251015_071224/*.keras
   # Should see: unet, attention_unet, attention_resunet (3 files)
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
   tail -f Density_MultiModel.o*
   ```

5. **Check results:**
   ```bash
   ls -la density_analysis_xukuang_multimodel_*/
   ```

### Locally (For Testing)

```bash
python density_analysis_xukuang.py
```

**Note:** Requires:
- TensorFlow with GPU support
- All three 512×512 RGB models from Xukuang experiment
- Sufficient memory for loading 3 models (~20GB)

## Pipeline Details

### 1. Model Loading
```
For each model (UNet, Attention UNet, Attention ResUNet):
  ↓
Load {model}_xukuang_params_shrunk.keras
  ↓
Register BinaryFocalLoss class for deserialization
  ↓
Verify input/output shapes (512×512×3 → 512×512×1)
```

### 2. Image Processing & Prediction
```
For each test image:
  ↓
Load TIFF and convert to RGB (if needed)
  ↓
Extract 512×512 non-overlapping tiles
  ↓
For each tile:
  ↓
  Normalize to [0, 1]
  ↓
  Predict with ALL 3 models in parallel
  ↓
  Calculate density for each prediction
  ↓
  Save tile-level results (3 rows per tile)
  ↓
Aggregate statistics per image per model
```

### 3. Density Calculation
```
Prediction Map (512×512, [0-1])
  ↓
Threshold at 0.5
  ↓
Binary Mask (512×512, {0,1})
  ↓
Density = Mean(Binary Mask)
  ↓
Store with tile metadata (position, dilution, model)
```

### 4. Visualization Generation

**Step 1: Individual Model Boxplots**
```
Tile-level densities
  ↓
Group by (model, dilution)
  ↓
Create categorical ordering (10x → 10240x)
  ↓
Generate 3 stacked boxplots (one per model)
```

**Step 2: Model Comparison Boxplot**
```
Tile-level densities
  ↓
Group by dilution, color by model
  ↓
Create side-by-side grouped boxplot
```

**Step 3: 4-Panel Tile Comparisons**
```
For each dilution:
  ↓
Select 5 representative tiles (spanning density range)
  ↓
For each tile:
  ↓
  Panel 1: Original RGB
  Panel 2: UNet prediction
  Panel 3: Attention UNet prediction
  Panel 4: Attention ResUNet prediction
  ↓
Save as tiles_4panel_{dilution}.png
```

## Key Improvements Over Previous Version

| Aspect | Previous (Single Model) | New (Multi-Model) |
|--------|------------------------|-------------------|
| **Models** | UNet only | **All 3 models** (UNet, Attn UNet, Attn ResUNet) |
| **Density Data** | Image-level means only | **Tile-level values** (n=28 per image) |
| **Boxplots** | Single model | **3 boxplots + comparison** |
| **Tile Visualizations** | 2-panel (original + pred) | **4-panel** (original + 3 predictions) |
| **CSV Output** | 1 file (image summary) | **2 files** (tile-level + image summary) |
| **Walltime** | 4 hours | **8 hours** (for 3 models) |

## Expected Runtime

- **Model Loading:** ~30-60 sec per model (3 models = ~2-3 min)
- **Tile Extraction:** ~1-2 min per large image
- **Prediction:** ~10-20 ms per tile per model (×3 models = 30-60 ms per tile)
- **Visualization:** ~1-2 min total
- **Total:** ~1-2 hours for 11 test images with 3 models
- **Walltime Allocated:** 8 hours (safe margin)

## Troubleshooting

### Issue: Model file not found
```bash
# Check model directory
ls -la xukuang_params_shrunk_20251015_071224/*.keras
# Should show 3 files: unet, attention_unet, attention_resunet
```

**Solution:** Models are saved on HPC during training. Ensure you're running on HPC at `/home/svu/phyzxi/scratch/unet-HPC/`.

### Issue: BinaryFocalLoss deserialization error
**Error:** `TypeError: Could not locate class 'BinaryFocalLoss'`

**Solution:** This is fixed in current version. The script properly defines and registers `BinaryFocalLoss` class with `@keras.saving.register_keras_serializable(package='Custom')` decorator.

See `DENSITY_ANALYSIS_FIXES.md` for detailed explanation.

### Issue: Wrong image format (grayscale model on RGB images)
**Error:** Input shape mismatch

**Solution:** This script expects RGB models. Verify:
```python
model.input_shape  # Should be (None, 512, 512, 3)
```

### Issue: Dilution labels still wrong in plot
**Check:** Verify boxplots show correct x-axis order: 10x, 20x, 80x, ..., 5120x, 10240x
(NOT: 10240x, 1280x, 160x, ...)

The script uses `pd.Categorical(..., categories=DILUTION_LABELS, ordered=True)` to enforce correct ordering.

### Issue: Out of memory
**Solutions:**
1. Reduce batch size: `'batch_size': 8` → `'batch_size': 4`
2. Load models one at a time (requires code modification)
3. Request more memory: `#PBS -l select=1:ncpus=36:mem=300gb:ngpus=1`

### Issue: Job times out
**Solutions:**
1. Current walltime is 8 hours (sufficient for most cases)
2. If needed, increase: `#PBS -l walltime=12:00:00`
3. Check if models are loading correctly (shouldn't take > 5 min)

## Verification Checklist

After analysis completes, verify:

### 1. Output Files
- [ ] `density_results_tile_level.csv` exists and has tile-level data
- [ ] `density_results_image_summary.csv` exists and has image summaries
- [ ] `density_boxplot_multimodel.png` shows 3 stacked boxplots
- [ ] `density_boxplot_comparison.png` shows grouped comparison
- [ ] `representative_tiles_4panel/` directory contains 10 PNG files (one per dilution)
- [ ] `EXPERIMENT_INFO.json` has correct metadata

### 2. Data Integrity
- [ ] Tile-level CSV has rows for each model (total rows = n_tiles × n_models)
- [ ] Image summary CSV has 3 rows per image (one per model)
- [ ] All dilutions present: 10x, 20x, 80x, 160x, 320x, 640x, 1280x, 2560x, 5120x, 10240x

### 3. Visualizations
- [ ] **Boxplot x-axis ordering:** Should go: 10x, 20x, 80x, ..., 5120x, 10240x
- [ ] **Density trend:** Higher dilution → Lower density (generally)
- [ ] **4-panel comparisons:** Each shows Original + 3 model predictions
- [ ] **Model differences visible:** Can see variation between UNet, Attn UNet, Attn ResUNet

### 4. Model Performance Insights
- [ ] Which model gives highest/lowest densities?
- [ ] Are attention models more/less sensitive than UNet?
- [ ] Do model predictions correlate? (check 4-panel comparisons)
- [ ] Variability across tiles: Which model is most consistent?

## Analysis Questions to Answer

After successful analysis, investigate:

1. **Model Comparison:**
   - Which model predicts highest densities on average?
   - Which model has lowest variance across tiles?
   - Do attention mechanisms improve consistency?

2. **Dilution Response:**
   - Is the density vs dilution relationship linear?
   - Which dilution range shows most variation?
   - Are replicates (80x_1 vs 80x_2) consistent?

3. **Tile-Level Patterns:**
   - How much variation exists within each image?
   - Are there systematic position effects?
   - Which images have highest tile-to-tile variance?

4. **Model Agreement:**
   - Do all 3 models agree on high/low density tiles?
   - Where do models disagree most?
   - Use 4-panel comparisons to identify disagreement patterns

## Next Steps

After successful analysis:

### 1. Review Results
```bash
# View boxplots
open density_analysis_xukuang_multimodel_*/density_boxplot_*.png

# View 4-panel comparisons
open density_analysis_xukuang_multimodel_*/representative_tiles_4panel/*.png

# Check CSV data
head -50 density_analysis_xukuang_multimodel_*/density_results_tile_level.csv
```

### 2. Statistical Analysis
- Correlation between dilution and density (per model)
- ANOVA: Compare models across dilutions
- Agreement analysis: Inter-model correlation
- Variance partitioning: Image vs tile vs model effects

### 3. Model Selection
Based on results:
- Which model performs best for this task?
- Should we use ensemble predictions?
- Are attention mechanisms helpful or harmful?

### 4. Report Generation
Include in final report:
- Multi-model comparison boxplot
- Best 4-panel comparisons showing model differences
- Table of mean densities per dilution per model
- Statistical comparison of models

## Technical Details

### Why 4-Panel Comparisons?

**Purpose:** Visual comparison of model predictions on identical tiles

**Layout:**
```
Row 1: [Original] [UNet] [Attn UNet] [Attn ResUNet]
Row 2: [Original] [UNet] [Attn UNet] [Attn ResUNet]
...
Row 5: [Original] [UNet] [Attn UNet] [Attn ResUNet]
```

**Benefits:**
- See where models agree/disagree
- Identify systematic biases
- Understand attention mechanism effects
- Select best model for deployment

### Why Tile-Level Tracking?

**Previous approach:** Only saved mean density per image

**Problem:** Lost information about within-image variation

**New approach:** Save EVERY tile density

**Benefits:**
- Full distribution visible in boxplots
- Can analyze positional effects
- Better statistical power
- Identify outlier tiles

### Dilution Ordering Fix

**Problem:** Previous code sorted dilutions as strings: `['10240x', '1280x', '160x', ...]`

**Solution:** Use pandas categorical with explicit ordering:
```python
DILUTION_ORDER = [10, 20, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
DILUTION_LABELS = ['10x', '20x', '80x', ..., '10240x']

df['Dilution_Cat'] = pd.Categorical(
    df['dilution_label'],
    categories=DILUTION_LABELS,
    ordered=True
)

sns.boxplot(..., order=DILUTION_LABELS)  # Enforces correct order
```

### BinaryFocalLoss Serialization

**Why needed:** Xukuang training used `BinaryFocalLoss` class (not function)

**Solution:**
```python
@keras.saving.register_keras_serializable(package='Custom')
class BinaryFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config
```

See `DENSITY_ANALYSIS_FIXES.md` for full explanation of serialization.

## References

- **Model Source:** `xukuang_params_shrunk_20251015_071224/`
- **Training Report:** `xukuang_params_shrunk_20251015_071224/report.md`
- **Training Parameters:** `xukuang_params_shrunk_20251015_071224/EXPERIMENT_INFO.json`
- **Serialization Fixes:** `DENSITY_ANALYSIS_FIXES.md`
- **Previous Single-Model Analysis:** `density_analysis_512_grayscale_20251015_052432/`

## Contact

For questions about:
- **Models:** See `xukuang_params_shrunk_20251015_071224/report.md`
- **Analysis Pipeline:** Review this README and script comments
- **Results Interpretation:** Check generated `EXPERIMENT_INFO.json`
- **Serialization Issues:** Read `DENSITY_ANALYSIS_FIXES.md`

---

**Created:** October 15, 2025
**Author:** Claude Code
**Version:** 2.0 (Multi-Model)
**Updated:** Added multi-model comparison, tile-level tracking, 4-panel visualizations
