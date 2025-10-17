# Attention ResUNet Density Analysis - Setup and Usage

**Date:** October 17, 2025
**Status:** ✅ Ready for HPC deployment
**Model Directory:** `./attention_resunet_hyperparam_20251015_235542`

---

## Overview

This density analysis script performs tile-level and image-level density quantification on test images using the **BEST Attention ResUNet model** from the hyperparameter search.

### Key Features

1. ✅ **Automatic best model selection** - Selects model with highest validation IoU from 27 trained configurations
2. ✅ **Tile-level density tracking** - All 28 tiles per image analyzed and saved
3. ✅ **6 density calculation methods**:
   - Threshold 0.2, 0.5, 0.8, 0.95 (simple thresholding)
   - CLAHE+Otsu on predicted mask (denoised, with 1-density fix)
   - CLAHE+Otsu on original image (baseline)
4. ✅ **12 comprehensive boxplots** (6 methods × 2 dilution ranges)
5. ✅ **3-panel tile visualizations** (5 representative tiles per image)

---

## Files

### Python Script
**File:** `density_analysis_attention_resunet_only.py`

**Purpose:** Performs density analysis on test images using best Attention ResUNet model

**Key Functions:**
- `find_best_attention_resunet_model()`: Searches checkpoints and selects best model by validation IoU
- `load_model()`: Loads Keras model with all custom objects (including RepeatElements)
- `predict_on_tiles()`: Generates predictions for 512×512 tiles
- `calculate_density_*()`: 6 different density quantification methods
- `create_boxplots()`: Generates visualizations for two dilution ranges

**Critical Bug Fixes Included:**
1. ✅ Searches `checkpoints/` directory (not `models/`)
2. ✅ Includes `RepeatElements` custom layer in `custom_objects` (required for Attention ResUNet)

### PBS Script
**File:** `pbs_density_analysis_attention_resunet_only.sh`

**Purpose:** HPC job submission script for running density analysis

**PBS Configuration:**
- Walltime: 4 hours
- Resources: 1 GPU node (A40/A100), 36 CPUs, 240 GB RAM
- Job name: `Density_AttnResUNet_Only`
- Email notifications: Start, abort, end

**Workflow:**
1. Environment setup (TensorFlow Singularity container)
2. Verify required files (model directory, test images, script)
3. Run density analysis with console logging
4. Display results summary and generated files

---

## Model Selection Logic

### Automatic Best Model Selection

The script automatically selects the best Attention ResUNet model based on validation IoU from the hyperparameter search.

**Search Pattern:**
```python
attention_resunet_hyperparam_20251015_235542/checkpoints/attention_resunet_*/
```

**Selection Criteria:**
- Reads `training_history.csv` from each checkpoint directory
- Extracts maximum `val_jacard_coef` (validation IoU)
- Selects configuration with highest IoU
- Loads `best_model.keras` from selected checkpoint

**Expected Best Configuration (based on hyperparameter search results):**
```
Model: attention_resunet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001
Hyperparameters:
  - n_filters: 16
  - dropout: 0.1
  - batch_norm: True
  - learning_rate: 0.001
Expected Best Val IoU: ~0.4885-0.4900 (highest among all architectures)
```

**Note:** The actual selected model may differ slightly if HPC training logs show different results than local analysis.

---

## Custom Objects Required

### Why Custom Objects Are Needed

Attention ResUNet (like Attention UNet) uses **attention gates** that require a custom Keras layer for tensor shape matching.

**Custom Layer:**
```python
from models_fixed import RepeatElements

@tf.keras.saving.register_keras_serializable(package='Custom')
class RepeatElements(layers.Layer):
    """
    Repeats elements of a tensor along an axis.
    Replaces: Lambda(lambda x: K.repeat_elements(x, rep, axis=3))
    """
    def __init__(self, rep, axis=3, **kwargs):
        super().__init__(**kwargs)
        self.rep = rep
        self.axis = axis

    def call(self, inputs):
        return K.repeat_elements(inputs, self.rep, axis=self.axis)
```

**Custom Objects Dictionary:**
```python
custom_objects = {
    'combined_dice_focal_loss': combined_dice_focal_loss,
    'jacard_coef': jacard_coef,
    'dice_coef': dice_coef,
    'focal_loss': focal_loss,
    'BinaryFocalLoss': BinaryFocalLoss,
    'RepeatElements': RepeatElements,  # Required for Attention ResUNet
}
```

**Without this fix:** Model loading fails with `TypeError: 'str' object is not callable`

---

## Density Calculation Methods

### 1. Simple Thresholding (4 methods)

Counts pixels in predicted mask above threshold:

```python
density = np.sum(pred_mask > threshold) / pred_mask.size
```

**Thresholds:** 0.2, 0.5, 0.8, 0.95

### 2. CLAHE+Otsu on Predicted Mask

Applies denoising before thresholding:

```python
# Convert to 8-bit
mask_8bit = (pred_mask * 255).astype(np.uint8)

# Apply CLAHE for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(mask_8bit)

# Otsu's method for automatic thresholding
_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# FIXED: Measure beads (white), not background
density = 1.0 - (np.count_nonzero(binary) / binary.size)
```

**Critical Fix:** Previous versions measured background instead of beads (missing `1 -`)

### 3. CLAHE+Otsu on Original Image (Baseline)

Same method applied directly to original grayscale image for comparison.

---

## Boxplot Visualizations

### Two Dilution Ranges

**Full Range (1/10240x to 1/10x):**
- All 8 test images included
- Shows full density spectrum

**Low Dilution Range (1/10240x to 1/80x):**
- Excludes high-density images (10x, 20x, 40x if present)
- Focuses on challenging low-density region

### Categorical X-Axis Style

Boxplots use **evenly spaced categorical x-axis** (not log scale):

```python
dilutions_categorical = ['10240x', '5120x', '2560x', '1280x', '640x', '320x', '160x', '80x']
x_positions = range(len(dilutions_categorical))
```

**Advantages:**
- Clear visual separation between groups
- Easy comparison across dilution levels
- No distortion from log scaling

---

## 3-Panel Tile Visualizations

For each test image, the script generates **5 representative tiles** with 3 panels:

**Panel 1: Original Tile**
- Raw 512×512 RGB image
- Shows actual input to model

**Panel 2: Inverted Predicted Mask**
- White beads on black background
- Threshold: 0.5
- Shows model predictions

**Panel 3: Inverted CLAHE+Otsu**
- White beads on black background
- Denoised with CLAHE+Otsu
- Shows cleaned segmentation

**Output Directory:** `representative_tiles_3panel/`

---

## Usage Instructions

### 1. Prerequisites

**On HPC:**
- ✅ Attention ResUNet hyperparameter search completed
- ✅ Models saved in `attention_resunet_hyperparam_20251015_235542/checkpoints/`
- ✅ Test images available in `test_images/`
- ✅ Both scripts uploaded to working directory

**Files Required:**
```
/home/svu/phyzxi/scratch/unet-HPC/
├── density_analysis_attention_resunet_only.py
├── pbs_density_analysis_attention_resunet_only.sh
├── attention_resunet_hyperparam_20251015_235542/
│   └── checkpoints/
│       ├── attention_resunet_n_filters16_dropout0p1_.../
│       ├── attention_resunet_n_filters16_dropout0p2_.../
│       └── ... (27 configurations)
└── test_images/
    ├── 10240x_2025-05-29_02-22-00_002.tif
    ├── 5120x_2025-05-16_00-59-00.tif
    └── ... (8 images)
```

### 2. Submit Job

**Command:**
```bash
ssh HPC
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_density_analysis_attention_resunet_only.sh
```

**Expected Output:**
```
293XXX.stdct-mgmt-02
```

### 3. Monitor Job

**Check queue:**
```bash
qstat -u phyzxi
```

**Monitor console output (live):**
```bash
tail -f density_analysis_attention_resunet_only_console_YYYYMMDD_HHMMSS.log
```

**Check job output (after completion):**
```bash
cat Density_AttnResUNet_Only.oJOBID
```

### 4. Expected Runtime

**Total Walltime:** ~3-4 hours

**Breakdown:**
- Model selection and loading: ~30 seconds
- Prediction on 8 images (224 tiles): ~2-3 hours
- Density calculations: ~15 minutes
- Boxplot generation: ~10 minutes
- Tile visualization creation: ~15 minutes

### 5. Output Files

**Output Directory:** `density_analysis_attention_resunet_only_YYYYMMDD_HHMMSS/`

**Generated Files:**
```
density_analysis_attention_resunet_only_20251017_HHMMSS/
├── EXPERIMENT_INFO.json                           # Metadata
├── density_results_tile_level.csv                 # 224 rows (28 tiles × 8 images)
├── density_results_image_summary.csv              # 8 rows (1 per image)
├── density_boxplot_full_range_threshold_0.2.png   # Full range
├── density_boxplot_full_range_threshold_0.5.png
├── density_boxplot_full_range_threshold_0.8.png
├── density_boxplot_full_range_threshold_0.95.png
├── density_boxplot_full_range_claheotsu_on_pred.png
├── density_boxplot_full_range_claheotsu_on_original.png
├── density_boxplot_low_dilution_threshold_0.2.png # Low dilution range
├── density_boxplot_low_dilution_threshold_0.5.png
├── density_boxplot_low_dilution_threshold_0.8.png
├── density_boxplot_low_dilution_threshold_0.95.png
├── density_boxplot_low_dilution_claheotsu_on_pred.png
├── density_boxplot_low_dilution_claheotsu_on_original.png
└── representative_tiles_3panel/                   # 5 tiles × 8 images = 40 visualizations
    ├── 10240x_2025-05-29_02-22-00_002_tile_0_3panel.png
    ├── 10240x_2025-05-29_02-22-00_002_tile_5_3panel.png
    └── ...
```

**Total:** 12 boxplots + 40 tile visualizations + 3 CSV/JSON files

---

## Troubleshooting

### Job Fails: "No best_model.keras files found"

**Error:**
```
ERROR: No best_model.keras files found in ./attention_resunet_hyperparam_20251015_235542/checkpoints
```

**Cause:** Hyperparameter search incomplete or models not saved to `checkpoints/`

**Fix:**
1. Check hyperparameter search job status
2. Verify checkpoint directory exists:
   ```bash
   ls -lh attention_resunet_hyperparam_20251015_235542/checkpoints/
   ```

### Job Fails: TypeError during model loading

**Error:**
```
TypeError: 'str' object is not callable
```

**Cause:** Missing `RepeatElements` in custom_objects (shouldn't happen - already fixed)

**Fix:** Verify Python script has:
```python
from models_fixed import RepeatElements
custom_objects = {..., 'RepeatElements': RepeatElements}
```

### GPU Out of Memory

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Cause:** GPU memory insufficient (unlikely with A40/A100)

**Fix:** Reduce batch size in prediction loop (line 656):
```python
# Change from 4 to 2
batch_size = 2
```

### Test Images Not Found

**Error:**
```
ERROR: Test images directory not found: ./test_images
```

**Fix:** Create symlink or copy test images:
```bash
ln -s /path/to/test_images ./test_images
```

---

## Comparison with UNet and Attention UNet

### Architecture Differences

**UNet:**
- Vanilla encoder-decoder
- No attention mechanisms
- No custom layers needed
- Expected Val IoU: ~0.467

**Attention UNet:**
- Attention gates between encoder/decoder
- Uses `RepeatElements` custom layer
- Expected Val IoU: ~0.476-0.488

**Attention ResUNet (This Script):**
- Residual connections + attention gates
- Uses `RepeatElements` custom layer
- **Expected Val IoU: ~0.489 (HIGHEST)**
- Best capacity/performance tradeoff

### Density Analysis Scripts

All three architectures use the **same analysis pipeline**:
- Same 6 density methods
- Same boxplot generation
- Same 3-panel tile visualizations
- Same output format

**Only difference:** Model architecture and expected performance.

---

## Expected Results

### Best Model Selection

**Expected configuration from hyperparameter search:**
```
attention_resunet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003
```

**Expected performance:**
- Validation IoU: ~0.4885-0.4900 (highest among all architectures)
- Training time: ~1.5 hours (100 epochs)
- Model size: ~1.5 MB (16 filters)

### Density Trends

**Expected behavior across dilution series:**
1. **High dilution (10240x, 5120x):** Very low density, challenging detection
2. **Mid dilution (2560x, 1280x, 640x):** Gradual density increase
3. **Low dilution (320x, 160x, 80x):** Higher density, easier detection

**Method comparison:**
- Simple thresholding: Fast, but sensitive to threshold choice
- CLAHE+Otsu: More robust, removes noise artifacts
- Original image baseline: May capture non-bead features

---

## Bug Fixes Incorporated

### Fix 1: Checkpoint Directory Path

**Issue:** Original Attention UNet script searched `models/` directory

**Fix Applied:** Changed to `checkpoints/` directory (line 145)

```python
# CORRECT (this script)
model_dirs = list((base_dir / 'checkpoints').glob('attention_resunet_*'))
```

**Documentation:** See `BUGFIX_ATTENTION_UNET_DENSITY_ANALYSIS.md`

### Fix 2: RepeatElements Custom Layer

**Issue:** Attention-based models require custom layer for deserialization

**Fix Applied:** Added RepeatElements import and to custom_objects (lines 49, 414)

```python
# Import
from models_fixed import RepeatElements

# Add to custom_objects
custom_objects = {
    ...,
    'RepeatElements': RepeatElements,  # Custom layer for Attention ResUNet
}
```

**Documentation:** See `BUGFIX_ATTENTION_UNET_CUSTOM_LAYER.md`

---

## Next Steps

### 1. Submit Job (Immediate)

```bash
qsub pbs_density_analysis_attention_resunet_only.sh
```

### 2. Compare Architectures (After All Complete)

Once UNet, Attention UNet, and Attention ResUNet density analyses finish:

**Create comparison script:**
```python
compare_architecture_density.py
# - Load CSVs from all three architectures
# - Generate side-by-side boxplots
# - Calculate performance metrics (IoU vs density correlation)
# - Identify which architecture performs best at each dilution level
```

### 3. Validate Density Methods

**Correlation analysis:**
- Compare 6 density methods
- Identify which method best matches expected dilution series
- Check for systematic biases

### 4. Publication-Ready Figures

**Multi-panel figure:**
- Panel A: Architecture comparison (3 boxplots side-by-side)
- Panel B: Method comparison (6 methods overlaid)
- Panel C: Representative tile visualizations (best/worst predictions)

---

## References

### Related Documentation

- **Loss Function Analysis:** `LOSS_FUNCTION_ANALYSIS_AND_SEARCH_STRATEGY.md`
- **Attention UNet Bugs:** `BUGFIX_ATTENTION_UNET_DENSITY_ANALYSIS.md`, `BUGFIX_ATTENTION_UNET_CUSTOM_LAYER.md`
- **Hyperparameter Search:** See `attention_resunet_hyperparam_20251015_235542/` logs

### Training Configuration

**From:** `train_attention_resunet_hyperparam.py`

```python
# Loss function
loss = BinaryFocalLoss(alpha=0.25, gamma=2.0)

# Optimizer
optimizer = Adam(learning_rate=lr)

# Metrics
metrics = ['accuracy', jacard_coef, dice_coef]

# Training
epochs = 100
batch_size = 4
image_size = (512, 512, 3)
```

---

## Conclusion

**Status:** ✅ Ready for deployment

**Script Quality:**
- ✅ Syntax verified (compiles without errors)
- ✅ All references updated (attention_unet → attention_resunet)
- ✅ Both bug fixes incorporated (checkpoints path + RepeatElements)
- ✅ PBS script configured correctly

**Confidence Level:** Very High

**Expected Outcome:**
- Job completes successfully in ~3-4 hours
- Generates 12 boxplots + 40 tile visualizations
- Selects best Attention ResUNet model (expected IoU ~0.489)
- Provides comprehensive density analysis across dilution series

**Deployment Command:**
```bash
qsub pbs_density_analysis_attention_resunet_only.sh
```

---

**Document Date:** October 17, 2025
**Author:** Automated by Claude Code
**Version:** 1.0
**Status:** ✅ Production Ready
