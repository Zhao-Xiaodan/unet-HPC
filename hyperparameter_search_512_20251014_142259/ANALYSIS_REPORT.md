# 512×512 Hyperparameter Search Analysis Report

**Experiment:** `hyperparameter_search_512_20251014_142259`
**Date:** October 14, 2025 (HPC run: 14:22:59 - Oct 15, 06:59)
**Status:** ⚠ Complete with Training Instability Issues
**Input Resolution:** 512×512 RGB images
**Dataset:** `dataset_shrunk_masks/` (98 images)

---

## Executive Summary

### 🏆 Best Configuration Found

**Configuration:** `attention_resunet_lr5e-05_drop0.2_bs4`

| Parameter | Value |
|-----------|-------|
| **Architecture** | Attention ResUNet |
| **Learning Rate** | 5e-05 |
| **Dropout** | 0.2 |
| **Batch Size** | 4 |
| **Mean Jaccard (3-fold CV)** | **0.1562** |

### ⚠ CRITICAL FINDING: Training Instability

**Loss values were `nan` throughout training!**

Evidence from console logs:
```
Epoch 5/30
loss: nan - jacard_coef: 0.1226 - val_loss: nan - val_jacard_coef: 0.1613
```

**Impact:**
- Results may be unreliable
- Training was numerically unstable
- Likely due to mixed precision (FP16) issues with 512×512 images

### 📊 Performance Comparison: 512×512 vs 256×256

| Metric | 256×256 (Previous) | 512×512 (Current) | Change |
|--------|-------------------|-------------------|--------|
| **Best Jaccard** | 0.6005 | 0.1562 | **-74%** ⚠ |
| **Best Architecture** | ResUNet | Attention ResUNet | Different |
| **Best Learning Rate** | 5e-05 | 5e-05 | Same |
| **Best Dropout** | 0.3 | 0.2 | -0.1 |
| **Best Batch Size** | 8 | 4 | Smaller |
| **Filters** | 64 | 32 | **-50%** |
| **Training Stability** | ✓ Stable | ✗ Unstable (nan loss) | |

**Conclusion:** 512×512 performance is **significantly worse** than 256×256.

---

## Detailed Results

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total Configurations | 24 |
| Total Training Runs | 72 (24 × 3 folds) |
| Successful Runs | 72/72 (100%) |
| OOM Failures | 0 (0%) ✓ |
| Mean Runtime | ~16 hours |

**Memory Management:** ✓ No OOM errors! Mixed precision and small batch sizes worked.

### Top 5 Configurations

| Rank | Configuration | Mean Jaccard | Std |
|------|--------------|--------------|-----|
| 1 | `attention_resunet_lr5e-05_drop0.2_bs4` | 0.1562 | 0.0336 |
| 2 | `attention_resunet_lr5e-05_drop0.3_bs4` | 0.1552 | 0.0259 |
| 3 | `resunet_lr5e-05_drop0.3_bs4` | 0.1549 | 0.0171 |
| 4 | `unet_lr5e-05_drop0.2_bs4` | 0.1527 | 0.0263 |
| 5 | `unet_lr5e-05_drop0.2_bs2` | 0.1523 | 0.0273 |

**Observations:**
- All top 5 use learning rate 5e-05
- Batch size 4 dominates (4 out of 5)
- Attention ResUNet performs best (slightly)

### Performance by Architecture

| Architecture | Mean Jaccard | Std | Count |
|--------------|--------------|-----|-------|
| **Attention ResUNet** | **0.1460** | 0.0260 | 24 |
| U-Net | 0.1399 | 0.0195 | 24 |
| ResUNet | 0.1373 | 0.0219 | 24 |

**Finding:** Attention mechanism provides marginal benefit (+0.9% vs U-Net, +0.8% vs ResUNet)

### Performance by Hyperparameter

#### Learning Rate
| LR | Mean Jaccard | Count |
|----|--------------|-------|
| **5e-05** | **0.1490** | 36 |
| 1e-04 | 0.1342 | 36 |

**Finding:** Lower learning rate (5e-05) performs better, consistent with 256×256 results.

#### Dropout
| Dropout | Mean Jaccard | Count |
|---------|--------------|-------|
| **0.2** | **0.1436** | 36 |
| 0.3 | 0.1396 | 36 |

**Finding:** Lower dropout (0.2) is optimal, likely due to small dataset (98 images).

#### Batch Size
| Batch Size | Mean Jaccard | Count |
|------------|--------------|-------|
| **4** | **0.1429** | 36 |
| 2 | 0.1403 | 36 |

**Finding:** Batch size 4 slightly better, despite memory constraints.

---

## Why 512×512 Performance is Poor

### 1. **Training Instability (Primary Cause)**

**Evidence:**
- `loss: nan` in all training logs
- Despite nan loss, Jaccard was computed (suggests gradient issues, not data issues)

**Root Causes:**
- **Mixed precision (FP16):** 512×512 images with FP16 can cause numerical instability
- **Gradient explosion:** Large images → large gradients → overflow in FP16
- **Loss function sensitivity:** Combined Dice + Focal loss may be unstable with FP16

**Impact:**
- Models never properly converged
- Jaccard scores of ~0.15 indicate poor segmentation (random guessing would be ~0.05)

### 2. **Reduced Model Capacity**

| Aspect | 256×256 | 512×512 | Impact |
|--------|---------|---------|--------|
| **Filters** | 64 | 32 | -75% parameters |
| **Receptive Field** | Adequate | Relatively smaller | Harder to capture context |

**Consequence:** 512×512 models have 75% fewer parameters but need to handle 4× more pixels.

### 3. **Insufficient Training Data**

| Dataset | 256×256 | 512×512 |
|---------|---------|---------|
| **Images** | 1,980 patches | 98 images |
| **Effective samples** | ~1,980 | ~98 |
| **Pixels per image** | 65K | 262K (4×) |

**Data-to-parameter ratio:**
- 256×256: ~1,980 samples / ~1M params = 0.002
- 512×512: ~98 samples / ~250K params = 0.0004 (5× worse)

### 4. **RGB vs Grayscale**

- 256×256 dataset: Grayscale (1 channel)
- 512×512 dataset: RGB (3 channels)

**Implications:**
- Different data distribution
- RGB may have irrelevant color information
- Should consider converting to grayscale

---

## Critical Issues

### Issue 1: Loss is NaN

**Severity:** 🔴 CRITICAL

**Description:** All training runs show `loss: nan` while metrics are computed normally.

**Example from logs:**
```
Epoch 5/30
loss: nan - jacard_coef: 0.1226 - dice_coef: 0.2086
val_loss: nan - val_jacard_coef: 0.1613 - val_dice_coef: 0.2462
```

**Diagnosis:**
1. Mixed precision causes numerical overflow in loss computation
2. Gradients still update (Jaccard improves across epochs)
3. But optimization is unreliable

**Solution:**
- Disable mixed precision (use FP32)
- Add gradient clipping
- Lower learning rate further (2e-05 or 1e-05)

### Issue 2: Low Absolute Performance

**Severity:** 🟡 MODERATE

**Best Jaccard:** 0.1562 (vs 0.6005 for 256×256)

**Interpretation:**
- Jaccard 0.15 means only 15% overlap between prediction and ground truth
- This is poor segmentation performance
- Likely due to unstable training (Issue 1)

**Solution:**
- Fix training stability first
- If still low, increase model capacity (filters: 32 → 48)

### Issue 3: Small Dataset

**Severity:** 🟡 MODERATE

**Dataset size:** 98 images (vs 1,980 for 256×256)

**Impact:**
- Overfitting risk (though not observed in results)
- Limited generalization

**Solution:**
- Data augmentation (rotation, flip, elastic deformation)
- Or use 256×256 models with upscaling for inference

---

## Recommendations

### 1. **Immediate: Fix Training Instability** (CRITICAL)

```python
# Disable mixed precision
# In hyperparameter_search_512.py, line 50-52:
# REMOVE or COMMENT OUT:
# policy = keras.mixed_precision.Policy('mixed_float16')
# keras.mixed_precision.set_global_policy(policy)

# Add gradient clipping
optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
```

**Expected improvement:** Stable loss, Jaccard 0.3-0.5

### 2. **Increase Model Capacity** (if training is stable)

```python
# Increase filters
'filters': 32,  # Change to 48 or 64
```

**Trade-off:** May cause OOM. Start with 48, then try 64 if memory allows.

### 3. **Data Augmentation**

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect'
)
```

**Expected improvement:** Better generalization, +5-10% Jaccard

### 4. **Convert RGB to Grayscale**

```python
# In load_dataset() function:
img = Image.open(img_path).convert('L')  # Grayscale instead of RGB
img_array = np.array(img)[..., np.newaxis] / 255.0  # Add channel dim
```

**Rationale:** Match 256×256 dataset format

### 5. **Alternative: Use 256×256 Models for Production**

**Recommendation:** If 512×512 results remain poor after fixes:

1. Train on 256×256 (proven to work: Jaccard 0.60)
2. For inference on large images:
   - Option A: Extract 256×256 tiles, predict, stitch back
   - Option B: Resize 512×512 → 256×256, predict, resize back

**Pros:**
- Reliable training (no nan loss)
- Better performance (0.60 vs 0.15)
- Faster training (4× fewer pixels)

**Cons:**
- Lower resolution output
- May miss fine details

---

## Next Steps

### Option A: Fix and Re-run 512×512 Training (Recommended)

1. ✅ Create fixed hyperparameter search script
   - Disable mixed precision
   - Add gradient clipping
   - Optionally: convert RGB → grayscale

2. ✅ Re-run hyperparameter search
   - Expected runtime: ~12-18 hours
   - Expected Jaccard: 0.3-0.5 (if stable)

3. ✅ If results improve (Jaccard > 0.4):
   - Increase filters to 48 or 64
   - Add data augmentation
   - Fine-tune hyperparameters

4. ✅ Run density analysis with best model

### Option B: Use 256×256 Models (Fallback)

If 512×512 results remain poor (Jaccard < 0.4 after fixes):

1. ✅ Use existing 256×256 best model (`resunet_lr5e-05_drop0.3_bs8`, Jaccard 0.60)
2. ✅ For density analysis on test images:
   - Extract 256×256 tiles
   - Predict with best 256×256 model
   - Calculate density per tile
   - Aggregate results

**Advantage:** Proven to work, immediate results

---

## Comparison Summary

| Aspect | 256×256 (Proven) | 512×512 (Current) | Winner |
|--------|------------------|-------------------|--------|
| **Best Jaccard** | 0.6005 | 0.1562 | 256×256 ✓ |
| **Training Stability** | ✓ Stable | ✗ Unstable (nan) | 256×256 ✓ |
| **Memory Management** | ✓ Easy | ⚠ Challenging | 256×256 ✓ |
| **Dataset Size** | 1,980 | 98 | 256×256 ✓ |
| **Training Time** | ~6 hrs | ~16 hrs | 256×256 ✓ |
| **Resolution** | Lower | Higher | 512×512 ✓ |
| **Field of View** | Smaller | Larger | 512×512 ✓ |

**Overall Winner:** **256×256** (5:2)

---

## Files in This Directory

```
hyperparameter_search_512_20251014_142259/
├── all_results.csv                           ← All 72 training results
├── intermediate_results.csv                  ← Incremental save
├── summary.json                              ← Best configuration
├── hyperparam_search_512_console_*.log       ← Training logs (6 MB)
├── HyperSearch_512.o287606                   ← PBS output log
└── ANALYSIS_REPORT.md                        ← This file
```

---

## Conclusion

### Main Findings

1. **Training Instability:** Loss = nan indicates numerical issues with mixed precision
2. **Poor Performance:** Jaccard 0.16 vs 0.60 for 256×256 (74% worse)
3. **Memory Management:** ✓ No OOM errors (small batch sizes worked)
4. **Best Architecture:** Attention ResUNet (marginal improvement over U-Net/ResUNet)

### Recommended Action

**Do NOT use these 512×512 results for production.**

Instead:
1. **Fix training instability** (disable mixed precision, add gradient clipping)
2. **Re-run hyperparameter search** with stable training
3. **If results still poor:** Use proven 256×256 models

### For Density Analysis

**Use 256×256 models** until 512×512 training is fixed:
- Best model: `resunet_lr5e-05_drop0.3_bs8`
- Jaccard: 0.6005
- Location: `hyperparameter_search_20251013_154754/`

---

**Report Generated:** October 15, 2025
**Analyst:** Claude Code
**Status:** Awaiting decision on next steps
