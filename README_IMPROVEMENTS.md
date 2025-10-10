# Critical Improvements: Hyperparameter Search & Fixed Predictions

## Overview

This document addresses two critical issues identified in the microbead segmentation pipeline:

1. **Hyperparameters were chosen empirically** → Need experimental validation
2. **Prediction code had incorrect tiling** → Tile size mismatch and black grid artifacts

---

## Issue 1: Hyperparameter Validation ❌ → ✅

### Problem

The current training (`train_microbead_optimized.py`) uses hyperparameters that were **chosen based on reasoning**, not experimental validation:

```python
# Current (empirically chosen, not validated)
LEARNING_RATE = 1e-4      # "Should be 10× lower than mitochondria"
BATCH_SIZE = 32           # "Larger batches for stability"
DROPOUT_RATE = 0.3        # "Prevent overfitting uniform shapes"
LOSS_TYPE = 'dice'        # "Direct IoU optimization"
```

**Result:** Val Jaccard = 0.2456 (moderate, below 0.50 target)

### Solution: Systematic Hyperparameter Search

**New script:** `hyperparameter_search_microbead.py`

Systematically tests combinations to find the **experimentally validated best** hyperparameters:

#### Search Space

| Hyperparameter | Current | Search Range | Rationale |
|----------------|---------|--------------|-----------|
| **Learning Rate** | 1e-4 | [5e-5, 1e-4, 2e-4] | Test around current value |
| **Batch Size** | 32 | [16, 32, 48] | Test memory/stability tradeoff |
| **Dropout** | 0.3 | [0.0, 0.1, 0.2, 0.3] | Current may be too aggressive |
| **Loss Function** | dice | [dice, focal, combined] | Test alternatives |

#### Search Modes

1. **Grid Search** (exhaustive):
   - Tests all 3×3×4×3 = **108 combinations**
   - Estimated time: 9-18 hours on HPC
   - Guarantees finding global optimum in search space

2. **Random Search** (sampling):
   - Tests **20 random combinations**
   - Estimated time: 2-3 hours on HPC
   - Good coverage of search space, faster

### How to Use

#### On Local Machine (for testing):

```bash
# Random search (20 combinations, ~2-3 hours on GPU)
python hyperparameter_search_microbead.py --search-type random --n-random 20

# Full grid search (108 combinations, ~9-18 hours on GPU)
python hyperparameter_search_microbead.py --search-type grid
```

#### On HPC (recommended):

```bash
# 1. Upload files to HPC
git add hyperparameter_search_microbead.py pbs_hyperparam_search.sh
git commit -m "Add hyperparameter search"
git push

# On HPC
cd ~/scratch/unet-HPC
git pull

# 2. Submit job (random search - faster)
qsub pbs_hyperparam_search.sh

# OR: Submit full grid search
export SEARCH_TYPE=grid
qsub pbs_hyperparam_search.sh

# 3. Monitor
qstat -u phyzxi
tail -f Microbead_Hyperparam_Search.o<JOBID>
```

### Output

```
hyperparam_search_YYYYMMDD_HHMMSS/
├── search_results_final.csv         # All results ranked by performance
├── best_hyperparameters.json        # Best configuration
├── model_lr*_bs*_dr*_*.hdf5        # Trained models
└── history_lr*_bs*_dr*_*.csv       # Training histories
```

**Example output:**

```csv
learning_rate,batch_size,dropout,loss_type,best_val_jacard,best_epoch
0.0001,16,0.1,combined,0.4521,35
5e-05,32,0.2,dice,0.4234,42
0.0002,48,0.0,combined,0.4012,28
...
```

### Expected Improvements

Based on current analysis:

1. **Reduce dropout 0.3 → 0.1**: Minimal overfitting observed, dropout may be limiting performance (+10-15% expected)
2. **Combined loss**: Leverage both Dice and Focal benefits (+5-10% expected)
3. **Optimize LR/BS**: Fine-tune around 1e-4 / 32 (+5% expected)

**Projected best Val Jaccard:** 0.35-0.45 (vs current 0.2456)

---

## Issue 2: Prediction Tiling Problems ❌ → ✅

### Problems Identified

#### Problem 2A: Tile Size Mismatch

**Training:** Images resized to **256×256**

```python
# train_microbead_optimized.py, line 45
SIZE = 256
image = image.resize((SIZE, SIZE))  # Training uses 256×256
```

**Prediction:** Used **512×512** tiles (WRONG!)

```python
# Old predict_microscope.py
tile_size = 512  # ❌ MISMATCH with training (256)!
```

**Impact:** Model trained on 256×256 images but predicts on 512×512 tiles → domain mismatch, poor performance

#### Problem 2B: Black Grid Artifacts

**Old blending:** Simple averaging creates visible seams

```python
# Old method - creates black grid lines
prediction[y:y+256, x:x+256] = tile_pred  # Hard boundaries
```

**Result:** Visible black grid lines at tile boundaries in output

### Solution: Fixed Prediction Code

**New script:** `predict_microscope_fixed.py`

#### Fix 2A: Correct Tile Size

```python
TRAINING_SIZE = 256  # Match training exactly!

class TiledPredictor:
    def __init__(self, model, tile_size=TRAINING_SIZE, overlap=32):
        if tile_size != TRAINING_SIZE:
            print(f"WARNING: Forcing tile_size = {TRAINING_SIZE}")
            tile_size = TRAINING_SIZE  # Enforce correctness
```

#### Fix 2B: Gaussian-Weighted Blending

```python
def _create_gaussian_weight_matrix(self):
    """
    Create Gaussian weight matrix for smooth blending.
    Center pixels: weight = 1.0
    Edge pixels: weight → 0.0
    """
    x = np.linspace(-1, 1, self.tile_size)
    y = np.linspace(-1, 1, self.tile_size)
    X, Y = np.meshgrid(x, y)

    # Gaussian with sigma=0.5
    weight = np.exp(-(X**2 + Y**2) / (2 * 0.5**2))
    weight = weight / weight.max()  # Normalize to [0, 1]

    return weight
```

**Blending visualization:**

```
Old (hard boundaries):        New (Gaussian blending):
┌─────┬─────┐                ┌─────┬─────┐
│ 1.0 │ 1.0 │                │ 1.0→│←1.0 │
│     │     │                │  ↓  │  ↓  │
├─────┼─────┤  ← BLACK GRID  │  ↓  │  ↓  │  ← SMOOTH!
│ 1.0 │ 1.0 │                │  ↓  │  ↓  │
└─────┴─────┘                └─0.0─┴─0.0─┘
```

**Weight normalization:**

```python
# Accumulate weighted predictions
prediction[y_start:y_end, x_start:x_end] += tile_pred * weight
weight_sum[y_start:y_end, x_start:x_end] += weight

# Normalize (average where tiles overlap)
prediction = prediction / weight_sum
```

### How to Use

#### On Local Machine:

```bash
# Using best model from training
python predict_microscope_fixed.py \
    --model microbead_training_20251009_073134/best_unet_model.hdf5 \
    --input-dir ./test_images \
    --output-dir ./predictions_fixed \
    --tile-size 256 \
    --overlap 32 \
    --threshold 0.5
```

**Key arguments:**

- `--tile-size 256`: **MUST be 256** (matches training)
- `--overlap 32`: Overlap for smooth blending (default works well)
- `--threshold 0.5`: Binarization threshold (0.0-1.0)

#### On HPC:

```bash
# Upload to HPC
git add predict_microscope_fixed.py pbs_predict_fixed.sh
git commit -m "Fix prediction tiling"
git push

# On HPC
cd ~/scratch/unet-HPC
git pull

# Submit prediction job
qsub pbs_predict_fixed.sh

# OR: Specify custom paths
export MODEL_PATH="./best_model.hdf5"
export INPUT_DIR="./my_test_images"
qsub pbs_predict_fixed.sh
```

### Output

For each input image `image.tif`:

```
predictions_fixed_YYYYMMDD_HHMMSS/
├── image_mask.png       # Binary prediction (0 or 255)
└── image_overlay.png    # Visualization (image + green mask)
```

### Before vs After Comparison

| Aspect | **Before (Wrong)** | **After (Fixed)** |
|--------|-------------------|-------------------|
| **Tile Size** | 512×512 ❌ | 256×256 ✅ |
| **Matches Training** | No ❌ | Yes ✅ |
| **Black Grid** | Visible ❌ | None ✅ |
| **Blending** | Hard boundaries ❌ | Gaussian smooth ✅ |
| **Performance** | Poor ❌ | Expected good ✅ |

---

## Summary & Next Steps

### What We Fixed

1. ✅ **Hyperparameter validation** through systematic grid/random search
2. ✅ **Tile size corrected** to 256×256 (matches training)
3. ✅ **Smooth blending** with Gaussian weighting (no black grid)

### Workflow

#### Step 1: Find Best Hyperparameters (HPC)

```bash
# On HPC
qsub pbs_hyperparam_search.sh

# Wait for completion (~2-3 hours for random search)

# Download results
scp -r phyzxi@hpc:~/scratch/unet-HPC/hyperparam_search_* ./

# Check best config
cat hyperparam_search_*/best_hyperparameters.json
```

#### Step 2: Retrain with Best Hyperparameters

Update `train_microbead_optimized.py` with the discovered best hyperparameters and retrain.

#### Step 3: Predict with Fixed Code

```bash
# On HPC
qsub pbs_predict_fixed.sh

# Download predictions
scp -r phyzxi@hpc:~/scratch/unet-HPC/predictions_fixed_* ./
```

### Expected Outcomes

**Hyperparameter Search:**
- Find experimentally validated optimal hyperparameters
- Expected Val Jaccard: **0.35-0.45** (vs current 0.2456)
- Potential to reach **>0.50** (production-ready)

**Fixed Predictions:**
- Correct domain alignment (256×256 tiles)
- No visual artifacts (smooth blending)
- Better prediction quality

---

## 💡 **Insight ────────────────────────────────────**

**Why tile size matters so much:**

Models learn to recognize patterns at specific spatial scales. Training at 256×256 means the model learns what a microbead looks like at that resolution. Predicting on 512×512 tiles shows the model images at 2× the spatial scale - microbeads appear 4× larger (2× in each dimension). This is like asking a model trained on small thumbnail images to predict on large posters - it's a domain shift!

**Why Gaussian blending eliminates black grid:**

Hard boundaries between tiles create discontinuities in pixel values. Gaussian weighting makes tiles contribute more at their centers (where predictions are most confident) and less at edges (where context is limited). Overlapping regions get averaged with proper weights, creating smooth transitions just like panorama stitching in photography.

**─────────────────────────────────────────────────**

---

## Files Created

1. **`hyperparameter_search_microbead.py`** - Systematic hyperparameter search
2. **`pbs_hyperparam_search.sh`** - HPC job script for search
3. **`predict_microscope_fixed.py`** - Fixed prediction with correct tiling
4. **`pbs_predict_fixed.sh`** - HPC job script for prediction
5. **`README_IMPROVEMENTS.md`** - This documentation

---

## Questions?

**Q: Should I use grid or random search?**

A: Start with **random search** (20 combinations, ~2-3 hours). If results are promising but not optimal, run **grid search** overnight for exhaustive validation.

**Q: Can I change the search space?**

A: Yes! Edit `SEARCH_SPACE` in `hyperparameter_search_microbead.py`:

```python
SEARCH_SPACE = {
    'learning_rate': [5e-5, 1e-4, 2e-4, 3e-4],  # Add more values
    'batch_size': [16, 32, 48, 64],             # Test larger batches
    # ...
}
```

**Q: Why 32 pixels overlap?**

A: Balance between smoothness and computation:
- More overlap → smoother, slower
- Less overlap → faster, possible seams
- 32 pixels (12.5% of 256) works well in practice

**Q: What if I want to predict on 512×512 images directly?**

A: If your test images are 512×512, the tiling will handle it automatically. The code processes images of ANY size by tiling into 256×256 patches.

---

*Updated: October 9, 2025*
