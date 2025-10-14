## 512×512 Training and Density Analysis Workflow

**Created:** October 14, 2025
**Purpose:** Train models on 512×512 images with OOM protection and perform density analysis

---

## Overview

This workflow trains U-Net, ResUNet, and Attention ResUNet on the `dataset_shrunk_masks` dataset (98 images, 512×512 RGB), then uses the best model for density analysis on dilution series test images.

### Key Challenge: Memory Management

512×512 images use **4× more memory** than 256×256 images. Without proper memory management, training will fail with Out-of-Memory (OOM) errors.

### Solutions Implemented

| Strategy | Memory Savings | Implementation |
|----------|----------------|----------------|
| **Mixed Precision (FP16)** | ~40% | Automatic in training scripts |
| **Reduced Filters** | ~4× | 32 filters (vs 64 in 256×256) |
| **Small Batch Sizes** | Variable | Batch sizes: 2, 4 (vs 4, 8, 16) |
| **OOM Auto-Recovery** | N/A | Catches errors, reduces batch size |
| **GPU Memory Growth** | N/A | Allocates memory as needed |
| **Gradient Accumulation** | Simulates larger batches | 2 steps accumulation |

---

## Files Created

### Python Scripts

1. **`hyperparameter_search_512.py`**
   - Performs 3-fold cross-validation hyperparameter search
   - Tests 24 configurations (3 arch × 2 LR × 2 dropout × 2 BS)
   - Total: 72 training runs
   - Includes OOM error handling with automatic batch size reduction
   - Saves best configuration to `summary.json`

2. **`density_analysis_512_best_model.py`**
   - Loads best configuration from hyperparameter search
   - Trains final model on full dataset (90/10 split)
   - Performs density analysis on `./test_images/`
   - Generates boxplots and representative tile comparisons
   - Includes CLAHE+OTSU baseline

### PBS Scripts

3. **`pbs_hyperparam_search_512.sh`**
   - Submits hyperparameter search job
   - Walltime: 24 hours
   - Memory: 240 GB
   - GPU: 1 GPU with memory growth

4. **`pbs_density_analysis_512.sh`**
   - Submits density analysis job using best model
   - Walltime: 8 hours
   - **IMPORTANT:** Update `HYPERPARAM_SEARCH_DIR` variable before submission!

---

## Workflow

### Step 1: Run Hyperparameter Search

```bash
# Submit hyperparameter search job
qsub pbs_hyperparam_search_512.sh

# Monitor job
qstat -u $USER

# Check progress (while running)
tail -f hyperparam_search_512_console_*.log

# Expected runtime: 12-18 hours
```

**What happens:**
- Loads 98 images from `dataset_shrunk_masks/`
- Trains 24 configurations × 3 folds = 72 models
- Tests learning rates: [1e-4, 5e-5]
- Tests dropouts: [0.2, 0.3]
- Tests batch sizes: [2, 4]
- If OOM error occurs: reduces batch size (4→2→1) and retries
- Saves results to `hyperparameter_search_512_YYYYMMDD_HHMMSS/`

**Output:**
```
hyperparameter_search_512_YYYYMMDD_HHMMSS/
├── all_results.csv           # All fold results
├── intermediate_results.csv  # Incremental save (in case of crash)
└── summary.json              # Best configuration ← IMPORTANT!
```

### Step 2: Review Hyperparameter Search Results

```bash
# Find output directory
ls -ltd hyperparameter_search_512_*

# Check best configuration
cat hyperparameter_search_512_*/summary.json

# Example output:
# {
#   "best_config": "resunet_lr5e-05_drop0.2_bs4",
#   "best_jaccard": 0.6234,
#   "total_configs": 24,
#   "successful_configs": 22
# }
```

**Interpret results:**
- `best_config`: Configuration name (architecture_LR_dropout_batchsize)
- `best_jaccard`: Mean Jaccard across 3 folds
- `successful_configs`: How many configs completed (vs OOM failures)

**If many OOM failures (successful_configs < 18):**
- Edit `hyperparameter_search_512.py`
- Reduce filters: `'filters': 32` → `'filters': 24` or `16`
- Re-submit job

### Step 3: Run Density Analysis with Best Model

**IMPORTANT:** Update PBS script with your results directory!

```bash
# Edit pbs_density_analysis_512.sh
# Find this line (around line 28):
HYPERPARAM_SEARCH_DIR="./hyperparameter_search_512_YYYYMMDD_HHMMSS"

# Replace YYYYMMDD_HHMMSS with your actual directory name!
# Example:
HYPERPARAM_SEARCH_DIR="./hyperparameter_search_512_20251014_123456"

# Submit density analysis job
qsub pbs_density_analysis_512.sh

# Monitor
tail -f density_analysis_512_console_*.log

# Expected runtime: 2-4 hours
```

**What happens:**
- Loads best configuration from `summary.json`
- Trains final model on full `dataset_shrunk_masks` (90/10 split)
- Saves trained model to `trained_model/best_model.keras`
- Predicts on `./test_images/` using 512×512 tiles
- Selects 5 representative tiles per image (min, 25th, median, 75th, max density)
- Generates boxplots and tile comparisons
- Includes CLAHE+OTSU baseline for comparison

**Output:**
```
density_analysis_512_best_YYYYMMDD_HHMMSS/
├── trained_model/
│   └── best_model.keras                    # Final trained model
├── representative_tiles/
│   ├── 10x_image_tile_00_comparison.png   # 2-panel: original | prediction
│   ├── 10x_image_tile_01_comparison.png
│   └── ... (~50 images total, 5 per test image)
├── boxplots/
│   ├── resunet_density_vs_dilution.png    # Best architecture
│   └── clahe_otsu_density_vs_dilution.png # Baseline
└── csv_data/
    └── density_analysis_comprehensive.csv  # All tile-level data
```

---

## Understanding the Results

### Hyperparameter Search Results

**`all_results.csv` columns:**
- `config_name`: Configuration identifier
- `architecture`: unet, resunet, attention_resunet
- `lr`: Learning rate
- `dropout`: Dropout rate
- `batch_size`: Batch size (may differ from config if OOM occurred)
- `batch_size_used`: Actual batch size used (after OOM reductions)
- `fold`: Fold number (1-3)
- `val_jaccard`: Validation Jaccard coefficient
- `train_jaccard`: Training Jaccard coefficient
- `overfitting_gap`: (train - val) × 100 (%)
- `best_epoch`: Epoch with best validation Jaccard
- `total_epochs`: Total epochs trained
- `oom_retries`: Number of OOM retries
- `success`: True/False (training completed?)
- `error`: Error message if failed

**Analysis tips:**
```python
import pandas as pd

# Load results
df = pd.read_csv('hyperparameter_search_512_*/all_results.csv')

# Show successful runs only
df_success = df[df['success'] == True]

# Group by configuration, show mean Jaccard
summary = df_success.groupby('config_name')['val_jaccard'].agg(['mean', 'std', 'count'])
print(summary.sort_values('mean', ascending=False))

# Check OOM failures
df_oom = df[df['success'] == False]
print(f"OOM failures: {len(df_oom)} / {len(df)}")
```

### Density Analysis Results

**`density_analysis_comprehensive.csv` columns:**
- `image`: Test image filename
- `dilution_factor`: Dilution factor (10, 20, 80, ..., 10240)
- `tile_idx`: Tile index
- `method`: Architecture name or 'clahe_otsu'
- `foreground_pct`: Foreground percentage (density metric)

**Analysis tips:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('density_analysis_512_best_*/csv_data/density_analysis_comprehensive.csv')

# Compare DL model vs CLAHE+OTSU
for method in df['method'].unique():
    df_method = df[df['method'] == method]
    mean_density = df_method.groupby('dilution_factor')['foreground_pct'].mean()
    plt.plot(mean_density.index, mean_density.values, marker='o', label=method)

plt.xlabel('Dilution Factor')
plt.ylabel('Mean Foreground %')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Density vs Dilution Factor (512×512)')
plt.show()
```

---

## Memory Management Details

### Why 512×512 is Challenging

| Aspect | 256×256 | 512×512 | Ratio |
|--------|---------|---------|-------|
| **Pixels per image** | 65,536 | 262,144 | 4× |
| **Memory per batch (BS=8)** | ~50 MB | ~200 MB | 4× |
| **Gradient memory** | ~100 MB | ~400 MB | 4× |
| **Activation memory** | ~200 MB | ~800 MB | 4× |
| **Total (approximate)** | ~350 MB | ~1,400 MB | 4× |

With 64 filters and batch size 8, a 512×512 model can easily exceed 8 GB GPU memory!

### Mixed Precision Training (FP16)

**How it works:**
- Stores activations in 16-bit floats (instead of 32-bit)
- Reduces memory by ~40%
- Speeds up training by ~2× on modern GPUs
- Maintains numerical stability using loss scaling

**Enabled automatically in scripts:**
```python
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)
```

### OOM Auto-Recovery

**How it works:**
1. Try training with configured batch size (e.g., 4)
2. If `ResourceExhaustedError` occurs:
   - Clear GPU memory
   - Reduce batch size by half (4 → 2)
   - Retry training
3. Repeat up to 2 times (can reduce to batch size 1)
4. If still fails, mark configuration as failed

**Code snippet:**
```python
batch_size = config['batch_size']  # e.g., 4
oom_retries = 0

while oom_retries <= max_retries:
    try:
        model.fit(X_train, y_train, batch_size=batch_size, ...)
        break  # Success!
    except tf.errors.ResourceExhaustedError:
        batch_size = batch_size // 2
        oom_retries += 1
```

### Gradient Accumulation

Simulates larger batch sizes without using more memory:
- Accumulate gradients over 2 steps
- Then update weights
- Effective batch size = actual batch size × accumulation steps
- Example: BS=2 × 2 steps = effective BS=4

**Trade-off:**
- Pros: Reduces memory, allows smaller physical batches
- Cons: Slightly slower training (2× gradient computations per update)

---

## Comparison: 256×256 vs 512×512

| Aspect | 256×256 | 512×512 |
|--------|---------|---------|
| **Dataset** | `dataset_full_stack/` | `dataset_shrunk_masks/` |
| **Images** | 1,980 (100 images × ~20 patches) | 98 (native tiles) |
| **Channels** | 1 (grayscale) | 3 (RGB) |
| **Filters** | 64 | 32 (reduced for memory) |
| **Batch Sizes** | [4, 8, 16] | [2, 4] (smaller!) |
| **Learning Rates** | [1e-5, 2e-5, 5e-5] | [5e-5, 1e-4] (higher!) |
| **Training Time** | ~15 min/epoch | ~60 min/epoch (4× slower) |
| **Memory Required** | ~2-4 GB | ~6-10 GB |
| **OOM Risk** | Low | High (managed) |
| **Expected Jaccard** | ~0.60 (best ResUNet) | TBD (likely similar) |

**Key insight:** Higher learning rates for 512×512 help convergence despite slower epochs.

---

## Troubleshooting

### Problem 1: All Configurations Fail with OOM

**Symptoms:**
- `successful_configs: 0` in summary.json
- All rows in `all_results.csv` have `success=False`

**Solutions:**
1. Reduce filters in `hyperparameter_search_512.py`:
   ```python
   'filters': 32,  # Change to 24 or 16
   ```

2. Reduce batch sizes further:
   ```python
   'batch_sizes': [2, 4],  # Change to [1, 2]
   ```

3. Check GPU memory:
   ```bash
   nvidia-smi
   # Make sure no other jobs are using GPU
   ```

### Problem 2: Density Analysis Can't Find Hyperparameter Results

**Symptoms:**
```
✗ ERROR: Hyperparameter search directory not found!
```

**Solutions:**
1. Update `HYPERPARAM_SEARCH_DIR` in `pbs_density_analysis_512.sh`:
   ```bash
   HYPERPARAM_SEARCH_DIR="./hyperparameter_search_512_YYYYMMDD_HHMMSS"
   ```

2. Check available directories:
   ```bash
   find . -maxdepth 1 -type d -name "hyperparameter_search_512_*"
   ```

### Problem 3: Image Size Mismatch

**Symptoms:**
```
⚠ WARNING: Expected 512×512, got (XXX, YYY)
```

**Solutions:**
1. Verify dataset images are 512×512:
   ```bash
   python3 -c "from PIL import Image; img = Image.open('dataset_shrunk_masks/images/[filename].png'); print(img.size)"
   ```

2. If images are different size, they'll be resized automatically (but may lose quality)

### Problem 4: Training Too Slow

**Symptoms:**
- Epochs take >90 minutes
- Job exceeds 24-hour walltime

**Solutions:**
1. Reduce number of configurations:
   ```python
   'learning_rates': [5e-5],  # Test only one LR
   'dropouts': [0.2],          # Test only one dropout
   ```

2. Reduce number of folds:
   ```python
   'n_folds': 2,  # Instead of 3
   ```

3. Reduce epochs:
   ```python
   'epochs': 20,  # Instead of 30
   ```

---

## Expected Outputs Timeline

### Hyperparameter Search (12-18 hours)

```
Hour 0-1:   Setup, load data, start first configuration
Hour 1-3:   Train 3-4 configurations (each ~30-45 min × 3 folds)
Hour 3-9:   Continue training (bulk of configurations)
Hour 9-12:  Final configurations
Hour 12-18: Complete, save results (if slow or OOM retries)
```

### Density Analysis (2-4 hours)

```
Hour 0-2:   Train final model on full dataset
Hour 2-3:   Predict on test images
Hour 3-4:   Generate visualizations and CSV
```

---

## Scientific Implications

### Why Test 512×512?

1. **Higher Resolution**
   - Captures finer particle details
   - May improve segmentation of small particles
   - Better boundary precision

2. **Native Test Image Resolution**
   - Test images are large (several thousand pixels)
   - 512×512 tiles cover 4× more area per tile
   - Fewer tiles needed → faster inference

3. **Comparison with 256×256**
   - Does higher resolution improve Jaccard?
   - Or is 256×256 sufficient?
   - Trade-off: accuracy vs computational cost

### Interpreting Results

**If 512×512 Jaccard ≈ 256×256 Jaccard:**
- Resolution doesn't significantly improve performance
- **Recommendation:** Use 256×256 for efficiency

**If 512×512 Jaccard > 256×256 Jaccard (+5% or more):**
- Higher resolution provides better segmentation
- **Recommendation:** Use 512×512 despite higher cost

**If 512×512 Jaccard < 256×256 Jaccard:**
- Possible overfitting (fewer training images)
- OR insufficient model capacity (reduced filters)
- **Recommendation:** Increase filters or augment data

---

## Next Steps After Completion

1. **Compare Results:**
   - 512×512 best Jaccard vs 256×256 best Jaccard
   - Visual inspection of representative tiles
   - Density trends across dilution series

2. **Publication-Ready Figures:**
   - Side-by-side comparison: 256×256 vs 512×512 tile segmentations
   - Multi-panel density plots (both resolutions)
   - Table: Performance metrics across resolutions

3. **Model Deployment:**
   - If 512×512 is better: Use for production inference
   - If 256×256 is sufficient: Deploy for efficiency
   - Consider hybrid: 256×256 for screening, 512×512 for detailed analysis

---

**Workflow Complete!**

For questions or issues, check console logs:
- `hyperparam_search_512_console_*.log`
- `density_analysis_512_console_*.log`
