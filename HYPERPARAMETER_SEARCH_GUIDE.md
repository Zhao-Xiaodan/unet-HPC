# Hyperparameter Search Guide - ResUNet & Attention ResUNet

## Overview

Based on the architecture comparison study, ResUNet and Attention ResUNet underperformed with baseline hyperparameters:

| Architecture | Performance | Issue |
|--------------|-------------|-------|
| U-Net (baseline) | 69.94% ± 5.02% | ✅ Works well |
| **ResUNet** | 39.95% ± 6.79% | ❌ **Catastrophic failure (-42.9%)** |
| **Attention ResUNet** | 62.69% ± 3.71% | ⚠️ Moderate underperformance (-10.4%) |

**Goal:** Find optimal hyperparameters to make ResUNet and Attention ResUNet match or exceed U-Net's performance.

---

## Why ResUNet Failed: Root Cause Analysis

### Evidence from Training Curves

**ResUNet behavior:**
- Peaks at epoch 2-3
- Then validation performance **crashes**
- Consistent across all 5 folds

**Root cause hypothesis:** **Learning rate mismatch**

Residual connections fundamentally change gradient flow:
- Create "shortcut" paths that amplify gradients
- Effective learning rate becomes **too high**
- Causes rapid learning → overshooting → collapse

**Baseline settings (designed for U-Net):**
- Learning rate: 5e-5
- Dropout: 0.3
- Batch size: 4

**These are likely suboptimal for residual architectures.**

---

## Search Strategy

### Hyperparameters to Search

Based on failure analysis, we focus on:

#### 1. **Learning Rate** (Most Critical)

```python
learning_rate = [1e-5, 2e-5, 5e-5]
```

**Rationale:**
- **1e-5:** 5× lower than baseline (conservative, may stabilize training)
- **2e-5:** 2.5× lower (moderate adjustment)
- **5e-5:** Baseline (for comparison)

**Expected impact:** Lower LR should prevent overshooting and allow sustained learning.

#### 2. **Dropout** (Regularization)

```python
dropout = [0.3, 0.4, 0.5]
```

**Rationale:**
- **0.3:** Baseline (already tested)
- **0.4:** Higher regularization (may reduce overfitting gap)
- **0.5:** Aggressive regularization (for complex architectures)

**Expected impact:** Higher dropout may improve generalization, especially with residual connections that risk overfitting.

#### 3. **Batch Size** (Gradient Noise)

```python
batch_size = [4, 8]
```

**Rationale:**
- **4:** Baseline (higher gradient noise → more regularization)
- **8:** Larger batch (smoother gradients → more stable training)

**Expected impact:** Larger batches may stabilize ResUNet training.

### Fixed Parameters

These remain constant (already validated):
- **Filters:** 64
- **Loss function:** Combined (0.7×Dice + 0.3×Focal)
- **CV folds:** 3 (for efficiency)
- **Max epochs:** 25
- **Early stopping patience:** 8

---

## Search Space Summary

**Total configurations:**
- 2 architectures (ResUNet, Attention ResUNet)
- 3 learning rates
- 3 dropout values
- 2 batch sizes
- **= 36 configurations**

**With 3-fold CV:**
- 36 configs × 3 folds = **108 models to train**

**Estimated time:**
- ~5-7 minutes per model
- **Total: 8-12 hours**

---

## How to Run

### Local Execution (if you have GPU)

```bash
# Activate environment
conda activate unetCNN

# Run search
python hyperparameter_search_residual_architectures.py

# Expected runtime: 8-12 hours
```

### HPC Execution (Recommended)

```bash
# 1. Transfer files to HPC
scp hyperparameter_search_residual_architectures.py \
    model_architectures.py loss_functions_fixed.py \
    pbs_hyperparameter_search.sh \
    phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/

# 2. SSH and submit
ssh phyzxi@nscc
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_hyperparameter_search.sh

# 3. Monitor
qstat -u phyzxi
tail -f Hyperparam_Search.o<JOB_ID>
```

**PBS Configuration:**
- Walltime: 12 hours
- GPU: 1
- CPUs: 36
- Memory: 240GB

---

## Expected Outcomes

### Scenario 1: ResUNet Fix Found ✅

**If lower learning rate works:**

Best config might be:
```python
resunet_optimal = {
    'learning_rate': 1e-5,  # 5× lower than baseline
    'dropout': 0.4,          # Slightly higher
    'batch_size': 8          # Larger batch
}
```

**Performance target:** 65-72% Jaccard (match or exceed U-Net's 69.94%)

**Indicators of success:**
- Best epoch shifts to 8-12 (vs current 2-3)
- Overfitting gap drops to ~2× (vs current 3.1×)
- Validation curves show stable plateaus (not crashes)

**Next step:** Re-run full 5-fold CV with optimal config, compare to U-Net statistically.

---

### Scenario 2: Partial Improvement ⚠️

**If performance improves but doesn't match U-Net:**

Example: ResUNet reaches 60-65% (better than 39.95%, worse than 69.94%)

**Interpretation:**
- Hyperparameters help but aren't the only issue
- Architecture may be inherently less suited to this task
- Simpler U-Net may be more appropriate for microbead segmentation

**Recommendation:** Stick with U-Net unless specific need for residual connections.

---

### Scenario 3: No Improvement ❌

**If all configs fail to exceed 50% Jaccard:**

**Interpretation:**
- ResUNet fundamentally incompatible with this task/data
- Problem isn't just hyperparameters
- May be dataset-specific issue (small dataset, simple task)

**Recommendation:**
- **Abandon ResUNet** for this application
- **Continue with U-Net** (69.94% is excellent)
- Focus efforts on:
  - Data augmentation
  - Ensemble methods
  - Post-processing improvements

---

## Interpreting Results

### Output Files

After search completes:

```
hyperparameter_search_YYYYMMDD_HHMMSS/
├── hyperparameter_search_summary.json    # KEY FILE - read this first
├── resunet_lr1e-05_drop0.3_bs4/
│   ├── fold_1/
│   │   ├── best_model.keras
│   │   ├── history.csv
│   │   └── results.json
│   └── fold_2, fold_3/
├── resunet_lr1e-05_drop0.4_bs4/
└── ... (all 36 configurations)
```

### Key Metrics in Summary JSON

```json
{
  "best_configs": {
    "resunet": {
      "config_name": "resunet_lr1e-05_drop0.4_bs8",
      "mean_best_jacard": 0.6500,  // Target: > 0.6994
      "std_best_jacard": 0.0350,
      "mean_best_epoch": 10.3,     // Healthy: 8-15
      "mean_overfitting_gap": 2.1, // Good: < 2.5×
      "config": {
        "learning_rate": 1e-05,
        "dropout": 0.4,
        "batch_size": 8
      }
    }
  }
}
```

**What to look for:**

✅ **Success indicators:**
- `mean_best_jacard` ≥ 0.68 (within 2% of U-Net)
- `mean_best_epoch` in 8-15 range (not 1-3!)
- `mean_overfitting_gap` < 2.5×
- Low `std_best_jacard` (<0.05)

❌ **Failure indicators:**
- `mean_best_jacard` < 0.55
- `mean_best_epoch` < 5 (still collapsing early)
- `mean_overfitting_gap` > 3.0×
- High variance across folds

---

## Analyzing Training Curves

For the best configuration, examine training curves:

```bash
# Load history CSV
import pandas as pd
df = pd.read_csv('hyperparameter_search_*/resunet_lr1e-05_drop0.4_bs8/fold_1/history.csv')

# Check for healthy training
import matplotlib.pyplot as plt
plt.plot(df['val_jacard_coef'], label='Validation')
plt.plot(df['jacard_coef'], label='Training')
plt.legend()
plt.show()
```

**Healthy pattern:**
- Validation curve **steadily rises** then plateaus
- Peak occurs at epoch 8-15 (not 1-3)
- No sharp drops after peak
- Train/val gap widens gradually

**Unhealthy pattern (same as baseline ResUNet):**
- Validation peaks at epoch 1-3
- Sharp drop after peak
- Large train/val gap early
- Erratic oscillations

---

## Decision Tree

After search completes:

```
Did best ResUNet config reach ≥68% Jaccard?
│
├─ YES ✅
│  └─ Run full 5-fold CV with optimal config
│     └─ Statistical test vs U-Net
│        ├─ If significantly better: Use ResUNet
│        └─ If not significant: U-Net is simpler, use that
│
└─ NO ❌
   └─ Did improvement occur vs baseline (39.95%)?
      │
      ├─ YES (e.g., 50-60%)
      │  └─ Partial fix found but not enough
      │     └─ Consider: Is 60% acceptable?
      │        ├─ NO → Stick with U-Net (69.94%)
      │        └─ YES → Maybe ResUNet if you need residuals for other reasons
      │
      └─ NO (still <45%)
         └─ ResUNet incompatible with this task
            └─ RECOMMENDATION: Use U-Net, abandon ResUNet
```

---

## What About Attention ResUNet?

Attention ResUNet (62.69% baseline) is **closer to U-Net** than ResUNet (39.95%).

**Search may help if:**
- Optimal config found: LR=2e-5, dropout=0.4, batch=8
- Performance improves to 67-70% (match U-Net)

**But consider:**
- Attention adds complexity (34.2M params vs U-Net's 31.4M)
- 36% slower training (49.2s vs 36.1s per epoch)
- Only worth it if **significantly better** than U-Net

**Threshold for adoption:**
- Must exceed 71% Jaccard (>1% absolute improvement)
- Must be statistically significant (p < 0.05)
- Training time overhead must be acceptable

---

## Alternative Approaches (If Search Fails)

If hyperparameter search doesn't help ResUNet reach U-Net's performance:

### 1. **Architecture Modifications**

Try different ResUNet variants:
```python
# Pre-activation residual blocks
# Bottleneck residual blocks
# Different number of residual layers
```

### 2. **Training Schedule Changes**

```python
# Learning rate warmup
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-6,
    decay_steps=1000,
    warmup_steps=200
)

# Cyclic learning rates
# Two-stage training (freeze encoder first)
```

### 3. **Loss Function Tuning**

```python
# Try different Dice/Focal ratios
loss = 0.5 * dice_loss + 0.5 * focal_loss

# Add boundary loss
# Try Tversky loss (adjustable precision/recall)
```

### 4. **Abandon Residual Architectures**

**Pragmatic approach:**
- U-Net works excellently (69.94%)
- ResUNet adds complexity without benefit (for this task)
- Focus on improving U-Net instead:
  - Ensemble multiple U-Net models
  - Better data augmentation
  - Test/train time augmentation (TTA)

---

## Success Criteria

Define success threshold before starting search:

### Minimum Viable Improvement

**For ResUNet:**
- **Target:** 65% Jaccard (vs baseline 39.95%)
- **Success:** 15+ percentage point improvement
- **Deploy:** Only if ≥68% (within 2% of U-Net)

**For Attention ResUNet:**
- **Target:** 68% Jaccard (vs baseline 62.69%)
- **Success:** 5+ percentage point improvement
- **Deploy:** Only if ≥70% (better than U-Net)

### Statistical Significance

If improvement found:
- Must be **reproducible** (consistent across folds)
- Must be **statistically significant** vs baseline
- Standard deviation should **decrease** (more stable)

---

## Cost-Benefit Analysis

Before deploying optimized ResUNet/Attention ResUNet:

| Factor | U-Net | Optimized ResUNet | Worth It? |
|--------|-------|-------------------|-----------|
| **Performance** | 69.94% | 68-72% (goal) | Only if >70% |
| **Training Time** | 36s/epoch | 45s/epoch (+25%) | Tolerable |
| **Model Size** | 31M params | 33M params (+6%) | Tolerable |
| **Complexity** | Simple | Residual blocks | Higher maintenance |
| **Stability** | Proven | Needs tuning | Risk |

**Decision rule:**
- If ResUNet ≥ 71%: Consider adoption (marginal benefit)
- If ResUNet 68-70%: Probably not worth complexity
- If ResUNet < 68%: Definitely stick with U-Net

---

## Quick Start

```bash
# 1. Submit HPC job
qsub pbs_hyperparameter_search.sh

# 2. Monitor progress (~8-12 hours)
tail -f Hyperparam_Search.o<JOB_ID>

# 3. After completion, check summary
cat hyperparameter_search_*/hyperparameter_search_summary.json

# 4. Look for best_configs section
python -c "
import json
with open('hyperparameter_search_*/hyperparameter_search_summary.json') as f:
    data = json.load(f)
    best = data['best_configs']
    for arch, config in best.items():
        perf = config['mean_best_jacard']
        print(f'{arch}: {perf:.4f} (target: >0.6994)')
"

# 5. If success found, run full CV
# (create new script with optimal hyperparameters)
```

---

## Troubleshooting

### Search Takes Too Long

**Problem:** 12-hour walltime insufficient

**Solutions:**
1. Reduce search space:
   ```python
   HYPERPARAMETER_GRID = {
       'learning_rate': [1e-5, 2e-5],  # Remove 5e-5
       'dropout': [0.3, 0.4],           # Remove 0.5
       'batch_size': [4],                # Remove 8
   }
   # Reduces from 36 to 12 configs
   ```

2. Use 2-fold CV instead of 3-fold (faster but less reliable)

3. Reduce max epochs to 20 (from 25)

### OOM Errors

**Problem:** GPU runs out of memory with batch_size=8

**Solution:** Remove batch_size=8 from search, only test batch_size=4

### No Improvement Found

**Problem:** All configs still underperform

**Accept reality:**
- ResUNet may not work for this task
- U-Net is excellent (69.94%)
- Don't force a solution that doesn't exist

---

## Summary

**Goal:** Find hyperparameters to fix ResUNet's catastrophic failure (39.95% → target 70%)

**Method:** Grid search over learning rate (lower), dropout (higher), batch size

**Timeline:** 8-12 hours on HPC

**Success criteria:**
- ResUNet ≥ 68% Jaccard
- Stable training (best epoch 8-15)
- Low overfitting gap (<2.5×)

**If successful:** Run full 5-fold CV, compare to U-Net statistically

**If unsuccessful:** Accept that U-Net is best for this task, focus elsewhere

---

*This guide provides a systematic approach to fixing residual architecture training failures. However, remember: if hyperparameters don't help, the architectures may simply be incompatible with your task. U-Net's 69.94% performance is excellent—don't overengineer if it's working well.*
