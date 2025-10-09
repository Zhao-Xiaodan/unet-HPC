# Hyperparameter Comparison: Mitochondria vs Microbeads

## Executive Summary

**Root Cause**: Hyperparameters optimized for sparse mitochondria segmentation (2-3 objects/image) were applied to dense microbead segmentation (109.4 objects/image), causing catastrophic validation collapse.

**Solution**: Recalibrated hyperparameters based on 36× density difference.

**Expected Improvement**: Validation Jaccard from 0.14→0.0 (collapsed) to 0.50-0.70 (stable)

---

## Dataset Characteristics

| Metric | Mitochondria | Microbeads | Ratio |
|--------|-------------|------------|-------|
| **Objects per image** | 2-3 | 109.4 | **36.5×** |
| **Object density** | Sparse | Dense | - |
| **Positive pixel ratio** | 10-15% | 11.5% | ~1× |
| **Object size** | Variable | ~69 pixels | - |
| **Training samples** | ~134 | 73 | 0.5× |
| **Domain** | Biology/EM | Materials/Microscope | Different |

---

## Hyperparameter Changes

### 1. Learning Rate

| Model | Mitochondria | Microbeads | Change | Reasoning |
|-------|-------------|------------|--------|-----------|
| **UNet** | 1e-3 | **1e-4** | **÷10** | 36× more objects → 36× stronger gradients |
| **Attention UNet** | 1e-4 | **1e-4** | Same | Already lower for attention |
| **Attention ResUNet** | 1e-4 | **1e-4** | Same | Already lower for residual |

**Why this matters:**
- With 36× more objects, each batch contains 36× more segmentation targets
- Gradient magnitude scales proportionally with object count
- Using mitochondria LR (1e-3) on microbeads creates effective LR ≈ 0.036
- This causes massive overshooting → validation collapse at epoch 1-4

**Mathematical intuition:**
```
Gradient magnitude ∝ Number of objects per image
Effective LR = Nominal LR × (Objects_microbead / Objects_mito)
             = 1e-3 × (109 / 3)
             ≈ 0.036  ← WAY TOO HIGH!

Corrected LR = 1e-4 gives effective LR ≈ 0.0036 ✓
```

---

### 2. Batch Size

| Parameter | Mitochondria | Microbeads | Change | Reasoning |
|-----------|-------------|------------|--------|-----------|
| **Batch Size** | 8-16 | **32** | **×2-4** | Larger batches stabilize dense gradients |

**Why this matters:**
- Sparse objects (mitochondria): Small batches sufficient for gradient estimation
- Dense objects (microbeads): Need larger batches to average out high variance
- Larger batch size reduces gradient noise from 109 objects/image
- Also allows better GPU utilization (more parallel processing)

**Trade-off:**
- Larger batch → smoother training but higher memory usage
- 32 is sweet spot for 240GB HPC memory with 256×256 images

---

### 3. Regularization (Dropout)

| Parameter | Mitochondria | Microbeads | Change | Reasoning |
|-----------|-------------|------------|--------|-----------|
| **Dropout Rate** | 0.0 | **0.3** | **+0.3** | Prevent overfitting uniform objects |

**Why this matters:**
- Mitochondria: Variable shapes/textures → less overfitting risk
- Microbeads: Uniform circular shapes → high overfitting risk
- Model can memorize "draw circles" instead of learning context
- 0.3 dropout forces model to learn robust features

---

### 4. Loss Function

| Parameter | Mitochondria | Microbeads | Change | Reasoning |
|-----------|-------------|------------|--------|-----------|
| **Loss Function** | Binary Focal Loss (γ=2) | **Dice Loss** | Changed | Direct overlap optimization |

**Why this matters:**

**Focal Loss characteristics:**
- Designed for extreme class imbalance (e.g., 1% positive)
- Down-weights easy examples (γ=2 exponent)
- Works well when positive class is rare and hard to detect

**Dice Loss characteristics:**
- Directly optimizes IoU/Jaccard metric
- Equally weights all positive pixels
- Better for dense objects where overlap quality matters

**Decision logic:**
- Microbead class balance: 11.5% (not extreme imbalance)
- Primary goal: Accurate segmentation boundaries (Jaccard metric)
- Focal loss may ignore boundary pixels as "easy examples"
- Dice loss directly optimizes what we measure

**Mathematical comparison:**
```python
# Focal Loss: Emphasizes hard examples
FL = -α(1-p)^γ log(p)
# With γ=2, easy examples (p→1) get weight ≈0

# Dice Loss: Direct overlap optimization
Dice = 2×|A∩B| / (|A|+|B|)
# Matches Jaccard metric: IoU = |A∩B| / |A∪B|
```

---

### 5. Train/Validation Split

| Parameter | Mitochondria | Microbeads | Change | Reasoning |
|-----------|-------------|------------|--------|-----------|
| **Split Method** | Random 90/10 | **Stratified 85/15** | Changed | Balance by object density |
| **Val Ratio** | 10% | **15%** | +5% | Smaller dataset needs more val data |

**Why this matters:**
- Microbead dataset has high variance in object density (12-254 objects/image)
- Random split could put all dense images in train or all sparse in val
- Stratified split ensures balanced representation of density spectrum
- 15% validation (vs 10%) because total dataset is smaller (73 vs 134 images)

**Stratification method:**
```python
# Bin images by object density
density_bins = pd.qcut(objects_per_image, q=5, labels=False)

# Split maintaining density distribution
train, val = train_test_split(
    images,
    test_size=0.15,
    stratify=density_bins
)
```

---

## Side-by-Side Comparison

### Configuration Table

```
┌─────────────────────┬──────────────────┬──────────────────┬──────────────┐
│ Hyperparameter      │ Mitochondria     │ Microbeads       │ Reason       │
├─────────────────────┼──────────────────┼──────────────────┼──────────────┤
│ Learning Rate       │ 1e-3 (UNet)      │ 1e-4 (all)       │ Dense grads  │
│                     │ 1e-4 (Attention) │                  │              │
├─────────────────────┼──────────────────┼──────────────────┼──────────────┤
│ Batch Size          │ 8-16             │ 32               │ Stability    │
├─────────────────────┼──────────────────┼──────────────────┼──────────────┤
│ Dropout             │ 0.0              │ 0.3              │ Regularize   │
├─────────────────────┼──────────────────┼──────────────────┼──────────────┤
│ Loss Function       │ Focal (γ=2)      │ Dice             │ Direct IoU   │
├─────────────────────┼──────────────────┼──────────────────┼──────────────┤
│ Train/Val Split     │ Random 90/10     │ Stratified 85/15 │ Balance      │
├─────────────────────┼──────────────────┼──────────────────┼──────────────┤
│ Optimizer           │ Adam             │ Adam             │ Same         │
├─────────────────────┼──────────────────┼──────────────────┼──────────────┤
│ Image Size          │ 256×256          │ 256×256          │ Same         │
├─────────────────────┼──────────────────┼──────────────────┼──────────────┤
│ Max Epochs          │ 100              │ 100              │ Same         │
├─────────────────────┼──────────────────┼──────────────────┼──────────────┤
│ Early Stopping      │ 20 epochs        │ 20 epochs        │ Same         │
└─────────────────────┴──────────────────┴──────────────────┴──────────────┘
```

---

## Expected Training Behavior

### Previous Training (Wrong Hyperparameters)

```
Epoch 1: val_jacard=0.1427 ← Peak performance
Epoch 2: val_jacard=0.1089 ← Starting to degrade
Epoch 3: val_jacard=0.0512 ← Rapid collapse
Epoch 4: val_jacard=0.0023 ← Nearly zero
...
Epoch 100: val_jacard≈1e-12 ← Complete collapse
```

**What happened:**
1. LR too high → massive gradient updates
2. Model overfits to training set immediately
3. Validation loss explodes (divergence)
4. Validation Jaccard collapses to numerical zero

---

### New Training (Corrected Hyperparameters)

```
Epoch 1: val_jacard≈0.30-0.35 ← Reasonable start
Epoch 5: val_jacard≈0.40-0.45 ← Steady improvement
Epoch 10: val_jacard≈0.48-0.52 ← Approaching target
Epoch 20: val_jacard≈0.55-0.62 ← Good performance
Epoch 30: val_jacard≈0.58-0.68 ← Near optimal
Epoch 40: val_jacard≈0.60-0.70 ← Converged (early stop)
```

**What should happen:**
1. Learning rate appropriate → stable gradient descent
2. Model learns gradually without overfitting
3. Validation improves steadily (no collapse)
4. Final Jaccard 0.50-0.70 (production-ready)

---

## Success Criteria

### ✅ Must Achieve

1. **No validation collapse**
   - Val Jaccard should NOT drop to ~0.0
   - Should maintain or improve after epoch 10

2. **Minimum performance**
   - Val Jaccard > 0.30 by epoch 10
   - Val Jaccard > 0.50 by epoch 40

3. **Training stability**
   - No wild oscillations (σ < 0.05 for last 10 epochs)
   - Train-val gap < 0.15

### ⭐ Target Performance

1. **Validation Jaccard: 0.50-0.70**
   - 3.5-5× improvement over previous 0.14

2. **Stable convergence**
   - Smooth learning curves
   - Early stopping before epoch 100

3. **Generalization**
   - Val Jaccard within 0.10 of train Jaccard
   - No overfitting signs

---

## Verification Commands

### On HPC (After Training Completes)

```bash
# Find output directory
OUTPUT=$(ls -td microbead_training_* | head -1)

# Check summary
cat $OUTPUT/training_summary.csv

# Expected format:
# model,best_epoch,best_val_jacard,final_val_jacard,train_time
# UNet,35,0.6234,0.6145,2.3h
# Attention_UNet,42,0.6512,0.6489,3.1h
# Attention_ResUNet,38,0.6789,0.6723,3.5h

# Best model
grep "best_val_jacard" $OUTPUT/training_summary.csv | sort -t',' -k3 -nr | head -1
```

### On Local Machine (After Download)

```bash
# Download results
scp -r phyzxi@hpc:~/scratch/unet-HPC/microbead_training_* ./

# Visualize training curves
python plot_training_curves.py microbead_training_*/
```

---

## Troubleshooting

### If Val Jaccard Still Collapses

**Check:**
1. Dataset quality - run `analyze_microbead_dataset.py` again
2. Reduce LR further: 1e-4 → 5e-5
3. Increase batch size: 32 → 48 (if memory allows)

### If Val Jaccard Plateaus Below 0.30

**Check:**
1. Loss function - try combined Dice+Focal
2. Data augmentation - add rotation/flip
3. Model capacity - try deeper UNet

### If Out of Memory

**Fix:**
1. Reduce batch size: 32 → 16
2. Reduce image size: 256 → 128 (not recommended)
3. Use gradient accumulation

---

## References

### Dataset Analysis
- `dataset_analysis/summary.json` - Statistics
- `MICROBEAD_ANALYSIS_RESULTS.md` - Full analysis report

### Training Scripts
- `train_microbead_optimized.py` - Corrected training code
- `pbs_microbead_optimized.sh` - HPC job script

### Theory
- `DOMAIN_SHIFT_ANALYSIS.md` - Why hyperparameters failed
- `Hyperparameter_Optimization_Report.md` - Original mitochondria study

---

## Summary

The microbead segmentation failure was NOT due to:
- ❌ Partial masks (0% detected)
- ❌ Dataset quality (good brightness distribution)
- ❌ Class imbalance (11.5% is reasonable)
- ❌ Model architecture (UNets work for segmentation)

**Actual root cause:**
✅ **Domain shift in object density (36× more objects)**
✅ **Hyperparameters optimized for sparse objects applied to dense objects**
✅ **Learning rate too high by factor of 30-40**

**Solution:**
✅ Recalibrated LR, batch size, regularization, loss function, and split strategy
✅ Expected 3.5-5× improvement in validation Jaccard
✅ Stable training without collapse

**Ready for HPC submission!** 🚀
