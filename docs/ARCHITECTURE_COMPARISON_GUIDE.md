# Architecture Comparison Study Guide

## Overview

This guide explains the architecture comparison study designed to systematically evaluate three U-Net variants for microbead segmentation:

1. **U-Net** (baseline)
2. **ResUNet** (Residual U-Net)
3. **Attention ResUNet** (Attention + Residual U-Net)

---

## Architectural Differences

### 1. U-Net (Baseline)

**Architecture:**
```
Standard convolutional blocks: Conv → BN → ReLU → Conv → BN → ReLU → Dropout
Encoder-decoder structure with skip connections
```

**Characteristics:**
- **Parameters:** ~31M (filters=64)
- **Strengths:** Simple, fast training, proven performance
- **Limitations:** Vanishing gradients in deep networks, equal weight to all features
- **Validated Performance:** 60.97% ± 11.5% Jaccard (5-fold CV)

**When to Use:**
- Baseline for any segmentation task
- When simplicity and speed are priorities
- Limited computational resources

---

### 2. ResUNet (Residual U-Net)

**Architecture:**
```
Residual blocks: Conv → BN → ReLU → Conv → BN + 1×1 Shortcut → Add → ReLU → Dropout
Same encoder-decoder structure as U-Net
```

**Key Innovation:** Residual connections (shortcuts)
```python
# Standard U-Net block
output = Conv2D(x)
output = ReLU(output)

# ResUNet block
output = Conv2D(x)
output = ReLU(output)
shortcut = Conv2D_1x1(x)
output = Add([output, shortcut])  # <-- Residual connection
output = ReLU(output)
```

**Characteristics:**
- **Parameters:** ~34M (filters=64, +10% vs U-Net)
- **Strengths:**
  - Better gradient flow through network
  - Easier to train deeper networks
  - Faster convergence (fewer epochs to peak)
  - More stable training
- **Limitations:** Slightly more parameters and computation

**Expected Performance:**
- **Improvement:** +2-5% over U-Net
- **Convergence:** Reach best performance 20-30% faster
- **Training Time:** +5-15% per epoch

**When to Use:**
- Need faster convergence (limited training time)
- Training deeper networks (>4 encoder levels)
- Gradient flow issues observed
- Willing to trade +10% computation for better performance

---

### 3. Attention ResUNet

**Architecture:**
```
Residual blocks (same as ResUNet)
+ Attention gates at each skip connection
```

**Key Innovation:** Attention gates filter skip connection features
```python
# Standard U-Net skip connection
decoder_features = Concatenate([upsampled, encoder_features])

# Attention ResUNet skip connection
attention_weights = AttentionGate(encoder_features, upsampled)  # Learn what to focus on
filtered_features = Multiply([encoder_features, attention_weights])  # Apply attention
decoder_features = Concatenate([upsampled, filtered_features])  # Use filtered features
```

**Attention Mechanism:**
```
For each skip connection:
1. Encoder features (high-res) and decoder features (low-res) are aligned
2. Attention gate learns spatial weights (0-1 for each pixel)
3. Weights suppress irrelevant regions, amplify relevant regions
4. Filtered features passed to decoder
```

**Characteristics:**
- **Parameters:** ~36M (filters=64, +16% vs U-Net)
- **Strengths:**
  - Focus on relevant spatial regions (boundaries, small objects)
  - Better segmentation of overlapping objects
  - Improved boundary precision
  - Handles cluttered scenes better
- **Limitations:** Most computational overhead (+15-25% training time)

**Expected Performance:**
- **Improvement:** +3-7% over U-Net
- **Specific Gains:**
  - Better at object boundaries (+10-15% edge Jaccard)
  - Better at overlapping beads (+5-10%)
  - Better at small/occluded beads (+8-12%)
- **Training Time:** +15-25% per epoch

**When to Use:**
- Object boundaries are critical (segmentation quality)
- Many overlapping or clustered objects
- Small objects difficult to detect
- Willing to trade +20% computation for best performance

---

## Why These Architectures Matter for Microbead Segmentation

### Your Data Characteristics:
- **100 images, 512×512 pixels** (resized to 256×256 for training)
- **Density:** 5.6% foreground (microbeads are sparse)
- **Challenges:**
  - Overlapping/touching beads (hard to separate)
  - Variable sizes (different dilution levels)
  - Clustered regions (high local density)
  - Sparse distribution (class imbalance)

### Expected Architecture Benefits:

| Challenge | U-Net | ResUNet | Attention ResUNet |
|-----------|-------|---------|-------------------|
| **Overlapping Beads** | ★★☆ | ★★★ | ★★★★ |
| **Small Beads** | ★★☆ | ★★★ | ★★★★ |
| **Boundary Precision** | ★★☆ | ★★★ | ★★★★ |
| **Training Speed** | ★★★★ | ★★★★ | ★★★☆ |
| **Inference Speed** | ★★★★ | ★★★☆ | ★★☆☆ |
| **Simplicity** | ★★★★ | ★★★☆ | ★★☆☆ |

---

## Expected Outcomes

### Scenario 1: ResUNet Outperforms (Most Likely)

**If ResUNet shows +3-5% improvement and faster convergence:**

**Interpretation:**
- Gradient flow was limiting U-Net
- Network can learn better features with residual connections
- Convergence speed matters (fewer epochs needed)

**Recommendation:** **Use ResUNet for production**
- Better performance-computation trade-off
- Simpler than Attention ResUNet
- Maintains U-Net's simplicity while improving performance

---

### Scenario 2: Attention ResUNet Best (If Boundaries Critical)

**If Attention ResUNet shows +5-7% improvement:**

**Interpretation:**
- Overlapping beads are major challenge
- Boundary precision is bottleneck
- Attention helps focus on relevant regions

**Recommendation:** **Use Attention ResUNet if performance is critical**
- Best segmentation quality
- Worth computational overhead for production
- Use for final model training

---

### Scenario 3: Minimal Difference (<2% improvement)

**If all architectures perform similarly:**

**Interpretation:**
- U-Net capacity is sufficient
- Architecture not the bottleneck
- Data quality/quantity may be limiting factor

**Recommendation:** **Stick with U-Net**
- Simplest and fastest
- Focus on data augmentation instead
- Consider ensemble methods

---

## How to Run the Comparison Study

### Step 1: Run Architecture Comparison

```bash
# Activate conda environment
conda activate unetCNN

# Run comparison (takes ~2-3 hours for 3 architectures × 5 folds)
python validate_architecture_comparison.py
```

**What This Does:**
1. Tests all 3 architectures
2. 5-fold cross-validation for each (15 models total)
3. Same hyperparameters (filters=64, dropout=0.3)
4. Tracks performance, training time, convergence

**Expected Runtime:**
- **Per fold:** ~10-15 minutes
- **Per architecture:** ~50-75 minutes (5 folds)
- **Total:** ~2.5-3.5 hours (all 3 architectures)

**Output Directory:** `validation_arch_comparison_YYYYMMDD_HHMMSS/`
```
validation_arch_comparison_20251013_120000/
├── architecture_comparison_summary.json
├── unet/
│   ├── fold_1/
│   │   ├── history.csv
│   │   ├── results.json
│   │   └── best_model.keras
│   ├── fold_2/...
│   └── fold_5/...
├── resunet/
│   └── fold_1/...fold_5/...
└── attention_resunet/
    └── fold_1/...fold_5/...
```

---

### Step 2: Analyze Results

```bash
# Run analysis script
python analyze_architecture_comparison.py validation_arch_comparison_20251013_120000/
```

**What This Does:**
1. Loads all results
2. Performs statistical significance testing (paired t-tests)
3. Generates visualizations:
   - Performance comparison plots
   - Training curves
   - Convergence analysis
4. Creates comprehensive report

**Generated Files:**
```
validation_arch_comparison_20251013_120000/
├── ARCHITECTURE_COMPARISON_REPORT.md       # Comprehensive report
├── architecture_performance_comparison.png # Performance plots
├── architecture_training_curves.png        # Learning curves
└── architecture_convergence_analysis.png   # Convergence analysis
```

---

### Step 3: Interpret Results

#### 1. Read the Report

Open `ARCHITECTURE_COMPARISON_REPORT.md`:
- Executive summary with best architecture
- Statistical significance testing
- Fold-by-fold breakdown
- Recommendations

#### 2. Check Statistical Significance

**Look for p-values < 0.05:**
```
U-Net vs ResUNet: p = 0.032 *
→ SIGNIFICANT improvement (p < 0.05)

U-Net vs Attention ResUNet: p = 0.089
→ NOT significant (p ≥ 0.05)
```

**Interpretation Guide:**
- **p < 0.05**: Improvement is real, not due to chance
- **p ≥ 0.05**: Improvement could be random variation
- With only 5 folds, you need ≥3% improvement for significance

#### 3. Evaluate Trade-offs

**Performance vs Computation:**

| Scenario | Performance Gain | Time Overhead | Recommendation |
|----------|------------------|---------------|----------------|
| +5%, p<0.05 | Large | +15% | ✅ Switch architecture |
| +3%, p<0.05 | Medium | +10% | ⚠️  Evaluate use case |
| +2%, p>0.05 | Small | +20% | ❌ Stay with U-Net |

---

## Advanced Options

### Faster Testing (3-Fold CV)

If you want faster results, edit `validate_architecture_comparison.py`:

```python
BASE_CONFIG = {
    'batch_size': 4,
    'dropout': 0.3,
    'loss_function': 'combined',
    'filters': 64,
    'n_folds': 3,  # Change from 5 to 3
}
```

**Trade-off:**
- **Time savings:** ~40% faster
- **Cost:** Less reliable statistics (harder to detect significance)

---

### Testing Specific Architectures Only

Edit `ARCHITECTURES` list in `validate_architecture_comparison.py`:

```python
# Test only ResUNet and Attention ResUNet (skip U-Net baseline)
ARCHITECTURES = ['resunet', 'attention_resunet']
```

---

### Different Filter Sizes

Test smaller/larger models:

```python
BASE_CONFIG = {
    ...
    'filters': 32,  # Smaller (8M params) - faster, may lose performance
    # 'filters': 64,  # Baseline (31M params)
    # 'filters': 128,  # Larger (124M params) - slower, may overfit
}
```

---

## HPC Execution (Optional)

For faster execution on HPC cluster, create `pbs_architecture_comparison.sh`:

```bash
#!/bin/bash
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -N arch_comparison
#PBS -l select=1:ncpus=36:ngpus=1:mem=240gb

cd /home/svu/phyzxi/scratch/unet-HPC

export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_VISIBLE_DEVICES=0

module load singularity

image=/app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif

singularity exec --nv $image bash <<EOF
source /opt/conda/etc/profile.d/conda.sh
conda activate unetCNN
python validate_architecture_comparison.py
EOF
```

Submit: `qsub pbs_architecture_comparison.sh`

---

## Understanding the Visualizations

### Figure 1: Performance Comparison

**Top Left - Fold-by-Fold Bars:**
- Shows consistency across folds
- Look for: Which architecture is consistently higher?

**Top Right - Box Plots:**
- Shows distribution and variance
- Red diamonds = mean performance
- Look for: Tighter boxes = more stable

**Bottom Left - Parameters vs Performance:**
- Shows efficiency (accuracy per parameter)
- Look for: Higher points = better, left points = more efficient

**Bottom Right - Training Time:**
- Shows computational cost
- Look for: Balance between performance gain and time cost

---

### Figure 2: Training Curves

**Shows learning dynamics for each architecture:**
- Light lines = training Jaccard (should increase)
- Dark lines = validation Jaccard (actual performance)
- Gold stars = best validation epoch

**Look for:**
- **Fast rise:** Good convergence
- **Stable plateau:** Not overfitting
- **Early peaks:** Fast learners (ResUNet)
- **Late peaks:** Slow but steady (may reach higher)

---

### Figure 3: Convergence Analysis

**Top Left - Average Curves:**
- Mean validation Jaccard over epochs
- Look for: Which reaches high performance fastest?

**Top Right - Epochs to 90%:**
- How many epochs to reach 90% of best performance
- Look for: Lower = faster convergence

**Bottom Left - Overfitting Progression:**
- Train/val gap over time
- Look for: Lower = better generalization

**Bottom Right - Best Epoch Distribution:**
- When each architecture peaks
- Look for: Later peaks = more learning capacity

---

## Next Steps After Comparison

### If ResUNet or Attention ResUNet Wins:

1. **Re-run with Best Architecture:**
```python
# In your main training script
CONFIG = {
    'architecture': 'resunet',  # or 'attention_resunet'
    'filters': 64,
    'dropout': 0.3,
    ...
}
```

2. **Hyperparameter Optimization:**
- Test different `filters` (32, 64, 128)
- Test different `dropout` (0.2, 0.3, 0.4)
- Test different batch sizes

3. **Ensemble Methods:**
- Train multiple ResUNet models with different seeds
- Average predictions for robustness

---

## Troubleshooting

### Issue: All architectures perform similarly (~60-61%)

**Possible causes:**
- Architecture not the bottleneck
- Data quality limiting
- Already near optimal for task

**Solutions:**
- Focus on data augmentation
- Try ensemble methods
- Collect more training data

---

### Issue: Training time too long

**Solutions:**
1. **Reduce folds:** 3-fold instead of 5-fold
2. **Reduce epochs:** EarlyStopping should handle this
3. **Test fewer architectures:** Skip one
4. **Use HPC:** Submit to cluster with GPU

---

### Issue: NaN losses detected

**This shouldn't happen (FP32 training), but if it does:**
1. Check for corrupted images in dataset
2. Reduce learning rate
3. Increase smoothing constants in loss

---

## Expected Timeline

| Step | Duration | Cumulative |
|------|----------|------------|
| Setup and data loading | 5 min | 5 min |
| U-Net training (5 folds) | 50-60 min | ~1 hour |
| ResUNet training (5 folds) | 55-70 min | ~2 hours |
| Attention ResUNet (5 folds) | 60-75 min | ~3 hours |
| Analysis and visualization | 5 min | 3+ hours |

**Plan accordingly:**
- Run overnight or during working hours
- Use HPC for parallel execution
- Start with 3-fold for faster initial results

---

## Key Takeaways

1. **Baseline is Strong:** Your U-Net already achieves 60.97% ± 11.5%
2. **Expect Modest Gains:** Realistic improvement is +2-5%, not +50%
3. **Statistical Significance Matters:** With 5 folds, need ~3% gain for p<0.05
4. **Trade-offs Exist:** Performance vs computation, complexity vs simplicity
5. **Context Matters:** Best architecture depends on your constraints

---

## Questions?

**Common questions:**

**Q: Should I always use the most complex architecture?**
A: No. Use simplest architecture that meets performance requirements.

**Q: How much improvement is worth +20% training time?**
A: Depends on use case. If inference is batch processing, worth it. If real-time, maybe not.

**Q: Can I combine architectures (ensemble)?**
A: Yes! Train multiple models and average predictions. Often +1-2% improvement.

**Q: What if none improve significantly?**
A: U-Net is already good. Focus on data quality, augmentation, or ensemble methods.

---

*Guide created for architecture comparison study. For questions, refer to model_architectures.py for implementation details.*
