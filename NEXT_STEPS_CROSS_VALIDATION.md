# Next Steps: 5-Fold Cross-Validation

**Status:** Ready to deploy 🚀
**Priority:** 🔴 CRITICAL (Must run before any further architecture experiments)
**Estimated Time:** 10-12 hours

---

## What We Learned from Small Model Test

### Critical Findings

1. **❌ Small model WORSE than large model**
   - Phase 1 (31M params): 13.8% best val Jaccard
   - Small model (2M params): 7.6% best val Jaccard (-45%)
   - **Conclusion:** Model capacity is NOT the bottleneck

2. **❌ "Best at epoch 1" problem persists**
   - All 3 tests (Phase 1, Focal Tversky, Small Model) peaked at epoch 1
   - Statistically impossible if validation set was representative
   - **Conclusion:** Validation set is fundamentally flawed

3. **✅ Overfitting reduced significantly**
   - Gap: 10.5× → 4.4× (-58%)
   - Degradation: 78% → 20% (-74%)
   - **Conclusion:** Smaller models ARE more stable (but less capable)

### The Diagnosis

**Model complexity is NOT the problem. Validation set is the problem.**

All evidence points to validation set having different characteristics than training set, making performance estimates unreliable.

---

## Why Cross-Validation is Critical

### Current Situation

```
Fixed Split (random_state=42):
Train: 83 images
Val:   15 images  ← Too small, possibly biased
```

**Problems:**
- 15 images is statistically insignificant
- One unlucky split can doom all experiments
- Can't tell if model is good or validation is bad

### Cross-Validation Solution

```
5-Fold CV:
Fold 1: Train on 79, validate on 20
Fold 2: Train on 79, validate on 20
Fold 3: Train on 79, validate on 20
Fold 4: Train on 79, validate on 20
Fold 5: Train on 79, validate on 20

Average across folds = Reliable estimate
```

**Benefits:**
- 5 different validation sets (33% more data per fold)
- Average performance (less variance)
- Confidence intervals (know uncertainty)
- Detect split bias (if CV >> Phase 1 single split)

---

## Expected Outcomes

### Scenario 1: Split Bias Detected ✅

```
CV Mean: 18-22% best val Jaccard
Phase 1: 13.8% best val Jaccard

Interpretation:
→ Phase 1 validation split was unlucky (biased)
→ True performance is better than we thought
→ All previous tests underestimated model quality
→ Should use CV for all future evaluations

Next step: Optimal model size search with CV
```

### Scenario 2: No Split Bias 📊

```
CV Mean: 12-15% best val Jaccard
Phase 1: 13.8% best val Jaccard

Interpretation:
→ Phase 1 validation split was representative
→ Task performance is genuinely ~13%
→ "Best at epoch 1" is a data distribution problem
→ Need to investigate train/val differences

Next step: Train/val distribution analysis
```

### Scenario 3: Worse Than Phase 1 ⚠️

```
CV Mean: 8-11% best val Jaccard
Phase 1: 13.8% best val Jaccard

Interpretation:
→ Phase 1 validation split was optimistic
→ True performance is worse than we thought
→ Task may be very difficult with current data

Next step: Data quality analysis
```

---

## Files Created

### 1. `validate_cross_validation.py`
**Main CV script with:**
- 5-fold stratified split (by density quartiles)
- BASELINE config (filters=64, dropout=0.3) for fair comparison
- Comprehensive fold-by-fold monitoring
- Statistical analysis across folds
- Automatic diagnosis of split bias

**Key features:**
- Stratification ensures balanced density distribution
- Progress tracking per fold
- NaN detection per fold
- Early stopping per fold
- Best model saved per fold

### 2. `pbs_cross_validation.sh`
**HPC submission script with:**
- 12-hour walltime (2 hours per fold + buffer)
- Same environment as previous tests (FP32, 8GB GPU mem)
- Comprehensive pre/post-run checks
- Automatic result extraction and summary

### 3. `SMALL_MODEL_RESULTS_ANALYSIS.md`
**Comprehensive 22-page analysis documenting:**
- Why small model test definitively ruled out model complexity
- Evidence for validation set problem
- Statistical analysis of "best at epoch 1" pattern
- Detailed recommendations for next steps

---

## Configuration: BASELINE from Phase 1

**Why use baseline instead of small model?**

The small model test proved that reducing to 2M params hurts performance.
For CV, we want to use a configuration with **proven capacity**:

```python
CONFIG = {
    'filters': 64,      # 31M params (proven capacity)
    'dropout': 0.3,     # Original value
    'loss': 'combined', # Dice + Focal
    'batch_size': 4,
}
```

This provides apples-to-apples comparison with Phase 1 single split.

---

## How to Run

### Upload Files to HPC

```bash
scp validate_cross_validation.py pbs_cross_validation.sh \
    phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/
```

### Submit Job

```bash
ssh phyzxi@nscc
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_cross_validation.sh

# Monitor progress
qstat -u phyzxi
```

### Expected Timeline

```
Hour 0-2:   Fold 1 training
Hour 2-4:   Fold 2 training
Hour 4-6:   Fold 3 training
Hour 6-8:   Fold 4 training
Hour 8-10:  Fold 5 training
Hour 10:    Final analysis
```

---

## Expected Output

```
validation_cv_YYYYMMDD_HHMMSS/
├── cv_summary.json           # ← KEY FILE: Overall statistics
│   ├── Best Val Jaccard: mean ± std
│   ├── Comparison with Phase 1
│   ├── Diagnosis (split bias, epoch 1 problem, variance)
│   └── Fold-by-fold results
│
├── fold_1/
│   ├── history.csv           # Epoch-by-epoch metrics
│   ├── results.json          # Fold summary
│   └── best_model.keras      # Best model checkpoint
│
├── fold_2/ ... fold_5/       # Same structure
```

### Key Metrics to Extract

From `cv_summary.json`:

```python
{
  "statistics": {
    "best_val_jacard": {
      "mean": 0.XXX,    # ← Main metric
      "std": 0.YYY,     # ← Confidence
      "values": [...]   # ← Per-fold performance
    },
    "best_epoch": {
      "values": [...]   # ← Check for epoch 1 pattern
    }
  },
  "diagnosis": {
    "split_bias": true/false,
    "epoch_1_problem_persists": true/false,
    "coefficient_of_variation": 0.ZZZ
  }
}
```

---

## Success Criteria

**CV Test will answer 3 critical questions:**

1. **Is Phase 1 split biased?**
   - CV mean > 16% → YES (split was unlucky)
   - CV mean 12-16% → NO (split was representative)
   - CV mean < 12% → Phase 1 was optimistic

2. **Does "best at epoch 1" persist?**
   - ≥4 folds peak at epoch 1 → YES (data problem)
   - 2-3 folds peak at epoch 1 → MIXED
   - ≤1 fold peaks at epoch 1 → NO (was split bias)

3. **Is performance stable across splits?**
   - CV std < 2% → YES (stable)
   - CV std 2-4% → MODERATE
   - CV std > 4% → NO (highly variable)

---

## After Cross-Validation

### If Split Bias Detected ✅

**Next priority:** Optimal model size search with CV

Test intermediate sizes with cross-validation:
- filters=24: ~4.5M params
- filters=32: ~8M params
- filters=48: ~18M params

Find sweet spot between small model (too small) and baseline (potentially too large).

### If No Split Bias (Epoch 1 Problem Persists) ⚠️

**Next priority:** Train/val distribution analysis

Create `analyze_train_val_split.py` to investigate:
- Density distribution (mean/std per split)
- Dilution factors
- Image quality metrics
- Spatial statistics

Understand WHY validation is different from training.

### If High Variance Across Folds 📊

**Indicates:** Data heterogeneity or class imbalance

**Next steps:**
- Analyze fold composition
- Consider stratification by additional factors
- May need more data

---

## Why This is the Right Next Step

### What We Know

| Hypothesis                | Evidence                                        | Status      |
|---------------------------|-------------------------------------------------|-------------|
| FP16 causes NaN           | Phase 1: No NaN with FP32                       | ✅ SOLVED    |
| Wrong loss function       | Focal Tversky: Same pattern as combined         | ❌ REJECTED  |
| Model too complex         | Small model: WORSE performance (-45%)           | ❌ REJECTED  |
| Validation set problem    | All 3 tests: Best at epoch 1 (impossible!)      | ✅ CONFIRMED |

### What We Need to Know

1. **Is current validation estimate reliable?**
   → CV will answer this

2. **Is the "epoch 1" pattern due to split bias or data distribution?**
   → CV will answer this

3. **What is the true performance range?**
   → CV will answer this

**We cannot proceed with architecture experiments until we have reliable performance estimates.**

---

## Summary

✅ Created: `validate_cross_validation.py` (5-fold CV implementation)
✅ Created: `pbs_cross_validation.sh` (HPC submission)
✅ Created: `SMALL_MODEL_RESULTS_ANALYSIS.md` (comprehensive analysis)

🎯 **Action required:** Upload files and submit CV job
⏱️ **Time:** 10-12 hours
🔍 **Output:** Reliable performance estimate + diagnosis of split bias

**After CV results, we'll know:**
- True model performance (not affected by unlucky split)
- Whether to proceed with model optimization or data analysis
- Confidence in our evaluation methodology

---

**Ready to deploy!** 🚀
