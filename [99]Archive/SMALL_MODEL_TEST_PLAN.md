# Small Model Test - Priority 1

## Executive Summary

After testing both `combined` and `focal_tversky` loss functions, we discovered that **loss function is NOT the bottleneck**. Both loss functions failed in the exact same way:
- Best validation at epoch 1
- Immediate collapse after first epoch
- Extreme overfitting (10-15× gap)

This pattern indicates the real problem: **Model too complex for dataset size**.

## Root Cause Analysis

### The Problem
```
Current model:  31M parameters for 83 training images
Ratio:          378,117 params/image

ResNet50 baseline: 25M parameters for 1.2M images
Ratio:             21 params/image

Our ratio is 18,000× WORSE than ResNet50 on ImageNet!
```

### Evidence from Previous Tests

| Test           | Loss Function    | Best Val Jaccard | Overfitting Gap | Best Epoch |
|----------------|------------------|------------------|-----------------|------------|
| Phase 1        | Combined         | 13.8%            | 10.5×           | 1          |
| Focal Tversky  | Focal Tversky    | 13.3%            | 15.1×           | 1          |

**Key observation:** Both tests peaked at epoch 1 and never improved, indicating validation set is too small or not representative.

## Solution: Test Smaller Model

### Hypothesis
Reducing model complexity from 31M to ~2M parameters will:
1. Reduce overfitting
2. Allow model to generalize better
3. Improve validation performance beyond epoch 1

### Implementation

#### Model Changes
- **Filters:** 64 → 16 (reduces params by ~93%)
- **Dropout:** 0.3 → 0.5 (stronger regularization)
- **Loss:** Combined (same as Phase 1 for fair comparison)

#### Expected Parameter Count
```
filters=64: 31,424,193 params (Phase 1 baseline)
filters=16:  1,964,033 params (This test)
Reduction:  93% fewer parameters
```

#### Expected Outcomes
- **Best Val Jaccard:** 20-30% (vs Phase 1's 13.8%)
- **Overfitting Gap:** 3-5× (vs Phase 1's 10.5×)
- **Best Epoch:** > 1 (improvement beyond first epoch)
- **Params/image:** ~23,680 (vs Phase 1's 378,117)

## Files Created

### 1. `validate_small_model.py`
- Main validation script with filters=16
- Enhanced monitoring and comparison with Phase 1
- Success criteria based on improved metrics

### 2. `pbs_test_small_model.sh`
- HPC job submission script
- Estimated runtime: 2 hours
- FP32 precision (same as Phase 1)

### 3. `model_architectures.py` (Updated)
- Added `filters` parameter to all model functions:
  - `UNet()`
  - `ResUNet()`
  - `AttentionResUNet()`
  - `get_model()` wrapper
- Backward compatible: defaults to filters=64

## Success Criteria

The test will pass if it meets **at least 4 out of 5 criteria**:

1. ✅ **No NaN/Inf detected** (sanity check)
2. ✅ **Best Val Jaccard ≥ 15%** (better than Phase 1's 13.8%)
3. ✅ **Overfitting Gap ≤ 7×** (better than Phase 1's 10.5×)
4. ✅ **Degradation ≤ 50%** (better than Phase 1's 78%)
5. ✅ **Best Epoch ≥ 2** (improvement beyond first epoch)

## Next Steps After This Test

### If Test Passes ✅
**Priority 2:** Implement 5-fold cross-validation
- More robust performance estimates
- 20 validation images per fold (vs 15 currently)
- Average metrics across folds

**Priority 3:** Fine-tune small model hyperparameters
- Test filters=24, 32 for optimal capacity
- Test dropout=0.4, 0.6
- Test different learning rates

### If Test Fails ❌
**If validation still collapses:**
- Validation set may not be representative
- Need 5-fold cross-validation immediately
- Analyze train/val distribution mismatch

**If overfitting persists (gap > 7×):**
- Increase dropout further (0.6-0.7)
- Add more data augmentation
- Consider even smaller model (filters=8)

## How to Run

### On HPC
```bash
# 1. Copy files to HPC
scp validate_small_model.py pbs_test_small_model.sh phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/

# 2. Submit job
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_test_small_model.sh

# 3. Monitor job
qstat -u phyzxi

# 4. View results
ls -lhtr validation_small_model_*/
cat validation_small_model_*/test_summary.json
```

### Locally (for testing)
```bash
cd /Users/xiaodan/unetCNN/unet-HPC
conda activate unetCNN
python validate_small_model.py
```

## Expected Output

The script will generate:
```
validation_small_model_YYYYMMDD_HHMMSS/
├── training_history.csv          # Epoch-by-epoch metrics
├── test_summary.json              # Summary and comparison
├── best_model.keras               # Best model checkpoint
└── SmallModelTest.oXXXXXX        # PBS output log (HPC only)
```

### Key Metrics to Watch

**Training Progress:**
```
Epoch 1: train_j=0.20, val_j=0.15, gap=1.3×  ← Good start
Epoch 5: train_j=0.30, val_j=0.22, gap=1.4×  ← Improving!
Epoch 10: train_j=0.35, val_j=0.25, gap=1.4× ← Stable
```

**Final Comparison:**
```
Metric                  | Phase 1   | This Test | Change
------------------------|-----------|-----------|--------
Best Val Jaccard        | 13.8%     | 25.0%     | +81%   ✅
Overfitting Gap         | 10.5×     | 4.2×      | -60%   ✅
Best Epoch              | 1         | 7         | +6     ✅
```

## Technical Details

### Model Architecture Changes

#### Phase 1 (filters=64):
```
Encoder: 64 → 128 → 256 → 512 → 1024
Decoder: 512 → 256 → 128 → 64
Total: 31,424,193 parameters
```

#### This Test (filters=16):
```
Encoder: 16 → 32 → 64 → 128 → 256
Decoder: 128 → 64 → 32 → 16
Total: ~1,964,033 parameters
```

### Configuration Comparison

| Parameter        | Phase 1    | Focal Tversky | This Test  | Change      |
|------------------|------------|---------------|------------|-------------|
| Architecture     | UNet       | UNet          | UNet       | Same        |
| Filters          | 64         | 64            | **16**     | **÷4**      |
| Dropout          | 0.3        | 0.3           | **0.5**    | **+67%**    |
| Loss             | Combined   | Focal Tversky | Combined   | Same as P1  |
| Batch Size       | 4          | 4             | 4          | Same        |
| Learning Rate    | 5e-5       | 5e-5          | 5e-5       | Same        |
| Precision        | FP32       | FP32          | FP32       | Same        |
| Total Params     | 31.4M      | 31.4M         | **2.0M**   | **-93%**    |
| Params/Image     | 378,117    | 378,117       | **23,680** | **-94%**    |

## Why This Should Work

### 1. Parameter Efficiency
```
Phase 1:     378,117 params/image  ← Way too many!
This test:    23,680 params/image  ← Much better
Target:      ~10,000 params/image  ← Ideal range
```

### 2. Regularization Strength
- Dropout increased from 0.3 to 0.5
- Fewer parameters = less capacity to memorize
- Model forced to learn generalizable features

### 3. Fair Comparison
- Same loss function as Phase 1 (combined)
- Same data split (random_state=42)
- Same training settings
- Only difference: model size and dropout

## Validation Strategy

### Current Issues with Validation Set
1. **Too small:** 15 images (statistically insignificant)
2. **Fixed split:** One unlucky split can doom all experiments
3. **Not representative:** Best performance at epoch 1 suggests bias

### Why Smaller Model Helps
Even with flawed validation:
- Reduced overfitting should improve validation metrics
- If it doesn't, confirms validation set is the real problem
- Either way, we learn something valuable!

### Ultimate Solution: Cross-Validation
After this test, implement 5-fold CV:
- Each fold has 20 validation images (vs 15 currently)
- 5 different validation sets
- Average performance more robust
- Standard practice for small datasets

## References

### Previous Analysis Documents
- `CRITICAL_TRAINING_FAILURE_ANALYSIS.md` - FP16/NaN root cause
- `PHASE1_RESULTS_ANALYSIS.md` - Phase 1 detailed analysis
- `PHASE1_SUMMARY_VISUAL.md` - Visual summary of Phase 1
- `FOCAL_TVERSKY_TEST_RESULTS.md` - Focal tversky analysis

### Key Findings from Previous Tests
1. **FP16 causes NaN** → Solved with FP32 ✅
2. **Loss function not the problem** → Both combined and focal_tversky fail the same way ✅
3. **Model too complex** → Testing smaller model now 🔄
4. **Validation set too small** → Need cross-validation (next priority)

---

**Created:** 2025-10-13
**Author:** Claude Code
**Priority:** 1 (Highest)
**Estimated Runtime:** 2 hours
**Expected Impact:** High (should see 2-3× improvement in validation metrics)
