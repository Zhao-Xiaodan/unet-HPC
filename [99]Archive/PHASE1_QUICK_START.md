# Phase 1 Quick Start Guide

## Files Created (3 files)

### 1. **loss_functions_fixed.py**
Fixed loss functions with numerical stability (smooth=1e-3, safe clipping)

### 2. **validate_training_fixes.py**
Training script testing all 5 solutions (FP32, NaN detection, gradient clipping, etc.)

### 3. **pbs_validate_fixes.sh**
PBS submission script for HPC (2-hour job, single configuration test)

---

## Quick Start (5 Steps)

### Step 1: Upload to HPC
```bash
scp loss_functions_fixed.py validate_training_fixes.py pbs_validate_fixes.sh \
    phyzxi@hpc:/home/svu/phyzxi/scratch/unet-HPC/
```

### Step 2: SSH and Navigate
```bash
ssh phyzxi@hpc
cd /home/svu/phyzxi/scratch/unet-HPC
```

### Step 3: Test Loss Functions (Optional but Recommended)
```bash
module load singularity
singularity exec /app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif \
    python3 loss_functions_fixed.py
```
**Expected:** All loss functions return finite values (no NaN, no inf)

### Step 4: Submit Job
```bash
chmod +x pbs_validate_fixes.sh
qsub pbs_validate_fixes.sh
```
**Note the job ID** (e.g., 285500)

### Step 5: Monitor Progress
```bash
# Check status:
qstat -u phyzxi

# Watch log:
tail -f Validate_Training_Fixes.o285500

# Look for:
# - "Epoch X/20" with finite loss values
# - "✓ VALIDATION PASSED" at the end
```

---

## What to Expect

### ✅ SUCCESS (All fixes work):
```
Epoch 1/20: loss=0.5234, val_loss=0.6012, val_jaccard=0.1534
Epoch 2/20: loss=0.4891, val_loss=0.5678, val_jaccard=0.1789
...
Epoch 20/20: loss=0.3234, val_loss=0.4123, val_jaccard=0.3156 ✓ NEW BEST

================================================================================
✓ VALIDATION PASSED (4/4 criteria met)
================================================================================
The fixes WORKED! Training is numerically stable.
Next step: Run full hyperparameter search with these fixes.
```

**Time:** 30-60 minutes

---

### ❌ FAILURE (Fixes didn't work):
```
Epoch 1/20: loss=nan, val_loss=nan, val_jaccard=0.138

❌ BATCH 5: INVALID LOSS DETECTED
Loss value: nan

Terminating training to prevent weight corruption.
```

**Action:** Further investigation needed (unlikely if following instructions)

---

## Success Criteria

| Check | Target |
|-------|--------|
| No NaN/inf in loss | ✅ Required |
| Loss decreases | ✅ Required |
| Jaccard increases | ✅ Required |
| Jaccard > 0.25 | ⚠️ Nice-to-have |

**Minimum:** 3/4 criteria
**Ideal:** 4/4 criteria

---

## After Validation Passes

1. ✅ Fixes are proven to work
2. ✅ Training is numerically stable with FP32
3. → **Proceed to Phase 2**: Full hyperparameter search (30 configs, 12-24 hours)

Expected Phase 2 results:
- Best Jaccard: **0.50-0.70** (vs current broken 0.31)
- No NaN in any configuration
- Predictions match CLAHE+OTSU reference (50-70% density)

---

## Troubleshooting

**Job fails immediately?**
→ Check dataset exists: `ls dataset_shrunk_masks/images/ | wc -l` (should be 98)

**Out of memory?**
→ Edit `validate_training_fixes.py` line 47: `'batch_size': 2` (reduce from 4)

**NaN still appears?**
→ Verify FP32 active: `grep "Mixed precision" Validate_Training_Fixes.o*`

---

## Full Documentation

See `PHASE1_VALIDATION_README.md` for:
- Detailed explanation of each solution
- Complete expected output
- Comprehensive troubleshooting
- Next steps guide

---

## Summary

**What:** Test that FP32 fixes the NaN issue
**Time:** 1 hour
**Config:** U-Net, BS=4, Combined loss, 20 epochs
**Goal:** No NaN, smooth convergence
**If success:** → Phase 2 (full search)
**If failure:** → Investigate (unlikely)

---

**Ready to start?** → Run Step 1 above! 🚀
