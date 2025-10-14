# Phase 1 Validation Checklist

Use this checklist to ensure everything is ready before running validation.

---

## Pre-Upload Checklist (Local Machine)

### Files to Upload
- [ ] `loss_functions_fixed.py` exists
- [ ] `validate_training_fixes.py` exists
- [ ] `pbs_validate_fixes.sh` exists

**Verify:**
```bash
cd /Users/xiaodan/unetCNN/unet-HPC
ls -lh loss_functions_fixed.py
ls -lh validate_training_fixes.py
ls -lh pbs_validate_fixes.sh
```

---

## Upload to HPC

- [ ] Files uploaded successfully

**Command:**
```bash
scp loss_functions_fixed.py validate_training_fixes.py pbs_validate_fixes.sh \
    phyzxi@hpc:/home/svu/phyzxi/scratch/unet-HPC/
```

---

## HPC Pre-Flight Checklist

### 1. File Verification
- [ ] SSH to HPC successful
- [ ] In correct directory: `/home/svu/phyzxi/scratch/unet-HPC`
- [ ] All 3 new files exist
- [ ] Existing files present: `model_architectures.py`

**Commands:**
```bash
ssh phyzxi@hpc
cd /home/svu/phyzxi/scratch/unet-HPC
pwd  # Should show: /home/svu/phyzxi/scratch/unet-HPC

ls -lh loss_functions_fixed.py
ls -lh validate_training_fixes.py
ls -lh pbs_validate_fixes.sh
ls -lh model_architectures.py
```

---

### 2. Dataset Verification
- [ ] Dataset directory exists
- [ ] 98 images found
- [ ] 98 masks found

**Commands:**
```bash
ls -ld dataset_shrunk_masks/
ls dataset_shrunk_masks/images/ | wc -l   # Should output: 98
ls dataset_shrunk_masks/masks/ | wc -l    # Should output: 98
```

---

### 3. Loss Function Test (Recommended)
- [ ] Singularity module loads
- [ ] TensorFlow container accessible
- [ ] Loss functions return finite values

**Commands:**
```bash
module load singularity

# Test loss functions:
singularity exec /app1/common/singularity-img/hopper/tensorflow/tensorflow_2.16.1-cuda_12.5.0_24.06.sif \
    python3 loss_functions_fixed.py
```

**Expected output snippet:**
```
focal                    : 0.060583  ✓ FINITE
tversky                  : 0.149925  ✓ FINITE
focal_tversky            : 0.183608  ✓ FINITE
combined                 : 0.122573  ✓ FINITE
combined_tversky         : 0.114150  ✓ FINITE
```

---

### 4. PBS Script Preparation
- [ ] PBS script has execute permission

**Command:**
```bash
chmod +x pbs_validate_fixes.sh
```

---

## Job Submission Checklist

### Before Submission
- [ ] All above checks passed
- [ ] No pending jobs that might conflict
- [ ] GPU node available

**Commands:**
```bash
# Check your current jobs:
qstat -u phyzxi

# Check GPU availability:
pbsnodes | grep -A 5 "gpu"
```

---

### Submit Job
- [ ] Job submitted successfully
- [ ] Job ID noted

**Command:**
```bash
qsub pbs_validate_fixes.sh
```

**Example output:**
```
285500.venus13
```
→ Note this job ID!

---

## Monitoring Checklist

### Initial Check (First 5 minutes)
- [ ] Job is running (not failed immediately)
- [ ] Log file created
- [ ] Pre-flight checks passed in log

**Commands:**
```bash
# Check status:
qstat -u phyzxi
# Should show: R (Running)

# Check log exists:
ls -lh Validate_Training_Fixes.o285500

# Check pre-flight:
head -100 Validate_Training_Fixes.o285500
```

**Look for:**
```
✓ Dataset directory found: dataset_shrunk_masks
✓ validate_training_fixes.py
✓ loss_functions_fixed.py
✓ Loss function test PASSED
```

---

### Training Progress (Every 10 minutes)
- [ ] Training started
- [ ] Epoch counter incrementing
- [ ] Loss values are finite (no NaN, no inf)
- [ ] Jaccard increasing

**Command:**
```bash
tail -50 Validate_Training_Fixes.o285500
```

**Look for:**
```
Epoch 1/20: loss=0.5234, val_loss=0.6012, val_jaccard=0.1534
Epoch 2/20: loss=0.4891, val_loss=0.5678, val_jaccard=0.1789
...
```

**Red flags:**
- `loss=nan` → Fixes didn't work
- Job stops at epoch 1 → NaN detected (good! callback working)
- `ResourceExhaustedError` → Out of memory (reduce batch size)

---

### Completion Check (After 30-60 minutes)
- [ ] Job completed (not in qstat anymore)
- [ ] Exit code is 0
- [ ] Validation result present

**Commands:**
```bash
# Check if job finished:
qstat -u phyzxi
# Should show nothing (job completed)

# Check exit code:
tail -5 Validate_Training_Fixes.o285500
# Should show: "Status: ✓ Completed successfully"

# Check validation result:
grep "VALIDATION" Validate_Training_Fixes.o285500
```

---

## Success Verification Checklist

### Output Files
- [ ] Output directory created
- [ ] `model_best.hdf5` exists (~350 MB)
- [ ] `training_history.csv` exists
- [ ] `validation_summary.json` exists

**Commands:**
```bash
# Find output directory:
ls -ld validation_fixes_*/

# Check files:
ls -lh validation_fixes_*/
```

---

### Validation Summary
- [ ] No NaN detected
- [ ] Criteria met ≥ 3/4
- [ ] Validation passed = true

**Command:**
```bash
cat validation_fixes_*/validation_summary.json
```

**Check:**
```json
{
  "nan_detected": false,        ← Should be false
  "criteria_met": 4,            ← Should be 3 or 4
  "validation_passed": true,    ← Should be true
  ...
}
```

---

### Training History
- [ ] No NaN in loss column
- [ ] Loss decreased from first to last epoch
- [ ] Jaccard increased from first to last epoch

**Commands:**
```bash
# Check for NaN:
grep -i "nan\|inf" validation_fixes_*/training_history.csv
# Should return nothing!

# View progression:
head -6 validation_fixes_*/training_history.csv
tail -5 validation_fixes_*/training_history.csv
```

---

## Final Decision Checklist

### ✅ VALIDATION PASSED
- [ ] All success verification checks passed
- [ ] No NaN detected anywhere
- [ ] Loss decreased smoothly
- [ ] Jaccard > 0.25

**Action:**
→ **Proceed to Phase 2** (full hyperparameter search)

---

### ⚠️ VALIDATION PARTIAL
- [ ] No NaN detected
- [ ] Loss decreased
- [ ] Jaccard increased (but < 0.25)

**Action:**
→ **Proceed to Phase 2** (stability is main goal)
→ Consider adjustments: more epochs, different loss

---

### ❌ VALIDATION FAILED
- [ ] NaN detected in training
- [ ] Job crashed
- [ ] Loss didn't decrease

**Action:**
→ **DO NOT proceed to Phase 2**
→ Investigate issue:
  1. Check TensorFlow version
  2. Verify FP32 active
  3. Try simpler config (dice loss, BS=2)
  4. Review full error log

---

## Phase 2 Preparation (If Validation Passed)

- [ ] Understand which fixes worked
- [ ] Ready to apply fixes to full search
- [ ] Prepared for 12-24 hour training run
- [ ] Have capacity to monitor occasionally

**Next files to create:**
- `hyperparam_search_fixed.py` (apply all 5 fixes to full search)
- `pbs_hyperparam_fixed.sh` (24-hour job for 30 configs)

---

## Quick Reference

### Key Commands
```bash
# Upload files:
scp {loss_functions_fixed,validate_training_fixes,pbs_validate_fixes}.{py,sh} \
    phyzxi@hpc:/home/svu/phyzxi/scratch/unet-HPC/

# Submit job:
qsub pbs_validate_fixes.sh

# Monitor:
tail -f Validate_Training_Fixes.o<JOBID>

# Check result:
cat validation_fixes_*/validation_summary.json
```

---

### Key Files
- **Upload:** 3 files (.py x2, .sh x1)
- **Existing:** model_architectures.py, dataset_shrunk_masks/
- **Output:** validation_fixes_*/ (3 files inside)
- **Log:** Validate_Training_Fixes.o<JOBID>

---

### Timeline
- **Upload:** 1 minute
- **Setup:** 5 minutes (checks, tests)
- **Training:** 30-60 minutes
- **Verification:** 5 minutes
- **Total:** ~1 hour

---

### Success Indicators
✅ No NaN/inf in any loss value
✅ Loss: 0.52 → 0.32 (decreasing)
✅ Jaccard: 0.15 → 0.31+ (increasing)
✅ Smooth training curves

---

## Status Tracking

### Current Status: [ ] Not Started

- [ ] Files uploaded to HPC
- [ ] Pre-flight checks passed
- [ ] Job submitted (Job ID: _______)
- [ ] Training started
- [ ] Training completed
- [ ] Validation passed
- [ ] Ready for Phase 2

---

**Date started:** __________
**Job ID:** __________
**Result:** [ ] Pass  [ ] Partial  [ ] Fail
**Next action:** __________

---

Print this checklist and mark items as you complete them! ✓
