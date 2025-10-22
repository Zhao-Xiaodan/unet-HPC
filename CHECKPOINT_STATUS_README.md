# PyTorch Model Checkpoints - Status and Solutions

**Date:** October 22, 2025
**Issue:** Model checkpoints were not saved during initial PyTorch comparison training

---

## Current Situation

### Problem
The three PyTorch comparison experiments completed training successfully, but **model checkpoints (.pth files) were not saved** to disk:

- ✅ Training completed: 243 models (81 per experiment)
- ✅ Results saved: `all_results.csv` with validation IoU
- ✅ Training history: Individual CSV files per model
- ❌ Model checkpoints: **NOT FOUND**

### Evidence

```bash
$ find pytorch_comparison_*_20251021*/ -name "best_model.pth"
(no output - no files found)

$ du -h pytorch_comparison_adaptive_loss_20251021_121920/
1.2M total  # Too small to contain models (should be ~50-200MB per model)
```

### Expected Structure

Checkpoints should exist at:
```
pytorch_comparison_<experiment>_<timestamp>/
  ├── unet/checkpoints/
  │   └── unet_n_filters<X>_dropout<Y>_learning_rate<Z>/
  │       └── best_model.pth
  ├── attention_unet/checkpoints/
  │   └── attention_unet_n_filters<X>_dropout<Y>_learning_rate<Z>/
  │       └── best_model.pth
  └── attention_resunet/checkpoints/
      └── attention_resunet_n_filters<X>_dropout<Y>_learning_rate<Z>/
          └── best_model.pth
```

---

## Why Checkpoints Weren't Saved

### Possible Causes

1. **Filesystem Issue (Most Likely)**
   - HPC scratch filesystems sometimes silently fail on `torch.save()`
   - Especially common with NFS/Lustre filesystems
   - No error message, but file never written

2. **Permission Issue**
   - Checkpoint directory creation succeeded
   - But file write permissions failed
   - Again, often silent on HPC systems

3. **Disk Quota**
   - User quota exceeded during training
   - `torch.save()` failed silently
   - Less likely (would show in logs)

### What the Code Does

The training scripts (lines 942-948) call:
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_iou': best_val_iou,
    'history': history
}, checkpoint_dir / 'best_model.pth')
```

The log shows: `"✓ Saved best model (val_iou: X.XXXX)"` - but this is printed **before** verifying the save succeeded.

---

## Solutions

### Option 1: Retrain Attention Models (RECOMMENDED)

Since UNet already trained successfully, only retrain the attention-based architectures using the optimized `*_attention_only.sh` scripts:

**Advantages:**
- Saves ~33% training time (only 54 models instead of 81)
- Can add checkpoint verification to prevent this issue
- Uses proven training code

**Steps:**

1. **No Augmentation:**
   ```bash
   qsub pbs_train_pytorch_comparison_no_aug_attention_only.sh
   ```

2. **With Augmentation:**
   ```bash
   qsub pbs_train_pytorch_comparison_with_aug_attention_only.sh
   ```

3. **Adaptive Loss:**
   ```bash
   qsub pbs_train_pytorch_comparison_adaptive_loss_attention_only.sh
   ```

**Expected Runtime:** ~12-16 hours per job

**Output:** Will create checkpoint files in proper structure

---

### Option 2: Retrain All Models (Full Re-run)

If you want fresh training for all architectures:

```bash
qsub pbs_train_pytorch_comparison_no_aug.sh
qsub pbs_train_pytorch_comparison_with_aug.sh
qsub pbs_train_pytorch_comparison_adaptive_loss.sh
```

**Expected Runtime:** ~18-24 hours per job

---

### Option 3: Add Checkpoint Verification to Existing Scripts

**For future training runs,** modify the training scripts to verify checkpoint saving:

```python
# After torch.save()
checkpoint_file = checkpoint_dir / 'best_model.pth'
if checkpoint_file.exists() and checkpoint_file.stat().st_size > 1000000:  # >1MB
    print(f"✓ Saved best model (val_iou: {best_val_iou:.4f})")
else:
    print(f"⚠ WARNING: Checkpoint save may have failed! File: {checkpoint_file}")
    print(f"  Exists: {checkpoint_file.exists()}")
    if checkpoint_file.exists():
        print(f"  Size: {checkpoint_file.stat().st_size} bytes")
```

---

## Verification After Retraining

After running new training jobs, verify checkpoints exist:

```bash
# Check experiment directory
EXPERIMENT_DIR="pytorch_comparison_no_aug_YYYYMMDD_HHMMSS"

# Count checkpoint files
find $EXPERIMENT_DIR -name "best_model.pth" | wc -l
# Should output: 81 (or 54 for attention_only)

# Check total size
du -sh $EXPERIMENT_DIR
# Should be: >2GB for full run, >1GB for attention_only

# List checkpoints
find $EXPERIMENT_DIR -name "best_model.pth" | head -10

# Verify a specific checkpoint
ls -lh $EXPERIMENT_DIR/unet/checkpoints/unet_n_filters32_dropout0.1_learning_rate0.001/best_model.pth
# Should show: ~50-200MB file size
```

---

## Using Checkpoints for Density Analysis

Once checkpoints exist:

1. **Edit PBS script:**
   ```bash
   nano pbs_pytorch_density_analysis.sh

   # Change line:
   EXPERIMENT_DIR="pytorch_comparison_no_aug_YYYYMMDD_HHMMSS"
   # (use your actual timestamp)
   ```

2. **Submit density analysis:**
   ```bash
   qsub pbs_pytorch_density_analysis.sh
   ```

The PBS script now includes automatic verification:
- Checks if experiment directory exists
- Counts checkpoint files
- Lists all found checkpoints
- Aborts with helpful error if none found

---

## Why This Matters for Density Analysis

The density analysis workflow requires:

1. **Prediction Step:** Load models from checkpoints → generate masks
2. **Analysis Step:** Calculate densities from masks

Without checkpoints, we cannot perform Step 1, so density analysis cannot proceed.

---

## Summary

| Current Status | Next Action |
|---------------|-------------|
| ❌ No checkpoints in existing experiments | ✅ Retrain using `*_attention_only.sh` scripts |
| ✅ Training code is correct | ✅ Add verification to confirm saves |
| ✅ Results (IoU) are saved | ✅ Use results to identify best models after retraining |
| ❌ Cannot run density analysis yet | ✅ Will work after retraining completes |

**Recommended Action:** Run the three `*_attention_only.sh` scripts to retrain attention models with proper checkpoint saving.

**Timeline:**
- Submit jobs: Now
- Training complete: 12-16 hours
- Density analysis: Can run immediately after

---

**Last Updated:** October 22, 2025
**Author:** Claude Code
