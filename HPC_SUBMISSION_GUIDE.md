# HPC Submission Guide - Architecture Comparison

## Quick Start

### Step 1: Transfer Files to HPC

```bash
# On your local machine
cd /Users/xiaodan/unetCNN/unet-HPC

# Transfer necessary files to HPC
scp validate_architecture_comparison.py phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/
scp model_architectures.py phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/
scp loss_functions_fixed.py phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/
scp pbs_architecture_comparison.sh phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/
scp analyze_architecture_comparison.py phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/

# Or transfer entire directory (if first time)
rsync -avz --exclude='.git' --exclude='validation_*' --exclude='*.keras' \
  /Users/xiaodan/unetCNN/unet-HPC/ \
  phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/
```

### Step 2: Submit Job on HPC

```bash
# SSH to HPC
ssh phyzxi@nscc

# Navigate to working directory
cd /home/svu/phyzxi/scratch/unet-HPC

# Make script executable
chmod +x pbs_architecture_comparison.sh

# Submit job
qsub pbs_architecture_comparison.sh
```

### Step 3: Monitor Job

```bash
# Check job status
qstat -u phyzxi

# View job details
qstat -f <JOB_ID>

# Monitor output in real-time (once job starts)
tail -f Architecture_Comparison.o<JOB_ID>
```

### Step 4: Retrieve Results

```bash
# On your local machine, after job completes
cd /Users/xiaodan/unetCNN/unet-HPC

# Download results directory
scp -r phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/validation_arch_comparison_YYYYMMDD_HHMMSS/ ./

# Download console log
scp phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/arch_comparison_console_*.log ./
```

### Step 5: Analyze Locally

```bash
# On your local machine
conda activate unetCNN

# Run analysis
python analyze_architecture_comparison.py validation_arch_comparison_YYYYMMDD_HHMMSS/

# View report
open validation_arch_comparison_YYYYMMDD_HHMMSS/ARCHITECTURE_COMPARISON_REPORT.md
```

---

## PBS Script Configuration

### Job Resource Allocation

```bash
#PBS -l walltime=6:00:00              # 6 hours (sufficient for 15 models)
#PBS -l select=1:ncpus=36:ngpus=1    # 1 node, 36 CPUs, 1 GPU
#PBS -l mem=240gb                      # 240 GB RAM
```

**Resource Justification:**
- **Walltime:** 6 hours allows ~24 min/model, enough for 8-12 min typical runtime + buffer
- **GPU:** 1 GPU is sufficient (batch_size=4 uses ~6-8 GB VRAM)
- **CPUs:** 36 CPUs for data loading and preprocessing
- **RAM:** 240 GB ensures no memory issues with full dataset in memory

### Modifying Resources (if needed)

If job times out or you want faster execution:

```bash
# For faster execution (use more resources)
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=48:ngpus=1:mem=300gb

# For longer safety buffer
#PBS -l walltime=8:00:00
```

---

## Job Output Files

### Automatically Generated

1. **Standard Output/Error:** `Architecture_Comparison.o<JOB_ID>`
   - Combined stdout and stderr (due to `#PBS -j oe`)
   - Contains all console output
   - Check this first if job fails

2. **Console Log:** `arch_comparison_console_YYYYMMDD_HHMMSS.log`
   - Detailed training log with timestamps
   - Captured by `tee` command
   - Useful for debugging

3. **Results Directory:** `validation_arch_comparison_YYYYMMDD_HHMMSS/`
   ```
   validation_arch_comparison_20251013_143022/
   ├── architecture_comparison_summary.json
   ├── unet/
   │   ├── fold_1/
   │   │   ├── history.csv
   │   │   ├── results.json
   │   │   └── best_model.keras
   │   ├── fold_2/
   │   └── ... (fold_3, fold_4, fold_5)
   ├── resunet/
   │   └── fold_1/...fold_5/
   └── attention_resunet/
       └── fold_1/...fold_5/
   ```

---

## Monitoring Job Progress

### Check Job Status

```bash
# Basic status
qstat -u phyzxi

# Shows:
# Job ID    Name                 Status  Time
# 12345.pbs Architecture_Comparison  R    1:23:45
```

**Status Codes:**
- `Q` - Queued (waiting for resources)
- `R` - Running
- `E` - Exiting (finishing up)
- `F` - Finished
- `H` - Held (needs admin intervention)

### Real-Time Monitoring

```bash
# Watch output file grow (once job starts)
tail -f Architecture_Comparison.o<JOB_ID>

# Press Ctrl+C to stop watching
```

### Check Progress in Output

Look for these markers in the output:

```
🚀 STARTING ARCHITECTURE COMPARISON STUDY
============================================

Testing 3 architectures with 5-fold CV each

[Progress markers:]
U-NET - FOLD 1/5: TRAINING
U-NET - FOLD 2/5: TRAINING
...
RESUNET - FOLD 1/5: TRAINING
...
ATTENTION RESUNET - FOLD 1/5: TRAINING
...

✅ ARCHITECTURE COMPARISON COMPLETE
```

---

## Expected Timeline

| Stage | Duration | Cumulative | What's Happening |
|-------|----------|------------|------------------|
| **Queue Wait** | 0-30 min | 0-30 min | Waiting for GPU node |
| **Environment Check** | 2 min | ~32 min | Loading modules, checking GPU |
| **U-Net Training** | 40-50 min | ~1h 22min | 5 folds × 8-10 min/fold |
| **ResUNet Training** | 45-55 min | ~2h 17min | 5 folds × 9-11 min/fold |
| **Attention ResUNet** | 50-60 min | ~3h 17min | 5 folds × 10-12 min/fold |
| **Finalization** | 2 min | ~3h 19min | Saving summary, cleanup |

**Total:** ~3-3.5 hours (with 6-hour allocation for safety)

---

## Troubleshooting

### Job Stuck in Queue (Status: Q)

**Check queue:**
```bash
qstat -q
```

**Possible causes:**
- No GPU nodes available (peak usage time)
- Resource request too high
- Queue limits reached

**Solutions:**
- Wait (GPU queues can be 1-2 hours during peak)
- Submit during off-peak hours (evenings, weekends)
- Reduce resource request temporarily

---

### Job Failed Immediately (exits in <5 min)

**Check error log:**
```bash
cat Architecture_Comparison.o<JOB_ID> | tail -50
```

**Common causes:**

1. **Missing files**
   ```
   ✗ ERROR: validate_architecture_comparison.py not found!
   ```
   **Fix:** Transfer files again, check paths

2. **Dataset not found**
   ```
   ✗ ERROR: Dataset directories not found!
   ```
   **Fix:** Verify dataset at `/home/svu/phyzxi/scratch/unet-HPC/dataset_full_stack/`

3. **Module load failure**
   ```
   ERROR: TensorFlow container not found
   ```
   **Fix:** Check container path, update if HPC upgraded

4. **GPU not available**
   ```
   Physical GPUs found: 0
   ```
   **Fix:** Ensure `#PBS -l ngpus=1` is set, check `--nv` flag in singularity command

---

### Job Times Out (hits 6-hour limit)

**If job is still running when walltime expires:**

**Causes:**
- Very slow GPU node
- Dataset loading issues
- Network filesystem slow

**Solutions:**
1. **Increase walltime:**
   ```bash
   #PBS -l walltime=8:00:00
   ```

2. **Reduce folds (for faster test):**
   Edit `validate_architecture_comparison.py`:
   ```python
   BASE_CONFIG = {
       'n_folds': 3,  # Instead of 5
   }
   ```

3. **Resume from checkpoint:**
   Check which folds completed, manually run remaining

---

### NaN Losses Detected

**If you see:**
```
❌ BATCH 123: INVALID LOSS: nan
```

**Causes:**
- Corrupted image in dataset
- Numerical instability (rare with FP32)

**Solutions:**
1. Check console log for which fold failed
2. Inspect dataset for problematic images
3. Increase loss smoothing constants (if needed)

---

### Out of Memory (OOM)

**If you see:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
1. **Reduce batch size:**
   Edit `validate_architecture_comparison.py`:
   ```python
   BASE_CONFIG = {
       'batch_size': 2,  # Instead of 4
   }
   ```

2. **Request more GPU memory:**
   Some nodes have 32GB GPUs, request specifically:
   ```bash
   #PBS -l select=1:ncpus=36:ngpus=1:gpu_mem=32gb
   ```

---

## Advanced Options

### Run Only Specific Architectures

Edit `validate_architecture_comparison.py` before submitting:

```python
# Test only ResUNet and Attention ResUNet (skip U-Net)
ARCHITECTURES = ['resunet', 'attention_resunet']
```

**Saves time if you already have U-Net results from previous CV run.**

---

### Run with Different Hyperparameters

Edit `BASE_CONFIG` in `validate_architecture_comparison.py`:

```python
BASE_CONFIG = {
    'batch_size': 4,
    'dropout': 0.4,      # Try higher dropout
    'loss_function': 'combined',
    'filters': 128,      # Try larger model
    'n_folds': 5,
}
```

**Note:** Larger models (filters=128) may take 2× longer.

---

### Parallel Execution (Advanced)

If you need faster results, submit multiple jobs:

**Job 1:** U-Net only
```bash
# Edit script to: ARCHITECTURES = ['unet']
qsub pbs_architecture_comparison.sh
```

**Job 2:** ResUNet only
```bash
# Edit script to: ARCHITECTURES = ['resunet']
qsub pbs_architecture_comparison.sh
```

**Job 3:** Attention ResUNet only
```bash
# Edit script to: ARCHITECTURES = ['attention_resunet']
qsub pbs_architecture_comparison.sh
```

**Combine results manually after all complete.**

---

## Retrieving Results

### Full Results Directory

```bash
# On local machine
scp -r phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/validation_arch_comparison_20251013_143022/ ./
```

**Size:** ~200-300 MB (15 .keras models + histories)

---

### Summary Only (Fast)

```bash
# Just the summary JSON
scp phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/validation_arch_comparison_*/architecture_comparison_summary.json ./

# Quick check
cat architecture_comparison_summary.json | python -m json.tool | grep -A 5 "best_val_jacard"
```

---

### Models Only (for deployment)

```bash
# Download best performing architecture's models
scp -r phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/validation_arch_comparison_*/attention_resunet/ ./
```

---

## Email Notifications

The PBS script is configured to send emails:

```bash
#PBS -M phyzxi@nus.edu.sg
#PBS -m abe
```

**Notification triggers:**
- `a` - Job aborted/failed
- `b` - Job begins
- `e` - Job ends

**Emails include:**
- Job ID
- Exit status
- Resource usage summary

**To disable notifications:**
```bash
# Comment out or remove:
# #PBS -m abe
```

---

## Cost Estimation (If Applicable)

**Resource units per job:**
- 1 GPU × 6 hours = 6 GPU-hours
- 36 CPUs × 6 hours = 216 CPU-hours

Check your allocation with:
```bash
gbalance -u phyzxi
```

---

## Checklist Before Submission

- [ ] All Python scripts transferred to HPC
- [ ] Dataset in correct location (`./dataset_full_stack/`)
- [ ] PBS script has correct paths
- [ ] Email address correct in PBS script
- [ ] Script is executable (`chmod +x`)
- [ ] Sufficient allocation balance (if applicable)
- [ ] Estimated time fits within walltime
- [ ] Tested Python script locally (if possible)

---

## Post-Completion Steps

1. **Download results** (see above)
2. **Run analysis locally:**
   ```bash
   python analyze_architecture_comparison.py validation_arch_comparison_YYYYMMDD_HHMMSS/
   ```
3. **Review report:**
   ```bash
   open validation_arch_comparison_YYYYMMDD_HHMMSS/ARCHITECTURE_COMPARISON_REPORT.md
   ```
4. **Share findings** or deploy best model

---

## Contact / Support

**For HPC issues:**
- NSCC helpdesk: help@nscc.sg
- User guide: https://help.nscc.sg

**For script/code issues:**
- Check console log first
- Review Python traceback
- Consult ARCHITECTURE_COMPARISON_GUIDE.md

---

**Good luck with your architecture comparison study! 🚀**
