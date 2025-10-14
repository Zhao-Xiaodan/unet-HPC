# Quickstart: Attention ResUNet Hyperparameter Search

**Purpose:** Test if attention gates can improve ResUNet's performance beyond 55% Jaccard

**Based on:** ResUNet hyperparameter search results (42.40% Jaccard achieved)

---

## TL;DR - Run the Search

### On HPC (Recommended)

```bash
# 1. Transfer files to HPC
scp hyperparameter_search_attention_resunet.py phyzxi@atlas7.nus.edu.sg:~/scratch/unet-HPC/
scp pbs_attention_resunet_search.sh phyzxi@atlas7.nus.edu.sg:~/scratch/unet-HPC/

# 2. SSH to HPC
ssh phyzxi@atlas7.nus.edu.sg

# 3. Navigate to project
cd ~/scratch/unet-HPC

# 4. Submit job
qsub pbs_attention_resunet_search.sh

# 5. Monitor job
qstat -u phyzxi          # Check job status
tail -f AttResUNet_Search.o*  # Watch live output (once job starts)
```

**Expected Runtime:** 6-10 hours (54 models, ~5-7 min each)

---

## What This Search Does

### Search Space (Focused on ResUNet Findings)

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| Learning Rate | 1.5e-05, **2e-05**, 2.5e-05 | Centered on ResUNet optimum |
| Dropout | 0.2, **0.3**, 0.4 | Lower range (higher dropout hurt ResUNet) |
| Batch Size | **4**, 8 | Small batches slightly better |

**Total:** 18 configurations × 3 folds = **54 models**

**Why This Search Space?**
- ResUNet search tested 1e-05, 2e-05 → 2e-05 was 2× better
- Fine-tune around 2e-05 with 1.5e-05 and 2.5e-05
- Higher dropout (0.5) hurt ResUNet → exclude from search
- This is a more efficient search informed by previous results

---

## Decision Criteria

After the search completes, the script will automatically assess results:

| Performance | Assessment | Recommendation |
|------------|------------|----------------|
| **>69.94%** | ✅ **SUCCESS** | Beats U-Net! Use as primary architecture |
| **>55.00%** | ✅ **PROMISING** | Run full 5-fold validation to confirm |
| **>42.40%** | ⚠️ **MARGINAL** | Small improvement; decide by cost vs accuracy |
| **≤42.40%** | ❌ **FAILURE** | Attention doesn't help; abandon residual architectures |

**Baselines:**
- **U-Net:** 69.94% ± 5.02% (goal)
- **ResUNet (baseline):** 39.95% ± 6.79%
- **ResUNet (optimized):** 42.40% ± 4.60%

---

## What Happens During the Search

### Automated Process

```
For each of 18 hyperparameter combinations:
  For each of 3 cross-validation folds:
    1. Build Attention ResUNet model with config
    2. Train up to 30 epochs (early stopping after 8 no-improvement epochs)
    3. Save best model and training history
    4. Record: best Jaccard, best epoch, overfitting gap, etc.

Aggregate results:
  - Calculate mean/std for each configuration across folds
  - Rank configurations by mean Jaccard
  - Compare best config to baselines
  - Generate summary JSON with all results
```

### Output Structure

```
attention_resunet_search_20251014_HHMMSS/
├── attention_resunet_search_summary.json    ← Overall results
├── attention_resunet_lr1.5e-05_drop0.2_bs4/
│   ├── fold_1/
│   │   ├── best_model.keras
│   │   ├── results.json
│   │   └── training_history.csv
│   ├── fold_2/
│   └── fold_3/
├── attention_resunet_lr2e-05_drop0.3_bs4/   ← Likely best config
│   ├── fold_1/
│   ├── fold_2/
│   └── fold_3/
└── ... (16 more configurations)
```

---

## Monitoring Progress

### Check Job Status

```bash
qstat -u phyzxi                    # Job status (Q=queued, R=running, C=complete)
qstat -f <job_id> | grep walltime  # Time used/remaining
```

### Watch Live Output

```bash
# Once job starts running
tail -f AttResUNet_Search.o*

# You'll see progress like:
# CONFIG 1/18: LR=1.5e-05, Dropout=0.2, Batch=4
#   Fold 1/3 - Best Val Jaccard: 0.4521 (epoch 12)
#   Fold 2/3 - Best Val Jaccard: 0.4389 (epoch 10)
#   Fold 3/3 - Best Val Jaccard: 0.4672 (epoch 14)
# PROGRESS: 3/54 models completed (5.6%)
```

### View Results While Running

```bash
# Check latest results
ls -lt attention_resunet_search_*/*/fold_*/results.json | head -5

# Quick peek at best performing configs so far
find attention_resunet_search_* -name "results.json" -exec grep -H "best_val_jacard" {} \; | sort -t: -k2 -rn | head -10
```

---

## After Completion

### 1. Check Summary

```bash
# View results summary
cat attention_resunet_search_*/attention_resunet_search_summary.json | python3 -m json.tool

# Quick assessment
grep -A 10 "best_config" attention_resunet_search_*/attention_resunet_search_summary.json
```

### 2. Interpret Results

The PBS script automatically prints an assessment at the end:

```
✅ SUCCESS! Attention ResUNet EXCEEDS U-Net!
   → Use Attention ResUNet as primary architecture

✅ PROMISING! Attention gates improve performance.
   → Run full 5-fold validation to confirm

⚠️  MARGINAL: Small improvement over ResUNet.
   → Decide based on accuracy vs computational cost

❌ FAILURE: Attention gates do not help.
   → Abandon residual architectures, stick with U-Net
```

### 3. Next Steps Based on Results

#### If Performance >55%:
```bash
# Run full 5-fold validation with best config
# (Create validation script using best hyperparameters)
```

#### If Performance ≤55%:
- **Abandon residual architectures** (ResUNet, Attention ResUNet)
- **Stick with U-Net** (69.94% Jaccard)
- **Focus on data improvements:**
  - Data augmentation
  - More training samples
  - Better preprocessing
  - Class balancing

---

## Why This Search Is Efficient

### Compared to Brute Force

**Brute force approach:**
- Test same 3×3×2 grid as ResUNet
- 18 configs × 3 folds = 54 models
- 6-10 hours runtime

**Our focused approach:**
- Learned from ResUNet: LR=2e-05 optimal, dropout=0.3 best
- Fine-tune around optimal values (±25% LR variation)
- Skip known bad configs (LR=1e-05 too slow, dropout=0.5 too high)
- Same 54 models but higher probability of finding good config

**Key Insight:** We're not searching blindly - we're using ResUNet's lessons to narrow the search space around the most promising region.

---

## Troubleshooting

### Job Fails Immediately

```bash
# Check error log
cat AttResUNet_Search.o*

# Common issues:
# 1. File not found: Ensure .py file was transferred
# 2. Module import error: Check model_architectures.py has attention_resunet
# 3. Dataset missing: Verify ./dataset_full_stack/ exists on HPC
```

### Job Runs But No Results

```bash
# Check if output directory was created
ls -ld attention_resunet_search_*

# Check latest console output
tail -100 attention_resunet_search_console_*.log

# Look for error messages
grep -i "error\|failed\|exception" attention_resunet_search_console_*.log
```

### Job Times Out (>12 hours)

If the job exceeds 12-hour walltime:

1. **Option A:** Increase walltime in PBS script
   ```bash
   #PBS -l walltime=24:00:00
   ```

2. **Option B:** Reduce search space (edit Python script)
   ```python
   HYPERPARAMETER_GRID = {
       'learning_rate': [2e-5],        # Just test optimal
       'dropout': [0.2, 0.3],          # Reduce to 2 values
       'batch_size': [4],              # Just test best batch size
   }
   # Now: 1 LR × 2 dropout × 1 batch = 2 configs × 3 folds = 6 models (~30-40 min)
   ```

---

## Comparison: ResUNet vs Attention ResUNet

| Aspect | ResUNet Search | Attention ResUNet Search |
|--------|---------------|-------------------------|
| **Search Space** | 3 LR × 3 dropout × 2 batch = 18 | Same 18 configs |
| **Learning Rates** | 1e-05, 2e-05, 5e-05 | **1.5e-05, 2e-05, 2.5e-05** ← Fine-tuned |
| **Dropout** | 0.3, 0.4, 0.5 | **0.2, 0.3, 0.4** ← Lower range |
| **Strategy** | Exploratory | **Exploitative** (focused on optimal region) |
| **Models Trained** | 108 (6 failed, 12 configs × 3 folds) | 54 (more efficient) |
| **Runtime** | 8-12 hours | 6-10 hours |
| **Best Found** | 42.40% ± 4.60% | **TBD** |

**Key Difference:** Attention ResUNet search is informed by ResUNet failures, making it a more targeted investigation.

---

## Understanding Attention Gates

**What They Do:**
Attention gates learn to focus on relevant spatial regions, suppressing irrelevant features.

**Why They Might Help:**
- ResUNet struggles with feature learning (identity mapping problem)
- Attention forces model to selectively emphasize useful features
- Could mitigate shortcut learning issue

**Why They Might Not Help:**
- ResUNet's problem is gradient explosion, not feature relevance
- Attention adds parameters → more overfitting risk
- May not address fundamental architectural mismatch

**The Experiment:** This search will determine which hypothesis is correct.

---

## Files Created

```
hyperparameter_search_attention_resunet.py  ← Main search script (9 KB)
pbs_attention_resunet_search.sh             ← HPC submission script (6 KB)
QUICKSTART_ATTENTION_RESUNET_SEARCH.md      ← This guide
```

**Transfer to HPC:**
```bash
scp hyperparameter_search_attention_resunet.py pbs_attention_resunet_search.sh \
    phyzxi@atlas7.nus.edu.sg:~/scratch/unet-HPC/
```

---

## Expected Outcomes

### Scenario 1: Success (>55% Jaccard)
**Interpretation:** Attention gates successfully address ResUNet's feature learning deficit
**Action:** Run full 5-fold validation, compare statistically to U-Net
**Timeline:** +1 day for full validation

### Scenario 2: Marginal (42-55% Jaccard)
**Interpretation:** Attention helps slightly but not enough to justify complexity
**Action:** Stick with U-Net, explore data improvements
**Timeline:** Immediate - move to next steps

### Scenario 3: Failure (≤42% Jaccard)
**Interpretation:** Residual architectures fundamentally incompatible with this task
**Action:** Abandon residual approaches entirely, focus on U-Net variants without residuals
**Timeline:** Immediate - explore U-Net++, Dense U-Net, or data improvements

---

## Questions?

**Q:** Why only 54 models instead of 108 like ResUNet?
**A:** We're fine-tuning around the optimal region found by ResUNet, not exploring broadly. This is more efficient.

**Q:** What if I want to test more learning rates?
**A:** Edit `HYPERPARAMETER_GRID` in the Python script to add more values, but expect longer runtime.

**Q:** Can I run this locally?
**A:** Yes, but it will take 3-4× longer without GPU. Run: `python3 hyperparameter_search_attention_resunet.py`

**Q:** What if attention gates don't improve performance?
**A:** Then we conclusively know residual architectures aren't suitable for this task. Stick with U-Net (69.94% Jaccard) and focus efforts on data quality/quantity improvements instead.

---

**Created:** October 14, 2025
**For:** Mitochondria segmentation project
**Based on:** ResUNet hyperparameter search analysis
