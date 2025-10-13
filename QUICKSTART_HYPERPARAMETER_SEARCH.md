# Hyperparameter Search - Quick Start

## 🎯 Goal

Fix ResUNet's catastrophic failure (39.95% → target 70%)
Find optimal settings for Attention ResUNet (62.69% → target 70%+)

---

## 🚀 Submit to HPC (Fast Path)

```bash
# 1. Transfer files
scp hyperparameter_search_residual_architectures.py \
    model_architectures.py loss_functions_fixed.py \
    pbs_hyperparameter_search.sh \
    phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/

# 2. SSH and submit
ssh phyzxi@nscc
cd /home/svu/phyzxi/scratch/unet-HPC
qsub pbs_hyperparameter_search.sh

# 3. Monitor (~8-12 hours)
qstat -u phyzxi
tail -f Hyperparam_Search.o<JOB_ID>

# 4. Download results
scp -r phyzxi@nscc:/home/svu/phyzxi/scratch/unet-HPC/hyperparameter_search_*/ ./

# 5. Check best configs
cat hyperparameter_search_*/hyperparameter_search_summary.json
```

---

## 📊 What Gets Tested

**Search Space:**
- **Learning Rate:** [1e-5, 2e-5, 5e-5] (lower rates to fix collapse)
- **Dropout:** [0.3, 0.4, 0.5] (higher regularization)
- **Batch Size:** [4, 8] (gradient noise tuning)

**Total:** 36 configurations (2 architectures × 3 LR × 3 dropout × 2 batch)

**Each with 3-fold CV = 108 models to train**

---

## ⏱️ Timeline

| Stage | Duration |
|-------|----------|
| Queue wait | 0-30 min |
| Search execution | 8-12 hours |
| **Total** | **~9-12 hours** |

---

## 📁 Output Files

```
hyperparameter_search_YYYYMMDD_HHMMSS/
├── hyperparameter_search_summary.json    # ← Read this first!
├── resunet_lr1e-05_drop0.3_bs4/
│   └── fold_1...fold_3/
│       ├── best_model.keras
│       ├── history.csv
│       └── results.json
└── ... (all 36 configs)
```

---

## 🎯 Success Criteria

### For ResUNet

**Target:** ≥68% Jaccard (vs baseline 39.95%, vs U-Net 69.94%)

✅ **Success if:**
- Best config ≥68% Jaccard
- Best epoch shifts to 8-15 (not 1-3!)
- Overfitting gap <2.5×

❌ **Failure if:**
- All configs <55% Jaccard
- Still peaking at epoch 1-3
- High overfitting gap (>3×)

### For Attention ResUNet

**Target:** ≥70% Jaccard (vs baseline 62.69%, vs U-Net 69.94%)

✅ **Success if:**
- Best config >70% Jaccard (beats U-Net)
- Statistically significant improvement

---

## 📊 Quick Results Check

```bash
# After job completes, run this:
python -c "
import json

with open('hyperparameter_search_*/hyperparameter_search_summary.json') as f:
    data = json.load(f)

best = data['best_configs']
baseline_unet = 0.6994

for arch, config in best.items():
    perf = config['mean_best_jacard']
    improvement = ((perf - baseline_unet) / baseline_unet) * 100

    print(f'{arch.upper()}:')
    print(f'  Performance: {perf:.4f}')
    print(f'  vs U-Net: {improvement:+.1f}%')

    if perf >= 0.70:
        print(f'  ✅ SUCCESS - Beats U-Net!')
    elif perf >= 0.68:
        print(f'  📊 Close to U-Net (within 2%)')
    else:
        print(f'  ❌ Still underperforming')
    print()
"
```

---

## 🔄 What to Do After Search

### Scenario 1: Success Found (≥68% Jaccard) ✅

```bash
# 1. Note best configuration from summary.json
# Example: resunet_lr1e-05_drop0.4_bs8

# 2. Run full 5-fold CV with optimal config
# (Modify validate_architecture_comparison.py with these settings)

# 3. Compare to U-Net statistically
# If significantly better → deploy new architecture
# If not significant → U-Net is simpler, use that
```

### Scenario 2: Partial Improvement (50-67%) ⚠️

```bash
# ResUNet improved but didn't match U-Net

# Decision: Stick with U-Net (69.94%)
# Reason: Simpler, proven, faster training

# Alternative: If you specifically need residual connections
# for other reasons, use improved config
```

### Scenario 3: No Improvement (<50%) ❌

```bash
# ResUNet fundamentally incompatible with this task

# RECOMMENDATION: Use U-Net, abandon ResUNet
# Focus on:
#   - Data augmentation
#   - Ensemble methods
#   - Post-processing improvements
```

---

## 🔑 Key Insights from Architecture Comparison

**Why this search is needed:**

1. **ResUNet catastrophic failure:**
   - Baseline: 39.95% (vs U-Net's 69.94%)
   - Peaks at epoch 2-3 then **crashes**
   - Root cause: Learning rate too high for residual connections

2. **Hypothesis:**
   - Lower LR (1e-5, 2e-5) will stabilize training
   - Higher dropout (0.4, 0.5) will reduce overfitting
   - Optimal combo will match U-Net's 70%

3. **If hypothesis correct:**
   - Best epoch shifts to 8-15
   - Overfitting gap drops to ~2×
   - Performance reaches 68-72%

---

## 📖 Detailed Guides

- **Full guide:** `HYPERPARAMETER_SEARCH_GUIDE.md` (13 KB)
- **Architecture analysis:** `validation_arch_comparison_20251013_093844/REPORT.md`
- **HPC submission:** Standard PBS workflow

---

## ⚠️ Important Notes

1. **This is exploratory:** Success not guaranteed
2. **U-Net is excellent:** 69.94% is already very good
3. **Don't overengineer:** If search fails, stick with U-Net
4. **Consider cost:** ResUNet +25% slower even if improved

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Job takes >12 hours | Reduce search space (remove batch_size=8) |
| OOM errors | Remove batch_size=8, keep only batch_size=4 |
| No improvement | Accept U-Net is best, move on |
| Job stuck in queue | Submit during off-peak hours |

---

## 💡 Expected Best Configs (Predictions)

Based on failure analysis:

**ResUNet (predicted optimal):**
```python
{
    'learning_rate': 1e-5,  # 5× lower than baseline
    'dropout': 0.4,          # Slightly higher
    'batch_size': 8          # Larger, more stable
}
```

**Attention ResUNet (predicted optimal):**
```python
{
    'learning_rate': 2e-5,  # 2.5× lower
    'dropout': 0.4,          # Slightly higher
    'batch_size': 8          # Larger
}
```

**These are predictions—actual optimal may differ!**

---

## 📞 Support

- **Search explanation:** `HYPERPARAMETER_SEARCH_GUIDE.md`
- **Why ResUNet failed:** `validation_arch_comparison_20251013_093844/REPORT.md`
- **HPC issues:** `HPC_SUBMISSION_GUIDE.md`

---

**Ready to find optimal hyperparameters? Submit the job!** 🔬

*Remember: If search doesn't find improvement, that's valuable information too. U-Net's 69.94% performance is excellent—not every architecture needs to work for every task.*
