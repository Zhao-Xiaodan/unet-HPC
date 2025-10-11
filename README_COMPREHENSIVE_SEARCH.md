# Comprehensive Hyperparameter Search - Quick Reference

🎯 **Goal:** Find optimal architecture, batch size, and loss function for microbead segmentation at 512×512

---

## 📦 New Files Created

| File | Purpose | Status |
|------|---------|--------|
| `loss_functions.py` | 5 advanced loss functions | ✅ Tested |
| `model_architectures.py` | 3 architectures (U-Net, ResU-Net, Attention ResU-Net) | ✅ Tested |
| `hyperparam_search_comprehensive.py` | Main search script | ✅ Ready |
| `pbs_hyperparam_comprehensive.sh` | HPC job script | ✅ Ready |
| `HYPERPARAM_SEARCH_COMPREHENSIVE.md` | Full documentation | ✅ Complete |

---

## 🚀 Quick Start

### To Run on HPC:

```bash
# 1. Verify dataset exists
ls dataset_shrunk_masks/images/ | wc -l
ls dataset_shrunk_masks/masks/ | wc -l

# 2. Submit job
qsub pbs_hyperparam_comprehensive.sh

# 3. Monitor
qstat -u $USER
tail -f Hyperparam_Comprehensive.o<JOBID>
```

###  To Test Locally:

```bash
conda activate unetCNN

# Test components
python loss_functions.py              # Test loss functions
python model_architectures.py         # Test architectures

# Quick test (2 configs)
python hyperparam_search_comprehensive.py --search-type random --n-random 2
```

---

## 🎯 What's Being Tested

### Search Space

- **Architectures:** U-Net (31M params), ResU-Net (33M params), Attention ResU-Net (34M params)
- **Batch Sizes:** 8, 16, 32
- **Loss Functions:** focal, combined, focal_tversky, combined_tversky
- **Dropout:** 0.3 (fixed)

### Fixed Parameters (From Previous Analysis)

- **Learning Rate:** 5e-5 (lower for stability)
- **Early Stopping:** 30 epochs patience (increased)
- **Image Size:** 512×512 (original resolution)
- **Dataset:** dataset_shrunk_masks

### Total Configurations

- **Random Search:** 30 combinations (default, ~7-15 hours)
- **Grid Search:** 36 combinations (all combos, ~9-18 hours)

---

## 💡 Why These Changes?

### Problem from Previous Search (Jaccard: 0.164)

1. **Batch Size = 4** → Noisy gradients, training instability → **Main bottleneck**
2. **Standard U-Net** → May have insufficient receptive field at 512×512
3. **Limited Loss Functions** → Only dice, focal, combined tested

### Solution

1. **Larger Batch Sizes (8, 16, 32)** → More stable gradients → **Expected: +20-30% improvement**
2. **Advanced Architectures** → Better receptive fields, attention mechanisms → **Expected: +5-10% improvement**
3. **Advanced Losses (Tversky variants)** → Better FP/FN control for touching objects → **Expected: +2-5% improvement**

**Total Expected Improvement:** 0.164 → 0.21-0.24 (matching or exceeding 256×256 baseline of 0.2456)

---

## 📊 Expected Best Configuration

```json
{
  "architecture": "attention_resunet",
  "batch_size": 16,
  "dropout": 0.3,
  "loss_function": "focal_tversky",
  "learning_rate": 5e-5,
  "expected_jaccard": "0.21-0.24"
}
```

---

## 📁 Output Structure

```
hyperparam_comprehensive_YYYYMMDD_HHMMSS/
├── search_results_final.csv          # All results, ranked
├── best_hyperparameters.json         # Best config
├── history_*.csv                      # Training curves (36 files)
└── model_*.hdf5                       # Saved models (36 files)
```

---

## 🔍 Quick Analysis After Completion

```bash
# View best configuration
cat hyperparam_comprehensive_*/best_hyperparameters.json

# View top 5 results
head -6 hyperparam_comprehensive_*/search_results_final.csv | column -t -s,

# Compare to previous
echo "Previous best: 0.164 (LR=0.0002, BS=4, Dropout=0.3, combined)"
```

---

## 📈 Component Test Results

### Loss Functions ✅
```
focal                    : 0.001247  ✓
tversky                  : 0.150000  ✓
focal_tversky            : 0.080205  ✓
combined                 : 0.105374  ✓
combined_tversky         : 0.090499  ✓
```

### Architectures ✅
```
UNet               : 31,401,345 params  ✓
ResUNet            : 33,156,929 params  ✓
AttentionResUNet   : 34,204,293 params  ✓
```

---

## 🎓 Key Insights from Previous Analysis

**★ Insight ─────────────────────────────────────**

1. **Batch Size Was the Main Bottleneck:**
   - Previous BS=4 caused 15% overfitting (0.164 → 0.140)
   - Larger batches (16-32) should provide stable training
   - Mathematical analysis showed 29× gradient amplification needed

2. **512×512 Challenges:**
   - 4× more pixels than 256×256
   - Class imbalance doubles (1:7.2 vs 1:3.3)
   - Boundary pixels only 2% of image
   - Requires advanced loss functions and architecture

3. **Combined Loss Power:**
   - Previous: Combined (0.7×Dice + 0.3×Focal) = 0.164
   - Pure Dice = 0.080 (failed at high resolution)
   - Pure Focal = 0.125 (consistent but no metric alignment)
   - Tversky variants should further improve FP/FN balance

─────────────────────────────────────────────────

---

## 🔗 Related Documentation

- **Full Guide:** `HYPERPARAM_SEARCH_COMPREHENSIVE.md`
- **Previous Analysis:** `hyperparam_search_20251010_043123/REPORT.md`
- **Mathematical Analysis:** `hyperparam_search_20251010_043123/MATHEMATICAL_ANALYSIS_SUMMARY.md`

---

**Ready to run!** 🚀

Submit with: `qsub pbs_hyperparam_comprehensive.sh`
