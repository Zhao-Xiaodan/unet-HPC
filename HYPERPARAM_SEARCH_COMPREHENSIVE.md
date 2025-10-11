# Comprehensive Hyperparameter Search - Documentation

**Date:** 2025-10-11
**Purpose:** Test architectures, batch sizes, and advanced loss functions for microbead segmentation

---

## 📋 Overview

This comprehensive hyperparameter search addresses the performance gap identified in the previous 512×512 training (0.164 Jaccard vs 0.2456 at 256×256) by testing:

1. **Multiple Architectures** - To find better receptive fields for 512×512
2. **Larger Batch Sizes** - To improve gradient stability (main bottleneck identified)
3. **Advanced Loss Functions** - To optimize boundary detection

---

## 🎯 Search Configuration

### Tested Parameters

| Parameter | Values | Count | Rationale |
|-----------|--------|-------|-----------|
| **Architecture** | U-Net, ResU-Net, Attention ResU-Net | 3 | Test different receptive fields and attention mechanisms |
| **Batch Size** | 8, 16, 32 | 3 | Improve gradient stability (previous BS=4 was problematic) |
| **Loss Function** | focal, combined, focal_tversky, combined_tversky | 4 | Test advanced loss combinations |
| **Dropout** | 0.3 (fixed) | 1 | Worked well in previous search |

### Fixed Parameters (From Analysis)

| Parameter | Value | Reason |
|-----------|-------|--------|
| **Learning Rate** | 5e-5 | Lower for stability, especially with larger batches |
| **Early Stopping Patience** | 30 epochs | Increased from 20 (best model peaked at epoch 52) |
| **Image Size** | 512×512 | Original dataset resolution |
| **Validation Split** | 15% | Standard split with stratification by density |

### Total Combinations

- **Grid Search:** 3 × 3 × 4 × 1 = **36 configurations**
- **Random Search:** **30 configurations** (default, faster)

---

## 📁 Files Created

### 1. `loss_functions.py` (Advanced Loss Functions)

Implements 5 advanced loss functions optimized for segmentation:

#### Available Loss Functions

| Loss Function | Description | Use Case |
|---------------|-------------|----------|
| **`focal`** | Focal Loss (α=0.25, γ=2.0) | Handles class imbalance, focuses on hard examples |
| **`tversky`** | Tversky Loss (α=0.7, β=0.3) | Controls FP/FN trade-off, good for small objects |
| **`focal_tversky`** | Focal Tversky (γ=1.33) | Combines focusing with FP/FN control |
| **`combined`** | 0.7×Dice + 0.3×Focal | Best from previous search |
| **`combined_tversky`** | 0.6×Tversky + 0.4×Focal | Alternative combination |

**Key Features:**
- All losses tested and validated
- Gradient-friendly implementations
- Configurable hyperparameters
- Compatible with TensorFlow/Keras

#### Test Results
```
focal                    : 0.001247  ✓
tversky                  : 0.150000  ✓
focal_tversky            : 0.080205  ✓
combined                 : 0.105374  ✓
combined_tversky         : 0.090499  ✓
```

### 2. `model_architectures.py` (Network Architectures)

Implements 3 segmentation architectures:

#### Available Architectures

| Architecture | Parameters | Description | Advantages |
|--------------|------------|-------------|------------|
| **`unet`** | 31.4M | Standard U-Net | Baseline, proven performance |
| **`resunet`** | 33.2M | Residual U-Net | Better gradient flow, deeper training |
| **`attention_resunet`** | 34.2M | Attention Residual U-Net | Focus on relevant regions, best for small objects |

**Key Features:**
- All models tested at 512×512 resolution
- Configurable dropout and batch normalization
- Named layers for easy inspection
- Memory-efficient implementations

#### Architecture Details

**U-Net:**
- 4 encoder blocks + bridge + 4 decoder blocks
- Max pooling for downsampling
- Upsampling + concatenation for skip connections
- Standard convolutional blocks

**ResU-Net:**
- Same structure as U-Net
- Residual connections in all blocks
- Improved gradient propagation
- Better for deeper networks

**Attention ResU-Net:**
- ResU-Net base
- Attention gates at each decoder level
- Focuses on relevant spatial regions
- Best for boundary refinement and small objects

### 3. `hyperparam_search_comprehensive.py` (Main Search Script)

**Features:**
- Automatic dataset loading from `dataset_shrunk_masks`
- Stratified train/val split by object density
- Strong data augmentation
- Intermediate result saving
- Comprehensive logging
- Memory-efficient training

**Usage:**
```bash
# Random search (default, 30 configs)
python hyperparam_search_comprehensive.py --search-type random --n-random 30

# Grid search (all 36 configs)
python hyperparam_search_comprehensive.py --search-type grid
```

**Outputs:**
- `hyperparam_comprehensive_YYYYMMDD_HHMMSS/` - Results directory
  - `search_results_final.csv` - All results ranked by performance
  - `best_hyperparameters.json` - Best configuration
  - `history_*.csv` - Training history for each config
  - `model_*.hdf5` - Saved model checkpoints

### 4. `pbs_hyperparam_comprehensive.sh` (HPC Job Script)

**Configuration:**
- Walltime: 48 hours (sufficient for random search)
- Resources: 1 node, 36 cores, 1 GPU, 240GB RAM
- Container: TensorFlow 2.16.1 with CUDA 12.5

**Features:**
- Dataset validation before training
- Memory optimization flags
- Comprehensive logging
- Automatic result summarization
- Error handling

**Submission:**
```bash
qsub pbs_hyperparam_comprehensive.sh
```

---

## 🔬 Expected Improvements

Based on previous analysis, this search addresses:

### 1. Batch Size Bottleneck ⭐ **Main Issue**

**Problem:**
- Previous search used BS=4 (forced by memory constraints)
- Resulted in noisy gradients and training instability
- Best model showed 15% overfitting (0.164 → 0.140)

**Solution:**
- Test BS=8, 16, 32
- Should provide more stable gradients
- **Expected improvement: 0.164 → 0.19-0.21**

### 2. Architecture Limitations

**Problem:**
- Standard U-Net may have insufficient receptive field at 512×512
- Previous: 33% performance drop from 256×256

**Solution:**
- ResU-Net: Better gradient flow
- Attention ResU-Net: Focus on boundaries and small objects
- **Expected improvement: +5-10% Jaccard**

### 3. Loss Function Optimization

**Problem:**
- Combined loss worked best but had high variance
- May benefit from FP/FN control (Tversky)

**Solution:**
- Test Tversky and Focal Tversky
- Better control for touching microbeads
- **Expected improvement: +2-5% Jaccard**

---

## 📊 Predicted Best Configuration

Based on analysis and theoretical understanding:

```json
{
  "architecture": "attention_resunet",
  "batch_size": 16,
  "dropout": 0.3,
  "loss_function": "focal_tversky",
  "learning_rate": 5e-5
}
```

**Reasoning:**
1. **Attention ResU-Net** - Best for small object segmentation with boundary focus
2. **BS=16** - Good compromise between memory and gradient stability
3. **Focal Tversky** - Combines hard example mining with FP/FN control
4. **LR=5e-5** - Stable for various batch sizes

**Expected Performance:** Jaccard 0.21-0.24 (matching or exceeding 256×256 baseline)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure dataset exists
ls dataset_shrunk_masks/images/
ls dataset_shrunk_masks/masks/

# Verify Python files
ls loss_functions.py
ls model_architectures.py
ls hyperparam_search_comprehensive.py
```

### Local Testing

```bash
# Test loss functions
conda activate unetCNN
python loss_functions.py

# Test model architectures
python model_architectures.py

# Test with 2 random configs (quick test)
python hyperparam_search_comprehensive.py --search-type random --n-random 2
```

### HPC Execution

```bash
# Submit job
qsub pbs_hyperparam_comprehensive.sh

# Monitor job
qstat -u $USER

# Check progress (while running)
tail -f Hyperparam_Comprehensive.o<JOBID>

# View results (after completion)
ls -ltr hyperparam_comprehensive_*/
cat hyperparam_comprehensive_*/best_hyperparameters.json
```

---

## 📈 Analysis Tools

After the search completes, analyze results:

```python
import pandas as pd

# Load results
df = pd.read_csv('hyperparam_comprehensive_YYYYMMDD_HHMMSS/search_results_final.csv')

# Top architectures
print(df.groupby('architecture')['best_val_jacard'].agg(['mean', 'max', 'count']))

# Top batch sizes
print(df.groupby('batch_size')['best_val_jacard'].agg(['mean', 'max', 'count']))

# Top loss functions
print(df.groupby('loss_function')['best_val_jacard'].agg(['mean', 'max', 'count']))

# Best overall
print(df.iloc[0])
```

---

## 💡 Key Insights

**★ Insight ─────────────────────────────────────**

1. **Batch Size is Critical:**
   - Previous search showed BS=4 was the main bottleneck
   - Larger batches (8, 16, 32) should dramatically improve stability
   - This alone could close the 33% performance gap

2. **Architecture Matters at High Resolution:**
   - Standard U-Net may have inadequate receptive field at 512×512
   - Attention mechanisms help focus on small objects (109 beads/image)
   - Residual connections enable deeper, more expressive networks

3. **Advanced Losses Address Specific Challenges:**
   - Focal: Handles class imbalance (88% background at 512×512)
   - Tversky: Controls FP/FN for touching microbeads
   - Combined approaches leverage multiple objectives

4. **Fixed LR Strategy:**
   - Using single LR (5e-5) simplifies search
   - Focus resources on architecture/batch size/loss
   - Can fine-tune LR later for best config

─────────────────────────────────────────────────

---

## 🔗 Related Files

- **Previous Analysis:** `hyperparam_search_20251010_043123/REPORT.md`
- **Mathematical Analysis:** `hyperparam_search_20251010_043123/MATHEMATICAL_ANALYSIS_SUMMARY.md`
- **Original Models:** `models.py`, `224_225_226_models.py`
- **Original Loss:** Uses basic functions (dice, focal, combined)

---

## 📞 Support

For issues or questions:

1. Check PBS output file: `Hyperparam_Comprehensive.o<JOBID>`
2. Verify dataset: `dataset_shrunk_masks/` with images and masks
3. Check TensorFlow version: Should be 2.16.1
4. Memory issues: Reduce batch sizes or use gradient accumulation

---

**Estimated Completion Time:**
- Random search (30 configs): 7-15 hours
- Grid search (36 configs): 9-18 hours

**Expected Best Performance:** Jaccard 0.21-0.24 (vs previous 0.164)

Good luck! 🎯
