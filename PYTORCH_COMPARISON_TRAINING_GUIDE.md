# PyTorch Training: Fair Comparison with Keras Models

**Date:** October 19, 2025
**Purpose:** Enable direct comparison between Keras and PyTorch training approaches

---

## Overview

This directory contains **3 pairs** of training scripts (`.py` + `.sh`) for fair comparison with the Keras models. All scripts use:
- ✅ **Same preprocessing** as `train.py` (grayscale, percentile normalization)
- ✅ **Same dataset** (`./dataset_shrunk_masks/`)
- ✅ **Same train/val split** (80/20, random_seed=42)
- ✅ **Same architectures** (UNet, Attention UNet, Attention ResUNet)
- ✅ **Same hyperparameter grid** (3×3×3 = 27 configs per architecture)

---

## Training Approaches

### Approach 1: No Augmentation + BinaryFocalLoss
**Files:**
- `train_pytorch_comparison_no_aug.py`
- `pbs_train_pytorch_comparison_no_aug.sh`

**Purpose:** Direct comparison with Keras models (which also use no augmentation)

**Configuration:**
| Setting | Value |
|---------|-------|
| **Preprocessing** | Grayscale, percentile normalization |
| **Augmentation** | None |
| **Loss Function** | BinaryFocalLoss (α=0.25, γ=2.0) |
| **Learning Rates** | [0.001, 0.003, 0.005] |
| **Dropout** | [0.1, 0.2, 0.3] |
| **Filters** | [16, 32, 64] |
| **Total Models** | 81 (3 architectures × 27 configs) |

**Expected Output:**
```
./pytorch_comparison_no_aug_YYYYMMDD_HHMMSS/
├── unet/
│   ├── checkpoints/
│   │   ├── unet_n_filters16_dropout0p1_learning_rate0p001/best_model.pth
│   │   └── ... (27 total)
│   └── logs/
│       └── *_history.csv (27 total)
├── attention_unet/
│   └── ... (same structure)
├── attention_resunet/
│   └── ... (same structure)
├── config.json
└── all_results.csv
```

**Submit:**
```bash
qsub pbs_train_pytorch_comparison_no_aug.sh
```

---

### Approach 2: WITH Augmentation + BinaryFocalLoss
**Files:**
- `train_pytorch_comparison_with_aug.py`
- `pbs_train_pytorch_comparison_with_aug.sh`

**Purpose:** Test impact of augmentation while keeping loss function simple

**Configuration:**
| Setting | Value |
|---------|-------|
| **Preprocessing** | Grayscale, percentile normalization |
| **Augmentation** | **Synthetic background artifacts** (same as train.py) |
| **Augmentation Mix** | 40% none, 30% old-style, 30% new-style with fading |
| **Loss Function** | BinaryFocalLoss (α=0.25, γ=2.0) |
| **Other Settings** | Same as Approach 1 |

**Augmentation Details:**
- **Old-style:** Gradients, sinusoidal bands, Gaussian noise
- **New-style:** Realistic artifacts + bead fading effect
- Applied to **training set only**, validation set unchanged

**Expected Output:**
```
./pytorch_comparison_with_aug_YYYYMMDD_HHMMSS/
├── (same structure as Approach 1)
```

**Submit:**
```bash
qsub pbs_train_pytorch_comparison_with_aug.sh
```

---

### Approach 3: WITH Augmentation + AdaptiveBGDiceLoss
**Files:**
- `train_pytorch_comparison_adaptive_loss.py`
- `pbs_train_pytorch_comparison_adaptive_loss.sh`

**Purpose:** Full reproduction of `train.py` setup but with architecture grid search

**Configuration:**
| Setting | Value |
|---------|-------|
| **Preprocessing** | Grayscale, percentile normalization |
| **Augmentation** | **Synthetic background artifacts** (same as train.py) |
| **Loss Function** | **AdaptiveBGDiceLoss** (5 components) |
| **Loss Components** | Focal BCE + BG-adaptive + TV + Tversky |
| **Focal Parameters** | α=0.4, γ=2.0 (train.py uses 0.4, not 0.25!) |
| **Other Settings** | Same hyperparameter grid |

**Loss Function Breakdown:**
```python
Total Loss = Main Loss (Focal BCE)
           + 0.05 × L_bg_adapt      # Background-adaptive penalty
           + 1.0 × L_tv             # Total variation on flat backgrounds
           + 0.4 × L_tversky        # Asymmetric Dice loss
```

**Expected Output:**
```
./pytorch_comparison_adaptive_loss_YYYYMMDD_HHMMSS/
├── (same structure as Approach 1)
```

**Submit:**
```bash
qsub pbs_train_pytorch_comparison_adaptive_loss.sh
```

---

## Comparison Matrix

| Aspect | Keras Models | PyTorch Approach 1 | PyTorch Approach 2 | PyTorch Approach 3 |
|--------|--------------|-------------------|-------------------|-------------------|
| **Framework** | TensorFlow/Keras | PyTorch | PyTorch | PyTorch |
| **Preprocessing** | RGB, /255 | **Grayscale, percentile** | **Grayscale, percentile** | **Grayscale, percentile** |
| **Augmentation** | None | None | **Synthetic artifacts** | **Synthetic artifacts** |
| **Loss** | BinaryFocalLoss (α=0.25) | BinaryFocalLoss (α=0.25) | BinaryFocalLoss (α=0.25) | **AdaptiveBGDiceLoss** (α=0.4) |
| **Architectures** | 3 (U-Net variants) | 3 (same) | 3 (same) | 3 (same) |
| **Configs** | 27 per arch | 27 per arch | 27 per arch | 27 per arch |
| **Total Models** | 81 | 81 | 81 | 81 |

---

## Expected Results Analysis

### Research Questions

**Q1: Does preprocessing matter (RGB vs Grayscale+Percentile)?**
- Compare: Keras vs PyTorch Approach 1
- Both use: No augmentation, BinaryFocalLoss
- Difference: **Preprocessing only**

**Q2: Does augmentation help?**
- Compare: PyTorch Approach 1 vs PyTorch Approach 2
- Both use: Percentile norm, BinaryFocalLoss
- Difference: **Augmentation only**

**Q3: Does advanced loss function help?**
- Compare: PyTorch Approach 2 vs PyTorch Approach 3
- Both use: Percentile norm, augmentation
- Difference: **Loss function only** (BinaryFocalLoss vs AdaptiveBGDiceLoss)

**Q4: What is the best overall combination?**
- Compare all approaches
- Identify: Best architecture + best settings

---

## File Structure Reference

### Preprocessing Comparison

**Keras (train_unet_hyperparam.py):**
```python
# RGB input
image = cv2.imread(path, 1)  # RGB
image = image.resize((512, 512))
image_normalized = image / 255.0  # Linear scaling
# Final: (512, 512, 3)
```

**PyTorch (all comparison scripts):**
```python
# Grayscale input
image = Image.open(path).convert("L")  # Grayscale
arr = _percentile_norm(arr)  # Percentile normalization
image = image.resize((512, 512))  # After normalization
# Final: (1, 512, 512)
```

### Loss Function Comparison

**BinaryFocalLoss (Approaches 1 & 2):**
```python
FL(p, y) = -α_t * (1 - p_t)^γ * BCE(p, y)
# α = 0.25, γ = 2.0
# Single component: focal-weighted cross entropy
```

**AdaptiveBGDiceLoss (Approach 3):**
```python
Total = Focal(α=0.4, γ=2.0)            # Main loss
      + 0.05 × BG_Adaptive              # Penalize over-prediction on bright backgrounds
      + 1.0 × TV_Regularization         # Smoothness on flat regions
      + 0.4 × Tversky(α=0.7, β=0.3)     # Asymmetric Dice
# Multi-component: domain-specific for microscopy
```

---

## Computational Requirements

**Per Training Run:**
- **GPU:** A100 (40-48 GB VRAM)
- **RAM:** 240 GB
- **Walltime:** 24 hours (conservative)
- **Storage:** ~5 GB per run
- **Total Models:** 81 models per run
- **Expected Runtime:** 18-24 hours (depending on convergence)

**Total for All 3 Approaches:**
- **Models:** 243 total (81 × 3)
- **Time:** ~3 days (if run sequentially)
- **Storage:** ~15 GB

**Recommendation:** Run all 3 approaches in parallel if resources available.

---

## Output Analysis

### Comparing Results

After all trainings complete, compare best models:

```bash
# Extract best models from each approach
python analyze_pytorch_comparison.py

# Expected comparison:
# 1. Best IoU per architecture per approach
# 2. Best hyperparameters per approach
# 3. Training curves comparison
# 4. Inference speed comparison
```

### Key Metrics to Track

1. **Validation IoU** (primary metric)
2. **Validation Dice** (secondary metric)
3. **Training time** per config
4. **Convergence speed** (epochs to best)
5. **Final model sizes**

---

## Troubleshooting

### Common Issues

**Issue 1: Out of Memory (OOM)**
```bash
# Solution: Reduce batch size in CONFIG
'batch_size': 2,  # Default is 4
```

**Issue 2: Slow Training**
```bash
# Check: Is GPU being utilized?
nvidia-smi  # Should show Python process using GPU

# Check: Are data workers sufficient?
# Increase num_workers in DataLoader
DataLoader(..., num_workers=8)  # Default is 4
```

**Issue 3: Import Errors**
```bash
# Ensure scipy is available (needed for box_blur_same)
singularity exec --nv $image python -c "import scipy; print(scipy.__version__)"

# If missing, augmentation functions will fallback gracefully
```

---

## Best Practices

### Before Submitting

1. **Verify dataset exists:**
```bash
ls -la ./dataset_shrunk_masks/images/ | head
ls -la ./dataset_shrunk_masks/masks/ | head
```

2. **Test script locally (1 config):**
```python
# Modify CONFIG temporarily
'hyper param_grid': {
    'n_filters': [32],  # Test with one value only
    'dropout': [0.2],
    'learning_rate': [0.001],
},
'architectures': ['unet'],  # Test with one architecture
'epochs': 2,  # Quick test
```

3. **Check PBS syntax:**
```bash
qstat -f <job_id>  # Verify job was submitted correctly
```

### During Training

1. **Monitor progress:**
```bash
tail -f pytorch_comparison_no_aug_*.log
```

2. **Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

3. **Estimate completion time:**
```bash
# If 1 config takes ~20-30 minutes
# 81 configs ≈ 27-40 hours
```

---

## Citation

If using these scripts for publication, please cite:

**U-Net:**
- Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"

**Attention U-Net:**
- Oktay et al. (2018). "Attention U-Net: Learning Where to Look for the Pancreas"

**Focal Loss:**
- Lin et al. (2017). "Focal Loss for Dense Object Detection"

**ResNet:**
- He et al. (2016). "Deep Residual Learning for Image Recognition"

---

## Contact

**Questions or issues?**
- Check job logs first: `pytorch_comparison_*_<jobid>.log`
- Review error messages in console logs
- Compare with Keras training logs for reference

---

**Created:** October 19, 2025
**Author:** Claude Code
**Purpose:** Fair comparison between Keras and PyTorch training approaches
**Total Scripts:** 6 (3 pairs of .py + .sh)
**Total Experiments:** 3 different training configurations
**Expected Total Models:** 243 (81 per approach)
