# Hyperparameter Search - Quick Reference Guide

**Date:** October 16, 2025

---

## 🎯 Best Models Summary

| Rank | Architecture | IoU | Dice | Filters | Dropout | LR | Epoch |
|------|-------------|-----|------|---------|---------|-----|-------|
| 🥇 1st | **UNet** | **0.5080** | 0.6649 | 32 | 0.3 | 0.001 | 75 |
| 🥈 2nd | **Attention ResUNet** | **0.5039** | 0.6653 | 32 | 0.1 | 0.003 | 76 |
| 🥉 3rd | **Attention UNet** | **0.4875** | 0.6471 | 16 | 0.3 | 0.003 | 74 |

**Performance Gap:** Only 2% between 1st and 3rd place - hyperparameters matter more than architecture choice!

---

## 📊 Architecture Comparison

### Mean Performance (across all 27 hyperparameter combinations)

| Architecture | Mean IoU | Std | Best | Worst | Consistency |
|-------------|----------|-----|------|-------|-------------|
| **Attention UNet** | 0.389 | 0.084 | 0.488 | 0.212 | ⭐⭐⭐ High |
| **UNet** | 0.381 | 0.079 | 0.508 | 0.212 | ⭐⭐⭐ High |
| **Attention ResUNet** | 0.314 | 0.118 | 0.504 | 0.162 | ⭐ Low |

**Key Insight:** Attention ResUNet has 3× more failure modes than UNet (12 vs 4 configs with IoU<0.25)

---

## 🔧 Recommended Hyperparameters

### Production Deployment (Highest Performance)
```python
model = UNet(
    n_filters=32,
    dropout=0.3,
    learning_rate=0.001
)
# Expected IoU: 0.508
```

### Resource-Constrained (50% Fewer Parameters)
```python
model = AttentionUNet(
    n_filters=16,
    dropout=0.2,
    learning_rate=0.001
)
# Expected IoU: 0.47 (96% of best)
```

### Research (High Risk/Reward)
```python
model = AttentionResUNet(
    n_filters=32,
    dropout=0.1,  # CRITICAL: Must be ≤0.1
    learning_rate=0.003
)
# Expected IoU: 0.504
```

---

## ⚠️ Critical Warnings

### Attention ResUNet Dropout Sensitivity
```
Dropout 0.1: IoU = 0.50 ✅
Dropout 0.2: IoU = 0.38 ⚠️
Dropout 0.3: IoU = 0.22 ❌ FAILURE
```
**Never use dropout >0.2 with Attention ResUNet!**

### Learning Rate Sensitivity
```
LR 0.001: IoU = 0.45 ✅
LR 0.003: IoU = 0.43 ✅
LR 0.005: IoU = 0.30 ❌ Unstable
```
**Always use LR ≤0.003 for all architectures**

### Model Capacity Overfitting
```
16 filters: IoU = 0.38 (underfitting)
32 filters: IoU = 0.43 ✅ OPTIMAL
64 filters: IoU = 0.35 (overfitting)
```
**32 filters is optimal for current dataset size**

---

## 📈 Hyperparameter Effects Ranking

### Impact on Performance (by magnitude)

1. **Learning Rate** (Δ IoU = 0.15)
   - 0.001 → 0.005: -33% performance drop
   - Universal sensitivity across all architectures

2. **Dropout** (Δ IoU = 0.10 for ResUNet, 0.05 for UNet)
   - Architecture-dependent
   - ResUNet: Extremely sensitive
   - UNet/Attention UNet: Tolerant

3. **Number of Filters** (Δ IoU = 0.08)
   - Sweet spot at 32 filters
   - Diminishing returns at 64

---

## 🚀 Training Tips

### Recommended Training Config
```python
config = {
    'epochs': 100,
    'batch_size': 4,
    'early_stopping_patience': 20,
    'reduce_lr_patience': 10,
    'loss': 'binary_focal_loss',
    'focal_gamma': 2,
    'focal_alpha': 0.25
}
```

### Expected Training Time (HPC)
- UNet: ~3.3 hours (27 models)
- Attention UNet: ~3.7 hours (+15% overhead)
- Attention ResUNet: ~3.6 hours

### Convergence Expectations
- Median best epoch: 40-45
- Early stopping saves ~55% of training time
- If converging in <10 epochs: likely poor hyperparameters

---

## 📋 Experiment Checklist

### Before Training
- [ ] Learning rate ≤0.003
- [ ] Dropout ≤0.2 for ResUNet architectures
- [ ] Dropout 0.2-0.3 for UNet/Attention UNet
- [ ] n_filters = 32 (or 16 for limited compute)
- [ ] Early stopping enabled (patience=20)
- [ ] ReduceLROnPlateau enabled (patience=10)

### After Training
- [ ] Best epoch between 30-80 (not 1-5 or 95-100)
- [ ] Validation IoU >0.35 (minimum acceptable)
- [ ] Validation IoU >0.45 (good performance)
- [ ] Final IoU within 10% of best IoU (stable)

---

## 📁 File Locations

### Results CSVs
```
unet_hyperparam_20251015_224125/unet_results.csv
attention_unet_hyperparam_20251015_230149/attention_unet_results.csv
attention_resunet_hyperparam_20251015_235542/attention_resunet_results.csv
```

### Best Models
```
unet_hyperparam_20251015_224125/checkpoints/
    unet_n_filters32_dropout0p3_batch_normTrue_learning_rate0p001/best_model.keras

attention_unet_hyperparam_20251015_230149/models/
    attention_unet_n_filters16_dropout0p3_batch_normTrue_learning_rate0p003/best_model.keras

attention_resunet_hyperparam_20251015_235542/models/
    attention_resunet_n_filters32_dropout0p1_batch_normTrue_learning_rate0p003/best_model.keras
```

### Analysis Figures
```
hyperparam_comparison_report/
    fig1_best_iou_comparison.png
    fig2_iou_distribution.png
    fig3_hyperparameter_heatmaps.png
    fig4_dropout_effect.png
    fig5_learning_rate_effect.png
    fig6_n_filters_effect.png
    fig7_convergence_epoch.png
```

---

## 🔬 Key Scientific Findings

### 1. Attention Mechanisms Don't Always Win
- UNet achieved best peak performance (0.508)
- Simpler architectures can outperform complex ones with proper tuning

### 2. Residual Connections Require Special Care
- Attention ResUNet is highly sensitive to dropout
- Hypothesis: Skip connections provide implicit regularization

### 3. Hyperparameter Tuning > Architecture Choice
- 24% gap within same architecture (best vs worst config)
- Only 4% gap between best models across architectures

### 4. Small Models Can Be Competitive
- Attention UNet with 16 filters achieves 96% of best performance
- Model efficiency crucial for deployment

### 5. Dataset Size Limits Model Capacity
- 64 filters cause overfitting across all architectures
- Current dataset insufficient for very large models

---

## 📖 Full Report

See `HYPERPARAMETER_SEARCH_COMPARISON_REPORT.md` for:
- Detailed analysis (490 lines)
- Statistical discussions
- Failure mode analysis
- Future research directions
- Complete reproducibility info

---

## 🎓 Practical Recommendations

### Scenario 1: Production Deployment (Priority: Best Performance)
**Use: UNet (32F, 0.3D, 0.001LR)**
- Highest IoU (0.508)
- Robust and well-tested
- Fast inference

### Scenario 2: Limited Compute (Priority: Efficiency)
**Use: Attention UNet (16F, 0.2D, 0.001LR)**
- 50% fewer parameters
- 96% of best performance
- More consistent across hyperparameters

### Scenario 3: Research (Priority: Exploration)
**Use: Attention ResUNet (32F, 0.1D, 0.003LR)**
- Competitive peak performance (0.504)
- Requires careful tuning
- Potential for improvement with better regularization

---

**Quick Reference Version:** 1.0
**Full Report:** HYPERPARAMETER_SEARCH_COMPARISON_REPORT.md
**Generated:** October 16, 2025
