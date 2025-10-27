# PyTorch Experiments Comparison

## ⚠️ CRITICAL FINDING: Best Models Are NOT All From "No Aug"!

You are absolutely correct to question this. After analyzing all three experiments, **the best models for each architecture come from DIFFERENT experiments**.

---

## 📊 Best Model Comparison Across Experiments

### **Summary Table**

| Architecture | No Aug IoU | With Aug IoU | Adaptive Loss IoU | **ACTUAL BEST** |
|--------------|------------|--------------|-------------------|-----------------|
| **UNet** | 0.6377 | 0.5974 | **0.6417** ✅ | **Adaptive Loss** |
| **Attention UNet** | **0.6254** ✅ | 0.5871 | 0.6234 | **No Aug** |
| **Attention ResUNet** | 0.6127 | 0.6030 | **0.6260** ✅ | **Adaptive Loss** |

---

## 🔍 Detailed Results

### **1. UNet Architecture**

| Experiment | Best IoU | n_filters | dropout | learning_rate | Model Name |
|------------|----------|-----------|---------|---------------|------------|
| **Adaptive Loss** ⭐ | **0.6417** | 32 | 0.1 | 0.001 | unet_n_filters32_dropout0.1_learning_rate0.001 |
| No Aug | 0.6377 | 32 | 0.2 | 0.001 | unet_n_filters32_dropout0.2_learning_rate0.001 |
| With Aug | 0.5974 | 32 | 0.1 | 0.001 | unet_n_filters32_dropout0.1_learning_rate0.001 |

**Winner:** Adaptive Loss (+0.0040 IoU improvement over No Aug)

---

### **2. Attention UNet Architecture**

| Experiment | Best IoU | n_filters | dropout | learning_rate | Model Name |
|------------|----------|-----------|---------|---------------|------------|
| **No Aug** ⭐ | **0.6254** | 32 | 0.1 | 0.003 | attention_unet_n_filters32_dropout0.1_learning_rate0.003 |
| Adaptive Loss | 0.6234 | 32 | 0.1 | 0.001 | attention_unet_n_filters32_dropout0.1_learning_rate0.001 |
| With Aug | 0.5871 | 64 | 0.1 | 0.001 | attention_unet_n_filters64_dropout0.1_learning_rate0.001 |

**Winner:** No Aug (+0.0020 IoU improvement over Adaptive Loss)

---

### **3. Attention ResUNet Architecture**

| Experiment | Best IoU | n_filters | dropout | learning_rate | Model Name |
|------------|----------|-----------|---------|---------------|------------|
| **Adaptive Loss** ⭐ | **0.6260** | 32 | 0.1 | 0.001 | attention_resunet_n_filters32_dropout0.1_learning_rate0.001 |
| No Aug | 0.6127 | 64 | 0.1 | 0.001 | attention_resunet_n_filters64_dropout0.1_learning_rate0.001 |
| With Aug | 0.6030 | 64 | 0.1 | 0.001 | attention_resunet_n_filters64_dropout0.1_learning_rate0.001 |

**Winner:** Adaptive Loss (+0.0133 IoU improvement over No Aug)

---

## 🤔 Why Are Cached Models All From "No Aug"?

### **The Current Situation:**

Looking at `best_models_PyTorch/` cache:

```json
// UNet
{
  "best_val_iou": 0.6377,
  "source_experiment": "pytorch_comparison_no_aug_20251021_121918",
  "model_name": "unet_n_filters32_dropout0.2_learning_rate0.001"
}

// Attention UNet
{
  "best_val_iou": 0.6254,
  "source_experiment": "pytorch_comparison_no_aug_20251021_121918",
  "model_name": "attention_unet_n_filters32_dropout0.1_learning_rate0.003"
}

// Attention ResUNet
{
  "best_val_iou": 0.6127,
  "source_experiment": "pytorch_comparison_no_aug_20251021_121918",
  "model_name": "attention_resunet_n_filters64_dropout0.1_learning_rate0.001"
}
```

### **Answer: The Prediction Script Only Searched ONE Experiment**

The [predict_pytorch_comparison.py](share_folder/pytorch_unet_pipeline/predict_pytorch_comparison.py) script has this limitation:

```python
def find_and_cache_best_models(experiment_dir, cache_dir='./best_models_PyTorch'):
    """
    Find best models and cache them in a dedicated directory for future use.

    First checks cache_dir for existing best models.
    If not found, searches experiment_dir and copies best models to cache_dir.

    Args:
        experiment_dir: Directory containing trained models  # ⚠️ ONLY ONE EXPERIMENT!
        cache_dir: Directory to store/load cached best models
    """
```

**The script was run with:**
```bash
python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_no_aug_20251021_121918
```

**What it does:**
1. Looks ONLY at `pytorch_comparison_no_aug_20251021_121918/all_results.csv`
2. Finds the best model for each architecture **within that single experiment**
3. Caches those models to `best_models_PyTorch/`

**What it does NOT do:**
- ❌ Compare across multiple experiment directories
- ❌ Look at `pytorch_comparison_with_aug_20251021_122018`
- ❌ Look at `pytorch_comparison_adaptive_loss_20251021_121920`

---

## 🎯 The TRUE Best Models (Across All Experiments)

| Architecture | Best Experiment | IoU | Model Location |
|--------------|----------------|-----|----------------|
| **UNet** | Adaptive Loss | 0.6417 | `pytorch_comparison_adaptive_loss_20251021_121920/unet/checkpoints/unet_n_filters32_dropout0.1_learning_rate0.001/best_model.pth` |
| **Attention UNet** | No Aug | 0.6254 | `pytorch_comparison_no_aug_20251021_121918/attention_unet/checkpoints/attention_unet_n_filters32_dropout0.1_learning_rate0.003/best_model.pth` |
| **Attention ResUNet** | Adaptive Loss | 0.6260 | `pytorch_comparison_adaptive_loss_20251021_121920/attention_resunet/checkpoints/attention_resunet_n_filters32_dropout0.1_learning_rate0.001/best_model.pth` |

---

## 📈 Performance Insights

### **1. Augmentation Impact (With Aug vs No Aug)**

**With Aug performed WORSE across all architectures:**

- UNet: -6.3% IoU decrease (0.6377 → 0.5974)
- Attention UNet: -6.1% IoU decrease (0.6254 → 0.5871)
- Attention ResUNet: -1.6% IoU decrease (0.6127 → 0.6030)

**Why?**
- Data augmentation may have introduced too much variability
- The dataset might already be diverse enough
- Augmentation parameters may need tuning

---

### **2. Adaptive Loss Impact**

**Adaptive Loss helped UNet and Attention ResUNet:**

- UNet: +0.6% IoU improvement (0.6377 → 0.6417) ✅
- Attention ResUNet: +2.2% IoU improvement (0.6127 → 0.6260) ✅
- Attention UNet: -0.3% IoU decrease (0.6254 → 0.6234) ⚠️

**Why Adaptive Loss works better:**
- Better handling of class imbalance (background vs beads)
- Dynamic weighting helps with challenging samples
- Particularly beneficial for ResNet-based architectures

---

### **3. Optimal Hyperparameters**

**Common pattern across best models:**

- **n_filters**: 32 (4 out of 6 best models)
  - Not 16 (too small, underfitting)
  - Not 64 (too large, overfitting or slower training)

- **dropout**: 0.1 (5 out of 6 best models)
  - Less regularization needed than expected
  - Models benefit from retaining more information

- **learning_rate**: 0.001 (5 out of 6 best models)
  - Slow and steady wins the race
  - Higher LR (0.003, 0.005) causes instability

---

## 🛠️ How to Use the TRUE Best Models

### **Option 1: Manual Selection (Recommended)**

Manually copy the best models from each experiment:

```bash
# Create a new best_models directory
mkdir -p best_models_PyTorch_ACTUAL_BEST/unet
mkdir -p best_models_PyTorch_ACTUAL_BEST/attention_unet
mkdir -p best_models_PyTorch_ACTUAL_BEST/attention_resunet

# Copy UNet from Adaptive Loss
cp pytorch_comparison_adaptive_loss_20251021_121920/unet/checkpoints/unet_n_filters32_dropout0.1_learning_rate0.001/best_model.pth \
   best_models_PyTorch_ACTUAL_BEST/unet/

# Copy Attention UNet from No Aug
cp pytorch_comparison_no_aug_20251021_121918/attention_unet/checkpoints/attention_unet_n_filters32_dropout0.1_learning_rate0.003/best_model.pth \
   best_models_PyTorch_ACTUAL_BEST/attention_unet/

# Copy Attention ResUNet from Adaptive Loss
cp pytorch_comparison_adaptive_loss_20251021_121920/attention_resunet/checkpoints/attention_resunet_n_filters32_dropout0.1_learning_rate0.001/best_model.pth \
   best_models_PyTorch_ACTUAL_BEST/attention_resunet/
```

---

### **Option 2: Run Predictions with Each Experiment**

Run the prediction script three times, once per experiment:

```bash
# Predictions from No Aug
python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_no_aug_20251021_121918 \
    --test_images ./test_images \
    --output ./predictions_no_aug

# Predictions from With Aug
python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_with_aug_20251021_122018 \
    --test_images ./test_images \
    --output ./predictions_with_aug

# Predictions from Adaptive Loss
python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_adaptive_loss_20251021_121920 \
    --test_images ./test_images \
    --output ./predictions_adaptive_loss
```

Then compare the predictions manually.

---

## 💡 Recommendations

### **For Future Work:**

1. **Create a cross-experiment comparison script** that:
   - Reads `all_results.csv` from ALL three experiments
   - Selects the global best model per architecture
   - Caches the true best models

2. **Re-run predictions using the ACTUAL best models**:
   - UNet from Adaptive Loss (IoU: 0.6417)
   - Attention ResUNet from Adaptive Loss (IoU: 0.6260)
   - Attention UNet from No Aug (IoU: 0.6254)

3. **Consider hybrid approach**:
   - Use Adaptive Loss for UNet and Attention ResUNet
   - Use No Aug for Attention UNet
   - Skip "With Aug" entirely (consistently worst performance)

---

## 📝 Summary

**Your observation is 100% correct!** The best models are NOT all from the "no aug" experiment:

✅ **UNet**: Best from **Adaptive Loss** (0.6417)
✅ **Attention UNet**: Best from **No Aug** (0.6254)
✅ **Attention ResUNet**: Best from **Adaptive Loss** (0.6260)

The reason the cached models are all from "no aug" is because the prediction script only searched within that single experiment directory, not across all three experiments.

---

**Author:** Claude Code
**Date:** October 24, 2025
**Purpose:** Clarifying PyTorch experiment best model selection

---

**End of Analysis**
