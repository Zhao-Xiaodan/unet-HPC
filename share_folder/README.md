# UNet Training & Analysis Pipelines

This folder contains **two separate UNet pipelines** using different deep learning frameworks:

---

## 📂 Folder Structure

```
./share_folder/
├── README.md (this file)
├── keras_unet_pipeline/          ⚠️ Keras/TensorFlow
│   ├── README.md
│   ├── train_unet_hyperparam.py
│   ├── train_attention_unet_hyperparam.py
│   ├── train_attention_resunet_hyperparam.py
│   ├── analyze_hyperparam_comparison.py
│   ├── models_fixed.py
│   └── loss_functions_fixed.py
│
└── pytorch_unet_pipeline/         ⭐ PyTorch
    ├── README.md
    ├── train_pytorch_comparison_no_aug.py
    ├── train_pytorch_comparison_with_aug.py
    ├── train_pytorch_comparison_adaptive_loss.py
    ├── predict_pytorch_comparison.py
    ├── density_analysis_pytorch_comparison.py
    └── analyze_pytorch_comparison.py
```

---

## 🔍 Which Pipeline Should I Use?

### **Keras Pipeline** ([keras_unet_pipeline/](keras_unet_pipeline/))

**For analyzing these specific experiments:**
- `unet_hyperparam_20251015_224125`
- `attention_unet_hyperparam_20251015_230149`
- `attention_resunet_hyperparam_20251015_235542`

**Characteristics:**
- ⚠️ Framework: **Keras/TensorFlow**
- Input: **RGB images (3 channels)**
- Loss: `BinaryFocalLoss`
- Training: Keras callbacks (`ModelCheckpoint`, `EarlyStopping`)
- Output: `.h5` model files
- Purpose: Hyperparameter search (27 models per architecture)

**Use this if:**
- You need to analyze the Oct 15-16, 2025 experiments
- You have existing Keras models to evaluate
- You prefer Keras' simpler API

---

### **PyTorch Pipeline** ([pytorch_unet_pipeline/](pytorch_unet_pipeline/)) ⭐ **RECOMMENDED**

**For new experiments and predictions:**

**Characteristics:**
- ⭐ Framework: **PyTorch**
- Input: **Grayscale images (1 channel)**
- Loss: `BinaryFocalLoss` or `AdaptiveBGDiceLoss`
- Training: Manual PyTorch loops with custom callbacks
- Output: `.pth` model files
- Purpose: Complete pipeline (train → predict → analyze density)

**Use this if:**
- You want to run new training experiments
- You need to generate predictions on test images
- You want to perform density analysis on predictions
- You prefer PyTorch's flexibility and research-friendly API

---

## 🆚 Framework Comparison

| Feature | Keras Pipeline | PyTorch Pipeline |
|---------|---------------|------------------|
| **Framework** | TensorFlow/Keras | PyTorch |
| **Experiments** | Oct 15-16, 2025 | Oct 21-22, 2025 |
| **Input Channels** | 3 (RGB) | 1 (Grayscale) |
| **Model Files** | `.h5`, `.keras` | `.pth` |
| **Loss Function** | BinaryFocalLoss | BinaryFocalLoss or AdaptiveBGDiceLoss |
| **Training API** | `model.fit()` | Manual loop |
| **Prediction Script** | ❌ Not included | ✅ `predict_pytorch_comparison.py` |
| **Density Analysis** | ❌ Not included | ✅ `density_analysis_pytorch_comparison.py` |
| **Preprocessing** | Basic RGB loading | Percentile normalization |
| **Architectures** | 3 (UNet, AttUNet, AttResUNet) | 3 (same) |

---

## 📚 Detailed Documentation

### Keras Pipeline

**See:** [keras_unet_pipeline/README.md](keras_unet_pipeline/README.md)

**Key Scripts:**
1. **Training:**
   - `train_unet_hyperparam.py` - Standard UNet
   - `train_attention_unet_hyperparam.py` - Attention UNet
   - `train_attention_resunet_hyperparam.py` - Attention ResUNet

2. **Analysis:**
   - `analyze_hyperparam_comparison.py` - Compare all 3 architectures

3. **Supporting:**
   - `models_fixed.py` - Keras model definitions
   - `loss_functions_fixed.py` - Custom loss functions

**Workflow:**
```
Training (Keras) → Results CSV → Analysis Script → Visualizations
```

---

### PyTorch Pipeline ⭐

**See:** [pytorch_unet_pipeline/README.md](pytorch_unet_pipeline/README.md)

**Key Scripts:**
1. **Training:**
   - `train_pytorch_comparison_no_aug.py` ⭐ Recommended
   - `train_pytorch_comparison_with_aug.py` - With augmentation
   - `train_pytorch_comparison_adaptive_loss.py` - Advanced loss

2. **Prediction:**
   - `predict_pytorch_comparison.py` - Generate predictions from trained models

3. **Analysis:**
   - `density_analysis_pytorch_comparison.py` - Analyze bead density
   - `analyze_pytorch_comparison.py` - Compare training results

**Workflow:**
```
Training (PyTorch) → Model Checkpoints (.pth)
                  ↓
Test Images → Prediction Script → Prediction Masks (.png)
                                ↓
                         Density Analysis → Boxplots & Visualizations
```

---

## 🎯 Quick Start Guide

### To Analyze Oct 15-16 Keras Experiments

```bash
cd share_folder/keras_unet_pipeline
python analyze_hyperparam_comparison.py
```

**Output:** Visualizations comparing UNet, Attention UNet, and Attention ResUNet

---

### To Train New PyTorch Models

```bash
cd share_folder/pytorch_unet_pipeline

# Option 1: No augmentation (recommended for comparison)
python train_pytorch_comparison_no_aug.py

# Option 2: With augmentation
python train_pytorch_comparison_with_aug.py

# Option 3: Adaptive loss
python train_pytorch_comparison_adaptive_loss.py
```

---

### To Generate Predictions (PyTorch Only)

```bash
cd share_folder/pytorch_unet_pipeline

python predict_pytorch_comparison.py \
    --experiment ../pytorch_comparison_no_aug_YYYYMMDD_HHMMSS \
    --test_images ../test_images \
    --output ../predictions
```

---

### To Analyze Density (PyTorch Only)

```bash
cd share_folder/pytorch_unet_pipeline

python density_analysis_pytorch_comparison.py \
    --predictions ../predictions \
    --test_images ../test_images \
    --output ../density_results
```

---

## 🧪 Key Insights

`★ Insight ─────────────────────────────────────`
**Why Two Different Pipelines?**

1. **Historical Reason:**
   - Keras pipeline: Early experiments (Oct 15-16)
   - PyTorch pipeline: Later refinements (Oct 21-22)

2. **Different Use Cases:**
   - Keras: Quick hyperparameter search with built-in callbacks
   - PyTorch: Full control over training + prediction + analysis

3. **Input Differences:**
   - Keras: RGB images (3 channels) - may include unnecessary color info
   - PyTorch: Grayscale (1 channel) - more efficient for microscopy

4. **Pipeline Completeness:**
   - Keras: Training + Analysis only
   - PyTorch: Training + Prediction + Density Analysis
`─────────────────────────────────────────────────`

---

## 🔧 Requirements

### Keras Pipeline
```bash
tensorflow >= 2.10.0
keras >= 2.10.0
numpy, pandas, opencv-python, matplotlib, seaborn
```

### PyTorch Pipeline
```bash
torch >= 1.13.0
torchvision >= 0.14.0
numpy, pandas, opencv-python, matplotlib, seaborn
```

---

## 📊 Expected Outputs

### Keras Pipeline

**Training Output:**
```
unet_hyperparam_20251015_224125/
├── unet_results.csv                  # Summary of all 27 experiments
├── logs/
│   └── *_history.csv                 # Training history per model
└── models/
    └── best_model_*.h5               # Best model per experiment
```

**Analysis Output:**
```
hyperparam_comparison_report/
├── summary_statistics.csv
├── best_models_summary.csv
├── fig1_best_iou_comparison.png
├── fig2_iou_distribution.png
├── fig3_hyperparameter_heatmaps.png
├── fig4_dropout_effect.png
├── fig5_learning_rate_effect.png
├── fig6_n_filters_effect.png
└── fig7_convergence_epoch.png
```

---

### PyTorch Pipeline

**Training Output:**
```
pytorch_comparison_no_aug_YYYYMMDD_HHMMSS/
├── all_results.csv                   # Summary of all experiments
├── unet/
│   └── checkpoints/
│       └── unet_f32_d0.2_lr0.0001/
│           └── best_model.pth
├── attention_unet/
│   └── checkpoints/...
└── attention_resunet/
    └── checkpoints/...
```

**Prediction Output:**
```
predictions/
├── unet/
│   ├── image_10x_pred.png
│   └── ...
├── attention_unet/
│   └── ...
└── attention_resunet/
    └── ...
```

**Density Analysis Output:**
```
density_results/
├── density_results_tile_level.csv
├── density_results_image_summary.csv
├── density_boxplot_full_range__threshold_0.5.png
├── density_boxplot_low_dilution_range__threshold_0.5.png
├── density_boxplot_unet_full_range_threshold_0.5.png
├── density_boxplot_attention_unet_full_range_threshold_0.5.png
├── density_boxplot_attention_resunet_full_range_threshold_0.5.png
└── representative_tiles_4panel/
    └── tiles_4panel_*.png
```

---

## 🐛 Common Issues

### "Cannot find experiment directory"

**Keras Pipeline:**
- Update paths in `analyze_hyperparam_comparison.py` lines 24-42
- Ensure experiment directories are in parent folder

**PyTorch Pipeline:**
- Use correct experiment directory name with `--experiment` flag
- Experiment names include timestamp: `pytorch_comparison_no_aug_YYYYMMDD_HHMMSS`

---

### "Model loading error"

**Keras:**
```python
# Need custom_objects for BinaryFocalLoss and RepeatElements
model = keras.models.load_model('model.h5', custom_objects={...})
```

**PyTorch:**
```python
# Need to instantiate model first, then load state_dict
model = UNet(...)
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 📝 Summary

### **Answer to Your Question:**

**⚠️ The experiments you asked about used KERAS/TensorFlow, NOT PyTorch:**

- `unet_hyperparam_20251015_224125` → **Keras**
- `attention_unet_hyperparam_20251015_230149` → **Keras**
- `attention_resunet_hyperparam_20251015_235542` → **Keras**

**Evidence:**
- Training scripts import `tensorflow` and `keras` (line 29-31 in training scripts)
- Model files are `.h5` format (Keras), not `.pth` (PyTorch)
- Training history saved to CSV files (Keras pattern)

**For PyTorch experiments, see:**
- [pytorch_unet_pipeline/](pytorch_unet_pipeline/) folder
- These are more recent experiments from Oct 21-22, 2025

---

**Author:** Claude Code
**Date:** October 2025
**Purpose:** Complete documentation of both UNet pipelines

---

**End of Documentation**
