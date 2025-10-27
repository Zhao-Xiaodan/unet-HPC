# Keras UNet Pipeline: Training & Analysis

**⚠️ IMPORTANT: These experiments use Keras/TensorFlow, NOT PyTorch!**

This folder contains the training and analysis code for the three UNet architecture experiments:
- `unet_hyperparam_20251015_224125`
- `attention_unet_hyperparam_20251015_230149`
- `attention_resunet_hyperparam_20251015_235542`

---

## ⚠️ Framework Identification: Keras/TensorFlow

### **Confirmation: These experiments used KERAS, not PyTorch**

**Evidence from the code:**

1. **Import statements** in [train_unet_hyperparam.py:29-31](train_unet_hyperparam.py#L29-L31):
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
```

2. **Model file format**:
   - Keras saves models as `.h5` or `.keras` files
   - PyTorch saves models as `.pth` files
   - These experiments produced `.h5` files

3. **Training history**:
   - Keras: Saves training history to CSV files with format `*_history.csv`
   - PyTorch: Typically logs to tensorboard or custom loggers
   - These experiments have `*_history.csv` files in the `logs/` directory

4. **Results format**:
   - The `unet_results.csv`, `attention_unet_results.csv`, and `attention_resunet_results.csv` files contain Keras-specific metrics

---

## 📁 Files Included

### Training Scripts (Stage 1)

1. **[train_unet_hyperparam.py](train_unet_hyperparam.py)** (15.6 KB)
   - Trains standard UNet with hyperparameter search
   - Framework: **Keras/TensorFlow**
   - Hyperparameter grid: n_filters × dropout × learning_rate

2. **[train_attention_unet_hyperparam.py](train_attention_unet_hyperparam.py)** (15.7 KB)
   - Trains Attention UNet with hyperparameter search
   - Framework: **Keras/TensorFlow**
   - Same hyperparameter grid as UNet

3. **[train_attention_resunet_hyperparam.py](train_attention_resunet_hyperparam.py)** (15.7 KB)
   - Trains Attention ResUNet with hyperparameter search
   - Framework: **Keras/TensorFlow**
   - Same hyperparameter grid as UNet

### Analysis Script (Stage 2)

4. **[analyze_hyperparam_comparison.py](analyze_hyperparam_comparison.py)** (15.3 KB)
   - Analyzes and compares results from all three architectures
   - Generates visualizations and summary statistics
   - **Configured for the specific experiments**: Lines 24-42 hardcode the experiment directories

### Supporting Files

5. **[models_fixed.py](models_fixed.py)** (17 KB)
   - Keras model architecture definitions
   - Contains: `build_unet()`, `build_attention_unet()`, `build_attention_resunet()`
   - Custom layers: `RepeatElements` (replaces Lambda layers for serialization)

6. **[loss_functions_fixed.py](loss_functions_fixed.py)** (12 KB)
   - Keras custom loss functions
   - Contains: `BinaryFocalLoss`, `combined_dice_focal_loss`, `dice_coef`, `jacard_coef`

---

## 🏗️ The 3 UNet Architectures (Keras Implementation)

### 1. **Standard UNet**
```python
# Defined in models_fixed.py
build_unet(img_size=512, img_channels=3, n_filters=32, dropout=0.2, batch_norm=True)
```

**Structure:**
- Encoder: 4 downsampling blocks (Conv → BatchNorm → ReLU → Conv → MaxPool)
- Bottleneck: 2 conv layers
- Decoder: 4 upsampling blocks (UpSampling → Conv → Concat with skip → Conv)
- Output: Sigmoid activation

### 2. **Attention UNet**
```python
# Defined in models_fixed.py
build_attention_unet(img_size=512, img_channels=3, n_filters=32, dropout=0.2, batch_norm=True)
```

**Key Difference:**
- Adds **Attention Gates** before concatenating skip connections
- Attention gates filter out irrelevant background features
- Better at focusing on bead regions

### 3. **Attention ResUNet**
```python
# Defined in models_fixed.py
build_attention_resunet(img_size=512, img_channels=3, n_filters=32, dropout=0.2, batch_norm=True)
```

**Key Differences:**
- Uses **Residual Blocks** in encoder/decoder
- Includes **Attention Gates**
- Best gradient flow and highest accuracy

---

## 🚀 Usage

### Step 1: Train Models (Already Done)

These experiments were already run on Oct 15-16, 2025:

```bash
# UNet training (completed)
python train_unet_hyperparam.py
# Output: unet_hyperparam_20251015_224125/

# Attention UNet training (completed)
python train_attention_unet_hyperparam.py
# Output: attention_unet_hyperparam_20251015_230149/

# Attention ResUNet training (completed)
python train_attention_resunet_hyperparam.py
# Output: attention_resunet_hyperparam_20251015_235542/
```

**Hyperparameter Grid:**
- `n_filters`: [16, 32, 64]
- `dropout`: [0.1, 0.2, 0.3]
- `learning_rate`: [0.001, 0.003, 0.005]
- `batch_norm`: [True]

**Total experiments per architecture:** 3 × 3 × 3 = **27 models each**

---

### Step 2: Analyze Results

```bash
python analyze_hyperparam_comparison.py
```

**Output Directory:** `./hyperparam_comparison_report/`

**Generated Files:**
```
hyperparam_comparison_report/
├── summary_statistics.csv                    # Overall stats per architecture
├── best_models_summary.csv                   # Best model per architecture
├── fig1_best_iou_comparison.png              # Bar chart of best IoU
├── fig2_iou_distribution.png                 # Box plot of IoU distributions
├── fig3_hyperparameter_heatmaps.png          # Heatmaps: n_filters × learning_rate
├── fig4_dropout_effect.png                   # Line plot: dropout vs IoU
├── fig5_learning_rate_effect.png             # Line plot: learning_rate vs IoU
├── fig6_n_filters_effect.png                 # Bar chart: n_filters vs IoU
└── fig7_convergence_epoch.png                # Box plot: convergence speed
```

---

## 📊 Experiment Results Summary

### Best Models Found (from `analyze_hyperparam_comparison.py`)

Based on the analysis script configuration, here's what it looks for:

```python
CONFIG = {
    'unet': {
        'results_csv': './unet_hyperparam_20251015_224125/unet_results.csv',
        'dir': Path('./unet_hyperparam_20251015_224125'),
    },
    'attention_unet': {
        'results_csv': './attention_unet_hyperparam_20251015_230149/attention_unet_results.csv',
        'dir': Path('./attention_unet_hyperparam_20251015_230149'),
    },
    'attention_resunet': {
        'results_csv': './attention_resunet_hyperparam_20251015_235542/attention_resunet_results.csv',
        'dir': Path('./attention_resunet_hyperparam_20251015_235542'),
    }
}
```

**Metrics tracked:**
- `best_val_iou`: Best validation Intersection over Union
- `best_val_dice`: Best validation Dice coefficient
- `best_epoch`: Epoch where best validation IoU was achieved
- `final_val_iou`: Validation IoU at final epoch (may show overfitting)

---

## 🔍 Code Deep Dive

### Training Configuration (from `train_unet_hyperparam.py`)

```python
CONFIG = {
    # Data
    'images_dir': './dataset_shrunk_masks/images/',
    'masks_dir': './dataset_shrunk_masks/masks/',
    'train_val_split': 0.8,  # 80% train, 20% validation

    # Model
    'img_size': 512,
    'img_channels': 3,  # RGB images (NOT grayscale!)

    # Training
    'epochs': 100,
    'batch_size': 4,
    'early_stopping_patience': 20,
    'reduce_lr_patience': 10,

    # Loss function
    'loss': 'binary_focal_loss',  # BinaryFocalLoss
    'focal_gamma': 2,
    'focal_alpha': 0.25,
}
```

**Key Differences from PyTorch Pipeline:**
1. **Input channels**: Uses **RGB (3 channels)**, not grayscale
2. **Loss function**: Uses `BinaryFocalLoss`, not `AdaptiveBGDiceLoss`
3. **Framework**: Keras callbacks (ModelCheckpoint, EarlyStopping) vs PyTorch manual loops

---

### Analysis Functions (from `analyze_hyperparam_comparison.py`)

#### 1. **Find Best Models** (Lines 72-89)
```python
def find_best_models(results):
    """Find best model for each architecture."""
    best_models = {}

    for arch, df in results.items():
        best_idx = df['best_val_iou'].idxmax()  # Find row with max IoU
        best_model = df.loc[best_idx]
        best_models[arch] = best_model

        print(f"\n{CONFIG[arch]['name']} Best Model:")
        print(f"  IoU: {best_model['best_val_iou']:.4f}")
        print(f"  Dice: {best_model['best_val_dice']:.4f}")
        print(f"  Filters: {best_model['n_filters']}")
        print(f"  Dropout: {best_model['dropout']}")
        print(f"  Learning Rate: {best_model['learning_rate']}")
        print(f"  Best Epoch: {best_model['best_epoch']}")

    return best_models
```

**Purpose:** Identifies which hyperparameter combination achieved the highest validation IoU for each architecture.

---

#### 2. **Hyperparameter Heatmaps** (Lines 175-198)
```python
def plot_hyperparameter_heatmaps(results):
    """Heatmaps showing IoU across hyperparameter combinations."""
    for arch, df in results.items():
        # Create pivot table: n_filters × learning_rate, averaged over dropout
        pivot = df.pivot_table(
            values='best_val_iou',
            index='n_filters',
            columns='learning_rate',
            aggfunc='mean'
        )

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0.2, vmax=0.55)
```

**Purpose:** Visualizes which combinations of `n_filters` and `learning_rate` work best, averaging over `dropout` values.

---

#### 3. **Dropout Effect Analysis** (Lines 200-231)
```python
def plot_dropout_effect(df_all):
    """Line plot showing effect of dropout on performance."""
    for arch in ['unet', 'attention_unet', 'attention_resunet']:
        df_arch = df_all[df_all['architecture'] == CONFIG[arch]['name']]

        # Group by dropout and compute mean IoU
        dropout_means = df_arch.groupby('dropout')['best_val_iou'].mean()
        dropout_stds = df_arch.groupby('dropout')['best_val_iou'].std()

        ax.plot(dropout_means.index, dropout_means.values, marker='o')
        ax.fill_between(dropout_means.index,
                       dropout_means.values - dropout_stds.values,
                       dropout_means.values + dropout_stds.values,
                       alpha=0.2)
```

**Purpose:** Shows how dropout rate (0.1, 0.2, 0.3) affects model performance across architectures.

---

## 🆚 Keras vs PyTorch: Key Differences

| Aspect | Keras (These Experiments) | PyTorch (Other Folder) |
|--------|---------------------------|------------------------|
| **Framework** | TensorFlow + Keras | PyTorch |
| **Model Files** | `.h5`, `.keras` | `.pth` |
| **Input Channels** | RGB (3 channels) | Grayscale (1 channel) |
| **Loss Function** | `BinaryFocalLoss` | `AdaptiveBGDiceLoss` or `BinaryFocalLoss` |
| **Training Loop** | Keras `model.fit()` with callbacks | Manual PyTorch loop with `optimizer.step()` |
| **Callbacks** | `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau` | Manual implementation |
| **History Logging** | Automatic `.csv` files | Custom logging or TensorBoard |
| **Model Definition** | Functional API (`keras.Model`) | Class-based (`nn.Module`) |
| **Preprocessing** | Not explicitly shown | Percentile normalization |

---

## 🧪 Key Insights

### Why Use Keras?

**Advantages:**
1. **Simpler API**: `model.fit()` handles training loop automatically
2. **Built-in callbacks**: EarlyStopping, ModelCheckpoint work out-of-the-box
3. **Easy serialization**: Save/load models with `model.save()`, `keras.models.load_model()`
4. **Quick prototyping**: Less boilerplate code

**Disadvantages:**
1. **Less flexibility**: Harder to customize training loops
2. **Debugging**: Harder to trace through graph execution
3. **Research adoption**: PyTorch is more popular in research community

### Why Use PyTorch?

**Advantages:**
1. **Dynamic graphs**: Easier debugging and visualization
2. **Flexibility**: Full control over training loop
3. **Research standard**: Most papers provide PyTorch implementations
4. **GPU efficiency**: More control over memory management

**Disadvantages:**
1. **More boilerplate**: Need to manually write training loops
2. **Callbacks**: Need to implement early stopping, checkpointing manually
3. **Steeper learning curve**: More concepts to learn

---

## 📝 How to Read the Results

### Understanding the CSV Files

**Example from `unet_results.csv`:**
```csv
model,experiment_name,best_epoch,best_val_iou,best_val_dice,final_val_iou,final_val_dice,n_filters,dropout,batch_norm,learning_rate
unet,unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001,68,0.4508,0.6129,0.4296,0.5933,16,0.1,True,0.001
unet,unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003,96,0.4680,0.6313,0.3926,0.5535,16,0.1,True,0.003
```

**Column Meanings:**
- `best_epoch`: Epoch where validation IoU peaked (before early stopping)
- `best_val_iou`: Highest validation IoU achieved
- `final_val_iou`: IoU at epoch 100 (or when early stopping triggered)
- `best_val_iou > final_val_iou`: Indicates overfitting after best epoch

---

## 🔧 Requirements

```bash
# Keras/TensorFlow dependencies
tensorflow >= 2.10.0
keras >= 2.10.0
numpy >= 1.21.0
pandas >= 1.3.0
opencv-python >= 4.5.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
pillow >= 9.0.0
scikit-learn >= 1.0.0
```

---

## 🐛 Troubleshooting

### Keras model loading issues

If you encounter serialization errors when loading models:

```python
# Use custom_objects when loading
from loss_functions_fixed import BinaryFocalLoss
from models_fixed import RepeatElements

model = keras.models.load_model(
    'best_model.h5',
    custom_objects={
        'BinaryFocalLoss': BinaryFocalLoss,
        'RepeatElements': RepeatElements
    }
)
```

### Missing experiment directories

The analysis script expects these directories in the parent folder:
- `../unet_hyperparam_20251015_224125/`
- `../attention_unet_hyperparam_20251015_230149/`
- `../attention_resunet_hyperparam_20251015_235542/`

If they're missing, update paths in `analyze_hyperparam_comparison.py` lines 24-42.

---

## 📧 Support

For questions about:
- **Training**: See comments in `train_*_hyperparam.py`
- **Analysis**: See comments in `analyze_hyperparam_comparison.py`
- **Model architectures**: See `models_fixed.py`
- **Loss functions**: See `loss_functions_fixed.py`

---

## 📄 Summary

**Framework:** Keras/TensorFlow (NOT PyTorch)

**Experiments:**
1. `unet_hyperparam_20251015_224125` - Standard UNet
2. `attention_unet_hyperparam_20251015_230149` - Attention UNet
3. `attention_resunet_hyperparam_20251015_235542` - Attention ResUNet

**Purpose:** Hyperparameter search to find optimal:
- Number of filters (16, 32, 64)
- Dropout rate (0.1, 0.2, 0.3)
- Learning rate (0.001, 0.003, 0.005)

**Total models trained:** 27 × 3 architectures = **81 models**

**Analysis:** `analyze_hyperparam_comparison.py` generates comprehensive comparison visualizations

---

**Author:** Claude Code
**Date:** October 2025
**Purpose:** Research and educational use

---

**End of Documentation**
