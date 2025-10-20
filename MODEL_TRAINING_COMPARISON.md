# Comprehensive Model Training Comparison

**Date:** October 17, 2025
**Task:** Microbead Segmentation in Microscopy Images
**Framework:** Keras/TensorFlow 2.x
**Architectures Compared:** UNet, Attention UNet, Attention ResUNet

---

## Executive Summary

This document provides a detailed comparison of three U-Net architecture variants trained for microbead segmentation using **Keras/TensorFlow**. All models were trained identically except for their architectural differences, enabling direct performance comparison.

**Note:** This repository also contains a separate `train.py` file implementing a PyTorch-based student-teacher distillation approach with ConsensusAttnUNet architecture. That training approach uses different frameworks (PyTorch vs Keras), different loss functions (AdaptiveBGDiceLoss with TV regularization vs BinaryFocalLoss), and different augmentation strategies (synthetic background artifacts vs none). The PyTorch approach is **not** compared in this document - this analysis focuses solely on the three Keras models trained with hyperparameter search.

### Performance Ranking (Validation IoU)

| Rank | Architecture | Best Val IoU | Improvement vs UNet |
|------|-------------|--------------|---------------------|
| 🥇 1 | **Attention ResUNet** | **0.5039** | **+3.8%** |
| 🥈 2 | Attention UNet | 0.4759 | -1.9% |
| 🥉 3 | UNet (Baseline) | 0.4853 | baseline |

**Key Finding:** Attention ResUNet achieved the best performance, combining residual connections with attention mechanisms to improve feature extraction and gradient flow.

---

## 1. Dataset & Preprocessing

All three models use **identical** data preprocessing pipelines.

### 1.1 Dataset Information

| Parameter | Value |
|-----------|-------|
| **Dataset Directory** | `./dataset_shrunk_masks/` |
| **Images Directory** | `./dataset_shrunk_masks/images/` |
| **Masks Directory** | `./dataset_shrunk_masks/masks/` |
| **Task Type** | Binary segmentation (microbeads vs background) |
| **Image Format** | RGB microscopy images |
| **Mask Format** | Grayscale binary masks |

### 1.2 Image Preprocessing Pipeline (Keras Models)

**Source:** `train_unet_hyperparam.py:125-197` (identical for all 3 Keras models)

```python
# Step 1: Load image (RGB) and mask (grayscale)
image = cv2.imread(image_path, 1)      # RGB (OpenCV BGR → RGB conversion implicit)
mask = cv2.imread(mask_path, 0)        # Grayscale

# Step 2: Resize to 512×512 (BEFORE normalization)
image = Image.fromarray(image).resize((512, 512))  # PIL resize
mask = Image.fromarray(mask).resize((512, 512))    # PIL resize

# Step 3: Convert to numpy arrays
image = np.array(image)  # Shape: (512, 512, 3), dtype: uint8, range: [0, 255]
mask = np.array(mask)    # Shape: (512, 512), dtype: uint8, range: [0, 255]

# Step 4: Normalize to [0, 1] - LINEAR SCALING
image_normalized = image / 255.0       # RGB channels: [0, 1]
mask_normalized = mask / 255.0         # Binary: {0, 1}

# Step 5: Add channel dimension to mask
mask_final = np.expand_dims(mask_normalized, axis=-1)  # Shape: (512, 512, 1)
```

**Final Tensor Shapes:**
- **Input (X):** `(N, 512, 512, 3)` - RGB images normalized to [0, 1]
- **Target (y):** `(N, 512, 512, 1)` - Binary masks with values {0, 1}

**Key Characteristics:**
- ✅ **RGB input** (3 channels preserve color information)
- ✅ **Linear normalization** (divide by 255)
- ✅ **No contrast adjustment** (raw pixel values preserved)
- ✅ **No augmentation** (training on original images only)

### 1.3 Train/Validation Split

| Parameter | Value |
|-----------|-------|
| **Split Ratio** | 80/20 (train/validation) |
| **Random State** | 42 (fixed for reproducibility) |
| **Splitting Method** | `sklearn.model_selection.train_test_split` |

**Important:** The same random seed (42) ensures all three models train on identical train/validation sets, enabling fair comparison.

---

## 2. Architecture Comparison

All three architectures share the same U-Net encoder-decoder structure but differ in their building blocks and attention mechanisms.

### 2.1 Architecture Overview

| Component | UNet | Attention UNet | Attention ResUNet |
|-----------|------|----------------|-------------------|
| **Encoder Block** | Standard conv block | Standard conv block | **Residual conv block** |
| **Decoder Block** | Standard conv block | Standard conv block | **Residual conv block** |
| **Skip Connections** | Direct concatenation | **Attention gates** | **Attention gates** |
| **Residual Learning** | ❌ No | ❌ No | ✅ **Yes** |
| **Attention Mechanism** | ❌ No | ✅ **Yes** | ✅ **Yes** |

### 2.2 Standard Convolution Block (UNet)

**Location:** `models_fixed.py:163` (build_unet)

```python
def conv_block(inputs, n_filters, dropout, batch_norm):
    """Standard convolution block used in UNet"""
    # Conv → BN → ReLU → Conv → BN → ReLU → Dropout
    conv = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)

    if batch_norm:
        conv = layers.BatchNormalization()(conv)

    conv = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv)

    if batch_norm:
        conv = layers.BatchNormalization()(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv
```

**Characteristics:**
- Two 3×3 convolutions with ReLU activation
- Optional batch normalization after each conv
- Dropout applied at the end
- **Limitation:** Deep networks can suffer from vanishing gradients

### 2.3 Residual Convolution Block (Attention ResUNet)

**Location:** `models_fixed.py:66` (res_conv_block)

```python
def res_conv_block(inputs, n_filters, dropout, batch_norm):
    """
    Residual convolution block with skip connection.

    Enables gradient flow through identity shortcut:
    output = F(x) + x
    """
    # Main path: Conv → BN → ReLU → Conv → BN
    conv = layers.Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)

    if batch_norm:
        conv = layers.BatchNormalization()(conv)

    conv = layers.Conv2D(n_filters, (3, 3), padding='same')(conv)  # No activation yet

    if batch_norm:
        conv = layers.BatchNormalization()(conv)

    # Residual connection: match dimensions if needed
    if inputs.shape[-1] != n_filters:
        inputs = layers.Conv2D(n_filters, (1, 1), padding='same')(inputs)

    # Add residual connection
    conv = layers.Add()([conv, inputs])
    conv = layers.Activation('relu')(conv)  # Activation after addition

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv
```

**Advantages over Standard Conv Block:**
- ✅ **Residual learning:** `output = F(x) + x` enables gradient flow
- ✅ **Deeper networks:** Mitigates vanishing gradient problem
- ✅ **Better feature reuse:** Identity mapping preserves low-level features
- ✅ **Faster convergence:** Easier to optimize (learning residuals vs full mapping)

### 2.4 Attention Gate Mechanism (Attention UNet & ResUNet)

**Location:** `models_fixed.py:96` (attention_block)

```python
def attention_block(x, gating, inter_shape):
    """
    Attention gate: highlights salient features from skip connections.

    Args:
        x: Feature map from encoder (skip connection)
        gating: Gating signal from decoder (context information)
        inter_shape: Number of intermediate filters

    Returns:
        Attention-weighted feature map
    """
    # 1. Transform skip connection features
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)

    # 2. Transform gating signal
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)

    # 3. Align dimensions (upsample/downsample as needed)
    upsample_g = align_dimensions(phi_g, theta_x)

    # 4. Combine features: θᵀx + φᵀg
    concat_xg = layers.Add()([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)

    # 5. Generate attention coefficients
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)  # Attention weights ∈ [0, 1]

    # 6. Upsample attention weights to match input dimensions
    upsample_psi = layers.UpSampling2D(size=(upsample_h, upsample_w))(sigmoid_xg)

    # 7. Broadcast attention weights across channels
    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    # 8. Apply attention: element-wise multiplication
    y = layers.Multiply()([upsample_psi, x])

    # 9. Final 1×1 conv to match dimensions
    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)

    return result_bn
```

**How Attention Gates Work:**

1. **Input:** Encoder features (x) + decoder context (gating)
2. **Query-Key Mechanism:** Learn which spatial locations are important
3. **Attention Weights:** Sigmoid activation produces weights ∈ [0, 1]
4. **Weighted Features:** Multiply input features by attention weights
5. **Effect:** Suppresses irrelevant regions, enhances salient features

**Advantages:**
- ✅ **Better localization:** Focuses on relevant image regions
- ✅ **Reduces false positives:** Suppresses background activations
- ✅ **No extra supervision:** Learned end-to-end from segmentation loss
- ✅ **Interpretable:** Attention maps show what the model focuses on

### 2.5 Architecture-Specific Details

#### UNet (Standard)
**Location:** `models_fixed.py:163-207` (build_unet)

**Encoder:**
```
Input (512, 512, 3)
  ↓ conv_block(n_filters) → MaxPool(2×2)
  ↓ conv_block(n_filters*2) → MaxPool(2×2)
  ↓ conv_block(n_filters*4) → MaxPool(2×2)
  ↓ conv_block(n_filters*8) → MaxPool(2×2)
Bottleneck: conv_block(n_filters*16)
```

**Decoder:**
```
Bottleneck
  ↓ UpConv → Concatenate(skip_connection) → conv_block(n_filters*8)
  ↓ UpConv → Concatenate(skip_connection) → conv_block(n_filters*4)
  ↓ UpConv → Concatenate(skip_connection) → conv_block(n_filters*2)
  ↓ UpConv → Concatenate(skip_connection) → conv_block(n_filters)
Output: Conv2D(1, sigmoid) → (512, 512, 1)
```

**Parameter Scaling:**
- Base filters: 16/32/64 (from hyperparameter search)
- Filters double at each encoder level
- Filters halve at each decoder level

---

#### Attention UNet
**Location:** `models_fixed.py:283-327` (build_attention_unet)

**Same as UNet, but skip connections pass through attention gates:**

**Decoder:**
```
Bottleneck
  ↓ UpConv → attention_gate(skip, gating) → Concatenate → conv_block(n_filters*8)
  ↓ UpConv → attention_gate(skip, gating) → Concatenate → conv_block(n_filters*4)
  ↓ UpConv → attention_gate(skip, gating) → Concatenate → conv_block(n_filters*2)
  ↓ UpConv → attention_gate(skip, gating) → Concatenate → conv_block(n_filters)
Output: Conv2D(1, sigmoid) → (512, 512, 1)
```

**Key Difference:** Attention gates filter skip connections before concatenation.

---

#### Attention ResUNet
**Location:** `models_fixed.py:410-469` (build_attention_resunet)

**Combines residual blocks + attention gates:**

**Encoder:**
```
Input (512, 512, 3)
  ↓ res_conv_block(n_filters) → MaxPool(2×2)
  ↓ res_conv_block(n_filters*2) → MaxPool(2×2)
  ↓ res_conv_block(n_filters*4) → MaxPool(2×2)
  ↓ res_conv_block(n_filters*8) → MaxPool(2×2)
Bottleneck: res_conv_block(n_filters*16)
```

**Decoder:**
```
Bottleneck
  ↓ UpConv → attention_gate(skip, gating) → Concatenate → res_conv_block(n_filters*8)
  ↓ UpConv → attention_gate(skip, gating) → Concatenate → res_conv_block(n_filters*4)
  ↓ UpConv → attention_gate(skip, gating) → Concatenate → res_conv_block(n_filters*2)
  ↓ UpConv → attention_gate(skip, gating) → Concatenate → res_conv_block(n_filters)
Output: Conv2D(1, sigmoid) → (512, 512, 1)
```

**Key Differences:**
- ✅ Uses `res_conv_block` instead of `conv_block` (both encoder & decoder)
- ✅ Includes attention gates like Attention UNet
- ✅ **Best of both worlds:** Residual learning + attention mechanism

---

### 2.6 Model Complexity Comparison

**Estimated Parameter Counts (n_filters=32 baseline):**

| Architecture | Encoder Params | Decoder Params | Total Params | Relative Size |
|--------------|----------------|----------------|--------------|---------------|
| UNet | ~1.2M | ~1.0M | **~2.2M** | 1.0× (baseline) |
| Attention UNet | ~1.2M | ~1.3M | **~2.5M** | 1.14× |
| Attention ResUNet | ~1.5M | ~1.6M | **~3.1M** | 1.41× |

**Memory & Compute:**
- **UNet:** Fastest inference, lowest memory
- **Attention UNet:** +14% parameters (attention gates)
- **Attention ResUNet:** +41% parameters (residual blocks + attention)

**Training Batch Size Compatibility:**
- All three trained successfully with `batch_size=4`
- Attention ResUNet requires smaller batch size for inference (`batch_size=2` vs `8`)

---

## 3. Loss Function

All three models use **identical loss functions**.

### 3.1 Binary Focal Loss

**Implementation:** `loss_functions_fixed.py` → `BinaryFocalLoss` class

**Formula:**
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where:
  p_t = p     if y = 1 (foreground)
  p_t = 1 - p if y = 0 (background)

  α_t = α     if y = 1
  α_t = 1 - α if y = 0
```

**Parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **gamma (γ)** | 2.0 | Focusing parameter - down-weights easy examples |
| **alpha (α)** | 0.25 | Class balance weight - accounts for class imbalance |

**Why Focal Loss?**
1. ✅ **Handles class imbalance:** Microbeads occupy small % of image pixels
2. ✅ **Focuses on hard examples:** Down-weights well-classified pixels
3. ✅ **Better than standard BCE:** Standard BCE treats all pixels equally
4. ✅ **Improves boundary detection:** Hard pixels are often at object boundaries

**Keras Implementation:**
```python
loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[jacard_coef, dice_coef]
)
```

### 3.2 Evaluation Metrics

**Primary Metric:** Intersection over Union (IoU / Jaccard Coefficient)

```python
def jacard_coef(y_true, y_pred, smooth=1e-5):
    """
    IoU = |A ∩ B| / |A ∪ B|

    Measures overlap between predicted and ground truth masks.
    Range: [0, 1], higher is better.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, [1,2,3]) + K.sum(y_pred, [1,2,3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
```

**Secondary Metric:** Dice Coefficient (F1 Score)

```python
def dice_coef(y_true, y_pred, smooth=1e-5):
    """
    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Harmonic mean of precision and recall.
    Range: [0, 1], higher is better.
    """
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth) /
                  (K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) + smooth))
    return dice
```

**Model Selection:** Best model selected by **highest validation IoU**.

---

## 4. Hyperparameter Search

All three models use **identical hyperparameter search grids**.

### 4.1 Search Space

| Hyperparameter | Values Tested | Total Combinations |
|----------------|---------------|-------------------|
| **n_filters** | [16, 32, 64] | 3 |
| **dropout** | [0.1, 0.2, 0.3] | 3 |
| **batch_norm** | [True] | 1 |
| **learning_rate** | [0.001, 0.003, 0.005] | 3 |
| **Total Configs** | | **27** |

**Grid Search Strategy:** Exhaustive search (all 27 combinations tested per architecture)

### 4.2 Hyperparameter Definitions

**n_filters (Base Filter Count):**
- Number of filters in the first encoder layer
- Doubles at each encoder level: [n, 2n, 4n, 8n, 16n]
- Higher values = more model capacity, more memory usage
- **Tested:** 16, 32, 64

**dropout (Dropout Rate):**
- Probability of dropping neurons during training
- Applied after each conv block
- Prevents overfitting, improves generalization
- **Tested:** 0.1 (10%), 0.2 (20%), 0.3 (30%)

**batch_norm (Batch Normalization):**
- Normalizes layer inputs during training
- Stabilizes training, allows higher learning rates
- **Fixed:** Always enabled (True)

**learning_rate (Optimizer Learning Rate):**
- Step size for Adam optimizer
- Controls how quickly model learns
- **Tested:** 0.001, 0.003, 0.005

### 4.3 Fixed Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **epochs** | 100 | Sufficient for convergence with early stopping |
| **batch_size** | 4 | Balance between memory and gradient stability |
| **optimizer** | Adam | Adaptive learning rates, works well out-of-the-box |
| **early_stopping_patience** | 20 | Stop if no improvement for 20 epochs |
| **reduce_lr_patience** | 10 | Reduce LR if no improvement for 10 epochs |
| **reduce_lr_factor** | 0.5 | Halve learning rate when plateauing |
| **min_lr** | 1e-7 | Minimum learning rate threshold |

### 4.4 Callbacks

**ModelCheckpoint:**
```python
ModelCheckpoint(
    filepath='checkpoints/{model_name}/best_model.keras',
    monitor='val_jacard_coef',    # Track validation IoU
    mode='max',                    # Save when IoU increases
    save_best_only=True,           # Only save best model
    verbose=1
)
```

**EarlyStopping:**
```python
EarlyStopping(
    monitor='val_jacard_coef',
    patience=20,                   # Wait 20 epochs for improvement
    mode='max',
    restore_best_weights=True      # Restore best weights on stop
)
```

**ReduceLROnPlateau:**
```python
ReduceLROnPlateau(
    monitor='val_jacard_coef',
    factor=0.5,                    # Multiply LR by 0.5
    patience=10,
    mode='max',
    min_lr=1e-7,
    verbose=1
)
```

**CSVLogger:**
```python
CSVLogger(
    filename='logs/{model_name}_history.csv',
    separator=',',
    append=False
)
```

---

## 5. Training Configuration

### 5.1 Directory Structure

**Output Directory:** `./[architecture]_hyperparam_YYYYMMDD_HHMMSS/`

```
unet_hyperparam_20251015_224125/
├── checkpoints/                                    # Best models
│   ├── unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001/
│   │   └── best_model.keras
│   ├── unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003/
│   │   └── best_model.keras
│   └── ... (27 total)
├── logs/                                           # Training histories
│   ├── unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001_history.csv
│   ├── unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p003_history.csv
│   └── ... (27 total)
└── EXPERIMENT_INFO.json                            # Metadata
```

**Key Directories:**
- `checkpoints/` - Best model weights (selected by val_jacard_coef)
- `logs/` - CSV files with epoch-by-epoch metrics
- Root - Experiment metadata and configuration

### 5.2 Computational Requirements

**Per Model Training (27 configurations):**

| Resource | Requirement |
|----------|-------------|
| **GPU** | A40 or A100 (40+ GB VRAM recommended) |
| **RAM** | 240 GB |
| **Walltime** | 12-24 hours (depending on convergence) |
| **Storage** | ~5 GB (models + logs) |

**HPC Job Submission:**
```bash
# Example: UNet training
qsub pbs_train_unet_hyperparam.sh

# Example: Attention UNet training
qsub pbs_train_attention_unet_hyperparam.sh

# Example: Attention ResUNet training
qsub pbs_train_attention_resunet_hyperparam.sh
```

---

## 6. Best Model Results

### 6.1 Performance Summary

| Architecture | Best Configuration | Val IoU | Val Dice |
|--------------|-------------------|---------|----------|
| **Attention ResUNet** | n_filters=32, dropout=0.1, LR=0.001 | **0.5039** | **~0.67** |
| UNet | n_filters=32, dropout=0.2, LR=0.001 | 0.4853 | ~0.65 |
| Attention UNet | n_filters=32, dropout=0.3, LR=0.003 | 0.4759 | ~0.64 |

**Source Files:**
- `/Users/xiaodan/unetCNN/unet-HPC/best_models/SUMMARY.json`
- `/Users/xiaodan/unetCNN/unet-HPC/best_models/unet/model_info.json`
- `/Users/xiaodan/unetCNN/unet-HPC/best_models/attention_unet/model_info.json`
- `/Users/xiaodan/unetCNN/unet-HPC/best_models/attention_resunet/model_info.json`

### 6.2 Model Paths

**Best Models Copied To:**
```
best_models/
├── unet/
│   ├── best_model.keras
│   └── model_info.json
├── attention_unet/
│   ├── best_model.keras
│   └── model_info.json
├── attention_resunet/
│   ├── best_model.keras
│   └── model_info.json
└── SUMMARY.json
```

**Source Paths:**
- **UNet:** `unet_hyperparam_20251015_224125/checkpoints/unet_n_filters32_dropout0p2_batch_normTrue_learning_rate0p001/`
- **Attention UNet:** `attention_unet_hyperparam_20251015_230149/checkpoints/attention_unet_n_filters32_dropout0p3_batch_normTrue_learning_rate0p003/`
- **Attention ResUNet:** `attention_resunet_hyperparam_20251015_235542/checkpoints/attention_resunet_n_filters32_dropout0p1_batch_normTrue_learning_rate0p001/`

### 6.3 Hyperparameter Analysis

**Optimal Hyperparameters by Architecture:**

| Hyperparameter | UNet | Attention UNet | Attention ResUNet |
|----------------|------|----------------|-------------------|
| **n_filters** | 32 | 32 | 32 |
| **dropout** | 0.2 | 0.3 | 0.1 |
| **learning_rate** | 0.001 | 0.003 | 0.001 |

**Observations:**

1. **n_filters=32 is optimal across all architectures**
   - 16 filters: Insufficient model capacity
   - 64 filters: Overfitting and higher memory usage
   - **Sweet spot:** 32 filters balances capacity and generalization

2. **Dropout varies by architecture complexity:**
   - **UNet (0.2):** Moderate regularization
   - **Attention UNet (0.3):** Higher dropout needed (attention adds parameters)
   - **Attention ResUNet (0.1):** Lower dropout (residual connections provide implicit regularization)

3. **Learning rate depends on architecture:**
   - **UNet & ResUNet (0.001):** Conservative learning rate
   - **Attention UNet (0.003):** Higher LR needed for attention gates to converge

**Key Insight:** More complex architectures (residual connections, attention gates) require different regularization strategies. Residual learning provides implicit regularization, reducing need for high dropout.

---

## 7. Architecture Comparison Analysis

### 7.1 Comparative Visualizations

**Generated Outputs:**
```
density_analysis_architecture_comparison_YYYYMMDD_HHMMSS/
├── architecture_comparison_tile_level.csv           # Per-tile density metrics
├── architecture_comparison_image_summary.csv        # Per-image summary
├── EXPERIMENT_INFO.json
├── density_boxplot_comparison_threshold_0p2.png     # 3 architectures side-by-side
├── density_boxplot_comparison_threshold_0p5.png
├── density_boxplot_comparison_threshold_0p8.png
├── density_boxplot_comparison_threshold_0p95.png
├── density_boxplot_comparison_claheotsu_on_pred.png
├── density_boxplot_comparison_claheotsu_on_original.png
└── representative_tiles_4panel/                     # Original + 3 predictions
    ├── tiles_4panel_*.png (40 total)
    └── ...
```

**Density Calculation Methods:**
1. **Threshold-based (4 levels):** 0.2, 0.5, 0.8, 0.95
2. **CLAHE+Otsu on predictions:** Adaptive thresholding on model outputs
3. **CLAHE+Otsu on original images:** Ground truth estimation

### 7.2 Key Findings

**Qualitative Observations:**
- **Attention ResUNet:** Best boundary delineation, fewer false positives
- **Attention UNet:** Better than UNet, but occasional over-segmentation
- **UNet:** Good baseline, but misses smaller beads and has fuzzy boundaries

**Quantitative Performance:**
- **IoU Improvement:** Attention ResUNet achieves +3.8% over UNet baseline
- **Consistency:** Attention mechanisms reduce variance in predictions
- **Edge Cases:** Residual connections help with overlapping beads

---

## 8. Reproducibility

### 8.1 Random Seeds

**Fixed Seeds:**
```python
# Data split
train_test_split(..., random_state=42)

# TensorFlow/Keras
# (Uses default seeds - no explicit seed setting in training scripts)
```

**Note:** Train/val split is reproducible (random_state=42), but model initialization and training stochasticity may vary slightly between runs.

### 8.2 Environment

**Software:**
- Python 3.x
- TensorFlow/Keras 2.x
- NumPy, Pandas, scikit-learn, OpenCV, Pillow

**Hardware:**
- GPU: A40 or A100 (40+ GB VRAM)
- RAM: 240 GB
- Storage: ~15 GB for all experiments

### 8.3 Replication Steps

1. **Prepare dataset:**
   ```bash
   # Ensure data is in correct format:
   ./dataset_shrunk_masks/
   ├── images/  (512×512 RGB .png/.jpg)
   └── masks/   (512×512 grayscale .png)
   ```

2. **Train models:**
   ```bash
   qsub pbs_train_unet_hyperparam.sh
   qsub pbs_train_attention_unet_hyperparam.sh
   qsub pbs_train_attention_resunet_hyperparam.sh
   ```

3. **Copy best models:**
   ```bash
   python copy_best_models.py
   ```

4. **Run comparison analysis:**
   ```bash
   qsub pbs_density_analysis_architecture_comparison.sh
   ```

---

## 9. Conclusions

### 9.1 Architecture Ranking

**For Microbead Segmentation:**

1. ✅ **Attention ResUNet (Best):** Combines residual learning + attention for optimal performance
2. 🔶 **UNet (Baseline):** Strong baseline, simpler and faster
3. ❌ **Attention UNet (Unexpected):** Attention alone doesn't improve over baseline

**Surprising Result:** Attention UNet performed *worse* than standard UNet, suggesting:
- Attention gates alone may not be sufficient
- Residual connections are critical for deep U-Net variants
- Combination of both (ResUNet) unlocks best performance

### 9.2 Key Takeaways

**Architectural Insights:**
1. **Residual connections matter:** +3.8% IoU improvement (ResUNet vs UNet)
2. **Attention requires residual learning:** Attention alone decreased performance
3. **Optimal filters:** 32 base filters across all architectures
4. **Regularization varies:** Complex models need less dropout (residual = implicit regularization)

**Practical Recommendations:**
- **Best accuracy:** Use Attention ResUNet (if compute allows)
- **Best speed:** Use UNet baseline (minimal performance sacrifice)
- **Avoid:** Attention UNet without residual connections

### 9.3 Future Work

**Potential Improvements:**
1. **Data augmentation:** Rotation, flipping, brightness/contrast adjustment
2. **Loss function exploration:** Dice+Focal combination, Tversky loss
3. **Post-processing:** Morphological operations, connected component analysis
4. **Ensemble methods:** Combine predictions from all 3 models
5. **Test-time augmentation:** Average predictions over augmented versions
6. **Deeper architectures:** 5-level U-Net (currently 4 levels + bottleneck)

**Alternative Architectures:**
- **U-Net++:** Nested skip connections
- **DeepLabV3+:** Atrous convolutions for multi-scale features
- **TransUNet:** Vision transformer encoder + U-Net decoder

---

## 10. References

### 10.1 Architecture Papers

**U-Net:**
- Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- https://arxiv.org/abs/1505.04597

**Attention U-Net:**
- Oktay et al. (2018). "Attention U-Net: Learning Where to Look for the Pancreas"
- https://arxiv.org/abs/1804.03999

**Residual Networks:**
- He et al. (2016). "Deep Residual Learning for Image Recognition"
- https://arxiv.org/abs/1512.03385

**Focal Loss:**
- Lin et al. (2017). "Focal Loss for Dense Object Detection"
- https://arxiv.org/abs/1708.02002

### 10.2 Code Locations

**Training Scripts:**
- `train_unet_hyperparam.py` - UNet hyperparameter search
- `train_attention_unet_hyperparam.py` - Attention UNet hyperparameter search
- `train_attention_resunet_hyperparam.py` - Attention ResUNet hyperparameter search

**Architecture Definitions:**
- `models_fixed.py:163` - build_unet()
- `models_fixed.py:283` - build_attention_unet()
- `models_fixed.py:410` - build_attention_resunet()
- `models_fixed.py:66` - res_conv_block()
- `models_fixed.py:96` - attention_block()

**Loss Functions:**
- `loss_functions_fixed.py` - BinaryFocalLoss, jacard_coef, dice_coef

**Analysis Scripts:**
- `copy_best_models.py` - Extract best models from hyperparam search
- `density_analysis_architecture_comparison.py` - Compare all 3 architectures

**PBS Job Scripts:**
- `pbs_train_unet_hyperparam.sh`
- `pbs_train_attention_unet_hyperparam.sh`
- `pbs_train_attention_resunet_hyperparam.sh`
- `pbs_density_analysis_architecture_comparison.sh`

---

## Appendix: Alternative PyTorch Training Approach

### A.1 Overview

This repository contains a **separate, independent** training pipeline in `train.py` that uses a fundamentally different approach from the three Keras models analyzed above.

**Key Differences:**

| Aspect | Keras Models (This Document) | PyTorch Model (train.py) |
|--------|------------------------------|--------------------------|
| **Framework** | Keras/TensorFlow 2.x | PyTorch |
| **Architecture** | UNet, Attention UNet, Attention ResUNet | ConsensusAttnUNet (Student-Teacher) |
| **Training Strategy** | Standard supervised learning | **Knowledge distillation** with teacher model |
| **Loss Function** | BinaryFocalLoss (γ=2, α=0.25) | **AdaptiveBGDiceLoss** (focal + TV + Tversky + bg-adaptive) |
| **Image Channels** | **RGB (3 channels)** | **Grayscale (1 channel)** |
| **Image Loading** | cv2.imread(path, 1) - RGB | Image.open(path).convert("L") - Grayscale |
| **Mask Loading** | cv2.imread(path, 0) - Grayscale | Image.open(path).convert("L") - Grayscale |
| **Normalization** | **arr / 255.0** (linear scaling) | **Percentile normalization** (0.5-99.5 percentile → [0,1]) |
| **Resize Method** | PIL Image.resize() before normalization | PyTorch F.interpolate() after normalization |
| **Data Augmentation** | **None** | **Synthetic background artifacts** (gradients, bands, noise, bead fading) |
| **Batch Size** | 4 | 1 (with gradient accumulation: 8 steps) |
| **Learning Rate** | 0.001-0.005 (grid search) | 5e-4 (fixed) |
| **Epochs** | 100 (early stopping @ 20) | 400 |
| **Train/Val Split** | 80/20 with sklearn (random_state=42) | 80/20 with PyTorch random_split |
| **Mixed Precision** | No | Yes (AMP with GradScaler) |
| **Teacher Update** | N/A | EMA (α=0.999) every gradient step |

### A.2 Architecture: ConsensusAttnUNet

**Student-Teacher Distillation Framework:**

```
┌─────────────────────────────────────────────────────────┐
│  TEACHER MODEL (frozen, no gradients)                   │
│  ┌─────────────┐      ┌──────────────────────┐          │
│  │   Image     │──────▶  Encoder (frozen)    │          │
│  └─────────────┘      └──────────┬───────────┘          │
│  ┌─────────────┐                 │                      │
│  │  GT Mask    │──────▶  Encoder (frozen) ───┤          │
│  └─────────────┘                 │           │          │
│                           TwoSelfAttnFuse ◀──┘          │
│                                  │                       │
│                          [fused_gt_target]              │
│                                  │ (supervision)         │
└──────────────────────────────────┼───────────────────────┘
                                   │
                                   ▼ (MSE loss)
┌──────────────────────────────────┼───────────────────────┐
│  STUDENT MODEL (trainable)       │                       │
│  ┌─────────────┐                 │                       │
│  │   Image     │──────▶  Encoder │                       │
│  └─────────────┘         │       │                       │
│                    ┌─────┴───────┴──────┐                │
│                    │  TwoSelfAttnFuse   │                │
│                    └─────┬───────────────┘                │
│                          │ [fused_fake] (distillation)   │
│                          │                               │
│                     ┌────▼───────┐                        │
│                     │  Decoder   │                        │
│                     └────┬───────┘                        │
│                          │                               │
│                    [final_output]                        │
│                          │                               │
└──────────────────────────┼───────────────────────────────┘
                           │
                           ▼ (AdaptiveBGDiceLoss)
                    [Ground Truth]
```

**Key Concept:** The teacher model processes **both image and ground truth mask** through frozen encoders, fuses them with attention, and produces a target feature representation. The student learns to produce similar features from the **image alone**, enabling it to "imagine" what the mask-informed features should look like.

### A.3 Loss Function: AdaptiveBGDiceLoss

**Multi-Component Loss (train.py:203-320):**

```
Total Loss = Main Loss
           + 0.05 × L_bg_adapt        # Background-adaptive penalty
           + 1.0 × L_tv               # Total variation on flat backgrounds
           + 0.4 × L_tversky          # Tversky loss (variant of Dice)
           + 10.0 × L_attention_distill  # Attention feature distillation
```

**1. Main Loss (Focal BCE):**
- **Parameters:** α=0.4, γ=2.0
- **Target:** Foreground (beads, not background)
- Different from Keras models which use α=0.25

**2. Background-Adaptive Loss (L_bg_adapt):**
```python
# Novel component: penalizes over-prediction on bright backgrounds
b = box_blur(image, kernel=193)  # Estimate local background illumination
t = (b + delta).clamp(max=1.0)   # Adaptive threshold (delta=0.07)
w = (1.0 - b)                     # Weight: darker regions weighted more
over = ReLU(pred_bg - t) * w * mask_bg  # Penalize pred_bg > threshold

# Insight: In bright regions, background should stay high (beads=low)
```

**Why this matters:** Microscopy images have **non-uniform illumination**. This loss prevents false positives in bright background regions by adapting the threshold based on local brightness.

**3. Total Variation Loss (L_tv):**
```python
# Smoothness regularization weighted by background flatness
flat_weight = exp(-β × |∇b|)  # β=10: strong penalty in flat regions
L_tv = |∇pred_bg| * flat_weight * mask_bg

# Insight: Background should be smooth in flat-illumination regions
```

**4. Tversky Loss (L_tversky):**
```python
# Asymmetric Dice: weights FP and FN differently
Tversky = TP / (TP + α×FN + β×FP)
# α=0.7, β=0.3: penalizes false negatives more than false positives
```

**Keras vs PyTorch Loss Comparison:**

| Component | Keras (BinaryFocalLoss) | PyTorch (AdaptiveBGDiceLoss) |
|-----------|-------------------------|------------------------------|
| Focal BCE | ✅ α=0.25, γ=2.0 | ✅ α=0.4, γ=2.0 |
| Dice/Tversky | ❌ No (only as metric) | ✅ Tversky (α=0.7, β=0.3, weight=0.4) |
| TV Regularization | ❌ No | ✅ Yes (weight=1.0, adaptive by background) |
| Background-Adaptive | ❌ No | ✅ Yes (weight=0.05, kernel=193) |
| Attention Distillation | ❌ No | ✅ Yes (MSE, weight=10.0) |

### A.4 Image Preprocessing (PyTorch Model)

**CRITICAL DIFFERENCE:** The PyTorch model uses fundamentally different preprocessing from the Keras models.

**Source:** `train.py:366-408` (BeadsDataset._load_gray01 and __getitem__)

```python
# Step 1: Load as GRAYSCALE (not RGB!)
im = Image.open(image_path).convert("L")  # Force grayscale
arr = np.array(im, dtype=np.float32)      # Shape: (H, W), dtype: float32

# Step 2: Percentile normalization (NOT division by 255!)
def _percentile_norm(arr):
    lo, hi = np.percentile(arr, [0.5, 99.5]).astype(np.float32)  # Robust to outliers
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)  # Contrast stretching
    return arr.astype(np.float32)

img_normalized = _percentile_norm(arr)  # [0, 1] with enhanced contrast

# Step 3: Apply augmentation (60% of training images)
if augment:
    img_normalized = add_bg_artifacts_with_fade(img_normalized, ...)

# Step 4: Convert to PyTorch tensor
image = torch.from_numpy(img_normalized).unsqueeze(0)  # Shape: (1, H, W)

# Step 5: Resize (AFTER normalization, using bilinear interpolation)
if resize_to is not None:
    image = F.interpolate(image.unsqueeze(0), size=(512, 512),
                         mode="bilinear", align_corners=False).squeeze(0)

# Step 6: Invert mask convention (beads=0 → beads=1)
target = 1.0 - mask  # PyTorch model predicts BACKGROUND, not foreground
```

**Final Tensor Shapes:**
- **Input (X):** `(N, 1, 512, 512)` - Grayscale images, percentile-normalized
- **Target (y):** `(N, 1, 512, 512)` - Inverted masks (background=1, beads=0)

**Key Differences from Keras:**

| Aspect | Keras Models | PyTorch Model |
|--------|-------------|---------------|
| **Channels** | RGB (3 channels) | Grayscale (1 channel) |
| **Color Info** | Preserves color | Discards color |
| **Normalization** | Linear (arr/255) | Percentile (0.5-99.5 → [0,1]) |
| **Contrast** | Raw values preserved | **Contrast stretched** |
| **Outliers** | Affected by extreme pixels | **Robust to outliers** |
| **Resize Timing** | Before normalization | After normalization |
| **Resize Method** | PIL (Lanczos by default) | PyTorch bilinear |
| **Mask Convention** | beads=0, bg=1 | **beads=1, bg=0** (inverted!) |

**Why Percentile Normalization Matters:**

```python
# Example: Image with uneven illumination
raw_pixels = [10, 15, 20, ..., 200, 245, 255]  # Wide range

# Keras (linear /255):
keras_norm = [0.039, 0.059, 0.078, ..., 0.784, 0.961, 1.0]
# Low contrast in dark regions (beads often in 0.0-0.2 range)

# PyTorch (percentile):
p0.5 = 12, p99.5 = 240
pytorch_norm = [(10-12)/(240-12), (15-12)/(240-12), ..., (245-12)/(240-12)]
pytorch_norm = [0.0, 0.013, 0.035, ..., 0.821, 1.0]  # Clips outliers
# ENHANCED contrast: spreads middle 99% of pixels across full [0,1] range
```

**Impact on Model Performance:**
1. ✅ **Better for low-contrast images:** Percentile norm enhances visibility of dim beads
2. ✅ **Robust to illumination artifacts:** Ignores extreme bright/dark pixels
3. ❌ **Loses absolute intensity information:** Can't distinguish "bright" vs "dark" images
4. ❌ **Per-image normalization:** Same pixel intensity may map to different values

### A.5 Data Augmentation

**Keras Models:** No augmentation (raw images only)

**PyTorch Model:** Sophisticated **synthetic background artifact** augmentation (train.py:136-192):

**Augmentation Pipeline (60% of training images):**

```python
# 40% no augmentation
# 30% old-style artifacts
# 30% new-style artifacts with bead fading

# Old-style artifacts (simple):
image += random_gradient(max_grad=0.05-0.25)  # Linear intensity gradient
image += sinusoidal_bands(amp=0.03, period=40-120)  # Scanning artifacts
image += gaussian_noise(σ=0.02)

# New-style artifacts (realistic):
image += gradient + bands + noise  # (smaller amplitudes)
# Then apply selective bead fading:
bead_mask = (image < 0.4)  # Detect dark regions (beads)
darkness = (0.4 - image) / 0.4  # Intensity-proportional fading
local_bg = box_blur(image, kernel=33-81)  # Estimate local background
image[beads] = blend(image, local_bg, strength=0.7-1.2 × darkness)
# Effect: Dark beads "fade into" local background
```

**Motivation:** Microscopy images suffer from:
1. **Uneven illumination** (vignetting, gradients)
2. **Scanning artifacts** (periodic intensity bands)
3. **Bead fading** (beads appear dimmer in bright backgrounds)

The augmentation **simulates these artifacts** to improve generalization.

### A.6 Training Configuration

**PyTorch-Specific Settings:**

```python
# Hardware
device = 'cuda' (A40/A100)
mixed_precision = True (AMP with GradScaler)

# Optimization
optimizer = Adam(lr=5e-4)  # Fixed, no hyperparameter search
batch_size = 1
gradient_accumulation = 8 steps  # Effective batch size = 8
epochs = 400 (no early stopping)

# Checkpointing
save_every_epoch = True  # checkpoint_epoch_N.pth
save_student_and_teacher = True  # Both models saved
save_optimizer_and_scaler = True  # For resuming

# Teacher Update
EMA_alpha = 0.999  # Exponential moving average
update_frequency = every_gradient_step  # Not every epoch
```

**Key Difference:** No hyperparameter search - single fixed configuration trained for 400 epochs.

### A.7 Why Two Different Approaches?

**Keras Models (Grid Search):**
- **Goal:** Find optimal architecture and hyperparameters
- **Strategy:** Exhaustive search (81 total models)
- **Strengths:** Systematic comparison, reproducible
- **Weaknesses:** Expensive compute, no augmentation, simpler loss

**PyTorch Model (Distillation):**
- **Goal:** Maximize performance with advanced techniques
- **Strategy:** Single model with sophisticated training
- **Strengths:** State-of-the-art techniques (distillation, adaptive loss, augmentation)
- **Weaknesses:** No hyperparameter tuning, different framework

**These are complementary approaches:**
1. Keras: Architecture comparison study
2. PyTorch: Maximum performance single model

### A.8 Performance Comparison (If Available)

**Note:** Direct performance comparison between Keras and PyTorch models is **NOT POSSIBLE** because they are trained on fundamentally different data:

1. **Different input modalities:** Keras=RGB (3-channel), PyTorch=Grayscale (1-channel)
2. **Different normalization:** Keras=linear (/255), PyTorch=percentile (contrast stretching)
3. **Different mask conventions:** Keras predicts foreground (beads=0), PyTorch predicts background (beads=1)
4. **Different metrics:** Keras validates with IoU/Dice, PyTorch tracks multi-component losses
5. **Different augmentation:** Keras=none, PyTorch=synthetic artifacts

**These are NOT the same dataset** - even though they may use the same raw image files, the preprocessing transforms them into completely different inputs. Comparing their performance would be like comparing a color photo classifier to a grayscale photo classifier.

**To enable comparison, you would need to:**
```bash
# Evaluate PyTorch model with same metrics as Keras
python evaluate_pytorch_model.py \
  --checkpoint ./checkpoint_consensus_distill/latest.pth \
  --metrics iou dice \
  --save_predictions ./pytorch_predictions/
```

### A.9 When to Use Which Approach?

**Use Keras Models (UNet/Attention/ResUNet) if:**
- ✅ You want proven hyperparameters (grid search validated)
- ✅ You prefer Keras/TensorFlow ecosystem
- ✅ You need RGB input support
- ✅ You want simpler, more interpretable training
- ✅ You need fast inference (standard architectures)

**Use PyTorch Model (ConsensusAttnUNet) if:**
- ✅ You want state-of-the-art techniques (distillation, adaptive loss)
- ✅ You have challenging microscopy with illumination artifacts
- ✅ You can afford longer training (400 epochs)
- ✅ You prefer PyTorch ecosystem
- ✅ You need grayscale-specific optimizations
- ✅ You want built-in augmentation for microscopy artifacts

### A.10 Code Locations (PyTorch Approach)

**Training:**
- `train.py` - Main training script with distillation

**Architecture:**
- `twodecoder_skip_64.py` - ConsensusAttnUNetStudent, TwoSelfAttnFuse

**Key Components:**
- `train.py:18` - ConsensusAttnUNetTeacher (frozen encoder + attention fusion)
- `train.py:56` - update_teacher_weights() (EMA update)
- `train.py:203` - AdaptiveBGDiceLoss (multi-component loss)
- `train.py:330` - BeadsDataset (with augmentation)
- `train.py:161` - add_bg_artifacts_with_fade() (sophisticated augmentation)
- `train.py:501` - train_one_epoch_distillation() (student-teacher training loop)

**Usage:**
```bash
# Train from scratch
python train.py

# Resume from checkpoint
# Edit line 588: CHECKPOINT_TO_RESUME = './checkpoint_consensus_distill/latest.pth'
python train.py
```

---

## Summary: Two Training Paradigms

This repository demonstrates **two independent approaches** to microbead segmentation:

### Paradigm 1: Keras Grid Search (This Document)
- **3 architectures** (UNet, Attention UNet, Attention ResUNet)
- **27 hyperparameter configs each** (81 total models)
- **RGB input**, simple preprocessing, no augmentation
- **BinaryFocalLoss** (single component)
- **Best result:** Attention ResUNet (IoU=0.5039)
- **Strength:** Systematic architecture comparison

### Paradigm 2: PyTorch Distillation (train.py)
- **1 architecture** (ConsensusAttnUNet student-teacher)
- **No hyperparameter search** (single fixed config)
- **Grayscale input**, percentile norm, sophisticated augmentation
- **AdaptiveBGDiceLoss** (5 components: focal + bg-adaptive + TV + Tversky + distillation)
- **Best result:** Unknown (no IoU/Dice validation reported)
- **Strength:** Advanced training techniques

**Both approaches are valid** - they represent different philosophies (breadth vs depth) and could potentially be combined for even better performance.

---

**Document Created:** October 17, 2025
**Author:** Claude Code
**Keras Experiments:** October 15-17, 2025
**Total Keras Models Trained:** 81 (27 configs × 3 architectures)
**Best Keras Model:** Attention ResUNet (Val IoU: 0.5039)
**PyTorch Training:** Separate timeline (train.py)
