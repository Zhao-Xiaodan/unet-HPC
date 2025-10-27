# PyTorch UNet Pipeline: Training, Prediction & Density Analysis

This folder contains a complete PyTorch pipeline for training 3 UNet architectures, generating predictions, and performing density analysis on microscopy bead images.

## 📋 Overview

**Complete Workflow:**
```
1. Training (train_*.py)
   └─> Trains 3 UNet architectures with hyperparameter search

2. Prediction (predict_pytorch_comparison.py)
   └─> Loads best models and generates mask predictions

3. Density Analysis (density_analysis_pytorch_comparison.py)
   └─> Analyzes predictions and creates visualizations

4. Results Analysis (analyze_pytorch_comparison.py)
   └─> Analyzes training results and model performance
```

---

## 📁 Files Included

### Training Scripts (Stage 1)

#### 1. `train_pytorch_comparison_no_aug.py` ⭐ **RECOMMENDED**
- **Purpose**: Fair comparison with Keras models
- **Features**:
  - No data augmentation (matching Keras training)
  - BinaryFocalLoss (same as Keras)
  - Trains all 3 architectures: UNet, Attention UNet, Attention ResUNet
  - Grid search over hyperparameters
- **Best for**: Direct comparison with existing Keras results

#### 2. `train_pytorch_comparison_with_aug.py`
- **Purpose**: Training with data augmentation
- **Features**:
  - Includes augmentation techniques
  - Better generalization for new data
  - Same 3 architectures

#### 3. `train_pytorch_comparison_adaptive_loss.py`
- **Purpose**: Advanced training with adaptive loss
- **Features**:
  - Uses AdaptiveBGDiceLoss (handles class imbalance better)
  - Synthetic background artifact augmentation
  - Best for challenging datasets with background noise

---

### Prediction Script (Stage 2)

#### 4. `predict_pytorch_comparison.py`
- **Purpose**: Generate predictions using trained models
- **Key Functions**:
  - `load_model()` - Loads trained PyTorch checkpoints
  - `predict_tiles()` - Runs inference on 512×512 tiles
  - `percentile_normalize()` - Pre-processes images (0.5-99.5th percentile)

**Workflow:**
```python
Test Image (.tif)
    ↓
Load & extract 512×512 tiles
    ↓
For each architecture (UNet, Attention UNet, Attention ResUNet):
    ├─ Percentile normalize [0.5, 99.5]
    ├─ Convert to tensor [B, 1, 512, 512]
    ├─ Run model.forward() → predictions
    └─ Reconstruct full image
    ↓
Save as PNG (_pred.png)
```

---

### Analysis Scripts (Stage 3)

#### 5. `density_analysis_pytorch_comparison.py`
- **Purpose**: Analyze prediction density and create visualizations
- **Outputs**:
  - **Density boxplots**: Compare bead density across dilution factors
  - **4-panel tile visualizations**: Original | UNet | Attention UNet | Attention ResUNet
  - **CSV files**: Tile-level and image-level density statistics

**Key Metrics:**
- Bead density = fraction of pixels above threshold (0.5)
- Analyzed at tile level (512×512) and image level
- Grouped by dilution factor (10x to 10240x)

#### 6. `analyze_pytorch_comparison.py`
- **Purpose**: Analyze training results and model performance
- **Outputs**:
  - Training curves (loss, IoU over epochs)
  - Architecture comparison plots
  - Best model selection

---

## 🏗️ The 3 UNet Architectures

### 1. **Standard UNet**
```
Encoder (4 levels with MaxPool)
    → Bottleneck
        → Decoder (4 levels with ConvTranspose + Skip Connections)
            → Output (Sigmoid)
```
- **Features**: Classic U-Net with skip connections
- **Parameters**: ~7.8M (with n_filters=32)

### 2. **Attention UNet**
```
Encoder (4 levels)
    → Bottleneck
        → Decoder with Attention Gates
            → Attention filters skip connections
                → Output (Sigmoid)
```
- **Features**: Attention gates highlight important spatial features
- **Parameters**: ~8.2M (with n_filters=32)
- **Advantage**: Better at focusing on bead regions, ignoring background

### 3. **Attention ResUNet**
```
Encoder (4 ResNet blocks)
    → Bottleneck (ResNet block)
        → Decoder (4 ResNet blocks + Attention Gates)
            → Output (Sigmoid)
```
- **Features**: Residual connections + Attention gates
- **Parameters**: ~8.5M (with n_filters=32)
- **Advantage**: Best gradient flow, most robust training

---

## 🚀 Usage

### Step 1: Train Models

```bash
# Option A: No augmentation (recommended for Keras comparison)
python train_pytorch_comparison_no_aug.py

# Option B: With augmentation (better generalization)
python train_pytorch_comparison_with_aug.py

# Option C: Adaptive loss (best for imbalanced data)
python train_pytorch_comparison_adaptive_loss.py
```

**Expected Output:**
```
pytorch_comparison_no_aug_YYYYMMDD_HHMMSS/
├── unet/
│   └── checkpoints/
│       └── unet_f32_d0.2_lr0.0001/
│           └── best_model.pth
├── attention_unet/
│   └── checkpoints/...
├── attention_resunet/
│   └── checkpoints/...
└── all_results.csv
```

---

### Step 2: Generate Predictions

```bash
python predict_pytorch_comparison.py \
    --experiment pytorch_comparison_no_aug_20251021_121918 \
    --test_images ./test_images \
    --output ./predictions_output
```

**Expected Output:**
```
predictions_output/
├── unet/
│   ├── image_10x_pred.png
│   ├── image_20x_pred.png
│   └── ...
├── attention_unet/
│   └── ...
├── attention_resunet/
│   └── ...
└── prediction_metadata.json
```

**Key Parameters:**
- `--experiment`: Path to training experiment directory
- `--test_images`: Directory with test `.tif` images
- `--output`: Where to save predictions
- `--batch_size`: GPU batch size (default: 8)

---

### Step 3: Analyze Density

```bash
python density_analysis_pytorch_comparison.py \
    --predictions ./predictions_output \
    --test_images ./test_images \
    --output ./density_analysis_results
```

**Expected Output:**
```
density_analysis_results/
├── density_results_tile_level.csv
├── density_results_image_summary.csv
├── density_boxplot_full_range__threshold_0.5.png
├── density_boxplot_low_dilution_range__threshold_0.5.png
├── density_boxplot_unet_full_range_threshold_0.5.png
├── density_boxplot_attention_unet_full_range_threshold_0.5.png
├── density_boxplot_attention_resunet_full_range_threshold_0.5.png
└── representative_tiles_4panel/
    ├── tiles_4panel_10x.png
    ├── tiles_4panel_20x.png
    └── ...
```

---

### Step 4: Analyze Training Results

```bash
python analyze_pytorch_comparison.py \
    --experiment pytorch_comparison_no_aug_20251021_121918
```

**Expected Output:**
- Training loss/IoU curves for all models
- Best model summary table
- Architecture comparison plots

---

## 🔍 Code Deep Dive

### Image Preprocessing (`predict_pytorch_comparison.py`)

```python
def percentile_normalize(image, lower=0.5, upper=99.5):
    """
    Normalizes image to [0, 1] using percentile clipping
    - Clips outliers (below 0.5th and above 99.5th percentile)
    - Scales to [0, 1] range
    - Same as training preprocessing
    """
    p_low, p_high = np.percentile(image, [lower, upper])
    image = np.clip(image, p_low, p_high)
    image = (image - p_low) / (p_high - p_low + 1e-8)
    return image
```

**Where used:**
- Line 500 in `predict_pytorch_comparison.py` before model inference
- Critical for matching training preprocessing

---

### Model Loading (`predict_pytorch_comparison.py:469-487`)

```python
def load_model(arch_name, checkpoint_path, n_filters, dropout, device):
    """
    1. Instantiate model architecture (UNet/AttentionUNet/AttentionResUNet)
    2. Load trained weights from .pth checkpoint
    3. Set to evaluation mode (disable dropout/batch norm training)
    4. Move to GPU/CPU
    """
    # Architecture selection
    if arch_name == 'unet':
        model = UNet(n_channels=1, n_filters=n_filters, dropout=dropout)
    elif arch_name == 'attention_unet':
        model = AttentionUNet(n_channels=1, n_filters=n_filters, dropout=dropout)
    elif arch_name == 'attention_resunet':
        model = AttentionResUNet(n_channels=1, n_filters=n_filters, dropout=dropout)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Important!

    return model
```

---

### Prediction with Batching (`predict_pytorch_comparison.py:489-516`)

```python
def predict_tiles(model, tiles, device, batch_size=8):
    """
    Batch prediction for memory efficiency

    Input:  List of numpy tiles [H, W]
    Output: List of numpy predictions [H, W] with values [0, 1]
    """
    predictions = []

    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i+batch_size]

        # Preprocess each tile
        batch_tensor = []
        for tile in batch_tiles:
            tile_norm = percentile_normalize(tile.astype(np.float32))
            tile_tensor = torch.from_numpy(tile_norm).unsqueeze(0)  # Add channel: [H,W] → [1,H,W]
            batch_tensor.append(tile_tensor)

        # Stack and move to device: [B, 1, H, W]
        batch_tensor = torch.stack(batch_tensor).to(device)

        # Inference (no gradients needed)
        with torch.no_grad():
            batch_pred = model(batch_tensor)  # Forward pass through UNet

        # Convert back to numpy
        batch_pred = batch_pred.squeeze(1).cpu().numpy()  # [B, 1, H, W] → [B, H, W]
        predictions.extend(batch_pred)

    return predictions
```

**Tensor Shape Transformations:**
```
Input tile:         [512, 512]          (numpy grayscale)
    ↓ unsqueeze(0)
Add channel:        [1, 512, 512]       (single tile tensor)
    ↓ stack(batch)
Batched input:      [8, 1, 512, 512]    (batch of 8 tiles)
    ↓ model.forward()
Model output:       [8, 1, 512, 512]    (sigmoid activation → [0,1])
    ↓ squeeze(1)
Remove channel:     [8, 512, 512]       (batch of predictions)
    ↓ cpu().numpy()
Final output:       [8, 512, 512]       (numpy arrays [0,1])
```

---

### Density Calculation (`density_analysis_pytorch_comparison.py:107-111`)

```python
def calculate_density(mask, threshold=0.5):
    """
    Calculate bead density as fraction of pixels above threshold

    Input:  Grayscale prediction [0, 1]
    Output: Density value [0, 1] (fraction of foreground pixels)
    """
    binary_mask = (mask > threshold).astype(np.float32)
    density = np.mean(binary_mask)  # Fraction of "bead" pixels
    return density
```

**Example:**
```
512×512 tile prediction:
- 10,000 pixels > 0.5 (predicted as beads)
- 262,144 pixels total
- Density = 10,000 / 262,144 = 0.038 (3.8% bead coverage)
```

---

## 📊 Expected Results

### Training Metrics (all_results.csv)
```csv
architecture,n_filters,dropout,learning_rate,best_val_iou,best_epoch
unet,32,0.2,0.0001,0.8234,45
attention_unet,32,0.2,0.0001,0.8456,42
attention_resunet,32,0.2,0.0001,0.8512,48
```

### Density Analysis
- **Full range plots**: Compare all dilution factors (10x to 10240x)
- **Low dilution plots**: Focus on sparse samples (80x to 10240x)
- **Individual architecture plots**: Separate visualization per model
- **Representative tiles**: 5 tiles per dilution (min, 25th%, median, 75th%, max)

---

## 🧪 Key Insights

### Why Tile-Based Processing?
1. **Memory efficiency**: Large images (2048×2048+) don't fit in GPU memory
2. **Training consistency**: Models trained on 512×512 patches
3. **No blending artifacts**: Non-overlapping tiles avoid edge issues

### Why Percentile Normalization?
1. **Robustness to outliers**: Removes hot pixels and detector noise
2. **Consistent intensity range**: All images normalized to [0, 1]
3. **Same as training**: Critical for model performance

### Architecture Comparison
- **UNet**: Fastest, good baseline
- **Attention UNet**: Best for cluttered images (filters background noise)
- **Attention ResUNet**: Best gradient flow, highest accuracy (but slowest)

---

## 🔧 Requirements

```bash
# Core dependencies
torch >= 1.13.0
torchvision >= 0.14.0
numpy >= 1.21.0
pandas >= 1.3.0
opencv-python >= 4.5.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
tqdm >= 4.62.0
pillow >= 9.0.0
```

---

## 📝 Notes

### Important Implementation Details

1. **Model output activation**: All models use `torch.sigmoid()` at the output layer (lines 169, 239, 309 in `predict_pytorch_comparison.py`)

2. **Prediction inversion**: In `density_analysis_pytorch_comparison.py` line 481, predictions are inverted (`1.0 - pred_tile`) for visualization to match original image appearance

3. **Threshold**: Default threshold of 0.5 is used for binary classification (can be adjusted)

4. **Dilution factors**: Code handles 10 dilution levels: 10x, 20x, 80x, 160x, 320x, 640x, 1280x, 2560x, 5120x, 10240x

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python predict_pytorch_comparison.py --batch_size 4
```

### Model checkpoint not found
```bash
# Check experiment directory structure
ls pytorch_comparison_no_aug_*/unet/checkpoints/
```

### Image format issues
- Ensure test images are `.tif` or `.tiff` format
- Images should be grayscale (single channel)
- Supported sizes: any multiple of 512 (e.g., 512, 1024, 2048)

---

## 📧 Support

For questions about:
- **Training**: See comments in `train_pytorch_comparison_*.py`
- **Prediction**: See comments in `predict_pytorch_comparison.py`
- **Analysis**: See comments in `density_analysis_pytorch_comparison.py`

---

## 📄 License

Author: Claude Code
Date: October 2025
Purpose: Research and educational use

---

**End of Documentation**
