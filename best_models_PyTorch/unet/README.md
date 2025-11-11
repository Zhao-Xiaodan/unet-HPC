# PyTorch U-Net Model for Microbead Segmentation

## Project Overview

This directory contains a trained PyTorch U-Net model for **microbead segmentation** in microscopy images. The model was trained to detect and segment microbeads in grayscale microscope images, achieving **63.77% IoU (Intersection over Union)** on the validation set.

### Project Background

**Application:** Automated microbead detection and counting in microscopy images
**Problem:** Manual counting of microbeads is time-consuming and error-prone
**Solution:** Deep learning semantic segmentation using U-Net architecture

**Dataset Characteristics:**
- **98 grayscale images** at 512×512 resolution
- **Sparse foreground:** ~5.6% of pixels are microbeads, 94.4% background
- **Challenges:** Overlapping beads, variable sizes, cluttered regions, class imbalance

**Key Achievement:** This model represents the best-performing U-Net configuration from extensive hyperparameter search across 3 architectures (U-Net, Attention U-Net, Attention ResUNet) with 27 hyperparameter combinations tested.

---

## Model Specifications

### Architecture: Standard U-Net

**Architecture Type:** Encoder-Decoder with Skip Connections

**Key Components:**
- **4-level encoder** with progressive downsampling (512 → 256 → 128 → 64 → 32)
- **Bottleneck** at 32×32 resolution
- **4-level decoder** with progressive upsampling back to 512×512
- **Skip connections** between encoder and decoder at each level

**Architecture Details:**

```
Input: [1, 512, 512] (grayscale image)

Encoder:
├─ enc1: ConvBlock(1 → 32) → [32, 512, 512]
├─ pool1: MaxPool2d(2) → [32, 256, 256]
├─ enc2: ConvBlock(32 → 64) → [64, 256, 256]
├─ pool2: MaxPool2d(2) → [64, 128, 128]
├─ enc3: ConvBlock(64 → 128) → [128, 128, 128]
├─ pool3: MaxPool2d(2) → [128, 64, 64]
├─ enc4: ConvBlock(128 → 256) → [256, 64, 64]
└─ pool4: MaxPool2d(2) → [256, 32, 32]

Bottleneck:
└─ bottleneck: ConvBlock(256 → 512) → [512, 32, 32]

Decoder:
├─ up4: ConvTranspose2d(512 → 256) → [256, 64, 64]
│  └─ skip connection from enc4: concat → [512, 64, 64]
│  └─ dec4: ConvBlock(512 → 256) → [256, 64, 64]
├─ up3: ConvTranspose2d(256 → 128) → [128, 128, 128]
│  └─ skip connection from enc3: concat → [256, 128, 128]
│  └─ dec3: ConvBlock(256 → 128) → [128, 128, 128]
├─ up2: ConvTranspose2d(128 → 64) → [64, 256, 256]
│  └─ skip connection from enc2: concat → [128, 256, 256]
│  └─ dec2: ConvBlock(128 → 64) → [64, 256, 256]
├─ up1: ConvTranspose2d(64 → 32) → [32, 512, 512]
│  └─ skip connection from enc1: concat → [64, 512, 512]
│  └─ dec1: ConvBlock(64 → 32) → [32, 512, 512]
└─ out: Conv2d(32 → 1) → [1, 512, 512]

Output: [1, 512, 512] (binary mask, logits before sigmoid)
```

**ConvBlock Details:**
```python
ConvBlock(in_channels, out_channels, dropout):
  ├─ Conv2d(in_channels, out_channels, kernel=3, padding=1)
  ├─ BatchNorm2d(out_channels)
  ├─ ReLU
  ├─ Conv2d(out_channels, out_channels, kernel=3, padding=1)
  ├─ BatchNorm2d(out_channels)
  ├─ ReLU
  └─ Dropout2d(dropout) [optional]
```

### Hyperparameters

**Optimal Configuration (from model_info.json):**
- **Base filters (n_filters):** 32
- **Dropout rate:** 0.2 (applied in each ConvBlock)
- **Learning rate:** 0.001 (Adam optimizer)
- **Training image size:** 512×512
- **Input channels:** 1 (grayscale)
- **Output channels:** 1 (binary mask)

**Training Configuration:**
- **Loss function:** Binary Focal Loss (α=0.25, γ=2.0)
- **Batch size:** 4
- **Epochs:** 100 (with early stopping)
- **Optimizer:** Adam
- **Mixed precision:** FP16 (for memory efficiency)

### Model Performance

**Validation Metrics:**
- **IoU (Jaccard Index):** 63.77%
- **Dice Coefficient:** ~77.9% (estimated from IoU)
- **Training date:** October 21, 2025
- **Source experiment:** `pytorch_comparison_no_aug_20251021_121918`

---

## File Contents

```
best_models_PyTorch/unet/
├── README.md              # This file
├── model_info.json        # Model metadata and hyperparameters
└── best_model.pth         # Trained model weights (to be synced from HPC)
```

**model_info.json** contains:
```json
{
  "n_filters": 32,
  "dropout": 0.2,
  "learning_rate": 0.001,
  "best_val_iou": 0.6377273201942444,
  "model_name": "unet_n_filters32_dropout0.2_learning_rate0.001",
  "source_experiment": "pytorch_comparison_no_aug_20251021_121918",
  "cached_date": "2025-10-22 12:57:51"
}
```

**best_model.pth** checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state (for resuming training)
- `epoch`: Training epoch when saved
- `val_iou`: Validation IoU at checkpoint
- Additional metadata

---

## Installation & Requirements

### System Requirements

**Minimum:**
- **CPU:** Any modern x86-64 processor
- **RAM:** 4GB (8GB recommended)
- **Storage:** 500MB for model and dependencies

**Recommended (for faster inference):**
- **GPU:** NVIDIA GPU with 2GB+ VRAM (CUDA support)
- **RAM:** 8GB+
- **Storage:** 2GB+ (for additional images)

### Software Requirements

**Python Version:** 3.8+

**Core Dependencies:**
```bash
torch>=2.0.0          # PyTorch (CPU or CUDA version)
torchvision>=0.15.0   # For transforms
numpy>=1.21.0         # Numerical computing
pillow>=9.0.0         # Image loading/saving
```

**Optional (for analysis/visualization):**
```bash
matplotlib>=3.5.0     # Plotting and visualization
pandas>=1.4.0         # Data analysis
opencv-python>=4.6.0  # Advanced image processing
```

### Installation Instructions

#### Option 1: Install with CPU Support (Works on all laptops)

```bash
# Create virtual environment (recommended)
python -m venv unet_env
source unet_env/bin/activate  # On Windows: unet_env\Scripts\activate

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install numpy pillow matplotlib pandas
```

#### Option 2: Install with GPU Support (For NVIDIA GPUs)

**Prerequisites:** NVIDIA GPU with CUDA-capable driver installed

```bash
# Create virtual environment
python -m venv unet_env
source unet_env/bin/activate

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy pillow matplotlib pandas
```

**Check GPU availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

#### Option 3: Quick Install (Auto-detect CPU/GPU)

```bash
# This will install PyTorch with CUDA support if available
pip install torch torchvision numpy pillow matplotlib pandas
```

---

## Usage Guide

### 1. Loading the Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

# === Step 1: Define the U-Net architecture ===
# (Copy the ConvBlock and UNet classes from train_pytorch_comparison_no_aug.py)

class ConvBlock(nn.Module):
    """Standard convolution block"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class UNet(nn.Module):
    """Standard UNet"""
    def __init__(self, in_channels=1, n_filters=32, dropout=0.1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, n_filters, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(n_filters, n_filters * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(n_filters * 4, n_filters * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = ConvBlock(n_filters * 8, n_filters * 16, dropout)
        # Decoder
        self.up4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 2, stride=2)
        self.dec4 = ConvBlock(n_filters * 16, n_filters * 8, dropout)
        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 2, stride=2)
        self.dec3 = ConvBlock(n_filters * 8, n_filters * 4, dropout)
        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, stride=2)
        self.dec2 = ConvBlock(n_filters * 4, n_filters * 2, dropout)
        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, 2, stride=2)
        self.dec1 = ConvBlock(n_filters * 2, n_filters, dropout)
        # Output
        self.out = nn.Conv2d(n_filters, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

# === Step 2: Load model configuration ===
with open('best_models_PyTorch/unet/model_info.json', 'r') as f:
    model_info = json.load(f)

n_filters = model_info['n_filters']      # 32
dropout = model_info['dropout']          # 0.2

# === Step 3: Initialize model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet(in_channels=1, n_filters=n_filters, dropout=dropout)

# === Step 4: Load trained weights ===
checkpoint = torch.load(
    'best_models_PyTorch/unet/best_model.pth',
    map_location=device  # Important: loads on CPU if no GPU available
)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()  # Set to evaluation mode (critical!)

print(f"Model loaded successfully!")
print(f"Validation IoU: {checkpoint.get('val_iou', model_info['best_val_iou']):.2%}")
```

### 2. Preprocessing Function

**CRITICAL:** Images must be preprocessed exactly as during training!

```python
import numpy as np
from PIL import Image

def percentile_norm(arr: np.ndarray):
    """
    Percentile normalization (0.5th to 99.5th percentile)
    This MUST match the training preprocessing!
    """
    lo, hi = np.percentile(arr, [0.5, 99.5]).astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)

def load_and_preprocess_image(image_path, target_size=512):
    """
    Load and preprocess image for model input

    Args:
        image_path: Path to image file
        target_size: Target size (default 512, matching training)

    Returns:
        tensor: [1, 1, 512, 512] PyTorch tensor ready for model
        original_size: (width, height) of original image
    """
    # Load as grayscale
    image = Image.open(image_path).convert('L')
    original_size = image.size  # (width, height)

    # Convert to numpy and normalize
    arr = np.array(image, dtype=np.float32)
    arr_norm = percentile_norm(arr)

    # Convert to tensor [1, H, W]
    tensor = torch.from_numpy(arr_norm).unsqueeze(0)

    # Resize to target size if needed
    if tensor.shape[1] != target_size or tensor.shape[2] != target_size:
        tensor = F.interpolate(
            tensor.unsqueeze(0),           # [1, 1, H, W]
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)                       # Back to [1, H, W]

    # Add batch dimension: [1, 1, H, W]
    tensor = tensor.unsqueeze(0)

    return tensor, original_size
```

### 3. Running Inference

#### Single Image Prediction

```python
def predict_single_image(model, image_path, device, threshold=0.5):
    """
    Predict segmentation mask for a single image

    Args:
        model: Trained UNet model
        image_path: Path to input image
        device: torch.device ('cpu' or 'cuda')
        threshold: Binarization threshold (default 0.5)

    Returns:
        mask: Binary mask as numpy array [H, W] with values {0, 1}
        prob_map: Probability map [H, W] with values [0, 1]
    """
    # Load and preprocess
    input_tensor, original_size = load_and_preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_tensor)        # [1, 1, 512, 512]
        prob_map = torch.sigmoid(logits)    # Convert to probabilities

    # Convert to numpy
    prob_map = prob_map.squeeze().cpu().numpy()  # [512, 512]

    # Binarize
    mask = (prob_map > threshold).astype(np.uint8)

    return mask, prob_map

# Example usage
image_path = 'path/to/your/image.tif'
mask, prob_map = predict_single_image(model, image_path, device)

print(f"Mask shape: {mask.shape}")
print(f"Foreground pixels: {mask.sum()} ({100*mask.mean():.2f}%)")

# Save results
Image.fromarray(mask * 255).save('output_mask.png')
Image.fromarray((prob_map * 255).astype(np.uint8)).save('output_probability.png')
```

#### Batch Prediction

```python
def predict_batch(model, image_paths, device, threshold=0.5, batch_size=8):
    """
    Predict on multiple images efficiently

    Args:
        model: Trained UNet model
        image_paths: List of image paths
        device: torch.device
        threshold: Binarization threshold
        batch_size: Number of images to process at once

    Returns:
        masks: List of binary masks
        prob_maps: List of probability maps
    """
    from torch.utils.data import Dataset, DataLoader

    class InferenceDataset(Dataset):
        def __init__(self, image_paths):
            self.image_paths = image_paths

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            tensor, _ = load_and_preprocess_image(self.image_paths[idx])
            return tensor.squeeze(0), self.image_paths[idx]  # Remove batch dim

    dataset = InferenceDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    masks = []
    prob_maps = []

    with torch.no_grad():
        for batch, paths in dataloader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            for prob in probs:
                prob = prob.squeeze()  # [512, 512]
                mask = (prob > threshold).astype(np.uint8)
                masks.append(mask)
                prob_maps.append(prob)

    return masks, prob_maps

# Example usage
image_paths = [f'image_{i}.tif' for i in range(10)]
masks, prob_maps = predict_batch(model, image_paths, device, batch_size=4)
```

### 4. Post-processing and Visualization

```python
import matplotlib.pyplot as plt

def visualize_prediction(image_path, mask, prob_map):
    """Visualize input, mask, and probability map"""
    # Load original image
    original = np.array(Image.open(image_path).convert('L'))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Binary mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'Predicted Mask ({mask.sum()} pixels)')
    axes[1].axis('off')

    # Probability map
    im = axes[2].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Probability Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=150)
    plt.show()

# Example
visualize_prediction('test_image.tif', mask, prob_map)
```

#### Create Overlay Visualization

```python
def create_overlay(image_path, mask, alpha=0.5):
    """
    Create overlay of mask on original image

    Args:
        image_path: Path to original image
        mask: Binary mask [H, W]
        alpha: Transparency of overlay (0=invisible, 1=opaque)

    Returns:
        overlay: RGB image with mask overlaid in green
    """
    # Load original
    original = Image.open(image_path).convert('RGB')
    original = np.array(original)

    # Resize mask to match original if needed
    if mask.shape != original.shape[:2]:
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((original.shape[1], original.shape[0]), Image.NEAREST)
        mask = np.array(mask_pil)

    # Create green overlay
    overlay = original.copy()
    overlay[mask > 0] = [0, 255, 0]  # Green

    # Blend
    result = (alpha * overlay + (1 - alpha) * original).astype(np.uint8)

    return result

# Example
overlay_img = create_overlay('test_image.tif', mask, alpha=0.4)
Image.fromarray(overlay_img).save('overlay.png')
```

### 5. Counting Microbeads

```python
from scipy import ndimage

def count_microbeads(mask, min_size=10, max_size=500):
    """
    Count individual microbeads using connected component analysis

    Args:
        mask: Binary mask [H, W]
        min_size: Minimum bead size in pixels
        max_size: Maximum bead size in pixels

    Returns:
        count: Number of detected beads
        labeled: Labeled image [H, W] with each bead having unique ID
        sizes: List of bead sizes in pixels
    """
    # Label connected components
    labeled, num_features = ndimage.label(mask)

    # Filter by size
    sizes = []
    valid_labels = []

    for label_id in range(1, num_features + 1):
        size = (labeled == label_id).sum()
        if min_size <= size <= max_size:
            sizes.append(size)
            valid_labels.append(label_id)

    # Create filtered mask
    filtered_labeled = np.zeros_like(labeled)
    for i, label_id in enumerate(valid_labels, start=1):
        filtered_labeled[labeled == label_id] = i

    count = len(valid_labels)

    return count, filtered_labeled, sizes

# Example
count, labeled, sizes = count_microbeads(mask, min_size=20, max_size=1000)
print(f"Detected {count} microbeads")
print(f"Average size: {np.mean(sizes):.1f} pixels")
print(f"Size range: {min(sizes)} - {max(sizes)} pixels")
```

---

## Performance Optimization Tips

### For CPU-Only Laptops

1. **Reduce image size** (if acceptable):
```python
# Process at 256×256 instead of 512×512 (4× faster)
tensor, _ = load_and_preprocess_image(image_path, target_size=256)
```

2. **Process in batches** with smaller batch size:
```python
# Use batch_size=1 or 2 to avoid memory issues
masks, probs = predict_batch(model, paths, device, batch_size=1)
```

3. **Use half precision** (if supported):
```python
# Convert model to FP16 (2× memory reduction, faster on some CPUs)
model = model.half()
input_tensor = input_tensor.half()
```

### For GPU Laptops

1. **Larger batch sizes**:
```python
# Process more images at once
masks, probs = predict_batch(model, paths, device, batch_size=16)
```

2. **Mixed precision inference**:
```python
from torch.cuda.amp import autocast

with torch.no_grad():
    with autocast():  # Automatic mixed precision
        logits = model(input_tensor)
```

3. **Keep model on GPU**:
```python
# Keep model on GPU between predictions
model = model.to('cuda')
model.eval()

# Process multiple images without moving model
for image_path in image_paths:
    input_tensor = load_and_preprocess_image(image_path)[0].to('cuda')
    with torch.no_grad():
        logits = model(input_tensor)
```

---

## Expected Runtime Performance

### CPU (Intel i7 / AMD Ryzen 5)
- **Single 512×512 image:** 2-5 seconds
- **Batch of 10 images:** 20-50 seconds
- **Memory usage:** ~500MB-1GB

### GPU (NVIDIA GTX 1650 / RTX 3050)
- **Single 512×512 image:** 0.1-0.3 seconds
- **Batch of 10 images:** 1-3 seconds
- **Memory usage:** ~1-2GB VRAM

### GPU (NVIDIA RTX 3070 / RTX 4060)
- **Single 512×512 image:** 0.05-0.1 seconds
- **Batch of 10 images:** 0.5-1 second
- **Memory usage:** ~1.5-2GB VRAM

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```python
# Reduce batch size
batch_size = 1

# Or process at smaller resolution
tensor, _ = load_and_preprocess_image(image_path, target_size=256)

# Or use CPU
device = torch.device('cpu')
```

### Issue: "Poor prediction quality"

**Possible causes:**
1. **Wrong preprocessing:** Ensure percentile normalization is used
2. **Wrong image size:** Model expects 512×512 (or close to it)
3. **Different image characteristics:** Model trained on specific microscopy images

**Debug steps:**
```python
# Check input range
print(f"Input min: {input_tensor.min()}, max: {input_tensor.max()}")
# Should be approximately [0, 1]

# Check model output
logits = model(input_tensor)
print(f"Logits min: {logits.min()}, max: {logits.max()}")
print(f"Probabilities min: {torch.sigmoid(logits).min()}, max: {torch.sigmoid(logits).max()}")
```

### Issue: "Model file not found"

**Solution:** After modifying `.gitignore`, you need to:
1. Copy `best_model.pth` from HPC to `best_models_PyTorch/unet/`
2. Or sync from your training experiment directory
```bash
# From HPC
scp user@hpc:~/path/to/experiment/best_model.pth best_models_PyTorch/unet/
```

### Issue: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"

**Cause:** Model architecture mismatch

**Solution:** Ensure `n_filters` and `dropout` match `model_info.json`:
```python
# Must use exact values from model_info.json
model = UNet(in_channels=1, n_filters=32, dropout=0.2)
```

---

## Complete Standalone Script

Create `predict.py`:

```python
#!/usr/bin/env python3
"""
Standalone script for microbead segmentation using trained U-Net model
Usage: python predict.py --input image.tif --output mask.png
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path

# === Model Architecture ===
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_filters=32, dropout=0.1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, n_filters, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(n_filters, n_filters * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(n_filters * 4, n_filters * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(n_filters * 8, n_filters * 16, dropout)
        self.up4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 2, stride=2)
        self.dec4 = ConvBlock(n_filters * 16, n_filters * 8, dropout)
        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 2, stride=2)
        self.dec3 = ConvBlock(n_filters * 8, n_filters * 4, dropout)
        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, stride=2)
        self.dec2 = ConvBlock(n_filters * 4, n_filters * 2, dropout)
        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, 2, stride=2)
        self.dec1 = ConvBlock(n_filters * 2, n_filters, dropout)
        self.out = nn.Conv2d(n_filters, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

# === Preprocessing ===
def percentile_norm(arr):
    lo, hi = np.percentile(arr, [0.5, 99.5]).astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)

def load_image(image_path, target_size=512):
    image = Image.open(image_path).convert('L')
    arr = np.array(image, dtype=np.float32)
    arr_norm = percentile_norm(arr)
    tensor = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0)
    if tensor.shape[2] != target_size or tensor.shape[3] != target_size:
        tensor = F.interpolate(tensor, size=(target_size, target_size),
                             mode='bilinear', align_corners=False)
    return tensor

# === Main ===
def main():
    parser = argparse.ArgumentParser(description='Microbead segmentation')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', default='mask.png', help='Output mask path')
    parser.add_argument('--model-dir', default='best_models_PyTorch/unet',
                       help='Model directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binarization threshold')
    parser.add_argument('--device', default='auto',
                       help='Device: cpu, cuda, or auto')
    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model config
    model_dir = Path(args.model_dir)
    with open(model_dir / 'model_info.json', 'r') as f:
        model_info = json.load(f)

    # Initialize and load model
    model = UNet(in_channels=1,
                n_filters=model_info['n_filters'],
                dropout=model_info['dropout'])
    checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (IoU: {model_info['best_val_iou']:.2%})")

    # Load and predict
    input_tensor = load_image(args.input).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Binarize and save
    mask = (prob_map > args.threshold).astype(np.uint8) * 255
    Image.fromarray(mask).save(args.output)
    print(f"Saved mask to {args.output}")
    print(f"Foreground: {(mask > 0).sum()} pixels ({100*(mask>0).mean():.2f}%)")

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
python predict.py --input test_image.tif --output mask.png
python predict.py --input test_image.tif --output mask.png --threshold 0.3
python predict.py --input test_image.tif --device cpu
```

---

## Citation & License

**Model Training:** October 2025
**Architecture:** U-Net (Ronneberger et al., 2015)
**Implementation:** PyTorch
**Application:** Microbead segmentation in microscopy images

If you use this model, please cite the original U-Net paper:
```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention},
  pages={234--241},
  year={2015}
}
```

---

## Support & Contact

For questions or issues:
1. Check the [main project documentation](../../docs/)
2. Review [training analysis reports](../../docs/)
3. Examine the [visualization guide](../../Visualizing_U-Net.md)
4. See [architecture comparison](../../docs/ARCHITECTURE_COMPARISON_GUIDE.md)

---

**Last Updated:** November 11, 2025
**Model Version:** v1.0 (pytorch_comparison_no_aug_20251021_121918)
**Performance:** 63.77% IoU on validation set
