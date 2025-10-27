#!/usr/bin/env python3
"""
PyTorch Model Prediction for Density Analysis
==============================================

Loads best models from PyTorch comparison experiments and generates predictions
for test images. Supports UNet, Attention UNet, and Attention ResUNet.

Usage:
    python predict_pytorch_comparison.py --experiment <exp_dir> --output <output_dir>

Example:
    python predict_pytorch_comparison.py \\
        --experiment pytorch_comparison_no_aug_20251021_121918 \\
        --output pytorch_density_predictions

Author: Claude Code
Date: October 22, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import argparse
import shutil

# ============================================================================
# MODEL ARCHITECTURES (copied from training script)
# ============================================================================

class ConvBlock(nn.Module):
    """Standard convolution block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class ResConvBlock(nn.Module):
    """Residual convolution block (matching training script)"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)
        return out

class AttentionGate(nn.Module):
    """Attention gate (matching training script)"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, stride=1, padding=0)
        self.W_x = nn.Conv2d(F_l, F_int, 2, stride=2, padding=0)
        self.psi = nn.Conv2d(F_int, 1, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        """
        g: gating signal from decoder
        x: skip connection from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Align dimensions
        if g1.shape[2] != x1.shape[2] or g1.shape[3] != x1.shape[3]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))

        # Upsample attention map to match skip connection size
        psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=False)

        return x * psi

class UNet(nn.Module):
    """Standard U-Net architecture"""
    def __init__(self, n_channels=1, n_filters=32, dropout=0.2):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(n_channels, n_filters, dropout)
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
        self.out = nn.Conv2d(n_filters, 1, kernel_size=1)

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

        return torch.sigmoid(self.out(d1))

class AttentionUNet(nn.Module):
    """U-Net with Attention Gates"""
    def __init__(self, n_channels=1, n_filters=32, dropout=0.2):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(n_channels, n_filters, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(n_filters, n_filters * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(n_filters * 4, n_filters * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(n_filters * 8, n_filters * 16, dropout)

        # Attention gates (FIXED channel counts)
        self.att4 = AttentionGate(F_g=n_filters * 8, F_l=n_filters * 8, F_int=n_filters * 4)
        self.att3 = AttentionGate(F_g=n_filters * 4, F_l=n_filters * 4, F_int=n_filters * 2)
        self.att2 = AttentionGate(F_g=n_filters * 2, F_l=n_filters * 2, F_int=n_filters)
        self.att1 = AttentionGate(F_g=n_filters, F_l=n_filters, F_int=n_filters // 2)

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
        self.out = nn.Conv2d(n_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with attention
        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out(d1))

class AttentionResUNet(nn.Module):
    """U-Net with Residual Blocks and Attention Gates"""
    def __init__(self, n_channels=1, n_filters=32, dropout=0.2):
        super().__init__()

        # Encoder with residual blocks
        self.enc1 = ResConvBlock(n_channels, n_filters, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResConvBlock(n_filters, n_filters * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResConvBlock(n_filters * 2, n_filters * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResConvBlock(n_filters * 4, n_filters * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResConvBlock(n_filters * 8, n_filters * 16, dropout)

        # Attention gates (FIXED channel counts)
        self.att4 = AttentionGate(F_g=n_filters * 8, F_l=n_filters * 8, F_int=n_filters * 4)
        self.att3 = AttentionGate(F_g=n_filters * 4, F_l=n_filters * 4, F_int=n_filters * 2)
        self.att2 = AttentionGate(F_g=n_filters * 2, F_l=n_filters * 2, F_int=n_filters)
        self.att1 = AttentionGate(F_g=n_filters, F_l=n_filters, F_int=n_filters // 2)

        # Decoder with residual blocks
        self.up4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 2, stride=2)
        self.dec4 = ResConvBlock(n_filters * 16, n_filters * 8, dropout)
        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 2, stride=2)
        self.dec3 = ResConvBlock(n_filters * 8, n_filters * 4, dropout)
        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, stride=2)
        self.dec2 = ResConvBlock(n_filters * 4, n_filters * 2, dropout)
        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, 2, stride=2)
        self.dec1 = ResConvBlock(n_filters * 2, n_filters, dropout)

        # Output
        self.out = nn.Conv2d(n_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with attention
        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out(d1))

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def percentile_normalize(image, lower=0.5, upper=99.5):
    """Percentile normalization (same as training)"""
    p_low, p_high = np.percentile(image, [lower, upper])
    image = np.clip(image, p_low, p_high)
    image = (image - p_low) / (p_high - p_low + 1e-8)
    return image

def load_test_image(image_path):
    """Load test image and convert to grayscale"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def extract_tiles(image, tile_size=512):
    """Extract non-overlapping tiles from image"""
    h, w = image.shape[:2]
    tiles = []
    positions = []

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions

def find_and_cache_best_models(experiment_dir, cache_dir='./best_models_PyTorch'):
    """
    Find best models and cache them in a dedicated directory for future use.

    First checks cache_dir for existing best models.
    If not found, searches experiment_dir and copies best models to cache_dir.

    Args:
        experiment_dir: Directory containing trained models
        cache_dir: Directory to store/load cached best models

    Returns:
        dict mapping architecture name to model path and metadata
    """
    cache_dir = Path(cache_dir)
    experiment_dir = Path(experiment_dir)

    # Check if cache exists and has all three architectures
    if cache_dir.exists():
        cached_models = {}
        all_cached = True

        for arch in ['unet', 'attention_unet', 'attention_resunet']:
            arch_cache = cache_dir / arch
            checkpoint_file = arch_cache / 'best_model.pth'
            metadata_file = arch_cache / 'model_info.json'

            if checkpoint_file.exists() and metadata_file.exists():
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                cached_models[arch] = {
                    'checkpoint_path': checkpoint_file,
                    'n_filters': metadata['n_filters'],
                    'dropout': metadata['dropout'],
                    'learning_rate': metadata['learning_rate'],
                    'best_val_iou': metadata['best_val_iou'],
                    'model_name': metadata['model_name']
                }

                print(f"✓ {arch}: Loaded from cache (IoU={metadata['best_val_iou']:.4f})")
            else:
                all_cached = False
                break

        if all_cached and len(cached_models) == 3:
            print(f"\n✓ All models loaded from cache: {cache_dir}")
            return cached_models
        else:
            print(f"\n⚠ Cache incomplete, searching experiment directory...")
    else:
        print(f"\n⚠ Cache not found, searching experiment directory for best models...")

    # Cache doesn't exist or is incomplete - search experiment directory
    results_file = experiment_dir / 'all_results.csv'

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Load results
    df = pd.read_csv(results_file)

    best_models = {}

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    for arch in ['unet', 'attention_unet', 'attention_resunet']:
        arch_df = df[df['architecture'] == arch]

        if len(arch_df) == 0:
            print(f"Warning: No results found for {arch}")
            continue

        # Get best model
        best_row = arch_df.loc[arch_df['best_val_iou'].idxmax()]

        model_name = best_row['model_name']
        checkpoint_path = experiment_dir / arch / 'checkpoints' / model_name / 'best_model.pth'

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found for {arch}: {checkpoint_path}")
            continue

        # Create architecture subdirectory in cache
        arch_cache_dir = cache_dir / arch
        arch_cache_dir.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint to cache
        cached_checkpoint = arch_cache_dir / 'best_model.pth'
        print(f"  Copying {arch} checkpoint to cache...")
        shutil.copy2(checkpoint_path, cached_checkpoint)

        # Save metadata
        metadata = {
            'n_filters': int(best_row['n_filters']),
            'dropout': float(best_row['dropout']),
            'learning_rate': float(best_row['learning_rate']),
            'best_val_iou': float(best_row['best_val_iou']),
            'model_name': model_name,
            'source_experiment': str(experiment_dir),
            'cached_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(arch_cache_dir / 'model_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        best_models[arch] = {
            'checkpoint_path': cached_checkpoint,
            'n_filters': metadata['n_filters'],
            'dropout': metadata['dropout'],
            'learning_rate': metadata['learning_rate'],
            'best_val_iou': metadata['best_val_iou'],
            'model_name': model_name
        }

        print(f"✓ {arch}: IoU={best_row['best_val_iou']:.4f}, "
              f"n_filters={int(best_row['n_filters'])}, "
              f"dropout={best_row['dropout']}, lr={best_row['learning_rate']}")

    print(f"\n✓ Best models cached to: {cache_dir}")
    print(f"  (Future runs will load from cache without searching)")

    return best_models

def load_model(arch_name, checkpoint_path, n_filters, dropout, device):
    """Load model from checkpoint"""
    # Create model
    if arch_name == 'unet':
        model = UNet(n_channels=1, n_filters=n_filters, dropout=dropout)
    elif arch_name == 'attention_unet':
        model = AttentionUNet(n_channels=1, n_filters=n_filters, dropout=dropout)
    elif arch_name == 'attention_resunet':
        model = AttentionResUNet(n_channels=1, n_filters=n_filters, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model

def predict_tiles(model, tiles, device, batch_size=8):
    """Predict on list of tiles"""
    predictions = []

    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i+batch_size]

        # Preprocess
        batch_tensor = []
        for tile in batch_tiles:
            # Normalize
            tile_norm = percentile_normalize(tile.astype(np.float32))
            # To tensor [1, H, W]
            tile_tensor = torch.from_numpy(tile_norm).unsqueeze(0)
            batch_tensor.append(tile_tensor)

        # Stack to [B, 1, H, W]
        batch_tensor = torch.stack(batch_tensor).to(device)

        # Predict
        with torch.no_grad():
            batch_pred = model(batch_tensor)

        # Convert to numpy [B, H, W]
        batch_pred = batch_pred.squeeze(1).cpu().numpy()
        predictions.extend(batch_pred)

    return predictions

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PyTorch Model Prediction for Density Analysis')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment directory containing trained models')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for predictions')
    parser.add_argument('--test_images', type=str, default='./test_images',
                       help='Directory containing test images')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for prediction')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find and cache best models
    print("="*80)
    print("FINDING BEST MODELS")
    print("="*80)
    best_models = find_and_cache_best_models(args.experiment, cache_dir='./best_models_PyTorch')
    print()

    if len(best_models) == 0:
        print("ERROR: No valid models found!")
        return

    # Load models
    print("="*80)
    print("LOADING MODELS")
    print("="*80)
    models = {}
    for arch_name, info in best_models.items():
        print(f"Loading {arch_name}...")
        model = load_model(
            arch_name,
            info['checkpoint_path'],
            info['n_filters'],
            info['dropout'],
            device
        )
        models[arch_name] = model
    print()

    # Get test images
    test_image_dir = Path(args.test_images)
    test_images = sorted(test_image_dir.glob('*.tif'))

    print("="*80)
    print(f"PROCESSING {len(test_images)} TEST IMAGES")
    print("="*80)
    print()

    # Process each image
    for img_path in tqdm(test_images, desc="Processing images"):
        # Load image
        image = load_test_image(img_path)

        # Extract tiles
        tiles, positions = extract_tiles(image, tile_size=512)

        # Predict with each architecture
        for arch_name, model in models.items():
            predictions = predict_tiles(model, tiles, device, args.batch_size)

            # Save predictions
            arch_output_dir = output_dir / arch_name
            arch_output_dir.mkdir(exist_ok=True)

            # Reconstruct full image
            h, w = image.shape
            full_pred = np.zeros((h, w), dtype=np.float32)

            for pred, (y, x) in zip(predictions, positions):
                full_pred[y:y+512, x:x+512] = pred

            # Save as 8-bit PNG
            pred_uint8 = (full_pred * 255).astype(np.uint8)
            output_path = arch_output_dir / f"{img_path.stem}_pred.png"
            cv2.imwrite(str(output_path), pred_uint8)

    # Save metadata
    metadata = {
        'experiment_dir': str(args.experiment),
        'test_images_dir': str(args.test_images),
        'n_test_images': len(test_images),
        'models': {k: {
            'model_name': v['model_name'],
            'n_filters': v['n_filters'],
            'dropout': v['dropout'],
            'best_val_iou': v['best_val_iou']
        } for k, v in best_models.items()},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(output_dir / 'prediction_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("="*80)
    print("PREDICTION COMPLETE")
    print("="*80)
    print(f"Predictions saved to: {output_dir}")
    print(f"Architectures: {list(models.keys())}")
    print(f"Images processed: {len(test_images)}")

if __name__ == "__main__":
    main()
