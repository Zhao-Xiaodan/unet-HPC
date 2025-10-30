#!/usr/bin/env python3
"""
Edge Detector Transfer Learning Experiment
==========================================

Controlled experiment to test whether Gabor filter initialization helps U-Net
performance for cell counting.

Two training runs:
1. Frozen layer 1: Gabor filters stay fixed throughout training
2. Trainable layer 1: Gabor filters can adapt during training

Baseline comparison: Existing U-Net with random initialization (IoU=0.6377)

Author: Claude Code
Date: October 30, 2025
"""

import os
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm

# Import Gabor initializer
from gabor_initializer import (
    create_gabor_filter_bank,
    initialize_layer_with_gabor,
    visualize_filter_bank
)

#=============================================================================
# CONFIGURATION
#=============================================================================

CONFIG = {
    # Experiment settings
    'experiment_name': 'edge_detector_experiment',
    'freeze_layer1': False,  # Will be set by command line argument

    # Data paths
    'images_dir': './dataset_shrunk_masks/images/',
    'masks_dir': './dataset_shrunk_masks/masks/',
    'train_val_split': 0.8,

    # Output
    'output_dir_base': './edge_detector_experiment',

    # Image settings
    'img_size': 512,
    'img_channels': 1,

    # Best hyperparameters (from best_models_PyTorch/unet/model_info.json)
    'n_filters': 32,
    'dropout': 0.2,
    'learning_rate': 0.001,

    # Training settings (same as train_pytorch_comparison_no_aug.py)
    'epochs': 100,
    'batch_size': 4,
    'early_stopping_patience': 20,
    'reduce_lr_patience': 10,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,

    # Loss function (matching Keras)
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,

    # Random seed
    'random_seed': 42,

    # Layer 1 weight tracking
    'save_layer1_weights_every': 10,  # epochs
}

#=============================================================================
# PREPROCESSING (Same as train.py)
#=============================================================================

def _percentile_norm(arr: np.ndarray):
    """Percentile normalization (same as train.py)"""
    lo, hi = np.percentile(arr, [0.5, 99.5]).astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)

class BeadsDataset(Dataset):
    """Dataset with same preprocessing as train.py but NO augmentation"""

    def __init__(self, img_dir, mask_dir, img_size=512):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size

        # Get all image files
        self.img_files = sorted([f for f in self.img_dir.glob('*.png')])

        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {img_dir}")

        print(f"Found {len(self.img_files)} images")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img = Image.open(img_path)

        # Convert to grayscale numpy array
        img_array = np.array(img)

        # Handle different image formats
        if len(img_array.shape) == 3:
            # RGB to grayscale
            img_array = np.mean(img_array, axis=2)

        # Normalize
        img_array = _percentile_norm(img_array)

        # Resize if needed
        if img_array.shape != (self.img_size, self.img_size):
            img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
            img_pil = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_array = np.array(img_pil) / 255.0

        # Load mask
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            raise ValueError(f"Mask not found: {mask_path}")

        mask = Image.open(mask_path)
        mask_array = np.array(mask)

        # Binarize mask
        mask_array = (mask_array > 127).astype(np.float32)

        # Resize mask if needed
        if mask_array.shape != (self.img_size, self.img_size):
            mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((self.img_size, self.img_size), Image.NEAREST)
            mask_array = (np.array(mask_pil) > 127).astype(np.float32)

        # Convert to torch tensors [C, H, W]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()

        return img_tensor, mask_tensor

#=============================================================================
# MODEL ARCHITECTURE
#=============================================================================

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
    """Standard UNet with optional Gabor initialization and layer freezing"""

    def __init__(self, in_channels=1, n_filters=32, dropout=0.1,
                 gabor_init=True, freeze_layer1=False):
        super().__init__()

        self.gabor_init = gabor_init
        self.freeze_layer1 = freeze_layer1

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

        # Initialize with Gabor filters if requested
        if gabor_init:
            self._initialize_gabor()

        # Freeze layer 1 if requested
        if freeze_layer1:
            self._freeze_layer1()

    def _initialize_gabor(self):
        """Initialize enc1.conv1 with Gabor filters"""
        print("\n" + "="*70)
        print("Initializing encoder layer 1 with Gabor filters...")
        print("="*70)
        initialize_layer_with_gabor(self.enc1.conv1, verbose=True)
        print("="*70)

    def _freeze_layer1(self):
        """Freeze all parameters in enc1"""
        print("\n" + "="*70)
        print("Freezing encoder layer 1...")
        print("="*70)
        for name, param in self.enc1.named_parameters():
            param.requires_grad = False
            print(f"  ✓ Frozen: enc1.{name} (shape: {list(param.shape)})")
        print("="*70)

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

    def get_layer1_weights(self):
        """Extract layer 1 conv weights for tracking"""
        return self.enc1.conv1.weight.data.clone().cpu()

#=============================================================================
# LOSS FUNCTION
#=============================================================================

class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss (matching Keras)"""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)

        # Focal loss formula
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha balancing
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        # Binary cross entropy
        bce = -torch.log(pt)

        # Combine
        loss = alpha_t * focal_weight * bce
        return loss.mean()

#=============================================================================
# METRICS
#=============================================================================

def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.item()

#=============================================================================
# TRAINING
#=============================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_iou = 0

    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item()
        total_iou += compute_iou(outputs, masks)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    return avg_loss, avg_iou

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_iou += compute_iou(outputs, masks)

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    return avg_loss, avg_iou

#=============================================================================
# MAIN TRAINING FUNCTION
#=============================================================================

def train_experiment(freeze_layer1=False):
    """
    Main training function for edge detector experiment

    Args:
        freeze_layer1: If True, freeze enc1 with Gabor filters
    """
    # Set random seeds
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    variant = 'frozen_layer1' if freeze_layer1 else 'trainable_layer1'
    output_dir = Path(CONFIG['output_dir_base']) / f"{timestamp}" / f"unet_{variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"EDGE DETECTOR EXPERIMENT: {variant.upper()}")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print("="*70)

    # Save Gabor filter visualization
    gabor_filters = create_gabor_filter_bank(
        n_filters=CONFIG['n_filters'],
        kernel_size=3,
        visualize=True,
        save_path=output_dir / 'gabor_filters_initial.png'
    )

    # Save experiment configuration
    config_save = CONFIG.copy()
    config_save['freeze_layer1'] = freeze_layer1
    config_save['variant'] = variant
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(config_save, f, indent=2)

    # Create dataset
    print("\nLoading dataset...")
    full_dataset = BeadsDataset(
        CONFIG['images_dir'],
        CONFIG['masks_dir'],
        CONFIG['img_size']
    )

    # Train/val split
    train_size = int(CONFIG['train_val_split'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )

    print(f"Train size: {train_size}, Val size: {val_size}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = UNet(
        in_channels=CONFIG['img_channels'],
        n_filters=CONFIG['n_filters'],
        dropout=CONFIG['dropout'],
        gabor_init=True,
        freeze_layer1=freeze_layer1
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")

    # Save initial layer 1 weights
    initial_layer1_weights = model.get_layer1_weights()
    torch.save(initial_layer1_weights, output_dir / 'layer1_weights_epoch000.pth')
    print(f"\n✓ Saved initial layer 1 weights")

    # Loss and optimizer
    criterion = BinaryFocalLoss(
        gamma=CONFIG['focal_gamma'],
        alpha=CONFIG['focal_alpha']
    )

    # Only optimize trainable parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG['reduce_lr_factor'],
        patience=CONFIG['reduce_lr_patience'],
        min_lr=CONFIG['min_lr'],
        verbose=True
    )

    # Mixed precision training
    scaler = GradScaler()

    # Training loop
    best_val_iou = 0.0
    patience_counter = 0
    history = []

    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")

        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )

        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        print(f"LR: {current_lr:.2e}")

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'lr': current_lr,
        })

        # Scheduler step
        scheduler.step(val_loss)

        # Save layer 1 weights periodically
        if (epoch + 1) % CONFIG['save_layer1_weights_every'] == 0:
            layer1_weights = model.get_layer1_weights()
            torch.save(layer1_weights,
                      output_dir / f'layer1_weights_epoch{epoch+1:03d}.pth')
            print(f"✓ Saved layer 1 weights (epoch {epoch+1})")

        # Early stopping & checkpointing
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'freeze_layer1': freeze_layer1,
            }, output_dir / 'best_model.pth')
            print(f"✓ New best model saved (IoU: {best_val_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\nEarly stopping triggered (patience: {CONFIG['early_stopping_patience']})")
                break

    # Save final layer 1 weights
    final_layer1_weights = model.get_layer1_weights()
    torch.save(final_layer1_weights, output_dir / 'layer1_weights_final.pth')

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)

    # Save model info
    model_info = {
        'variant': variant,
        'freeze_layer1': freeze_layer1,
        'n_filters': CONFIG['n_filters'],
        'dropout': CONFIG['dropout'],
        'learning_rate': CONFIG['learning_rate'],
        'best_val_iou': best_val_iou,
        'total_epochs': len(history),
        'baseline_iou': 0.6377,  # For comparison
        'improvement_vs_baseline': best_val_iou - 0.6377,
    }
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Variant: {variant}")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Baseline IoU: 0.6377")
    print(f"Improvement: {best_val_iou - 0.6377:+.4f}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    return best_val_iou, output_dir

#=============================================================================
# CLI
#=============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Edge Detector Transfer Learning Experiment'
    )
    parser.add_argument('--freeze_layer1', action='store_true',
                       help='Freeze layer 1 with Gabor filters')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory')

    args = parser.parse_args()

    if args.output_dir:
        CONFIG['output_dir_base'] = args.output_dir

    # Run experiment
    best_iou, output_dir = train_experiment(freeze_layer1=args.freeze_layer1)

    print("\n✓ Experiment complete!")
    print(f"\nNext steps:")
    print(f"1. Run feature visualization:")
    print(f"   python unet_feature_visualization.py \\")
    print(f"       --model_path {output_dir}/best_model.pth \\")
    print(f"       --n_filters {CONFIG['n_filters']} \\")
    print(f"       --dropout {CONFIG['dropout']}")
    print()
