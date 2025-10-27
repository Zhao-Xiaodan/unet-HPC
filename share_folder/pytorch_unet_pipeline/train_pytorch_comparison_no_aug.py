#!/usr/bin/env python3
"""
PyTorch Training: Fair Comparison with Keras Models (No Augmentation)
======================================================================

This script enables direct comparison with Keras models by using:
1. Same preprocessing as train.py (grayscale, percentile normalization)
2. Same dataset split (80/20 from ./dataset_shrunk_masks)
3. Same loss function as Keras (BinaryFocalLoss)
4. NO data augmentation (matching Keras training)
5. Training all 3 architectures: UNet, Attention UNet, Attention ResUNet

Key differences from original train.py:
- No student-teacher distillation
- No AdaptiveBGDiceLoss (uses BinaryFocalLoss like Keras)
- No augmentation
- Grid search over same hyperparameters as Keras

Author: Claude Code
Date: October 19, 2025
"""

import os
import json
import math
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

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    'images_dir': './dataset_shrunk_masks/images/',
    'masks_dir': './dataset_shrunk_masks/masks/',
    'train_val_split': 0.8,  # 80% train, 20% validation

    # Output
    'output_dir_base': './pytorch_comparison_no_aug',

    # Image settings
    'img_size': 512,
    'img_channels': 1,  # Grayscale (matching train.py preprocessing)

    # Architectures to train
    'architectures': ['unet', 'attention_unet', 'attention_resunet'],

    # Hyperparameter grid (same as Keras)
    'hyperparam_grid': {
        'n_filters': [16, 32, 64],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.003, 0.005],
    },

    # Training settings
    'epochs': 100,
    'batch_size': 4,
    'early_stopping_patience': 20,
    'reduce_lr_patience': 10,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,

    # Loss function (matching Keras)
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,

    # Random seed for reproducibility
    'random_seed': 42,
}

# ============================================================================
# PREPROCESSING (Same as train.py)
# ============================================================================

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

        # Find paired images and masks
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

        def list_images(root: Path):
            files = []
            for p in sorted(root.iterdir()):
                if p.name.startswith("."):
                    continue
                if p.is_dir():
                    continue
                if p.suffix.lower() in exts:
                    files.append(p)
            return files

        img_files = list_images(self.img_dir)
        mask_files = list_images(self.mask_dir)

        img_map = {p.stem: p for p in img_files}
        mask_map = {p.stem: p for p in mask_files}
        common = sorted(set(img_map.keys()) & set(mask_map.keys()))

        if not common:
            raise RuntimeError(f"No paired images/masks found under {self.img_dir} and {self.mask_dir}")

        self.pairs = [(img_map[k], mask_map[k]) for k in common]
        print(f"[Dataset] Found {len(self.pairs)} image-mask pairs")

    def __len__(self):
        return len(self.pairs)

    def _load_gray01(self, p: Path):
        """Load and normalize (same as train.py)"""
        if not p.is_file():
            raise FileNotFoundError(f"Not a file: {p}")
        im = Image.open(str(p)).convert("L")
        arr = np.array(im, dtype=np.float32)
        arr = _percentile_norm(arr)
        return arr  # [H,W] float32 [0,1]

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load with percentile normalization
        img01 = self._load_gray01(img_path)
        mask01 = self._load_gray01(mask_path)

        # Convert to tensors
        image = torch.from_numpy(img01).unsqueeze(0)  # [1, H, W]
        mask = torch.from_numpy(mask01).unsqueeze(0)  # [1, H, W]

        # Resize to target size
        if image.shape[1] != self.img_size or image.shape[2] != self.img_size:
            image = F.interpolate(image.unsqueeze(0), size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=(self.img_size, self.img_size),
                               mode="bilinear", align_corners=False).squeeze(0)

        return image.float(), mask.float()

# ============================================================================
# LOSS FUNCTION (BinaryFocalLoss - matching Keras)
# ============================================================================

class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss (matching Keras implementation)"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        """
        Args:
            logits: [B, 1, H, W] - raw model outputs (before sigmoid)
            target: [B, 1, H, W] - ground truth masks [0, 1]
        """
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        # Focal weight
        p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        pt = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weight
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Combine
        focal_loss = alpha_weight * focal_weight * bce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# METRICS (IoU and Dice - matching Keras)
# ============================================================================

def iou_score(pred, target, smooth=1e-6):
    """Intersection over Union (Jaccard coefficient)"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def dice_score(pred, target, smooth=1e-6):
    """Dice coefficient (F1 score)"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)

    return dice.mean()

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class ConvBlock(nn.Module):
    """Standard convolution block (matching Keras UNet)"""

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

class ResConvBlock(nn.Module):
    """Residual convolution block (matching Keras Attention ResUNet)"""

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
    """Attention gate (matching Keras Attention UNet)"""

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

        # Upsample attention to match x
        psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=False)

        return x * psi

class UNet(nn.Module):
    """Standard UNet (matching Keras implementation)"""

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

class AttentionUNet(nn.Module):
    """Attention UNet (matching Keras implementation)"""

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

        # Attention gates (F_g is gating signal channels AFTER upsampling)
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
        self.out = nn.Conv2d(n_filters, 1, 1)

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

        return self.out(d1)

class AttentionResUNet(nn.Module):
    """Attention Residual UNet (matching Keras implementation)"""

    def __init__(self, in_channels=1, n_filters=32, dropout=0.1):
        super().__init__()

        # Encoder
        self.enc1 = ResConvBlock(in_channels, n_filters, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResConvBlock(n_filters, n_filters * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResConvBlock(n_filters * 2, n_filters * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResConvBlock(n_filters * 4, n_filters * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResConvBlock(n_filters * 8, n_filters * 16, dropout)

        # Attention gates (F_g is gating signal channels AFTER upsampling)
        self.att4 = AttentionGate(F_g=n_filters * 8, F_l=n_filters * 8, F_int=n_filters * 4)
        self.att3 = AttentionGate(F_g=n_filters * 4, F_l=n_filters * 4, F_int=n_filters * 2)
        self.att2 = AttentionGate(F_g=n_filters * 2, F_l=n_filters * 2, F_int=n_filters)
        self.att1 = AttentionGate(F_g=n_filters, F_l=n_filters, F_int=n_filters // 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 2, stride=2)
        self.dec4 = ResConvBlock(n_filters * 16, n_filters * 8, dropout)

        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 2, stride=2)
        self.dec3 = ResConvBlock(n_filters * 8, n_filters * 4, dropout)

        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, stride=2)
        self.dec2 = ResConvBlock(n_filters * 4, n_filters * 2, dropout)

        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, 2, stride=2)
        self.dec1 = ResConvBlock(n_filters * 2, n_filters, dropout)

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

        return self.out(d1)

def build_model(arch_name, n_filters, dropout):
    """Build model by architecture name"""
    if arch_name == 'unet':
        return UNet(in_channels=1, n_filters=n_filters, dropout=dropout)
    elif arch_name == 'attention_unet':
        return AttentionUNet(in_channels=1, n_filters=n_filters, dropout=dropout)
    elif arch_name == 'attention_resunet':
        return AttentionResUNet(in_channels=1, n_filters=n_filters, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        with torch.no_grad():
            running_loss += loss.item()
            running_iou += iou_score(outputs, masks).item()
            running_dice += dice_score(outputs, masks).item()

    n = len(dataloader)
    return {
        'loss': running_loss / n,
        'iou': running_iou / n,
        'dice': running_dice / n
    }

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(enabled=False):
            outputs = model(images)
            loss = criterion(outputs, masks)

        running_loss += loss.item()
        running_iou += iou_score(outputs, masks).item()
        running_dice += dice_score(outputs, masks).item()

    n = len(dataloader)
    return {
        'loss': running_loss / n,
        'iou': running_iou / n,
        'dice': running_dice / n
    }

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_model(arch_name, n_filters, dropout, learning_rate, config, train_loader, val_loader, device):
    """Train a single model configuration"""

    # Model name
    model_name = f"{arch_name}_n_filters{n_filters}_dropout{dropout}_learning_rate{learning_rate}"
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")

    # Build model
    model = build_model(arch_name, n_filters, dropout).to(device)

    # Loss and optimizer
    criterion = BinaryFocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['reduce_lr_factor'],
        patience=config['reduce_lr_patience'],
        min_lr=config['min_lr'],
        verbose=True
    )

    # Output directories
    output_dir = Path(config['output_dir']) / arch_name
    checkpoint_dir = output_dir / 'checkpoints' / model_name
    log_dir = output_dir / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # History
    history = {
        'train_loss': [], 'train_iou': [], 'train_dice': [],
        'val_loss': [], 'val_iou': [], 'val_dice': []
    }

    # Training loop
    best_val_iou = 0.0
    patience_counter = 0

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_iou'].append(train_metrics['iou'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_dice'].append(val_metrics['dice'])

        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")

        # Learning rate scheduler
        scheduler.step(val_metrics['iou'])

        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            patience_counter = 0

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'history': history
            }, checkpoint_dir / 'best_model.pth')

            print(f"✓ Saved best model (val_iou: {best_val_iou:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Save final history
    history_df = pd.DataFrame(history)
    history_df.to_csv(log_dir / f"{model_name}_history.csv", index=False)

    return best_val_iou, model_name

def main():
    """Main training function"""

    # Set random seeds
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create timestamp-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{CONFIG['output_dir_base']}_{timestamp}"
    CONFIG['output_dir'] = output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    with open(Path(output_dir) / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)

    print(f"\nOutput directory: {output_dir}")

    # Load dataset
    print("\nLoading dataset...")
    full_dataset = BeadsDataset(
        img_dir=CONFIG['images_dir'],
        mask_dir=CONFIG['masks_dir'],
        img_size=CONFIG['img_size']
    )

    # Split dataset (80/20 with fixed seed)
    total_size = len(full_dataset)
    train_size = int(CONFIG['train_val_split'] * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['random_seed'])
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=8,  # Increased for 36 CPUs
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=8,  # Increased for 36 CPUs
        pin_memory=True,
        drop_last=False
    )

    # Train all architectures and hyperparameter combinations
    results = []

    for arch in CONFIG['architectures']:
        print(f"\n\n{'#'*80}")
        print(f"# Training Architecture: {arch.upper()}")
        print(f"{'#'*80}")

        for n_filters in CONFIG['hyperparam_grid']['n_filters']:
            for dropout in CONFIG['hyperparam_grid']['dropout']:
                for lr in CONFIG['hyperparam_grid']['learning_rate']:

                    best_iou, model_name = train_model(
                        arch, n_filters, dropout, lr,
                        CONFIG, train_loader, val_loader, device
                    )

                    results.append({
                        'architecture': arch,
                        'n_filters': n_filters,
                        'dropout': dropout,
                        'learning_rate': lr,
                        'best_val_iou': best_iou,
                        'model_name': model_name
                    })

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(output_dir) / 'all_results.csv', index=False)

    # Find best model per architecture
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}\n")

    for arch in CONFIG['architectures']:
        arch_results = results_df[results_df['architecture'] == arch]
        best_row = arch_results.loc[arch_results['best_val_iou'].idxmax()]

        print(f"{arch.upper()}:")
        print(f"  Best Val IoU: {best_row['best_val_iou']:.4f}")
        print(f"  n_filters: {best_row['n_filters']}")
        print(f"  dropout: {best_row['dropout']}")
        print(f"  learning_rate: {best_row['learning_rate']}")
        print(f"  model_name: {best_row['model_name']}")
        print()

    print(f"\nAll results saved to: {output_dir}")
    print("Training complete!")

if __name__ == "__main__":
    main()
