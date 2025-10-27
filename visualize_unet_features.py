#!/usr/bin/env python3
"""
U-Net Feature Visualization for PyTorch Models
==============================================

This script visualizes the internal workings of trained U-Net models by showing:
1. Feature maps at each encoder and decoder layer
2. Reconstructed input from layer activations (feature inversion)

The script processes a representative 512x512 tile from test images and generates:
- Feature map visualizations for all encoder/decoder layers
- Feature inversion results showing what each layer "sees"

Based on models trained by train_pytorch_comparison_no_aug.py

Author: Claude Code
Date: October 27, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import argparse
from tqdm import tqdm

# ============================================================================
# PREPROCESSING (Same as train_pytorch_comparison_no_aug.py)
# ============================================================================

def _percentile_norm(arr: np.ndarray):
    """Percentile normalization (same as train.py)"""
    lo, hi = np.percentile(arr, [0.5, 99.5]).astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)

def preprocess_image(img_path, img_size=512):
    """
    Loads and preprocesses a single image, matching BeadsDataset implementation.

    Args:
        img_path: Path to image file
        img_size: Target size (default 512)

    Returns:
        tensor: [1, 1, H, W] preprocessed image tensor
    """
    # Load as grayscale
    im = Image.open(str(img_path)).convert("L")
    arr = np.array(im, dtype=np.float32)

    # Apply percentile normalization
    arr_norm = _percentile_norm(arr)

    # Convert to tensor
    tensor = torch.from_numpy(arr_norm).unsqueeze(0)  # [1, H, W]

    # Resize if needed
    if tensor.shape[1] != img_size or tensor.shape[2] != img_size:
        tensor = F.interpolate(tensor.unsqueeze(0),
                             size=(img_size, img_size),
                             mode="bilinear",
                             align_corners=False).squeeze(0)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)  # [1, 1, H, W]
    return tensor

def postprocess_tensor(tensor):
    """
    Converts a tensor from [0, 1] range back to a plottable numpy image.
    """
    image = tensor.detach().cpu().squeeze().numpy()
    image = np.clip(image, 0, 1)
    return image

# ============================================================================
# MODEL ARCHITECTURES (Same as train_pytorch_comparison_no_aug.py)
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
# FEATURE VISUALIZATION: HOOKS AND PLOTTING
# ============================================================================

def plot_feature_maps(maps, layer_name, output_path, max_channels=16):
    """
    Plot feature maps from a layer.

    Args:
        maps: [1, C, H, W] tensor
        layer_name: Name of the layer
        output_path: Path to save the plot
        max_channels: Maximum number of channels to plot (default 16)
    """
    maps = maps.cpu().squeeze(0)  # Remove batch dim: [C, H, W]
    num_channels = maps.shape[0]

    # Limit to max_channels
    channels_to_plot = min(num_channels, max_channels)

    # Determine grid size
    cols = 8 if channels_to_plot > 8 else channels_to_plot
    rows = int(np.ceil(channels_to_plot / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(f"Feature Maps: {layer_name} (Showing {channels_to_plot}/{num_channels} channels)",
                 fontsize=14, fontweight='bold')

    # Flatten axes array for easy iteration
    if rows > 1:
        axes = axes.flatten()
    elif channels_to_plot == 1:
        axes = [axes]

    for i in range(channels_to_plot):
        ax = axes[i]
        fmap = maps[i].detach().numpy()

        # Plot with viridis colormap
        im = ax.imshow(fmap, cmap='viridis')
        ax.set_title(f"Ch {i}", fontsize=9)
        ax.axis('off')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off unused subplots
    for j in range(channels_to_plot, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()

def visualize_feature_maps(model, image_tensor, output_dir, device):
    """
    Visualize feature maps at each encoder and decoder layer using hooks.

    Args:
        model: Trained model
        image_tensor: Input image [1, 1, H, W]
        output_dir: Output directory for plots
        device: torch device
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nVisualizing feature maps...")

    # Dictionary to store activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks for all encoder and decoder layers
    layers_to_visualize = {
        'encoder_1': model.enc1,
        'encoder_2': model.enc2,
        'encoder_3': model.enc3,
        'encoder_4': model.enc4,
        'bottleneck': model.bottleneck,
        'decoder_4': model.dec4,
        'decoder_3': model.dec3,
        'decoder_2': model.dec2,
        'decoder_1': model.dec1,
    }

    hooks = {}
    for name, layer in layers_to_visualize.items():
        hooks[name] = layer.register_forward_hook(get_activation(name))

    # Run forward pass
    with torch.no_grad():
        _ = model(image_tensor.to(device))

    # Plot feature maps
    for name, activation in tqdm(activations.items(), desc="Plotting feature maps"):
        output_path = output_dir / f"feature_map_{name}.png"
        plot_feature_maps(activation, name, output_path, max_channels=16)

    # Remove hooks
    for handle in hooks.values():
        handle.remove()

    print(f"✓ Saved {len(activations)} feature map visualizations to: {output_dir}")

# ============================================================================
# FEATURE INVERSION: RECONSTRUCT INPUT FROM LAYER ACTIVATIONS
# ============================================================================

def total_variation_loss(img):
    """
    Computes the Total Variation Loss for a batch of images.
    Helps to reduce noise in the generated image.
    """
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (img.shape[0] * img.shape[1] * img.shape[2] * img.shape[3])

def reconstruct_from_layer(model, target_image_tensor, layer_name, layer_module, device,
                           n_steps=2000, lr=0.01, tv_weight=0.001):
    """
    Reconstruct input image from layer activations using feature inversion.

    Args:
        model: Trained model
        target_image_tensor: Target image [1, 1, H, W]
        layer_name: Name of the layer
        layer_module: Layer module to reconstruct from
        device: torch device
        n_steps: Number of optimization steps
        lr: Learning rate
        tv_weight: Total variation loss weight

    Returns:
        optimized_image: Reconstructed image
        target_feature_map: Target feature map
    """
    print(f"\n--- Starting Feature Inversion for: {layer_name} ---")

    # Step 1: Get target feature map
    target_activations = {}

    def get_target_hook(name):
        def hook(model, input, output):
            target_activations[name] = output.detach()
        return hook

    hook_handle = layer_module.register_forward_hook(get_target_hook(layer_name))

    # Run forward pass on the REAL image
    with torch.no_grad():
        _ = model(target_image_tensor.to(device))

    # This is the feature map we want to match
    target_feature_map = target_activations[layer_name].clone()
    hook_handle.remove()

    print(f"Target feature map shape: {target_feature_map.shape}")

    # Step 2: Create image to optimize
    # Start with random noise
    optimized_image = torch.randn(
        target_image_tensor.shape,
        device=device,
        requires_grad=True
    )

    # Step 3: Setup optimizer
    optimizer = torch.optim.Adam([optimized_image], lr=lr)

    # Register hook for current activations
    current_activations = {}

    def get_current_hook(name):
        def hook(model, input, output):
            current_activations[name] = output  # Keep in graph
        return hook

    hook_handle = layer_module.register_forward_hook(get_current_hook(layer_name))

    # Loss function
    feature_loss_fn = nn.MSELoss()

    # Step 4: Optimization loop
    for i in range(n_steps + 1):
        optimizer.zero_grad()

        # Pass the noise image through the model
        _ = model(optimized_image)

        # Get the feature map for the noise image
        current_feature_map = current_activations[layer_name]

        # Calculate loss
        feature_loss = feature_loss_fn(current_feature_map, target_feature_map)
        tv_loss = total_variation_loss(optimized_image)

        # Combine losses
        total_loss = feature_loss + (tv_weight * tv_loss)

        # Backpropagate
        total_loss.backward()
        optimizer.step()

        # Clamp image values to [0, 1] range after each step
        with torch.no_grad():
            optimized_image.data.clamp_(0, 1)

        if i % 200 == 0:
            print(f"  Step {i:4d}/{n_steps} | Total Loss: {total_loss.item():.4f} | "
                  f"Feature Loss: {feature_loss.item():.4f} | TV Loss: {tv_loss.item():.4f}")

    hook_handle.remove()
    print("Optimization finished.")

    return optimized_image, target_feature_map

def visualize_feature_inversions(model, image_tensor, output_dir, device):
    """
    Visualize feature inversions for selected layers.

    Args:
        model: Trained model
        image_tensor: Input image [1, 1, H, W]
        output_dir: Output directory for plots
        device: torch device
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nVisualizing feature inversions...")

    # Select key layers to invert
    layers_to_invert = {
        'encoder_1': model.enc1,
        'encoder_2': model.enc2,
        'encoder_3': model.enc3,
        'encoder_4': model.enc4,
        'bottleneck': model.bottleneck,
        'decoder_1': model.dec1,
    }

    original_img_plot = postprocess_tensor(image_tensor)

    for layer_name, layer_module in tqdm(layers_to_invert.items(), desc="Running feature inversions"):
        # Run feature inversion
        reconstructed_image, _ = reconstruct_from_layer(
            model, image_tensor, layer_name, layer_module, device,
            n_steps=2000, lr=0.01, tv_weight=0.001
        )

        # Plot results
        recon_img_plot = postprocess_tensor(reconstructed_image)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Feature Inversion from Layer: '{layer_name}'", fontsize=16, fontweight='bold')

        ax1.imshow(original_img_plot, cmap='gray')
        ax1.set_title("Original Image", fontsize=14)
        ax1.axis('off')

        ax2.imshow(recon_img_plot, cmap='gray')
        ax2.set_title("Reconstructed Image", fontsize=14)
        ax2.axis('off')

        plt.tight_layout()
        output_path = output_dir / f"feature_inversion_{layer_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()

    print(f"✓ Saved {len(layers_to_invert)} feature inversion visualizations to: {output_dir}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def find_best_model(cache_dir='./best_models_PyTorch'):
    """
    Find the best trained model in the cache directory.

    Returns:
        model_path: Path to best model checkpoint
        metadata: Model metadata (architecture, hyperparameters)
    """
    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Model cache directory not found: {cache_dir}\n"
            "Please run pbs_pytorch_density_analysis.sh first to cache best models."
        )

    # Look for UNet models (you can modify to use other architectures)
    unet_dir = cache_dir / 'unet'

    if not unet_dir.exists():
        raise FileNotFoundError(f"UNet models not found in: {cache_dir}")

    # Find best model (assumes single best model per architecture)
    model_files = list(unet_dir.glob('**/best_model.pth'))

    if not model_files:
        raise FileNotFoundError(f"No best_model.pth found in: {unet_dir}")

    model_path = model_files[0]

    # Extract hyperparameters from path
    # Path format: unet/unet_n_filters32_dropout0.1_learning_rate0.001/best_model.pth
    parent_dir = model_path.parent.name

    # Parse hyperparameters (basic parsing, adjust if needed)
    import re
    n_filters_match = re.search(r'n_filters(\d+)', parent_dir)
    dropout_match = re.search(r'dropout([\d.]+)', parent_dir)

    n_filters = int(n_filters_match.group(1)) if n_filters_match else 32
    dropout = float(dropout_match.group(1)) if dropout_match else 0.1

    metadata = {
        'architecture': 'unet',
        'n_filters': n_filters,
        'dropout': dropout,
        'model_path': str(model_path)
    }

    return model_path, metadata

def main():
    parser = argparse.ArgumentParser(description='Visualize U-Net Feature Maps')
    parser.add_argument('--model_cache', type=str, default='./best_models_PyTorch',
                       help='Directory containing cached best models')
    parser.add_argument('--test_image', type=str, required=True,
                       help='Path to test image (will extract 512x512 tile)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for visualizations')
    parser.add_argument('--tile_x', type=int, default=0,
                       help='X position of 512x512 tile to extract')
    parser.add_argument('--tile_y', type=int, default=0,
                       help='Y position of 512x512 tile to extract')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("U-NET FEATURE VISUALIZATION")
    print("="*80)
    print(f"Model cache: {args.model_cache}")
    print(f"Test image: {args.test_image}")
    print(f"Tile position: ({args.tile_x}, {args.tile_y})")
    print(f"Output: {args.output}")
    print("="*80)

    # Step 1: Find and load best model
    print("\n" + "="*80)
    print("STEP 1: LOADING MODEL")
    print("="*80)

    model_path, metadata = find_best_model(args.model_cache)
    print(f"\nFound model:")
    print(f"  Architecture: {metadata['architecture']}")
    print(f"  n_filters: {metadata['n_filters']}")
    print(f"  dropout: {metadata['dropout']}")
    print(f"  Path: {metadata['model_path']}")

    # Build model
    model = build_model(
        metadata['architecture'],
        metadata['n_filters'],
        metadata['dropout']
    ).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("\n✓ Model loaded successfully")

    # Step 2: Load and preprocess image
    print("\n" + "="*80)
    print("STEP 2: LOADING IMAGE")
    print("="*80)

    # Load full image
    full_image = Image.open(args.test_image).convert("L")
    full_array = np.array(full_image, dtype=np.float32)

    print(f"\nFull image shape: {full_array.shape}")

    # Extract 512x512 tile
    tile_size = 512
    tile = full_array[args.tile_y:args.tile_y+tile_size, args.tile_x:args.tile_x+tile_size]

    # Save tile as reference
    tile_img = Image.fromarray(tile.astype(np.uint8))
    tile_img.save(output_dir / 'input_tile_original.png')

    # Preprocess tile
    tile_norm = _percentile_norm(tile)
    tile_tensor = torch.from_numpy(tile_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Save preprocessed tile
    plt.figure(figsize=(6, 6))
    plt.imshow(tile_norm, cmap='gray')
    plt.title('Preprocessed Input Tile (512x512)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'input_tile_preprocessed.png', bbox_inches='tight', dpi=200)
    plt.close()

    print(f"Extracted tile: {tile.shape}")
    print(f"Preprocessed tile: {tile_tensor.shape}")
    print(f"✓ Saved input tiles to: {output_dir}")

    # Step 3: Generate prediction
    print("\n" + "="*80)
    print("STEP 3: GENERATING PREDICTION")
    print("="*80)

    with torch.no_grad():
        prediction = model(tile_tensor.to(device))
        prediction_sigmoid = torch.sigmoid(prediction)

    # Save prediction
    pred_img = postprocess_tensor(prediction_sigmoid)
    plt.figure(figsize=(6, 6))
    plt.imshow(pred_img, cmap='gray')
    plt.title('Model Prediction', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction.png', bbox_inches='tight', dpi=200)
    plt.close()

    print(f"✓ Saved prediction to: {output_dir}")

    # Step 4: Visualize feature maps
    print("\n" + "="*80)
    print("STEP 4: VISUALIZING FEATURE MAPS")
    print("="*80)

    feature_maps_dir = output_dir / 'feature_maps'
    visualize_feature_maps(model, tile_tensor, feature_maps_dir, device)

    # Step 5: Feature inversions
    print("\n" + "="*80)
    print("STEP 5: VISUALIZING FEATURE INVERSIONS")
    print("="*80)

    feature_inversions_dir = output_dir / 'feature_inversions'
    visualize_feature_inversions(model, tile_tensor, feature_inversions_dir, device)

    # Save metadata
    viz_metadata = {
        'model_metadata': metadata,
        'test_image': str(args.test_image),
        'tile_position': {'x': args.tile_x, 'y': args.tile_y},
        'tile_size': tile_size,
        'output_dir': str(args.output),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(output_dir / 'visualization_metadata.json', 'w') as f:
        json.dump(viz_metadata, f, indent=2)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  Input:")
    print("    - input_tile_original.png")
    print("    - input_tile_preprocessed.png")
    print("    - prediction.png")
    print("  Feature Maps:")
    print("    - feature_maps/ (9 layers)")
    print("  Feature Inversions:")
    print("    - feature_inversions/ (6 layers)")
    print("  - visualization_metadata.json")

if __name__ == "__main__":
    main()
