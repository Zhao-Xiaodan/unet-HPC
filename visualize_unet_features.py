#!/usr/bin/env python3
"""
U-Net Feature Visualization for PyTorch Models
==============================================

This script visualizes the internal workings of trained U-Net models by showing:
1. Feature maps at each encoder and decoder layer (both conv1 and conv2)
2. Reconstructed input from layer activations (feature inversion)

The script processes all test images and generates:
- Feature map visualizations for all encoder/decoder layers (both convolutions)
- Feature inversion results showing what each layer "sees"
- 3-panel input/preprocessed/prediction figures

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

def extract_center_tile(image_array, tile_size=512, row=3, col=4):
    """
    Extract a tile at specific row and column position.

    Args:
        image_array: Input image array [H, W]
        tile_size: Size of tile (default 512)
        row: Row index (1-indexed for user, 0-indexed internally)
        col: Column index (1-indexed for user, 0-indexed internally)

    Returns:
        tile: Extracted tile [tile_size, tile_size]
        position: (y, x) position of tile
    """
    h, w = image_array.shape[:2]

    # Calculate number of tiles that fit
    n_rows = h // tile_size
    n_cols = w // tile_size

    # Adjust row/col if they exceed available tiles
    row_idx = min(row - 1, n_rows - 1)  # Convert to 0-indexed
    col_idx = min(col - 1, n_cols - 1)  # Convert to 0-indexed

    # Calculate position
    y = row_idx * tile_size
    x = col_idx * tile_size

    # Extract tile
    tile = image_array[y:y+tile_size, x:x+tile_size]

    return tile, (y, x)

def preprocess_tile(tile_array):
    """
    Preprocess a tile matching BeadsDataset implementation.

    Args:
        tile_array: Tile array [H, W]

    Returns:
        tensor: [1, 1, H, W] preprocessed tensor
    """
    # Apply percentile normalization
    arr_norm = _percentile_norm(tile_array)

    # Convert to tensor
    tensor = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

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

def plot_feature_maps(maps, layer_name, output_path, max_channels=32):
    """
    Plot feature maps from a layer.

    Args:
        maps: [1, C, H, W] tensor
        layer_name: Name of the layer
        output_path: Path to save the plot
        max_channels: Maximum number of channels to plot (default 32)
    """
    maps = maps.cpu().squeeze(0)  # Remove batch dim: [C, H, W]
    num_channels = maps.shape[0]

    # Limit to max_channels
    channels_to_plot = min(num_channels, max_channels)

    # Determine grid size
    cols = 8
    rows = int(np.ceil(channels_to_plot / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(f"Feature Maps: {layer_name}\n(Showing {channels_to_plot}/{num_channels} channels)",
                 fontsize=16, fontweight='bold')

    # Flatten axes array for easy iteration
    if rows > 1:
        axes = axes.flatten()
    elif channels_to_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(channels_to_plot):
        ax = axes[i]
        fmap = maps[i].detach().numpy()

        # Plot with viridis colormap
        im = ax.imshow(fmap, cmap='viridis')
        ax.set_title(f"Ch {i}", fontsize=10)
        ax.axis('off')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off unused subplots
    for j in range(channels_to_plot, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def register_conv_hooks(model):
    """
    Register hooks for BOTH conv1 and conv2 in each ConvBlock/ResConvBlock.

    Returns:
        layers_dict: Dictionary mapping layer names to modules
    """
    layers_dict = {}

    # Helper function to get conv layers from a block
    def get_conv_layers(block, prefix):
        if hasattr(block, 'conv1') and hasattr(block, 'bn1'):
            # Register hook after conv1+bn1+relu
            layers_dict[f'{prefix}_conv1'] = block.bn1
        if hasattr(block, 'conv2') and hasattr(block, 'bn2'):
            # Register hook after conv2+bn2+relu (or before adding residual)
            layers_dict[f'{prefix}_conv2'] = block.bn2

    # Encoder blocks
    get_conv_layers(model.enc1, 'encoder_1')
    get_conv_layers(model.enc2, 'encoder_2')
    get_conv_layers(model.enc3, 'encoder_3')
    get_conv_layers(model.enc4, 'encoder_4')

    # Bottleneck
    get_conv_layers(model.bottleneck, 'bottleneck')

    # Decoder blocks
    get_conv_layers(model.dec4, 'decoder_4')
    get_conv_layers(model.dec3, 'decoder_3')
    get_conv_layers(model.dec2, 'decoder_2')
    get_conv_layers(model.dec1, 'decoder_1')

    return layers_dict

def visualize_feature_maps(model, image_tensor, output_dir, device):
    """
    Visualize feature maps at each conv layer using hooks.

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
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks for all conv layers
    layers_to_visualize = register_conv_hooks(model)

    hooks = {}
    for name, layer in layers_to_visualize.items():
        hooks[name] = layer.register_forward_hook(get_activation(name))

    # Run forward pass
    with torch.no_grad():
        _ = model(image_tensor.to(device))

    # Plot feature maps
    for name, activation in tqdm(activations.items(), desc="Plotting feature maps"):
        output_path = output_dir / f"feature_map_{name}.png"
        plot_feature_maps(activation, name, output_path, max_channels=32)

    # Remove hooks
    for handle in hooks.values():
        handle.remove()

    print(f"  ✓ Saved {len(activations)} feature map visualizations to: {output_dir}")

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
    print(f"  Starting feature inversion: {layer_name}")

    # Step 1: Get target feature map
    target_activations = {}

    def get_target_hook(name):
        def hook(module, input, output):
            target_activations[name] = output.detach()
        return hook

    hook_handle = layer_module.register_forward_hook(get_target_hook(layer_name))

    # Run forward pass on the REAL image
    with torch.no_grad():
        _ = model(target_image_tensor.to(device))

    # This is the feature map we want to match
    target_feature_map = target_activations[layer_name].clone()
    hook_handle.remove()

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
        def hook(module, input, output):
            current_activations[name] = output  # Keep in graph
        return hook

    hook_handle = layer_module.register_forward_hook(get_current_hook(layer_name))

    # Loss function
    feature_loss_fn = nn.MSELoss()

    # Step 4: Optimization loop (silent, only print at key milestones)
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

    hook_handle.remove()

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

    print("\nVisualizing feature inversions (this may take 30-60 minutes)...")

    # Get all conv layers
    all_layers = register_conv_hooks(model)

    # Select key layers to invert (all encoder and decoder conv2 layers + bottleneck)
    layers_to_invert = {
        'encoder_1_conv2': all_layers['encoder_1_conv2'],
        'encoder_2_conv2': all_layers['encoder_2_conv2'],
        'encoder_3_conv2': all_layers['encoder_3_conv2'],
        'encoder_4_conv2': all_layers['encoder_4_conv2'],
        'bottleneck_conv2': all_layers['bottleneck_conv2'],
        'decoder_4_conv2': all_layers['decoder_4_conv2'],
        'decoder_3_conv2': all_layers['decoder_3_conv2'],
        'decoder_2_conv2': all_layers['decoder_2_conv2'],
        'decoder_1_conv2': all_layers['decoder_1_conv2'],
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
        fig.suptitle(f"Feature Inversion: {layer_name}", fontsize=16, fontweight='bold')

        ax1.imshow(original_img_plot, cmap='gray')
        ax1.set_title("Original Image", fontsize=14)
        ax1.axis('off')

        ax2.imshow(recon_img_plot, cmap='gray')
        ax2.set_title("Reconstructed Image", fontsize=14)
        ax2.axis('off')

        plt.tight_layout()
        output_path = output_dir / f"feature_inversion_{layer_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

    print(f"  ✓ Saved {len(layers_to_invert)} feature inversion visualizations to: {output_dir}")

def create_3panel_figure(tile_original, tile_preprocessed, prediction, image_name, output_path):
    """
    Create a 3-panel figure showing original, preprocessed, and prediction.

    Args:
        tile_original: Original tile array
        tile_preprocessed: Preprocessed tile tensor
        prediction: Prediction tensor
        image_name: Name of original image
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    fig.suptitle(f"Input and Prediction: {image_name}", fontsize=18, fontweight='bold', y=0.98)

    # Panel 1: Original tile
    axes[0].imshow(tile_original, cmap='gray')
    axes[0].set_title("Original Tile (512x512)", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Preprocessed tile
    tile_prep_img = postprocess_tensor(tile_preprocessed)
    axes[1].imshow(tile_prep_img, cmap='gray')
    axes[1].set_title("Preprocessed (Percentile Norm)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Prediction
    pred_img = postprocess_tensor(prediction)
    axes[2].imshow(pred_img, cmap='gray')
    axes[2].set_title("Model Prediction", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

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

def process_single_image(test_image_path, model, metadata, output_base_dir, device, tile_row=3, tile_col=4):
    """
    Process a single test image: extract tile, generate visualizations.

    Args:
        test_image_path: Path to test image
        model: Trained model
        metadata: Model metadata
        output_base_dir: Base output directory
        device: torch device
        tile_row: Row index for tile extraction (1-indexed)
        tile_col: Column index for tile extraction (1-indexed)
    """
    # Create output subdirectory named by image
    image_name = test_image_path.stem
    image_output_dir = output_base_dir / image_name
    image_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Processing: {image_name}")
    print(f"{'='*80}")

    # Load full image
    full_image = Image.open(test_image_path).convert("L")
    full_array = np.array(full_image, dtype=np.float32)

    print(f"  Full image shape: {full_array.shape}")

    # Extract center tile
    tile, (tile_y, tile_x) = extract_center_tile(full_array, tile_size=512, row=tile_row, col=tile_col)
    print(f"  Extracted tile at position: row={tile_row}, col={tile_col} (pixel position: y={tile_y}, x={tile_x})")

    # Preprocess tile
    tile_tensor = preprocess_tile(tile)

    # Generate prediction
    print("  Generating prediction...")
    with torch.no_grad():
        prediction = model(tile_tensor.to(device))
        prediction_sigmoid = torch.sigmoid(prediction)

    # Create 3-panel figure
    print("  Creating 3-panel input/prediction figure...")
    create_3panel_figure(
        tile, tile_tensor, prediction_sigmoid, image_name,
        image_output_dir / f"{image_name}_3panel.png"
    )

    # Visualize feature maps
    print("  Visualizing feature maps...")
    feature_maps_dir = image_output_dir / 'feature_maps'
    visualize_feature_maps(model, tile_tensor, feature_maps_dir, device)

    # Feature inversions
    print("  Visualizing feature inversions...")
    feature_inversions_dir = image_output_dir / 'feature_inversions'
    visualize_feature_inversions(model, tile_tensor, feature_inversions_dir, device)

    # Save tile metadata
    tile_metadata = {
        'image_name': image_name,
        'full_image_shape': full_array.shape,
        'tile_position': {'y': int(tile_y), 'x': int(tile_x)},
        'tile_row': tile_row,
        'tile_col': tile_col,
        'tile_size': 512
    }

    with open(image_output_dir / 'tile_metadata.json', 'w') as f:
        json.dump(tile_metadata, f, indent=2)

    print(f"  ✓ Completed processing: {image_name}")

def main():
    parser = argparse.ArgumentParser(description='Visualize U-Net Feature Maps')
    parser.add_argument('--model_cache', type=str, default='./best_models_PyTorch',
                       help='Directory containing cached best models')
    parser.add_argument('--test_images', type=str, default='./test_images',
                       help='Directory containing test images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for visualizations')
    parser.add_argument('--tile_row', type=int, default=3,
                       help='Row index for tile extraction (1-indexed, default: 3)')
    parser.add_argument('--tile_col', type=int, default=4,
                       help='Column index for tile extraction (1-indexed, default: 4)')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("U-NET FEATURE VISUALIZATION - ALL TEST IMAGES")
    print("="*80)
    print(f"Model cache: {args.model_cache}")
    print(f"Test images: {args.test_images}")
    print(f"Tile position: row {args.tile_row}, column {args.tile_col}")
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

    # Step 2: Find all test images
    print("\n" + "="*80)
    print("STEP 2: FINDING TEST IMAGES")
    print("="*80)

    test_images_dir = Path(args.test_images)
    test_images = sorted(test_images_dir.glob('*.tif'))

    if not test_images:
        print(f"ERROR: No .tif images found in {test_images_dir}")
        return

    print(f"\nFound {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img.name}")

    # Step 3: Process all images
    print("\n" + "="*80)
    print("STEP 3: PROCESSING ALL TEST IMAGES")
    print("="*80)

    for test_image_path in test_images:
        process_single_image(
            test_image_path, model, metadata, output_dir, device,
            tile_row=args.tile_row, tile_col=args.tile_col
        )

    # Save overall metadata
    viz_metadata = {
        'model_metadata': metadata,
        'test_images_dir': str(args.test_images),
        'num_images_processed': len(test_images),
        'tile_position': {'row': args.tile_row, 'col': args.tile_col},
        'tile_size': 512,
        'output_dir': str(args.output),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(output_dir / 'visualization_metadata.json', 'w') as f:
        json.dump(viz_metadata, f, indent=2)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"\nProcessed {len(test_images)} images:")
    for img in test_images:
        img_name = img.stem
        print(f"\n  {img_name}/")
        print(f"    - {img_name}_3panel.png (original + preprocessed + prediction)")
        print(f"    - feature_maps/ (18 images: encoder_1-4, bottleneck, decoder_1-4, each with conv1 & conv2)")
        print(f"    - feature_inversions/ (9 images: all conv2 layers)")
        print(f"    - tile_metadata.json")

    print(f"\n  - visualization_metadata.json")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
