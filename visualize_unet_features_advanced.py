#!/usr/bin/env python3
"""
U-Net Feature Visualization - Advanced Version
===============================================

Enhanced visualization with:
1. Feature inversions at correct spatial resolutions (matching layer dimensions)
2. UMAP clustering to identify representative feature maps

Features:
- Inverts from dimensions matching each layer (512→256→128→64→32)
- Clusters similar feature maps using UMAP + KMeans
- Shows only representative feature maps per cluster (reduces redundancy)
- Generates UMAP scatter plots showing feature map clusters

Author: Claude Code
Date: October 27, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import argparse
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, provide fallback
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: UMAP not available. Install with: pip install umap-learn")

# ============================================================================
# PREPROCESSING (Same as training)
# ============================================================================

def _percentile_norm(arr: np.ndarray):
    """Percentile normalization"""
    lo, hi = np.percentile(arr, [0.5, 99.5]).astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return arr.astype(np.float32)

def extract_center_tile(image_array, tile_size=512, row=3, col=4):
    """Extract tile at specific row/column position"""
    h, w = image_array.shape[:2]
    n_rows = h // tile_size
    n_cols = w // tile_size
    row_idx = min(row - 1, n_rows - 1)
    col_idx = min(col - 1, n_cols - 1)
    y = row_idx * tile_size
    x = col_idx * tile_size
    tile = image_array[y:y+tile_size, x:x+tile_size]
    return tile, (y, x)

def preprocess_tile(tile_array):
    """Preprocess tile"""
    arr_norm = _percentile_norm(tile_array)
    tensor = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0)
    return tensor

def postprocess_tensor(tensor):
    """Convert tensor to plottable image"""
    image = tensor.detach().cpu().squeeze().numpy()
    image = np.clip(image, 0, 1)
    return image

# ============================================================================
# MODEL ARCHITECTURES (Identical to training code)
# ============================================================================

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

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
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
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, stride=1, padding=0)
        self.W_x = nn.Conv2d(F_l, F_int, 2, stride=2, padding=0)
        self.psi = nn.Conv2d(F_int, 1, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2] != x1.shape[2] or g1.shape[3] != x1.shape[3]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=False)
        return x * psi

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

class AttentionUNet(nn.Module):
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
        self.att4 = AttentionGate(F_g=n_filters * 8, F_l=n_filters * 8, F_int=n_filters * 4)
        self.att3 = AttentionGate(F_g=n_filters * 4, F_l=n_filters * 4, F_int=n_filters * 2)
        self.att2 = AttentionGate(F_g=n_filters * 2, F_l=n_filters * 2, F_int=n_filters)
        self.att1 = AttentionGate(F_g=n_filters, F_l=n_filters, F_int=n_filters // 2)
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
    def __init__(self, in_channels=1, n_filters=32, dropout=0.1):
        super().__init__()
        self.enc1 = ResConvBlock(in_channels, n_filters, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResConvBlock(n_filters, n_filters * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResConvBlock(n_filters * 2, n_filters * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResConvBlock(n_filters * 4, n_filters * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ResConvBlock(n_filters * 8, n_filters * 16, dropout)
        self.att4 = AttentionGate(F_g=n_filters * 8, F_l=n_filters * 8, F_int=n_filters * 4)
        self.att3 = AttentionGate(F_g=n_filters * 4, F_l=n_filters * 4, F_int=n_filters * 2)
        self.att2 = AttentionGate(F_g=n_filters * 2, F_l=n_filters * 2, F_int=n_filters)
        self.att1 = AttentionGate(F_g=n_filters, F_l=n_filters, F_int=n_filters // 2)
        self.up4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 2, stride=2)
        self.dec4 = ResConvBlock(n_filters * 16, n_filters * 8, dropout)
        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 2, stride=2)
        self.dec3 = ResConvBlock(n_filters * 8, n_filters * 4, dropout)
        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, stride=2)
        self.dec2 = ResConvBlock(n_filters * 4, n_filters * 2, dropout)
        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, 2, stride=2)
        self.dec1 = ResConvBlock(n_filters * 2, n_filters, dropout)
        self.out = nn.Conv2d(n_filters, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
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
    if arch_name == 'unet':
        return UNet(in_channels=1, n_filters=n_filters, dropout=dropout)
    elif arch_name == 'attention_unet':
        return AttentionUNet(in_channels=1, n_filters=n_filters, dropout=dropout)
    elif arch_name == 'attention_resunet':
        return AttentionResUNet(in_channels=1, n_filters=n_filters, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

# ============================================================================
# LAYER DIMENSION MAPPING
# ============================================================================

def get_layer_dimensions(n_filters=32):
    """
    Return expected spatial dimensions for each layer.

    Architecture flow:
    - Input: 512x512
    - After pool1: 256x256 (enc2)
    - After pool2: 128x128 (enc3)
    - After pool3: 64x64 (enc4)
    - After pool4: 32x32 (bottleneck)
    - After up4: 64x64 (dec4)
    - After up3: 128x128 (dec3)
    - After up2: 256x256 (dec2)
    - After up1: 512x512 (dec1)
    """
    return {
        'encoder_1_conv2': (512, n_filters),
        'encoder_2_conv2': (256, n_filters * 2),
        'encoder_3_conv2': (128, n_filters * 4),
        'encoder_4_conv2': (64, n_filters * 8),
        'bottleneck_conv2': (32, n_filters * 16),
        'decoder_4_conv2': (64, n_filters * 8),
        'decoder_3_conv2': (128, n_filters * 4),
        'decoder_2_conv2': (256, n_filters * 2),
        'decoder_1_conv2': (512, n_filters),
    }

# ============================================================================
# UMAP CLUSTERING FOR FEATURE MAPS
# ============================================================================

def cluster_feature_maps(activations_dict, n_clusters=8):
    """
    Cluster feature maps using UMAP + KMeans to find representatives.

    Args:
        activations_dict: Dict of {layer_name: activation_tensor}
        n_clusters: Number of clusters to form

    Returns:
        representatives: Dict of {layer_name: [(channel_idx, cluster_id), ...]}
        umap_embeddings: Dict of {layer_name: umap_2d_coords}
        cluster_labels: Dict of {layer_name: cluster_labels_array}
    """
    if not UMAP_AVAILABLE:
        print("  WARNING: UMAP not available, returning all channels")
        # Return all channels without clustering
        representatives = {}
        for name, activation in activations_dict.items():
            n_channels = activation.shape[1]
            representatives[name] = [(i, 0) for i in range(n_channels)]
        return representatives, {}, {}

    representatives = {}
    umap_embeddings = {}
    cluster_labels_dict = {}

    for layer_name, activation in tqdm(activations_dict.items(), desc="Clustering feature maps"):
        # activation: [1, C, H, W]
        fmaps = activation.cpu().squeeze(0).numpy()  # [C, H, W]
        n_channels = fmaps.shape[0]

        # Skip if too few channels to cluster
        if n_channels < n_clusters:
            representatives[layer_name] = [(i, 0) for i in range(n_channels)]
            continue

        # Flatten each channel to 1D vector
        fmaps_flat = fmaps.reshape(n_channels, -1)  # [C, H*W]

        # Standardize
        scaler = StandardScaler()
        fmaps_scaled = scaler.fit_transform(fmaps_flat)

        # UMAP dimensionality reduction
        n_neighbors = min(15, n_channels - 1)
        umap = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        embedding = umap.fit_transform(fmaps_scaled)
        umap_embeddings[layer_name] = embedding

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embedding)
        cluster_labels_dict[layer_name] = cluster_labels

        # Find representative for each cluster (closest to centroid)
        reps = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            # Find channel closest to cluster centroid
            cluster_center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(embedding[cluster_indices] - cluster_center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]

            reps.append((int(closest_idx), int(cluster_id)))

        representatives[layer_name] = reps

    return representatives, umap_embeddings, cluster_labels_dict

def plot_umap_clusters(umap_embeddings, cluster_labels_dict, output_dir):
    """Plot UMAP scatter plots showing feature map clusters"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_name, embedding in umap_embeddings.items():
        labels = cluster_labels_dict[layer_name]

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(f'Feature Map Clusters: {layer_name}', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster', fontsize=12)

        # Annotate each point with channel number
        for i, (x, y) in enumerate(embedding):
            ax.annotate(str(i), (x, y), fontsize=8, alpha=0.6, ha='center')

        plt.tight_layout()
        output_path = output_dir / f'umap_clusters_{layer_name}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

    print(f"  ✓ Saved {len(umap_embeddings)} UMAP cluster plots to: {output_dir}")

def plot_representative_feature_maps(activation, layer_name, representatives, output_path):
    """
    Plot only representative feature maps from each cluster.

    Args:
        activation: [1, C, H, W] tensor
        layer_name: Name of layer
        representatives: List of (channel_idx, cluster_id) tuples
        output_path: Path to save plot
    """
    maps = activation.cpu().squeeze(0)  # [C, H, W]

    n_reps = len(representatives)
    cols = min(8, n_reps)
    rows = int(np.ceil(n_reps / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(f"Representative Feature Maps: {layer_name}\n({n_reps} clusters)",
                 fontsize=16, fontweight='bold')

    # Flatten axes
    if rows > 1:
        axes = axes.flatten()
    elif n_reps == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, (channel_idx, cluster_id) in enumerate(representatives):
        ax = axes[i]
        fmap = maps[channel_idx].detach().numpy()

        im = ax.imshow(fmap, cmap='viridis')
        ax.set_title(f"Ch {channel_idx}\nCluster {cluster_id}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off unused subplots
    for j in range(n_reps, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

# ============================================================================
# FEATURE VISUALIZATION WITH UMAP
# ============================================================================

def register_conv_hooks(model):
    """Register hooks for conv2 layers"""
    layers_dict = {}

    def get_conv_layers(block, prefix):
        if hasattr(block, 'bn2'):
            layers_dict[f'{prefix}_conv2'] = block.bn2

    get_conv_layers(model.enc1, 'encoder_1')
    get_conv_layers(model.enc2, 'encoder_2')
    get_conv_layers(model.enc3, 'encoder_3')
    get_conv_layers(model.enc4, 'encoder_4')
    get_conv_layers(model.bottleneck, 'bottleneck')
    get_conv_layers(model.dec4, 'decoder_4')
    get_conv_layers(model.dec3, 'decoder_3')
    get_conv_layers(model.dec2, 'decoder_2')
    get_conv_layers(model.dec1, 'decoder_1')

    return layers_dict

def visualize_feature_maps_with_clustering(model, image_tensor, output_dir, device, n_clusters=8):
    """Visualize representative feature maps using UMAP clustering"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nVisualizing feature maps with UMAP clustering...")

    # Capture activations
    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    layers_to_visualize = register_conv_hooks(model)
    hooks = {}
    for name, layer in layers_to_visualize.items():
        hooks[name] = layer.register_forward_hook(get_activation(name))

    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor.to(device))

    # Remove hooks
    for handle in hooks.values():
        handle.remove()

    # Cluster feature maps
    print("  Clustering feature maps...")
    representatives, umap_embeddings, cluster_labels = cluster_feature_maps(activations, n_clusters=n_clusters)

    # Plot UMAP clusters
    if umap_embeddings:
        umap_dir = output_dir / 'umap_clusters'
        plot_umap_clusters(umap_embeddings, cluster_labels, umap_dir)

    # Plot representative feature maps
    feature_maps_dir = output_dir / 'representative_feature_maps'
    feature_maps_dir.mkdir(parents=True, exist_ok=True)

    for layer_name, activation in tqdm(activations.items(), desc="Plotting representative feature maps"):
        reps = representatives[layer_name]
        output_path = feature_maps_dir / f"feature_map_{layer_name}.png"
        plot_representative_feature_maps(activation, layer_name, reps, output_path)

    print(f"  ✓ Saved {len(activations)} representative feature map visualizations")

# ============================================================================
# FEATURE INVERSION AT CORRECT DIMENSIONS
# ============================================================================

def total_variation_loss(img):
    """Total Variation Loss for regularization"""
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (img.shape[0] * img.shape[1] * img.shape[2] * img.shape[3])

def reconstruct_from_layer_with_correct_dims(model, target_image_tensor, layer_name, layer_module,
                                              spatial_size, device, n_steps=2000, lr=0.01, tv_weight=0.001):
    """
    Feature inversion starting from the CORRECT spatial dimensions.

    Args:
        model: Trained model
        target_image_tensor: Full 512x512 input [1, 1, 512, 512]
        layer_name: Layer name
        layer_module: Layer module
        spatial_size: Target spatial size for this layer
        device: Device
        n_steps: Optimization steps
        lr: Learning rate
        tv_weight: TV loss weight

    Returns:
        optimized_image_fullres: Reconstructed image upsampled to 512x512
    """
    print(f"  Feature inversion: {layer_name} (inverting from {spatial_size}x{spatial_size})")

    # Get target feature map
    target_activations = {}
    def get_target_hook(name):
        def hook(module, input, output):
            target_activations[name] = output.detach()
        return hook

    hook_handle = layer_module.register_forward_hook(get_target_hook(layer_name))
    with torch.no_grad():
        _ = model(target_image_tensor.to(device))
    target_feature_map = target_activations[layer_name].clone()
    hook_handle.remove()

    # Initialize optimized image at CORRECT dimensions
    # Start from noise at the layer's actual spatial resolution
    optimized_image = torch.randn(1, 1, spatial_size, spatial_size, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([optimized_image], lr=lr)

    # Hook for current activations
    current_activations = {}
    def get_current_hook(name):
        def hook(module, input, output):
            current_activations[name] = output
        return hook

    hook_handle = layer_module.register_forward_hook(get_current_hook(layer_name))
    feature_loss_fn = nn.MSELoss()

    # Optimization loop
    for i in range(n_steps + 1):
        optimizer.zero_grad()

        # Upsample to 512x512 for model input
        if spatial_size != 512:
            input_image = F.interpolate(optimized_image, size=(512, 512), mode='bilinear', align_corners=False)
        else:
            input_image = optimized_image

        _ = model(input_image)
        current_feature_map = current_activations[layer_name]

        feature_loss = feature_loss_fn(current_feature_map, target_feature_map)
        tv_loss = total_variation_loss(optimized_image)
        total_loss = feature_loss + (tv_weight * tv_loss)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            optimized_image.data.clamp_(0, 1)

    hook_handle.remove()

    # Upsample final result to 512x512 for display
    if spatial_size != 512:
        optimized_image_fullres = F.interpolate(optimized_image, size=(512, 512), mode='bilinear', align_corners=False)
    else:
        optimized_image_fullres = optimized_image

    return optimized_image_fullres

def visualize_feature_inversions_correct_dims(model, image_tensor, output_dir, device, n_filters=32):
    """Feature inversions at correct spatial resolutions"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nVisualizing feature inversions at correct dimensions...")

    layer_dims = get_layer_dimensions(n_filters)
    all_layers = register_conv_hooks(model)

    original_img_plot = postprocess_tensor(image_tensor)

    for layer_name, layer_module in tqdm(all_layers.items(), desc="Running feature inversions"):
        spatial_size, n_channels = layer_dims[layer_name]

        reconstructed_image = reconstruct_from_layer_with_correct_dims(
            model, image_tensor, layer_name, layer_module, spatial_size, device,
            n_steps=2000, lr=0.01, tv_weight=0.001
        )

        recon_img_plot = postprocess_tensor(reconstructed_image)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Feature Inversion: {layer_name} (from {spatial_size}×{spatial_size})",
                     fontsize=16, fontweight='bold')

        ax1.imshow(original_img_plot, cmap='gray')
        ax1.set_title("Original Image (512×512)", fontsize=14)
        ax1.axis('off')

        ax2.imshow(recon_img_plot, cmap='gray')
        ax2.set_title(f"Reconstructed (inverted from {spatial_size}×{spatial_size})", fontsize=14)
        ax2.axis('off')

        plt.tight_layout()
        output_path = output_dir / f"feature_inversion_{layer_name}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

    print(f"  ✓ Saved {len(all_layers)} feature inversion visualizations")

def create_3panel_figure(tile_original, tile_preprocessed, prediction, image_name, output_path):
    """Create 3-panel figure"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Input and Prediction: {image_name}", fontsize=18, fontweight='bold', y=0.98)

    axes[0].imshow(tile_original, cmap='gray')
    axes[0].set_title("Original Tile (512x512)", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    tile_prep_img = postprocess_tensor(tile_preprocessed)
    axes[1].imshow(tile_prep_img, cmap='gray')
    axes[1].set_title("Preprocessed (Percentile Norm)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    pred_img = postprocess_tensor(prediction)
    axes[2].imshow(pred_img, cmap='gray')
    axes[2].set_title("Model Prediction", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def find_best_model(cache_dir='./best_models_PyTorch'):
    """Find best model in cache"""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Model cache not found: {cache_dir}")

    unet_dir = cache_dir / 'unet'
    if not unet_dir.exists():
        raise FileNotFoundError(f"UNet models not found: {cache_dir}")

    model_files = list(unet_dir.glob('**/best_model.pth'))
    if not model_files:
        raise FileNotFoundError(f"No best_model.pth found in: {unet_dir}")

    model_path = model_files[0]
    parent_dir = model_path.parent.name

    import re
    n_filters_match = re.search(r'n_filters(\d+)', parent_dir)
    dropout_match = re.search(r'dropout([\d.]+)', parent_dir)

    n_filters = int(n_filters_match.group(1)) if n_filters_match else 32
    dropout = float(dropout_match.group(1)) if dropout_match else 0.1

    return model_path, {'architecture': 'unet', 'n_filters': n_filters, 'dropout': dropout, 'model_path': str(model_path)}

def process_single_image(test_image_path, model, metadata, output_base_dir, device, tile_row=3, tile_col=4, n_clusters=8):
    """Process single image with advanced visualizations"""
    image_name = test_image_path.stem
    image_output_dir = output_base_dir / image_name
    image_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Processing: {image_name}")
    print(f"{'='*80}")

    # Load and extract tile
    full_image = Image.open(test_image_path).convert("L")
    full_array = np.array(full_image, dtype=np.float32)
    print(f"  Full image shape: {full_array.shape}")

    tile, (tile_y, tile_x) = extract_center_tile(full_array, tile_size=512, row=tile_row, col=tile_col)
    print(f"  Extracted tile at: row={tile_row}, col={tile_col} (y={tile_y}, x={tile_x})")

    tile_tensor = preprocess_tile(tile)

    # Generate prediction
    print("  Generating prediction...")
    with torch.no_grad():
        prediction = model(tile_tensor.to(device))
        prediction_sigmoid = torch.sigmoid(prediction)

    # 3-panel figure
    print("  Creating 3-panel figure...")
    create_3panel_figure(tile, tile_tensor, prediction_sigmoid, image_name,
                        image_output_dir / f"{image_name}_3panel.png")

    # Feature maps with clustering
    print("  Visualizing feature maps with UMAP clustering...")
    feature_maps_dir = image_output_dir / 'feature_maps'
    visualize_feature_maps_with_clustering(model, tile_tensor, feature_maps_dir, device, n_clusters=n_clusters)

    # Feature inversions at correct dimensions
    print("  Visualizing feature inversions at correct dimensions...")
    feature_inversions_dir = image_output_dir / 'feature_inversions'
    visualize_feature_inversions_correct_dims(model, tile_tensor, feature_inversions_dir, device,
                                             n_filters=metadata['n_filters'])

    # Save metadata
    tile_metadata = {
        'image_name': image_name,
        'full_image_shape': full_array.shape,
        'tile_position': {'y': int(tile_y), 'x': int(tile_x)},
        'tile_row': tile_row,
        'tile_col': tile_col,
        'tile_size': 512,
        'n_clusters': n_clusters
    }

    with open(image_output_dir / 'tile_metadata.json', 'w') as f:
        json.dump(tile_metadata, f, indent=2)

    print(f"  ✓ Completed: {image_name}")

def main():
    parser = argparse.ArgumentParser(description='Advanced U-Net Feature Visualization')
    parser.add_argument('--model_cache', type=str, default='./best_models_PyTorch')
    parser.add_argument('--test_images', type=str, default='./test_images')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--tile_row', type=int, default=3)
    parser.add_argument('--tile_col', type=int, default=4)
    parser.add_argument('--n_clusters', type=int, default=8, help='Number of clusters for UMAP')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not UMAP_AVAILABLE:
        print("\nWARNING: UMAP not available. Clustering will be skipped.")
        print("Install with: pip install umap-learn scikit-learn")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ADVANCED U-NET FEATURE VISUALIZATION")
    print("="*80)
    print(f"Model cache: {args.model_cache}")
    print(f"Test images: {args.test_images}")
    print(f"Tile position: row {args.tile_row}, col {args.tile_col}")
    print(f"UMAP clusters: {args.n_clusters}")
    print(f"Output: {args.output}")
    print("="*80)

    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)

    model_path, metadata = find_best_model(args.model_cache)
    print(f"\nModel: {metadata['architecture']}")
    print(f"  n_filters: {metadata['n_filters']}")
    print(f"  dropout: {metadata['dropout']}")

    model = build_model(metadata['architecture'], metadata['n_filters'], metadata['dropout']).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded")

    # Find test images
    print("\n" + "="*80)
    print("FINDING TEST IMAGES")
    print("="*80)

    test_images_dir = Path(args.test_images)
    test_images = sorted(test_images_dir.glob('*.tif'))

    if not test_images:
        print(f"ERROR: No .tif images in {test_images_dir}")
        return

    print(f"\nFound {len(test_images)} images:")
    for img in test_images:
        print(f"  - {img.name}")

    # Process all images
    print("\n" + "="*80)
    print("PROCESSING ALL TEST IMAGES")
    print("="*80)

    for test_image_path in test_images:
        process_single_image(test_image_path, model, metadata, output_dir, device,
                           tile_row=args.tile_row, tile_col=args.tile_col, n_clusters=args.n_clusters)

    # Save overall metadata
    viz_metadata = {
        'model_metadata': metadata,
        'test_images_dir': str(args.test_images),
        'num_images_processed': len(test_images),
        'tile_position': {'row': args.tile_row, 'col': args.tile_col},
        'n_clusters': args.n_clusters,
        'umap_available': UMAP_AVAILABLE,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(output_dir / 'visualization_metadata.json', 'w') as f:
        json.dump(viz_metadata, f, indent=2)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Results: {output_dir}")
    print(f"\nProcessed {len(test_images)} images with:")
    print("  - 3-panel figures (original + preprocessed + prediction)")
    print(f"  - UMAP clustered feature maps ({args.n_clusters} clusters per layer)")
    print("  - Feature inversions at correct dimensions (512→256→128→64→32)")
    print("="*80)

if __name__ == "__main__":
    main()
