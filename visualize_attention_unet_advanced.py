#!/usr/bin/env python3
"""
Attention U-Net Advanced Feature Visualization
===============================================

Enhanced visualization for Attention U-Net with:
1. Feature inversions at correct spatial resolutions
2. PCA/UMAP clustering for representative feature maps
3. Attention gate visualizations
4. Full feature map exports

Based on visualize_unet_features_advanced.py but adapted for Attention U-Net architecture.

Author: Claude Code
Date: October 30, 2025
"""

import sys
import os

# Add path for shared utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the base visualization script as a module
import visualize_unet_features_advanced as base_viz

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import PCA and clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Try to import UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ============================================================================
# ATTENTION U-NET ARCHITECTURE
# ============================================================================

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

class AttentionGate(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, stride=1, padding=0)
        self.W_x = nn.Conv2d(F_l, F_int, 2, stride=2, padding=0)
        self.psi = nn.Conv2d(F_int, 1, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        """g: gating signal from decoder, x: skip connection from encoder"""
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

class AttentionUNet(nn.Module):
    """Attention U-Net with attention gates on skip connections"""
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

        # Decoder with attention gates
        self.up4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 2, stride=2)
        self.att4 = AttentionGate(F_g=n_filters * 8, F_l=n_filters * 8, F_int=n_filters * 4)
        self.dec4 = ConvBlock(n_filters * 16, n_filters * 8, dropout)

        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 2, stride=2)
        self.att3 = AttentionGate(F_g=n_filters * 4, F_l=n_filters * 4, F_int=n_filters * 2)
        self.dec3 = ConvBlock(n_filters * 8, n_filters * 4, dropout)

        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, stride=2)
        self.att2 = AttentionGate(F_g=n_filters * 2, F_l=n_filters * 2, F_int=n_filters)
        self.dec2 = ConvBlock(n_filters * 4, n_filters * 2, dropout)

        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, 2, stride=2)
        self.att1 = AttentionGate(F_g=n_filters, F_l=n_filters, F_int=n_filters // 2)
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
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

# ============================================================================
# ATTENTION U-NET SPECIFIC FUNCTIONS
# ============================================================================

def register_attention_unet_hooks(model):
    """Register hooks for Attention U-Net layers including attention gates"""
    layers_dict = {}

    # Encoder blocks
    def get_conv_layers(block, prefix):
        if hasattr(block, 'bn2'):
            layers_dict[f'{prefix}_conv2'] = block.bn2

    get_conv_layers(model.enc1, 'encoder_1')
    get_conv_layers(model.enc2, 'encoder_2')
    get_conv_layers(model.enc3, 'encoder_3')
    get_conv_layers(model.enc4, 'encoder_4')
    get_conv_layers(model.bottleneck, 'bottleneck')

    # Decoder blocks
    get_conv_layers(model.dec4, 'decoder_4')
    get_conv_layers(model.dec3, 'decoder_3')
    get_conv_layers(model.dec2, 'decoder_2')
    get_conv_layers(model.dec1, 'decoder_1')

    # Attention gates - register the sigmoid output
    layers_dict['attention_gate_4'] = model.att4.sigmoid
    layers_dict['attention_gate_3'] = model.att3.sigmoid
    layers_dict['attention_gate_2'] = model.att2.sigmoid
    layers_dict['attention_gate_1'] = model.att1.sigmoid

    return layers_dict

def get_attention_unet_layer_dimensions(n_filters=32):
    """Get spatial dimensions and channel counts for each layer"""
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
        # Attention gates - same spatial dims as corresponding encoder
        'attention_gate_4': (64, 1),  # Attention map is single channel
        'attention_gate_3': (128, 1),
        'attention_gate_2': (256, 1),
        'attention_gate_1': (512, 1),
    }

# ============================================================================
# MAIN PROCESSING FUNCTIONS (Adapted from base)
# ============================================================================

def visualize_attention_unet_feature_maps(model, image_tensor, output_dir, device, n_clusters=8):
    """
    Visualize feature maps with PCA/UMAP clustering for Attention U-Net
    """
    output_dir = Path(output_dir)

    # Create subdirectories
    feature_maps_dir = output_dir / 'feature_maps'
    pca_dir = feature_maps_dir / 'representative_feature_maps_pca'
    pca_clusters_dir = feature_maps_dir / 'pca_clusters'

    pca_dir.mkdir(parents=True, exist_ok=True)
    pca_clusters_dir.mkdir(parents=True, exist_ok=True)

    print("\nVisualizing Attention U-Net feature maps with PCA clustering...")

    # Capture activations
    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    layers_to_visualize = register_attention_unet_hooks(model)
    hooks = {}
    for name, layer in layers_to_visualize.items():
        hooks[name] = layer.register_forward_hook(get_activation(name))

    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor.to(device))

    # Remove hooks
    for hook in hooks.values():
        hook.remove()

    # Cluster and visualize each layer using PCA
    results = base_viz.cluster_feature_maps_dual(activations, n_clusters=n_clusters)

    for result in results:
        layer_name = result['layer_name']
        method_name = result['method_name']
        embeddings = result['embeddings']
        labels = result['labels']
        representatives = result['representatives']

        # Plot cluster scatter
        cluster_path = pca_clusters_dir / f"pca_clusters_{layer_name}.png"
        base_viz.plot_cluster_scatter(embeddings, labels, layer_name, cluster_path, method_name)

        # Plot representative feature maps
        representative_path = pca_dir / f"feature_map_{layer_name}_pca.png"
        base_viz.plot_representative_feature_maps(
            activations[layer_name], layer_name, representatives,
            representative_path, method_name='PCA'
        )

    print(f"  ✓ Saved PCA visualizations to: {output_dir / 'feature_maps'}")

def visualize_attention_unet_inversions(model, image_tensor, output_dir, device, n_filters=32):
    """Feature inversions at correct dimensions for Attention U-Net"""
    output_dir = Path(output_dir / 'feature_inversions')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nVisualizing Attention U-Net feature inversions...")

    layer_dims = get_attention_unet_layer_dimensions(n_filters)
    all_layers = register_attention_unet_hooks(model)

    original_img_plot = base_viz.postprocess_tensor(image_tensor)

    # Only invert conv layers (skip attention gates for inversion)
    conv_layers = {k: v for k, v in all_layers.items() if 'attention_gate' not in k}

    for layer_name, layer_module in tqdm(conv_layers.items(), desc="Running feature inversions"):
        spatial_size, n_channels = layer_dims[layer_name]

        reconstructed_image = base_viz.reconstruct_from_layer_with_correct_dims(
            model, image_tensor, layer_name, layer_module, spatial_size, device,
            n_steps=2000, lr=0.01, tv_weight=0.001
        )

        recon_img_plot = base_viz.postprocess_tensor(reconstructed_image)

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

    print(f"  ✓ Saved {len(conv_layers)} feature inversion visualizations")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Attention U-Net Advanced Feature Visualization')
    parser.add_argument('--model_path', type=str, default='./best_models_PyTorch/attention_unet/best_model.pth',
                       help='Path to Attention U-Net model checkpoint')
    parser.add_argument('--test_images_dir', type=str, default='./test_images',
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: attention_unet_visualization_advanced_<timestamp>)')
    parser.add_argument('--n_filters', type=int, default=32,
                       help='Number of base filters')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--n_clusters', type=int, default=8,
                       help='Number of clusters for feature map grouping')
    parser.add_argument('--tile_row', type=int, default=3,
                       help='Tile row position')
    parser.add_argument('--tile_col', type=int, default=4,
                       help='Tile column position')

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'attention_unet_visualization_advanced_{timestamp}'

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ATTENTION U-NET ADVANCED FEATURE VISUALIZATION")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Test images: {args.test_images_dir}")
    print(f"Output: {args.output_dir}")
    print(f"PCA clusters: {args.n_clusters}")
    print("="*80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    print("\nLoading Attention U-Net model...")
    model = AttentionUNet(in_channels=1, n_filters=args.n_filters, dropout=args.dropout).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✓ Model loaded successfully")

    # Get test images (support multiple formats)
    test_images_dir = Path(args.test_images_dir)
    image_files = []
    for ext in ['*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg']:
        image_files.extend(test_images_dir.glob(ext))
    image_files = sorted(image_files)

    if len(image_files) == 0:
        print(f"❌ No images found in {test_images_dir}")
        print(f"   Searched for: .png, .tif, .tiff, .jpg, .jpeg")
        return

    print(f"\nFound {len(image_files)} test images")

    # Process each image
    for img_idx, img_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing image {img_idx}/{len(image_files)}: {img_path.name}")
        print('='*80)

        # Load and preprocess
        image = np.array(Image.open(img_path))
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)

        # Extract tile
        tile, (y, x) = base_viz.extract_center_tile(image, tile_size=512, row=args.tile_row, col=args.tile_col)
        tile_tensor = base_viz.preprocess_tile(tile).to(device)

        # Create output directory for this image
        image_name = img_path.stem
        image_output_dir = output_base / image_name
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate prediction
        with torch.no_grad():
            prediction = torch.sigmoid(model(tile_tensor))

        # Save 3-panel figure
        panel_path = image_output_dir / f"{image_name}_3panel.png"
        base_viz.create_3panel_figure(tile, tile_tensor, prediction, image_name, panel_path)
        print(f"✓ Saved 3-panel figure")

        # Feature inversions
        visualize_attention_unet_inversions(model, tile_tensor, image_output_dir, device, args.n_filters)

        # Feature maps with clustering
        visualize_attention_unet_feature_maps(model, tile_tensor, image_output_dir, device, args.n_clusters)

        # Save metadata
        metadata = {
            'image_name': image_name,
            'tile_position': {'row': args.tile_row, 'col': args.tile_col, 'y': int(y), 'x': int(x)},
            'model_config': {'n_filters': args.n_filters, 'dropout': args.dropout},
            'pca_clusters': args.n_clusters,
        }

        with open(image_output_dir / 'tile_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Completed {image_name}")

    # Save global metadata
    global_metadata = {
        'model_path': str(args.model_path),
        'test_images_dir': str(args.test_images_dir),
        'n_images_processed': len(image_files),
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_base / 'visualization_metadata.json', 'w') as f:
        json.dump(global_metadata, f, indent=2)

    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_base}")
    print(f"Processed {len(image_files)} images")
    print("="*80)

if __name__ == "__main__":
    main()
