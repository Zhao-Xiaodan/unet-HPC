#!/usr/bin/env python3
"""
Advanced Feature Visualization for Edge Detector Experiment

Generates comprehensive visualizations for Gabor-initialized models:
1. Feature inversions at correct spatial resolutions (9 layers)
2. PCA clustering analysis (9 encoder/decoder layers)
3. Representative feature maps from PCA
4. Layer 1 Gabor filter comparison (initial vs final)

Usage:
    python visualize_edge_detector_advanced.py \
        --model_path ./edge_detector_experiment/.../best_model.pth \
        --variant frozen_layer1  # or trainable_layer1
        --filter_image 320x
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import base visualization functions
import visualize_unet_features_advanced as base_viz

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and Dropout"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    """Standard U-Net architecture"""
    def __init__(self, in_channels=1, n_filters=32, dropout=0.2):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, n_filters, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(n_filters, n_filters*2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(n_filters*2, n_filters*4, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(n_filters*4, n_filters*8, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(n_filters*8, n_filters*16, dropout=dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(n_filters*16, n_filters*8, 2, stride=2)
        self.dec4 = ConvBlock(n_filters*16, n_filters*8, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(n_filters*8, n_filters*4, 2, stride=2)
        self.dec3 = ConvBlock(n_filters*8, n_filters*4, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 2, stride=2)
        self.dec2 = ConvBlock(n_filters*4, n_filters*2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(n_filters*2, n_filters, 2, stride=2)
        self.dec1 = ConvBlock(n_filters*2, n_filters, dropout=dropout)

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

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))

# ============================================================================
# LAYER REGISTRATION FOR VISUALIZATION
# ============================================================================

def register_unet_hooks(model):
    """Register hooks on conv2 layers for feature extraction"""
    layers = {
        'encoder_1_conv2': model.enc1.conv2,
        'encoder_2_conv2': model.enc2.conv2,
        'encoder_3_conv2': model.enc3.conv2,
        'encoder_4_conv2': model.enc4.conv2,
        'bottleneck_conv2': model.bottleneck.conv2,
        'decoder_4_conv2': model.dec4.conv2,
        'decoder_3_conv2': model.dec3.conv2,
        'decoder_2_conv2': model.dec2.conv2,
        'decoder_1_conv2': model.dec1.conv2,
    }
    return layers

# ============================================================================
# FEATURE MAP VISUALIZATION WITH PCA
# ============================================================================

def visualize_feature_maps_pca(model, image_tensor, output_dir, device, n_clusters=8):
    """
    Visualize feature maps using PCA clustering
    """
    output_dir = Path(output_dir / 'feature_maps')
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_clusters_dir = output_dir / 'pca_clusters'
    pca_dir = output_dir / 'representative_feature_maps_pca'
    pca_clusters_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)

    print("\nVisualizing feature maps with PCA clustering...")

    # Capture activations
    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    layers_to_visualize = register_unet_hooks(model)
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

    # Extract PCA results (always available)
    pca_results = results['pca']
    method_name = pca_results['method_name']

    # Plot all PCA cluster scatter plots at once
    base_viz.plot_clustering_results(
        pca_results['embeddings'],
        pca_results['cluster_labels'],
        pca_clusters_dir,
        method_name='PCA'
    )

    # Plot representative feature maps for each layer
    for layer_name in pca_results['representatives'].keys():
        representatives = pca_results['representatives'][layer_name]
        representative_path = pca_dir / f"feature_map_{layer_name}_pca.png"
        base_viz.plot_representative_feature_maps(
            activations[layer_name], layer_name, representatives,
            representative_path, method_name='PCA'
        )

    print(f"  ✓ Saved PCA visualizations to: {output_dir}")

# ============================================================================
# FEATURE INVERSIONS
# ============================================================================

def visualize_feature_inversions(model, image_tensor, output_dir, device, n_filters=32):
    """Feature inversions at correct dimensions for U-Net"""
    output_dir = Path(output_dir / 'feature_inversions')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nVisualizing feature inversions...")

    # Use base visualization module
    base_viz.visualize_feature_inversions_correct_dims(
        model, image_tensor, output_dir, device, n_filters=n_filters
    )

# ============================================================================
# LAYER 1 GABOR COMPARISON
# ============================================================================

def visualize_layer1_gabor_comparison(model, output_dir, initial_weights_path=None):
    """
    Compare initial Gabor filters with final learned filters

    For frozen models: Should be identical
    For trainable models: Shows adaptation trajectory
    """
    output_dir = Path(output_dir / 'layer1_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nVisualizing Layer 1 Gabor comparison...")

    # Get current layer 1 weights
    current_weights = model.enc1.conv1.weight.data.cpu()  # [32, 1, 3, 3]

    # Load initial Gabor weights if provided
    if initial_weights_path and Path(initial_weights_path).exists():
        initial_weights = torch.load(initial_weights_path, map_location='cpu')

        # Plot comparison
        fig, axes = plt.subplots(2, 16, figsize=(24, 6))
        fig.suptitle('Layer 1: Initial Gabor Filters vs Final Learned Filters', fontsize=16, fontweight='bold')

        for i in range(min(16, current_weights.shape[0])):
            # Initial Gabor
            initial_filter = initial_weights[i, 0].numpy()
            axes[0, i].imshow(initial_filter, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Initial\n(Gabor)', fontsize=12, fontweight='bold')

            # Final learned
            final_filter = current_weights[i, 0].numpy()
            axes[1, i].imshow(final_filter, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Final\n(Learned)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'layer1_gabor_comparison_first16.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Compute distance metrics
        l2_distance = torch.norm(current_weights - initial_weights).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            current_weights.flatten().unsqueeze(0),
            initial_weights.flatten().unsqueeze(0)
        ).item()

        metrics = {
            'l2_distance': l2_distance,
            'cosine_similarity': cosine_sim,
            'weights_identical': torch.equal(current_weights, initial_weights),
            'mean_abs_change': (current_weights - initial_weights).abs().mean().item(),
            'max_abs_change': (current_weights - initial_weights).abs().max().item(),
        }

        with open(output_dir / 'layer1_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  ✓ Layer 1 comparison saved to: {output_dir}")
        print(f"    L2 distance: {l2_distance:.6f}")
        print(f"    Cosine similarity: {cosine_sim:.6f}")
        print(f"    Weights identical: {metrics['weights_identical']}")

    else:
        # Just visualize current weights
        fig, axes = plt.subplots(2, 16, figsize=(24, 6))
        fig.suptitle('Layer 1: Current Learned Filters', fontsize=16, fontweight='bold')

        for i in range(min(32, current_weights.shape[0])):
            row = i // 16
            col = i % 16
            filter_weight = current_weights[i, 0].numpy()
            axes[row, col].imshow(filter_weight, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'layer1_current_filters.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Layer 1 filters saved to: {output_dir}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Advanced visualization for edge detector experiment')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--variant', type=str, required=True,
                       choices=['frozen_layer1', 'trainable_layer1'],
                       help='Model variant')
    parser.add_argument('--test_images_dir', type=str, default='./test_images',
                       help='Directory with test images')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n_clusters', type=int, default=8,
                       help='Number of PCA clusters')
    parser.add_argument('--filter_image', type=str, default=None,
                       help='Process only images matching this pattern (e.g., "320x")')
    parser.add_argument('--tile_row', type=int, default=3)
    parser.add_argument('--tile_col', type=int, default=4)
    parser.add_argument('--initial_weights', type=str, default=None,
                       help='Path to initial Gabor weights (for comparison)')

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'edge_detector_viz_advanced_{args.variant}_{timestamp}'

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"EDGE DETECTOR EXPERIMENT: ADVANCED VISUALIZATION")
    print("="*80)
    print(f"Variant: {args.variant}")
    print(f"Model: {args.model_path}")
    print(f"Test images: {args.test_images_dir}")
    print(f"Output: {args.output_dir}")
    print(f"PCA clusters: {args.n_clusters}")
    print("="*80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    print(f"\nLoading model...")
    model = UNet(in_channels=1, n_filters=args.n_filters, dropout=args.dropout).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✓ Model loaded successfully")

    # Get test images
    test_images_dir = Path(args.test_images_dir)
    image_files = []
    for ext in ['*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg']:
        image_files.extend(test_images_dir.glob(ext))
    image_files = sorted(image_files)

    # Apply image filter if specified
    if args.filter_image:
        original_count = len(image_files)
        image_files = [f for f in image_files if args.filter_image in f.name]
        print(f"\nFiltered from {original_count} to {len(image_files)} image(s) matching '{args.filter_image}'")

    if len(image_files) == 0:
        filter_msg = f" matching filter '{args.filter_image}'" if args.filter_image else ""
        print(f"❌ No images found{filter_msg} in {test_images_dir}")
        return

    print(f"\nFound {len(image_files)} test images")

    # Process each image
    for img_idx, img_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing image {img_idx}/{len(image_files)}: {img_path.name}")
        print('='*80)

        # Load and preprocess image
        img = Image.open(img_path)
        img_array = np.array(img).astype(np.float32)

        # Convert RGB to grayscale if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = np.mean(img_array, axis=2)  # Average RGB channels

        if img_array.max() > 1:
            img_array /= 255.0

        # Extract center tile (returns tuple: tile, (y, x))
        tile, (tile_y, tile_x) = base_viz.extract_center_tile(img_array, tile_size=512, row=args.tile_row, col=args.tile_col)
        tile_tensor = base_viz.preprocess_tile(tile)

        # Run prediction
        with torch.no_grad():
            prediction = model(tile_tensor.to(device))

        # Create output directory for this image
        image_output_dir = output_base / img_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create 3-panel figure
        pred_np = prediction.cpu().squeeze().numpy()
        base_viz.create_3panel_figure(
            tile, tile, pred_np, img_path.stem,
            image_output_dir / f"{img_path.stem}_3panel.png"
        )
        print("✓ Saved 3-panel figure")

        # 2. Feature inversions
        visualize_feature_inversions(model, tile_tensor, image_output_dir, device, args.n_filters)

        # 3. Feature maps with PCA clustering
        visualize_feature_maps_pca(model, tile_tensor, image_output_dir, device, args.n_clusters)

        # 4. Layer 1 Gabor comparison (only once)
        if img_idx == 1:
            visualize_layer1_gabor_comparison(model, image_output_dir, args.initial_weights)

    print(f"\n{'='*80}")
    print("✓ All visualizations complete!")
    print(f"  Output directory: {output_base}")
    print('='*80)

if __name__ == '__main__':
    main()
