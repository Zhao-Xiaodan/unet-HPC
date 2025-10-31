#!/usr/bin/env python3
"""
Visualize Layer 1 Conv1 Kernels (Similar to AlexNet Figure 3)

Replicates the visualization style from Krizhevsky et al. (2012) Figure 3
showing learned convolutional kernels from the first layer.

Usage:
    python visualize_layer1_kernels.py --model_path <path> --output <path> --title <title>
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
from pathlib import Path

# ============================================================================
# MODEL ARCHITECTURE (must match training)
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
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_conv1_kernels(weights, output_path, title="Layer 1: Convolutional Kernels"):
    """
    Visualize Layer 1 conv1 kernels in AlexNet Figure 3 style

    Args:
        weights: Tensor of shape [n_filters, in_channels, kernel_h, kernel_w]
                 For our U-Net: [32, 1, 3, 3]
        output_path: Path to save visualization
        title: Title for the figure
    """
    weights = weights.cpu().numpy()
    n_filters, in_channels, kernel_h, kernel_w = weights.shape

    print(f"Kernel shape: {weights.shape}")
    print(f"  n_filters: {n_filters}")
    print(f"  in_channels: {in_channels}")
    print(f"  kernel_size: {kernel_h}×{kernel_w}")

    # Calculate grid layout (aim for roughly square)
    n_cols = 8  # Show 8 filters per row (like AlexNet uses rows of 8)
    n_rows = int(np.ceil(n_filters / n_cols))

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2*n_rows))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Find global min/max for consistent color scaling
    vmin = weights.min()
    vmax = weights.max()

    # Use symmetric normalization around zero for better contrast
    # This ensures positive and negative weights are equally visible
    abs_max = max(abs(vmin), abs(vmax))
    vmin_sym = -abs_max
    vmax_sym = abs_max

    print(f"Weight range: [{vmin:.4f}, {vmax:.4f}]")
    print(f"Symmetric range for visualization: [{vmin_sym:.4f}, {vmax_sym:.4f}]")

    # Plot each filter
    for i in range(n_filters):
        ax = axes_flat[i]

        # Get kernel weights for this filter
        # Shape: [in_channels, kernel_h, kernel_w]
        kernel = weights[i]

        if in_channels == 1:
            # Grayscale input: show single channel as 2D image
            img = kernel[0]  # Shape: [3, 3]
        else:
            # RGB input: show as RGB image (for comparison with AlexNet)
            img = kernel.transpose(1, 2, 0)  # Shape: [3, 3, 3]

        # Display kernel with grayscale colormap (white=positive, black=negative, gray=zero)
        # Using 'gray' colormap for AlexNet-style visualization
        im = ax.imshow(img, cmap='gray', vmin=vmin_sym, vmax=vmax_sym, interpolation='nearest')
        ax.set_title(f'Ch {i}', fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for i in range(n_filters, len(axes_flat)):
        axes_flat[i].axis('off')

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Filter Weight\n(White=Positive, Black=Negative)', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization to: {output_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize Layer 1 conv1 kernels')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for visualization')
    parser.add_argument('--title', type=str, default='Layer 1: Convolutional Kernels',
                       help='Title for the figure')
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)

    args = parser.parse_args()

    print("="*80)
    print("LAYER 1 KERNEL VISUALIZATION")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output}")
    print(f"Title: {args.title}")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model = UNet(in_channels=1, n_filters=args.n_filters, dropout=args.dropout)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("✓ Model loaded successfully")

    # Extract Layer 1 conv1 weights
    print("\nExtracting Layer 1 conv1 weights...")
    conv1_weights = model.enc1.conv1.weight.data  # Shape: [32, 1, 3, 3]

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Visualize kernels
    print("\nVisualizing kernels...")
    visualize_conv1_kernels(conv1_weights, output_path, title=args.title)

    print("\n" + "="*80)
    print("✓ Visualization complete!")
    print("="*80)

if __name__ == '__main__':
    main()
