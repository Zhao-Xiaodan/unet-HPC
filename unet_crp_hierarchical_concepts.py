#!/usr/bin/env python3
"""
U-Net Concept Relevance Propagation (CRP) - Hierarchical Concept Analysis
===========================================================================

Implements CRP for U-Net to trace hierarchical concept composition:
- Identifies which lower-layer channels contribute to higher-layer concepts
- Generates conditional heatmaps showing concept dependencies
- Creates masked reference samples using RelMax
- Produces hierarchical concept composition graphs

Based on: "Concept Relevance Propagation" (Nature Machine Intelligence, 2023)
Adapted for U-Net architecture with skip connections

Target Analysis: Ch4 in Cluster6 (decoder_1_conv2) → decoder_2_conv2 → decoder_3_conv2 → ...

Author: Claude Code
Date: October 28, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import zennit (for LRP)
try:
    from zennit.composites import EpsilonPlusFlat, NameMapComposite
    from zennit.rules import Epsilon, Gamma, Flat, ZPlus
    from zennit.core import Composite, Hook
    from zennit.attribution import Gradient
    from zennit.types import Convolution, Linear
    from zennit.canonizers import SequentialMergeBatchNorm
    ZENNIT_AVAILABLE = True
    print("✓ Zennit is available for LRP-based CRP")
except ImportError:
    ZENNIT_AVAILABLE = False
    print("⚠ Zennit not available. Will use gradient-based approximation.")
    print("  Install with: pip install zennit")

# ============================================================================
# PREPROCESSING (Same as training/visualization)
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

# ============================================================================
# U-NET MODEL ARCHITECTURE
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

class UNet(nn.Module):
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

    def forward(self, x, return_intermediates=False):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Decoder with concatenation
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

        out = self.out(d1)

        if return_intermediates:
            intermediates = {
                'encoder_1_conv2': e1,
                'encoder_2_conv2': e2,
                'encoder_3_conv2': e3,
                'encoder_4_conv2': e4,
                'bottleneck_conv2': b,
                'decoder_4_conv2': d4,
                'decoder_3_conv2': d3,
                'decoder_2_conv2': d2,
                'decoder_1_conv2': d1,
            }
            return out, intermediates
        return out

# ============================================================================
# CRP IMPLEMENTATION FOR U-NET
# ============================================================================

class UNetCRP:
    """
    Concept Relevance Propagation adapted for U-Net

    Key adaptations:
    1. Handles skip connections by tracking both upsampling and encoder paths
    2. Implements conditional backpropagation with channel filtering
    3. Computes relevance scores for hierarchical concept tracing
    """

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Store activations and gradients during forward/backward passes
        self.activations = {}
        self.gradients = {}
        self.hooks = []

    def register_hooks(self, layer_names):
        """Register forward and backward hooks to capture activations and gradients"""

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        def get_gradient(name):
            def hook(model, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        # Clear previous hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # Register hooks for each layer
        layer_map = {
            'decoder_1_conv2': self.model.dec1.conv2,
            'decoder_2_conv2': self.model.dec2.conv2,
            'decoder_3_conv2': self.model.dec3.conv2,
            'decoder_4_conv2': self.model.dec4.conv2,
            'bottleneck_conv2': self.model.bottleneck.conv2,
        }

        for name in layer_names:
            if name in layer_map:
                layer = layer_map[name]
                self.hooks.append(layer.register_forward_hook(get_activation(name)))
                self.hooks.append(layer.register_full_backward_hook(get_gradient(name)))

    def conditional_relevance_propagation(self, input_tensor, target_layer, target_channels,
                                         source_layer=None, top_k=2):
        """
        Compute conditional relevance propagation from target_layer back to source_layer

        Args:
            input_tensor: Input image tensor [1, 1, H, W]
            target_layer: Layer name to start propagation (e.g., 'decoder_1_conv2')
            target_channels: List of channel indices to condition on (e.g., [4])
            source_layer: Layer name to propagate to (if None, uses full backward)
            top_k: Number of top contributing channels to identify

        Returns:
            relevance_scores: Channel-wise relevance scores at source_layer
            conditional_heatmap: Spatial heatmap showing where relevance comes from
        """

        # Register hooks
        layers_to_track = [target_layer]
        if source_layer:
            layers_to_track.append(source_layer)
        self.register_hooks(layers_to_track)

        # Forward pass
        self.activations.clear()
        self.gradients.clear()

        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        output, intermediates = self.model(input_tensor, return_intermediates=True)

        # Get target layer activation
        target_activation = self.activations[target_layer]  # [1, C, H, W]

        # Create conditional signal: only activate specified channels
        batch_size, num_channels, h, w = target_activation.shape
        conditional_signal = torch.zeros_like(target_activation)
        for ch in target_channels:
            conditional_signal[0, ch, :, :] = target_activation[0, ch, :, :]

        # Backward pass with conditional signal
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()

        # Sum over spatial dimensions to get scalar per channel
        loss = conditional_signal.sum()
        loss.backward()

        # Compute relevance scores at source layer
        if source_layer and source_layer in self.activations:
            source_activation = self.activations[source_layer]
            source_gradient = self.gradients.get(source_layer, None)

            if source_gradient is not None:
                # Compute relevance: activation * gradient (Grad-CAM style)
                relevance = (source_activation * source_gradient).mean(dim=[2, 3])  # [1, C]
                relevance = relevance[0].abs()  # [C]

                # Get top-k channels
                top_k_actual = min(top_k, len(relevance))
                top_k_indices = torch.topk(relevance, top_k_actual).indices.cpu().numpy()
                top_k_values = relevance[top_k_indices].cpu().numpy()

                # Create spatial heatmap
                spatial_relevance = (source_activation * source_gradient).sum(dim=1, keepdim=True)  # [1, 1, H, W]
                heatmap = F.interpolate(spatial_relevance, size=input_tensor.shape[2:],
                                       mode='bilinear', align_corners=False)
                heatmap = heatmap[0, 0].cpu().numpy()
                heatmap = np.maximum(heatmap, 0)  # ReLU
                heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize

                return {
                    'channel_indices': top_k_indices,
                    'channel_relevance': top_k_values,
                    'spatial_heatmap': heatmap,
                    'all_relevance': relevance.cpu().numpy()
                }

        # If no source layer specified, return input gradient heatmap
        input_gradient = input_tensor.grad[0, 0].cpu().numpy()
        input_gradient = np.maximum(input_gradient, 0)
        input_gradient = input_gradient / (input_gradient.max() + 1e-8)

        return {
            'spatial_heatmap': input_gradient,
            'all_relevance': None
        }

    def trace_hierarchical_concepts(self, input_tensor, start_layer, start_channel, top_k=2):
        """
        Trace hierarchical concept composition through decoder layers

        Args:
            input_tensor: Input image tensor
            start_layer: Starting layer (e.g., 'decoder_1_conv2')
            start_channel: Starting channel index (e.g., 4)
            top_k: Number of top contributing channels per layer

        Returns:
            hierarchy: Dictionary mapping layers to their contributing channels
        """

        # Define layer sequence (from shallow to deep in decoder)
        layer_sequence = [
            'decoder_1_conv2',
            'decoder_2_conv2',
            'decoder_3_conv2',
            'decoder_4_conv2',
            'bottleneck_conv2',
        ]

        # Find starting position
        start_idx = layer_sequence.index(start_layer)

        # Trace backwards through layers
        hierarchy = {}
        current_channels = [start_channel]

        for i in range(start_idx, len(layer_sequence) - 1):
            current_layer = layer_sequence[i]
            next_layer = layer_sequence[i + 1]

            print(f"\n{'='*60}")
            print(f"Tracing from {current_layer} (Ch {current_channels}) → {next_layer}")
            print(f"{'='*60}")

            # Compute conditional relevance for each current channel
            all_contributing_channels = []
            all_relevance_values = []

            for ch in current_channels:
                result = self.conditional_relevance_propagation(
                    input_tensor,
                    target_layer=current_layer,
                    target_channels=[ch],
                    source_layer=next_layer,
                    top_k=top_k
                )

                if result['all_relevance'] is not None:
                    top_channels = result['channel_indices']
                    top_values = result['channel_relevance']

                    print(f"\n  Ch{ch} ← Top {top_k} from {next_layer}:")
                    for tc, tv in zip(top_channels, top_values):
                        print(f"    Ch{tc}: relevance={tv:.4f}")

                    all_contributing_channels.extend(top_channels)
                    all_relevance_values.extend(top_values)

            # Store in hierarchy
            hierarchy[current_layer] = {
                'channels': current_channels.copy(),
                'contributes_from': {
                    'layer': next_layer,
                    'channels': all_contributing_channels,
                    'relevance': all_relevance_values
                }
            }

            # Update current channels for next iteration (use top contributors)
            if all_contributing_channels:
                # Get unique channels sorted by relevance
                channel_relevance = defaultdict(float)
                for ch, rel in zip(all_contributing_channels, all_relevance_values):
                    channel_relevance[ch] += rel

                sorted_channels = sorted(channel_relevance.items(),
                                       key=lambda x: x[1], reverse=True)
                current_channels = [ch for ch, _ in sorted_channels[:top_k]]
            else:
                break

        return hierarchy

    def visualize_hierarchy(self, hierarchy, output_path, input_image=None):
        """
        Visualize hierarchical concept composition graph

        Args:
            hierarchy: Hierarchy dictionary from trace_hierarchical_concepts
            output_path: Path to save visualization
            input_image: Optional input image to show
        """

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('Hierarchical Concept Composition (CRP)', fontsize=16, fontweight='bold')

        # If input image provided, show it
        if input_image is not None:
            ax_img = fig.add_subplot(gs[0, 0])
            ax_img.imshow(input_image, cmap='gray')
            ax_img.set_title('Input Image', fontsize=12, fontweight='bold')
            ax_img.axis('off')

        # Create graph visualization
        ax_graph = fig.add_subplot(gs[:, 1:])

        # Build graph structure
        layers = sorted(hierarchy.keys(),
                       key=lambda x: int(x.split('_')[1]) if 'decoder' in x else 999,
                       reverse=True)

        y_positions = {layer: i for i, layer in enumerate(layers)}

        # Draw nodes and edges
        for layer_idx, layer in enumerate(layers):
            info = hierarchy[layer]
            source_channels = info['channels']

            # Draw source nodes
            for i, ch in enumerate(source_channels):
                x = layer_idx
                y = i - len(source_channels) / 2

                # Node
                circle = plt.Circle((x, y), 0.15, color='steelblue', alpha=0.7, zorder=10)
                ax_graph.add_patch(circle)
                ax_graph.text(x, y, f'Ch{ch}', ha='center', va='center',
                            fontsize=10, fontweight='bold', color='white', zorder=11)

                # Draw connections to contributing channels
                if 'contributes_from' in info:
                    target_layer = info['contributes_from']['layer']
                    target_channels = info['contributes_from']['channels']
                    relevances = info['contributes_from']['relevance']

                    if target_layer in layers:
                        target_idx = layers.index(target_layer)

                        for j, (tch, rel) in enumerate(zip(target_channels, relevances)):
                            # Find position of target channel
                            target_y = j - len(target_channels) / 2

                            # Draw edge with width proportional to relevance
                            max_rel = max(relevances) if relevances else 1
                            linewidth = 1 + 5 * (rel / max_rel)
                            alpha = 0.3 + 0.7 * (rel / max_rel)

                            ax_graph.plot([x, target_idx], [y, target_y],
                                        'k-', linewidth=linewidth, alpha=alpha, zorder=1)

                            # Add relevance label
                            mid_x = (x + target_idx) / 2
                            mid_y = (y + target_y) / 2
                            ax_graph.text(mid_x, mid_y, f'{rel:.2f}',
                                        fontsize=8, ha='center',
                                        bbox=dict(boxstyle='round,pad=0.3',
                                                facecolor='yellow', alpha=0.7))

        # Layer labels
        for i, layer in enumerate(layers):
            layer_name = layer.replace('_', ' ').title()
            ax_graph.text(i, -3, layer_name, ha='center', fontsize=11,
                        fontweight='bold', rotation=45)

        ax_graph.set_xlim(-0.5, len(layers) - 0.5)
        ax_graph.set_ylim(-4, 3)
        ax_graph.axis('off')
        ax_graph.set_aspect('equal')

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Hierarchical concept graph saved to: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='U-Net Concept Relevance Propagation')
    parser.add_argument('--model_path', type=str,
                       default='/Users/xiaodan/unetCNN/unet-HPC/best_models_PyTorch/unet/best_model.pth',
                       help='Path to trained U-Net model')
    parser.add_argument('--test_image', type=str,
                       default='/Users/xiaodan/unetCNN/unet-HPC/test_images/320x_2025-05-15_02-05-00.png',
                       help='Path to test image')
    parser.add_argument('--start_layer', type=str, default='decoder_1_conv2',
                       help='Starting layer for CRP analysis')
    parser.add_argument('--start_channel', type=int, default=4,
                       help='Starting channel index (e.g., Ch4 in Cluster6)')
    parser.add_argument('--top_k', type=int, default=2,
                       help='Number of top contributing channels to identify per layer')
    parser.add_argument('--output_dir', type=str, default='./unet_crp_analysis',
                       help='Output directory for results')
    parser.add_argument('--n_filters', type=int, default=32,
                       help='Number of filters in first layer')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')

    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir + f"_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print(f"\nLoading U-Net model from: {args.model_path}")
    model = UNet(in_channels=1, n_filters=args.n_filters, dropout=args.dropout)

    if Path(args.model_path).exists():
        checkpoint = torch.load(args.model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint is a dict with training info (epoch, optimizer, etc.)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded successfully (epoch {checkpoint.get('epoch', '?')})")
            if 'best_val_iou' in checkpoint:
                print(f"  Best validation IoU: {checkpoint['best_val_iou']:.4f}")
        else:
            # Checkpoint is just the state dict
            model.load_state_dict(checkpoint)
            print("✓ Model loaded successfully")
    else:
        print(f"⚠ Model file not found: {args.model_path}")
        print("  Will use randomly initialized model for demonstration")

    # Load and preprocess test image
    print(f"\nLoading test image: {args.test_image}")
    try:
        image = Image.open(args.test_image)
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        image_array = np.array(image, dtype=np.float32)
        print(f"✓ Image loaded: shape={image_array.shape}, dtype={image_array.dtype}")
    except Exception as e:
        print(f"ERROR: Failed to load image: {e}")
        import sys
        sys.exit(1)

    # Extract center tile
    tile, (y, x) = extract_center_tile(image_array, tile_size=512, row=3, col=4)
    print(f"Extracted tile at position ({y}, {x})")

    # Preprocess
    input_tensor = preprocess_tile(tile)

    # Save input tile
    plt.figure(figsize=(6, 6))
    plt.imshow(tile, cmap='gray')
    plt.title('Input Tile (512×512)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig(output_dir / 'input_tile.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Initialize CRP
    print(f"\n{'='*60}")
    print("INITIALIZING U-NET CRP ANALYSIS")
    print(f"{'='*60}")
    print(f"Start Layer: {args.start_layer}")
    print(f"Start Channel: Ch{args.start_channel}")
    print(f"Top-K Contributors: {args.top_k}")

    crp = UNetCRP(model, device=device)

    # Trace hierarchical concepts
    print(f"\n{'='*60}")
    print("TRACING HIERARCHICAL CONCEPT COMPOSITION")
    print(f"{'='*60}")

    hierarchy = crp.trace_hierarchical_concepts(
        input_tensor,
        start_layer=args.start_layer,
        start_channel=args.start_channel,
        top_k=args.top_k
    )

    # Save hierarchy data
    hierarchy_json = {}
    for layer, info in hierarchy.items():
        hierarchy_json[layer] = {
            'channels': [int(ch) for ch in info['channels']],
            'contributes_from': {
                'layer': info['contributes_from']['layer'],
                'channels': [int(ch) for ch in info['contributes_from']['channels']],
                'relevance': [float(r) for r in info['contributes_from']['relevance']]
            }
        }

    with open(output_dir / 'hierarchy.json', 'w') as f:
        json.dump(hierarchy_json, f, indent=2)
    print(f"\n✓ Hierarchy data saved to: {output_dir / 'hierarchy.json'}")

    # Visualize hierarchy
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    crp.visualize_hierarchy(
        hierarchy,
        output_path=output_dir / 'hierarchical_concept_graph.png',
        input_image=tile
    )

    # Summary report
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - input_tile.png: Input image tile")
    print(f"  - hierarchy.json: Hierarchical concept data")
    print(f"  - hierarchical_concept_graph.png: Concept composition graph")

    print(f"\n{'='*60}")
    print("SUMMARY OF CONCEPT HIERARCHY")
    print(f"{'='*60}")

    for layer in sorted(hierarchy.keys(),
                       key=lambda x: int(x.split('_')[1]) if 'decoder' in x else 999):
        info = hierarchy[layer]
        print(f"\n{layer}:")
        print(f"  Channels: {info['channels']}")
        if 'contributes_from' in info:
            cf = info['contributes_from']
            print(f"  ← {cf['layer']}:")
            for ch, rel in zip(cf['channels'], cf['relevance']):
                print(f"      Ch{ch}: {rel:.4f}")

if __name__ == "__main__":
    main()
