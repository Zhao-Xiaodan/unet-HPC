#!/usr/bin/env python3
"""
Complete U-Net CRP Analysis with Encoders and Skip Connections
===============================================================

Extends comprehensive CRP to include:
1. Encoder path: bottleneck → encoder_4 → encoder_3 → encoder_2 → encoder_1
2. Skip connections: decoder layers ← corresponding encoder layers
3. Feature map extraction for visualization
4. Multi-hop path support

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
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PREPROCESSING
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
# U-NET MODEL
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

    def forward(self, x, return_intermediates=False):
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
# COMPLETE CRP ANALYZER (INCLUDING ENCODERS)
# ============================================================================

class CompleteCRP:
    """
    Computes COMPLETE relevance matrices including encoder path and skip connections
    """

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.gradients = {}
        self.hooks = []

    def register_hooks(self, layer_names):
        """Register backward hooks to capture gradients"""
        def get_gradient(name):
            def hook(model, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        # Clear previous hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        layer_map = {
            'encoder_1_conv2': self.model.enc1.conv2,
            'encoder_2_conv2': self.model.enc2.conv2,
            'encoder_3_conv2': self.model.enc3.conv2,
            'encoder_4_conv2': self.model.enc4.conv2,
            'bottleneck_conv2': self.model.bottleneck.conv2,
            'decoder_4_conv2': self.model.dec4.conv2,
            'decoder_3_conv2': self.model.dec3.conv2,
            'decoder_2_conv2': self.model.dec2.conv2,
            'decoder_1_conv2': self.model.dec1.conv2,
        }

        for name in layer_names:
            if name in layer_map:
                layer = layer_map[name]
                self.hooks.append(layer.register_full_backward_hook(get_gradient(name)))

    def compute_full_relevance_matrix(self, input_tensor, target_layer, source_layer):
        """
        Compute COMPLETE relevance matrix from target_layer to source_layer

        Returns:
            relevance_matrix: [target_channels, source_channels] matrix
        """

        # Register hooks
        self.register_hooks([target_layer, source_layer])

        # Forward pass
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        output, intermediates = self.model(input_tensor, return_intermediates=True)

        target_activation = intermediates[target_layer]
        num_target_channels = target_activation.shape[1]
        num_source_channels = intermediates[source_layer].shape[1]

        # Initialize relevance matrix
        relevance_matrix = np.zeros((num_target_channels, num_source_channels), dtype=np.float32)

        # Iterate through each target channel
        for target_ch in tqdm(range(num_target_channels), desc=f"  {target_layer}", leave=False):
            # Clear gradients
            self.model.zero_grad()
            self.gradients.clear()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()

            # Forward pass
            output, intermediates = self.model(input_tensor, return_intermediates=True)
            target_activation = intermediates[target_layer]

            # Create conditional signal for this channel only
            conditional_signal = torch.zeros_like(target_activation)
            conditional_signal[0, target_ch, :, :] = target_activation[0, target_ch, :, :]

            # Backward pass
            loss = conditional_signal.sum()
            loss.backward()

            # Get source gradient
            if source_layer in self.gradients:
                source_activation = intermediates[source_layer].detach()
                source_gradient = self.gradients[source_layer]

                # Compute relevance
                relevance_per_channel = (source_activation * source_gradient).mean(dim=[2, 3])
                relevance_matrix[target_ch, :] = relevance_per_channel[0].abs().cpu().numpy()

        return relevance_matrix

    def save_feature_maps(self, input_tensor, output_dir):
        """
        Extract and save feature maps for all layers and channels

        Returns:
            feature_maps_metadata: Dict with file paths for each layer/channel
        """

        print(f"\nExtracting feature maps...")

        # Move input to correct device (model is on GPU, input might be on CPU)
        input_tensor = input_tensor.to(self.device)

        # Forward pass to get all intermediates
        with torch.no_grad():
            output, intermediates = self.model(input_tensor, return_intermediates=True)

        feature_maps_dir = output_dir / 'feature_maps'
        feature_maps_dir.mkdir(exist_ok=True)

        feature_maps_metadata = {}

        for layer_name, activation in tqdm(intermediates.items(), desc="  Saving feature maps"):
            layer_dir = feature_maps_dir / layer_name
            layer_dir.mkdir(exist_ok=True)

            activation_cpu = activation[0].cpu().numpy()  # [C, H, W]
            num_channels = activation_cpu.shape[0]

            feature_maps_metadata[layer_name] = {
                'num_channels': num_channels,
                'spatial_shape': activation_cpu.shape[1:],
                'channels': {}
            }

            # Save each channel as PNG
            for ch in range(num_channels):
                ch_map = activation_cpu[ch]  # [H, W]

                # Normalize to 0-1
                ch_map_norm = (ch_map - ch_map.min()) / (ch_map.max() - ch_map.min() + 1e-8)

                # Save as PNG with viridis colormap
                fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
                ax.imshow(ch_map_norm, cmap='viridis')
                ax.axis('off')
                ax.set_title(f'{layer_name} Ch{ch}', fontsize=10)
                plt.tight_layout(pad=0)

                ch_path = layer_dir / f'ch{ch:03d}.png'
                plt.savefig(ch_path, bbox_inches='tight', pad_inches=0.05)
                plt.close()

                # Store relative path
                feature_maps_metadata[layer_name]['channels'][ch] = str(ch_path.relative_to(output_dir))

        print(f"  ✓ Saved feature maps for {len(intermediates)} layers")

        return feature_maps_metadata

    def compute_all_transitions(self, input_tensor, output_dir):
        """
        Compute relevance matrices for ALL transitions including:
        1. Decoder path: decoder_1 → decoder_2 → decoder_3 → decoder_4 → bottleneck
        2. Encoder path: bottleneck → encoder_4 → encoder_3 → encoder_2 → encoder_1
        3. Skip connections: decoder layers ← encoder layers
        """

        transitions = {}

        # Define complete architecture paths
        architecture_connections = [
            # Decoder path (original)
            ('decoder_1_conv2', 'decoder_2_conv2'),
            ('decoder_2_conv2', 'decoder_3_conv2'),
            ('decoder_3_conv2', 'decoder_4_conv2'),
            ('decoder_4_conv2', 'bottleneck_conv2'),

            # Encoder path (NEW!)
            ('bottleneck_conv2', 'encoder_4_conv2'),
            ('encoder_4_conv2', 'encoder_3_conv2'),
            ('encoder_3_conv2', 'encoder_2_conv2'),
            ('encoder_2_conv2', 'encoder_1_conv2'),

            # Skip connections (NEW!)
            ('decoder_4_conv2', 'encoder_4_conv2'),  # Skip from encoder to decoder
            ('decoder_3_conv2', 'encoder_3_conv2'),
            ('decoder_2_conv2', 'encoder_2_conv2'),
            ('decoder_1_conv2', 'encoder_1_conv2'),
        ]

        for target_layer, source_layer in architecture_connections:
            print(f"\n{'='*60}")
            print(f"Computing: {target_layer} ← {source_layer}")
            print(f"{'='*60}")

            relevance_matrix = self.compute_full_relevance_matrix(
                input_tensor, target_layer, source_layer
            )

            transitions[(target_layer, source_layer)] = relevance_matrix

            # Save immediately
            matrix_path = output_dir / f"{target_layer}_from_{source_layer}.npy"
            np.save(matrix_path, relevance_matrix)

            print(f"  Shape: {relevance_matrix.shape}")
            print(f"  Range: [{relevance_matrix.min():.4f}, {relevance_matrix.max():.4f}]")
            print(f"  Mean: {relevance_matrix.mean():.4f}")
            print(f"  ✓ Saved: {matrix_path.name}")

        # Extract and save feature maps
        feature_maps_metadata = self.save_feature_maps(input_tensor, output_dir)

        return transitions, feature_maps_metadata

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Complete CRP Analysis with Encoders')
    parser.add_argument('--model_path', type=str,
                       default='./best_models_PyTorch/unet/best_model.pth')
    parser.add_argument('--test_images_dir', type=str,
                       default='./test_images')
    parser.add_argument('--output_dir', type=str,
                       default='./unet_crp_complete')
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--tile_row', type=int, default=3)
    parser.add_argument('--tile_col', type=int, default=4)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir + f"_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = UNet(in_channels=1, n_filters=args.n_filters, dropout=args.dropout)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded (epoch {checkpoint.get('epoch', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Model loaded")

    # Find test images
    test_images_dir = Path(args.test_images_dir)
    test_images = sorted(test_images_dir.glob('*.tif')) + sorted(test_images_dir.glob('*.tiff'))

    if not test_images:
        print(f"ERROR: No test images found in {test_images_dir}")
        return

    print(f"\nFound {len(test_images)} test images")

    # Initialize CRP
    crp = CompleteCRP(model, device=device)

    # Process each test image
    all_results = {}

    for img_idx, img_path in enumerate(test_images):
        print(f"\n{'='*70}")
        print(f"Processing image {img_idx + 1}/{len(test_images)}: {img_path.name}")
        print(f"{'='*70}")

        # Load image
        try:
            image = Image.open(img_path)
            if image.mode != 'L':
                image = image.convert('L')
            image_array = np.array(image, dtype=np.float32)
        except Exception as e:
            print(f"  ERROR loading image: {e}")
            continue

        # Extract tile
        tile, (y, x) = extract_center_tile(image_array, tile_size=512,
                                          row=args.tile_row, col=args.tile_col)
        input_tensor = preprocess_tile(tile)

        print(f"  Tile extracted at ({y}, {x}), shape: {tile.shape}")

        # Save tile
        tile_save_path = output_dir / f"{img_path.stem}_tile.png"
        plt.figure(figsize=(6, 6))
        plt.imshow(tile, cmap='gray')
        plt.title(f'{img_path.stem} - Tile ({y}, {x})', fontsize=12)
        plt.axis('off')
        plt.savefig(tile_save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Create image-specific directory
        image_output_dir = output_dir / img_path.stem
        image_output_dir.mkdir(exist_ok=True)

        # Compute all transitions (including encoders and skip connections!)
        transitions, feature_maps_metadata = crp.compute_all_transitions(input_tensor, image_output_dir)

        # Store results
        all_results[img_path.stem] = {
            'image_path': str(img_path),
            'tile_position': (int(y), int(x)),
            'tile_shape': tile.shape,
            'transitions': {
                f"{target}_from_{source}": {
                    'shape': rel_matrix.shape,
                    'file': f"{img_path.stem}/{target}_from_{source}.npy"
                }
                for (target, source), rel_matrix in transitions.items()
            },
            'feature_maps': feature_maps_metadata
        }

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_info': {
            'n_filters': args.n_filters,
            'dropout': args.dropout,
            'model_path': args.model_path
        },
        'layer_sequence_full': [
            'decoder_1_conv2',
            'decoder_2_conv2',
            'decoder_3_conv2',
            'decoder_4_conv2',
            'bottleneck_conv2',
            'encoder_4_conv2',
            'encoder_3_conv2',
            'encoder_2_conv2',
            'encoder_1_conv2',
        ],
        'layer_channels': {
            'encoder_1_conv2': args.n_filters,
            'encoder_2_conv2': args.n_filters * 2,
            'encoder_3_conv2': args.n_filters * 4,
            'encoder_4_conv2': args.n_filters * 8,
            'bottleneck_conv2': args.n_filters * 16,
            'decoder_4_conv2': args.n_filters * 8,
            'decoder_3_conv2': args.n_filters * 4,
            'decoder_2_conv2': args.n_filters * 2,
            'decoder_1_conv2': args.n_filters,
        },
        'connections': [
            # Decoder path
            ['decoder_1_conv2', 'decoder_2_conv2'],
            ['decoder_2_conv2', 'decoder_3_conv2'],
            ['decoder_3_conv2', 'decoder_4_conv2'],
            ['decoder_4_conv2', 'bottleneck_conv2'],
            # Encoder path
            ['bottleneck_conv2', 'encoder_4_conv2'],
            ['encoder_4_conv2', 'encoder_3_conv2'],
            ['encoder_3_conv2', 'encoder_2_conv2'],
            ['encoder_2_conv2', 'encoder_1_conv2'],
            # Skip connections
            ['decoder_4_conv2', 'encoder_4_conv2'],
            ['decoder_3_conv2', 'encoder_3_conv2'],
            ['decoder_2_conv2', 'encoder_2_conv2'],
            ['decoder_1_conv2', 'encoder_1_conv2'],
        ],
        'images': all_results
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Processed {len(all_results)} images")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nComputed connections:")
    print(f"  - Decoder path: 4 transitions")
    print(f"  - Encoder path: 4 transitions")
    print(f"  - Skip connections: 4 transitions")
    print(f"  - Total: 12 transition matrices per image")
    print(f"  - Feature maps: All layers and channels saved")

if __name__ == "__main__":
    main()
