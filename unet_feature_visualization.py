#!/usr/bin/env python3
"""
U-Net Feature Visualization
============================

Implementation of feature visualization techniques from:
"Feature Visualization" - Olah et al., Distill 2017
https://distill.pub/2017/feature-visualization/

Applies optimization-based visualization to U-Net architecture to understand:
- What patterns activate specific channels
- What features different layers learn
- How the network processes microscopy images

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
import cv2
from scipy.ndimage import gaussian_filter

# ============================================================================
# U-NET MODEL DEFINITION
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
# FEATURE VISUALIZATION ENGINE
# ============================================================================

class FeatureVisualizer:
    """
    Optimization-based feature visualization for U-Net

    Implements techniques from Olah et al. 2017:
    - Gradient ascent optimization
    - Multiple regularization techniques
    - Diverse visualizations per channel
    """

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def visualize_channel(
        self,
        layer_name,
        channel_idx,
        size=(512, 512),
        iterations=500,
        lr=0.05,
        regularization_config=None
    ):
        """
        Generate synthetic image that maximally activates a specific channel

        Args:
            layer_name: Name of layer to visualize (e.g., 'decoder_1_conv2')
            channel_idx: Index of channel to maximize
            size: Size of generated image (H, W)
            iterations: Number of optimization iterations
            lr: Learning rate for gradient ascent
            regularization_config: Dict of regularization parameters

        Returns:
            optimized_image: [H, W] numpy array
            activation_history: List of activation values over iterations
        """

        # Default regularization config
        if regularization_config is None:
            regularization_config = {
                'l2_weight': 1e-4,        # L2 penalty on pixel values
                'tv_weight': 1e-2,        # Total variation (smoothness)
                'blur_every': 4,          # Apply Gaussian blur frequency
                'blur_sigma': 0.5,        # Blur strength
                'jitter_range': 8,        # Random jitter (pixels)
                'rotate_range': 5,        # Random rotation (degrees)
            }

        # Initialize with random noise
        img = torch.randn(1, 1, size[0], size[1], device=self.device) * 0.1
        img.requires_grad_(True)

        # Optimizer
        optimizer = torch.optim.Adam([img], lr=lr)

        activation_history = []

        print(f"\nVisualizing {layer_name} - Channel {channel_idx}")
        print(f"Image size: {size}, Iterations: {iterations}")

        for iteration in tqdm(range(iterations), desc="Optimizing"):
            optimizer.zero_grad()

            # Apply random jitter (spatial translation)
            jitter = regularization_config['jitter_range']
            if jitter > 0:
                ox, oy = np.random.randint(-jitter, jitter+1, 2)
                img_jittered = torch.roll(img, shifts=(ox, oy), dims=(2, 3))
            else:
                img_jittered = img

            # Forward pass
            output, intermediates = self.model(img_jittered, return_intermediates=True)

            # Get activation for target channel
            if layer_name not in intermediates:
                raise ValueError(f"Layer {layer_name} not found. Available: {list(intermediates.keys())}")

            activation = intermediates[layer_name]  # [1, C, H, W]

            # Objective: Maximize mean activation of target channel
            channel_activation = activation[0, channel_idx].mean()
            loss = -channel_activation  # Negative for gradient ascent

            # Add L2 regularization (prevent extreme values)
            l2_loss = regularization_config['l2_weight'] * (img ** 2).mean()
            loss = loss + l2_loss

            # Add total variation loss (encourage smoothness)
            tv_loss = regularization_config['tv_weight'] * self.total_variation_loss(img)
            loss = loss + tv_loss

            # Backward pass
            loss.backward()

            # Gradient ascent step
            optimizer.step()

            # Undo jitter
            if jitter > 0:
                img.data = torch.roll(img.data, shifts=(-ox, -oy), dims=(2, 3))

            # Apply Gaussian blur periodically
            if (iteration + 1) % regularization_config['blur_every'] == 0:
                img_np = img.detach().cpu().numpy()[0, 0]
                img_blurred = gaussian_filter(img_np, sigma=regularization_config['blur_sigma'])
                img.data = torch.from_numpy(img_blurred[None, None]).to(self.device).float()

            # Record activation
            activation_history.append(-loss.item())

        # Final image
        optimized_image = img.detach().cpu().numpy()[0, 0]

        # Normalize for visualization
        optimized_image = self.normalize_image(optimized_image)

        return optimized_image, activation_history

    def visualize_layer_diverse(
        self,
        layer_name,
        num_channels=None,
        num_diverse=3,
        **kwargs
    ):
        """
        Generate diverse visualizations for multiple channels in a layer

        Args:
            layer_name: Name of layer
            num_channels: Number of channels to visualize (None = all)
            num_diverse: Number of diverse examples per channel
            **kwargs: Arguments for visualize_channel

        Returns:
            results: Dict mapping channel_idx -> list of visualizations
        """

        # Get number of channels in layer
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 512, 512, device=self.device)
            _, intermediates = self.model(dummy_input, return_intermediates=True)
            total_channels = intermediates[layer_name].shape[1]

        if num_channels is None:
            channels_to_viz = list(range(total_channels))
        else:
            channels_to_viz = list(range(min(num_channels, total_channels)))

        print(f"\nGenerating diverse visualizations for {layer_name}")
        print(f"Channels: {len(channels_to_viz)}, Diverse examples: {num_diverse}")

        results = {}

        for ch_idx in channels_to_viz:
            diverse_images = []

            for div_idx in range(num_diverse):
                # Use different random seed for diversity
                torch.manual_seed(ch_idx * 1000 + div_idx)
                np.random.seed(ch_idx * 1000 + div_idx)

                img, history = self.visualize_channel(
                    layer_name, ch_idx, **kwargs
                )

                diverse_images.append({
                    'image': img,
                    'activation_history': history,
                    'final_activation': history[-1] if history else 0
                })

            results[ch_idx] = diverse_images

        return results

    def deepdream(
        self,
        input_image,
        layer_name,
        channel_idx=None,
        iterations=100,
        lr=0.01,
        octave_scale=1.4,
        num_octaves=3
    ):
        """
        DeepDream: Amplify patterns detected by the network

        Args:
            input_image: Input image [H, W] numpy array
            layer_name: Layer to amplify
            channel_idx: Specific channel (None = all channels)
            iterations: Optimization iterations
            lr: Learning rate
            octave_scale: Scale factor between octaves
            num_octaves: Number of scales to process

        Returns:
            dreamed_image: Enhanced image
        """

        # Convert to tensor
        img = torch.from_numpy(input_image[None, None]).float().to(self.device)

        # Multi-octave processing (coarse to fine)
        octaves = []
        for i in range(num_octaves - 1):
            h, w = img.shape[2:]
            new_h, new_w = int(h / octave_scale), int(w / octave_scale)
            octaves.append(img)
            img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        octaves.append(img)

        detail = torch.zeros_like(octaves[-1])

        for octave_idx, octave_base in enumerate(reversed(octaves)):
            # Upsample detail from previous octave
            if octave_idx > 0:
                h, w = octave_base.shape[2:]
                detail = F.interpolate(detail, size=(h, w), mode='bilinear', align_corners=False)

            img = octave_base + detail
            img.requires_grad_(True)

            optimizer = torch.optim.Adam([img], lr=lr)

            for i in range(iterations):
                optimizer.zero_grad()

                output, intermediates = self.model(img, return_intermediates=True)
                activation = intermediates[layer_name]

                if channel_idx is not None:
                    loss = -activation[0, channel_idx].mean()
                else:
                    loss = -activation.mean()

                loss.backward()
                optimizer.step()

            detail = img.detach() - octave_base

        dreamed_image = img.detach().cpu().numpy()[0, 0]
        dreamed_image = self.normalize_image(dreamed_image)

        return dreamed_image

    @staticmethod
    def total_variation_loss(img):
        """Compute total variation loss (encourages spatial smoothness)"""
        diff_h = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
        diff_w = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
        return diff_h + diff_w

    @staticmethod
    def normalize_image(img, percentile=2):
        """Normalize image to [0, 1] using percentile clipping"""
        lo, hi = np.percentile(img, [percentile, 100 - percentile])
        img_norm = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
        return img_norm

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def create_visualization_grid(results, layer_name, output_dir, num_per_row=3):
    """Create grid visualization of channel features"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    channels = sorted(results.keys())
    num_diverse = len(results[channels[0]])

    # Create grid: rows = channels, cols = diverse examples
    fig, axes = plt.subplots(
        len(channels), num_diverse,
        figsize=(num_diverse * 3, len(channels) * 3)
    )

    if len(channels) == 1:
        axes = axes[None, :]

    for row_idx, ch_idx in enumerate(channels):
        for col_idx in range(num_diverse):
            ax = axes[row_idx, col_idx]

            img_data = results[ch_idx][col_idx]
            img = img_data['image']
            activation = img_data['final_activation']

            ax.imshow(img, cmap='gray')
            ax.set_title(f'Ch{ch_idx} #{col_idx+1}\nAct={activation:.3f}', fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    save_path = output_dir / f'{layer_name}_diverse_visualizations.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved grid: {save_path.name}")

def plot_activation_history(activation_history, output_path):
    """Plot activation value over optimization iterations"""
    plt.figure(figsize=(8, 4))
    plt.plot(activation_history)
    plt.xlabel('Iteration')
    plt.ylabel('Channel Activation')
    plt.title('Optimization Progress')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Feature Visualization for U-Net'
    )
    parser.add_argument('--model_path', type=str,
                       default='./best_models_PyTorch/unet/best_model.pth')
    parser.add_argument('--output_dir', type=str,
                       default='./unet_feature_viz')
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Visualization parameters
    parser.add_argument('--layers', type=str, nargs='+',
                       default=['encoder_1_conv2', 'encoder_3_conv2',
                               'bottleneck_conv2', 'decoder_1_conv2'],
                       help='Layers to visualize')
    parser.add_argument('--channels_per_layer', type=int, default=8,
                       help='Number of channels to visualize per layer')
    parser.add_argument('--diverse_per_channel', type=int, default=3,
                       help='Number of diverse examples per channel')
    parser.add_argument('--iterations', type=int, default=500,
                       help='Optimization iterations')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Size of generated images')

    # DeepDream parameters
    parser.add_argument('--deepdream', action='store_true',
                       help='Also run DeepDream on test images')
    parser.add_argument('--test_images_dir', type=str,
                       default='./test_images',
                       help='Test images for DeepDream')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"U-Net Feature Visualization")
    print(f"{'='*70}")
    print(f"Device: {device}")

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

    # Initialize visualizer
    visualizer = FeatureVisualizer(model, device=device)

    # Visualize each layer
    all_results = {}

    for layer_name in args.layers:
        print(f"\n{'='*70}")
        print(f"Layer: {layer_name}")
        print(f"{'='*70}")

        results = visualizer.visualize_layer_diverse(
            layer_name=layer_name,
            num_channels=args.channels_per_layer,
            num_diverse=args.diverse_per_channel,
            size=(args.image_size, args.image_size),
            iterations=args.iterations,
            lr=0.05
        )

        all_results[layer_name] = results

        # Create grid visualization
        create_visualization_grid(
            results, layer_name, output_dir,
            num_per_row=args.diverse_per_channel
        )

        # Save individual channel visualizations
        layer_dir = output_dir / layer_name
        layer_dir.mkdir(exist_ok=True)

        for ch_idx, diverse_list in results.items():
            for div_idx, img_data in enumerate(diverse_list):
                # Save image
                img_path = layer_dir / f'ch{ch_idx:03d}_div{div_idx}.png'
                plt.imsave(img_path, img_data['image'], cmap='gray')

                # Save activation history
                hist_path = layer_dir / f'ch{ch_idx:03d}_div{div_idx}_history.png'
                plot_activation_history(img_data['activation_history'], hist_path)

    # DeepDream on test images (optional)
    if args.deepdream:
        print(f"\n{'='*70}")
        print("Running DeepDream on test images")
        print(f"{'='*70}")

        test_images_dir = Path(args.test_images_dir)
        test_images = list(test_images_dir.glob('*.tif')) + list(test_images_dir.glob('*.tiff'))

        if test_images:
            dream_dir = output_dir / 'deepdream'
            dream_dir.mkdir(exist_ok=True)

            for img_path in test_images[:3]:  # Process first 3 images
                print(f"\nProcessing: {img_path.name}")

                # Load image
                image = Image.open(img_path)
                if image.mode != 'L':
                    image = image.convert('L')
                image_array = np.array(image, dtype=np.float32)

                # Extract center tile
                h, w = image_array.shape
                y, x = (h - 512) // 2, (w - 512) // 2
                tile = image_array[y:y+512, x:x+512]

                # Normalize
                tile = (tile - tile.min()) / (tile.max() - tile.min() + 1e-8)

                # Apply DeepDream for each layer
                for layer_name in args.layers:
                    print(f"  Layer: {layer_name}")

                    dreamed = visualizer.deepdream(
                        tile, layer_name,
                        iterations=100,
                        num_octaves=3
                    )

                    # Save comparison
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(tile, cmap='gray')
                    axes[0].set_title('Original')
                    axes[0].axis('off')
                    axes[1].imshow(dreamed, cmap='gray')
                    axes[1].set_title(f'DeepDream - {layer_name}')
                    axes[1].axis('off')
                    plt.tight_layout()

                    save_path = dream_dir / f'{img_path.stem}_{layer_name}_deepdream.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
        else:
            print("  No test images found")

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_info': {
            'n_filters': args.n_filters,
            'dropout': args.dropout,
            'model_path': args.model_path
        },
        'visualization_params': {
            'layers': args.layers,
            'channels_per_layer': args.channels_per_layer,
            'diverse_per_channel': args.diverse_per_channel,
            'iterations': args.iterations,
            'image_size': args.image_size
        },
        'results': {
            layer: {
                'num_channels': len(results),
                'channels': list(results.keys())
            }
            for layer, results in all_results.items()
        }
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Visualized {len(args.layers)} layers")
    print(f"Generated {args.channels_per_layer * args.diverse_per_channel * len(args.layers)} total visualizations")
    print(f"\nResults:")
    print(f"  - Grid visualizations: {len(args.layers)} files")
    print(f"  - Individual channel images: {output_dir / 'layer_name'}/")
    if args.deepdream:
        print(f"  - DeepDream results: {output_dir / 'deepdream'}/")
    print(f"  - Metadata: {metadata_path}")

if __name__ == "__main__":
    main()
