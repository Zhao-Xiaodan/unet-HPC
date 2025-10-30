#!/usr/bin/env python3
"""
Gabor Filter Initialization for U-Net Layer 1
==============================================

This module creates Gabor filters for initializing the first convolutional
layer of U-Net with edge detectors.

Gabor filters are commonly found in the first layer of CNNs trained on natural
images (e.g., ImageNet models) and are effective edge/orientation detectors.

Author: Claude Code
Date: October 30, 2025
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple, List


def create_gabor_kernel(size: int, theta: float, freq: float,
                       phi: float = 0, sigma: float = 1.0) -> np.ndarray:
    """
    Create a single Gabor filter kernel

    Gabor filter formula:
    g(x, y) = exp(-(x'^2 + γ²y'^2) / (2σ²)) * cos(2πf*x' + φ)

    where:
    x' = x*cos(θ) + y*sin(θ)  [rotation]
    y' = -x*sin(θ) + y*cos(θ)  [rotation]

    Args:
        size: Kernel size (e.g., 3 for 3x3)
        theta: Orientation in radians (0 to π)
        freq: Spatial frequency (cycles per pixel)
        phi: Phase offset (0 = edge detector, π/2 = line detector)
        sigma: Gaussian envelope standard deviation

    Returns:
        kernel: [size, size] Gabor filter normalized to zero-mean
    """
    # Create coordinate grid
    grid_size = size // 2
    x, y = np.meshgrid(np.arange(-grid_size, grid_size + 1),
                      np.arange(-grid_size, grid_size + 1))

    # Rotate coordinates by theta
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gaussian envelope
    gamma = 1.0  # Aspect ratio
    gaussian = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))

    # Sinusoidal carrier
    sinusoid = np.cos(2 * np.pi * freq * x_theta + phi)

    # Gabor filter
    gabor = gaussian * sinusoid

    # Normalize to zero-mean (DC-free)
    gabor = gabor - gabor.mean()

    # Normalize to unit energy
    gabor = gabor / (np.sqrt((gabor**2).sum()) + 1e-8)

    return gabor.astype(np.float32)


def create_gabor_filter_bank(n_filters: int = 32, kernel_size: int = 3,
                             visualize: bool = False,
                             save_path: str = None) -> torch.Tensor:
    """
    Create a bank of Gabor filters for CNN initialization

    Strategy:
    - Sample multiple orientations (0° to 180°)
    - Sample multiple frequencies (low, medium, high)
    - Sample multiple phases (edge vs line detectors)
    - Ensure diversity for n_filters coverage

    Args:
        n_filters: Number of filters to generate (e.g., 32 for U-Net enc1)
        kernel_size: Spatial size of filter (must be odd)
        visualize: If True, create visualization of all filters
        save_path: Optional path to save visualization

    Returns:
        filters: Tensor of shape [n_filters, 1, kernel_size, kernel_size]
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd (e.g., 3, 5, 7)")

    filters = []

    # Design space for diversity
    # For n_filters=32, we want good coverage of orientation/frequency space
    n_orientations = 8  # Every 22.5 degrees (180°/8)
    n_frequencies = 3   # Low, medium, high frequency
    n_phases = 2        # Edge (0) and line (π/2) detectors
    # Total combinations: 8 * 3 * 2 = 48 (will sample n_filters from these)

    orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)
    frequencies = [0.15, 0.25, 0.35]  # Tuned for 3x3 kernels
    phases = [0, np.pi/2]
    sigma = 1.5  # Gaussian width (tuned for kernel_size=3)

    # Generate all combinations
    filter_params = []
    for theta in orientations:
        for freq in frequencies:
            for phi in phases:
                filter_params.append((theta, freq, phi))

    # Sample n_filters from combinations (evenly spaced)
    if len(filter_params) > n_filters:
        indices = np.linspace(0, len(filter_params)-1, n_filters, dtype=int)
        filter_params = [filter_params[i] for i in indices]
    elif len(filter_params) < n_filters:
        # Duplicate some filters if we need more
        while len(filter_params) < n_filters:
            filter_params.append(filter_params[len(filter_params) % len(orientations)])

    # Create filters
    for theta, freq, phi in filter_params[:n_filters]:
        gabor = create_gabor_kernel(kernel_size, theta, freq, phi, sigma)
        filters.append(gabor)

    # Convert to torch tensor [n_filters, 1, H, W]
    filters_array = np.array(filters)
    filters_tensor = torch.from_numpy(filters_array).unsqueeze(1)

    # Visualization
    if visualize:
        fig = visualize_filter_bank(filters_array, n_filters, kernel_size)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Gabor filter bank visualization saved to: {save_path}")
        plt.close(fig)

    return filters_tensor


def visualize_filter_bank(filters: np.ndarray, n_filters: int,
                          kernel_size: int) -> plt.Figure:
    """
    Create a grid visualization of all Gabor filters

    Args:
        filters: Array of shape [n_filters, kernel_size, kernel_size]
        n_filters: Number of filters
        kernel_size: Size of each filter

    Returns:
        fig: Matplotlib figure
    """
    # Determine grid size
    ncols = 8
    nrows = int(np.ceil(n_filters / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 1.5))
    fig.suptitle(f'Gabor Filter Bank: {n_filters} filters ({kernel_size}×{kernel_size})',
                 fontsize=14, fontweight='bold')

    axes = axes.flatten() if n_filters > 1 else [axes]

    for i in range(n_filters):
        ax = axes[i]
        im = ax.imshow(filters[i], cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set_title(f'Ch {i}', fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')

    # Colorbar
    fig.colorbar(im, ax=axes, orientation='horizontal',
                 pad=0.02, fraction=0.02, label='Filter Weight')

    plt.tight_layout()
    return fig


def initialize_layer_with_gabor(layer: nn.Conv2d,
                                 preserve_bias: bool = True,
                                 verbose: bool = True) -> None:
    """
    Initialize a Conv2d layer's weights with Gabor filters

    Args:
        layer: PyTorch Conv2d layer to initialize
        preserve_bias: If True, keep existing bias (usually zeros)
        verbose: Print confirmation message
    """
    # Get layer properties
    out_channels = layer.out_channels
    in_channels = layer.in_channels
    kernel_size = layer.kernel_size[0]  # Assume square kernel

    if in_channels != 1:
        raise ValueError(f"Gabor initialization only supports grayscale input "
                        f"(in_channels=1), got {in_channels}")

    if kernel_size != layer.kernel_size[1]:
        raise ValueError(f"Gabor initialization only supports square kernels, "
                        f"got {layer.kernel_size}")

    # Generate Gabor filters
    gabor_filters = create_gabor_filter_bank(
        n_filters=out_channels,
        kernel_size=kernel_size,
        visualize=False
    )

    # Replace layer weights
    with torch.no_grad():
        layer.weight.data = gabor_filters.to(layer.weight.device)

    if verbose:
        print(f"✓ Initialized Conv2d layer with Gabor filters:")
        print(f"  - Filters: {out_channels}")
        print(f"  - Kernel size: {kernel_size}×{kernel_size}")
        print(f"  - Input channels: {in_channels} (grayscale)")


def analyze_gabor_properties(filters: torch.Tensor) -> dict:
    """
    Analyze properties of Gabor filter bank

    Args:
        filters: Tensor of shape [n_filters, 1, H, W]

    Returns:
        properties: Dictionary with statistics
    """
    filters_np = filters.squeeze(1).numpy()

    # Basic statistics
    stats = {
        'n_filters': filters.shape[0],
        'kernel_size': filters.shape[2],
        'mean': filters_np.mean(),
        'std': filters_np.std(),
        'min': filters_np.min(),
        'max': filters_np.max(),
    }

    # Energy per filter
    energies = (filters_np**2).sum(axis=(1, 2))
    stats['mean_energy'] = energies.mean()
    stats['std_energy'] = energies.std()

    # Orientation diversity (approximate via FFT)
    orientations = []
    for f in filters_np:
        # Compute 2D FFT
        fft = np.fft.fft2(f)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Find dominant orientation (simple center-of-mass method)
        cy, cx = np.array(magnitude.shape) // 2
        y, x = np.meshgrid(np.arange(magnitude.shape[0]) - cy,
                          np.arange(magnitude.shape[1]) - cx,
                          indexing='ij')

        # Weighted angle
        theta = np.arctan2(y, x)
        dominant_orientation = np.average(theta, weights=magnitude)
        orientations.append(dominant_orientation)

    stats['orientations'] = orientations
    stats['orientation_diversity'] = np.std(orientations)

    return stats


# ============================================================================
# CLI for testing and visualization
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate and visualize Gabor filter banks for U-Net initialization'
    )
    parser.add_argument('--n_filters', type=int, default=32,
                       help='Number of filters (default: 32 for U-Net enc1)')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='Kernel size (default: 3)')
    parser.add_argument('--output', type=str, default='gabor_filters.png',
                       help='Output visualization path')
    parser.add_argument('--save_weights', type=str, default=None,
                       help='Save filter weights as .pth file')

    args = parser.parse_args()

    print("="*70)
    print("Gabor Filter Bank Generator")
    print("="*70)
    print(f"Generating {args.n_filters} Gabor filters ({args.kernel_size}×{args.kernel_size})")
    print()

    # Generate filters
    filters = create_gabor_filter_bank(
        n_filters=args.n_filters,
        kernel_size=args.kernel_size,
        visualize=True,
        save_path=args.output
    )

    # Analyze properties
    props = analyze_gabor_properties(filters)
    print()
    print("Filter Bank Properties:")
    print(f"  - Number of filters: {props['n_filters']}")
    print(f"  - Kernel size: {props['kernel_size']}×{props['kernel_size']}")
    print(f"  - Mean: {props['mean']:.6f}")
    print(f"  - Std: {props['std']:.6f}")
    print(f"  - Range: [{props['min']:.3f}, {props['max']:.3f}]")
    print(f"  - Mean energy: {props['mean_energy']:.3f}")
    print(f"  - Orientation diversity: {props['orientation_diversity']:.3f} rad")
    print()

    # Save weights if requested
    if args.save_weights:
        torch.save({
            'gabor_filters': filters,
            'n_filters': args.n_filters,
            'kernel_size': args.kernel_size,
            'properties': props,
        }, args.save_weights)
        print(f"✓ Gabor filter weights saved to: {args.save_weights}")
        print()

    print("="*70)
    print("✓ Gabor filter bank generation complete!")
    print("="*70)
    print()
    print("Usage example:")
    print("  import torch.nn as nn")
    print("  from gabor_initializer import initialize_layer_with_gabor")
    print()
    print("  conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)")
    print("  initialize_layer_with_gabor(conv1)")
    print("  # conv1 now has Gabor filter weights!")
    print()
