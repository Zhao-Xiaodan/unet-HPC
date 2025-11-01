#!/usr/bin/env python3
"""
3-Architecture Density Analysis with Dual Contrast and Edge Fix
================================================================

Script: density_analysis_3arch_dual_contrast.py
PBS Script: pbs_density_analysis_3arch_dual_contrast.sh

FIXES & FEATURES:
1. ✅ Analyzes 3 UNet architectures (UNet, Attention UNet, Attention ResUNet)
2. ✅ 4-panel visualization (Original + 3 predictions) like PyTorch analysis
3. ✅ FIXES CLAHE edge artifacts with border padding/cropping
4. ✅ DUAL tile sets (fixed contrast + auto contrast) for comparison
5. ✅ Shows actual min/max ranges in auto-contrast titles

Architecture comparison:
- unet_hyperparam_20251015_224125/
- attention_unet_hyperparam_20251015_230149/
- attention_resunet_hyperparam_20251015_235542/

Author: Claude Code
Date: November 1, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# Import custom modules
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef, focal_loss
from models_fixed import RepeatElements  # Custom layer for attention models

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Model directories for 3 architectures
    'model_dirs': {
        'unet': './unet_hyperparam_20251015_224125',
        'attention_unet': './attention_unet_hyperparam_20251015_230149',
        'attention_resunet': './attention_resunet_hyperparam_20251015_235542',
    },

    'test_images_dir': './test_images',
    'output_dir': f'./density_analysis_3arch_dual_contrast_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Image settings
    'img_size': 512,
    'img_channels': 3,

    # Prediction settings
    'batch_size': 8,
    'thresholds': [0.5],  # Using 0.5 as primary threshold

    # Representative tiles
    'n_representative_tiles': 5,

    # Plotting
    'dpi': 300,
    'figsize_boxplot_full': (14, 8),
    'figsize_boxplot_low': (14, 8),
    'figsize_tiles': (20, 25),  # For 4-panel tiles
}

# Dilution factors
DILUTION_ORDER = [10240, 5120, 2560, 1280, 640, 320, 160, 80, 20, 10]
DILUTION_LABELS = ['10240x', '5120x', '2560x', '1280x', '640x', '320x', '160x', '80x', '20x', '10x']
DILUTION_ORDER_LOW = [10240, 5120, 2560, 1280, 640, 320, 160, 80]
DILUTION_LABELS_LOW = ['10240x', '5120x', '2560x', '1280x', '640x', '320x', '160x', '80x']

DILUTION_PATTERNS = {
    '10240x': 10240, '5120x': 5120, '2560x': 2560, '1280x': 1280,
    '640x': 640, '320x': 320, '160x': 160, '80x': 80, '20x': 20, '10x': 10,
}

ARCHITECTURE_NAMES = {
    'unet': 'UNet',
    'attention_unet': 'Attention UNet',
    'attention_resunet': 'Attention ResUNet'
}

def create_extreme_highlight_colormap():
    """
    Create a custom colormap that highlights min/max pixels with distinct colors.

    Color scheme:
    - MIN pixels (darkest, 0%) → RED
    - Middle pixels (1-99%) → Grayscale gradient
    - MAX pixels (brightest, 100%) → CYAN

    This makes it easy to see where the extreme pixels are located,
    even if there are only a few of them.
    """
    # Define colors at specific positions
    # Position 0.0 (min): RED
    # Position 0.001-0.999 (middle): Gray gradient
    # Position 1.0 (max): CYAN

    colors = [
        (0.000, (1.0, 0.0, 0.0)),  # MIN → RED
        (0.001, (0.0, 0.0, 0.0)),  # Just above min → BLACK (start of grayscale)
        (0.999, (1.0, 1.0, 1.0)),  # Just below max → WHITE (end of grayscale)
        (1.000, (0.0, 1.0, 1.0)),  # MAX → CYAN
    ]

    cmap = LinearSegmentedColormap.from_list('extreme_highlight', colors, N=256)
    return cmap


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_dilution_from_filename(filename):
    """Extract dilution factor from filename."""
    filename_lower = filename.lower()
    for pattern, dilution in DILUTION_PATTERNS.items():
        if pattern in filename_lower:
            return dilution
    return 1

def load_test_image(image_path):
    """Load test image and convert to RGB."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_tiles(image, tile_size):
    """Extract non-overlapping tiles from image."""
    h, w = image.shape[:2]
    tiles_with_pos = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles_with_pos.append((tile, (y, x)))
    return tiles_with_pos

def preprocess_tile(tile):
    """Preprocess tile for model input."""
    if len(tile.shape) == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
    tile = tile.astype(np.float32) / 255.0
    return tile

def calculate_foreground_density(prediction, threshold=0.5):
    """Calculate foreground density from prediction mask."""
    binary_mask = (prediction > threshold).astype(np.float32)
    density = np.mean(binary_mask)
    return density

def rescale_image_full_range(img):
    """Rescale image to full 0-255 range."""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)

def apply_clahe_otsu_fixed(img_gray, clipLimit=2.0, tileGridSize=(8, 8), border_size=8):
    """
    Apply CLAHE + OTSU with edge artifact fix.
    
    FIX: Pads borders before CLAHE to avoid edge effects, then crops after.
    
    Args:
        img_gray: Grayscale image (0-255 uint8)
        clipLimit: CLAHE clip limit
        tileGridSize: CLAHE tile grid size
        border_size: Pixels to pad/crop (default: 8 = same as tileGridSize)
    
    Returns:
        binary_mask: Binary mask after CLAHE+Otsu WITHOUT edge artifacts
    """
    # STEP 1: Pad borders to avoid CLAHE edge effects
    img_padded = cv2.copyMakeBorder(
        img_gray, 
        border_size, border_size, border_size, border_size,
        cv2.BORDER_REPLICATE  # Replicate edge pixels
    )
    
    # STEP 2: Apply CLAHE on padded image
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_img = clahe.apply(img_padded)
    
    # STEP 3: Apply Otsu thresholding
    _, binary_mask_padded = cv2.threshold(
        clahe_img, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # STEP 4: Crop borders to remove edge effects
    binary_mask = binary_mask_padded[
        border_size:-border_size,
        border_size:-border_size
    ]
    
    return binary_mask

def calculate_density_with_clahe_otsu(prediction, clipLimit=2.0, tileGridSize=(8, 8)):
    """Calculate density from predicted mask after applying CLAHE+Otsu (with edge fix)."""
    pred_rescaled = rescale_image_full_range(prediction.squeeze())
    binary_mask = apply_clahe_otsu_fixed(pred_rescaled, clipLimit=clipLimit, tileGridSize=tileGridSize)
    density = 1.0 - (np.count_nonzero(binary_mask) / binary_mask.size)
    return density, binary_mask


# ============================================================================
# MODEL SELECTION & LOADING
# ============================================================================

def find_best_model(base_dir, arch_name):
    """Find the best model for given architecture."""
    print(f"\nSearching for best {arch_name} model...")
    base_dir = Path(base_dir)
    
    checkpoint_dir = base_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        # Try without checkpoints subdirectory
        checkpoint_dir = base_dir
    
    # Find model files
    model_files = list(checkpoint_dir.glob(f'{arch_name}_*/best_model.keras'))
    if not model_files:
        model_files = list(checkpoint_dir.glob('*/best_model.keras'))
    
    if not model_files:
        raise FileNotFoundError(f"No models found in {checkpoint_dir}")
    
    print(f"Found {len(model_files)} {arch_name} model(s)")
    
    best_model_info = None
    best_iou = -1.0
    
    for model_file in model_files:
        dir_name = model_file.parent.name
        
        # Find history CSV
        history_files = list((base_dir / 'logs').glob(f'{dir_name}*_history.csv'))
        if not history_files:
            history_files = list(base_dir.glob(f'{dir_name}*_history.csv'))
        if not history_files:
            history_files = list(base_dir.glob('*_history.csv'))
        
        if history_files:
            try:
                history_df = pd.read_csv(history_files[0])
                if 'val_jacard_coef' in history_df.columns:
                    max_iou = history_df['val_jacard_coef'].max()
                    if max_iou > best_iou:
                        best_iou = max_iou
                        best_model_info = {
                            'model_path': model_file,
                            'model_name': dir_name,
                            'best_val_iou': max_iou,
                            'architecture': arch_name
                        }
                        print(f"  Best so far: {dir_name} (IoU: {max_iou:.4f})")
            except Exception as e:
                continue
    
    if best_model_info is None:
        # Fallback: just use first model found
        print(f"  Warning: No IoU data found, using first model")
        best_model_info = {
            'model_path': model_files[0],
            'model_name': model_files[0].parent.name,
            'best_val_iou': 0.0,
            'architecture': arch_name
        }
    
    print(f"✓ Selected: {best_model_info['model_name']} (IoU: {best_model_info['best_val_iou']:.4f})")
    return best_model_info

def load_model(model_path):
    """Load trained model."""
    print(f"Loading model from: {model_path}")
    
    @keras.saving.register_keras_serializable(package='Custom')
    class BinaryFocalLoss(keras.losses.Loss):
        def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.alpha = alpha
        def call(self, y_true, y_pred):
            return focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)
        def get_config(self):
            config = super().get_config()
            config.update({'gamma': self.gamma, 'alpha': self.alpha})
            return config
        @classmethod
        def from_config(cls, config):
            return cls(**config)
    
    custom_objects = {
        'combined_dice_focal_loss': combined_dice_focal_loss,
        'jacard_coef': jacard_coef,
        'dice_coef': dice_coef,
        'focal_loss': focal_loss,
        'BinaryFocalLoss': BinaryFocalLoss,
        'RepeatElements': RepeatElements,  # Required for attention models
    }
    
    model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
    print(f"  ✓ Loaded successfully")
    return model


# ============================================================================
# 4-PANEL TILE VISUALIZATION (Original + 3 Architectures)
# ============================================================================

def create_4panel_tiles_fixed_contrast(tile_data_by_arch, output_dir, config):
    """
    Create multi-panel visualizations with FIXED contrast (vmin=0, vmax=1).
    Dynamically adjusts to number of loaded architectures.
    Panels: Original | [Architecture 1] | [Architecture 2] | [Architecture 3]
    """
    # Determine which architectures loaded successfully
    loaded_archs = list(tile_data_by_arch.keys())
    n_archs = len(loaded_archs)
    n_panels = n_archs + 1  # +1 for original image

    print(f"\nGenerating {n_panels}-panel tiles (FIXED CONTRAST: vmin=0, vmax=1)...")
    print(f"  Loaded architectures: {', '.join(loaded_archs)}")

    if n_archs == 0:
        print("  WARNING: No architectures loaded! Skipping visualization.")
        return

    output_dir = Path(output_dir)
    tiles_dir = output_dir / f'representative_tiles_{n_panels}panel_fixed_contrast'
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Use first architecture as reference (all archs should have same images/tiles)
    ref_data = tile_data_by_arch[loaded_archs[0]]

    # Group by image
    tiles_by_image = defaultdict(list)
    for tile_info in ref_data:
        tiles_by_image[tile_info['image']].append(tile_info)

    for image_name, tiles in tiles_by_image.items():
        print(f"  Creating tiles for: {image_name}")

        # Select 5 representative tiles
        tiles_sorted = sorted(tiles, key=lambda x: x['density_clahe_otsu_pred'])
        n_tiles = len(tiles_sorted)
        indices = [0, n_tiles//4, n_tiles//2, 3*n_tiles//4, n_tiles-1]
        representative_tiles = [tiles_sorted[i] for i in indices]

        dilution_label = tiles[0]['dilution_label']

        # Create dynamic multi-panel figure (5 rows × n_panels columns)
        fig, axes = plt.subplots(5, n_panels, figsize=(5*n_panels, 25))

        # Handle single architecture case (axes would be 1D)
        if n_archs == 1:
            axes = axes.reshape(5, n_panels)

        for row_idx, tile_info in enumerate(representative_tiles):
            tile_idx = tile_info['tile_idx']
            pos = tile_info['position']

            # Panel 0: Original
            axes[row_idx, 0].imshow(tile_info['tile'], cmap='gray')
            axes[row_idx, 0].set_title(f'Original Tile {tile_idx}\nPos: ({pos[0]}, {pos[1]})', fontsize=10)
            axes[row_idx, 0].axis('off')

            # Panels 1+: Each architecture
            for col_idx, arch_name in enumerate(loaded_archs, start=1):
                # FIX: Find matching tile by image name AND tile_idx (not by list index!)
                matching_tile = None
                for arch_tile in tile_data_by_arch[arch_name]:
                    if arch_tile['image'] == image_name and arch_tile['tile_idx'] == tile_idx:
                        matching_tile = arch_tile
                        break

                if matching_tile is None:
                    print(f"    WARNING: No matching tile found for {arch_name} (image={image_name}, tile_idx={tile_idx})")
                    continue

                pred = matching_tile['prediction']
                density = matching_tile['density_threshold_0.5']

                # Invert and display with fixed contrast
                pred_inv = 1.0 - pred.squeeze()
                axes[row_idx, col_idx].imshow(pred_inv, cmap='gray', vmin=0, vmax=1)
                axes[row_idx, col_idx].set_title(f'{ARCHITECTURE_NAMES.get(arch_name, arch_name)}\nDensity: {density:.4f}', fontsize=10)
                axes[row_idx, col_idx].axis('off')

        fig.suptitle(f'{image_name} - {n_panels}-Panel Comparison (FIXED CONTRAST)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.995])

        output_filename = f'tiles_{n_panels}panel_fixed_{dilution_label}_{image_name.replace(".tif", "").replace(".tiff", "")}.png'
        output_path = tiles_dir / output_filename
        plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved: {output_path.name}")

    print(f"\n  ✓ All {n_panels}-panel fixed-contrast tiles saved to: {tiles_dir}")

def create_4panel_tiles_auto_contrast(tile_data_by_arch, output_dir, config):
    """
    Create multi-panel visualizations with AUTO contrast (vmin/vmax from data).
    Dynamically adjusts to number of loaded architectures.
    Panels: Original | [UNet] | [Attention UNet] | [Attention ResUNet]
    """
    print("\nGenerating multi-panel tiles (AUTO CONTRAST: vmin/vmax from data)...")

    # Determine which architectures loaded successfully
    loaded_archs = list(tile_data_by_arch.keys())
    n_archs = len(loaded_archs)
    n_panels = n_archs + 1  # +1 for original image

    if n_archs == 0:
        print("  ⚠ WARNING: No architectures loaded successfully. Skipping auto-contrast tiles.")
        return

    print(f"  Loaded architectures ({n_archs}): {loaded_archs}")
    print(f"  Creating {n_panels}-panel visualizations (Original + {n_archs} predictions)")

    output_dir = Path(output_dir)
    tiles_dir = output_dir / f'representative_tiles_{n_panels}panel_auto_contrast'
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Use first available architecture for tile selection
    first_arch = loaded_archs[0]
    arch_data = tile_data_by_arch[first_arch]
    tiles_by_image = defaultdict(list)
    for tile_info in arch_data:
        tiles_by_image[tile_info['image']].append(tile_info)

    for image_name, tiles in tiles_by_image.items():
        print(f"  Creating tiles for: {image_name}")

        tiles_sorted = sorted(tiles, key=lambda x: x['density_clahe_otsu_pred'])
        n_tiles = len(tiles_sorted)
        indices = [0, n_tiles//4, n_tiles//2, 3*n_tiles//4, n_tiles-1]
        representative_tiles = [tiles_sorted[i] for i in indices]

        dilution_label = tiles[0]['dilution_label']
        fig, axes = plt.subplots(5, n_panels, figsize=(5*n_panels, 25))

        for row_idx, tile_info in enumerate(representative_tiles):
            tile_idx = tile_info['tile_idx']
            pos = tile_info['position']

            # Panel 1: Original
            axes[row_idx, 0].imshow(tile_info['tile'], cmap='gray')
            axes[row_idx, 0].set_title(f'Original Tile {tile_idx}\nPos: ({pos[0]}, {pos[1]})', fontsize=10)
            axes[row_idx, 0].axis('off')

            # Panels 2+: Predictions for each loaded architecture (AUTO contrast)
            for col_idx, arch_name in enumerate(loaded_archs, start=1):
                # FIX: Find matching tile by image name AND tile_idx (not by list index!)
                matching_tile = None
                for arch_tile in tile_data_by_arch[arch_name]:
                    if arch_tile['image'] == image_name and arch_tile['tile_idx'] == tile_idx:
                        matching_tile = arch_tile
                        break

                if matching_tile is None:
                    print(f"    WARNING: No matching tile found for {arch_name} (image={image_name}, tile_idx={tile_idx})")
                    continue

                pred = matching_tile['prediction']
                density = matching_tile['density_threshold_0.5']

                pred_inv = 1.0 - pred.squeeze()
                v_min, v_max = pred_inv.min(), pred_inv.max()

                # Use custom colormap that highlights min/max pixels
                # RED = minimum pixels, CYAN = maximum pixels, Grayscale = middle values
                extreme_cmap = create_extreme_highlight_colormap()
                axes[row_idx, col_idx].imshow(pred_inv, cmap=extreme_cmap, vmin=v_min, vmax=v_max)
                axes[row_idx, col_idx].set_title(
                    f'{ARCHITECTURE_NAMES.get(arch_name, arch_name)} (AUTO)\n'
                    f'Density: {density:.4f}\n'
                    f'Range: [{v_min:.3f}, {v_max:.3f}]\n'
                    f'🔴MIN  🔵MAX',
                    fontsize=9
                )
                axes[row_idx, col_idx].axis('off')

        fig.suptitle(f'{image_name} - {n_panels}-Panel Comparison (AUTO CONTRAST)',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.995])

        output_filename = f'tiles_{n_panels}panel_auto_{dilution_label}_{image_name.replace(".tif", "").replace(".tiff", "")}.png'
        output_path = tiles_dir / output_filename
        plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved: {output_path.name}")

    print(f"\n  ✓ All {n_panels}-panel auto-contrast tiles saved to: {tiles_dir}")


# ============================================================================
# BOXPLOT VISUALIZATION FOR 3 ARCHITECTURES
# ============================================================================

def create_3arch_boxplot_full_range(tile_data_by_arch, output_dir, config):
    """
    Create grouped boxplot comparing 3 architectures across full dilution range.
    Shows UNet, Attention UNet, and Attention ResUNet side-by-side for each dilution.
    """
    print("\nGenerating 3-architecture boxplot (Full Range: 1/10240 to 1/10)...")

    loaded_archs = list(tile_data_by_arch.keys())
    n_archs = len(loaded_archs)

    if n_archs == 0:
        print("  WARNING: No architectures loaded. Skipping boxplot.")
        return

    output_dir = Path(output_dir)

    # Prepare data for each architecture
    arch_data = {}
    for arch_name in loaded_archs:
        data_by_dilution = {d: [] for d in DILUTION_ORDER}
        for tile_info in tile_data_by_arch[arch_name]:
            dilution = tile_info['dilution']
            if dilution in DILUTION_ORDER:
                data_by_dilution[dilution].append(tile_info['density_threshold_0.5'])
        arch_data[arch_name] = data_by_dilution

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Colors for each architecture
    arch_colors = {
        'unet': '#3498db',  # Blue
        'attention_unet': '#e74c3c',  # Red
        'attention_resunet': '#2ecc71'  # Green
    }

    # Create grouped boxplot
    positions_base = np.arange(len(DILUTION_ORDER))
    width = 0.25

    for arch_idx, arch_name in enumerate(loaded_archs):
        # Calculate positions for this architecture
        positions = positions_base + (arch_idx - n_archs/2 + 0.5) * width

        # Prepare data
        data = [arch_data[arch_name][d] for d in DILUTION_ORDER]

        # Create boxplot
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.8,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(facecolor=arch_colors.get(arch_name, '#95a5a6'),
                         color='black', linewidth=1, alpha=0.7),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1),
            capprops=dict(color='black', linewidth=1),
            flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5),
            label=ARCHITECTURE_NAMES.get(arch_name, arch_name)
        )

    # Customize
    ax.set_title('3-Architecture Comparison: Particle Density vs Dilution (Threshold 0.5)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('1/Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Percentage (log scale)', fontsize=14, fontweight='bold')

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)

    # Set x-axis ticks and labels
    inv_dilution_labels = [f'1/{d}' for d in DILUTION_ORDER]
    ax.set_xticks(positions_base)
    ax.set_xticklabels(inv_dilution_labels, fontsize=12, rotation=45, ha='right')

    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    plt.tight_layout()

    # Save
    filename = 'density_boxplot_3arch_full_range__threshold_0.5.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


def create_3arch_boxplot_low_dilution_range(tile_data_by_arch, output_dir, config):
    """
    Create grouped boxplot for LOW dilution range (1/10240 to 1/80).
    Shows UNet, Attention UNet, and Attention ResUNet side-by-side for each dilution.
    """
    print("\nGenerating 3-architecture boxplot (Low Range: 1/10240 to 1/80)...")

    loaded_archs = list(tile_data_by_arch.keys())
    n_archs = len(loaded_archs)

    if n_archs == 0:
        print("  WARNING: No architectures loaded. Skipping boxplot.")
        return

    output_dir = Path(output_dir)

    # Low dilution range: 10240x to 80x
    LOW_DILUTION_ORDER = [10240, 5120, 2560, 1280, 640, 320, 160, 80]

    # Prepare data for each architecture
    arch_data = {}
    for arch_name in loaded_archs:
        data_by_dilution = {d: [] for d in LOW_DILUTION_ORDER}
        for tile_info in tile_data_by_arch[arch_name]:
            dilution = tile_info['dilution']
            if dilution in LOW_DILUTION_ORDER:
                data_by_dilution[dilution].append(tile_info['density_threshold_0.5'])
        arch_data[arch_name] = data_by_dilution

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Colors for each architecture
    arch_colors = {
        'unet': '#3498db',  # Blue
        'attention_unet': '#e74c3c',  # Red
        'attention_resunet': '#2ecc71'  # Green
    }

    # Create grouped boxplot
    positions_base = np.arange(len(LOW_DILUTION_ORDER))
    width = 0.25

    for arch_idx, arch_name in enumerate(loaded_archs):
        # Calculate positions for this architecture
        positions = positions_base + (arch_idx - n_archs/2 + 0.5) * width

        # Prepare data
        data = [arch_data[arch_name][d] for d in LOW_DILUTION_ORDER]

        # Create boxplot
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.8,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(facecolor=arch_colors.get(arch_name, '#95a5a6'),
                         color='black', linewidth=1, alpha=0.7),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1),
            capprops=dict(color='black', linewidth=1),
            flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5),
            label=ARCHITECTURE_NAMES.get(arch_name, arch_name)
        )

    # Customize
    ax.set_title('3-Architecture Comparison: Low Dilution Range (Threshold 0.5)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('1/Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Percentage (log scale)', fontsize=14, fontweight='bold')

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)

    # Set x-axis ticks and labels
    inv_dilution_labels = [f'1/{d}' for d in LOW_DILUTION_ORDER]
    ax.set_xticks(positions_base)
    ax.set_xticklabels(inv_dilution_labels, fontsize=12, rotation=45, ha='right')

    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    plt.tight_layout()

    # Save
    filename = 'density_boxplot_3arch_low_range__threshold_0.5.png'
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# PREDICTION FOR 3 ARCHITECTURES
# ============================================================================

def predict_all_architectures(models, test_images_dir, config):
    """
    Predict on all test images using all 3 architectures.
    
    Returns:
        tile_data_by_arch: Dict of {arch_name: list of tile data}
    """
    print("\n" + "="*80)
    print("PREDICTION ON ALL 3 ARCHITECTURES")
    print("="*80)
    
    test_images_dir = Path(test_images_dir)
    image_files = sorted(list(test_images_dir.glob('*.tif')) + list(test_images_dir.glob('*.tiff')))
    
    print(f"Found {len(image_files)} test images")
    print(f"Architectures: {list(models.keys())}")
    print()
    
    tile_data_by_arch = {arch: [] for arch in models.keys()}
    
    for img_file in tqdm(image_files, desc="Processing images"):
        print(f"\nProcessing: {img_file.name}")
        
        dilution = extract_dilution_from_filename(img_file.name)
        dilution_label = f"{dilution}x"
        
        image = load_test_image(img_file)
        tiles_with_pos = extract_tiles(image, config['img_size'])
        
        print(f"  Dilution: {dilution}x, Tiles: {len(tiles_with_pos)}")
        
        # Preprocess tiles once
        tiles_preprocessed = np.array([preprocess_tile(tile) for tile, pos in tiles_with_pos])
        
        # Predict with each architecture
        for arch_name, model in models.items():
            print(f"    Predicting with {ARCHITECTURE_NAMES[arch_name]}...")
            predictions = model.predict(tiles_preprocessed, batch_size=config['batch_size'], verbose=0)
            
            for tile_idx, ((tile, pos), prediction) in enumerate(zip(tiles_with_pos, predictions)):
                density_05 = calculate_foreground_density(prediction, 0.5)
                density_clahe, binary_mask = calculate_density_with_clahe_otsu(prediction)
                
                tile_data_by_arch[arch_name].append({
                    'image': img_file.name,
                    'dilution': dilution,
                    'dilution_label': dilution_label,
                    'tile_idx': tile_idx,
                    'position': pos,
                    'tile': tile,
                    'prediction': prediction,
                    'density_threshold_0.5': density_05,
                    'density_clahe_otsu_pred': density_clahe,
                    'binary_mask_pred': binary_mask,
                })
            
            mean_density = np.mean([td['density_threshold_0.5'] for td in tile_data_by_arch[arch_name] if td['image'] == img_file.name])
            print(f"      Mean density: {mean_density:.4f}")
    
    return tile_data_by_arch

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main analysis pipeline."""
    print("="*80)
    print("3-ARCHITECTURE DENSITY ANALYSIS - DUAL CONTRAST + EDGE FIX")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {CONFIG['output_dir']}")
    print()
    print("FEATURES:")
    print("  ✅ 3 architectures: UNet, Attention UNet, Attention ResUNet")
    print("  ✅ 4-panel visualization (Original + 3 predictions)")
    print("  ✅ DUAL contrast sets (Fixed + Auto)")
    print("  ✅ CLAHE edge artifacts FIXED")
    print("  ✅ Auto-contrast shows actual min/max ranges")
    print("="*80)
    print()
    
    # Find and load best models for all 3 architectures
    print("="*80)
    print("LOADING MODELS")
    print("="*80)
    
    models = {}
    best_models_info = {}
    
    for arch_name, model_dir in CONFIG['model_dirs'].items():
        try:
            best_model_info = find_best_model(model_dir, arch_name)
            model = load_model(best_model_info['model_path'])
            models[arch_name] = model
            best_models_info[arch_name] = best_model_info
        except Exception as e:
            print(f"ERROR loading {arch_name}: {e}")
            print(f"Skipping {arch_name}")
            continue
    
    if len(models) == 0:
        print("ERROR: No models loaded successfully!")
        return
    
    print(f"\n✓ Successfully loaded {len(models)} model(s)")
    print("="*80)
    
    # Predict on all test images with all architectures
    tile_data_by_arch = predict_all_architectures(models, CONFIG['test_images_dir'], CONFIG)
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment info
    experiment_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'density_analysis_3arch_dual_contrast.py',
        'architectures': list(models.keys()),
        'models': {arch: {
            'name': info['model_name'],
            'iou': info['best_val_iou'],
            'path': str(info['model_path'])
        } for arch, info in best_models_info.items()},
        'features': [
            'CLAHE edge artifacts fixed (padding + cropping)',
            'Dual contrast visualization (fixed + auto)',
            '4-panel architecture comparison',
            'Auto-contrast shows actual ranges',
            'Tile indexing bug FIXED (correct matching between architectures)',
            'Boxplots for threshold 0.5 (full range + low dilution range)',
            'Custom colormap highlights min/max pixels (RED=min, CYAN=max)'
        ]
    }
    
    with open(output_dir / 'EXPERIMENT_INFO.json', 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING 4-PANEL VISUALIZATIONS")
    print("="*80)

    create_4panel_tiles_fixed_contrast(tile_data_by_arch, CONFIG['output_dir'], CONFIG)
    create_4panel_tiles_auto_contrast(tile_data_by_arch, CONFIG['output_dir'], CONFIG)

    # Generate boxplots
    print("\n" + "="*80)
    print("GENERATING 3-ARCHITECTURE BOXPLOTS")
    print("="*80)

    create_3arch_boxplot_full_range(tile_data_by_arch, CONFIG['output_dir'], CONFIG)
    create_3arch_boxplot_low_dilution_range(tile_data_by_arch, CONFIG['output_dir'], CONFIG)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Output directory: {CONFIG['output_dir']}")
    print("\nGenerated files:")
    print("  - EXPERIMENT_INFO.json")
    print("  - representative_tiles_4panel_fixed_contrast/")
    print("      * 4 panels: Original | UNet | Attention UNet | Attention ResUNet")
    print("      * Fixed contrast (vmin=0, vmax=1)")
    print("  - representative_tiles_4panel_auto_contrast/")
    print("      * 4 panels: Original | UNet | Attention UNet | Attention ResUNet")
    print("      * Auto contrast (vmin/vmax from data)")
    print("      * Shows actual range values in titles")
    print("      * 🔴 RED pixels = MIN value (proves auto-contrast working)")
    print("      * 🔵 CYAN pixels = MAX value (proves auto-contrast working)")
    print("  - density_boxplot_3arch_full_range__threshold_0.5.png")
    print("      * Grouped boxplot: Full dilution range (1/10240 to 1/10)")
    print("      * Compares all 3 architectures side-by-side")
    print("  - density_boxplot_3arch_low_range__threshold_0.5.png")
    print("      * Grouped boxplot: Low dilution range (1/10240 to 1/80)")
    print("      * Compares all 3 architectures side-by-side")
    print()
    print("FIXES APPLIED:")
    print("  ✓ CLAHE edge artifacts removed (padding + cropping)")
    print("  ✓ Auto-contrast proves it works (shows min/max ranges)")
    print("  ✓ Tile indexing bug FIXED (tiles now match correctly between architectures)")
    print("  ✓ Boxplots added for threshold 0.5 density comparison")
    print("  ✓ Custom colormap highlights min/max pixels (🔴RED=min, 🔵CYAN=max)")
    print("="*80)

if __name__ == '__main__':
    main()

