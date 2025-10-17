#!/usr/bin/env python3
"""
Architecture Comparison Density Analysis

Purpose: Compare UNet, Attention UNet, and Attention ResUNet on test images

Output:
1. Combined boxplots (3 architectures side-by-side for each method)
2. 4-panel tile visualizations (Original + 3 architecture predictions)
3. Comparative statistics CSV

Date: October 17, 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from tensorflow import keras

# Import custom objects
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef, focal_loss
from models_fixed import RepeatElements

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Best models directory (created by copy_best_models.py)
    'best_models_dir': './best_models',

    # Test images
    'test_images_dir': './test_images',

    # Output
    'output_dir': f'./density_analysis_architecture_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Image settings
    'img_size': 512,
    'img_channels': 3,

    # Prediction settings (conservative for Attention ResUNet)
    'batch_size': 2,
    'thresholds': [0.2, 0.5, 0.8, 0.95],

    # Visualization
    'n_representative_tiles': 5,
    'dpi': 300,
    'figsize_boxplot': (16, 8),
    'figsize_tiles': (20, 5),  # 4 panels wide
}

# Dilution order for plotting
DILUTION_ORDER_FULL = [10240, 5120, 2560, 1280, 640, 320, 160, 80]
DILUTION_ORDER_LOW = [10240, 5120, 2560, 1280, 640, 320, 160, 80]  # All 8 dilutions

# Architecture names and colors
ARCHITECTURES = ['UNet', 'Attention_UNet', 'Attention_ResUNet']
ARCH_COLORS = {
    'UNet': '#3498db',  # Blue
    'Attention_UNet': '#e74c3c',  # Red
    'Attention_ResUNet': '#2ecc71'  # Green
}
ARCH_LABELS = {
    'UNet': 'UNet',
    'Attention_UNet': 'Attention UNet',
    'Attention_ResUNet': 'Attention ResUNet'
}

# ============================================================================
# HEADER
# ============================================================================

def print_header(config):
    """Print analysis header"""
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON DENSITY ANALYSIS")
    print("="*80)
    print(f"Script: density_analysis_architecture_comparison.py")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test images: {config['test_images_dir']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Image format: {config['img_size']}×{config['img_size']} RGB ({config['img_channels']} channels)")
    print("="*80)
    print()

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_best_model_info(best_models_dir, architecture_name):
    """
    Load best model info from ./best_models/ directory.

    Args:
        best_models_dir: Directory containing copied best models
        architecture_name: Architecture name (e.g., 'UNet', 'Attention_UNet')

    Returns:
        dict with model_path, model_name, best_val_iou
    """
    print(f"\nLoading {architecture_name} model info...")

    best_models_dir = Path(best_models_dir)
    arch_dir = best_models_dir / architecture_name.lower().replace(' ', '_')

    if not arch_dir.exists():
        raise FileNotFoundError(f"Architecture directory not found: {arch_dir}")

    model_file = arch_dir / 'best_model.keras'
    info_file = arch_dir / 'model_info.json'

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    if not info_file.exists():
        raise FileNotFoundError(f"Model info file not found: {info_file}")

    # Load metadata
    with open(info_file, 'r') as f:
        info = json.load(f)

    print(f"  Model: {info['model_name']}")
    print(f"  Val IoU: {info['best_val_iou']:.4f}")
    print(f"  Path: {model_file}")

    return {
        'model_path': model_file,
        'model_name': info['model_name'],
        'best_val_iou': info['best_val_iou']
    }

def load_model(model_path):
    """Load Keras model with custom objects"""

    # Define custom loss class
    class BinaryFocalLoss(keras.losses.Loss):
        def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.alpha = alpha

        def call(self, y_true, y_pred):
            return focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)

        def get_config(self):
            config = super().get_config()
            config.update({"gamma": self.gamma, "alpha": self.alpha})
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
        'RepeatElements': RepeatElements,  # For Attention UNet and Attention ResUNet
    }

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
    print(f"  ✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    return model

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def extract_dilution_from_filename(filename):
    """Extract dilution factor from filename (e.g., '10240x_...' -> 10240)"""
    parts = filename.split('_')
    for part in parts:
        if part.endswith('x'):
            try:
                return int(part[:-1])
            except ValueError:
                continue
    return None

def load_test_image(image_path):
    """Load test image as RGB float32"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def extract_tiles(image, tile_size):
    """Extract non-overlapping tiles from image"""
    h, w = image.shape[:2]
    tiles_with_pos = []

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if y + tile_size <= h and x + tile_size <= w:
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles_with_pos.append((tile, (y, x)))

    return tiles_with_pos

def preprocess_tile(tile):
    """Preprocess tile for model input (already normalized)"""
    return tile

# ============================================================================
# DENSITY CALCULATION
# ============================================================================

def calculate_foreground_density(prediction, threshold=0.5):
    """Calculate density using simple thresholding"""
    return np.sum(prediction > threshold) / prediction.size

def rescale_image_full_range(img):
    """Rescale image to full 0-1 range"""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-6:
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)

def apply_clahe_otsu(img_gray, clipLimit=2.0, tileGridSize=(8, 8)):
    """Apply CLAHE and Otsu thresholding"""
    # Convert to 8-bit
    img_8bit = (img_gray * 255).astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    enhanced = clahe.apply(img_8bit)

    # Otsu's thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def calculate_density_with_clahe_otsu(prediction, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    Calculate density using CLAHE + Otsu on predicted mask (denoising).

    FIXED: Returns 1 - (white pixels / total) to measure beads (not background)
    """
    # Rescale prediction to full 0-1 range for better CLAHE performance
    pred_rescaled = rescale_image_full_range(prediction.squeeze())

    # Apply CLAHE + Otsu
    binary_mask = apply_clahe_otsu(pred_rescaled, clipLimit, tileGridSize)

    # Calculate density: 1 - (white pixels / total)
    # Otsu gives white=255 for bright regions (background in our case)
    # We want to measure beads (dark), so we invert
    density = 1.0 - (np.count_nonzero(binary_mask) / binary_mask.size)

    return density, binary_mask

def calculate_density_clahe_otsu_on_original(tile):
    """Calculate density using CLAHE + Otsu directly on original image (baseline)"""
    # Convert to grayscale
    if len(tile.shape) == 3:
        tile_gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    else:
        tile_gray = tile

    # Rescale to full range
    tile_rescaled = rescale_image_full_range(tile_gray)

    # Apply CLAHE + Otsu
    binary_mask = apply_clahe_otsu(tile_rescaled)

    # Calculate density (same as above: measure beads, not background)
    density = 1.0 - (np.count_nonzero(binary_mask) / binary_mask.size)

    return density, binary_mask

# ============================================================================
# PREDICTION ON TEST IMAGES (ALL 3 ARCHITECTURES)
# ============================================================================

def predict_on_test_images_all_architectures(models, test_images_dir, config):
    """
    Predict on all test images using all three architectures.

    Args:
        models: dict with keys 'UNet', 'Attention_UNet', 'Attention_ResUNet'
        test_images_dir: Path to test images
        config: Configuration dict

    Returns:
        df_results: DataFrame with architecture comparison results
        tile_data: List of dicts with tile data for visualization
    """
    print("\n" + "="*80)
    print("PREDICTION ON TEST IMAGES (ALL ARCHITECTURES)")
    print("="*80)

    test_images_dir = Path(test_images_dir)
    image_files = sorted(list(test_images_dir.glob('*.tif')) + list(test_images_dir.glob('*.tiff')))

    print(f"Found {len(image_files)} test images")
    print(f"Architectures: {', '.join(ARCHITECTURES)}")
    print()

    tile_results = []
    tile_data = []

    for img_file in tqdm(image_files, desc="Processing images"):
        print(f"\nProcessing: {img_file.name}")

        # Extract dilution
        dilution = extract_dilution_from_filename(img_file.name)
        dilution_label = f"{dilution}x"
        print(f"  Dilution: {dilution}x")

        # Load image
        image = load_test_image(img_file)
        print(f"  Image shape: {image.shape}")

        # Extract tiles
        tiles_with_pos = extract_tiles(image, config['img_size'])
        print(f"  Extracted {len(tiles_with_pos)} tiles")

        # Prepare tiles for prediction
        tiles = np.array([preprocess_tile(tile) for tile, pos in tiles_with_pos])

        # Predict with all three architectures
        predictions_all = {}
        for arch_name in ARCHITECTURES:
            print(f"  Predicting with {arch_name}...")
            predictions_all[arch_name] = models[arch_name].predict(tiles, batch_size=config['batch_size'], verbose=0)

        # Calculate densities for each tile and architecture
        for tile_idx, (tile, pos) in enumerate(tiles_with_pos):
            # Store predictions from all architectures
            arch_predictions = {arch: predictions_all[arch][tile_idx] for arch in ARCHITECTURES}

            # Calculate densities for each architecture and method
            for arch_name in ARCHITECTURES:
                prediction = arch_predictions[arch_name]

                # Threshold methods
                density_02 = calculate_foreground_density(prediction, 0.2)
                density_05 = calculate_foreground_density(prediction, 0.5)
                density_08 = calculate_foreground_density(prediction, 0.8)
                density_095 = calculate_foreground_density(prediction, 0.95)

                # CLAHE+Otsu on predicted mask
                density_clahe_pred, binary_pred = calculate_density_with_clahe_otsu(prediction)

                # CLAHE+Otsu on original (same for all architectures)
                if arch_name == 'UNet':  # Calculate only once
                    density_clahe_orig, binary_orig = calculate_density_clahe_otsu_on_original(tile)
                else:
                    density_clahe_orig = tile_results[-3]['density_clahe_otsu_orig']  # Reuse from UNet

                # Store tile-level results
                tile_results.append({
                    'image': img_file.name,
                    'dilution': dilution,
                    'dilution_label': dilution_label,
                    'tile_idx': tile_idx,
                    'position_y': pos[0],
                    'position_x': pos[1],
                    'architecture': arch_name,
                    'density_threshold_0.2': density_02,
                    'density_threshold_0.5': density_05,
                    'density_threshold_0.8': density_08,
                    'density_threshold_0.95': density_095,
                    'density_clahe_otsu_pred': density_clahe_pred,
                    'density_clahe_otsu_orig': density_clahe_orig,
                })

            # Store tile data for visualization (first occurrence only)
            if tile_idx < config['n_representative_tiles']:
                tile_data.append({
                    'image': img_file.name,
                    'dilution': dilution,
                    'dilution_label': dilution_label,
                    'tile_idx': tile_idx,
                    'position': pos,
                    'tile': tile,
                    'predictions': arch_predictions,  # All 3 architecture predictions
                })

        # Print summary statistics for this image
        print(f"  Summary across all tiles (threshold 0.5):")
        for arch_name in ARCHITECTURES:
            img_results = [r for r in tile_results if r['image'] == img_file.name and r['architecture'] == arch_name]
            mean_density = np.mean([r['density_threshold_0.5'] for r in img_results])
            print(f"    {arch_name:20s}: {mean_density:.4f}")

    df_results = pd.DataFrame(tile_results)

    return df_results, tile_data

# ============================================================================
# COMBINED BOXPLOTS (3 ARCHITECTURES SIDE-BY-SIDE)
# ============================================================================

def create_combined_boxplot(df_results, output_dir, config, density_column='density_threshold_0.5', method_name='Threshold 0.5'):
    """
    Create boxplot comparing all 3 architectures for given density method.
    """
    output_dir = Path(output_dir)

    # Filter to low dilution range
    df_plot = df_results[df_results['dilution'].isin(DILUTION_ORDER_LOW)].copy()

    print(f"\nGenerating combined boxplot: {method_name}")

    fig, ax = plt.subplots(figsize=config['figsize_boxplot'])

    # Prepare data grouped by dilution and architecture
    dilutions = DILUTION_ORDER_LOW
    positions_base = np.arange(len(dilutions))
    box_width = 0.25

    # Colors for each architecture
    colors = [ARCH_COLORS[arch] for arch in ARCHITECTURES]

    # Create boxplots for each architecture
    bp_list = []
    for i, arch_name in enumerate(ARCHITECTURES):
        df_arch = df_plot[df_plot['architecture'] == arch_name]

        # Prepare data for each dilution
        data_by_dilution = []
        for dilution in dilutions:
            dilution_data = df_arch[df_arch['dilution'] == dilution][density_column].values
            data_by_dilution.append(dilution_data)

        # Calculate positions for this architecture
        positions = positions_base + (i - 1) * box_width

        # Create boxplot
        bp = ax.boxplot(
            data_by_dilution,
            positions=positions,
            widths=box_width * 0.8,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(facecolor=colors[i], color='black', linewidth=1, alpha=0.7),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1),
            capprops=dict(color='black', linewidth=1),
            flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5)
        )
        bp_list.append(bp)

    # Customize
    ax.set_title(f'Architecture Comparison: {method_name}\\nParticle Density vs Dilution Factor',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('1/Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Percentage (log scale)', fontsize=14, fontweight='bold')

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)

    # Set x-axis ticks and labels
    ax.set_xticks(positions_base)
    ax.set_xticklabels([f'1/{d}' for d in dilutions], fontsize=11, rotation=45, ha='right')

    # Legend
    legend_patches = [plt.Rectangle((0,0),1,1, facecolor=ARCH_COLORS[arch], edgecolor='black', alpha=0.7)
                     for arch in ARCHITECTURES]
    ax.legend(legend_patches, [ARCH_LABELS[arch] for arch in ARCHITECTURES],
             loc='upper left', fontsize=12, framealpha=0.9)

    plt.tight_layout()

    # Save
    safe_name = method_name.replace(' ', '_').replace('+', '').replace('.', 'p').lower()
    output_path = output_dir / f'density_boxplot_comparison_{safe_name}.png'
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")

# ============================================================================
# 4-PANEL TILE VISUALIZATIONS (ORIGINAL + 3 ARCHITECTURES)
# ============================================================================

def create_4panel_tile_visualizations(tile_data, output_dir, config):
    """
    Create 4-panel tile visualizations: Original + UNet + Attention UNet + Attention ResUNet.
    All predictions are inverted (white beads on black background).
    """
    output_dir = Path(output_dir)
    tiles_dir = output_dir / 'representative_tiles_4panel'
    tiles_dir.mkdir(exist_ok=True, parents=True)

    print("\n--- 4-Panel Tile Visualizations (Architecture Comparison) ---")
    print("Generating representative tile visualizations (4-panel)...")

    # Group by image
    images = {}
    for tile in tile_data:
        img_name = tile['image']
        if img_name not in images:
            images[img_name] = []
        images[img_name].append(tile)

    for img_name, tiles in images.items():
        print(f"  Creating 4-panel tiles for: {img_name}")

        # Take first n_representative_tiles
        tiles = tiles[:config['n_representative_tiles']]

        n_tiles = len(tiles)
        fig, axes = plt.subplots(n_tiles, 4, figsize=(config['figsize_tiles'][0], config['figsize_tiles'][1] * n_tiles))

        if n_tiles == 1:
            axes = axes.reshape(1, -1)

        for row, tile in enumerate(tiles):
            # Panel 1: Original tile (dark beads)
            axes[row, 0].imshow(tile['tile'])
            axes[row, 0].axis('off')
            if row == 0:
                axes[row, 0].set_title('Original Tile\\n(Dark Beads)', fontsize=10, fontweight='bold')

            # Panels 2-4: Architecture predictions (inverted - white beads)
            for col, arch_name in enumerate(ARCHITECTURES, start=1):
                prediction = tile['predictions'][arch_name].squeeze()

                # Invert: 1 - prediction (white beads on black background)
                inverted_pred = 1.0 - prediction

                axes[row, col].imshow(inverted_pred, cmap='gray', vmin=0, vmax=1)
                axes[row, col].axis('off')

                if row == 0:
                    axes[row, col].set_title(f'{ARCH_LABELS[arch_name]}\\n(White Beads)', fontsize=10, fontweight='bold')

        plt.tight_layout()

        # Save
        dilution_label = tile['dilution_label']
        safe_name = img_name.replace('.tif', '').replace('.tiff', '')
        output_path = tiles_dir / f'tiles_4panel_{dilution_label}_{safe_name}.png'
        plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved: {output_path.name}")

    print(f"\n  ✓ All 4-panel tile visualizations saved to: {tiles_dir}")
    print(f"    Total images: {len(images)}")
    print(f"    Format: 4 panels (Original, UNet, Attention UNet, Attention ResUNet) - Inverted Predictions")

# ============================================================================
# MAIN
# ============================================================================

def main():
    config = CONFIG

    # Print header
    print_header(config)

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    # ========================================================================
    # LOAD BEST MODELS FROM ./best_models/ DIRECTORY
    # ========================================================================

    print("\n" + "="*80)
    print("LOADING BEST MODELS")
    print("="*80)

    best_models = {}

    # UNet
    best_models['UNet'] = load_best_model_info(
        config['best_models_dir'],
        'UNet'
    )

    # Attention UNet
    best_models['Attention_UNet'] = load_best_model_info(
        config['best_models_dir'],
        'Attention_UNet'
    )

    # Attention ResUNet
    best_models['Attention_ResUNet'] = load_best_model_info(
        config['best_models_dir'],
        'Attention_ResUNet'
    )

    # Load all models
    print("\n" + "="*80)
    print("LOADING KERAS MODELS")
    print("="*80)

    models = {}
    for arch_name in ARCHITECTURES:
        print(f"\n--- {arch_name} ---")
        models[arch_name] = load_model(best_models[arch_name]['model_path'])

    # ========================================================================
    # PREDICT ON TEST IMAGES
    # ========================================================================

    df_results, tile_data = predict_on_test_images_all_architectures(
        models, config['test_images_dir'], config
    )

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Tile-level results
    tile_csv = output_dir / 'architecture_comparison_tile_level.csv'
    df_results.to_csv(tile_csv, index=False)
    print(f"✓ Tile-level results saved: {tile_csv}")
    print(f"  Total rows: {len(df_results)} (tiles × architectures)")

    # Image-level summary (group by image, dilution, architecture)
    df_summary = df_results.groupby(['image', 'dilution', 'dilution_label', 'architecture']).agg({
        'density_threshold_0.2': ['mean', 'median', 'std'],
        'density_threshold_0.5': ['mean', 'median', 'std'],
        'density_threshold_0.8': ['mean', 'median', 'std'],
        'density_threshold_0.95': ['mean', 'median', 'std'],
        'density_clahe_otsu_pred': ['mean', 'median', 'std'],
        'density_clahe_otsu_orig': ['mean', 'median', 'std'],
    }).reset_index()

    # Flatten column names
    df_summary.columns = ['_'.join(col).strip('_') for col in df_summary.columns.values]

    summary_csv = output_dir / 'architecture_comparison_image_summary.csv'
    df_summary.to_csv(summary_csv, index=False)
    print(f"✓ Image-level summary saved: {summary_csv}")
    print(f"  Total rows: {len(df_summary)} (images × architectures)")

    # Experiment info
    exp_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'density_analysis_architecture_comparison.py',
        'architectures': {
            arch_name: {
                'model_path': str(best_models[arch_name]['model_path']),
                'model_name': best_models[arch_name]['model_name'],
                'best_val_iou': best_models[arch_name]['best_val_iou'],
            }
            for arch_name in ARCHITECTURES
        },
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'total_images': len(df_results['image'].unique()),
        'total_tiles_per_arch': len(df_results) // len(ARCHITECTURES),
    }

    exp_json = output_dir / 'EXPERIMENT_INFO.json'
    with open(exp_json, 'w') as f:
        json.dump(exp_info, f, indent=2)
    print(f"✓ Experiment info saved: {exp_json}")

    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Combined boxplots for each density method
    methods = [
        ('density_threshold_0.2', 'Threshold 0.2'),
        ('density_threshold_0.5', 'Threshold 0.5'),
        ('density_threshold_0.8', 'Threshold 0.8'),
        ('density_threshold_0.95', 'Threshold 0.95'),
        ('density_clahe_otsu_pred', 'CLAHE+Otsu on Pred'),
        ('density_clahe_otsu_orig', 'CLAHE+Otsu on Original'),
    ]

    for density_col, method_name in methods:
        create_combined_boxplot(df_results, output_dir, config, density_col, method_name)

    # 4-panel tile visualizations
    create_4panel_tile_visualizations(tile_data, output_dir, config)

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print()
    print("Generated files:")
    print(f"  - architecture_comparison_tile_level.csv")
    print(f"  - architecture_comparison_image_summary.csv")
    print(f"  - EXPERIMENT_INFO.json")
    print()
    print("Generated visualizations:")
    print(f"  - 6 combined boxplots (3 architectures per plot)")
    print(f"  - 4-panel tile visualizations (Original + 3 architectures)")
    print("="*80)

if __name__ == '__main__':
    main()
