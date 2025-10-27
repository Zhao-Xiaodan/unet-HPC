#!/usr/bin/env python3
"""
Density Analysis Using Best 512×512 Grayscale Models
=====================================================

Uses the top 5 configurations from hyperparameter_search_512_20251014_235755
to predict on test images and generate density analysis.

Generates:
- Individual box plots for each model (5 separate PNG files)
- Representative 512×512 tiles with 4-panel comparison (Original + top 3 models)
- CSV with all density data

Process:
1. Extract 512×512 tiles from test images (non-overlapping)
2. Predict on each tile with all 5 models
3. Calculate foreground density for each prediction
4. Select 5 representative tiles per dilution
5. Generate box plots and tile visualizations

Models used (from hyperparameter_search_512_20251014_235755):
1. unet_lr0.0001_drop0.3_bs4 (Best: 0.1533 ± 0.0578)
2. unet_lr5e-05_drop0.2_bs4 (0.1327 ± 0.0176)
3. unet_lr5e-05_drop0.3_bs4 (0.1308 ± 0.0137)
4. resunet_lr5e-05_drop0.3_bs4 (0.1117 ± 0.0131)
5. attention_resunet_lr5e-05_drop0.2_bs4 (0.1091 ± 0.0064)

Author: Claude Code
Date: October 15, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json

import tensorflow as tf
from tensorflow import keras

# Import custom modules
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef

# ============================================================================
# CONFIGURATION
# ============================================================================

# Top 5 configurations from hyperparameter search
TOP_CONFIGS = [
    {
        'name': 'unet_lr0.0001_drop0.3_bs4',
        'short_name': 'unet_best',
        'architecture': 'unet',
        'mean_jaccard': 0.1533,
        'label': 'U-Net (best)',
    },
    {
        'name': 'unet_lr5e-05_drop0.2_bs4',
        'short_name': 'unet_lr5e-05_d0.2',
        'architecture': 'unet',
        'mean_jaccard': 0.1327,
        'label': 'U-Net (lr5e-05, d0.2)',
    },
    {
        'name': 'unet_lr5e-05_drop0.3_bs4',
        'short_name': 'unet_lr5e-05_d0.3',
        'architecture': 'unet',
        'mean_jaccard': 0.1308,
        'label': 'U-Net (lr5e-05, d0.3)',
    },
    {
        'name': 'resunet_lr5e-05_drop0.3_bs4',
        'short_name': 'resunet',
        'architecture': 'resunet',
        'mean_jaccard': 0.1117,
        'label': 'ResUNet',
    },
    {
        'name': 'attention_resunet_lr5e-05_drop0.2_bs4',
        'short_name': 'attention_resunet',
        'architecture': 'attention_resunet',
        'mean_jaccard': 0.1091,
        'label': 'Attention ResUNet',
    }
]

CONFIG = {
    # Directories
    'model_dir': './hyperparameter_search_512_20251014_235755',
    'test_images_dir': './test_images',
    'output_dir': f'./density_analysis_512_grayscale_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Image settings (from training)
    'img_size': 512,  # Tile size
    'img_channels': 1,  # Grayscale

    # Prediction settings
    'batch_size': 4,
    'threshold': 0.5,  # Binary threshold for predictions

    # Representative tiles
    'n_representative_tiles': 5,

    # Plotting
    'dpi': 300,
    'figsize_comparison': (16, 4),  # 4-panel comparison
    'figsize_boxplot': (14, 8),
}

# Dilution factor patterns (for extracting from filenames)
DILUTION_PATTERNS = {
    '10x': 10, '20x': 20, '40x': 40, '80x': 80, '160x': 160,
    '320x': 320, '640x': 640, '1280x': 1280, '2560x': 2560,
    '5120x': 5120, '10240x': 10240
}

print("="*80)
print("DENSITY ANALYSIS - 512×512 GRAYSCALE MODELS")
print("="*80)
print(f"Script: density_analysis_512_grayscale.py")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model directory: {CONFIG['model_dir']}")
print(f"Test images: {CONFIG['test_images_dir']}")
print(f"Output directory: {CONFIG['output_dir']}")
print()

print("Models to be used (Top 5 from hyperparameter search):")
for i, config in enumerate(TOP_CONFIGS, 1):
    print(f"  {i}. {config['label']}: Jaccard {config['mean_jaccard']:.4f}")
print("="*80)
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_dilution_from_filename(filename):
    """Extract dilution factor from filename."""
    for pattern, dilution in DILUTION_PATTERNS.items():
        if pattern in filename.lower():
            return dilution
    return None

def rescale_image_full_range(img):
    """Rescale image to full 0-255 range."""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)

def extract_tiles_512(image, tile_size=512):
    """Extract 512×512 tiles (non-overlapping) from larger image."""
    h, w = image.shape[:2]
    tiles, positions = [], []

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions

def calculate_foreground_percentage(mask, threshold=0.5):
    """Calculate percentage of foreground pixels."""
    binary_mask = (mask > threshold).astype(np.float32)
    foreground_pixels = np.sum(binary_mask)
    total_pixels = mask.size
    percentage = (foreground_pixels / total_pixels) * 100
    return percentage

def select_representative_tiles(tiles, densities, n=5):
    """Select n representative tiles spanning density range."""
    if len(tiles) == 0:
        return []
    if len(tiles) <= n:
        return list(range(len(tiles)))

    # Sort by density
    sorted_indices = np.argsort(densities)

    # Select evenly spaced indices
    step = len(sorted_indices) / n
    selected = [sorted_indices[int(i * step)] for i in range(n)]

    return selected

def load_model_for_config(config_name, model_dir, fold=1):
    """Load trained model for a specific configuration and fold."""
    model_pattern = f"{config_name.replace('_lr', f'_fold{fold}_lr')}_model.keras"
    model_path = Path(model_dir) / model_pattern

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"  Loading: {model_path.name}")

    # Load with custom objects
    model = keras.models.load_model(
        str(model_path),
        custom_objects={
            'combined_dice_focal_loss': combined_dice_focal_loss,
            'jacard_coef': jacard_coef,
            'dice_coef': dice_coef
        }
    )

    return model

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_tile_comparison(original_tile, masks_dict, density_dict, image_name,
                          tile_idx, dilution, output_dir):
    """
    Create 4-panel comparison: Original + top 3 models.
    Format: [Original | Model1 | Model2 | Model3]
    """
    fig, axes = plt.subplots(1, 4, figsize=CONFIG['figsize_comparison'])

    # Panel 1: Original
    axes[0].imshow(original_tile, cmap='gray')
    axes[0].set_title('Original Tile\n512×512', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panels 2-4: Top 3 models (all U-Net variants)
    model_order = ['unet_best', 'unet_lr5e-05_d0.2', 'unet_lr5e-05_d0.3']
    model_names = ['U-Net (best)', 'U-Net (lr5e-05, d0.2)', 'U-Net (lr5e-05, d0.3)']

    for i, (model_key, name) in enumerate(zip(model_order, model_names), 1):
        if model_key in masks_dict:
            # Show binary mask
            binary_mask = (masks_dict[model_key] > CONFIG['threshold']).astype(np.uint8) * 255
            axes[i].imshow(binary_mask, cmap='gray')
            density = density_dict.get(model_key, 0)
            axes[i].set_title(f'{name}\nDensity: {density:.2f}%',
                            fontsize=11, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, f'{name}\nN/A',
                        ha='center', va='center', fontsize=12)
        axes[i].axis('off')

    plt.suptitle(f'{dilution}x Dilution - {image_name} - Tile {tile_idx}',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    filename = f'{image_name}_tile_{tile_idx:02d}_comparison.png'
    save_path = output_dir / filename
    plt.savefig(str(save_path), dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()

def create_boxplot(df, model_name, model_label, output_path):
    """Create boxplot for one model across all dilutions."""
    df_model = df[df['model_name'] == model_name].copy()

    if len(df_model) == 0:
        print(f"  ⚠ No data for {model_label}")
        return

    df_model = df_model.sort_values('dilution_factor')
    dilution_factors = sorted(df_model['dilution_factor'].unique())

    fig, ax = plt.subplots(figsize=CONFIG['figsize_boxplot'])

    data_to_plot = []
    labels = []
    positions = []

    for i, dilution in enumerate(dilution_factors):
        df_dilution = df_model[df_model['dilution_factor'] == dilution]
        data_to_plot.append(df_dilution['foreground_pct'].values)
        labels.append(f"{int(dilution)}x")
        positions.append(i)

    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5)
    )

    ax.set_xlabel('Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Percentage (log scale)', fontsize=14, fontweight='bold')
    ax.set_title(f'Particle Density vs. Dilution Factor\n(Model: {model_label})',
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')

    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {output_path.name}")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_density_analysis():
    """Run complete density analysis."""

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = output_dir / 'csv_data'
    csv_dir.mkdir(exist_ok=True)

    plots_dir = output_dir / 'boxplots'
    plots_dir.mkdir(exist_ok=True)

    tiles_dir = output_dir / 'representative_tiles'
    tiles_dir.mkdir(exist_ok=True)

    # ========================================================================
    # PHASE 1: LOAD MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: LOADING MODELS")
    print("="*80)

    models = {}
    for config in TOP_CONFIGS:
        try:
            model = load_model_for_config(config['name'], CONFIG['model_dir'], fold=1)
            models[config['short_name']] = {
                'model': model,
                'config': config
            }
            print(f"  ✓ Loaded {config['label']}")
        except FileNotFoundError as e:
            print(f"  ✗ Failed to load {config['label']}: {e}")

    if len(models) == 0:
        raise ValueError("No models loaded successfully!")

    print(f"\n✓ Successfully loaded {len(models)} models")
    print()

    # ========================================================================
    # PHASE 2: LOAD TEST IMAGES
    # ========================================================================
    print("="*80)
    print("PHASE 2: LOADING TEST IMAGES")
    print("="*80)

    test_dir = Path(CONFIG['test_images_dir'])
    image_files = sorted(list(test_dir.glob('*.tif')) + list(test_dir.glob('*.tiff')))

    if len(image_files) == 0:
        raise ValueError(f"No test images found in {test_dir}")

    print(f"Found {len(image_files)} test images")

    # Group images by dilution
    images_by_dilution = {}
    for img_path in image_files:
        dilution = extract_dilution_from_filename(img_path.name)
        if dilution is not None:
            if dilution not in images_by_dilution:
                images_by_dilution[dilution] = []
            images_by_dilution[dilution].append(img_path)

    print(f"Dilution factors found: {sorted(images_by_dilution.keys())}")
    for dilution in sorted(images_by_dilution.keys()):
        print(f"  {dilution}x: {len(images_by_dilution[dilution])} images")
    print()

    # ========================================================================
    # PHASE 3: PREDICT ON ALL IMAGES
    # ========================================================================
    print("="*80)
    print("PHASE 3: PREDICTION AND DENSITY ANALYSIS")
    print("="*80)

    all_results = []

    for dilution in tqdm(sorted(images_by_dilution.keys()), desc="Dilution factors"):
        for img_path in tqdm(images_by_dilution[dilution],
                            desc=f"  {dilution}x images",
                            leave=False):

            print(f"\nProcessing: {img_path.name} (dilution: {dilution}x)")

            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  ✗ Could not read {img_path.name}")
                continue

            # Rescale and normalize
            img_rescaled = rescale_image_full_range(img)
            img_normalized = img_rescaled.astype(np.float32) / 255.0

            # Extract 512×512 tiles
            tiles, positions = extract_tiles_512(img_normalized, tile_size=CONFIG['img_size'])

            if len(tiles) == 0:
                print(f"  ⚠ No tiles extracted")
                continue

            print(f"  Extracted {len(tiles)} tiles (512×512)")

            # Predict with all models and calculate densities
            tile_data = []

            for tile_idx, tile in enumerate(tiles):
                tile_info = {
                    'tile_idx': tile_idx,
                    'tile': tile,
                    'masks': {},
                    'densities': {}
                }

                # Add channel dimension for prediction
                tile_batch = np.expand_dims(tile, axis=(0, -1))  # (1, 512, 512, 1)

                # Predict with each model
                for model_key, model_data in models.items():
                    pred = model_data['model'].predict(tile_batch, verbose=0)
                    pred_mask = pred[0, :, :, 0]  # Remove batch and channel dims

                    tile_info['masks'][model_key] = pred_mask
                    foreground_pct = calculate_foreground_percentage(pred_mask, CONFIG['threshold'])
                    tile_info['densities'][model_key] = foreground_pct

                    # Record result
                    all_results.append({
                        'image': img_path.stem,
                        'dilution_factor': dilution,
                        'tile_idx': tile_idx,
                        'model_name': model_key,
                        'model_label': model_data['config']['label'],
                        'architecture': model_data['config']['architecture'],
                        'foreground_pct': foreground_pct
                    })

                tile_data.append(tile_info)

            # Select 5 representative tiles based on U-Net (best) densities
            unet_best_densities = [t['densities']['unet_best'] for t in tile_data]
            representative_indices = select_representative_tiles(
                tiles, unet_best_densities, CONFIG['n_representative_tiles']
            )

            print(f"  Selected {len(representative_indices)} representative tiles")

            # Create visualizations for representative tiles
            for tile_idx in representative_indices:
                tile_info = tile_data[tile_idx]

                create_tile_comparison(
                    tile_info['tile'],
                    tile_info['masks'],
                    tile_info['densities'],
                    img_path.stem,
                    tile_idx,
                    dilution,
                    tiles_dir
                )

            print(f"  ✓ Saved {len(representative_indices)} comparison images")

    print(f"\n✓ Completed predictions on {len(all_results)} tile-model combinations")
    print()

    # ========================================================================
    # PHASE 4: SAVE RESULTS AND CREATE BOXPLOTS
    # ========================================================================
    print("="*80)
    print("PHASE 4: GENERATING OUTPUTS")
    print("="*80)

    # Save CSV
    results_df = pd.DataFrame(all_results)
    csv_path = csv_dir / 'density_analysis_all_models.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    print()

    # Create individual boxplots for each model
    print("Creating box plots...")
    for config in TOP_CONFIGS:
        plot_path = plots_dir / f"{config['short_name']}_density_vs_dilution.png"
        create_boxplot(results_df, config['short_name'], config['label'], plot_path)

    # Save experiment metadata
    metadata = {
        'experiment_name': 'Density Analysis - 512×512 Grayscale Models',
        'python_script': 'density_analysis_512_grayscale.py',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_source': CONFIG['model_dir'],
        'test_images': CONFIG['test_images_dir'],
        'output_directory': str(output_dir),
        'tile_size': CONFIG['img_size'],
        'models_used': [
            {
                'name': config['name'],
                'short_name': config['short_name'],
                'label': config['label'],
                'architecture': config['architecture'],
                'mean_jaccard': config['mean_jaccard']
            }
            for config in TOP_CONFIGS
        ],
        'num_test_images': len(image_files),
        'dilution_factors': sorted(list(images_by_dilution.keys())),
        'total_predictions': len(all_results),
        'n_representative_tiles_per_image': CONFIG['n_representative_tiles'],
    }

    with open(output_dir / 'EXPERIMENT_INFO.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: EXPERIMENT_INFO.json")
    print()

    print("="*80)
    print("DENSITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"  - Box plots (5 models): {plots_dir}")
    print(f"  - Representative tiles: {tiles_dir}")
    print(f"  - CSV data: {csv_dir}")
    print("="*80)

if __name__ == '__main__':
    run_density_analysis()
