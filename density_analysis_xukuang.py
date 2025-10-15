#!/usr/bin/env python3
"""
Multi-Model Density Analysis Using Xukuang Parameters
=====================================================

Script: density_analysis_xukuang.py
PBS Script: pbs_density_analysis_xukuang.sh

Uses THREE models from xukuang_params_shrunk_20251015_071224:
1. UNet (Final Val IoU: 0.6065, Best: 0.6789)
2. Attention UNet
3. Attention ResUNet

to predict on test images and generate comprehensive density analysis.

Training Parameters:
- Learning Rate: 0.005
- Epochs: 200
- Batch Size: 4
- Loss: BinaryFocalLoss(gamma=2)
- Image Size: 512×512 RGB

NEW FEATURES:
- Save ALL tile-level density values (n=28 per image)
- Box plots for each model showing tile-level distributions
- 4-panel tile comparisons (Original, UNet, Attention UNet, Attention ResUNet)
- Comprehensive multi-model comparison

Process:
1. Load all three models from Xukuang experiment
2. Extract 512×512 tiles from test images (non-overlapping)
3. Predict on each tile with ALL models
4. Save tile-level density values to CSV
5. Generate multi-model visualizations with CORRECT dilution ordering

Author: Claude Code
Date: October 15, 2025
Updated: Adding multi-model comparison
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
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# Import custom modules
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef, focal_loss

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Directories
    'model_dir': './xukuang_params_shrunk_20251015_071224',
    'test_images_dir': './test_images',
    'output_dir': f'./density_analysis_xukuang_multimodel_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Models to load (all three architectures)
    'models': ['unet', 'attention_unet', 'attention_resunet'],

    # Image settings (from Xukuang training - RGB!)
    'img_size': 512,  # Tile size
    'img_channels': 3,  # RGB (NOT grayscale!)

    # Prediction settings
    'batch_size': 8,
    'threshold': 0.5,  # Binary threshold for predictions

    # Representative tiles (for 4-panel comparisons)
    'n_representative_tiles': 5,

    # Plotting
    'dpi': 300,
    'figsize_4panel': (24, 6),  # For 4-panel tile comparisons
    'figsize_boxplot': (18, 12),  # Larger for multi-model comparison
}

# CORRECTED: Dilution factors in proper order (10x - 10240x)
# Previous issue: ordering was wrong in boxplot due to string sorting
DILUTION_ORDER = [10, 20, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
DILUTION_LABELS = ['10x', '20x', '80x', '160x', '320x', '640x', '1280x', '2560x', '5120x', '10240x']

# Dilution factor patterns (for extracting from filenames)
DILUTION_PATTERNS = {
    '10240x': 10240,
    '5120x': 5120,
    '2560x': 2560,
    '1280x': 1280,
    '640x': 640,
    '320x': 320,
    '160x': 160,
    '80x': 80,
    '20x': 20,
    '10x': 10,
}

# ============================================================================
# HEADER
# ============================================================================

def print_header(config):
    """Print analysis header."""
    print("="*80)
    print("MULTI-MODEL DENSITY ANALYSIS - XUKUANG MODELS (RGB)")
    print("="*80)
    print(f"Script: density_analysis_xukuang.py")
    print(f"PBS Script: pbs_density_analysis_xukuang.sh")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model directory: {config['model_dir']}")
    print(f"Test images: {config['test_images_dir']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Image format: {config['img_size']}×{config['img_size']} RGB ({config['img_channels']} channels)")
    print()
    print("Models: UNet, Attention UNet, Attention ResUNet (FINAL epoch 200 models)")
    print("  - Final Val IoU (Epoch 200): 0.6065 (UNet)")
    print("  - Best Val IoU (Epoch 140): 0.6789 (UNet)")
    print("  - Training: LR=0.005, 200 epochs, BinaryFocalLoss")
    print("  - NOTE: Using FINAL epoch models, not best checkpoint")
    print("="*80)
    print()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_dilution_from_filename(filename):
    """Extract dilution factor from filename."""
    filename_lower = filename.lower()

    # Try to match dilution patterns (highest first to avoid partial matches)
    for pattern, dilution in DILUTION_PATTERNS.items():
        if pattern in filename_lower:
            return dilution

    # Default to 1x (undiluted)
    return 1

def load_test_image(image_path):
    """Load test image and convert to RGB if needed."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    # Convert to RGB
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 3:  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def extract_tiles(image, tile_size):
    """
    Extract non-overlapping tiles from image.
    Returns list of (tile, position) tuples.
    """
    h, w = image.shape[:2]
    tiles_with_pos = []

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles_with_pos.append((tile, (y, x)))

    return tiles_with_pos

def preprocess_tile(tile):
    """Preprocess tile for model input (normalize to [0, 1])."""
    # Ensure RGB format
    if len(tile.shape) == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)

    # Normalize to [0, 1]
    tile = tile.astype(np.float32) / 255.0

    return tile

def calculate_foreground_density(prediction, threshold=0.5):
    """
    Calculate foreground density from prediction mask.
    Returns density as fraction of pixels above threshold.
    """
    binary_mask = (prediction > threshold).astype(np.float32)
    density = np.mean(binary_mask)
    return density

# ============================================================================
# MODEL LOADING
# ============================================================================

def find_model_file(model_dir, model_name):
    """
    Find model file in directory.
    Xukuang experiment naming: {model_name}_xukuang_params_shrunk.keras
    """
    model_dir = Path(model_dir)

    # Xukuang experiment naming convention
    xukuang_name = f"{model_name}_xukuang_params_shrunk.keras"
    model_path = model_dir / xukuang_name

    if model_path.exists():
        return model_path

    raise FileNotFoundError(f"Could not find model file: {xukuang_name} in {model_dir}")

def load_model(model_dir, model_name):
    """Load the trained model."""
    print(f"Loading model: {model_name}")

    model_path = find_model_file(model_dir, model_name)
    print(f"  Model file: {model_path}")

    # Define BinaryFocalLoss class for deserialization
    @keras.saving.register_keras_serializable(package='Custom')
    class BinaryFocalLoss(keras.losses.Loss):
        """Binary Focal Loss for model loading compatibility."""
        def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.alpha = alpha

        def call(self, y_true, y_pred):
            return focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)

        def get_config(self):
            config = super().get_config()
            config.update({
                'gamma': self.gamma,
                'alpha': self.alpha,
            })
            return config

    # Custom objects for loading
    # Note: 'K' (Keras backend) is required for Lambda layers in Attention models
    # that use K.repeat_elements() and other backend functions
    custom_objects = {
        'BinaryFocalLoss': BinaryFocalLoss,
        'binary_focal_loss': BinaryFocalLoss,
        'combined_dice_focal_loss': combined_dice_focal_loss,
        'jacard_coef': jacard_coef,
        'dice_coef': dice_coef,
        'focal_loss': focal_loss,
        'K': K,  # Keras backend for Lambda layers
    }

    # Load model with safe_mode=False to allow Lambda layers
    # Note: Attention models use Lambda layers for attention mechanisms
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        safe_mode=False  # Required for Lambda layers in Attention models
    )
    print(f"  ✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print()

    return model

def load_all_models(config):
    """Load all three models."""
    models = {}
    print("="*80)
    print("LOADING MODELS")
    print("="*80)
    print()

    for model_name in config['models']:
        models[model_name] = load_model(config['model_dir'], model_name)

    print(f"✓ Loaded {len(models)} models successfully")
    print()

    return models

# ============================================================================
# PREDICTION AND DENSITY CALCULATION
# ============================================================================

def predict_on_test_images_multimodel(models, test_images_dir, config):
    """
    Predict on all test images with ALL models.
    Returns DataFrame with TILE-LEVEL results for each model.
    """
    test_images_dir = Path(test_images_dir)
    image_files = sorted(test_images_dir.glob("*.tif")) + sorted(test_images_dir.glob("*.tiff"))

    print(f"Found {len(image_files)} test images")
    print()

    tile_results = []  # Store ALL tile-level results
    tile_data = []  # For visualization (store predictions from all models)

    for img_file in tqdm(image_files, desc="Processing images"):
        # Extract dilution
        dilution = extract_dilution_from_filename(img_file.name)
        dilution_label = f"{dilution}x" if dilution > 1 else "undiluted"

        print(f"\nProcessing: {img_file.name} (Dilution: {dilution_label})")

        # Load image
        image = load_test_image(img_file)
        print(f"  Image shape: {image.shape}")

        # Extract tiles
        tiles_with_pos = extract_tiles(image, config['img_size'])
        print(f"  Extracted {len(tiles_with_pos)} tiles")

        # Predict with ALL models
        for tile_idx, (tile, pos) in enumerate(tiles_with_pos):
            # Preprocess
            tile_prep = preprocess_tile(tile)
            tile_batch = np.expand_dims(tile_prep, axis=0)

            # Store predictions from all models
            predictions = {}
            densities = {}

            for model_name, model in models.items():
                # Predict
                pred = model.predict(tile_batch, verbose=0)[0, :, :, 0]
                predictions[model_name] = pred

                # Calculate density
                density = calculate_foreground_density(pred, config['threshold'])
                densities[model_name] = density

                # Store tile-level result
                tile_results.append({
                    'image': img_file.name,
                    'dilution': dilution,
                    'dilution_label': dilution_label,
                    'tile_idx': tile_idx,
                    'position_y': pos[0],
                    'position_x': pos[1],
                    'model': model_name,
                    'density': density
                })

            # Store tile data for visualization (include all model predictions)
            tile_data.append({
                'image': img_file.name,
                'dilution': dilution,
                'dilution_label': dilution_label,
                'tile_idx': tile_idx,
                'position': pos,
                'tile': tile,
                'predictions': predictions,  # Dict with all model predictions
                'densities': densities,  # Dict with all model densities
            })

        # Print summary for this image
        print(f"  Densities (mean across {len(tiles_with_pos)} tiles):")
        for model_name in config['models']:
            model_densities = [d['density'] for d in tile_results
                             if d['image'] == img_file.name and d['model'] == model_name]
            mean_density = np.mean(model_densities)
            std_density = np.std(model_densities)
            print(f"    {model_name:20s}: {mean_density:.4f} ± {std_density:.4f}")

    df_tile_results = pd.DataFrame(tile_results)

    return df_tile_results, tile_data

# ============================================================================
# VISUALIZATION - BOX PLOTS
# ============================================================================

def create_multimodel_boxplot(df_tile_results, output_dir, config):
    """
    Create box plots for ALL models showing tile-level density distributions.
    One subplot per model.
    Uses 1/Dilution on x-axis with log scale (matching reference style).
    """
    print("\nGenerating multi-model box plots...")

    # Filter to only include dilutions in our defined order
    df_plot = df_tile_results[df_tile_results['dilution'].isin(DILUTION_ORDER)].copy()

    # Calculate 1/dilution for x-axis
    df_plot['inv_dilution'] = 1.0 / df_plot['dilution']

    # Create inverse dilution labels
    inv_dilution_labels = [f'1/{d}' for d in DILUTION_ORDER]

    # Create figure with subplots (one per model)
    n_models = len(config['models'])
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 6*n_models))

    if n_models == 1:
        axes = [axes]

    # Colors: blue boxes with orange median like reference
    box_color = '#5FA3D9'  # Light blue
    median_color = '#FF8C42'  # Orange

    for idx, model_name in enumerate(config['models']):
        ax = axes[idx]

        # Filter data for this model
        df_model = df_plot[df_plot['model'] == model_name].copy()

        # Sort by dilution (highest to lowest dilution = lowest to highest 1/dilution)
        df_model = df_model.sort_values('dilution', ascending=False)

        # Create positions for boxplot (log scale)
        positions = [1.0/d for d in DILUTION_ORDER]

        # Prepare data for each dilution
        data_by_dilution = []
        for dilution in DILUTION_ORDER:
            dilution_data = df_model[df_model['dilution'] == dilution]['density'].values
            data_by_dilution.append(dilution_data)

        # Create boxplot
        bp = ax.boxplot(
            data_by_dilution,
            positions=positions,
            widths=[p*0.15 for p in positions],  # Width proportional to position
            patch_artist=True,
            boxprops=dict(facecolor=box_color, color='black', linewidth=1),
            medianprops=dict(color=median_color, linewidth=2),
            whiskerprops=dict(color='black', linewidth=1),
            capprops=dict(color='black', linewidth=1),
            flierprops=dict(marker='o', markerfacecolor='black', markersize=3,
                          linestyle='none', markeredgecolor='black')
        )

        # Set log scale on x-axis
        ax.set_xscale('log')

        # Customize
        ax.set_title(f'{model_name.upper().replace("_", " ")} - Foreground Percentage vs 1/Dilution (Log Scale)',
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('1 / Dilution Factor', fontsize=14)
        ax.set_ylabel('Foreground Percentage', fontsize=14)

        # Set y-axis to log scale like reference
        ax.set_yscale('log')
        ax.set_ylim(0.002, 1.5)  # Slightly wider range for visibility

        # Grid
        ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)

        # Set x-axis ticks and labels
        ax.set_xticks(positions)
        ax.set_xticklabels(inv_dilution_labels, fontsize=12)

        # Rotate x-axis labels if needed
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'density_boxplot_multimodel.png'
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")

def create_model_comparison_boxplot(df_tile_results, output_dir, config):
    """
    Create combined box plot comparing all models side-by-side.
    Uses 1/Dilution on x-axis with log scale (matching reference style).
    """
    print("\nGenerating model comparison box plot...")

    # Filter to only include dilutions in our defined order
    df_plot = df_tile_results[df_tile_results['dilution'].isin(DILUTION_ORDER)].copy()

    # Calculate 1/dilution for x-axis
    df_plot['inv_dilution'] = 1.0 / df_plot['dilution']

    # Create inverse dilution labels
    inv_dilution_labels = [f'1/{d}' for d in DILUTION_ORDER]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Model colors
    model_colors = {'unet': '#5FA3D9', 'attention_unet': '#8BC34A', 'attention_resunet': '#FF6B9D'}

    # Create positions for boxplot (log scale)
    base_positions = [1.0/d for d in DILUTION_ORDER]
    n_models = len(config['models'])

    # Offset for grouped boxplots
    offsets = np.linspace(-0.025, 0.025, n_models)  # Small offsets for grouping

    for model_idx, model_name in enumerate(config['models']):
        df_model = df_plot[df_plot['model'] == model_name].copy()

        # Prepare data for each dilution
        data_by_dilution = []
        for dilution in DILUTION_ORDER:
            dilution_data = df_model[df_model['dilution'] == dilution]['density'].values
            data_by_dilution.append(dilution_data)

        # Adjust positions for this model
        positions = [p * (1 + offsets[model_idx]) for p in base_positions]

        # Create boxplot
        bp = ax.boxplot(
            data_by_dilution,
            positions=positions,
            widths=[p*0.04 for p in base_positions],  # Narrower for grouped
            patch_artist=True,
            boxprops=dict(facecolor=model_colors.get(model_name, '#999999'),
                         color='black', linewidth=1),
            medianprops=dict(color='#FF8C42', linewidth=1.5),
            whiskerprops=dict(color='black', linewidth=0.8),
            capprops=dict(color='black', linewidth=0.8),
            flierprops=dict(marker='o', markerfacecolor='black', markersize=2,
                          linestyle='none', markeredgecolor='black'),
            label=model_name.replace('_', ' ').title()
        )

    # Set log scale on both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Customize
    ax.set_title('Model Comparison - Foreground Percentage vs 1/Dilution (Log Scale)',
                fontsize=18, fontweight='bold')
    ax.set_xlabel('1 / Dilution Factor', fontsize=14)
    ax.set_ylabel('Foreground Percentage', fontsize=14)

    # Set y-axis range
    ax.set_ylim(0.002, 1.5)

    # Grid
    ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)

    # Set x-axis ticks and labels
    ax.set_xticks(base_positions)
    ax.set_xticklabels(inv_dilution_labels, fontsize=12)

    # Legend
    ax.legend(fontsize=12, title='Model', title_fontsize=12, loc='lower left')

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'density_boxplot_comparison.png'
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")

# ============================================================================
# VISUALIZATION - 4-PANEL TILE COMPARISONS
# ============================================================================

def select_representative_tiles(tile_data, dilution, n_tiles=5):
    """
    Select representative tiles spanning density range for a given dilution.
    """
    # Filter tiles for this dilution
    dilution_tiles = [t for t in tile_data if t['dilution'] == dilution]

    if len(dilution_tiles) == 0:
        return []

    # Use UNet densities for selection (first model)
    densities = np.array([t['densities']['unet'] for t in dilution_tiles])

    # Select tiles spanning density range
    if len(dilution_tiles) <= n_tiles:
        return dilution_tiles

    # Select tiles at percentiles
    percentiles = np.linspace(0, 100, n_tiles)
    selected_indices = []

    for p in percentiles:
        target_density = np.percentile(densities, p)
        idx = np.argmin(np.abs(densities - target_density))
        selected_indices.append(idx)

    # Remove duplicates while preserving order
    selected_indices = sorted(set(selected_indices))

    return [dilution_tiles[i] for i in selected_indices[:n_tiles]]

def create_4panel_comparison(tile_data, output_dir, config):
    """
    Create 4-panel comparisons for representative tiles of each dilution.
    Panels: Original, UNet, Attention UNet, Attention ResUNet
    """
    print("\nGenerating 4-panel tile comparisons...")

    output_subdir = Path(output_dir) / 'representative_tiles_4panel'
    output_subdir.mkdir(exist_ok=True)

    # Get unique dilutions
    dilutions = sorted(set(t['dilution'] for t in tile_data))

    for dilution in dilutions:
        dilution_label = f"{dilution}x" if dilution > 1 else "undiluted"

        # Skip if not in our standard dilution list
        if dilution not in DILUTION_ORDER and dilution != 1:
            continue

        print(f"  Creating 4-panel for: {dilution_label}")

        # Select representative tiles
        rep_tiles = select_representative_tiles(tile_data, dilution, config['n_representative_tiles'])

        if len(rep_tiles) == 0:
            print(f"    ⚠ No tiles found for {dilution_label}")
            continue

        # Create figure (n_tiles rows × 4 columns)
        n_tiles = len(rep_tiles)
        fig, axes = plt.subplots(n_tiles, 4, figsize=(16, 4*n_tiles))

        if n_tiles == 1:
            axes = axes.reshape(1, -1)

        for row_idx, tile_info in enumerate(rep_tiles):
            tile = tile_info['tile']
            predictions = tile_info['predictions']
            densities = tile_info['densities']

            # Column 0: Original
            axes[row_idx, 0].imshow(tile)
            axes[row_idx, 0].set_title('Original', fontsize=12)
            axes[row_idx, 0].axis('off')

            # Column 1: UNet
            axes[row_idx, 1].imshow(predictions['unet'], cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 1].set_title(f'UNet\nDensity: {densities["unet"]:.3f}', fontsize=10)
            axes[row_idx, 1].axis('off')

            # Column 2: Attention UNet
            axes[row_idx, 2].imshow(predictions['attention_unet'], cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 2].set_title(f'Attention UNet\nDensity: {densities["attention_unet"]:.3f}', fontsize=10)
            axes[row_idx, 2].axis('off')

            # Column 3: Attention ResUNet
            axes[row_idx, 3].imshow(predictions['attention_resunet'], cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 3].set_title(f'Attention ResUNet\nDensity: {densities["attention_resunet"]:.3f}', fontsize=10)
            axes[row_idx, 3].axis('off')

        # Super title
        fig.suptitle(f'4-Panel Comparison - {dilution_label}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        # Save
        output_path = output_subdir / f'tiles_4panel_{dilution_label}.png'
        plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved: {output_path}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(df_tile_results, output_dir, config):
    """Save tile-level results to CSV."""
    print("\nSaving results...")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save tile-level results
    csv_path = output_dir / 'density_results_tile_level.csv'
    df_tile_results.to_csv(csv_path, index=False)
    print(f"  ✓ Saved tile-level results: {csv_path}")

    # Also save image-level summary (mean across tiles)
    image_summary = []
    for (image, model), group in df_tile_results.groupby(['image', 'model']):
        image_summary.append({
            'image': image,
            'dilution': group['dilution'].iloc[0],
            'dilution_label': group['dilution_label'].iloc[0],
            'model': model,
            'n_tiles': len(group),
            'mean_density': group['density'].mean(),
            'median_density': group['density'].median(),
            'std_density': group['density'].std(),
            'min_density': group['density'].min(),
            'max_density': group['density'].max(),
        })

    df_image_summary = pd.DataFrame(image_summary)
    summary_path = output_dir / 'density_results_image_summary.csv'
    df_image_summary.to_csv(summary_path, index=False)
    print(f"  ✓ Saved image-level summary: {summary_path}")

    # Save experiment metadata
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': config['models'],
        'model_dir': config['model_dir'],
        'test_images_dir': config['test_images_dir'],
        'n_images': len(df_tile_results['image'].unique()),
        'n_tiles_total': len(df_tile_results) // len(config['models']),
        'dilution_order': DILUTION_ORDER,
        'dilution_labels': DILUTION_LABELS,
        'config': {k: str(v) for k, v in config.items() if k not in ['models']},
    }

    metadata_path = output_dir / 'EXPERIMENT_INFO.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata: {metadata_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main analysis pipeline."""
    # Print header
    print_header(CONFIG)

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")
    print()

    # Load all models
    models = load_all_models(CONFIG)

    # Predict on test images with all models
    print("="*80)
    print("PROCESSING TEST IMAGES")
    print("="*80)
    print()

    df_tile_results, tile_data = predict_on_test_images_multimodel(
        models, CONFIG['test_images_dir'], CONFIG
    )

    # Save results
    save_results(df_tile_results, output_dir, CONFIG)

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. Individual model boxplots (stacked)
    create_multimodel_boxplot(df_tile_results, output_dir, CONFIG)

    # 2. Model comparison boxplot (side-by-side)
    create_model_comparison_boxplot(df_tile_results, output_dir, CONFIG)

    # 3. 4-panel tile comparisons
    create_4panel_comparison(tile_data, output_dir, CONFIG)

    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - density_results_tile_level.csv       (ALL tile-level densities)")
    print("  - density_results_image_summary.csv    (Image-level summaries)")
    print("  - density_boxplot_multimodel.png       (Individual model boxplots)")
    print("  - density_boxplot_comparison.png       (Side-by-side comparison)")
    print("  - representative_tiles_4panel/         (4-panel comparisons)")
    print("  - EXPERIMENT_INFO.json                 (Metadata)")
    print()
    print("✓ Multi-model density analysis completed successfully!")
    print("="*80)

if __name__ == '__main__':
    main()
