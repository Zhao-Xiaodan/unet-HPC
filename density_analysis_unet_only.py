#!/usr/bin/env python3
"""
UNet-Only Density Analysis with Improved Visualizations
========================================================

Script: density_analysis_unet_only.py
PBS Script: pbs_density_analysis_unet_only.sh

Updates the previous density analysis (density_analysis_xukuang_20251015_142119)
with the following improvements:
1. ✅ Tile-Level Density Tracking (all 28 tiles per image saved)
2. ✅ Updated Boxplot Style (log-scale 1/dilution x-axis, matching reference)
3. ✅ Two boxplot ranges:
   - Full range: 1/10240 to 1/10
   - Low dilution range: 1/10240 to 1/80

Uses the BEST UNet model from hyperparameter search:
- Source: unet_hyperparam_20251015_224125/
- Selects model with highest validation IoU
- Training: 100 epochs, BinaryFocalLoss, 512×512 RGB

Author: Claude Code
Date: October 16, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    'model_base_dir': './unet_hyperparam_20251015_224125',
    'test_images_dir': './test_images',
    'output_dir': f'./density_analysis_unet_only_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Image settings (RGB from training)
    'img_size': 512,  # Tile size
    'img_channels': 3,  # RGB

    # Prediction settings
    'batch_size': 8,
    'threshold': 0.5,  # Binary threshold for predictions

    # Representative tiles (reuse from previous analysis)
    'n_representative_tiles': 5,

    # Plotting
    'dpi': 300,
    'figsize_boxplot_full': (14, 8),
    'figsize_boxplot_low': (14, 8),
}

# Dilution factors in proper order (10x - 10240x)
DILUTION_ORDER = [10, 20, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
DILUTION_LABELS = ['10x', '20x', '80x', '160x', '320x', '640x', '1280x', '2560x', '5120x', '10240x']

# Low dilution range (1/10240 to 1/80)
DILUTION_ORDER_LOW = [80, 160, 320, 640, 1280, 2560, 5120, 10240]
DILUTION_LABELS_LOW = ['80x', '160x', '320x', '640x', '1280x', '2560x', '5120x', '10240x']

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

def print_header(config, best_model_info):
    """Print analysis header."""
    print("="*80)
    print("UNET-ONLY DENSITY ANALYSIS - BEST MODEL FROM HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Script: density_analysis_unet_only.py")
    print(f"PBS Script: pbs_density_analysis_unet_only.sh")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model base directory: {config['model_base_dir']}")
    print(f"Test images: {config['test_images_dir']}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Image format: {config['img_size']}×{config['img_size']} RGB ({config['img_channels']} channels)")
    print()
    print("="*80)
    print("BEST MODEL SELECTION")
    print("="*80)
    print(f"Model: {best_model_info['model_name']}")
    print(f"Hyperparameters:")
    print(f"  - n_filters: {best_model_info['n_filters']}")
    print(f"  - dropout: {best_model_info['dropout']}")
    print(f"  - batch_norm: {best_model_info['batch_norm']}")
    print(f"  - learning_rate: {best_model_info['learning_rate']}")
    print(f"Best Val IoU: {best_model_info['best_val_iou']:.4f}")
    print(f"Model path: {best_model_info['model_path']}")
    print("="*80)
    print()

# ============================================================================
# MODEL SELECTION
# ============================================================================

def find_best_unet_model(base_dir):
    """
    Find the best UNet model from hyperparameter search.
    Looks through all checkpoints and finds the one with highest validation IoU.
    """
    print("Searching for best UNet model...")
    base_dir = Path(base_dir)

    # Find all best_model.keras files in checkpoints
    checkpoint_dirs = list((base_dir / 'checkpoints').glob('unet_*'))

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No UNet checkpoint directories found in {base_dir / 'checkpoints'}")

    print(f"Found {len(checkpoint_dirs)} UNet model configurations")

    best_model_info = None
    best_iou = -1.0

    # Search through each checkpoint directory
    for ckpt_dir in checkpoint_dirs:
        model_file = ckpt_dir / 'best_model.keras'

        if not model_file.exists():
            continue

        # Parse hyperparameters from directory name
        # Format: unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001
        dir_name = ckpt_dir.name

        try:
            # Extract hyperparameters
            n_filters = int(re.search(r'n_filters(\d+)', dir_name).group(1))
            dropout_str = re.search(r'dropout(\d+p\d+)', dir_name).group(1)
            dropout = float(dropout_str.replace('p', '.'))
            batch_norm = 'batch_normTrue' in dir_name
            lr_str = re.search(r'learning_rate(\d+p\d+)', dir_name).group(1)
            learning_rate = float(lr_str.replace('p', '.'))

            # Find corresponding history CSV to get IoU
            history_files = list((base_dir / 'logs').glob(f'unet_n_filters{n_filters}_dropout{dropout_str}_*_history.csv'))

            if history_files:
                history_df = pd.read_csv(history_files[0])
                if 'val_jacard_coef' in history_df.columns:
                    max_iou = history_df['val_jacard_coef'].max()

                    if max_iou > best_iou:
                        best_iou = max_iou
                        best_model_info = {
                            'model_path': model_file,
                            'model_name': dir_name,
                            'n_filters': n_filters,
                            'dropout': dropout,
                            'batch_norm': batch_norm,
                            'learning_rate': learning_rate,
                            'best_val_iou': max_iou
                        }
                        print(f"  New best: {dir_name} (IoU: {max_iou:.4f})")
        except (AttributeError, IndexError, ValueError) as e:
            print(f"  Warning: Could not parse {dir_name}: {e}")
            continue

    if best_model_info is None:
        raise FileNotFoundError("Could not find any valid UNet models with validation IoU data")

    print()
    print(f"✓ Selected best model: {best_model_info['model_name']}")
    print(f"  Best Val IoU: {best_model_info['best_val_iou']:.4f}")
    print()

    return best_model_info

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

def load_model(model_path):
    """Load the trained UNet model."""
    print(f"Loading model from: {model_path}")

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

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    # Load model with custom objects
    custom_objects = {
        'combined_dice_focal_loss': combined_dice_focal_loss,
        'jacard_coef': jacard_coef,
        'dice_coef': dice_coef,
        'focal_loss': focal_loss,
        'BinaryFocalLoss': BinaryFocalLoss,
    }

    model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
    print(f"  ✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    return model

# ============================================================================
# PREDICTION
# ============================================================================

def predict_on_test_images(model, test_images_dir, config):
    """
    Predict on all test images and save tile-level densities.

    Returns:
        df_tile_results: DataFrame with tile-level densities
        tile_data: List of dicts with tile data for visualization
    """
    print("\n" + "="*80)
    print("PREDICTION ON TEST IMAGES")
    print("="*80)

    test_images_dir = Path(test_images_dir)
    image_files = sorted(list(test_images_dir.glob('*.tif')) + list(test_images_dir.glob('*.tiff')))

    print(f"Found {len(image_files)} test images")
    print()

    tile_results = []
    tile_data = []

    for img_file in tqdm(image_files, desc="Processing images"):
        print(f"\nProcessing: {img_file.name}")

        # Extract dilution from filename
        dilution = extract_dilution_from_filename(img_file.name)
        dilution_label = f"{dilution}x"
        print(f"  Dilution: {dilution}x")

        # Load image
        image = load_test_image(img_file)
        print(f"  Image shape: {image.shape}")

        # Extract tiles
        tiles_with_pos = extract_tiles(image, config['img_size'])
        print(f"  Extracted {len(tiles_with_pos)} tiles")

        # Predict on tiles
        tiles = np.array([preprocess_tile(tile) for tile, pos in tiles_with_pos])
        predictions = model.predict(tiles, batch_size=config['batch_size'], verbose=0)

        # Calculate densities for each tile
        for tile_idx, ((tile, pos), prediction) in enumerate(zip(tiles_with_pos, predictions)):
            density = calculate_foreground_density(prediction, config['threshold'])

            # Store tile-level result
            tile_results.append({
                'image': img_file.name,
                'dilution': dilution,
                'dilution_label': dilution_label,
                'tile_idx': tile_idx,
                'position_y': pos[0],
                'position_x': pos[1],
                'density': density
            })

            # Store tile data for visualization (reuse previous visualizations)
            tile_data.append({
                'image': img_file.name,
                'dilution': dilution,
                'dilution_label': dilution_label,
                'tile_idx': tile_idx,
                'position': pos,
                'tile': tile,
                'prediction': prediction,
                'density': density,
            })

        # Print summary for this image
        tile_densities = [r['density'] for r in tile_results if r['image'] == img_file.name]
        mean_density = np.mean(tile_densities)
        std_density = np.std(tile_densities)
        print(f"  Mean density: {mean_density:.4f} ± {std_density:.4f} (across {len(tiles_with_pos)} tiles)")

    df_tile_results = pd.DataFrame(tile_results)

    return df_tile_results, tile_data

# ============================================================================
# VISUALIZATION - BOX PLOTS WITH TWO DILUTION RANGES
# ============================================================================

def create_boxplot_full_range(df_tile_results, output_dir, config):
    """
    Create box plot for FULL dilution range (1/10240 to 1/10).
    Uses 1/Dilution on x-axis with log scale (matching reference style).
    """
    print("\nGenerating box plot (Full Range: 1/10240 to 1/10)...")

    # Filter to only include dilutions in our defined order
    df_plot = df_tile_results[df_tile_results['dilution'].isin(DILUTION_ORDER)].copy()

    # Calculate 1/dilution for x-axis
    df_plot['inv_dilution'] = 1.0 / df_plot['dilution']

    # Create inverse dilution labels
    inv_dilution_labels = [f'1/{d}' for d in DILUTION_ORDER]

    # Create figure
    fig, ax = plt.subplots(figsize=config['figsize_boxplot_full'])

    # Colors: blue boxes with orange median like reference
    box_color = '#5FA3D9'  # Light blue
    median_color = '#FF8C42'  # Orange

    # Create positions for boxplot (log scale)
    positions = [1.0/d for d in DILUTION_ORDER]

    # Prepare data for each dilution
    data_by_dilution = []
    for dilution in DILUTION_ORDER:
        dilution_data = df_plot[df_plot['dilution'] == dilution]['density'].values
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
    ax.set_title('UNet - Foreground Percentage vs 1/Dilution (Full Range: 1/10 - 1/10240)',
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
    output_path = Path(output_dir) / 'density_boxplot_full_range.png'
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")

def create_boxplot_low_dilution_range(df_tile_results, output_dir, config):
    """
    Create box plot for LOW dilution range (1/10240 to 1/80).
    Uses 1/Dilution on x-axis with log scale (matching reference style).
    """
    print("\nGenerating box plot (Low Dilution Range: 1/10240 to 1/80)...")

    # Filter to only include low dilutions
    df_plot = df_tile_results[df_tile_results['dilution'].isin(DILUTION_ORDER_LOW)].copy()

    # Calculate 1/dilution for x-axis
    df_plot['inv_dilution'] = 1.0 / df_plot['dilution']

    # Create inverse dilution labels
    inv_dilution_labels_low = [f'1/{d}' for d in DILUTION_ORDER_LOW]

    # Create figure
    fig, ax = plt.subplots(figsize=config['figsize_boxplot_low'])

    # Colors: blue boxes with orange median like reference
    box_color = '#5FA3D9'  # Light blue
    median_color = '#FF8C42'  # Orange

    # Create positions for boxplot (log scale)
    positions = [1.0/d for d in DILUTION_ORDER_LOW]

    # Prepare data for each dilution
    data_by_dilution = []
    for dilution in DILUTION_ORDER_LOW:
        dilution_data = df_plot[df_plot['dilution'] == dilution]['density'].values
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
    ax.set_title('UNet - Foreground Percentage vs 1/Dilution (Low Dilution: 1/80 - 1/10240)',
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
    ax.set_xticklabels(inv_dilution_labels_low, fontsize=12)

    # Rotate x-axis labels if needed
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'density_boxplot_low_dilution_range.png'
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(df_tile_results, output_dir, best_model_info, config):
    """Save analysis results to CSV and JSON."""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tile-level results
    tile_csv_path = output_dir / 'density_results_tile_level.csv'
    df_tile_results.to_csv(tile_csv_path, index=False)
    print(f"✓ Tile-level results saved: {tile_csv_path}")
    print(f"  Total tiles: {len(df_tile_results)}")

    # Create image-level summary
    summary_data = []
    for image in df_tile_results['image'].unique():
        img_data = df_tile_results[df_tile_results['image'] == image]
        dilution = img_data['dilution'].iloc[0]
        dilution_label = img_data['dilution_label'].iloc[0]

        summary_data.append({
            'image': image,
            'dilution': dilution,
            'dilution_label': dilution_label,
            'n_tiles': len(img_data),
            'mean_density': img_data['density'].mean(),
            'median_density': img_data['density'].median(),
            'std_density': img_data['density'].std(),
            'min_density': img_data['density'].min(),
            'max_density': img_data['density'].max(),
        })

    df_summary = pd.DataFrame(summary_data)
    summary_csv_path = output_dir / 'density_results_image_summary.csv'
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"✓ Image-level summary saved: {summary_csv_path}")
    print(f"  Total images: {len(df_summary)}")

    # Save experiment info
    experiment_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'density_analysis_unet_only.py',
        'model_info': best_model_info,
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'total_images': len(df_summary),
        'total_tiles': len(df_tile_results),
        'dilutions': sorted(df_tile_results['dilution'].unique().tolist()),
    }

    # Convert Path objects to strings for JSON serialization
    experiment_info['model_info']['model_path'] = str(experiment_info['model_info']['model_path'])

    info_json_path = output_dir / 'EXPERIMENT_INFO.json'
    with open(info_json_path, 'w') as f:
        json.dump(experiment_info, f, indent=2)
    print(f"✓ Experiment info saved: {info_json_path}")
    print()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main analysis pipeline."""

    # Find best model
    best_model_info = find_best_unet_model(CONFIG['model_base_dir'])

    # Print header
    print_header(CONFIG, best_model_info)

    # Load model
    model = load_model(best_model_info['model_path'])

    # Predict on test images
    df_tile_results, tile_data = predict_on_test_images(model, CONFIG['test_images_dir'], CONFIG)

    # Save results
    save_results(df_tile_results, CONFIG['output_dir'], best_model_info, CONFIG)

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Two boxplots with different dilution ranges
    create_boxplot_full_range(df_tile_results, CONFIG['output_dir'], CONFIG)
    create_boxplot_low_dilution_range(df_tile_results, CONFIG['output_dir'], CONFIG)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {CONFIG['output_dir']}")
    print("\nGenerated files:")
    print("  - density_results_tile_level.csv")
    print("  - density_results_image_summary.csv")
    print("  - density_boxplot_full_range.png (1/10 - 1/10240)")
    print("  - density_boxplot_low_dilution_range.png (1/80 - 1/10240)")
    print("  - EXPERIMENT_INFO.json")
    print("\nNote: Tile visualizations can be reused from previous analysis")
    print("      (density_analysis_xukuang_20251015_142119/representative_tiles/)")
    print("="*80)

if __name__ == '__main__':
    main()
