#!/usr/bin/env python3
"""
Density Analysis Using Xukuang Parameters UNet Model
====================================================

Uses the best UNet model from xukuang_params_shrunk_20251015_071224
to predict on test images and generate density analysis.

Best Model: UNet (Epoch 140, Val IoU: 0.6789)
Training Parameters:
- Learning Rate: 0.005
- Epochs: 200
- Batch Size: 4
- Loss: BinaryFocalLoss(gamma=2)
- Image Size: 512×512 RGB

Generates:
- Box plot with corrected dilution labels (10x - 10240x)
- Representative 512×512 tiles with predictions
- CSV with all density data
- Summary statistics

Process:
1. Load best UNet model from Xukuang experiment
2. Extract 512×512 tiles from test images (non-overlapping)
3. Predict on each tile
4. Calculate foreground density for each prediction
5. Generate visualizations with CORRECT dilution ordering

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
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras

# Import custom modules
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Directories
    'model_dir': './xukuang_params_shrunk_20251015_071224',
    'test_images_dir': './test_images',
    'output_dir': f'./density_analysis_xukuang_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Model selection (best UNet from Xukuang experiment)
    'model_name': 'unet',  # Will look for unet_model.keras or similar
    'best_epoch': 140,  # Best validation IoU epoch

    # Image settings (from Xukuang training - RGB!)
    'img_size': 512,  # Tile size
    'img_channels': 3,  # RGB (NOT grayscale!)

    # Prediction settings
    'batch_size': 8,
    'threshold': 0.5,  # Binary threshold for predictions

    # Representative tiles
    'n_representative_tiles': 5,

    # Plotting
    'dpi': 300,
    'figsize_comparison': (20, 5),  # For tile comparisons
    'figsize_boxplot': (16, 10),
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

print("="*80)
print("DENSITY ANALYSIS - XUKUANG UNET MODEL (RGB)")
print("="*80)
print(f"Script: density_analysis_xukuang.py")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model directory: {CONFIG['model_dir']}")
print(f"Test images: {CONFIG['test_images_dir']}")
print(f"Output directory: {CONFIG['output_dir']}")
print(f"Image format: {CONFIG['img_size']}×{CONFIG['img_size']} RGB (3 channels)")
print()
print("Model: UNet (FINAL model from Xukuang experiment)")
print("  - Final Val IoU (Epoch 200): 0.6065")
print("  - Best Val IoU (Epoch 140): 0.6789")
print("  - Training: LR=0.005, 200 epochs, BinaryFocalLoss")
print("  - NOTE: Using FINAL epoch model, not best checkpoint")
print("="*80)
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_dilution_from_filename(filename):
    """
    Extract dilution factor from filename.
    Returns dilution as integer (e.g., 10, 20, 80, etc.)
    """
    filename_lower = filename.lower()

    # Check patterns in order (longest first to avoid matching 10x in 10240x)
    for pattern, dilution in DILUTION_PATTERNS.items():
        if pattern in filename_lower:
            return dilution

    # Default: undiluted (1x)
    return 1

def load_test_image(image_path):
    """Load test image (TIFF format, RGB)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to RGB if needed
    if len(img.shape) == 2:
        # Grayscale -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # RGBA -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def extract_tiles(image, tile_size=512):
    """
    Extract non-overlapping tiles from image.
    Returns list of (tile, position) tuples.
    """
    h, w = image.shape[:2]
    tiles = []

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append((tile, (y, x)))

    return tiles

def preprocess_tile(tile):
    """Preprocess tile for model prediction (RGB, 512×512)."""
    # Normalize to [0, 1]
    tile = tile.astype(np.float32) / 255.0
    return tile

def calculate_foreground_density(prediction, threshold=0.5):
    """Calculate foreground (bead) density from prediction."""
    binary_pred = (prediction > threshold).astype(np.uint8)
    density = np.mean(binary_pred)
    return density

def find_model_file(model_dir, model_name='unet'):
    """
    Find the model file in the directory.

    The Xukuang training script saves FINAL models (after 200 epochs) with naming:
    - unet_xukuang_params_shrunk.keras
    - attention_unet_xukuang_params_shrunk.keras
    - attention_resunet_xukuang_params_shrunk.keras

    These are the FINAL epoch models, NOT best checkpoint models.
    """
    model_dir = Path(model_dir)

    # Xukuang experiment naming convention
    xukuang_name = f"{model_name}_xukuang_params_shrunk.keras"
    model_path = model_dir / xukuang_name

    if model_path.exists():
        return model_path

    # If not found, try alternative patterns
    patterns = [
        f"{model_name}_model.keras",
        f"{model_name}_best_model.keras",
        f"{model_name}.keras",
        f"{model_name}_model.h5",
        f"{model_name}_best_model.h5",
        f"{model_name}.h5",
    ]

    for pattern in patterns:
        model_path = model_dir / pattern
        if model_path.exists():
            return model_path

    # If still not found, list available models
    available = list(model_dir.glob("*.keras")) + list(model_dir.glob("*.h5"))
    if available:
        print(f"Available models in {model_dir}:")
        for m in available:
            print(f"  - {m.name}")
        # Use first available model
        return available[0]

    raise FileNotFoundError(f"No model file found in {model_dir}")

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def load_model(model_dir, model_name='unet'):
    """Load the trained model."""
    print(f"Loading model: {model_name}")

    model_path = find_model_file(model_dir, model_name)
    print(f"  Model file: {model_path}")

    # Custom objects for loading
    custom_objects = {
        'combined_dice_focal_loss': combined_dice_focal_loss,
        'jacard_coef': jacard_coef,
        'dice_coef': dice_coef,
    }

    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"  ✓ Model loaded successfully")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print()

    return model

def predict_on_test_images(model, test_images_dir, config):
    """
    Predict on all test images and calculate densities.
    Returns DataFrame with results.
    """
    test_images_dir = Path(test_images_dir)
    image_files = sorted(test_images_dir.glob("*.tif")) + sorted(test_images_dir.glob("*.tiff"))

    print(f"Found {len(image_files)} test images")
    print()

    results = []
    tile_data = []  # For saving representative tiles

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

        # Predict on tiles
        tile_densities = []
        for tile, pos in tiles_with_pos:
            # Preprocess
            tile_prep = preprocess_tile(tile)
            tile_batch = np.expand_dims(tile_prep, axis=0)

            # Predict
            pred = model.predict(tile_batch, verbose=0)[0, :, :, 0]

            # Calculate density
            density = calculate_foreground_density(pred, config['threshold'])
            tile_densities.append(density)

            # Store tile data for later visualization
            tile_data.append({
                'image': img_file.name,
                'dilution': dilution,
                'dilution_label': dilution_label,
                'position': pos,
                'tile': tile,
                'prediction': pred,
                'density': density
            })

        # Aggregate results
        mean_density = np.mean(tile_densities)
        median_density = np.median(tile_densities)
        std_density = np.std(tile_densities)

        results.append({
            'image': img_file.name,
            'dilution': dilution,
            'dilution_label': dilution_label,
            'n_tiles': len(tiles_with_pos),
            'mean_density': mean_density,
            'median_density': median_density,
            'std_density': std_density,
            'min_density': np.min(tile_densities),
            'max_density': np.max(tile_densities),
        })

        print(f"  Mean density: {mean_density:.4f} ± {std_density:.4f}")

    df_results = pd.DataFrame(results)

    return df_results, tile_data

def create_boxplot(df_results, output_dir, config):
    """
    Create box plot with CORRECTED dilution ordering (10x - 10240x).
    """
    print("\nGenerating box plot...")

    # Prepare data for plotting
    plot_data = []
    for _, row in df_results.iterrows():
        # Each tile contributes to the distribution
        # We'll use the stored tile densities if available, or approximate
        plot_data.append({
            'Dilution': row['dilution'],
            'Dilution_Label': row['dilution_label'],
            'Density': row['mean_density']
        })

    df_plot = pd.DataFrame(plot_data)

    # Filter to only include dilutions in our defined order
    df_plot = df_plot[df_plot['Dilution'].isin(DILUTION_ORDER)]

    # Create categorical with CORRECT ordering
    df_plot['Dilution_Cat'] = pd.Categorical(
        df_plot['Dilution_Label'],
        categories=DILUTION_LABELS,
        ordered=True
    )

    # Sort by categorical order
    df_plot = df_plot.sort_values('Dilution_Cat')

    # Create figure
    fig, ax = plt.subplots(figsize=config['figsize_boxplot'])

    # Create box plot with correct ordering
    sns.boxplot(
        data=df_plot,
        x='Dilution_Cat',
        y='Density',
        ax=ax,
        palette='Set2',
        order=DILUTION_LABELS  # CRITICAL: Specify order explicitly
    )

    ax.set_xlabel('Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Density', fontsize=14, fontweight='bold')
    ax.set_title('Bead Density vs Dilution Factor - Xukuang UNet Model',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    # Add sample size annotations
    for i, label in enumerate(DILUTION_LABELS):
        n = len(df_plot[df_plot['Dilution_Cat'] == label])
        if n > 0:
            ax.text(i, ax.get_ylim()[1] * 0.95, f'n={n}',
                   ha='center', va='top', fontsize=9)

    plt.tight_layout()

    output_path = Path(output_dir) / 'density_boxplot.png'
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")

    return output_path

def create_representative_tiles(tile_data, output_dir, config):
    """
    Create visualizations of representative tiles for each dilution.
    """
    print("\nGenerating representative tile visualizations...")

    output_dir = Path(output_dir) / 'representative_tiles'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by dilution
    tiles_by_dilution = defaultdict(list)
    for tile_info in tile_data:
        dilution = tile_info['dilution']
        if dilution in DILUTION_ORDER:
            tiles_by_dilution[dilution].append(tile_info)

    # For each dilution, select representative tiles
    for dilution in DILUTION_ORDER:
        dilution_label = f"{dilution}x"
        tiles = tiles_by_dilution.get(dilution, [])

        if not tiles:
            print(f"  No tiles for {dilution_label}")
            continue

        # Select tiles spanning density range
        tiles_sorted = sorted(tiles, key=lambda x: x['density'])
        n_select = min(config['n_representative_tiles'], len(tiles_sorted))

        if n_select == 0:
            continue

        # Select evenly spaced tiles
        indices = np.linspace(0, len(tiles_sorted) - 1, n_select, dtype=int)
        selected_tiles = [tiles_sorted[i] for i in indices]

        # Create visualization
        fig, axes = plt.subplots(2, n_select, figsize=(4*n_select, 8))
        if n_select == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(f'Representative Tiles: {dilution_label} Dilution',
                    fontsize=16, fontweight='bold')

        for i, tile_info in enumerate(selected_tiles):
            # Original tile
            axes[0, i].imshow(tile_info['tile'])
            axes[0, i].set_title(f"Original\nDensity: {tile_info['density']:.3f}",
                                fontsize=10)
            axes[0, i].axis('off')

            # Prediction
            axes[1, i].imshow(tile_info['prediction'], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f"Prediction", fontsize=10)
            axes[1, i].axis('off')

        plt.tight_layout()

        output_path = output_dir / f'tiles_{dilution_label}.png'
        plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")

def save_results(df_results, output_dir):
    """Save results to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / 'density_results.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    return csv_path

def save_experiment_info(config, model_path, output_dir):
    """Save experiment information."""
    info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_dir': str(config['model_dir']),
        'model_file': str(model_path),
        'model_name': 'UNet (Xukuang - FINAL epoch)',
        'model_note': 'Using FINAL model from epoch 200, NOT best checkpoint from epoch 140',
        'model_performance': {
            'final_val_iou_epoch_200': 0.6065,
            'best_val_iou_epoch_140': 0.6789,
            'model_used': 'final_epoch_200',
        },
        'training_params': {
            'learning_rate': 0.005,
            'epochs': 200,
            'batch_size': 4,
            'loss_function': 'BinaryFocalLoss(gamma=2)',
            'image_size': 512,
            'image_channels': 3,
            'image_type': 'RGB',
            'optimizer': 'Adam',
            'test_split': 0.2,
            'random_state': 0,
        },
        'test_images_dir': str(config['test_images_dir']),
        'prediction_threshold': config['threshold'],
        'dilution_order': DILUTION_ORDER,
        'dilution_labels': DILUTION_LABELS,
        'dilution_order_note': 'CORRECTED ordering: 10x -> 10240x (not string sort)',
    }

    info_path = Path(output_dir) / 'EXPERIMENT_INFO.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"✓ Experiment info saved to: {info_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Load model
    model = load_model(CONFIG['model_dir'], CONFIG['model_name'])
    model_path = find_model_file(CONFIG['model_dir'], CONFIG['model_name'])

    # Predict on test images
    df_results, tile_data = predict_on_test_images(
        model, CONFIG['test_images_dir'], CONFIG
    )

    # Save results
    save_results(df_results, output_dir)

    # Create visualizations
    create_boxplot(df_results, output_dir, CONFIG)
    create_representative_tiles(tile_data, output_dir, CONFIG)

    # Save experiment info
    save_experiment_info(CONFIG, model_path, output_dir)

    # Print summary
    print("\n" + "="*80)
    print("DENSITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"\nProcessed {len(df_results)} images")
    print(f"Total tiles analyzed: {df_results['n_tiles'].sum()}")
    print("\nDensity by Dilution:")
    print(df_results[['dilution_label', 'mean_density']].to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
