#!/usr/bin/env python3
"""
Density Prediction Using Existing Trained Models
=================================================

Loads pre-trained models from validation_arch_comparison experiment
and performs density analysis on test images with 256×256 tiles.

NO TRAINING - Just prediction and analysis (~5 minutes)

Output:
- Representative tile comparisons (5 per image)
- Boxplots (4 total)
- Comprehensive CSV

Author: Claude Code
Date: October 14, 2025
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

CONFIG = {
    # Directories to search for trained models (in priority order)
    'model_search_paths': [
        './validation_arch_comparison_20251013_093844',  # Primary
        './microscope_training_20251008_074915',         # Fallback
        './saved_models_validation_config',               # Fallback
        './',                                             # Current directory
    ],

    # Test images and output
    'test_images_dir': './test_images',
    'output_dir': f'./density_prediction_existing_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Architecture config
    'architectures': ['unet', 'resunet', 'attention_resunet'],
    'img_size': 256,
    'img_channels': 1,

    # CLAHE+OTSU
    'clahe': {'clipLimit': 2.0, 'tileGridSize': (8, 8)},

    # Visualization
    'n_representative_tiles': 5,
    'dpi': 300,
    'figsize_comparison': (16, 4),
    'figsize_boxplot': (12, 8),
}

DILUTION_PATTERNS = {
    '10x': 10, '20x': 20, '40x': 40, '80x': 80,
    '160x': 160, '320x': 320, '640x': 640, '1280x': 1280,
    '2560x': 2560, '5120x': 5120, '10240x': 10240
}

COLORS = {
    'unet': '#440154',
    'resunet': '#31688e',
    'attention_resunet': '#35b779',
    'clahe_otsu': '#fde724'
}

# ============================================================================
# GPU SETUP
# ============================================================================

def setup_gpu():
    """Configure GPU memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠ GPU warning: {e}")
    else:
        print("⚠ No GPU detected")

# ============================================================================
# MODEL LOADING
# ============================================================================

def find_best_fold_from_results(arch_dir):
    """
    Find best fold for an architecture by reading results.json files.
    Returns best fold number and its Jaccard score.
    """
    arch_path = Path(arch_dir)
    best_fold = None
    best_jaccard = -1

    for fold_dir in arch_path.glob('fold_*'):
        results_file = fold_dir / 'results.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    jaccard = results.get('best_val_jacard', -1)
                    fold_num = int(fold_dir.name.split('_')[1])

                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
                        best_fold = fold_num
            except Exception as e:
                print(f"  ⚠ Could not read {results_file}: {e}")

    return best_fold, best_jaccard


def search_for_model(architecture):
    """
    Search for trained model in multiple locations.
    Returns path to model file if found, None otherwise.
    """
    print(f"\nSearching for {architecture} model...")

    # Model filename patterns to search for
    patterns = [
        f'{architecture}_best_model.keras',
        f'{architecture}_best_model.h5',
        f'{architecture}_best_model.hdf5',
        f'best_{architecture}_model.keras',
        f'best_{architecture}_model.h5',
        f'best_{architecture}_model.hdf5',
        f'model_{architecture}*.keras',
        f'model_{architecture}*.hdf5',
    ]

    for search_path in CONFIG['model_search_paths']:
        search_path = Path(search_path)

        if not search_path.exists():
            continue

        print(f"  Searching in {search_path}...")

        # Try direct model files
        for pattern in patterns:
            matches = list(search_path.glob(pattern))
            if matches:
                print(f"  ✓ Found: {matches[0].name}")
                return matches[0]

        # Try looking in fold subdirectories
        arch_dir = search_path / architecture
        if arch_dir.exists():
            print(f"  Found architecture directory: {arch_dir}")

            # Find best fold
            best_fold, best_jaccard = find_best_fold_from_results(arch_dir)

            if best_fold is not None:
                print(f"  Best fold: {best_fold} (Jaccard: {best_jaccard:.4f})")

                # Look for model in best fold
                fold_dir = arch_dir / f'fold_{best_fold}'
                for pattern in patterns:
                    matches = list(fold_dir.glob(pattern))
                    if matches:
                        print(f"  ✓ Found: {matches[0]}")
                        return matches[0]

    print(f"  ✗ No model found for {architecture}")
    return None


def load_model_with_custom_objects(model_path):
    """Load model with custom loss and metrics."""
    print(f"  Loading model from {model_path}...")

    try:
        model = keras.models.load_model(
            str(model_path),
            custom_objects={
                'combined_dice_focal_loss': combined_dice_focal_loss,
                'jacard_coef': jacard_coef,
                'dice_coef': dice_coef,
                # Add other possible loss names
                'combined_loss': combined_dice_focal_loss,
                'dice_loss': combined_dice_focal_loss,
            },
            compile=False  # Don't need compilation for inference
        )
        print(f"  ✓ Model loaded successfully")
        return model

    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return None


def load_all_models():
    """Load all architecture models."""
    print("\n" + "="*80)
    print("LOADING TRAINED MODELS")
    print("="*80)

    models = {}

    for arch in CONFIG['architectures']:
        model_path = search_for_model(arch)

        if model_path is None:
            raise FileNotFoundError(
                f"Could not find trained model for {arch}\n"
                f"Searched in: {CONFIG['model_search_paths']}\n"
                f"Please ensure models exist in one of these directories"
            )

        model = load_model_with_custom_objects(model_path)

        if model is None:
            raise RuntimeError(f"Failed to load model for {arch}")

        models[arch] = model

    print(f"\n✓ Successfully loaded all {len(models)} models")
    return models

# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def create_output_dirs(base_dir):
    """Create output directory structure."""
    base = Path(base_dir)
    subdirs = {
        'plots': base / 'boxplots',
        'tiles': base / 'representative_tiles',
        'csv': base / 'csv_data',
    }

    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    print(f"\n✓ Created output directory: {base}")
    return subdirs

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def rescale_image_full_range(img):
    """Rescale image to full 0-255 range."""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)


def apply_clahe_otsu(img_gray, clipLimit=2.0, tileGridSize=(8, 8)):
    """Apply CLAHE + OTSU thresholding."""
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_img = clahe.apply(img_gray)
    _, binary_mask = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_mask


def extract_tiles_256(image, tile_size=256):
    """Extract 256×256 tiles."""
    h, w = image.shape[:2]
    tiles, positions = [], []

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions

# ============================================================================
# PREDICTION
# ============================================================================

def predict_on_tile_256(model, tile_256):
    """Predict on 256×256 tile."""
    tile_input = tile_256[np.newaxis, ..., np.newaxis]
    pred = model.predict(tile_input, verbose=0)
    pred_mask = pred[0, ..., 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    return pred_mask, binary_mask

# ============================================================================
# REPRESENTATIVE TILE SELECTION
# ============================================================================

def select_representative_tiles(tiles, densities, n_tiles=5):
    """Select representative tiles based on density percentiles."""
    if len(tiles) < n_tiles:
        return list(range(len(tiles)))

    sorted_indices = np.argsort(densities)
    percentiles = [0, 25, 50, 75, 100]
    selected_indices = []

    for pct in percentiles:
        idx_position = int(len(sorted_indices) * pct / 100)
        if idx_position >= len(sorted_indices):
            idx_position = len(sorted_indices) - 1
        selected_indices.append(sorted_indices[idx_position])

    return selected_indices

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_tile_comparison(original_tile, masks_dict, density_dict, image_name, tile_idx, output_dir):
    """Create 4-panel comparison."""
    fig, axes = plt.subplots(1, 4, figsize=CONFIG['figsize_comparison'])

    # Original
    axes[0].imshow(original_tile, cmap='gray')
    axes[0].set_title('Original Tile\n256×256', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Predicted masks
    arch_order = ['unet', 'resunet', 'attention_resunet']
    arch_names = ['U-Net', 'ResUNet', 'Attention ResUNet']

    for i, (arch, name) in enumerate(zip(arch_order, arch_names), 1):
        if arch in masks_dict:
            axes[i].imshow(masks_dict[arch], cmap='gray')
            density = density_dict.get(arch, 0)
            axes[i].set_title(f'{name}\nDensity: {density:.2f}%', fontsize=11, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, f'{name}\nN/A', ha='center', va='center', fontsize=12)
        axes[i].axis('off')

    plt.suptitle(f'{image_name} - Tile {tile_idx}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    filename = f'{image_name}_tile_{tile_idx:02d}_comparison.png'
    save_path = output_dir / filename
    plt.savefig(str(save_path), dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()

# ============================================================================
# BOXPLOT
# ============================================================================

def create_boxplot(df, method, output_path):
    """Create boxplot for one method."""
    df_method = df[df['method'] == method].copy()

    if len(df_method) == 0:
        print(f"  ⚠ No data for {method}")
        return

    df_method = df_method.sort_values('dilution_factor')
    dilution_factors = sorted(df_method['dilution_factor'].unique())

    fig, ax = plt.subplots(figsize=CONFIG['figsize_boxplot'])

    data_to_plot, labels, positions = [], [], []

    for i, dilution in enumerate(dilution_factors):
        df_dilution = df_method[df_method['dilution_factor'] == dilution]
        data_to_plot.append(df_dilution['foreground_pct'].values)
        labels.append(f"{int(dilution)}x")
        positions.append(i)

    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=COLORS.get(method, '#3498db'), alpha=0.7),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5)
    )

    ax.set_xlabel('1/Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Percentage (log scale)', fontsize=14, fontweight='bold')
    ax.set_title(f'Particle Density vs. Dilution Factor\n(Method: {method.replace("_", " ").title()})',
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
# DILUTION EXTRACTION
# ============================================================================

def extract_dilution_factor(filename):
    """Extract dilution factor from filename."""
    for pattern, value in DILUTION_PATTERNS.items():
        if pattern.lower() in filename.lower():
            return value

    match = re.search(r'(\d+)x', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("DENSITY PREDICTION - USING EXISTING TRAINED MODELS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    setup_gpu()
    subdirs = create_output_dirs(CONFIG['output_dir'])

    # Load existing models (NO TRAINING)
    trained_models = load_all_models()

    # Prediction
    print("\n" + "="*80)
    print("PREDICTION AND DENSITY ANALYSIS")
    print("="*80)

    test_dir = Path(CONFIG['test_images_dir'])
    test_images = sorted(test_dir.glob('*.tif'))

    if len(test_images) == 0:
        raise ValueError(f"No test images found in {test_dir}")

    print(f"Found {len(test_images)} test images\n")

    all_data = []

    for img_path in tqdm(test_images, desc="Processing test images"):
        dilution = extract_dilution_factor(img_path.stem)
        if dilution is None:
            print(f"  ⚠ Skipping {img_path.name} - no dilution factor")
            continue

        print(f"Processing: {img_path.name} (dilution: {dilution}x)")

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_rescaled = rescale_image_full_range(img)
        img_normalized = img_rescaled.astype(np.float32) / 255.0

        tiles, positions = extract_tiles_256(img_normalized, tile_size=CONFIG['img_size'])

        if len(tiles) == 0:
            continue

        print(f"  Extracted {len(tiles)} tiles (256×256)")

        tile_data = []

        for tile_idx, tile in enumerate(tiles):
            tile_info = {
                'tile_idx': tile_idx,
                'tile': tile,
                'masks': {},
                'densities': {}
            }

            # DL models
            for arch in CONFIG['architectures']:
                _, binary_mask = predict_on_tile_256(trained_models[arch], tile)
                foreground_pct = (np.count_nonzero(binary_mask) / binary_mask.size) * 100

                tile_info['masks'][arch] = binary_mask
                tile_info['densities'][arch] = foreground_pct

                all_data.append({
                    'image': img_path.stem,
                    'dilution_factor': dilution,
                    'tile_idx': tile_idx,
                    'method': arch,
                    'foreground_pct': foreground_pct
                })

            # CLAHE+OTSU
            tile_uint8 = (tile * 255).astype(np.uint8)
            binary_mask_clahe = apply_clahe_otsu(
                tile_uint8,
                clipLimit=CONFIG['clahe']['clipLimit'],
                tileGridSize=CONFIG['clahe']['tileGridSize']
            )
            foreground_pct_clahe = (np.count_nonzero(binary_mask_clahe) / binary_mask_clahe.size) * 100

            tile_info['masks']['clahe_otsu'] = binary_mask_clahe
            tile_info['densities']['clahe_otsu'] = foreground_pct_clahe

            all_data.append({
                'image': img_path.stem,
                'dilution_factor': dilution,
                'tile_idx': tile_idx,
                'method': 'clahe_otsu',
                'foreground_pct': foreground_pct_clahe
            })

            tile_data.append(tile_info)

        # Select representative tiles
        unet_densities = [t['densities']['unet'] for t in tile_data]
        representative_indices = select_representative_tiles(
            tiles, unet_densities, CONFIG['n_representative_tiles']
        )

        print(f"  Selected {len(representative_indices)} representative tiles")

        # Create visualizations
        for i, tile_idx in enumerate(representative_indices):
            tile_info = tile_data[tile_idx]

            create_tile_comparison(
                tile_info['tile'],
                {k: v for k, v in tile_info['masks'].items() if k != 'clahe_otsu'},
                {k: v for k, v in tile_info['densities'].items() if k != 'clahe_otsu'},
                img_path.stem,
                tile_idx,
                subdirs['tiles']
            )

        print(f"  ✓ Saved {len(representative_indices)} comparison images\n")

    # Generate outputs
    print("\n" + "="*80)
    print("GENERATING OUTPUTS")
    print("="*80)

    df = pd.DataFrame(all_data)
    print(f"\n✓ Collected {len(df)} density measurements")

    csv_path = subdirs['csv'] / 'density_analysis_comprehensive.csv'
    df.to_csv(str(csv_path), index=False)
    print(f"✓ Saved CSV: {csv_path}")

    methods = ['unet', 'resunet', 'attention_resunet', 'clahe_otsu']

    for method in methods:
        plot_path = subdirs['plots'] / f'{method}_density_vs_dilution.png'
        create_boxplot(df, method, plot_path)

    print("\n" + "="*80)
    print("DENSITY PREDICTION COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput directory: {CONFIG['output_dir']}")
    print(f"  - Representative tiles: {subdirs['tiles']}")
    print(f"  - Boxplots: {subdirs['plots']}")
    print(f"  - CSV: {subdirs['csv']}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
