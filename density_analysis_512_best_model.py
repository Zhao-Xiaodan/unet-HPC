#!/usr/bin/env python3
"""
Density Analysis Using Best 512×512 Model
==========================================

Loads the best model from hyperparameter search and performs density analysis
on test images with dilution series.

Features:
- Uses best configuration from hyperparameter_search_512 results
- Trains final model on full dataset (no CV split)
- Predicts on ./test_images/ with 512×512 tiles
- Generates boxplots and representative tile comparisons
- Includes CLAHE+OTSU baseline

Author: Claude Code
Date: October 14, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import re
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras

# Import custom modules
from model_architectures import get_model
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Directories
    'hyperparam_search_dir': None,  # Will be set from command line
    'test_images_dir': './test_images',
    'output_dir': f'./density_analysis_512_best_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Will be loaded from best config
    'architecture': None,
    'lr': None,
    'dropout': None,
    'batch_size': None,
    'filters': 32,  # Same as hyperparameter search

    # Fixed settings
    'img_size': 512,
    'img_channels': 3,
    'epochs': 40,  # More epochs for final model
    'early_stopping_patience': 10,

    # CLAHE+OTSU
    'clahe': {'clipLimit': 2.0, 'tileGridSize': (8, 8)},

    # Visualization
    'n_representative_tiles': 5,
    'dpi': 300,
}

# Colors
COLORS = {
    'unet': '#440154',
    'resunet': '#31688e',
    'attention_resunet': '#35b779',
    'clahe_otsu': '#fde724'
}

# Dilution patterns
DILUTION_PATTERNS = {
    '10x': 10, '20x': 20, '40x': 40, '80x': 80,
    '160x': 160, '320x': 320, '640x': 640, '1280x': 1280,
    '2560x': 2560, '5120x': 5120, '10240x': 10240
}

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def setup_gpu():
    """Configure GPU memory growth and mixed precision."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠ GPU configuration warning: {e}")

    # Mixed precision
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print(f"✓ Mixed precision enabled: {policy.name}")

def clear_session():
    """Clear Keras session and garbage collect."""
    keras.backend.clear_session()
    gc.collect()

# ============================================================================
# LOAD BEST CONFIGURATION
# ============================================================================

def load_best_config(search_dir):
    """Load best configuration from hyperparameter search results."""
    search_path = Path(search_dir)

    summary_file = search_path / 'summary.json'
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    best_config_name = summary['best_config']
    best_jaccard = summary['best_jaccard']

    # Parse config name: e.g., "resunet_lr0.0001_drop0.2_bs4"
    parts = best_config_name.split('_')
    architecture = parts[0]
    lr = float(parts[1].replace('lr', ''))
    dropout = float(parts[2].replace('drop', ''))
    batch_size = int(parts[3].replace('bs', ''))

    print(f"\n{'='*80}")
    print(f"LOADED BEST CONFIGURATION FROM HYPERPARAMETER SEARCH")
    print(f"{'='*80}")
    print(f"Search directory: {search_dir}")
    print(f"Best configuration: {best_config_name}")
    print(f"Best Jaccard (CV): {best_jaccard:.4f}")
    print(f"")
    print(f"Parameters:")
    print(f"  Architecture: {architecture}")
    print(f"  Learning Rate: {lr}")
    print(f"  Dropout: {dropout}")
    print(f"  Batch Size: {batch_size}")
    print(f"{'='*80}\n")

    return {
        'architecture': architecture,
        'lr': lr,
        'dropout': dropout,
        'batch_size': batch_size,
        'config_name': best_config_name,
        'cv_jaccard': best_jaccard
    }

# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(dataset_dir, img_size=512):
    """Load 512×512 training data."""
    from PIL import Image

    images_dir = Path(dataset_dir) / 'images'
    masks_dir = Path(dataset_dir) / 'masks'

    image_files = sorted(images_dir.glob('*.png'))

    print(f"Loading {len(image_files)} training images (512×512)...")

    images = []
    masks = []

    for img_path in tqdm(image_files, desc="Loading training data"):
        # Load image
        img = Image.open(img_path).convert('RGB')
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size))
        img_array = np.array(img) / 255.0

        # Load mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            continue

        mask = Image.open(mask_path).convert('L')
        if mask.size != (img_size, img_size):
            mask = mask.resize((img_size, img_size))
        mask_array = np.array(mask) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)

        images.append(img_array)
        masks.append(mask_array)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)[..., np.newaxis]

    print(f"✓ Loaded {len(images)} image-mask pairs")
    return images, masks

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_final_model(X, y, config, output_dir):
    """Train final model on full dataset (no CV split)."""
    print(f"\n{'='*80}")
    print(f"TRAINING FINAL MODEL: {config['architecture']}")
    print(f"{'='*80}")

    clear_session()

    # Use 90/10 split for validation monitoring
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Build model
    input_shape = (CONFIG['img_size'], CONFIG['img_size'], CONFIG['img_channels'])
    model = get_model(
        model_name=config['architecture'],
        input_shape=input_shape,
        NUM_CLASSES=1,
        dropout_rate=config['dropout'],
        batch_norm=True
    )

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=config['lr'])
    optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=combined_dice_focal_loss,
        metrics=[jacard_coef, dice_coef]
    )

    # Model checkpoint
    model_path = output_dir / 'best_model.keras'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor='val_jacard_coef',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_jacard_coef',
            patience=CONFIG['early_stopping_patience'],
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_jacard_coef',
            factor=0.5,
            patience=5,
            mode='max',
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    best_jaccard = max(history.history['val_jacard_coef'])
    print(f"\n✓ Training complete")
    print(f"  Best Jaccard: {best_jaccard:.4f}")
    print(f"  Model saved: {model_path}")

    return model, best_jaccard

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def rescale_image_full_range(img):
    """Rescale image to 0-255."""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)

def apply_clahe_otsu(img_gray, clipLimit=2.0, tileGridSize=(8, 8)):
    """Apply CLAHE + OTSU."""
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_img = clahe.apply(img_gray)
    _, binary_mask = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_mask

def extract_tiles_512(image, tile_size=512):
    """Extract 512×512 tiles."""
    h, w = image.shape[:2]
    tiles, positions = [], []

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions

def extract_dilution_factor(filename):
    """Extract dilution from filename."""
    match = re.search(r'(?:^|_)(\d+)x(?:_|\.|-)', filename.lower())
    if match:
        return int(match.group(1))
    return None

# ============================================================================
# PREDICTION
# ============================================================================

def predict_on_tile(model, tile_rgb):
    """Predict on 512×512 RGB tile."""
    tile_input = tile_rgb[np.newaxis, ...]
    pred = model.predict(tile_input, verbose=0)
    pred_mask = pred[0, ..., 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    return pred_mask, binary_mask

def select_representative_tiles(tiles, densities, n_tiles=5):
    """Select tiles at density percentiles."""
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

def create_tile_comparison(original_tile, pred_mask, density, image_name, tile_idx, arch_name, output_dir):
    """Create 2-panel comparison: original + prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_tile)
    axes[0].set_title('Original Tile\n512×512', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title(f'{arch_name}\nDensity: {density:.2f}%', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle(f'{image_name} - Tile {tile_idx}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    filename = f'{image_name}_tile_{tile_idx:02d}_comparison.png'
    save_path = output_dir / filename
    plt.savefig(str(save_path), dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()

def create_boxplot(df, method, output_path, arch_name):
    """Create density boxplot."""
    df_method = df[df['method'] == method].copy()

    if len(df_method) == 0:
        return

    df_method = df_method.sort_values('dilution_factor')
    dilution_factors = sorted(df_method['dilution_factor'].unique())

    fig, ax = plt.subplots(figsize=(12, 8))

    data_to_plot = []
    labels = []
    positions = []

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
    ax.set_title(f'Particle Density vs. Dilution Factor\n(Best Model: {arch_name})',
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
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python density_analysis_512_best_model.py <hyperparameter_search_dir>")
        print("Example: python density_analysis_512_best_model.py ./hyperparameter_search_512_20251014_123456")
        sys.exit(1)

    CONFIG['hyperparam_search_dir'] = sys.argv[1]

    print(f"\n{'='*80}")
    print(f"DENSITY ANALYSIS WITH BEST 512×512 MODEL")
    print(f"{'='*80}\n")

    setup_gpu()

    # Load best config
    best_config = load_best_config(CONFIG['hyperparam_search_dir'])
    CONFIG.update(best_config)

    # Create output dirs
    output_dir = Path(CONFIG['output_dir'])
    subdirs = {
        'models': output_dir / 'trained_model',
        'tiles': output_dir / 'representative_tiles',
        'plots': output_dir / 'boxplots',
        'csv': output_dir / 'csv_data'
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    # Train final model
    print(f"\n{'='*80}")
    print("PHASE 1: TRAIN FINAL MODEL ON FULL DATASET")
    print(f"{'='*80}")

    X, y = load_training_data('./dataset_shrunk_masks', CONFIG['img_size'])
    model, final_jaccard = train_final_model(X, y, CONFIG, subdirs['models'])

    # Predict on test images
    print(f"\n{'='*80}")
    print("PHASE 2: DENSITY ANALYSIS ON TEST IMAGES")
    print(f"{'='*80}")

    test_dir = Path(CONFIG['test_images_dir'])
    test_images = sorted(test_dir.glob('*.tif'))

    print(f"Found {len(test_images)} test images")

    all_data = []

    for img_path in tqdm(test_images, desc="Processing test images"):
        dilution = extract_dilution_factor(img_path.stem)
        if dilution is None:
            continue

        print(f"\nProcessing: {img_path.name} (dilution: {dilution}x)")

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rescaled = rescale_image_full_range(img_rgb)
        img_normalized = img_rescaled.astype(np.float32) / 255.0

        # Extract tiles
        tiles, positions = extract_tiles_512(img_normalized, CONFIG['img_size'])

        print(f"  Extracted {len(tiles)} tiles (512×512)")

        tile_data = []

        for tile_idx, tile in enumerate(tiles):
            # DL prediction
            _, binary_mask = predict_on_tile(model, tile)
            foreground_pct = (np.count_nonzero(binary_mask) / binary_mask.size) * 100

            tile_data.append({
                'tile_idx': tile_idx,
                'tile': tile,
                'mask': binary_mask,
                'density': foreground_pct
            })

            all_data.append({
                'image': img_path.stem,
                'dilution_factor': dilution,
                'tile_idx': tile_idx,
                'method': CONFIG['architecture'],
                'foreground_pct': foreground_pct
            })

            # CLAHE+OTSU
            tile_gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            binary_mask_clahe = apply_clahe_otsu(tile_gray, **CONFIG['clahe'])
            foreground_pct_clahe = (np.count_nonzero(binary_mask_clahe) / binary_mask_clahe.size) * 100

            all_data.append({
                'image': img_path.stem,
                'dilution_factor': dilution,
                'tile_idx': tile_idx,
                'method': 'clahe_otsu',
                'foreground_pct': foreground_pct_clahe
            })

        # Select representative tiles
        densities = [t['density'] for t in tile_data]
        representative_indices = select_representative_tiles(tiles, densities, CONFIG['n_representative_tiles'])

        print(f"  Selected {len(representative_indices)} representative tiles")

        for tile_idx in representative_indices:
            t = tile_data[tile_idx]
            create_tile_comparison(
                t['tile'], t['mask'], t['density'],
                img_path.stem, tile_idx,
                CONFIG['architecture'].upper(),
                subdirs['tiles']
            )

    # Save CSV
    df = pd.DataFrame(all_data)
    csv_path = subdirs['csv'] / 'density_analysis_comprehensive.csv'
    df.to_csv(str(csv_path), index=False)
    print(f"\n✓ Saved CSV: {csv_path}")

    # Create boxplots
    methods = [CONFIG['architecture'], 'clahe_otsu']
    for method in methods:
        plot_path = subdirs['plots'] / f'{method}_density_vs_dilution.png'
        create_boxplot(df, method, plot_path, CONFIG['architecture'].upper())

    print(f"\n{'='*80}")
    print("DENSITY ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
