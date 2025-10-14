#!/usr/bin/env python3
"""
Density Analysis for Architecture Comparison Models
====================================================

Creates density analysis using models trained with configurations from
validation_arch_comparison_20251013_093844.

Generates:
- 4 separate PNG files (one per architecture/method)
- 1 comprehensive CSV with all density data
- Y-axis: Foreground Percentage (log scale)
- X-axis: 1/Dilution Factor

Architectures:
- U-Net (baseline)
- ResUNet (residual connections)
- Attention ResUNet (residual + attention gates)
- CLAHE+OTSU (traditional CV method)

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
from sklearn.model_selection import train_test_split

# Import custom modules
from model_architectures import get_model
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Directories
    'dataset_dir': './dataset_full_stack',
    'test_images_dir': './test_images',  # Directory with dilution series images
    'output_dir': f'./density_analysis_arch_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Model configuration (from validation_arch_comparison)
    'architectures': ['unet', 'resunet', 'attention_resunet'],
    'img_size': 256,
    'img_channels': 1,
    'filters': 64,
    'dropout': 0.2,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'epochs': 50,
    'early_stopping_patience': 10,

    # CLAHE+OTSU parameters
    'clahe': {'clipLimit': 2.0, 'tileGridSize': (8, 8)},

    # Plotting
    'dpi': 300,
    'figsize': (12, 8),
}

# Dilution factor patterns
DILUTION_PATTERNS = {
    '10x': 10,
    '20x': 20,
    '40x': 40,
    '80x': 80,
    '160x': 160,
    '320x': 320,
    '640x': 640,
    '1280x': 1280,
    '2560x': 2560,
    '5120x': 5120,
    '10240x': 10240
}

# Color scheme (matching reference plot - viridis-like gradient)
COLORS = {
    'unet': '#440154',          # Dark purple
    'resunet': '#31688e',       # Blue
    'attention_resunet': '#35b779',  # Green
    'clahe_otsu': '#fde724'     # Yellow
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
            return True
        except RuntimeError as e:
            print(f"⚠ GPU configuration warning: {e}")
            return False
    else:
        print("⚠ No GPU detected - training will be slow!")
        return False

# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def create_output_dirs(base_dir):
    """Create output directory structure."""
    base = Path(base_dir)
    subdirs = {
        'models': base / 'trained_models',
        'plots': base / 'plots',
        'csv': base / 'csv_data',
    }

    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    print(f"✓ Created output directory: {base}")
    return subdirs

# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(dataset_dir, img_size=256):
    """Load training data from dataset_full_stack."""
    from PIL import Image

    images_dir = Path(dataset_dir) / 'images'
    masks_dir = Path(dataset_dir) / 'masks'

    image_files = sorted(images_dir.glob('*.tif'))

    if len(image_files) == 0:
        raise ValueError(f"No .tif files found in {images_dir}")

    images = []
    masks = []

    print(f"Loading {len(image_files)} training images...")

    for img_path in tqdm(image_files, desc="Loading training data"):
        # Load image
        img = Image.open(img_path).convert('L')
        img = img.resize((img_size, img_size))
        img_array = np.array(img) / 255.0

        # Load mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            continue

        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((img_size, img_size))
        mask_array = np.array(mask) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)

        images.append(img_array)
        masks.append(mask_array)

    images = np.array(images)[..., np.newaxis]
    masks = np.array(masks)[..., np.newaxis]

    print(f"✓ Loaded {len(images)} image-mask pairs")
    print(f"  Shape: {images.shape}")

    return images, masks

def load_test_images(test_dir):
    """Load test images with dilution factors."""
    test_path = Path(test_dir)

    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    image_files = sorted(test_path.glob('*.tif'))

    if len(image_files) == 0:
        raise ValueError(f"No .tif files found in {test_dir}")

    print(f"Found {len(image_files)} test images")

    return image_files

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(architecture, X_train, y_train, X_val, y_val, output_dir):
    """Train a single architecture."""
    print(f"\n{'='*70}")
    print(f"Training {architecture.upper()}")
    print(f"{'='*70}")

    # Build model
    input_shape = (CONFIG['img_size'], CONFIG['img_size'], CONFIG['img_channels'])

    model = get_model(
        model_name=architecture,
        input_shape=input_shape,
        NUM_CLASSES=1,
        dropout_rate=CONFIG['dropout'],
        batch_norm=True
    )

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss=combined_dice_focal_loss,
        metrics=[jacard_coef, dice_coef]
    )

    # Callbacks
    model_path = output_dir / f'{architecture}_best_model.keras'

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(model_path),  # Convert Path to string for Keras compatibility
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
    print(f"Training {architecture}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    # Get best performance
    val_jacards = history.history['val_jacard_coef']
    best_epoch = np.argmax(val_jacards)
    best_jaccard = val_jacards[best_epoch]

    print(f"✓ {architecture} training complete")
    print(f"  Best Jaccard: {best_jaccard:.4f} (epoch {best_epoch})")
    print(f"  Model saved: {model_path}")

    return model, best_jaccard

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

def extract_tiles(image, tile_size=256):
    """Extract tiles from image with stride."""
    h, w = image.shape[:2]
    tiles = []
    positions = []

    stride = tile_size  # Non-overlapping tiles

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions

# ============================================================================
# PREDICTION AND DENSITY CALCULATION
# ============================================================================

def predict_on_image(model, image_path):
    """Predict on image tiles and calculate foreground percentage."""
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_path}")

    img_rescaled = rescale_image_full_range(img)
    img_normalized = img_rescaled.astype(np.float32) / 255.0

    # Extract tiles
    tiles, positions = extract_tiles(img_normalized, tile_size=CONFIG['img_size'])

    if len(tiles) == 0:
        print(f"  ⚠ No tiles extracted from {image_path.name}")
        return []

    # Predict
    densities = []
    for tile in tiles:
        tile_input = tile[np.newaxis, ..., np.newaxis]
        pred = model.predict(tile_input, verbose=0)
        pred_mask = pred[0, ..., 0]

        # Calculate foreground percentage (mean of predicted probability)
        foreground_pct = np.mean(pred_mask) * 100  # Convert to percentage
        densities.append(foreground_pct)

    return densities

def clahe_otsu_on_image(image_path):
    """Apply CLAHE+OTSU on image tiles and calculate foreground percentage."""
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_path}")

    img_rescaled = rescale_image_full_range(img)

    # Extract tiles
    tiles, positions = extract_tiles(img_rescaled, tile_size=CONFIG['img_size'])

    if len(tiles) == 0:
        return []

    # Process
    densities = []
    for tile in tiles:
        binary_mask = apply_clahe_otsu(
            tile,
            clipLimit=CONFIG['clahe']['clipLimit'],
            tileGridSize=CONFIG['clahe']['tileGridSize']
        )

        # Calculate foreground percentage
        foreground_pct = (np.count_nonzero(binary_mask) / binary_mask.size) * 100
        densities.append(foreground_pct)

    return densities

# ============================================================================
# DILUTION FACTOR EXTRACTION
# ============================================================================

def extract_dilution_factor(filename):
    """Extract dilution factor from filename."""
    # Try known patterns first
    for pattern, value in DILUTION_PATTERNS.items():
        if pattern.lower() in filename.lower():
            return value

    # Try to extract number followed by 'x'
    match = re.search(r'(\d+)x', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None

# ============================================================================
# PLOTTING
# ============================================================================

def create_individual_plot(df, method, output_path):
    """
    Create individual boxplot for one method.

    Y-axis: Foreground Percentage (log scale)
    X-axis: 1/Dilution Factor
    """
    # Filter data for this method
    df_method = df[df['method'] == method].copy()

    if len(df_method) == 0:
        print(f"  ⚠ No data for {method}")
        return

    # Sort by dilution factor
    df_method = df_method.sort_values('dilution_factor')

    # Get unique dilution factors
    dilution_factors = sorted(df_method['dilution_factor'].unique())

    # Create figure
    fig, ax = plt.subplots(figsize=CONFIG['figsize'])

    # Prepare data for boxplot
    data_to_plot = []
    labels = []
    positions = []

    for i, dilution in enumerate(dilution_factors):
        df_dilution = df_method[df_method['dilution_factor'] == dilution]
        data_to_plot.append(df_dilution['foreground_pct'].values)
        labels.append(f"{int(dilution)}x")
        positions.append(i)

    # Create boxplot
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

    # Set labels
    ax.set_xlabel('1/Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Percentage', fontsize=14, fontweight='bold')
    ax.set_title(f'Particle Density vs. Dilution Factor\n(Method: {method.replace("_", " ").title()})',
                 fontsize=16, fontweight='bold', pad=20)

    # Set x-axis ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12, rotation=45, ha='right')

    # Set y-axis to log scale
    ax.set_yscale('log')
    ax.set_ylabel('Foreground Percentage (log scale)', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save
    plt.savefig(str(output_path), dpi=CONFIG['dpi'], bbox_inches='tight')  # Convert Path to string
    plt.close()

    print(f"✓ Saved: {output_path.name}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("DENSITY ANALYSIS - ARCHITECTURE COMPARISON")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Setup
    setup_gpu()
    subdirs = create_output_dirs(CONFIG['output_dir'])

    # Load training data
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    X, y = load_training_data(CONFIG['dataset_dir'], CONFIG['img_size'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train models
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)

    trained_models = {}
    for arch in CONFIG['architectures']:
        model, best_jaccard = train_model(
            arch, X_train, y_train, X_val, y_val, subdirs['models']
        )
        trained_models[arch] = model

    # Load test images
    print("\n" + "="*80)
    print("LOADING TEST IMAGES")
    print("="*80)
    test_images = load_test_images(CONFIG['test_images_dir'])

    # Perform predictions and collect density data
    print("\n" + "="*80)
    print("PERFORMING PREDICTIONS AND DENSITY ANALYSIS")
    print("="*80)

    all_data = []

    for img_path in tqdm(test_images, desc="Processing test images"):
        # Extract dilution factor
        dilution = extract_dilution_factor(img_path.stem)
        if dilution is None:
            print(f"  ⚠ Skipping {img_path.name} - no dilution factor")
            continue

        print(f"\nProcessing: {img_path.name} (dilution: {dilution}x)")

        # DL models
        for arch in CONFIG['architectures']:
            print(f"  {arch}...", end=' ')
            densities = predict_on_image(trained_models[arch], img_path)

            for density in densities:
                all_data.append({
                    'image': img_path.stem,
                    'dilution_factor': dilution,
                    'method': arch,
                    'foreground_pct': density
                })

            print(f"{len(densities)} tiles, mean: {np.mean(densities):.4f}%")

        # CLAHE+OTSU
        print(f"  clahe_otsu...", end=' ')
        densities = clahe_otsu_on_image(img_path)

        for density in densities:
            all_data.append({
                'image': img_path.stem,
                'dilution_factor': dilution,
                'method': 'clahe_otsu',
                'foreground_pct': density
            })

        print(f"{len(densities)} tiles, mean: {np.mean(densities):.4f}%")

    # Create DataFrame
    df = pd.DataFrame(all_data)

    print(f"\n✓ Collected {len(df)} density measurements")
    print(f"  Methods: {df['method'].unique().tolist()}")
    print(f"  Dilution factors: {sorted(df['dilution_factor'].unique())}")

    # Save comprehensive CSV
    csv_path = subdirs['csv'] / 'density_analysis_comprehensive.csv'
    df.to_csv(str(csv_path), index=False)  # Convert Path to string
    print(f"\n✓ Saved comprehensive CSV: {csv_path}")

    # Create individual plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)

    methods = ['unet', 'resunet', 'attention_resunet', 'clahe_otsu']

    for method in methods:
        plot_path = subdirs['plots'] / f'{method}_density_vs_dilution.png'
        create_individual_plot(df, method, plot_path)

    print("\n" + "="*80)
    print("DENSITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput directory: {CONFIG['output_dir']}")
    print(f"  - Models: {subdirs['models']}")
    print(f"  - Plots: {subdirs['plots']}")
    print(f"  - CSV: {subdirs['csv']}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
