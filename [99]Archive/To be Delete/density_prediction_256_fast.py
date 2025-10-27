#!/usr/bin/env python3
"""
Fast Density Prediction with 256×256 Tiles
===========================================

Uses existing trained models (or trains once if needed) for:
- U-Net, ResUNet, Attention ResUNet
- 256×256 tile predictions (native model resolution - no interpolation)
- Representative tile visualization (5 per image)
- Density analysis with boxplots

Runtime:
- First run: ~3-4 hours (training + saving models)
- Subsequent runs: ~5 minutes (load models + predict)

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
    'test_images_dir': './test_images',
    'models_dir': './saved_models_validation_config',  # Where to save/load models
    'output_dir': f'./density_prediction_256_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Model configuration (from validation_arch_comparison_20251013_093844)
    'architectures': ['unet', 'resunet', 'attention_resunet'],
    'img_size': 256,  # Training AND prediction size
    'img_channels': 1,
    'filters': 64,
    'dropout': 0.2,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'epochs': 50,
    'early_stopping_patience': 10,

    # CLAHE+OTSU parameters
    'clahe': {'clipLimit': 2.0, 'tileGridSize': (8, 8)},

    # Visualization
    'n_representative_tiles': 5,
    'dpi': 300,
    'figsize_comparison': (16, 4),
    'figsize_boxplot': (12, 8),
}

# Dilution patterns
DILUTION_PATTERNS = {
    '10x': 10, '20x': 20, '40x': 40, '80x': 80,
    '160x': 160, '320x': 320, '640x': 640, '1280x': 1280,
    '2560x': 2560, '5120x': 5120, '10240x': 10240
}

# Colors
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
            print(f"⚠ GPU configuration warning: {e}")
    else:
        print("⚠ No GPU detected")

# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def create_output_dirs(base_dir):
    """Create output directory structure."""
    base = Path(base_dir)
    subdirs = {
        'models': Path(CONFIG['models_dir']),  # Separate persistent models dir
        'plots': base / 'boxplots',
        'tiles': base / 'representative_tiles',
        'csv': base / 'csv_data',
    }

    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    print(f"✓ Created output directory: {base}")
    return subdirs

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def check_existing_models(models_dir):
    """Check if trained models already exist."""
    models_dir = Path(models_dir)
    existing = {}

    for arch in CONFIG['architectures']:
        model_path = models_dir / f'{arch}_best_model.keras'
        if model_path.exists():
            existing[arch] = model_path

    return existing

def load_existing_model(model_path, architecture):
    """Load a saved model."""
    print(f"  Loading {architecture} from {model_path.name}...")

    try:
        model = keras.models.load_model(
            str(model_path),
            custom_objects={
                'combined_dice_focal_loss': combined_dice_focal_loss,
                'jacard_coef': jacard_coef,
                'dice_coef': dice_coef
            }
        )
        print(f"  ✓ {architecture} loaded successfully")
        return model

    except Exception as e:
        print(f"  ✗ Failed to load {architecture}: {e}")
        return None

def load_training_data(dataset_dir, img_size=256):
    """Load training data."""
    from PIL import Image

    images_dir = Path(dataset_dir) / 'images'
    masks_dir = Path(dataset_dir) / 'masks'
    image_files = sorted(images_dir.glob('*.tif'))

    if len(image_files) == 0:
        raise ValueError(f"No .tif files found in {images_dir}")

    images, masks = [], []

    print(f"Loading {len(image_files)} training images...")

    for img_path in tqdm(image_files, desc="Loading training data"):
        img = Image.open(img_path).convert('L')
        img = img.resize((img_size, img_size))
        img_array = np.array(img) / 255.0

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
    return images, masks

def train_model(architecture, X_train, y_train, X_val, y_val, output_dir):
    """Train a single architecture."""
    print(f"\n{'='*70}")
    print(f"Training {architecture.upper()}")
    print(f"{'='*70}")

    input_shape = (CONFIG['img_size'], CONFIG['img_size'], CONFIG['img_channels'])

    model = get_model(
        model_name=architecture,
        input_shape=input_shape,
        NUM_CLASSES=1,
        dropout_rate=CONFIG['dropout'],
        batch_norm=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss=combined_dice_focal_loss,
        metrics=[jacard_coef, dice_coef]
    )

    model_path = output_dir / f'{architecture}_best_model.keras'

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

    print(f"Training {architecture}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    best_jaccard = max(history.history['val_jacard_coef'])
    print(f"✓ {architecture} complete - Best Jaccard: {best_jaccard:.4f}")
    print(f"✓ Model saved: {model_path}")

    return model

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
    """Extract 256×256 tiles (non-overlapping)."""
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
    """Predict on 256×256 tile directly (native resolution - no resizing)."""
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
    """Create 4-panel comparison: original + 3 predicted masks."""
    fig, axes = plt.subplots(1, 4, figsize=CONFIG['figsize_comparison'])

    # Panel 1: Original
    axes[0].imshow(original_tile, cmap='gray')
    axes[0].set_title('Original Tile\n256×256', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panels 2-4: Predicted masks
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
# BOXPLOT GENERATION
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
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("FAST DENSITY PREDICTION - 256×256 TILES")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    setup_gpu()
    subdirs = create_output_dirs(CONFIG['output_dir'])

    # ========================================================================
    # PHASE 1: LOAD OR TRAIN MODELS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: MODEL LOADING/TRAINING")
    print("="*80)

    existing_models = check_existing_models(subdirs['models'])

    if len(existing_models) == len(CONFIG['architectures']):
        print(f"✓ Found all {len(existing_models)} trained models")
        print("  Loading existing models (fast path)...\n")

        trained_models = {}
        for arch in CONFIG['architectures']:
            model = load_existing_model(existing_models[arch], arch)
            if model is None:
                raise RuntimeError(f"Failed to load {arch} model")
            trained_models[arch] = model

        print("\n✓ All models loaded successfully")
        print("  Skipping training (models already exist)")

    else:
        print(f"⚠ Found {len(existing_models)}/{len(CONFIG['architectures'])} models")
        print("  Training missing models (this will take ~3-4 hours)...")
        print("  Future runs will be fast (~5 min) once models are saved\n")

        # Load training data
        X, y = load_training_data(CONFIG['dataset_dir'], CONFIG['img_size'])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Train: {len(X_train)}, Val: {len(X_val)}")

        trained_models = {}
        for arch in CONFIG['architectures']:
            if arch in existing_models:
                print(f"\n✓ {arch} already exists, loading...")
                trained_models[arch] = load_existing_model(existing_models[arch], arch)
            else:
                trained_models[arch] = train_model(arch, X_train, y_train, X_val, y_val, subdirs['models'])

        print("\n✓ All models ready (saved for future use)")

    # ========================================================================
    # PHASE 2: PREDICTION ON TEST IMAGES
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: PREDICTION AND DENSITY ANALYSIS")
    print("="*80)

    test_dir = Path(CONFIG['test_images_dir'])
    test_images = sorted(test_dir.glob('*.tif'))

    if len(test_images) == 0:
        raise ValueError(f"No test images found in {test_dir}")

    print(f"Found {len(test_images)} test images")

    all_data = []

    for img_path in tqdm(test_images, desc="Processing test images"):
        dilution = extract_dilution_factor(img_path.stem)
        if dilution is None:
            print(f"  ⚠ Skipping {img_path.name} - no dilution factor")
            continue

        print(f"\nProcessing: {img_path.name} (dilution: {dilution}x)")

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  ✗ Could not read {img_path.name}")
            continue

        img_rescaled = rescale_image_full_range(img)
        img_normalized = img_rescaled.astype(np.float32) / 255.0

        # Extract 256×256 tiles
        tiles, positions = extract_tiles_256(img_normalized, tile_size=CONFIG['img_size'])

        if len(tiles) == 0:
            print(f"  ⚠ No tiles extracted")
            continue

        print(f"  Extracted {len(tiles)} tiles (256×256)")

        # Predict with all models and calculate densities
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

        # Select 5 representative tiles
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

        print(f"  ✓ Saved {len(representative_indices)} comparison images")

    # ========================================================================
    # PHASE 3: CREATE CSV AND BOXPLOTS
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: GENERATING OUTPUTS")
    print("="*80)

    df = pd.DataFrame(all_data)
    print(f"\n✓ Collected {len(df)} density measurements")

    # Save CSV
    csv_path = subdirs['csv'] / 'density_analysis_comprehensive.csv'
    df.to_csv(str(csv_path), index=False)
    print(f"✓ Saved CSV: {csv_path}")

    # Create boxplots
    methods = ['unet', 'resunet', 'attention_resunet', 'clahe_otsu']

    for method in methods:
        plot_path = subdirs['plots'] / f'{method}_density_vs_dilution.png'
        create_boxplot(df, method, plot_path)

    print("\n" + "="*80)
    print("FAST DENSITY PREDICTION COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput directory: {CONFIG['output_dir']}")
    print(f"  - Persistent models: {subdirs['models']}")
    print(f"  - Representative tiles: {subdirs['tiles']}")
    print(f"  - Boxplots: {subdirs['plots']}")
    print(f"  - CSV: {subdirs['csv']}")
    print("\n💡 Next run will be FAST (~5 min) - models already saved!")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
