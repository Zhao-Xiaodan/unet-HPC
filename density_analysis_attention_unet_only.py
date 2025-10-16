#!/usr/bin/env python3
"""
Attention UNet-Only Density Analysis with Improved Visualizations
==================================================================

Script: density_analysis_attention_unet_only.py
PBS Script: pbs_density_analysis_attention_unet_only.sh

Performs density analysis using the BEST Attention UNet model from hyperparameter search.

Features:
1. ✅ Tile-Level Density Tracking (all 28 tiles per image saved)
2. ✅ Multiple threshold analysis (0.2, 0.5, 0.8, 0.95)
3. ✅ CLAHE+Otsu denoising on predicted masks and original images
4. ✅ Categorical boxplot x-axis (1/10240x → 1/10x)
5. ✅ 3-panel tile visualizations (Original, Inverted Pred, Inverted CLAHE+Otsu)

Uses the BEST Attention UNet model from hyperparameter search:
- Source: attention_unet_hyperparam_20251015_230149/models/
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
    'model_base_dir': './attention_unet_hyperparam_20251015_230149',
    'test_images_dir': './test_images',
    'output_dir': f'./density_analysis_attention_unet_only_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Image settings (RGB from training)
    'img_size': 512,  # Tile size
    'img_channels': 3,  # RGB

    # Prediction settings
    'batch_size': 8,
    'thresholds': [0.2, 0.5, 0.8, 0.95],  # Multiple thresholds to test

    # Representative tiles (reuse from previous analysis)
    'n_representative_tiles': 5,

    # Plotting
    'dpi': 300,
    'figsize_boxplot_full': (14, 8),
    'figsize_boxplot_low': (14, 8),
}

# Dilution factors in proper order (10240x - 10x)
# X-axis goes from LOW density (1/10240x) to HIGH density (1/10x)
DILUTION_ORDER = [10240, 5120, 2560, 1280, 640, 320, 160, 80, 20, 10]
DILUTION_LABELS = ['10240x', '5120x', '2560x', '1280x', '640x', '320x', '160x', '80x', '20x', '10x']

# Low dilution range (1/10240 to 1/80) - reversed for low→high density
DILUTION_ORDER_LOW = [10240, 5120, 2560, 1280, 640, 320, 160, 80]
DILUTION_LABELS_LOW = ['10240x', '5120x', '2560x', '1280x', '640x', '320x', '160x', '80x']

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
    print("ATTENTION UNET-ONLY DENSITY ANALYSIS - BEST MODEL FROM HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Script: density_analysis_attention_unet_only.py")
    print(f"PBS Script: pbs_density_analysis_attention_unet_only.sh")
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

def find_best_attention_unet_model(base_dir):
    """
    Find the best Attention UNet model from hyperparameter search.
    Looks through all models and finds the one with highest validation IoU.
    """
    print("Searching for best Attention UNet model...")
    base_dir = Path(base_dir)

    # Find all best_model.keras files in models directory
    model_dirs = list((base_dir / 'models').glob('attention_unet_*'))

    if not model_dirs:
        raise FileNotFoundError(f"No Attention UNet model directories found in {base_dir / 'models'}")

    print(f"Found {len(model_dirs)} Attention UNet model configurations")

    best_model_info = None
    best_iou = -1.0

    # Search through each model directory
    for model_dir in model_dirs:
        model_file = model_dir / 'best_model.keras'

        if not model_file.exists():
            continue

        # Parse hyperparameters from directory name
        # Format: attention_unet_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001
        dir_name = model_dir.name

        try:
            # Extract hyperparameters
            n_filters = int(re.search(r'n_filters(\d+)', dir_name).group(1))
            dropout_str = re.search(r'dropout(\d+p\d+)', dir_name).group(1)
            dropout = float(dropout_str.replace('p', '.'))
            batch_norm = 'batch_normTrue' in dir_name
            lr_str = re.search(r'learning_rate(\d+p\d+)', dir_name).group(1)
            learning_rate = float(lr_str.replace('p', '.'))

            # Find corresponding history CSV to get IoU
            history_files = list((base_dir / 'logs').glob(f'attention_unet_n_filters{n_filters}_dropout{dropout_str}_*_history.csv'))

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
        raise FileNotFoundError("Could not find any valid Attention UNet models with validation IoU data")

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

def rescale_image_full_range(img):
    """
    Rescale image to full 0-255 range.
    This is CRITICAL for CLAHE+Otsu to work properly.

    Args:
        img: Image (any dtype, any range)

    Returns:
        img_rescaled: uint8 image with full 0-255 range
    """
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)

def apply_clahe_otsu(img_gray, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    Apply CLAHE + OTSU thresholding to denoise predicted masks or original images.

    IMPORTANT: Input should be rescaled to 0-255 first using rescale_image_full_range()

    Args:
        img_gray: Grayscale image (0-255 uint8)
        clipLimit: CLAHE clip limit
        tileGridSize: CLAHE tile grid size

    Returns:
        binary_mask: Binary mask after CLAHE+Otsu (0-255 uint8)
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_img = clahe.apply(img_gray)
    # THRESH_BINARY_INV because beads are black (foreground becomes white)
    _, binary_mask = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_mask

def calculate_density_with_clahe_otsu(prediction, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    Calculate density from predicted mask after applying CLAHE+Otsu denoising.

    This reduces noise in low-density scenarios.

    Args:
        prediction: Predicted mask (0-1 float32)
        clipLimit: CLAHE clip limit
        tileGridSize: CLAHE tile grid size

    Returns:
        density: Foreground density after CLAHE+Otsu denoising
        binary_mask: Binary mask after CLAHE+Otsu (for visualization)
    """
    # CRITICAL: Rescale to full 0-255 range FIRST
    pred_rescaled = rescale_image_full_range(prediction.squeeze())

    # Apply CLAHE+Otsu
    binary_mask = apply_clahe_otsu(pred_rescaled, clipLimit=clipLimit, tileGridSize=tileGridSize)

    # Calculate density (INVERTED because we want bead density, not background)
    # binary_mask has 255 for foreground after THRESH_BINARY_INV
    # But we need to invert to get actual bead density
    density = 1.0 - (np.count_nonzero(binary_mask) / binary_mask.size)
    return density, binary_mask

def calculate_density_clahe_otsu_on_original(tile):
    """
    Apply CLAHE+Otsu directly on original image tile (baseline method).

    This serves as a traditional CV baseline for comparison.

    IMPORTANT: For original microscope images, beads are DARK on BRIGHT background.
    We need to INVERT the image first so that THRESH_BINARY_INV works correctly.

    Args:
        tile: Original tile (RGB, 0-1 float32)

    Returns:
        density: Foreground density from CLAHE+Otsu on original
        binary_mask: Binary mask after CLAHE+Otsu (for visualization)
    """
    # Convert to grayscale if RGB
    if len(tile.shape) == 3:
        tile_gray = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        tile_gray = (tile * 255).astype(np.uint8)

    # CRITICAL: Invert the grayscale image so beads become BRIGHT on DARK background
    # This is necessary because original microscope images have dark beads on bright background,
    # but our CLAHE+Otsu pipeline expects bright foreground on dark background
    tile_gray_inverted = 255 - tile_gray

    # CRITICAL: Rescale to full 0-255 range AFTER inversion
    tile_rescaled = rescale_image_full_range(tile_gray_inverted)

    # Apply CLAHE+Otsu
    binary_mask = apply_clahe_otsu(tile_rescaled, clipLimit=2.0, tileGridSize=(8, 8))

    # Calculate density
    density = np.count_nonzero(binary_mask) / binary_mask.size
    return density, binary_mask

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

        # Calculate densities for each tile (multiple methods)
        for tile_idx, ((tile, pos), prediction) in enumerate(zip(tiles_with_pos, predictions)):
            # Methods 1-4: Simple thresholds (0.2, 0.5, 0.8, 0.95)
            density_threshold_02 = calculate_foreground_density(prediction, 0.2)
            density_threshold_05 = calculate_foreground_density(prediction, 0.5)
            density_threshold_08 = calculate_foreground_density(prediction, 0.8)
            density_threshold_095 = calculate_foreground_density(prediction, 0.95)

            # Method 5: CLAHE+Otsu on predicted mask (denoising)
            density_clahe_otsu_pred, binary_mask_pred = calculate_density_with_clahe_otsu(prediction)

            # Method 6: CLAHE+Otsu on original tile (baseline CV method)
            density_clahe_otsu_orig, binary_mask_orig = calculate_density_clahe_otsu_on_original(tile)

            # Store tile-level results for all methods
            tile_results.append({
                'image': img_file.name,
                'dilution': dilution,
                'dilution_label': dilution_label,
                'tile_idx': tile_idx,
                'position_y': pos[0],
                'position_x': pos[1],
                'density_threshold_0.2': density_threshold_02,
                'density_threshold_0.5': density_threshold_05,
                'density_threshold_0.8': density_threshold_08,
                'density_threshold_0.95': density_threshold_095,
                'density_clahe_otsu_pred': density_clahe_otsu_pred,
                'density_clahe_otsu_orig': density_clahe_otsu_orig,
            })

            # Store tile data for visualization
            tile_data.append({
                'image': img_file.name,
                'dilution': dilution,
                'dilution_label': dilution_label,
                'tile_idx': tile_idx,
                'position': pos,
                'tile': tile,
                'prediction': prediction,
                'density_threshold_0.2': density_threshold_02,
                'density_threshold_0.5': density_threshold_05,
                'density_threshold_0.8': density_threshold_08,
                'density_threshold_0.95': density_threshold_095,
                'density_clahe_otsu_pred': density_clahe_otsu_pred,
                'density_clahe_otsu_orig': density_clahe_otsu_orig,
                'binary_mask_pred': binary_mask_pred,  # For visualization
                'binary_mask_orig': binary_mask_orig,  # For visualization
            })

        # Print summary for this image (all methods)
        img_results = [r for r in tile_results if r['image'] == img_file.name]
        mean_threshold_02 = np.mean([r['density_threshold_0.2'] for r in img_results])
        mean_threshold_05 = np.mean([r['density_threshold_0.5'] for r in img_results])
        mean_threshold_08 = np.mean([r['density_threshold_0.8'] for r in img_results])
        mean_threshold_095 = np.mean([r['density_threshold_0.95'] for r in img_results])
        mean_clahe_pred = np.mean([r['density_clahe_otsu_pred'] for r in img_results])
        mean_clahe_orig = np.mean([r['density_clahe_otsu_orig'] for r in img_results])

        print(f"  Mean densities (across {len(tiles_with_pos)} tiles):")
        print(f"    Threshold (0.2):        {mean_threshold_02:.4f}")
        print(f"    Threshold (0.5):        {mean_threshold_05:.4f}")
        print(f"    Threshold (0.8):        {mean_threshold_08:.4f}")
        print(f"    Threshold (0.95):       {mean_threshold_095:.4f}")
        print(f"    CLAHE+Otsu on pred:     {mean_clahe_pred:.4f}")
        print(f"    CLAHE+Otsu on original: {mean_clahe_orig:.4f}")

    df_tile_results = pd.DataFrame(tile_results)

    return df_tile_results, tile_data

# ============================================================================
# VISUALIZATION - BOX PLOTS WITH TWO DILUTION RANGES
# ============================================================================

def create_boxplot_full_range(df_tile_results, output_dir, config, density_column='density_threshold', title_suffix=''):
    """
    Create box plot for FULL dilution range (1/10240 to 1/10).
    Uses categorical x-axis with dilution labels (matching reference style).

    Args:
        df_tile_results: DataFrame with tile-level results
        output_dir: Output directory
        config: Configuration dict
        density_column: Which density column to plot
        title_suffix: Suffix for title and filename
    """
    print(f"\nGenerating box plot (Full Range: 1/10240 to 1/10) - {title_suffix}...")

    # Filter to only include dilutions in our defined order
    df_plot = df_tile_results[df_tile_results['dilution'].isin(DILUTION_ORDER)].copy()

    # Create inverse dilution labels
    inv_dilution_labels = [f'1/{d}x' for d in DILUTION_ORDER]

    # Create figure
    fig, ax = plt.subplots(figsize=config['figsize_boxplot_full'])

    # Colors: blue boxes with black median like reference
    box_color = '#3498db'  # Blue
    median_color = 'black'

    # Create CATEGORICAL positions (0, 1, 2, 3, ...) like reference
    positions = list(range(len(DILUTION_ORDER)))

    # Prepare data for each dilution
    data_by_dilution = []
    for dilution in DILUTION_ORDER:
        dilution_data = df_plot[df_plot['dilution'] == dilution][density_column].values
        data_by_dilution.append(dilution_data)

    # Create boxplot
    bp = ax.boxplot(
        data_by_dilution,
        positions=positions,
        widths=0.6,  # Fixed width like reference
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=box_color, color='black', linewidth=1, alpha=0.7),
        medianprops=dict(color=median_color, linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5)
    )

    # Customize
    ax.set_title(f'Particle Density vs Dilution Factor{title_suffix}',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('1/Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Percentage (log scale)', fontsize=14, fontweight='bold')

    # Set y-axis to log scale like reference
    ax.set_yscale('log')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set x-axis ticks and labels (categorical)
    ax.set_xticks(positions)
    ax.set_xticklabels(inv_dilution_labels, fontsize=12, rotation=45, ha='right')

    plt.tight_layout()

    # Save
    filename = f'density_boxplot_full_range{title_suffix.lower().replace(" ", "_").replace("-", "")}.png'
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")

def create_boxplot_low_dilution_range(df_tile_results, output_dir, config, density_column='density_threshold', title_suffix=''):
    """
    Create box plot for LOW dilution range (1/10240 to 1/80).
    Uses categorical x-axis with dilution labels (matching reference style).

    Args:
        df_tile_results: DataFrame with tile-level results
        output_dir: Output directory
        config: Configuration dict
        density_column: Which density column to plot
        title_suffix: Suffix for title and filename
    """
    print(f"\nGenerating box plot (Low Dilution Range: 1/10240 to 1/80) - {title_suffix}...")

    # Filter to only include low dilutions
    df_plot = df_tile_results[df_tile_results['dilution'].isin(DILUTION_ORDER_LOW)].copy()

    # Create inverse dilution labels
    inv_dilution_labels_low = [f'1/{d}x' for d in DILUTION_ORDER_LOW]

    # Create figure
    fig, ax = plt.subplots(figsize=config['figsize_boxplot_low'])

    # Colors: blue boxes with black median like reference
    box_color = '#3498db'  # Blue
    median_color = 'black'

    # Create CATEGORICAL positions (0, 1, 2, 3, ...) like reference
    positions = list(range(len(DILUTION_ORDER_LOW)))

    # Prepare data for each dilution
    data_by_dilution = []
    for dilution in DILUTION_ORDER_LOW:
        dilution_data = df_plot[df_plot['dilution'] == dilution][density_column].values
        data_by_dilution.append(dilution_data)

    # Create boxplot
    bp = ax.boxplot(
        data_by_dilution,
        positions=positions,
        widths=0.6,  # Fixed width like reference
        patch_artist=True,
        showfliers=True,
        boxprops=dict(facecolor=box_color, color='black', linewidth=1, alpha=0.7),
        medianprops=dict(color=median_color, linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5)
    )

    # Customize
    ax.set_title(f'Particle Density vs Dilution Factor (Low Dilution Range){title_suffix}',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('1/Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Percentage (log scale)', fontsize=14, fontweight='bold')

    # Set y-axis to log scale like reference
    ax.set_yscale('log')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set x-axis ticks and labels (categorical)
    ax.set_xticks(positions)
    ax.set_xticklabels(inv_dilution_labels_low, fontsize=12, rotation=45, ha='right')

    plt.tight_layout()

    # Save
    filename = f'density_boxplot_low_dilution_range{title_suffix.lower().replace(" ", "_").replace("-", "")}.png'
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path}")

# ============================================================================
# VISUALIZATION - REPRESENTATIVE TILES (3-PANEL)
# ============================================================================

def create_representative_tile_visualizations(tile_data, output_dir, config):
    """
    Create 3-panel tile visualizations for each test image.
    Shows 5 representative tiles per image with:
    - Panel 1: Original tile (dark beads on bright background)
    - Panel 2: Inverted Predicted mask threshold 0.5 (white beads on dark background)
    - Panel 3: Inverted CLAHE+Otsu mask (white beads on dark background)

    IMPORTANT: Panels 2-3 show INVERTED masks (white beads) for better visibility.
    """
    print("\nGenerating representative tile visualizations (3-panel)...")

    output_dir = Path(output_dir)
    tiles_dir = output_dir / 'representative_tiles_3panel'
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Group tiles by image
    tiles_by_image = defaultdict(list)
    for tile_info in tile_data:
        tiles_by_image[tile_info['image']].append(tile_info)

    # Process each image
    for image_name, tiles in tiles_by_image.items():
        print(f"  Creating tiles for: {image_name}")

        # Sort tiles by CLAHE+Otsu density (denoised) to get representative range
        tiles_sorted = sorted(tiles, key=lambda x: x['density_clahe_otsu_pred'])

        # Select 5 representative tiles (min, 25th percentile, median, 75th percentile, max)
        n_tiles = len(tiles_sorted)
        indices = [
            0,                          # Min density
            n_tiles // 4,               # 25th percentile
            n_tiles // 2,               # Median
            3 * n_tiles // 4,           # 75th percentile
            n_tiles - 1,                # Max density
        ]

        representative_tiles = [tiles_sorted[i] for i in indices]

        # Get dilution label for filename
        dilution_label = tiles[0]['dilution_label']

        # Create 3-panel figure (5 rows × 3 columns)
        fig, axes = plt.subplots(5, 3, figsize=(15, 25))

        for row_idx, tile_info in enumerate(representative_tiles):
            tile = tile_info['tile']
            prediction = tile_info['prediction']
            density_threshold_05 = tile_info['density_threshold_0.5']
            density_clahe_pred = tile_info['density_clahe_otsu_pred']
            binary_mask_pred = tile_info['binary_mask_pred']
            tile_idx = tile_info['tile_idx']
            pos = tile_info['position']

            # Panel 1: Original tile (dark beads on bright background)
            axes[row_idx, 0].imshow(tile, cmap='gray')
            axes[row_idx, 0].set_title(f'Original Tile {tile_idx}\nPosition: ({pos[0]}, {pos[1]})',
                                       fontsize=11)
            axes[row_idx, 0].axis('off')

            # Panel 2: Inverted Predicted mask threshold 0.5 (white beads on dark background)
            pred_squeeze = prediction.squeeze()
            inverted_pred = 1.0 - pred_squeeze  # Invert: white beads
            axes[row_idx, 1].imshow(inverted_pred, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 1].set_title(f'Inverted Predicted (0.5)\n(White = Beads)\nDensity: {density_threshold_05:.4f}',
                                       fontsize=11)
            axes[row_idx, 1].axis('off')

            # Panel 3: Inverted CLAHE+Otsu mask (white beads on dark background)
            # binary_mask_pred has white foreground after THRESH_BINARY_INV
            # Keep as-is to show white beads
            axes[row_idx, 2].imshow(binary_mask_pred, cmap='gray', vmin=0, vmax=255)
            axes[row_idx, 2].set_title(f'Inverted CLAHE+Otsu\n(White = Beads, Denoised)\nDensity: {density_clahe_pred:.4f}',
                                       fontsize=11)
            axes[row_idx, 2].axis('off')

        # Add overall title
        fig.suptitle(f'{image_name} - Representative Tiles (5 tiles, White = Beads)',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.995])

        # Save
        output_filename = f'tiles_3panel_{dilution_label}_{image_name.replace(".tif", "").replace(".tiff", "")}.png'
        output_path = tiles_dir / output_filename
        plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
        plt.close()

        print(f"    ✓ Saved: {output_path.name}")

    print(f"\n  ✓ All tile visualizations saved to: {tiles_dir}")
    print(f"    Total images: {len(tiles_by_image)}")
    print(f"    Format: 3 panels (Original, Predicted 0.5, CLAHE+Otsu) - Dark = Beads")

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
            # Threshold 0.2
            'mean_density_threshold_0.2': img_data['density_threshold_0.2'].mean(),
            'median_density_threshold_0.2': img_data['density_threshold_0.2'].median(),
            'std_density_threshold_0.2': img_data['density_threshold_0.2'].std(),
            # Threshold 0.5
            'mean_density_threshold_0.5': img_data['density_threshold_0.5'].mean(),
            'median_density_threshold_0.5': img_data['density_threshold_0.5'].median(),
            'std_density_threshold_0.5': img_data['density_threshold_0.5'].std(),
            # Threshold 0.8
            'mean_density_threshold_0.8': img_data['density_threshold_0.8'].mean(),
            'median_density_threshold_0.8': img_data['density_threshold_0.8'].median(),
            'std_density_threshold_0.8': img_data['density_threshold_0.8'].std(),
            # Threshold 0.95
            'mean_density_threshold_0.95': img_data['density_threshold_0.95'].mean(),
            'median_density_threshold_0.95': img_data['density_threshold_0.95'].median(),
            'std_density_threshold_0.95': img_data['density_threshold_0.95'].std(),
            # CLAHE+Otsu on predicted mask
            'mean_density_clahe_otsu_pred': img_data['density_clahe_otsu_pred'].mean(),
            'median_density_clahe_otsu_pred': img_data['density_clahe_otsu_pred'].median(),
            'std_density_clahe_otsu_pred': img_data['density_clahe_otsu_pred'].std(),
            # CLAHE+Otsu on original
            'mean_density_clahe_otsu_orig': img_data['density_clahe_otsu_orig'].mean(),
            'median_density_clahe_otsu_orig': img_data['density_clahe_otsu_orig'].median(),
            'std_density_clahe_otsu_orig': img_data['density_clahe_otsu_orig'].std(),
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
    best_model_info = find_best_attention_unet_model(CONFIG['model_base_dir'])

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

    # Generate boxplots for all 6 density calculation methods
    print("\n--- Threshold Method (0.2) ---")
    create_boxplot_full_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                              density_column='density_threshold_0.2',
                              title_suffix=' - Threshold 0.2')
    create_boxplot_low_dilution_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                                      density_column='density_threshold_0.2',
                                      title_suffix=' - Threshold 0.2')

    print("\n--- Threshold Method (0.5) ---")
    create_boxplot_full_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                              density_column='density_threshold_0.5',
                              title_suffix=' - Threshold 0.5')
    create_boxplot_low_dilution_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                                      density_column='density_threshold_0.5',
                                      title_suffix=' - Threshold 0.5')

    print("\n--- Threshold Method (0.8) ---")
    create_boxplot_full_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                              density_column='density_threshold_0.8',
                              title_suffix=' - Threshold 0.8')
    create_boxplot_low_dilution_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                                      density_column='density_threshold_0.8',
                                      title_suffix=' - Threshold 0.8')

    print("\n--- Threshold Method (0.95) ---")
    create_boxplot_full_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                              density_column='density_threshold_0.95',
                              title_suffix=' - Threshold 0.95')
    create_boxplot_low_dilution_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                                      density_column='density_threshold_0.95',
                                      title_suffix=' - Threshold 0.95')

    print("\n--- CLAHE+Otsu on Predicted Mask (Denoised) ---")
    create_boxplot_full_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                              density_column='density_clahe_otsu_pred',
                              title_suffix=' - CLAHE+Otsu on Pred')
    create_boxplot_low_dilution_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                                      density_column='density_clahe_otsu_pred',
                                      title_suffix=' - CLAHE+Otsu on Pred')

    print("\n--- CLAHE+Otsu on Original Image (Baseline) ---")
    create_boxplot_full_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                              density_column='density_clahe_otsu_orig',
                              title_suffix=' - CLAHE+Otsu on Original')
    create_boxplot_low_dilution_range(df_tile_results, CONFIG['output_dir'], CONFIG,
                                      density_column='density_clahe_otsu_orig',
                                      title_suffix=' - CLAHE+Otsu on Original')

    # 3-panel tile visualizations (5 representative tiles per image)
    print("\n--- Tile Visualizations ---")
    create_representative_tile_visualizations(tile_data, CONFIG['output_dir'], CONFIG)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {CONFIG['output_dir']}")
    print("\nGenerated files:")
    print("  - density_results_tile_level.csv (6 density methods per tile)")
    print("  - density_results_image_summary.csv (summary statistics)")
    print("  - EXPERIMENT_INFO.json")
    print("\nGenerated boxplots (12 total - 6 methods × 2 dilution ranges):")
    print("  1. Threshold 0.2:")
    print("     - density_boxplot_full_range_threshold_0.2.png")
    print("     - density_boxplot_low_dilution_range_threshold_0.2.png")
    print("  2. Threshold 0.5:")
    print("     - density_boxplot_full_range_threshold_0.5.png")
    print("     - density_boxplot_low_dilution_range_threshold_0.5.png")
    print("  3. Threshold 0.8:")
    print("     - density_boxplot_full_range_threshold_0.8.png")
    print("     - density_boxplot_low_dilution_range_threshold_0.8.png")
    print("  4. Threshold 0.95:")
    print("     - density_boxplot_full_range_threshold_0.95.png")
    print("     - density_boxplot_low_dilution_range_threshold_0.95.png")
    print("  5. CLAHE+Otsu on predicted mask (denoised):")
    print("     - density_boxplot_full_range_claheotsu_on_pred.png")
    print("     - density_boxplot_low_dilution_range_claheotsu_on_pred.png")
    print("  6. CLAHE+Otsu on original image (baseline):")
    print("     - density_boxplot_full_range_claheotsu_on_original.png")
    print("     - density_boxplot_low_dilution_range_claheotsu_on_original.png")
    print("\nTile visualizations:")
    print("  - representative_tiles_3panel/ (5 tiles per image, 3 panels each)")
    print("      * Panel 1: Original tile (dark beads)")
    print("      * Panel 2: Inverted Predicted mask threshold 0.5 (WHITE beads)")
    print("      * Panel 3: Inverted CLAHE+Otsu (WHITE beads, denoised)")
    print("="*80)

if __name__ == '__main__':
    main()
