#!/usr/bin/env python3
"""
Prediction and Density Analysis Script
========================================
Performs prediction using trained U-Net, ResU-Net, and Attention ResU-Net models
on test images and calculates particle density using CLAHE+OTSU method.

Features:
- Loads best models for each architecture (BS=8, combined_tversky)
- Predicts on 512x512 tiles from test images
- Calculates density for both predicted masks and ground truth
- Generates boxplot comparison for each test image
- Exports results and visualizations

Usage:
    python predict_with_density_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

# Import model architectures and loss functions
from model_architectures import get_model, UNet, ResUNet, AttentionResUNet
from loss_functions import get_loss_function, jacard_coef, dice_coef

# Configuration
CONFIG = {
    'test_images_dir': './test_images',
    'models_dir': './hyperparam_comprehensive_20251012_005054',
    'output_dir': f'./prediction_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    'img_height': 512,
    'img_width': 512,
    'img_channels': 1,

    # Best models (BS=8, look for combined_tversky or combined from search results)
    # Will search for best available model for each architecture
    'models': {
        'unet': None,  # Will be auto-detected
        'resunet': None,  # Will be auto-detected
        'attention_resunet': None  # Will be auto-detected
    },

    # CLAHE parameters (from Particle-density-calculation.py)
    'clahe': {
        'clipLimit': 2.0,
        'tileGridSize': (8, 8)
    },

    # Visualization
    'figsize': (15, 10),
    'dpi': 150
}


def create_output_dirs(base_dir):
    """Create output directory structure"""
    base = Path(base_dir)
    subdirs = {
        'masks': base / 'predicted_masks',
        'density_maps': base / 'density_maps',
        'boxplots': base / 'boxplots',
        'summary': base / 'summary'
    }

    for arch in ['unet', 'resunet', 'attention_resunet']:
        subdirs[f'masks_{arch}'] = subdirs['masks'] / arch
        subdirs[f'masks_{arch}'].mkdir(parents=True, exist_ok=True)

    for key, path in subdirs.items():
        if not key.startswith('masks_'):
            path.mkdir(parents=True, exist_ok=True)

    print(f"✓ Created output directory: {base}")
    return subdirs


def rescale_image_full_range(img):
    """
    Rescale image to full 0-255 range
    (matching Particle-density-calculation.py approach)
    """
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()

    if img_max - img_min > 0:
        img_rescaled = 255.0 * (img - img_min) / (img_max - img_min)
    else:
        img_rescaled = img.copy()

    return img_rescaled.astype(np.uint8)


def apply_clahe_otsu(img_gray, clip=2.0, tile_size=(8, 8)):
    """
    Apply CLAHE + OTSU thresholding for density calculation
    Following Particle-density-calculation.py methodology

    Args:
        img_gray: Grayscale image (already rescaled to 0-255)
        clip: CLAHE clip limit
        tile_size: CLAHE tile grid size

    Returns:
        binary_mask: Binary mask (255=particle, 0=background)
        density: Fraction of particle pixels (0-1)
    """
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_size)
    clahe_img = clahe.apply(img_gray)

    # Apply OTSU threshold (inverse: white=particles)
    _, binary_mask = cv2.threshold(
        clahe_img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Calculate density (fraction of particle pixels)
    density = float((binary_mask > 0).sum()) / binary_mask.size

    return binary_mask, density


def find_best_model_for_architecture(models_dir, architecture):
    """
    Find best model file for given architecture

    Priority:
    1. BS=8, combined_tversky
    2. BS=8, combined
    3. BS=8, any loss
    4. Any model with this architecture
    """
    models_path = Path(models_dir)

    # Search patterns in order of priority
    patterns = [
        f"model_{architecture}_bs8_dr0.3_combined_tversky.hdf5",
        f"model_{architecture}_bs8_dr0.3_combined.hdf5",
        f"model_{architecture}_bs8_dr0.3_*.hdf5",
        f"model_{architecture}_bs*_dr*_*.hdf5"
    ]

    for pattern in patterns:
        matches = list(models_path.glob(pattern))
        if matches:
            print(f"  Found {architecture} model: {matches[0].name}")
            return matches[0]

    print(f"  ⚠ No model found for {architecture}")
    return None


def load_model_with_custom_objects(model_path, architecture):
    """Load model with custom loss functions"""
    print(f"Loading {architecture} model from: {model_path}")

    # Custom objects for loading (support all loss functions and metrics)
    custom_objects = {
        'jacard_coef': jacard_coef,
        'dice_coef': dice_coef,
        'combined_tversky_focal_loss': get_loss_function('combined_tversky'),
        'combined_dice_focal_loss': get_loss_function('combined'),
        'focal_loss': get_loss_function('focal'),
        'focal_tversky_loss': get_loss_function('focal_tversky'),
        'tversky_loss': get_loss_function('tversky')
    }

    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"✓ {architecture} model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load {architecture} model: {e}")
        print(f"  Rebuilding model architecture and loading weights...")

        # Rebuild architecture using get_model()
        img_height = CONFIG['img_height']
        img_width = CONFIG['img_width']
        img_channels = CONFIG['img_channels']
        input_shape = (img_height, img_width, img_channels)

        model = get_model(
            model_name=architecture,
            input_shape=input_shape,
            NUM_CLASSES=1,
            dropout_rate=0.3,
            batch_norm=True
        )

        # Load weights only
        model.load_weights(model_path)
        print(f"✓ {architecture} weights loaded successfully")
        return model


def load_and_preprocess_image(image_path, rescale_full=True):
    """
    Load and preprocess test image

    Args:
        image_path: Path to image file
        rescale_full: If True, rescale to full 0-255 range first

    Returns:
        img_original: Original image (grayscale, 0-255)
        img_normalized: Normalized image for model prediction (0-1)
    """
    # Load as grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Rescale to full 0-255 range (for consistent density calculation)
    if rescale_full:
        img = rescale_image_full_range(img)

    img_original = img.copy()

    # Normalize to 0-1 for model prediction
    img_normalized = img.astype(np.float32) / 255.0

    return img_original, img_normalized


def extract_tiles_512(image, tile_size=512, overlap=0):
    """
    Extract 512x512 tiles from large image

    Args:
        image: Input image (H, W) or (H, W, C)
        tile_size: Size of square tiles
        overlap: Overlap between tiles in pixels

    Returns:
        tiles: List of tile images
        positions: List of (y, x) top-left positions
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap

    tiles = []
    positions = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Ensure we don't go out of bounds
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            # Extract tile
            if len(image.shape) == 2:
                tile = image[y:y_end, x:x_end]
            else:
                tile = image[y:y_end, x:x_end, :]

            # Pad if necessary to reach 512x512
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                if len(image.shape) == 2:
                    pad_h = tile_size - tile.shape[0]
                    pad_w = tile_size - tile.shape[1]
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='reflect')
                else:
                    pad_h = tile_size - tile.shape[0]
                    pad_w = tile_size - tile.shape[1]
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions


def predict_on_tiles(model, tiles):
    """
    Run prediction on all tiles

    Args:
        model: Trained model
        tiles: List of normalized tiles (512, 512)

    Returns:
        predictions: List of predicted masks (512, 512)
    """
    predictions = []

    for tile in tiles:
        # Add batch and channel dimensions
        tile_input = tile[np.newaxis, :, :, np.newaxis]  # (1, 512, 512, 1)

        # Predict
        pred = model.predict(tile_input, verbose=0)

        # Remove batch dimension and threshold
        pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255

        predictions.append(pred_mask)

    return predictions


def calculate_densities_for_tiles(tiles_original, tiles_predicted):
    """
    Calculate densities for original (CLAHE+OTSU) and predicted tiles

    Args:
        tiles_original: List of original tiles (0-255, grayscale)
        tiles_predicted: List of predicted masks (0 or 255)

    Returns:
        densities_original: List of densities from CLAHE+OTSU
        densities_predicted: List of densities from predicted masks
    """
    densities_original = []
    densities_predicted = []

    for orig_tile, pred_tile in zip(tiles_original, tiles_predicted):
        # Original: CLAHE + OTSU
        _, density_orig = apply_clahe_otsu(
            orig_tile,
            clip=CONFIG['clahe']['clipLimit'],
            tile_size=CONFIG['clahe']['tileGridSize']
        )
        densities_original.append(density_orig)

        # Predicted: direct calculation
        density_pred = float((pred_tile > 0).sum()) / pred_tile.size
        densities_predicted.append(density_pred)

    return densities_original, densities_predicted


def create_density_boxplot(densities_dict, test_image_name, output_path):
    """
    Create boxplot comparing densities across architectures

    Args:
        densities_dict: Dict with keys ['unet', 'resunet', 'attention_resunet', 'clahe_otsu']
                       each containing list of tile densities
        test_image_name: Name of test image
        output_path: Path to save plot
    """
    # Prepare data for plotting
    plot_data = []

    for arch, densities in densities_dict.items():
        for density in densities:
            plot_data.append({
                'Architecture': arch,
                'Density': density
            })

    df = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Boxplot with swarm overlay
    sns.boxplot(
        data=df, x='Architecture', y='Density',
        palette='Set2', ax=ax, width=0.6
    )
    sns.swarmplot(
        data=df, x='Architecture', y='Density',
        color='black', alpha=0.3, size=3, ax=ax
    )

    # Formatting
    ax.set_title(f'Particle Density Distribution - {test_image_name}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Particle Density (fraction)', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics
    stats_text = []
    for arch in ['clahe_otsu', 'unet', 'resunet', 'attention_resunet']:
        if arch in densities_dict:
            data = densities_dict[arch]
            mean_val = np.mean(data)
            std_val = np.std(data)
            stats_text.append(f'{arch}: μ={mean_val:.4f}, σ={std_val:.4f}')

    ax.text(0.02, 0.98, '\n'.join(stats_text),
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved density boxplot: {output_path}")


def process_test_image(image_path, models, output_dirs):
    """
    Process single test image with all architectures

    Args:
        image_path: Path to test image
        models: Dict of loaded models {arch_name: model}
        output_dirs: Dict of output directories

    Returns:
        results: Dict with density statistics
    """
    image_name = Path(image_path).stem
    print(f"\nProcessing: {image_name}")

    # Load and preprocess
    img_original, img_normalized = load_and_preprocess_image(image_path)
    print(f"  Image shape: {img_original.shape}")

    # Extract 512x512 tiles
    tiles_original, positions = extract_tiles_512(img_original, tile_size=512)
    tiles_normalized, _ = extract_tiles_512(img_normalized, tile_size=512)

    print(f"  Extracted {len(tiles_original)} tiles (512x512)")

    # Calculate CLAHE+OTSU densities (ground truth method)
    densities_clahe_otsu = []
    for tile_orig in tiles_original:
        _, density = apply_clahe_otsu(
            tile_orig,
            clip=CONFIG['clahe']['clipLimit'],
            tile_size=CONFIG['clahe']['tileGridSize']
        )
        densities_clahe_otsu.append(density)

    # Process with each architecture
    densities_dict = {'clahe_otsu': densities_clahe_otsu}
    results = {}

    for arch_name, model in models.items():
        print(f"  Predicting with {arch_name}...")

        # Predict on tiles
        predictions = predict_on_tiles(model, tiles_normalized)

        # Calculate densities from predictions
        densities_pred = []
        for pred_tile in predictions:
            density = float((pred_tile > 0).sum()) / pred_tile.size
            densities_pred.append(density)

        densities_dict[arch_name] = densities_pred

        # Save predicted masks
        for idx, (pred_mask, pos) in enumerate(zip(predictions, positions)):
            mask_filename = f"{image_name}_tile{idx:03d}_y{pos[0]}_x{pos[1]}.png"
            mask_path = output_dirs[f'masks_{arch_name}'] / mask_filename
            cv2.imwrite(str(mask_path), pred_mask)

        # Store results
        results[arch_name] = {
            'mean_density': np.mean(densities_pred),
            'std_density': np.std(densities_pred),
            'median_density': np.median(densities_pred),
            'min_density': np.min(densities_pred),
            'max_density': np.max(densities_pred),
            'num_tiles': len(densities_pred)
        }

    # Add CLAHE+OTSU results
    results['clahe_otsu'] = {
        'mean_density': np.mean(densities_clahe_otsu),
        'std_density': np.std(densities_clahe_otsu),
        'median_density': np.median(densities_clahe_otsu),
        'min_density': np.min(densities_clahe_otsu),
        'max_density': np.max(densities_clahe_otsu),
        'num_tiles': len(densities_clahe_otsu)
    }

    # Create boxplot
    boxplot_path = output_dirs['boxplots'] / f"{image_name}_density_comparison.png"
    create_density_boxplot(densities_dict, image_name, boxplot_path)

    return results, densities_dict


def save_summary_report(all_results, output_dir):
    """
    Save comprehensive summary report

    Args:
        all_results: Dict {image_name: {arch_name: stats}}
        output_dir: Output directory path
    """
    summary_path = Path(output_dir) / 'summary' / 'density_analysis_summary.csv'

    # Flatten results into DataFrame
    rows = []
    for image_name, results in all_results.items():
        for arch_name, stats in results.items():
            row = {
                'image': image_name,
                'architecture': arch_name,
                **stats
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)

    print(f"\n✓ Saved summary report: {summary_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for arch in ['clahe_otsu', 'unet', 'resunet', 'attention_resunet']:
        arch_data = df[df['architecture'] == arch]
        if len(arch_data) > 0:
            print(f"\n{arch.upper()}")
            print(f"  Mean density: {arch_data['mean_density'].mean():.4f} ± {arch_data['mean_density'].std():.4f}")
            print(f"  Median density: {arch_data['median_density'].mean():.4f}")
            print(f"  Range: [{arch_data['min_density'].min():.4f}, {arch_data['max_density'].max():.4f}]")

    # Create overall comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Overall Density Analysis - All Test Images',
                 fontsize=16, fontweight='bold')

    metrics = ['mean_density', 'std_density', 'median_density', 'max_density']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        sns.barplot(data=df, x='architecture', y=metric,
                   palette='Set2', ax=ax, errorbar='sd')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Architecture', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    overall_plot_path = Path(output_dir) / 'summary' / 'overall_density_comparison.png'
    plt.savefig(overall_plot_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()

    print(f"✓ Saved overall comparison: {overall_plot_path}")


def main():
    print("="*80)
    print("PREDICTION AND DENSITY ANALYSIS")
    print("="*80)
    print(f"Test images directory: {CONFIG['test_images_dir']}")
    print(f"Models directory: {CONFIG['models_dir']}")
    print(f"Output directory: {CONFIG['output_dir']}")
    print("="*80)

    # Create output directories
    output_dirs = create_output_dirs(CONFIG['output_dir'])

    # Load models
    print("\nLoading trained models...")
    models = {}

    for arch_name in ['unet', 'resunet', 'attention_resunet']:
        print(f"\nSearching for {arch_name} model...")

        # Find best available model
        model_path = find_best_model_for_architecture(CONFIG['models_dir'], arch_name)

        if model_path is None:
            print(f"  ✗ No {arch_name} model found. Skipping.")
            continue

        # Load model
        try:
            models[arch_name] = load_model_with_custom_objects(str(model_path), arch_name)
        except Exception as e:
            print(f"  ✗ Failed to load {arch_name}: {e}")
            continue

    if not models:
        print("✗ No models loaded. Exiting.")
        sys.exit(1)

    print(f"\n✓ Loaded {len(models)} models: {list(models.keys())}")

    # Find test images
    test_images_path = Path(CONFIG['test_images_dir'])
    if not test_images_path.exists():
        print(f"✗ Test images directory not found: {test_images_path}")
        sys.exit(1)

    test_images = sorted(test_images_path.glob('*.tif')) + \
                  sorted(test_images_path.glob('*.tiff')) + \
                  sorted(test_images_path.glob('*.png'))

    if not test_images:
        print(f"✗ No test images found in {test_images_path}")
        sys.exit(1)

    print(f"\n✓ Found {len(test_images)} test images")
    for img_path in test_images:
        print(f"  - {img_path.name}")

    # Process each test image
    all_results = {}
    all_densities = {}

    for image_path in test_images:
        try:
            results, densities = process_test_image(image_path, models, output_dirs)
            all_results[image_path.stem] = results
            all_densities[image_path.stem] = densities
        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary report
    if all_results:
        save_summary_report(all_results, CONFIG['output_dir'])

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {CONFIG['output_dir']}")
    print(f"  - Predicted masks: {output_dirs['masks']}")
    print(f"  - Density boxplots: {output_dirs['boxplots']}")
    print(f"  - Summary report: {output_dirs['summary']}")
    print("="*80)


if __name__ == '__main__':
    main()
