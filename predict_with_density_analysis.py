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
- Exports a single comprehensive CSV with per-tile density data for all images

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
# Ensure these files are in the same directory or Python path
# from model_architectures import get_model, UNet, ResUNet, AttentionResUNet
# from loss_functions import get_loss_function, jacard_coef, dice_coef

# Dummy placeholders if the above files are not available
# This allows the script to be parsed, but it will fail on run.
def get_model(model_name, input_shape, NUM_CLASSES, dropout_rate, batch_norm): return None
def get_loss_function(name): return None
def jacard_coef(): return None
def dice_coef(): return None

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
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_size)
    clahe_img = clahe.apply(img_gray)

    _, binary_mask = cv2.threshold(
        clahe_img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    density = float((binary_mask > 0).sum()) / binary_mask.size
    return binary_mask, density


def find_best_model_for_architecture(models_dir, architecture):
    """
    Find best model file for given architecture based on a priority list.
    """
    models_path = Path(models_dir)
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
    """Load model with custom loss functions, with a fallback to rebuild."""
    print(f"Loading {architecture} model from: {model_path}")
    custom_objects = {
        'jacard_coef': jacard_coef, 'dice_coef': dice_coef,
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
        print(f"✗ Failed to load {architecture} model directly: {e}")
        print("  Attempting to rebuild architecture and load weights...")
        input_shape = (CONFIG['img_height'], CONFIG['img_width'], CONFIG['img_channels'])
        model = get_model(
            model_name=architecture, input_shape=input_shape,
            NUM_CLASSES=1, dropout_rate=0.3, batch_norm=True
        )
        model.load_weights(model_path)
        print(f"✓ {architecture} weights loaded successfully into rebuilt model")
        return model


def load_and_preprocess_image(image_path, rescale_full=True):
    """Load, rescale, and normalize a test image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if rescale_full:
        img = rescale_image_full_range(img)
    img_original = img.copy()
    img_normalized = img.astype(np.float32) / 255.0
    return img_original, img_normalized


def extract_tiles_512(image, tile_size=512, overlap=0):
    """Extract square tiles from a larger image, padding if necessary."""
    h, w = image.shape[:2]
    stride = tile_size - overlap
    tiles, positions = [], []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end, x_end = min(y + tile_size, h), min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]

            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                padding = ((0, pad_h), (0, pad_w))
                if image.ndim == 3:
                    padding += ((0, 0),)
                tile = np.pad(tile, padding, mode='reflect')

            tiles.append(tile)
            positions.append((y, x))
    return tiles, positions


def predict_on_tiles(model, tiles):
    """Run model prediction on a list of image tiles."""
    predictions = []
    for tile in tiles:
        tile_input = tile[np.newaxis, ..., np.newaxis]
        pred = model.predict(tile_input, verbose=0)
        pred_mask = (pred[0, ..., 0] > 0.5).astype(np.uint8) * 255
        predictions.append(pred_mask)
    return predictions


def create_density_boxplot(densities_dict, test_image_name, output_path):
    """Create and save a boxplot comparing density distributions."""
    plot_data = [{'Architecture': arch, 'Density': density}
                 for arch, densities in densities_dict.items()
                 for density in densities]
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='Architecture', y='Density', palette='Set2', ax=ax, width=0.6)
    sns.swarmplot(data=df, x='Architecture', y='Density', color='black', alpha=0.3, size=3, ax=ax)

    ax.set_title(f'Particle Density Distribution - {test_image_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Particle Density (fraction)', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    stats_text = []
    for arch in ['clahe_otsu', 'unet', 'resunet', 'attention_resunet']:
        if arch in densities_dict:
            data = densities_dict[arch]
            stats_text.append(f'{arch}: μ={np.mean(data):.4f}, σ={np.std(data):.4f}')

    ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved density boxplot: {output_path}")


def process_test_image(image_path, models, output_dirs):
    """
    Process a single test image: predict, calculate densities, and collect data.

    Returns:
        results (dict): Aggregated statistics for the image.
        densities_dict (dict): Lists of densities for boxplot generation.
        individual_tile_data (list): Per-tile data for the comprehensive CSV.
    """
    image_name = Path(image_path).stem
    print(f"\nProcessing: {image_name}")

    individual_tile_data = []

    img_original, img_normalized = load_and_preprocess_image(image_path)
    tiles_original, positions = extract_tiles_512(img_original)
    tiles_normalized, _ = extract_tiles_512(img_normalized)
    print(f"  Extracted {len(tiles_original)} tiles (512x512)")

    # --- CLAHE+OTSU (Baseline) ---
    densities_clahe_otsu = []
    for i, tile_orig in enumerate(tiles_original):
        _, density = apply_clahe_otsu(
            tile_orig, clip=CONFIG['clahe']['clipLimit'], tile_size=CONFIG['clahe']['tileGridSize']
        )
        densities_clahe_otsu.append(density)
        pos = positions[i]
        individual_tile_data.append({
            'image': image_name, 'tile_index': i, 'pos_y': pos[0], 'pos_x': pos[1],
            'method': 'clahe_otsu', 'density': density
        })

    densities_dict = {'clahe_otsu': densities_clahe_otsu}
    results = {}

    # --- Model Predictions ---
    for arch_name, model in models.items():
        print(f"  Predicting with {arch_name}...")
        predictions = predict_on_tiles(model, tiles_normalized)
        densities_pred = []
        for i, pred_tile in enumerate(predictions):
            density = float((pred_tile > 0).sum()) / pred_tile.size
            densities_pred.append(density)
            pos = positions[i]
            individual_tile_data.append({
                'image': image_name, 'tile_index': i, 'pos_y': pos[0], 'pos_x': pos[1],
                'method': arch_name, 'density': density
            })

        densities_dict[arch_name] = densities_pred

        for idx, (pred_mask, pos) in enumerate(zip(predictions, positions)):
            mask_filename = f"{image_name}_tile{idx:03d}_y{pos[0]}_x{pos[1]}.png"
            mask_path = output_dirs[f'masks_{arch_name}'] / mask_filename
            cv2.imwrite(str(mask_path), pred_mask)

    # --- Aggregate and Finalize ---
    for method, data in densities_dict.items():
        results[method] = {
            'mean_density': np.mean(data), 'std_density': np.std(data),
            'median_density': np.median(data), 'min_density': np.min(data),
            'max_density': np.max(data), 'num_tiles': len(data)
        }

    boxplot_path = output_dirs['boxplots'] / f"{image_name}_density_comparison.png"
    create_density_boxplot(densities_dict, image_name, boxplot_path)

    return results, densities_dict, individual_tile_data


def save_summary_report(all_results, output_dir):
    """Save aggregated summary statistics report."""
    summary_path = Path(output_dir) / 'summary' / 'density_analysis_summary.csv'
    rows = [{'image': image_name, 'architecture': arch_name, **stats}
            for image_name, results in all_results.items()
            for arch_name, stats in results.items()]
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary report: {summary_path}")

    print("\n" + "="*80 + "\nSUMMARY STATISTICS\n" + "="*80)
    for arch in ['clahe_otsu', 'unet', 'resunet', 'attention_resunet']:
        arch_data = df[df['architecture'] == arch]
        if not arch_data.empty:
            print(f"\n{arch.upper()}")
            print(f"  Mean density: {arch_data['mean_density'].mean():.4f} ± {arch_data['mean_density'].std():.4f}")

    # --- Overall Comparison Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Overall Density Analysis - All Test Images', fontsize=16, fontweight='bold')
    metrics = ['mean_density', 'std_density', 'median_density', 'max_density']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        sns.barplot(data=df, x='architecture', y=metric, palette='Set2', ax=ax, errorbar='sd')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    overall_plot_path = Path(output_dir) / 'summary' / 'overall_density_comparison.png'
    plt.savefig(overall_plot_path, dpi=CONFIG['dpi'])
    plt.close()
    print(f"✓ Saved overall comparison plot: {overall_plot_path}")


def main():
    """Main execution function."""
    print("="*80 + "\nPREDICTION AND DENSITY ANALYSIS\n" + "="*80)
    print(f"Test images directory: {CONFIG['test_images_dir']}")
    print(f"Models directory: {CONFIG['models_dir']}")
    print(f"Output directory: {CONFIG['output_dir']}\n" + "="*80)

    output_dirs = create_output_dirs(CONFIG['output_dir'])

    # --- Load Models ---
    print("\nLoading trained models...")
    models = {}
    for arch_name in ['unet', 'resunet', 'attention_resunet']:
        model_path = find_best_model_for_architecture(CONFIG['models_dir'], arch_name)
        if model_path:
            try:
                models[arch_name] = load_model_with_custom_objects(str(model_path), arch_name)
            except Exception as e:
                print(f"  ✗ FATAL: Could not load {arch_name} model: {e}")
    if not models:
        print("\n✗ No models were loaded successfully. Exiting.")
        sys.exit(1)
    print(f"\n✓ Loaded {len(models)} models: {list(models.keys())}")

    # --- Find Test Images ---
    test_images_path = Path(CONFIG['test_images_dir'])
    test_images = sorted(list(test_images_path.glob('*.tif*')) + list(test_images_path.glob('*.png')))
    if not test_images:
        print(f"\n✗ No test images found in {test_images_path}. Exiting.")
        sys.exit(1)
    print(f"\n✓ Found {len(test_images)} test images.")

    # --- Process Images and Collect All Tile Data ---
    all_results = {}
    all_tile_data = []  # Master list for all per-tile data from all images

    for image_path in test_images:
        try:
            results, _, tile_data_for_image = process_test_image(image_path, models, output_dirs)
            all_results[image_path.stem] = results
            all_tile_data.extend(tile_data_for_image)
        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # --- Save Reports ---
    if all_results:
        save_summary_report(all_results, CONFIG['output_dir'])

    if all_tile_data:
        print("\nSaving comprehensive per-tile density report...")
        comprehensive_df = pd.DataFrame(all_tile_data)
        comprehensive_csv_path = output_dirs['summary'] / 'comprehensive_tile_densities.csv'
        comprehensive_df.to_csv(comprehensive_csv_path, index=False)
        print(f"✓ Saved comprehensive tile data: {comprehensive_csv_path}")

    print("\n" + "="*80 + "\nANALYSIS COMPLETE\n" + "="*80)
    print(f"Results saved to: {CONFIG['output_dir']}")
    print(f"  - Predicted masks: {output_dirs['masks']}")
    print(f"  - Density boxplots: {output_dirs['boxplots']}")
    print(f"  - Summary reports: {output_dirs['summary']}")
    print("="*80)


if __name__ == '__main__':
    main()
