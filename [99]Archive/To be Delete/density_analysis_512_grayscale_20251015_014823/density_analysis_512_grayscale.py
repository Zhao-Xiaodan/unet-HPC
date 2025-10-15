#!/usr/bin/env python3
"""
Density Analysis Using Best 512×512 Grayscale Models
=====================================================

Uses the top 5 configurations from hyperparameter_search_512_20251014_235755
to predict on test images and generate density analysis.

Generates:
- Box plots of density vs dilution factor (all models combined)
- 5 representative 512×512 tiles with side-by-side predictions from all models
- CSV with all density data

Models used (from hyperparameter_search_512_20251014_235755):
1. unet_lr0.0001_drop0.3_bs4 (Best: 0.1533 ± 0.0578)
2. unet_lr5e-05_drop0.2_bs4 (0.1327 ± 0.0176)
3. unet_lr5e-05_drop0.3_bs4 (0.1308 ± 0.0137)
4. resunet_lr5e-05_drop0.3_bs4 (0.1117 ± 0.0131)
5. attention_resunet_lr5e-05_drop0.2_bs4 (0.1091 ± 0.0064)

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

import tensorflow as tf
from tensorflow import keras

# Import custom modules
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef

# ============================================================================
# CONFIGURATION
# ============================================================================

# Top 5 configurations from hyperparameter search
TOP_CONFIGS = [
    {
        'name': 'unet_lr0.0001_drop0.3_bs4',
        'architecture': 'unet',
        'lr': 0.0001,
        'dropout': 0.3,
        'mean_jaccard': 0.1533,
        'std_jaccard': 0.0578,
        'color': '#440154',  # Dark purple
        'label': 'U-Net (best)',
    },
    {
        'name': 'unet_lr5e-05_drop0.2_bs4',
        'architecture': 'unet',
        'lr': 5e-05,
        'dropout': 0.2,
        'mean_jaccard': 0.1327,
        'std_jaccard': 0.0176,
        'color': '#31688e',  # Blue
        'label': 'U-Net (lr5e-05)',
    },
    {
        'name': 'unet_lr5e-05_drop0.3_bs4',
        'architecture': 'unet',
        'lr': 5e-05,
        'dropout': 0.3,
        'mean_jaccard': 0.1308,
        'std_jaccard': 0.0137,
        'color': '#35b779',  # Green
        'label': 'U-Net (d0.3)',
    },
    {
        'name': 'resunet_lr5e-05_drop0.3_bs4',
        'architecture': 'resunet',
        'lr': 5e-05,
        'dropout': 0.3,
        'mean_jaccard': 0.1117,
        'std_jaccard': 0.0131,
        'color': '#fde724',  # Yellow
        'label': 'ResUNet',
    },
    {
        'name': 'attention_resunet_lr5e-05_drop0.2_bs4',
        'architecture': 'attention_resunet',
        'lr': 5e-05,
        'dropout': 0.2,
        'mean_jaccard': 0.1091,
        'std_jaccard': 0.0064,
        'color': '#b5de2b',  # Light yellow-green
        'label': 'Att-ResUNet',
    }
]

CONFIG = {
    # Directories
    'model_dir': './hyperparameter_search_512_20251014_235755',
    'test_images_dir': './test_images',
    'output_dir': f'./density_analysis_512_grayscale_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Image settings (from training)
    'img_size': 512,
    'img_channels': 1,  # Grayscale

    # Prediction settings
    'batch_size': 4,
    'threshold': 0.5,  # Binary threshold for predictions

    # Dilution factors to analyze
    'dilution_factors': [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240],

    # Representative tiles for visualization
    'num_representative_tiles': 5,

    # Plotting
    'dpi': 300,
}

# Dilution factor patterns (for extracting from filenames)
DILUTION_PATTERNS = {
    '10x': 10, '20x': 20, '40x': 40, '80x': 80, '160x': 160,
    '320x': 320, '640x': 640, '1280x': 1280, '2560x': 2560,
    '5120x': 5120, '10240x': 10240
}

print("="*80)
print("DENSITY ANALYSIS - 512×512 GRAYSCALE MODELS")
print("="*80)
print(f"Script: density_analysis_512_grayscale.py")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model directory: {CONFIG['model_dir']}")
print(f"Test images: {CONFIG['test_images_dir']}")
print(f"Output directory: {CONFIG['output_dir']}")
print()

print("Models to be used (Top 5 from hyperparameter search):")
for i, config in enumerate(TOP_CONFIGS, 1):
    print(f"  {i}. {config['label']}: {config['mean_jaccard']:.4f} ± {config['std_jaccard']:.4f}")
print("="*80)
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_dilution_from_filename(filename):
    """Extract dilution factor from filename."""
    for pattern, dilution in DILUTION_PATTERNS.items():
        if pattern in filename.lower():
            return dilution
    return None

def load_image_grayscale(image_path, img_size=512):
    """Load and preprocess image to grayscale."""
    from PIL import Image

    img = Image.open(image_path).convert('L')  # Convert to grayscale
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BILINEAR)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = img_array[..., np.newaxis]  # Add channel dimension

    return img_array

def calculate_foreground_percentage(mask, threshold=0.5):
    """Calculate percentage of foreground pixels."""
    binary_mask = (mask > threshold).astype(np.float32)
    foreground_pixels = np.sum(binary_mask)
    total_pixels = mask.size
    percentage = (foreground_pixels / total_pixels) * 100
    return percentage

def load_model_for_config(config_name, model_dir, fold=1):
    """Load trained model for a specific configuration and fold."""
    model_pattern = f"{config_name.replace('_lr', f'_fold{fold}_lr')}_model.keras"
    model_path = Path(model_dir) / model_pattern

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"  Loading: {model_path.name}")

    # Load with custom objects
    model = keras.models.load_model(
        str(model_path),
        custom_objects={
            'combined_dice_focal_loss': combined_dice_focal_loss,
            'jacard_coef': jacard_coef,
            'dice_coef': dice_coef
        }
    )

    return model

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_density_analysis():
    """Run complete density analysis."""

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = output_dir / 'csv_data'
    csv_dir.mkdir(exist_ok=True)

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    tiles_dir = output_dir / 'representative_tiles'
    tiles_dir.mkdir(exist_ok=True)

    # Load test images
    print("\n" + "="*80)
    print("LOADING TEST IMAGES")
    print("="*80)

    test_dir = Path(CONFIG['test_images_dir'])
    image_files = sorted(list(test_dir.glob('*.tif')) + list(test_dir.glob('*.tiff')))

    if len(image_files) == 0:
        raise ValueError(f"No test images found in {test_dir}")

    print(f"Found {len(image_files)} test images")

    # Group images by dilution
    images_by_dilution = {}
    for img_path in image_files:
        dilution = extract_dilution_from_filename(img_path.name)
        if dilution is not None:
            if dilution not in images_by_dilution:
                images_by_dilution[dilution] = []
            images_by_dilution[dilution].append(img_path)

    print(f"Dilution factors found: {sorted(images_by_dilution.keys())}")
    for dilution in sorted(images_by_dilution.keys()):
        print(f"  {dilution}x: {len(images_by_dilution[dilution])} images")
    print()

    # Load all models (use fold 1 for each config)
    print("="*80)
    print("LOADING MODELS")
    print("="*80)

    models = {}
    for config in TOP_CONFIGS:
        try:
            model = load_model_for_config(config['name'], CONFIG['model_dir'], fold=1)
            models[config['name']] = {
                'model': model,
                'config': config
            }
            print(f"  ✓ Loaded {config['label']}")
        except FileNotFoundError as e:
            print(f"  ✗ Failed to load {config['label']}: {e}")

    if len(models) == 0:
        raise ValueError("No models loaded successfully!")

    print(f"\n✓ Successfully loaded {len(models)} models")
    print()

    # Run predictions on all images
    print("="*80)
    print("RUNNING PREDICTIONS")
    print("="*80)

    all_results = []
    representative_tiles = []  # Store tiles for visualization

    for dilution in tqdm(sorted(images_by_dilution.keys()), desc="Dilution factors"):
        for img_idx, img_path in enumerate(tqdm(images_by_dilution[dilution],
                                                  desc=f"  {dilution}x images",
                                                  leave=False)):
            # Load image
            img = load_image_grayscale(img_path, CONFIG['img_size'])
            img_batch = np.expand_dims(img, axis=0)

            # Store original image for visualization
            img_for_viz = (img.squeeze() * 255).astype(np.uint8)

            # Predict with each model
            predictions = {}
            densities = {}

            for model_name, model_data in models.items():
                pred = model_data['model'].predict(img_batch, verbose=0)
                pred_mask = pred[0, :, :, 0]  # Remove batch and channel dims

                predictions[model_name] = pred_mask
                densities[model_name] = calculate_foreground_percentage(
                    pred_mask, CONFIG['threshold']
                )

                # Record result
                all_results.append({
                    'image': img_path.name,
                    'dilution_factor': dilution,
                    'inverse_dilution': 1.0 / dilution,
                    'model': model_data['config']['label'],
                    'model_name': model_name,
                    'architecture': model_data['config']['architecture'],
                    'foreground_percentage': densities[model_name],
                    'mean_jaccard': model_data['config']['mean_jaccard'],
                })

            # Save representative tiles (evenly spaced across dilution range)
            if len(representative_tiles) < CONFIG['num_representative_tiles']:
                # Select tiles from different dilutions
                target_dilutions = [10, 80, 320, 1280, 5120]
                if dilution in target_dilutions and img_idx == 0:
                    representative_tiles.append({
                        'image': img_for_viz,
                        'predictions': predictions,
                        'dilution': dilution,
                        'filename': img_path.name
                    })

    print(f"\n✓ Completed predictions on {len(all_results)} image-model combinations")
    print()

    # Save results to CSV
    print("="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_df = pd.DataFrame(all_results)
    csv_path = csv_dir / 'density_analysis_all_models.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    print()

    # ========================================================================
    # FIGURE 1: Box Plot - Density vs Dilution (All Models)
    # ========================================================================

    print("="*80)
    print("GENERATING FIGURES")
    print("="*80)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for box plot
    dilutions = sorted(results_df['dilution_factor'].unique())
    positions = np.arange(len(dilutions))
    width = 0.15

    # Plot box plots for each model
    for i, config in enumerate(TOP_CONFIGS):
        model_label = config['label']
        model_data = []

        for dilution in dilutions:
            data = results_df[
                (results_df['dilution_factor'] == dilution) &
                (results_df['model'] == model_label)
            ]['foreground_percentage'].values
            model_data.append(data)

        # Create box plot
        bp = ax.boxplot(model_data,
                        positions=positions + (i - 2) * width,
                        widths=width * 0.9,
                        patch_artist=True,
                        showfliers=True,
                        boxprops=dict(facecolor=config['color'], alpha=0.7),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color=config['color']),
                        capprops=dict(color=config['color']))

        # Add to legend
        ax.plot([], [], color=config['color'], linewidth=10, alpha=0.7,
                label=f"{model_label} (J={config['mean_jaccard']:.3f})")

    ax.set_xlabel('Dilution Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Foreground Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Density Analysis: 512×512 Grayscale Models (Top 5 Configurations)',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{d}x' for d in dilutions], rotation=45, ha='right')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plot_path = plots_dir / 'density_vs_dilution_all_models.png'
    plt.savefig(plot_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()

    print(f"✓ Saved Figure 1: {plot_path}")

    # ========================================================================
    # FIGURE 2: Representative Tiles with Predictions
    # ========================================================================

    if len(representative_tiles) > 0:
        n_tiles = len(representative_tiles)
        n_models = len(models)

        fig, axes = plt.subplots(n_tiles, n_models + 1,
                                 figsize=(4 * (n_models + 1), 4 * n_tiles))

        if n_tiles == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Representative 512×512 Tiles: Model Comparison',
                     fontsize=16, fontweight='bold', y=0.995)

        for row, tile_data in enumerate(representative_tiles):
            # Original image
            ax = axes[row, 0]
            ax.imshow(tile_data['image'], cmap='gray')
            ax.set_title(f"Original\n{tile_data['dilution']}x dilution",
                        fontsize=11, fontweight='bold')
            ax.axis('off')

            # Model predictions
            for col, (model_name, model_data) in enumerate(models.items(), 1):
                ax = axes[row, col]
                pred_mask = tile_data['predictions'][model_name]

                # Show prediction as binary mask
                binary_pred = (pred_mask > CONFIG['threshold']).astype(np.uint8) * 255
                ax.imshow(binary_pred, cmap='gray')

                # Calculate density for this tile
                density = calculate_foreground_percentage(pred_mask, CONFIG['threshold'])

                ax.set_title(f"{model_data['config']['label']}\n"
                            f"Density: {density:.2f}%",
                            fontsize=10)
                ax.axis('off')

        plt.tight_layout()
        tiles_path = tiles_dir / 'representative_tiles_comparison.png'
        plt.savefig(tiles_path, dpi=CONFIG['dpi'], bbox_inches='tight')
        plt.close()

        print(f"✓ Saved Figure 2: {tiles_path}")

    # ========================================================================
    # FIGURE 3: Model Performance Comparison
    # ========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (A) Mean density across all dilutions
    ax = axes[0]
    model_means = results_df.groupby('model')['foreground_percentage'].mean().sort_values(ascending=False)
    model_stds = results_df.groupby('model')['foreground_percentage'].std()

    colors = [config['color'] for config in TOP_CONFIGS]
    bars = ax.barh(range(len(model_means)), model_means, xerr=model_stds,
                   color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(model_means)))
    ax.set_yticklabels(model_means.index, fontsize=11)
    ax.set_xlabel('Mean Foreground Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('(A) Average Density Across All Dilutions', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars, model_means, model_stds)):
        ax.text(val + std + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}±{std:.2f}%', va='center', fontsize=10)

    # (B) Correlation with training performance
    ax = axes[1]
    model_names = []
    train_jaccards = []
    pred_densities = []

    for config in TOP_CONFIGS:
        model_label = config['label']
        model_names.append(model_label)
        train_jaccards.append(config['mean_jaccard'])

        mean_density = results_df[results_df['model'] == model_label]['foreground_percentage'].mean()
        pred_densities.append(mean_density)

    for i, (name, jac, dens, color) in enumerate(zip(model_names, train_jaccards,
                                                       pred_densities,
                                                       [c['color'] for c in TOP_CONFIGS])):
        ax.scatter(jac, dens, s=200, color=color, alpha=0.7,
                  edgecolor='black', linewidth=2, label=name)

    ax.set_xlabel('Training Jaccard (3-fold CV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Prediction Density (%)', fontsize=12, fontweight='bold')
    ax.set_title('(B) Training Performance vs Prediction Density', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = plots_dir / 'model_performance_comparison.png'
    plt.savefig(comparison_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()

    print(f"✓ Saved Figure 3: {comparison_path}")

    # Save experiment metadata
    metadata = {
        'experiment_name': 'Density Analysis - 512×512 Grayscale Models',
        'python_script': 'density_analysis_512_grayscale.py',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_source': CONFIG['model_dir'],
        'test_images': CONFIG['test_images_dir'],
        'output_directory': str(output_dir),
        'models_used': [
            {
                'name': config['name'],
                'label': config['label'],
                'architecture': config['architecture'],
                'mean_jaccard': config['mean_jaccard'],
                'std_jaccard': config['std_jaccard']
            }
            for config in TOP_CONFIGS
        ],
        'num_test_images': len(image_files),
        'dilution_factors': sorted(list(images_by_dilution.keys())),
        'total_predictions': len(all_results),
    }

    with open(output_dir / 'EXPERIMENT_INFO.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: EXPERIMENT_INFO.json")
    print()

    print("="*80)
    print("DENSITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Figures: {plots_dir}")
    print(f"CSV data: {csv_dir}")
    print(f"Representative tiles: {tiles_dir}")
    print("="*80)

if __name__ == '__main__':
    run_density_analysis()
