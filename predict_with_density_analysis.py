#!/usr/bin/env python3
"""
Prediction and Density Analysis Script
========================================
Performs prediction using trained U-Net, ResU-Net, and Attention ResU-Net models
on test images and calculates particle density using CLAHE+OTSU method.

Features:
- Loads best models for each architecture.
- Predicts on 512x512 tiles from test images.
- Saves 3 representative masks (min, median, max density) per image/model.
- Generates 4 boxplots, one per method, showing density vs. dilution factor.
- Exports a single comprehensive CSV with per-tile density data for all images.

Usage:
    python predict_with_density_analysis.py
"""
import os
import sys
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

# Dummy placeholders if model architecture files are not available.
# Replace with your actual imports.
# from model_architectures import get_model
# from loss_functions import get_loss_function, jacard_coef, dice_coef
def get_model(model_name, input_shape, NUM_CLASSES, dropout_rate, batch_norm): return None
def get_loss_function(name): return None
def jacard_coef(): return None
def dice_coef(): return None

# --- CONFIGURATION ---
CONFIG = {
    'test_images_dir': './test_images',
    'models_dir': './hyperparam_comprehensive_20251012_005054',
    'output_dir': f'./prediction_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    'img_height': 512,
    'img_width': 512,
    'img_channels': 1,
    'clahe': {'clipLimit': 2.0, 'tileGridSize': (8, 8)},
    'dpi': 200
}

# --- DIRECTORY AND FILE HANDLING ---

def create_output_dirs(base_dir):
    """Create output directory structure."""
    base = Path(base_dir)
    subdirs = {
        'representative_masks': base / 'representative_masks',
        'dilution_boxplots': base / 'dilution_boxplots',
        'summary': base / 'summary'
    }
    for arch in ['unet', 'resunet', 'attention_resunet']:
        # Create a subfolder for each architecture's masks
        (subdirs['representative_masks'] / arch).mkdir(parents=True, exist_ok=True)

    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    print(f"✓ Created output directory: {base}")
    return subdirs

# --- IMAGE PROCESSING ---

def rescale_image_full_range(img):
    """Rescale image to full 0-255 range."""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)

def apply_clahe_otsu(img_gray, clip=2.0, tile_size=(8, 8)):
    """Apply CLAHE + OTSU thresholding to calculate particle density."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_size)
    clahe_img = clahe.apply(img_gray)
    _, binary_mask = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    density = np.count_nonzero(binary_mask) / binary_mask.size
    return binary_mask, density

def load_and_preprocess_image(image_path):
    """Load, rescale, and normalize a test image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_original = rescale_image_full_range(img)
    img_normalized = img_original.astype(np.float32) / 255.0
    return img_original, img_normalized

def extract_tiles_512(image, tile_size=512):
    """Extract square tiles from a larger image, padding if necessary."""
    h, w = image.shape[:2]
    tiles, positions = [], []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y_end, x_end = min(y + tile_size, h), min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]

            pad_h, pad_w = tile_size - tile.shape[0], tile_size - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                padding = ((0, pad_h), (0, pad_w))
                if image.ndim == 3:
                    padding += ((0, 0),)
                tile = np.pad(tile, padding, mode='reflect')

            tiles.append(tile)
            positions.append((y, x))
    return tiles, positions

# --- MODEL LOADING AND PREDICTION ---

def find_best_model_for_architecture(models_dir, architecture):
    """Find the best model file for an architecture based on a priority list."""
    models_path = Path(models_dir)
    patterns = [f"model_{architecture}_bs8_dr0.3_combined_tversky.hdf5",
                f"model_{architecture}_bs8_dr0.3_combined.hdf5",
                f"model_{architecture}_bs8_dr0.3_*.hdf5",
                f"model_{architecture}_bs*_dr*_*.hdf5"]
    for pattern in patterns:
        matches = list(models_path.glob(pattern))
        if matches:
            print(f"  Found {architecture} model: {matches[0].name}")
            return matches[0]
    print(f"  ⚠ No model found for {architecture}")
    return None

def load_model_with_custom_objects(model_path, architecture):
    """Load a Keras model with custom objects and a fallback to rebuild."""
    print(f"Loading {architecture} model from: {model_path}")
    custom_objects = {
        'jacard_coef': jacard_coef, 'dice_coef': dice_coef,
        'combined_tversky_focal_loss': get_loss_function('combined_tversky'),
        'combined_dice_focal_loss': get_loss_function('combined'),
    }
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"✓ {architecture} model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load {architecture} model directly: {e}. Rebuilding...")
        input_shape = (CONFIG['img_height'], CONFIG['img_width'], CONFIG['img_channels'])
        model = get_model(model_name=architecture, input_shape=input_shape, NUM_CLASSES=1, dropout_rate=0.3, batch_norm=True)
        model.load_weights(model_path)
        print(f"✓ {architecture} weights loaded into rebuilt model")
        return model

def predict_on_tiles(model, tiles):
    """Run model prediction on a list of image tiles."""
    predictions = []
    for tile in tqdm(tiles, desc="  Predicting tiles", leave=False):
        tile_input = tile[np.newaxis, ..., np.newaxis]
        pred = model.predict(tile_input, verbose=0)
        pred_mask = (pred[0, ..., 0] > 0.5).astype(np.uint8) * 255
        predictions.append(pred_mask)
    return predictions

# --- ANALYSIS AND VISUALIZATION ---

def save_representative_masks(image_name, arch_name, predictions, densities, output_dir):
    """Saves masks for tiles with min, median, and max densities."""
    if not densities:
        return

    sorted_indices = np.argsort(densities)
    min_idx, max_idx = sorted_indices[0], sorted_indices[-1]
    median_idx = sorted_indices[len(sorted_indices) // 2]

    reps = {'min': min_idx, 'median': median_idx, 'max': max_idx}

    for name, idx in reps.items():
        mask_to_save = predictions[idx]
        density = densities[idx]
        filename = f"{image_name}_{arch_name}_{name}_density_{density:.4f}.png"
        save_path = output_dir / arch_name / filename
        cv2.imwrite(str(save_path), mask_to_save)
    print(f"  ✓ Saved representative masks for {arch_name}")

def create_dilution_boxplots(df, output_dir):
    """Creates a boxplot for each method, showing density vs. dilution factor."""
    print("\nGenerating dilution-based boxplots...")

    def extract_dilution(image_name):
        # Finds numbers followed by 'x' (e.g., '10x', '160x')
        match = re.search(r'(\d+)x', image_name, re.IGNORECASE)
        return int(match.group(1)) if match else None

    df['dilution'] = df['image'].apply(extract_dilution)
    df.dropna(subset=['dilution'], inplace=True)
    if df.empty:
        print("  ⚠ No dilution factors found in image names. Skipping boxplots.")
        return

    df = df.sort_values('dilution')
    df['dilution_label'] = df['dilution'].astype(str) + 'x'

    methods = df['method'].unique()
    for method in methods:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        method_df = df[df['method'] == method]

        sns.boxplot(data=method_df, x='dilution_label', y='density', ax=ax, palette='viridis')

        ax.set_title(f'Particle Density vs. Dilution Factor\n(Method: {method.replace("_", " ").title()})',
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Dilution Factor', fontsize=12)
        ax.set_ylabel('Particle Density (Fraction)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plot_path = output_dir / f"{method}_density_vs_dilution.png"
        plt.savefig(plot_path, dpi=CONFIG['dpi'])
        plt.close()
        print(f"  ✓ Saved boxplot: {plot_path}")

# --- MAIN WORKFLOW ---

def process_test_image(image_path, models, output_dirs):
    """Processes a single test image, returning aggregated and per-tile data."""
    image_name = Path(image_path).stem
    print(f"\nProcessing: {image_name}")

    img_original, img_normalized = load_and_preprocess_image(image_path)
    tiles_original, positions = extract_tiles_512(img_original)
    tiles_normalized, _ = extract_tiles_512(img_normalized)
    print(f"  Extracted {len(tiles_original)} tiles")

    all_tile_data_for_image = []

    # --- CLAHE+OTSU (Baseline) ---
    densities_clahe_otsu = []
    for i, tile_orig in enumerate(tiles_original):
        _, density = apply_clahe_otsu(tile_orig, **CONFIG['clahe'])
        densities_clahe_otsu.append(density)
        all_tile_data_for_image.append({'image': image_name, 'method': 'clahe_otsu', 'density': density})

    # --- Model Predictions ---
    for arch_name, model in models.items():
        print(f"  Predicting with {arch_name}...")
        predictions = predict_on_tiles(model, tiles_normalized)
        densities_pred = [np.count_nonzero(p) / p.size for p in predictions]

        for density in densities_pred:
            all_tile_data_for_image.append({'image': image_name, 'method': arch_name, 'density': density})

        # Save 3 representative masks instead of all of them
        save_representative_masks(image_name, arch_name, predictions, densities_pred, output_dirs['representative_masks'])

    return all_tile_data_for_image

def main():
    """Main execution function."""
    print("="*80 + "\nPREDICTION AND DENSITY ANALYSIS\n" + "="*80)

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

    # --- Find and Process Test Images ---
    test_images_path = Path(CONFIG['test_images_dir'])
    test_images = sorted(list(test_images_path.glob('*.tif*')) + list(test_images_path.glob('*.png')))
    if not test_images:
        print(f"\n✗ No test images found in {test_images_path}. Exiting.")
        sys.exit(1)
    print(f"\n✓ Found {len(test_images)} test images to process.")

    all_tile_data = []
    for image_path in test_images:
        try:
            tile_data_for_image = process_test_image(image_path, models, output_dirs)
            all_tile_data.extend(tile_data_for_image)
        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # --- Final Reports and Plots ---
    if not all_tile_data:
        print("\nNo data was processed. Exiting analysis.")
        return

    comprehensive_df = pd.DataFrame(all_tile_data)

    # Save comprehensive per-tile CSV
    comprehensive_csv_path = output_dirs['summary'] / 'comprehensive_tile_densities.csv'
    comprehensive_df.to_csv(comprehensive_csv_path, index=False)
    print(f"\n✓ Saved comprehensive tile data to: {comprehensive_csv_path}")

    # Generate dilution boxplots from the comprehensive data
    create_dilution_boxplots(comprehensive_df, output_dirs['dilution_boxplots'])

    print("\n" + "="*80 + "\nANALYSIS COMPLETE\n" + "="*80)
    print(f"Results saved to: {CONFIG['output_dir']}")
    print(f"  - Representative masks: {output_dirs['representative_masks']}")
    print(f"  - Dilution boxplots: {output_dirs['dilution_boxplots']}")
    print(f"  - Summary CSV: {output_dirs['summary']}")
    print("="*80)

if __name__ == '__main__':
    main()
