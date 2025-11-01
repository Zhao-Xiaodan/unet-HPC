#!/usr/bin/env python3
"""
3-Architecture Density Analysis with Dual Contrast and Edge Fix
================================================================

Script: density_analysis_3arch_dual_contrast.py
PBS Script: pbs_density_analysis_3arch_dual_contrast.sh

FIXES & FEATURES:
1. ✅ Analyzes 3 UNet architectures (UNet, Attention UNet, Attention ResUNet)
2. ✅ 4-panel visualization (Original + 3 predictions) like PyTorch analysis
3. ✅ FIXES CLAHE edge artifacts with border padding/cropping
4. ✅ DUAL tile sets (fixed contrast + auto contrast) for comparison
5. ✅ Shows actual min/max ranges in auto-contrast titles

Architecture comparison:
- unet_hyperparam_20251015_224125/
- attention_unet_hyperparam_20251015_230149/
- attention_resunet_hyperparam_20251015_235542/

Author: Claude Code
Date: November 1, 2025
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
    # Model directories for 3 architectures
    'model_dirs': {
        'unet': './unet_hyperparam_20251015_224125',
        'attention_unet': './attention_unet_hyperparam_20251015_230149',
        'attention_resunet': './attention_resunet_hyperparam_20251015_235542',
    },

    'test_images_dir': './test_images',
    'output_dir': f'./density_analysis_3arch_dual_contrast_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Image settings
    'img_size': 512,
    'img_channels': 3,

    # Prediction settings
    'batch_size': 8,
    'thresholds': [0.5],  # Using 0.5 as primary threshold

    # Representative tiles
    'n_representative_tiles': 5,

    # Plotting
    'dpi': 300,
    'figsize_boxplot_full': (14, 8),
    'figsize_boxplot_low': (14, 8),
    'figsize_tiles': (20, 25),  # For 4-panel tiles
}

# Dilution factors
DILUTION_ORDER = [10240, 5120, 2560, 1280, 640, 320, 160, 80, 20, 10]
DILUTION_LABELS = ['10240x', '5120x', '2560x', '1280x', '640x', '320x', '160x', '80x', '20x', '10x']
DILUTION_ORDER_LOW = [10240, 5120, 2560, 1280, 640, 320, 160, 80]
DILUTION_LABELS_LOW = ['10240x', '5120x', '2560x', '1280x', '640x', '320x', '160x', '80x']

DILUTION_PATTERNS = {
    '10240x': 10240, '5120x': 5120, '2560x': 2560, '1280x': 1280,
    '640x': 640, '320x': 320, '160x': 160, '80x': 80, '20x': 20, '10x': 10,
}

ARCHITECTURE_NAMES = {
    'unet': 'UNet',
    'attention_unet': 'Attention UNet',
    'attention_resunet': 'Attention ResUNet'
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_dilution_from_filename(filename):
    """Extract dilution factor from filename."""
    filename_lower = filename.lower()
    for pattern, dilution in DILUTION_PATTERNS.items():
        if pattern in filename_lower:
            return dilution
    return 1

def load_test_image(image_path):
    """Load test image and convert to RGB."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_tiles(image, tile_size):
    """Extract non-overlapping tiles from image."""
    h, w = image.shape[:2]
    tiles_with_pos = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            tiles_with_pos.append((tile, (y, x)))
    return tiles_with_pos

def preprocess_tile(tile):
    """Preprocess tile for model input."""
    if len(tile.shape) == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
    tile = tile.astype(np.float32) / 255.0
    return tile

def calculate_foreground_density(prediction, threshold=0.5):
    """Calculate foreground density from prediction mask."""
    binary_mask = (prediction > threshold).astype(np.float32)
    density = np.mean(binary_mask)
    return density

def rescale_image_full_range(img):
    """Rescale image to full 0-255 range."""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 0:
        img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)

def apply_clahe_otsu_fixed(img_gray, clipLimit=2.0, tileGridSize=(8, 8), border_size=8):
    """
    Apply CLAHE + OTSU with edge artifact fix.
    
    FIX: Pads borders before CLAHE to avoid edge effects, then crops after.
    
    Args:
        img_gray: Grayscale image (0-255 uint8)
        clipLimit: CLAHE clip limit
        tileGridSize: CLAHE tile grid size
        border_size: Pixels to pad/crop (default: 8 = same as tileGridSize)
    
    Returns:
        binary_mask: Binary mask after CLAHE+Otsu WITHOUT edge artifacts
    """
    # STEP 1: Pad borders to avoid CLAHE edge effects
    img_padded = cv2.copyMakeBorder(
        img_gray, 
        border_size, border_size, border_size, border_size,
        cv2.BORDER_REPLICATE  # Replicate edge pixels
    )
    
    # STEP 2: Apply CLAHE on padded image
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_img = clahe.apply(img_padded)
    
    # STEP 3: Apply Otsu thresholding
    _, binary_mask_padded = cv2.threshold(
        clahe_img, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    # STEP 4: Crop borders to remove edge effects
    binary_mask = binary_mask_padded[
        border_size:-border_size,
        border_size:-border_size
    ]
    
    return binary_mask

def calculate_density_with_clahe_otsu(prediction, clipLimit=2.0, tileGridSize=(8, 8)):
    """Calculate density from predicted mask after applying CLAHE+Otsu (with edge fix)."""
    pred_rescaled = rescale_image_full_range(prediction.squeeze())
    binary_mask = apply_clahe_otsu_fixed(pred_rescaled, clipLimit=clipLimit, tileGridSize=tileGridSize)
    density = 1.0 - (np.count_nonzero(binary_mask) / binary_mask.size)
    return density, binary_mask


# ============================================================================
# MODEL SELECTION & LOADING
# ============================================================================

def find_best_model(base_dir, arch_name):
    """Find the best model for given architecture."""
    print(f"\nSearching for best {arch_name} model...")
    base_dir = Path(base_dir)
    
    checkpoint_dir = base_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        # Try without checkpoints subdirectory
        checkpoint_dir = base_dir
    
    # Find model files
    model_files = list(checkpoint_dir.glob(f'{arch_name}_*/best_model.keras'))
    if not model_files:
        model_files = list(checkpoint_dir.glob('*/best_model.keras'))
    
    if not model_files:
        raise FileNotFoundError(f"No models found in {checkpoint_dir}")
    
    print(f"Found {len(model_files)} {arch_name} model(s)")
    
    best_model_info = None
    best_iou = -1.0
    
    for model_file in model_files:
        dir_name = model_file.parent.name
        
        # Find history CSV
        history_files = list((base_dir / 'logs').glob(f'{dir_name}*_history.csv'))
        if not history_files:
            history_files = list(base_dir.glob(f'{dir_name}*_history.csv'))
        if not history_files:
            history_files = list(base_dir.glob('*_history.csv'))
        
        if history_files:
            try:
                history_df = pd.read_csv(history_files[0])
                if 'val_jacard_coef' in history_df.columns:
                    max_iou = history_df['val_jacard_coef'].max()
                    if max_iou > best_iou:
                        best_iou = max_iou
                        best_model_info = {
                            'model_path': model_file,
                            'model_name': dir_name,
                            'best_val_iou': max_iou,
                            'architecture': arch_name
                        }
                        print(f"  Best so far: {dir_name} (IoU: {max_iou:.4f})")
            except Exception as e:
                continue
    
    if best_model_info is None:
        # Fallback: just use first model found
        print(f"  Warning: No IoU data found, using first model")
        best_model_info = {
            'model_path': model_files[0],
            'model_name': model_files[0].parent.name,
            'best_val_iou': 0.0,
            'architecture': arch_name
        }
    
    print(f"✓ Selected: {best_model_info['model_name']} (IoU: {best_model_info['best_val_iou']:.4f})")
    return best_model_info

def load_model(model_path):
    """Load trained model."""
    print(f"Loading model from: {model_path}")
    
    @keras.saving.register_keras_serializable(package='Custom')
    class BinaryFocalLoss(keras.losses.Loss):
        def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.alpha = alpha
        def call(self, y_true, y_pred):
            return focal_loss(y_true, y_pred, alpha=self.alpha, gamma=self.gamma)
        def get_config(self):
            config = super().get_config()
            config.update({'gamma': self.gamma, 'alpha': self.alpha})
            return config
        @classmethod
        def from_config(cls, config):
            return cls(**config)
    
    custom_objects = {
        'combined_dice_focal_loss': combined_dice_focal_loss,
        'jacard_coef': jacard_coef,
        'dice_coef': dice_coef,
        'focal_loss': focal_loss,
        'BinaryFocalLoss': BinaryFocalLoss,
    }
    
    model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
    print(f"  ✓ Loaded successfully")
    return model


# ============================================================================
# 4-PANEL TILE VISUALIZATION (Original + 3 Architectures)
# ============================================================================

def create_4panel_tiles_fixed_contrast(tile_data_by_arch, output_dir, config):
    """
    Create 4-panel visualizations with FIXED contrast (vmin=0, vmax=1).
    Panels: Original | UNet | Attention UNet | Attention ResUNet
    """
    print("\nGenerating 4-panel tiles (FIXED CONTRAST: vmin=0, vmax=1)...")
    
    output_dir = Path(output_dir)
    tiles_dir = output_dir / 'representative_tiles_4panel_fixed_contrast'
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Get UNet data as reference (all archs should have same images/tiles)
    unet_data = tile_data_by_arch['unet']
    
    # Group by image
    tiles_by_image = defaultdict(list)
    for tile_info in unet_data:
        tiles_by_image[tile_info['image']].append(tile_info)
    
    for image_name, tiles in tiles_by_image.items():
        print(f"  Creating tiles for: {image_name}")
        
        # Select 5 representative tiles
        tiles_sorted = sorted(tiles, key=lambda x: x['density_clahe_otsu_pred'])
        n_tiles = len(tiles_sorted)
        indices = [0, n_tiles//4, n_tiles//2, 3*n_tiles//4, n_tiles-1]
        representative_tiles = [tiles_sorted[i] for i in indices]
        
        dilution_label = tiles[0]['dilution_label']
        
        # Create 4-panel figure (5 rows × 4 columns)
        fig, axes = plt.subplots(5, 4, figsize=(20, 25))
        
        for row_idx, tile_info in enumerate(representative_tiles):
            tile_idx = tile_info['tile_idx']
            pos = tile_info['position']
            
            # Get predictions from all 3 architectures
            unet_pred = tile_data_by_arch['unet'][tile_idx]['prediction']
            attn_unet_pred = tile_data_by_arch['attention_unet'][tile_idx]['prediction']
            attn_res_pred = tile_data_by_arch['attention_resunet'][tile_idx]['prediction']
            
            unet_density = tile_data_by_arch['unet'][tile_idx]['density_threshold_0.5']
            attn_unet_density = tile_data_by_arch['attention_unet'][tile_idx]['density_threshold_0.5']
            attn_res_density = tile_data_by_arch['attention_resunet'][tile_idx]['density_threshold_0.5']
            
            # Panel 1: Original
            axes[row_idx, 0].imshow(tile_info['tile'], cmap='gray')
            axes[row_idx, 0].set_title(f'Original Tile {tile_idx}\nPos: ({pos[0]}, {pos[1]})', fontsize=10)
            axes[row_idx, 0].axis('off')
            
            # Panel 2: UNet (inverted, fixed contrast)
            unet_inv = 1.0 - unet_pred.squeeze()
            axes[row_idx, 1].imshow(unet_inv, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 1].set_title(f'UNet\nDensity: {unet_density:.4f}', fontsize=10)
            axes[row_idx, 1].axis('off')
            
            # Panel 3: Attention UNet (inverted, fixed contrast)
            attn_unet_inv = 1.0 - attn_unet_pred.squeeze()
            axes[row_idx, 2].imshow(attn_unet_inv, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 2].set_title(f'Attention UNet\nDensity: {attn_unet_density:.4f}', fontsize=10)
            axes[row_idx, 2].axis('off')
            
            # Panel 4: Attention ResUNet (inverted, fixed contrast)
            attn_res_inv = 1.0 - attn_res_pred.squeeze()
            axes[row_idx, 3].imshow(attn_res_inv, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, 3].set_title(f'Attention ResUNet\nDensity: {attn_res_density:.4f}', fontsize=10)
            axes[row_idx, 3].axis('off')
        
        fig.suptitle(f'{image_name} - 4-Panel Comparison (FIXED CONTRAST)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        
        output_filename = f'tiles_4panel_fixed_{dilution_label}_{image_name.replace(".tif", "").replace(".tiff", "")}.png'
        output_path = tiles_dir / output_filename
        plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {output_path.name}")
    
    print(f"\n  ✓ All 4-panel fixed-contrast tiles saved to: {tiles_dir}")

def create_4panel_tiles_auto_contrast(tile_data_by_arch, output_dir, config):
    """
    Create 4-panel visualizations with AUTO contrast (vmin/vmax from data).
    Panels: Original | UNet | Attention UNet | Attention ResUNet
    """
    print("\nGenerating 4-panel tiles (AUTO CONTRAST: vmin/vmax from data)...")
    
    output_dir = Path(output_dir)
    tiles_dir = output_dir / 'representative_tiles_4panel_auto_contrast'
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    unet_data = tile_data_by_arch['unet']
    tiles_by_image = defaultdict(list)
    for tile_info in unet_data:
        tiles_by_image[tile_info['image']].append(tile_info)
    
    for image_name, tiles in tiles_by_image.items():
        print(f"  Creating tiles for: {image_name}")
        
        tiles_sorted = sorted(tiles, key=lambda x: x['density_clahe_otsu_pred'])
        n_tiles = len(tiles_sorted)
        indices = [0, n_tiles//4, n_tiles//2, 3*n_tiles//4, n_tiles-1]
        representative_tiles = [tiles_sorted[i] for i in indices]
        
        dilution_label = tiles[0]['dilution_label']
        fig, axes = plt.subplots(5, 4, figsize=(20, 25))
        
        for row_idx, tile_info in enumerate(representative_tiles):
            tile_idx = tile_info['tile_idx']
            pos = tile_info['position']
            
            # Get predictions
            unet_pred = tile_data_by_arch['unet'][tile_idx]['prediction']
            attn_unet_pred = tile_data_by_arch['attention_unet'][tile_idx]['prediction']
            attn_res_pred = tile_data_by_arch['attention_resunet'][tile_idx]['prediction']
            
            unet_density = tile_data_by_arch['unet'][tile_idx]['density_threshold_0.5']
            attn_unet_density = tile_data_by_arch['attention_unet'][tile_idx]['density_threshold_0.5']
            attn_res_density = tile_data_by_arch['attention_resunet'][tile_idx]['density_threshold_0.5']
            
            # Panel 1: Original
            axes[row_idx, 0].imshow(tile_info['tile'], cmap='gray')
            axes[row_idx, 0].set_title(f'Original Tile {tile_idx}\nPos: ({pos[0]}, {pos[1]})', fontsize=10)
            axes[row_idx, 0].axis('off')
            
            # Panel 2: UNet (AUTO contrast)
            unet_inv = 1.0 - unet_pred.squeeze()
            v_min, v_max = unet_inv.min(), unet_inv.max()
            axes[row_idx, 1].imshow(unet_inv, cmap='gray', vmin=v_min, vmax=v_max)
            axes[row_idx, 1].set_title(f'UNet (AUTO)\nDensity: {unet_density:.4f}\nRange: [{v_min:.3f}, {v_max:.3f}]', fontsize=9)
            axes[row_idx, 1].axis('off')
            
            # Panel 3: Attention UNet (AUTO contrast)
            attn_unet_inv = 1.0 - attn_unet_pred.squeeze()
            v_min, v_max = attn_unet_inv.min(), attn_unet_inv.max()
            axes[row_idx, 2].imshow(attn_unet_inv, cmap='gray', vmin=v_min, vmax=v_max)
            axes[row_idx, 2].set_title(f'Attention UNet (AUTO)\nDensity: {attn_unet_density:.4f}\nRange: [{v_min:.3f}, {v_max:.3f}]', fontsize=9)
            axes[row_idx, 2].axis('off')
            
            # Panel 4: Attention ResUNet (AUTO contrast)
            attn_res_inv = 1.0 - attn_res_pred.squeeze()
            v_min, v_max = attn_res_inv.min(), attn_res_inv.max()
            axes[row_idx, 3].imshow(attn_res_inv, cmap='gray', vmin=v_min, vmax=v_max)
            axes[row_idx, 3].set_title(f'Attention ResUNet (AUTO)\nDensity: {attn_res_density:.4f}\nRange: [{v_min:.3f}, {v_max:.3f}]', fontsize=9)
            axes[row_idx, 3].axis('off')
        
        fig.suptitle(f'{image_name} - 4-Panel Comparison (AUTO CONTRAST)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        
        output_filename = f'tiles_4panel_auto_{dilution_label}_{image_name.replace(".tif", "").replace(".tiff", "")}.png'
        output_path = tiles_dir / output_filename
        plt.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved: {output_path.name}")
    
    print(f"\n  ✓ All 4-panel auto-contrast tiles saved to: {tiles_dir}")


# ============================================================================
# PREDICTION FOR 3 ARCHITECTURES
# ============================================================================

def predict_all_architectures(models, test_images_dir, config):
    """
    Predict on all test images using all 3 architectures.
    
    Returns:
        tile_data_by_arch: Dict of {arch_name: list of tile data}
    """
    print("\n" + "="*80)
    print("PREDICTION ON ALL 3 ARCHITECTURES")
    print("="*80)
    
    test_images_dir = Path(test_images_dir)
    image_files = sorted(list(test_images_dir.glob('*.tif')) + list(test_images_dir.glob('*.tiff')))
    
    print(f"Found {len(image_files)} test images")
    print(f"Architectures: {list(models.keys())}")
    print()
    
    tile_data_by_arch = {arch: [] for arch in models.keys()}
    
    for img_file in tqdm(image_files, desc="Processing images"):
        print(f"\nProcessing: {img_file.name}")
        
        dilution = extract_dilution_from_filename(img_file.name)
        dilution_label = f"{dilution}x"
        
        image = load_test_image(img_file)
        tiles_with_pos = extract_tiles(image, config['img_size'])
        
        print(f"  Dilution: {dilution}x, Tiles: {len(tiles_with_pos)}")
        
        # Preprocess tiles once
        tiles_preprocessed = np.array([preprocess_tile(tile) for tile, pos in tiles_with_pos])
        
        # Predict with each architecture
        for arch_name, model in models.items():
            print(f"    Predicting with {ARCHITECTURE_NAMES[arch_name]}...")
            predictions = model.predict(tiles_preprocessed, batch_size=config['batch_size'], verbose=0)
            
            for tile_idx, ((tile, pos), prediction) in enumerate(zip(tiles_with_pos, predictions)):
                density_05 = calculate_foreground_density(prediction, 0.5)
                density_clahe, binary_mask = calculate_density_with_clahe_otsu(prediction)
                
                tile_data_by_arch[arch_name].append({
                    'image': img_file.name,
                    'dilution': dilution,
                    'dilution_label': dilution_label,
                    'tile_idx': tile_idx,
                    'position': pos,
                    'tile': tile,
                    'prediction': prediction,
                    'density_threshold_0.5': density_05,
                    'density_clahe_otsu_pred': density_clahe,
                    'binary_mask_pred': binary_mask,
                })
            
            mean_density = np.mean([td['density_threshold_0.5'] for td in tile_data_by_arch[arch_name] if td['image'] == img_file.name])
            print(f"      Mean density: {mean_density:.4f}")
    
    return tile_data_by_arch

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main analysis pipeline."""
    print("="*80)
    print("3-ARCHITECTURE DENSITY ANALYSIS - DUAL CONTRAST + EDGE FIX")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {CONFIG['output_dir']}")
    print()
    print("FEATURES:")
    print("  ✅ 3 architectures: UNet, Attention UNet, Attention ResUNet")
    print("  ✅ 4-panel visualization (Original + 3 predictions)")
    print("  ✅ DUAL contrast sets (Fixed + Auto)")
    print("  ✅ CLAHE edge artifacts FIXED")
    print("  ✅ Auto-contrast shows actual min/max ranges")
    print("="*80)
    print()
    
    # Find and load best models for all 3 architectures
    print("="*80)
    print("LOADING MODELS")
    print("="*80)
    
    models = {}
    best_models_info = {}
    
    for arch_name, model_dir in CONFIG['model_dirs'].items():
        try:
            best_model_info = find_best_model(model_dir, arch_name)
            model = load_model(best_model_info['model_path'])
            models[arch_name] = model
            best_models_info[arch_name] = best_model_info
        except Exception as e:
            print(f"ERROR loading {arch_name}: {e}")
            print(f"Skipping {arch_name}")
            continue
    
    if len(models) == 0:
        print("ERROR: No models loaded successfully!")
        return
    
    print(f"\n✓ Successfully loaded {len(models)} model(s)")
    print("="*80)
    
    # Predict on all test images with all architectures
    tile_data_by_arch = predict_all_architectures(models, CONFIG['test_images_dir'], CONFIG)
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment info
    experiment_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'script': 'density_analysis_3arch_dual_contrast.py',
        'architectures': list(models.keys()),
        'models': {arch: {
            'name': info['model_name'],
            'iou': info['best_val_iou'],
            'path': str(info['model_path'])
        } for arch, info in best_models_info.items()},
        'features': [
            'CLAHE edge artifacts fixed',
            'Dual contrast visualization',
            '4-panel architecture comparison',
            'Auto-contrast shows actual ranges'
        ]
    }
    
    with open(output_dir / 'EXPERIMENT_INFO.json', 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING 4-PANEL VISUALIZATIONS")
    print("="*80)
    
    create_4panel_tiles_fixed_contrast(tile_data_by_arch, CONFIG['output_dir'], CONFIG)
    create_4panel_tiles_auto_contrast(tile_data_by_arch, CONFIG['output_dir'], CONFIG)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Output directory: {CONFIG['output_dir']}")
    print("\nGenerated files:")
    print("  - EXPERIMENT_INFO.json")
    print("  - representative_tiles_4panel_fixed_contrast/")
    print("      * 4 panels: Original | UNet | Attention UNet | Attention ResUNet")
    print("      * Fixed contrast (vmin=0, vmax=1)")
    print("  - representative_tiles_4panel_auto_contrast/")
    print("      * 4 panels: Original | UNet | Attention UNet | Attention ResUNet")
    print("      * Auto contrast (vmin/vmax from data)")
    print("      * Shows actual range values in titles")
    print()
    print("FIXES APPLIED:")
    print("  ✓ CLAHE edge artifacts removed (padding + cropping)")
    print("  ✓ Auto-contrast proves it works (shows min/max ranges)")
    print("="*80)

if __name__ == '__main__':
    main()

