#!/usr/bin/env python3
"""
3-Architecture Density Analysis with Dual Contrast Visualization
=================================================================

Script: density_analysis_3architectures_dual_contrast.py
PBS Script: pbs_density_analysis_3architectures_dual_contrast.sh

Analyzes 3 UNet architectures with dual representative tile visualization:
- UNet (baseline)
- Attention UNet
- Attention ResUNet

KEY FEATURES:
1. ✅ Compares 3 architectures (like PyTorch analysis)
2. ✅ TWO sets of representative tiles for direct comparison:
   - FIXED contrast (vmin=0, vmax=1) - matches previous results
   - AUTO contrast (vmin/vmax from data) - enhanced visibility
3. ✅ Same tile selection for both sets (fair comparison)
4. ✅ 5-panel visualization (Original + 3 architectures + CLAHE+Otsu)

Uses the BEST model from each hyperparameter search:
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

    # Image settings (RGB from training)
    'img_size': 512,  # Tile size
    'img_channels': 3,  # RGB

    # Prediction settings
    'batch_size': 8,
    'thresholds': [0.2, 0.5, 0.8, 0.95],  # Multiple thresholds to test

    # Representative tiles
    'n_representative_tiles': 5,

    # Plotting
    'dpi': 300,
    'figsize_boxplot_full': (14, 8),
    'figsize_boxplot_low': (14, 8),
    'figsize_tiles': (25, 25),  # For 5-panel tiles
}

# Dilution factors in proper order
DILUTION_ORDER = [10240, 5120, 2560, 1280, 640, 320, 160, 80, 20, 10]
DILUTION_LABELS = ['10240x', '5120x', '2560x', '1280x', '640x', '320x', '160x', '80x', '20x', '10x']
DILUTION_ORDER_LOW = [10240, 5120, 2560, 1280, 640, 320, 160, 80]
DILUTION_LABELS_LOW = ['10240x', '5120x', '2560x', '1280x', '640x', '320x', '160x', '80x']

# Dilution factor patterns
DILUTION_PATTERNS = {
    '10240x': 10240, '5120x': 5120, '2560x': 2560, '1280x': 1280,
    '640x': 640, '320x': 320, '160x': 160, '80x': 80, '20x': 20, '10x': 10,
}

# ============================================================================
# HEADER
# ============================================================================

def print_header(config, best_models_info):
    """Print analysis header."""
    print("="*80)
    print("3-ARCHITECTURE DENSITY ANALYSIS - DUAL CONTRAST VISUALIZATION")
    print("="*80)
    print(f"Script: density_analysis_3architectures_dual_contrast.py")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test images: {config['test_images_dir']}")
    print(f"Output directory: {config['output_dir']}")
    print()
    print("="*80)
    print("DUAL VISUALIZATION STRATEGY")
    print("="*80)
    print("TWO sets of representative tiles will be generated:")
    print("  1. FIXED contrast (vmin=0, vmax=1)")
    print("     └─> For direct comparison with previous results")
    print("  2. AUTO contrast (vmin/vmax from actual data)")
    print("     └─> For enhanced visibility of model predictions")
    print("Same tiles selected for both sets (fair comparison)")
    print("="*80)
    print()
    print("="*80)
    print("BEST MODELS SELECTED")
    print("="*80)
    for arch_name, model_info in best_models_info.items():
        print(f"\n{arch_name.upper().replace('_', ' ')}:")
        print(f"  Model: {model_info['model_name']}")
        print(f"  Best Val IoU: {model_info['best_val_iou']:.4f}")
        print(f"  Path: {model_info['model_path']}")
    print("="*80)
    print()

# ============================================================================
# MODEL SELECTION
# ============================================================================

def find_best_model(base_dir, arch_prefix):
    """
    Find the best model from hyperparameter search for a given architecture.

    Args:
        base_dir: Base directory containing checkpoints
        arch_prefix: Architecture prefix ('unet', 'attention_unet', etc.)
    """
    print(f"Searching for best {arch_prefix} model...")
    base_dir = Path(base_dir)

    # Find all checkpoint directories
    checkpoint_dirs = list((base_dir / 'checkpoints').glob(f'{arch_prefix}_*'))

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No {arch_prefix} checkpoint directories found in {base_dir / 'checkpoints'}")

    print(f"Found {len(checkpoint_dirs)} {arch_prefix} model configurations")

    best_model_info = None
    best_iou = -1.0

    # Search through each checkpoint directory
    for ckpt_dir in checkpoint_dirs:
        model_file = ckpt_dir / 'best_model.keras'

        if not model_file.exists():
            continue

        dir_name = ckpt_dir.name

        try:
            # Find corresponding history CSV to get IoU
            history_files = list((base_dir / 'logs').glob(f'{dir_name}_*_history.csv'))

            if history_files:
                history_df = pd.read_csv(history_files[0])
                if 'val_jacard_coef' in history_df.columns:
                    max_iou = history_df['val_jacard_coef'].max()

                    if max_iou > best_iou:
                        best_iou = max_iou
                        best_model_info = {
                            'model_path': model_file,
                            'model_name': dir_name,
                            'best_val_iou': max_iou,
                            'architecture': arch_prefix
                        }
                        print(f"  New best: {dir_name} (IoU: {max_iou:.4f})")
        except Exception as e:
            print(f"  Warning: Could not parse {dir_name}: {e}")
            continue

    if best_model_info is None:
        raise FileNotFoundError(f"Could not find any valid {arch_prefix} models with validation IoU data")

    print(f"✓ Selected best {arch_prefix}: {best_model_info['model_name']}")
    print(f"  Best Val IoU: {best_model_info['best_val_iou']:.4f}\n")

    return best_model_info

# ... (continuing in next message due to length)
