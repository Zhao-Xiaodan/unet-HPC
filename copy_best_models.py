#!/usr/bin/env python3
"""
Copy Best Models to Centralized Location

Purpose: Find best model from each architecture's hyperparameter search
         and copy to ./best_models/ directory for easy access

Date: October 17, 2025
"""

import os
import sys
import json
import shutil
import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Source directories (hyperparameter search results)
    'unet_dir': './unet_hyperparam_20251015_224125',
    'attention_unet_dir': './attention_unet_hyperparam_20251015_230149',
    'attention_resunet_dir': './attention_resunet_hyperparam_20251015_235542',

    # Destination directory
    'output_dir': './best_models',
}

# ============================================================================
# FIND BEST MODEL
# ============================================================================

def find_best_model(base_dir, architecture_name, glob_pattern):
    """
    Find best model from hyperparameter search by validation IoU.

    Returns:
        dict with model_path, model_name, best_val_iou, config
    """
    print(f"\nSearching for best {architecture_name} model...")
    print(f"  Base directory: {base_dir}")

    base_dir = Path(base_dir)
    checkpoints_dir = base_dir / 'checkpoints'

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    model_dirs = list(checkpoints_dir.glob(glob_pattern))

    if not model_dirs:
        raise FileNotFoundError(f"No {architecture_name} models found in {checkpoints_dir}")

    print(f"  Found {len(model_dirs)} model configurations")

    best_model = None
    best_iou = -1

    for model_dir in model_dirs:
        model_file = model_dir / 'best_model.keras'

        if not model_file.exists():
            continue

        # Parse hyperparameters from directory name to find history CSV
        # Format: {arch}_n_filters16_dropout0p1_batch_normTrue_learning_rate0p001
        dir_name = model_dir.name

        try:
            # Extract key parameters for history file matching
            import re
            n_filters_match = re.search(r'n_filters(\d+)', dir_name)
            dropout_match = re.search(r'dropout(\d+p\d+)', dir_name)

            if not n_filters_match or not dropout_match:
                print(f"    Skip: {dir_name} (cannot parse hyperparameters)")
                continue

            n_filters = n_filters_match.group(1)
            dropout_str = dropout_match.group(1)

            # Find history CSV in logs directory
            # Pattern: {glob_pattern}_n_filters{}_dropout{}_*_history.csv
            arch_prefix = glob_pattern.replace('*', '')  # e.g., 'unet_' from 'unet_*'
            history_pattern = f'{arch_prefix}n_filters{n_filters}_dropout{dropout_str}_*_history.csv'
            history_files = list((base_dir / 'logs').glob(history_pattern))

            if not history_files:
                print(f"    Skip: {dir_name} (no history CSV in logs/)")
                continue

            # Read history and get max IoU
            history_csv = history_files[0]
            df = pd.read_csv(history_csv)

            if 'val_jacard_coef' not in df.columns:
                print(f"    Skip: {dir_name} (no val_jacard_coef in history)")
                continue

            max_iou = df['val_jacard_coef'].max()

            if max_iou > best_iou:
                best_iou = max_iou
                best_model = {
                    'model_path': model_file,
                    'model_name': model_dir.name,
                    'best_val_iou': max_iou,
                    'history_csv': history_csv,
                }
                print(f"    New best: {model_dir.name} (IoU: {max_iou:.4f})")

        except Exception as e:
            print(f"    Warning: Failed to process {dir_name}: {e}")
            continue

    if best_model is None:
        raise FileNotFoundError(f"No valid {architecture_name} models found")

    print(f"  ✓ Selected: {best_model['model_name']}")
    print(f"    Best Val IoU: {best_model['best_val_iou']:.4f}")

    return best_model

# ============================================================================
# COPY MODEL
# ============================================================================

def copy_best_model(best_model_info, output_dir, architecture_name):
    """
    Copy best model and metadata to output directory.

    Args:
        best_model_info: dict from find_best_model()
        output_dir: destination directory
        architecture_name: architecture name (for subdirectory)

    Returns:
        Path to copied model
    """
    output_dir = Path(output_dir)
    arch_dir = output_dir / architecture_name.lower().replace(' ', '_')
    arch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {architecture_name} model to {arch_dir}...")

    # Copy model file
    src_model = best_model_info['model_path']
    dst_model = arch_dir / 'best_model.keras'

    print(f"  Copying: {src_model.name}")
    shutil.copy2(src_model, dst_model)
    print(f"  ✓ Model copied: {dst_model}")

    # Copy training history
    src_history = best_model_info['history_csv']
    dst_history = arch_dir / 'training_history.csv'

    print(f"  Copying: {src_history.name}")
    shutil.copy2(src_history, dst_history)
    print(f"  ✓ History copied: {dst_history}")

    # Save metadata
    metadata = {
        'architecture': architecture_name,
        'model_name': best_model_info['model_name'],
        'best_val_iou': float(best_model_info['best_val_iou']),
        'source_path': str(best_model_info['model_path']),
        'copied_path': str(dst_model),
    }

    metadata_file = arch_dir / 'model_info.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved: {metadata_file}")

    return dst_model

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("COPY BEST MODELS TO CENTRALIZED LOCATION")
    print("="*80)
    print()

    config = CONFIG

    # Create output directory
    output_dir = Path(config['output_dir'])
    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}")
        print("Removing existing directory...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created output directory: {output_dir}")

    # Find and copy best models
    architectures = [
        ('unet_dir', 'UNet', 'unet_*'),
        ('attention_unet_dir', 'Attention_UNet', 'attention_unet_*'),
        ('attention_resunet_dir', 'Attention_ResUNet', 'attention_resunet_*'),
    ]

    best_models = {}

    for config_key, arch_name, glob_pattern in architectures:
        base_dir = config[config_key]

        # Find best model
        best_model_info = find_best_model(base_dir, arch_name, glob_pattern)

        # Copy to output directory
        copied_model_path = copy_best_model(best_model_info, output_dir, arch_name)

        best_models[arch_name] = {
            'model_path': str(copied_model_path),
            'model_name': best_model_info['model_name'],
            'best_val_iou': float(best_model_info['best_val_iou']),
        }

    # Save summary
    summary = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'output_directory': str(output_dir),
        'best_models': best_models,
    }

    summary_file = output_dir / 'SUMMARY.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print()
    print("Best models copied:")
    for arch_name, info in best_models.items():
        print(f"  {arch_name}:")
        print(f"    Model: {info['model_name']}")
        print(f"    Val IoU: {info['best_val_iou']:.4f}")
        print(f"    Path: {info['model_path']}")
    print()
    print(f"✓ Summary saved: {summary_file}")
    print("="*80)

if __name__ == '__main__':
    main()
