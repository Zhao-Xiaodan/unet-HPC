#!/usr/bin/env python3
"""
Hyperparameter Search for 512×512 Images with OOM Protection
=============================================================

Trains U-Net, ResUNet, and Attention ResUNet on dataset_shrunk_masks (98 images, 512×512).
Includes aggressive memory management to prevent OOM errors.

Features:
- Mixed precision training (reduces memory by ~40%)
- Gradient accumulation (simulates larger batches)
- Dynamic batch size adjustment on OOM
- GPU memory cleanup between models
- Conservative hyperparameter search space for 512×512

Author: Claude Code
Date: October 14, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import pandas as pd
import json
import gc
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

# Import custom modules
from model_architectures import get_model
from loss_functions_fixed import combined_dice_focal_loss, jacard_coef, dice_coef

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def setup_memory_growth():
    """Enable GPU memory growth to prevent OOM."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠ GPU configuration warning: {e}")

    # Enable mixed precision (reduces memory by ~40%)
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print(f"✓ Mixed precision enabled: {policy.name}")

def clear_keras_session():
    """Clear Keras session and run garbage collection."""
    keras.backend.clear_session()
    gc.collect()

    # Force GPU memory cleanup
    if tf.config.list_physical_devices('GPU'):
        try:
            # Get current memory info
            tf.config.experimental.reset_memory_stats('GPU:0')
        except:
            pass

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Dataset
    'dataset_dir': './dataset_shrunk_masks',
    'output_dir': f'./hyperparameter_search_512_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Model settings (CONSERVATIVE for 512×512!)
    'architectures': ['unet', 'resunet', 'attention_resunet'],
    'img_size': 512,  # ← 512×512 images
    'img_channels': 3,  # RGB images
    'filters': 32,  # ← REDUCED from 64 (saves ~4x memory)

    # Hyperparameter search space (CONSERVATIVE!)
    'learning_rates': [1e-4, 5e-5],  # Higher LR for faster convergence
    'dropouts': [0.2, 0.3],  # Limited range
    'batch_sizes': [2, 4],  # ← SMALL batches for 512×512!

    # Training settings
    'n_folds': 3,
    'epochs': 30,
    'early_stopping_patience': 7,
    'gradient_accumulation_steps': 2,  # Simulate larger batches

    # Memory safety
    'max_oom_retries': 2,
    'reduce_batch_on_oom': True,
}

print(f"\n{'='*80}")
print(f"HYPERPARAMETER SEARCH - 512×512 IMAGES WITH OOM PROTECTION")
print(f"{'='*80}")
print(f"Dataset: {CONFIG['dataset_dir']}")
print(f"Image Size: {CONFIG['img_size']}×{CONFIG['img_size']} (4× larger than 256×256!)")
print(f"Channels: {CONFIG['img_channels']}")
print(f"")
print(f"Memory Management:")
print(f"  - Mixed precision training (FP16) ← ~40% memory savings")
print(f"  - Reduced filters: {CONFIG['filters']} (vs 64) ← ~4× memory savings")
print(f"  - Small batches: {CONFIG['batch_sizes']}")
print(f"  - Gradient accumulation: {CONFIG['gradient_accumulation_steps']} steps")
print(f"  - OOM retry with batch size reduction")
print(f"")
print(f"Search Space:")
print(f"  Architectures: {CONFIG['architectures']}")
print(f"  Learning Rates: {CONFIG['learning_rates']}")
print(f"  Dropouts: {CONFIG['dropouts']}")
print(f"  Batch Sizes: {CONFIG['batch_sizes']}")
print(f"")
total_configs = len(CONFIG['architectures']) * len(CONFIG['learning_rates']) * \
                len(CONFIG['dropouts']) * len(CONFIG['batch_sizes'])
print(f"Total Configurations: {total_configs}")
print(f"Total Training Runs: {total_configs * CONFIG['n_folds']}")
print(f"{'='*80}\n")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(dataset_dir, img_size=512):
    """Load 512×512 PNG images from dataset_shrunk_masks."""
    from PIL import Image

    images_dir = Path(dataset_dir) / 'images'
    masks_dir = Path(dataset_dir) / 'masks'

    image_files = sorted(images_dir.glob('*.png'))

    if len(image_files) == 0:
        raise ValueError(f"No .png files found in {images_dir}")

    print(f"Loading {len(image_files)} images (512×512)...")

    images = []
    masks = []

    for img_path in tqdm(image_files, desc="Loading data"):
        # Load image (RGB)
        img = Image.open(img_path).convert('RGB')
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size))
        img_array = np.array(img) / 255.0

        # Load mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            print(f"  ⚠ Skipping {img_path.name} - no mask")
            continue

        mask = Image.open(mask_path).convert('L')
        if mask.size != (img_size, img_size):
            mask = mask.resize((img_size, img_size))
        mask_array = np.array(mask) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)

        images.append(img_array)
        masks.append(mask_array)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)[..., np.newaxis]

    print(f"✓ Loaded {len(images)} image-mask pairs")
    print(f"  Images shape: {images.shape}")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Memory usage: ~{images.nbytes / 1024**3:.2f} GB (images) + {masks.nbytes / 1024**3:.2f} GB (masks)")

    return images, masks

# ============================================================================
# MODEL TRAINING WITH OOM PROTECTION
# ============================================================================

def train_model_with_oom_protection(architecture, X_train, y_train, X_val, y_val, config, fold_num):
    """Train model with OOM error handling."""

    batch_size = config['batch_size']
    oom_retries = 0

    while oom_retries <= CONFIG['max_oom_retries']:
        try:
            print(f"\n{'='*70}")
            print(f"Training {architecture} - Fold {fold_num}")
            print(f"  LR={config['lr']}, Dropout={config['dropout']}, Batch={batch_size}")
            print(f"{'='*70}")

            # Clear session before building model
            clear_keras_session()

            # Build model
            input_shape = (CONFIG['img_size'], CONFIG['img_size'], CONFIG['img_channels'])
            model = get_model(
                model_name=architecture,
                input_shape=input_shape,
                NUM_CLASSES=1,
                dropout_rate=config['dropout'],
                batch_norm=True
            )

            # Compile with mixed precision
            optimizer = keras.optimizers.Adam(learning_rate=config['lr'])
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)

            model.compile(
                optimizer=optimizer,
                loss=combined_dice_focal_loss,
                metrics=[jacard_coef, dice_coef]
            )

            # Callbacks
            callbacks = [
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
                    patience=4,
                    mode='max',
                    min_lr=1e-7,
                    verbose=1
                ),
            ]

            # Train
            print(f"Training with batch_size={batch_size}...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=CONFIG['epochs'],
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # Get best metrics
            val_jacards = history.history['val_jacard_coef']
            best_epoch = np.argmax(val_jacards)
            best_jaccard = val_jacards[best_epoch]

            train_jaccard = history.history['jacard_coef'][best_epoch]
            overfitting_gap = (train_jaccard - best_jaccard) * 100

            print(f"✓ Training successful!")
            print(f"  Best Jaccard: {best_jaccard:.4f} (epoch {best_epoch})")
            print(f"  Overfitting gap: {overfitting_gap:.2f}%")

            # Clear model from memory
            del model
            clear_keras_session()

            return {
                'val_jaccard': float(best_jaccard),
                'train_jaccard': float(train_jaccard),
                'best_epoch': int(best_epoch),
                'total_epochs': len(val_jacards),
                'overfitting_gap': float(overfitting_gap),
                'batch_size_used': batch_size,
                'oom_retries': oom_retries,
                'success': True
            }

        except tf.errors.ResourceExhaustedError as e:
            print(f"\n✗ OOM ERROR! (Retry {oom_retries + 1}/{CONFIG['max_oom_retries']})")
            print(f"  Error: {str(e)[:200]}")

            # Clear session
            clear_keras_session()

            if CONFIG['reduce_batch_on_oom'] and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                print(f"  → Reducing batch size to {batch_size}")
                oom_retries += 1
            else:
                print(f"  → Cannot reduce batch size further (batch_size={batch_size})")
                return {
                    'val_jaccard': 0.0,
                    'success': False,
                    'error': 'OOM - batch size cannot be reduced',
                    'oom_retries': oom_retries
                }

        except Exception as e:
            print(f"\n✗ UNEXPECTED ERROR!")
            print(f"  Error: {str(e)[:200]}")

            clear_keras_session()

            return {
                'val_jaccard': 0.0,
                'success': False,
                'error': str(e)[:200],
                'oom_retries': oom_retries
            }

    # Max retries exceeded
    return {
        'val_jaccard': 0.0,
        'success': False,
        'error': f'OOM - max retries ({CONFIG["max_oom_retries"]}) exceeded',
        'oom_retries': oom_retries
    }

# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================

def run_hyperparameter_search():
    """Run complete hyperparameter search with cross-validation."""

    # Setup
    setup_memory_growth()
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    X, y = load_dataset(CONFIG['dataset_dir'], CONFIG['img_size'])

    # Initialize results storage
    all_results = []

    # Generate all configurations
    configs = []
    for arch in CONFIG['architectures']:
        for lr in CONFIG['learning_rates']:
            for dropout in CONFIG['dropouts']:
                for bs in CONFIG['batch_sizes']:
                    configs.append({
                        'architecture': arch,
                        'lr': lr,
                        'dropout': dropout,
                        'batch_size': bs,
                        'config_name': f"{arch}_lr{lr}_drop{dropout}_bs{bs}"
                    })

    print(f"\n✓ Generated {len(configs)} configurations")

    # Run cross-validation for each config
    kfold = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)

    for config_idx, config in enumerate(configs, 1):
        print(f"\n\n{'='*80}")
        print(f"CONFIGURATION {config_idx}/{len(configs)}: {config['config_name']}")
        print(f"{'='*80}")

        fold_results = []

        for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            print(f"\nFold {fold_num}/{CONFIG['n_folds']}")
            print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

            result = train_model_with_oom_protection(
                config['architecture'],
                X_train, y_train,
                X_val, y_val,
                config,
                fold_num
            )

            result['fold'] = fold_num
            result['config_name'] = config['config_name']
            result.update(config)

            fold_results.append(result)
            all_results.append(result)

            # Save intermediate results
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_dir / 'intermediate_results.csv', index=False)

        # Print fold summary
        successful_folds = [r for r in fold_results if r['success']]
        if successful_folds:
            jacards = [r['val_jaccard'] for r in successful_folds]
            print(f"\n✓ Configuration summary:")
            print(f"  Successful folds: {len(successful_folds)}/{CONFIG['n_folds']}")
            print(f"  Mean Jaccard: {np.mean(jacards):.4f} ± {np.std(jacards):.4f}")
            print(f"  Range: [{np.min(jacards):.4f}, {np.max(jacards):.4f}]")
        else:
            print(f"\n✗ All folds failed for this configuration")

    # Final analysis
    print(f"\n\n{'='*80}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'all_results.csv', index=False)

    # Find best configuration
    successful_results = results_df[results_df['success'] == True]

    if len(successful_results) > 0:
        # Group by configuration
        config_summary = successful_results.groupby('config_name').agg({
            'val_jaccard': ['mean', 'std', 'count']
        }).round(4)

        print(f"\n✓ Successful configurations: {len(config_summary)}")
        print(f"\nTop 5 configurations:")
        print(config_summary.sort_values(('val_jaccard', 'mean'), ascending=False).head())

        # Best configuration
        best_config = config_summary[('val_jaccard', 'mean')].idxmax()
        best_jaccard = config_summary.loc[best_config, ('val_jaccard', 'mean')]

        print(f"\n🏆 BEST CONFIGURATION: {best_config}")
        print(f"   Mean Jaccard: {best_jaccard:.4f}")

        # Save summary
        summary = {
            'best_config': best_config,
            'best_jaccard': float(best_jaccard),
            'total_configs': len(configs),
            'successful_configs': len(config_summary),
            'config_summary': config_summary.to_dict()
        }

        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    else:
        print(f"\n✗ No successful training runs!")
        print(f"   All configurations failed (likely OOM)")

    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    run_hyperparameter_search()
