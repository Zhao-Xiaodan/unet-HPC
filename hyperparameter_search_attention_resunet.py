#!/usr/bin/env python3
"""
Hyperparameter Search for Attention ResUNet
============================================

Based on ResUNet hyperparameter search results (hyperparameter_search_20251013_154754),
this script searches for optimal hyperparameters for Attention ResUNet.

Key Insights from ResUNet Search:
---------------------------------
1. Learning Rate DOMINATES performance (2× effect)
   - Optimal: 2e-05
   - Too low (1e-05): Undertraining
   - Too high (5e-05): Training collapse

2. Lower Dropout BETTER (inverse relationship)
   - Best: 0.3
   - Model appears to underfit, not overfit

3. Batch Size: Minimal effect
   - Slightly better with BS=4

Search Strategy for Attention ResUNet:
---------------------------------------
- Focus learning rate around 2e-05 (test 1.5e-05, 2e-05, 2.5e-05)
- Lower dropout range (0.2, 0.3, 0.4)
- Keep batch sizes 4 and 8
- Total: 3 LR × 3 dropout × 2 batch = 18 configs × 3 folds = 54 models

Target Performance:
------------------
- U-Net baseline: 69.94% Jaccard
- ResUNet optimal: 42.40% Jaccard
- Goal: >55% Jaccard (to justify further investigation)

Author: Claude Code
Date: October 14, 2025
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from pathlib import Path
import sys

# Import custom modules
from model_architectures import attention_resunet
from loss_functions_fixed import combined_loss, dice_loss, focal_loss
from loss_functions_fixed import jacard_coef, dice_coef

# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

# Focused search space based on ResUNet findings
HYPERPARAMETER_GRID = {
    'learning_rate': [1.5e-5, 2e-5, 2.5e-5],  # Centered around ResUNet optimum
    'dropout': [0.2, 0.3, 0.4],                # Lower range (higher dropout hurt)
    'batch_size': [4, 8],                      # Small batches slightly better
}

# Fixed parameters (based on successful ResUNet configuration)
FIXED_CONFIG = {
    'architecture': 'attention_resunet',
    'loss_function': 'combined',  # 0.7×Dice + 0.3×Focal
    'filters': 64,                 # Base filters
    'n_folds': 3,                  # 3-fold CV for efficiency
    'epochs': 30,                  # Slightly more than ResUNet (25)
    'early_stopping_patience': 8,  # Standard patience
    'image_size': 256,
    'channels': 1,
}

# Dataset configuration
DATASET_CONFIG = {
    'images_dir': './dataset_full_stack/images/',
    'masks_dir': './dataset_full_stack/masks/',
    'target_size': (256, 256),
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_gpu():
    """Configure GPU memory growth to prevent OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            print(f"⚠ GPU configuration warning: {e}")
            return False
    else:
        print("⚠ No GPU detected - training will be VERY slow!")
        return False


def create_output_directory():
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'attention_resunet_search_{timestamp}')
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    return output_dir


def load_dataset(images_dir, masks_dir, target_size=(256, 256)):
    """
    Load images and masks from directories.
    Returns: images, masks, filenames
    """
    from PIL import Image

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    # Find all .tif files
    image_files = sorted(images_dir.glob('*.tif'))

    if len(image_files) == 0:
        raise ValueError(f"No .tif files found in {images_dir}")

    images = []
    masks = []
    filenames = []

    print(f"Loading {len(image_files)} image pairs...")

    for img_path in image_files:
        # Load image
        img = Image.open(img_path).convert('L')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0

        # Load corresponding mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            print(f"⚠ Warning: Missing mask for {img_path.name}, skipping...")
            continue

        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(target_size)
        mask_array = np.array(mask) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)

        images.append(img_array)
        masks.append(mask_array)
        filenames.append(img_path.name)

    images = np.array(images)[..., np.newaxis]
    masks = np.array(masks)[..., np.newaxis]

    print(f"✓ Loaded {len(images)} images with shape {images.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
    print(f"  Positive pixels: {masks.mean()*100:.2f}%")

    return images, masks, filenames


def calculate_stratification_bins(masks, n_bins=5):
    """
    Create stratification bins based on positive pixel density.
    This ensures balanced CV folds.
    """
    densities = masks.mean(axis=(1, 2, 3))
    bins = np.percentile(densities, np.linspace(0, 100, n_bins + 1))
    bins[0] = -0.01  # Ensure all values are captured
    bins[-1] = 1.01

    strat_labels = np.digitize(densities, bins) - 1

    print(f"✓ Stratification bins: {len(np.unique(strat_labels))} bins")
    for i in range(len(np.unique(strat_labels))):
        count = np.sum(strat_labels == i)
        print(f"  Bin {i}: {count} images")

    return strat_labels


def get_loss_function(loss_name):
    """Return loss function by name."""
    if loss_name == 'combined':
        return combined_loss
    elif loss_name == 'dice':
        return dice_loss
    elif loss_name == 'focal':
        return focal_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def check_for_nans(history):
    """Check if training produced NaN values."""
    for metric_values in history.history.values():
        if any(np.isnan(metric_values)):
            return True
    return False


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_single_configuration(config, fold_data, output_dir):
    """
    Train Attention ResUNet for one configuration and one fold.

    Args:
        config: Dict with hyperparameters (learning_rate, dropout, batch_size)
        fold_data: Dict with train/val data (X_train, y_train, X_val, y_val)
        output_dir: Path to save results

    Returns:
        Dict with results (best_val_jacard, best_epoch, etc.)
    """
    # Unpack configuration
    learning_rate = config['learning_rate']
    dropout = config['dropout']
    batch_size = config['batch_size']

    X_train = fold_data['X_train']
    y_train = fold_data['y_train']
    X_val = fold_data['X_val']
    y_val = fold_data['y_val']
    fold = fold_data['fold']

    print(f"\n{'='*70}")
    print(f"Configuration: LR={learning_rate}, Dropout={dropout}, BS={batch_size}, Fold={fold}")
    print(f"{'='*70}")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # Create model
    input_shape = (FIXED_CONFIG['image_size'], FIXED_CONFIG['image_size'], FIXED_CONFIG['channels'])

    model = attention_resunet(
        input_shape=input_shape,
        filters=FIXED_CONFIG['filters'],
        dropout=dropout
    )

    # Compile model
    loss_fn = get_loss_function(FIXED_CONFIG['loss_function'])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[jacard_coef, dice_coef]
    )

    print(f"✓ Model compiled with {FIXED_CONFIG['loss_function']} loss")

    # Callbacks
    config_name = f"attention_resunet_lr{learning_rate:.0e}_drop{dropout}_bs{batch_size}"
    fold_dir = output_dir / config_name / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_jacard_coef',
            patience=FIXED_CONFIG['early_stopping_patience'],
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
        ),
        keras.callbacks.CSVLogger(
            fold_dir / 'training_history.csv'
        )
    ]

    # Train model
    print(f"Training for up to {FIXED_CONFIG['epochs']} epochs...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=FIXED_CONFIG['epochs'],
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Check for NaN
    nan_detected = check_for_nans(history)
    if nan_detected:
        print("⚠ WARNING: NaN values detected during training!")

    # Extract results
    val_jacards = history.history['val_jacard_coef']
    best_epoch = np.argmax(val_jacards)
    best_val_jacard = val_jacards[best_epoch]
    best_val_dice = history.history['val_dice_coef'][best_epoch]

    final_val_jacard = val_jacards[-1]
    final_train_jacard = history.history['jacard_coef'][-1]

    # Calculate overfitting gap
    overfitting_gap = final_train_jacard / final_val_jacard if final_val_jacard > 0 else np.inf

    print(f"\n{'='*70}")
    print(f"FOLD {fold} RESULTS:")
    print(f"{'='*70}")
    print(f"Best Val Jaccard:  {best_val_jacard:.4f} (epoch {best_epoch})")
    print(f"Best Val Dice:     {best_val_dice:.4f}")
    print(f"Final Val Jaccard: {final_val_jacard:.4f}")
    print(f"Final Train Jaccard: {final_train_jacard:.4f}")
    print(f"Overfitting Gap:   {overfitting_gap:.2f}×")
    print(f"Epochs Trained:    {len(val_jacards)}")
    print(f"NaN Detected:      {nan_detected}")
    print(f"{'='*70}\n")

    # Save model
    model.save(fold_dir / 'best_model.keras')
    print(f"✓ Model saved to {fold_dir / 'best_model.keras'}")

    # Save results
    results = {
        'architecture': 'attention_resunet',
        'config': {
            'learning_rate': float(learning_rate),
            'dropout': float(dropout),
            'batch_size': int(batch_size),
        },
        'config_name': config_name,
        'fold': int(fold),
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'best_val_jacard': float(best_val_jacard),
        'best_val_dice': float(best_val_dice),
        'best_epoch': int(best_epoch),
        'final_val_jacard': float(final_val_jacard),
        'final_train_jacard': float(final_train_jacard),
        'overfitting_gap': float(overfitting_gap),
        'nan_detected': bool(nan_detected),
        'epochs_trained': int(len(val_jacards)),
    }

    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {fold_dir / 'results.json'}")

    # Clear session to free memory
    keras.backend.clear_session()

    return results


# =============================================================================
# MAIN HYPERPARAMETER SEARCH
# =============================================================================

def main():
    """Main hyperparameter search loop."""

    print("\n" + "="*80)
    print("ATTENTION RESUNET HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup
    print("="*80)
    print("ENVIRONMENT SETUP")
    print("="*80)
    setup_gpu()
    output_dir = create_output_directory()
    print()

    # Load dataset
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    images, masks, filenames = load_dataset(
        DATASET_CONFIG['images_dir'],
        DATASET_CONFIG['masks_dir'],
        DATASET_CONFIG['target_size']
    )
    print()

    # Create stratification bins
    strat_labels = calculate_stratification_bins(masks, n_bins=5)

    # Generate all hyperparameter combinations
    configs = []
    for lr in HYPERPARAMETER_GRID['learning_rate']:
        for dropout in HYPERPARAMETER_GRID['dropout']:
            for bs in HYPERPARAMETER_GRID['batch_size']:
                configs.append({
                    'learning_rate': lr,
                    'dropout': dropout,
                    'batch_size': bs,
                })

    print(f"\n{'='*80}")
    print(f"SEARCH SPACE")
    print(f"{'='*80}")
    print(f"Learning Rates: {HYPERPARAMETER_GRID['learning_rate']}")
    print(f"Dropout:        {HYPERPARAMETER_GRID['dropout']}")
    print(f"Batch Sizes:    {HYPERPARAMETER_GRID['batch_size']}")
    print(f"")
    print(f"Total Configurations: {len(configs)}")
    print(f"Folds per Config:     {FIXED_CONFIG['n_folds']}")
    print(f"Total Models:         {len(configs) * FIXED_CONFIG['n_folds']}")
    print(f"{'='*80}\n")

    # Setup cross-validation
    skf = StratifiedKFold(n_splits=FIXED_CONFIG['n_folds'], shuffle=True, random_state=42)

    # Store all results
    all_results = []

    # Loop over configurations
    for config_idx, config in enumerate(configs, 1):
        print(f"\n{'#'*80}")
        print(f"# CONFIG {config_idx}/{len(configs)}: LR={config['learning_rate']:.0e}, "
              f"Dropout={config['dropout']}, Batch={config['batch_size']}")
        print(f"{'#'*80}\n")

        # Loop over folds
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(images, strat_labels), 1):
            X_train = images[train_indices]
            y_train = masks[train_indices]
            X_val = images[val_indices]
            y_val = masks[val_indices]

            fold_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'fold': fold_idx,
            }

            try:
                results = train_single_configuration(config, fold_data, output_dir)
                all_results.append(results)

                # Print progress
                completed = (config_idx - 1) * FIXED_CONFIG['n_folds'] + fold_idx
                total = len(configs) * FIXED_CONFIG['n_folds']
                print(f"\n{'>'*80}")
                print(f"PROGRESS: {completed}/{total} models completed ({completed/total*100:.1f}%)")
                print(f"{'>'*80}\n")

            except Exception as e:
                print(f"\n{'!'*80}")
                print(f"ERROR in config {config_idx}, fold {fold_idx}:")
                print(f"{str(e)}")
                print(f"{'!'*80}\n")
                continue

    # Aggregate results by configuration
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}\n")

    config_summaries = {}

    for config in configs:
        config_name = f"attention_resunet_lr{config['learning_rate']:.0e}_drop{config['dropout']}_bs{config['batch_size']}"

        # Find all results for this config
        config_results = [r for r in all_results if r['config_name'] == config_name]

        if not config_results:
            print(f"⚠ No results for {config_name}")
            continue

        # Aggregate metrics
        best_jacards = [r['best_val_jacard'] for r in config_results]
        best_epochs = [r['best_epoch'] for r in config_results]
        overfitting_gaps = [r['overfitting_gap'] for r in config_results if np.isfinite(r['overfitting_gap'])]

        config_summaries[config_name] = {
            'config': config,
            'n_folds': len(config_results),
            'mean_best_jacard': float(np.mean(best_jacards)),
            'std_best_jacard': float(np.std(best_jacards)),
            'min_best_jacard': float(np.min(best_jacards)),
            'max_best_jacard': float(np.max(best_jacards)),
            'mean_best_epoch': float(np.mean(best_epochs)),
            'mean_overfitting_gap': float(np.mean(overfitting_gaps)) if overfitting_gaps else np.inf,
        }

        print(f"{config_name}:")
        print(f"  Mean Jaccard: {config_summaries[config_name]['mean_best_jacard']:.4f} ± {config_summaries[config_name]['std_best_jacard']:.4f}")
        print(f"  Range: [{config_summaries[config_name]['min_best_jacard']:.4f}, {config_summaries[config_name]['max_best_jacard']:.4f}]")
        print(f"  Mean Best Epoch: {config_summaries[config_name]['mean_best_epoch']:.1f}")
        print(f"  Overfitting Gap: {config_summaries[config_name]['mean_overfitting_gap']:.2f}×")
        print()

    # Find best configuration
    if config_summaries:
        best_config_name = max(config_summaries, key=lambda k: config_summaries[k]['mean_best_jacard'])
        best_config_data = config_summaries[best_config_name]

        print(f"\n{'='*80}")
        print("BEST CONFIGURATION FOUND")
        print(f"{'='*80}")
        print(f"Config: {best_config_name}")
        print(f"Mean Jaccard: {best_config_data['mean_best_jacard']:.4f} ± {best_config_data['std_best_jacard']:.4f}")
        print(f"Range: [{best_config_data['min_best_jacard']:.4f}, {best_config_data['max_best_jacard']:.4f}]")
        print(f"Mean Best Epoch: {best_config_data['mean_best_epoch']:.1f}")
        print(f"Overfitting Gap: {best_config_data['mean_overfitting_gap']:.2f}×")
        print()

        # Compare to baselines
        unet_baseline = 0.6994
        resunet_baseline = 0.3995
        resunet_optimized = 0.4240

        best_perf = best_config_data['mean_best_jacard']

        print(f"COMPARISON TO BASELINES:")
        print(f"{'='*80}")
        print(f"U-Net:               {unet_baseline:.4f} (goal)")
        print(f"ResUNet (baseline):  {resunet_baseline:.4f}")
        print(f"ResUNet (optimized): {resunet_optimized:.4f}")
        print(f"Attention ResUNet:   {best_perf:.4f} ← THIS RESULT")
        print()
        print(f"vs U-Net:            {(best_perf - unet_baseline)/unet_baseline*100:+.1f}%")
        print(f"vs ResUNet baseline: {(best_perf - resunet_baseline)/resunet_baseline*100:+.1f}%")
        print(f"vs ResUNet optimized: {(best_perf - resunet_optimized)/resunet_optimized*100:+.1f}%")
        print()

        # Decision criteria
        print(f"ASSESSMENT:")
        print(f"{'='*80}")
        if best_perf > unet_baseline:
            print(f"✅ SUCCESS! Attention ResUNet EXCEEDS U-Net baseline!")
            print(f"   Recommend: Use Attention ResUNet as primary architecture")
        elif best_perf > 0.55:
            print(f"✅ PROMISING! Attention ResUNet shows improvement over ResUNet.")
            print(f"   Recommend: Run full 5-fold validation to confirm")
        elif best_perf > resunet_optimized:
            print(f"⚠️  MARGINAL: Attention gates provide small improvement.")
            print(f"   Recommend: Decide based on computational cost vs accuracy trade-off")
        else:
            print(f"❌ FAILURE: Attention ResUNet no better than ResUNet.")
            print(f"   Recommend: Abandon residual architectures, stick with U-Net")
        print(f"{'='*80}\n")

        # Save summary
        summary = {
            'search_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'search_space': HYPERPARAMETER_GRID,
            'fixed_config': FIXED_CONFIG,
            'n_configurations': len(configs),
            'n_folds': FIXED_CONFIG['n_folds'],
            'total_models_trained': len(all_results),
            'best_config': {
                'config_name': best_config_name,
                **best_config_data
            },
            'all_configs': config_summaries,
            'baselines': {
                'unet': unet_baseline,
                'resunet_baseline': resunet_baseline,
                'resunet_optimized': resunet_optimized,
            }
        }

        with open(output_dir / 'attention_resunet_search_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Summary saved to {output_dir / 'attention_resunet_search_summary.json'}")
    else:
        print("\n❌ ERROR: No configurations completed successfully!")

    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
