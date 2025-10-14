#!/usr/bin/env python3
"""
Hyperparameter Search for ResUNet and Attention ResUNet
========================================================

PURPOSE:
Find optimal hyperparameters for residual architectures that failed with
baseline settings. Architecture comparison revealed:
- ResUNet: 39.95% (vs U-Net's 69.94%) - catastrophic failure
- Attention ResUNet: 62.69% (vs U-Net's 69.94%) - moderate underperformance

ROOT CAUSE HYPOTHESIS:
Residual connections change gradient flow, making baseline learning rate
(5e-5) too high. This causes:
- Early peaking (epoch 2-3)
- Training collapse
- High overfitting

SEARCH STRATEGY:
Grid search over critical hyperparameters:
- Learning rate (focus on lower values)
- Dropout (higher values for regularization)
- Batch size (affects gradient noise)

Use 3-fold CV for efficiency (instead of 5-fold).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import numpy as np
import pandas as pd
import cv2
import json
import itertools
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras.callbacks import (
    Callback, ModelCheckpoint, CSVLogger,
    EarlyStopping, ReduceLROnPlateau
)
import tensorflow as tf

# Import models
import sys
sys.path.append('/Users/xiaodan/unetCNN/unet-HPC')
from model_architectures import get_model
from loss_functions_fixed import (
    combined_dice_focal_loss,
    jacard_coef,
    dice_coef
)

# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================

ARCHITECTURES = ['resunet', 'attention_resunet']  # Focus on failed architectures

HYPERPARAMETER_GRID = {
    'learning_rate': [1e-5, 2e-5, 5e-5],  # Lower rates for ResUNet
    'dropout': [0.3, 0.4, 0.5],            # Higher dropout for regularization
    'batch_size': [4, 8],                  # Test gradient noise effect
}

# Fixed parameters
FIXED_CONFIG = {
    'loss_function': 'combined',
    'filters': 64,
    'n_folds': 3,  # Reduced for efficiency
    'epochs': 25,  # Slightly longer to observe full dynamics
}

EPOCHS = FIXED_CONFIG['epochs']
IMG_SIZE = 256
NUM_CLASSES = 1

print("="*80)
print("HYPERPARAMETER SEARCH FOR RESIDUAL ARCHITECTURES")
print("="*80)
print(f"\nArchitectures: {', '.join(ARCHITECTURES)}")
print(f"\nSearch space:")
for param, values in HYPERPARAMETER_GRID.items():
    print(f"  {param}: {values}")
print(f"\nFixed parameters:")
for param, value in FIXED_CONFIG.items():
    print(f"  {param}: {value}")

# Calculate total configurations
total_configs = (len(ARCHITECTURES) *
                 len(HYPERPARAMETER_GRID['learning_rate']) *
                 len(HYPERPARAMETER_GRID['dropout']) *
                 len(HYPERPARAMETER_GRID['batch_size']))

print(f"\n📊 Total configurations to test: {total_configs}")
print(f"   {len(ARCHITECTURES)} architectures × "
      f"{len(HYPERPARAMETER_GRID['learning_rate'])} LR × "
      f"{len(HYPERPARAMETER_GRID['dropout'])} dropout × "
      f"{len(HYPERPARAMETER_GRID['batch_size'])} batch_size")
print(f"   Each with {FIXED_CONFIG['n_folds']}-fold CV")
print(f"   Total models to train: {total_configs * FIXED_CONFIG['n_folds']}")

expected_time = total_configs * FIXED_CONFIG['n_folds'] * 10  # ~10 min per fold
print(f"\n⏱️  Estimated time: {expected_time/60:.1f} hours")
print("="*80)


class TerminateOnNaN(Callback):
    """Detect NaN/Inf in loss and terminate training immediately."""

    def __init__(self):
        super().__init__()
        self.nan_detected = False

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')

        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print(f'\n❌ BATCH {batch}: INVALID LOSS: {loss}')
                self.nan_detected = True
                self.model.stop_training = True


class SearchProgressCallback(Callback):
    """Track progress for hyperparameter search."""

    def __init__(self, config_name, fold_num):
        super().__init__()
        self.config_name = config_name
        self.fold_num = fold_num
        self.best_jacard = 0.0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_jacard = logs.get('val_jacard_coef', 0)

        if val_jacard > self.best_jacard:
            self.best_jacard = val_jacard
            self.best_epoch = epoch

        # Only print every 5 epochs to reduce clutter
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'  [{self.config_name}] Fold {self.fold_num} - '
                  f'Epoch {epoch+1}: val_jacard={val_jacard:.4f} '
                  f'(best={self.best_jacard:.4f} @ epoch {self.best_epoch+1})')


def load_data(images_dir, masks_dir, img_size=256):
    """Load and preprocess images and masks."""
    print("\n🔄 Loading dataset...")

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_files = sorted(images_dir.glob('*.tif'))

    images = []
    masks = []
    filenames = []
    densities = []

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        density = mask.mean()

        images.append(img)
        masks.append(mask)
        filenames.append(img_path.name)
        densities.append(density)

    images = np.array(images)[..., np.newaxis]
    masks = np.array(masks)[..., np.newaxis]
    densities = np.array(densities)

    print(f"✅ Loaded {len(images)} image-mask pairs")

    return images, masks, filenames, densities


def create_stratified_folds(densities, n_folds=3):
    """Create stratified folds based on density."""
    density_bins = pd.qcut(densities, q=4, labels=False, duplicates='drop')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    return skf.split(np.zeros(len(densities)), density_bins)


def train_config_fold(architecture, config, fold_num, X_train, y_train, X_val, y_val, save_dir):
    """Train one configuration on one fold."""

    # Build model
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    model = get_model(
        architecture,
        input_shape,
        NUM_CLASSES=1,
        dropout_rate=config['dropout'],
        batch_norm=True,
        filters=FIXED_CONFIG['filters']
    )

    # Compile
    loss_fn = combined_dice_focal_loss

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config['learning_rate'],
            clipnorm=1.0
        ),
        loss=loss_fn,
        metrics=['accuracy', jacard_coef, dice_coef]
    )

    # Callbacks
    config_name = (f"{architecture}_lr{config['learning_rate']:.0e}_"
                   f"drop{config['dropout']}_bs{config['batch_size']}")

    fold_dir = save_dir / config_name / f'fold_{fold_num}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    nan_callback = TerminateOnNaN()
    progress_callback = SearchProgressCallback(config_name, fold_num)

    callbacks = [
        nan_callback,
        progress_callback,
        ModelCheckpoint(
            str(fold_dir / 'best_model.keras'),
            monitor='val_jacard_coef',
            mode='max',
            save_best_only=True,
            verbose=0
        ),
        CSVLogger(
            str(fold_dir / 'history.csv'),
            append=False
        ),
        EarlyStopping(
            monitor='val_jacard_coef',
            mode='max',
            patience=8,  # Longer patience to observe dynamics
            restore_best_weights=True,
            verbose=0
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config['batch_size'],
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0  # Suppress epoch-by-epoch output
    )

    # Get results
    history_df = pd.read_csv(fold_dir / 'history.csv')

    results = {
        'architecture': architecture,
        'config': config,
        'config_name': config_name,
        'fold': fold_num,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'best_val_jacard': float(history_df['val_jacard_coef'].max()),
        'best_val_dice': float(history_df['val_dice_coef'].max()),
        'best_epoch': int(history_df['val_jacard_coef'].idxmax()),
        'final_val_jacard': float(history_df['val_jacard_coef'].iloc[-1]),
        'final_train_jacard': float(history_df['jacard_coef'].iloc[-1]),
        'overfitting_gap': float(history_df['jacard_coef'].iloc[-1] / max(history_df['val_jacard_coef'].iloc[-1], 1e-6)),
        'nan_detected': nan_callback.nan_detected,
        'epochs_trained': len(history_df)
    }

    # Save fold results
    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_hyperparameter_search(X, y, densities, save_dir):
    """Run full hyperparameter search."""

    # Create stratified folds (same for all configs)
    fold_splits = list(create_stratified_folds(densities, n_folds=FIXED_CONFIG['n_folds']))

    # Generate all configurations
    all_results = []
    config_counter = 0
    total_configs = len(ARCHITECTURES) * len(list(itertools.product(
        HYPERPARAMETER_GRID['learning_rate'],
        HYPERPARAMETER_GRID['dropout'],
        HYPERPARAMETER_GRID['batch_size']
    )))

    for architecture in ARCHITECTURES:
        print(f"\n{'='*80}")
        print(f"🏗️  ARCHITECTURE: {architecture.upper()}")
        print(f"{'='*80}")

        for lr in HYPERPARAMETER_GRID['learning_rate']:
            for dropout in HYPERPARAMETER_GRID['dropout']:
                for batch_size in HYPERPARAMETER_GRID['batch_size']:
                    config_counter += 1

                    config = {
                        'learning_rate': lr,
                        'dropout': dropout,
                        'batch_size': batch_size,
                    }

                    config_name = (f"{architecture}_lr{lr:.0e}_"
                                   f"drop{dropout}_bs{batch_size}")

                    print(f"\n[{config_counter}/{total_configs}] Testing: {config_name}")
                    print(f"  LR={lr:.0e}, Dropout={dropout}, BatchSize={batch_size}")

                    fold_results = []

                    for fold_num, (train_idx, val_idx) in enumerate(fold_splits, 1):
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        result = train_config_fold(
                            architecture, config, fold_num,
                            X_train, y_train, X_val, y_val,
                            save_dir
                        )
                        fold_results.append(result)

                    # Calculate CV statistics
                    best_jacards = [r['best_val_jacard'] for r in fold_results]
                    best_epochs = [r['best_epoch'] for r in fold_results]
                    overfitting_gaps = [r['overfitting_gap'] for r in fold_results]

                    cv_summary = {
                        'architecture': architecture,
                        'config': config,
                        'config_name': config_name,
                        'n_folds': FIXED_CONFIG['n_folds'],
                        'mean_best_jacard': float(np.mean(best_jacards)),
                        'std_best_jacard': float(np.std(best_jacards)),
                        'mean_best_epoch': float(np.mean(best_epochs)),
                        'mean_overfitting_gap': float(np.mean(overfitting_gaps)),
                        'fold_results': fold_results
                    }

                    all_results.append(cv_summary)

                    # Print summary
                    print(f"  ✅ CV Results: {cv_summary['mean_best_jacard']:.4f} ± {cv_summary['std_best_jacard']:.4f}")
                    print(f"     Best epoch: {cv_summary['mean_best_epoch']:.1f}, "
                          f"Overfitting gap: {cv_summary['mean_overfitting_gap']:.2f}×")

    return all_results


def analyze_results(all_results, save_dir):
    """Analyze hyperparameter search results."""

    print("\n" + "="*80)
    print("📊 HYPERPARAMETER SEARCH RESULTS ANALYSIS")
    print("="*80)

    # Organize by architecture
    by_arch = {}
    for result in all_results:
        arch = result['architecture']
        if arch not in by_arch:
            by_arch[arch] = []
        by_arch[arch].append(result)

    # Find best configuration for each architecture
    print("\n🏆 BEST CONFIGURATIONS:")
    print("-" * 80)

    best_configs = {}

    for arch in ARCHITECTURES:
        results = by_arch[arch]

        # Sort by mean best Jaccard
        results_sorted = sorted(results, key=lambda x: x['mean_best_jacard'], reverse=True)

        best = results_sorted[0]
        best_configs[arch] = best

        print(f"\n{arch.upper()}:")
        print(f"  Best config: {best['config_name']}")
        print(f"  Performance: {best['mean_best_jacard']:.4f} ± {best['std_best_jacard']:.4f}")
        print(f"  Hyperparameters:")
        for param, value in best['config'].items():
            print(f"    {param}: {value}")
        print(f"  Training dynamics:")
        print(f"    Best epoch: {best['mean_best_epoch']:.1f}")
        print(f"    Overfitting gap: {best['mean_overfitting_gap']:.2f}×")

    # Compare to baseline U-Net
    print("\n" + "="*80)
    print("📈 COMPARISON TO BASELINE U-NET")
    print("="*80)
    print("\nBaseline U-Net (from architecture comparison):")
    print("  Performance: 69.94% ± 5.02%")
    print("  Config: LR=5e-5, Dropout=0.3, BatchSize=4")

    for arch in ARCHITECTURES:
        best = best_configs[arch]
        baseline_perf = 0.6994
        improvement = ((best['mean_best_jacard'] - baseline_perf) / baseline_perf) * 100

        print(f"\n{arch.upper()} (optimized):")
        print(f"  Performance: {best['mean_best_jacard']:.4f} ± {best['std_best_jacard']:.4f}")
        print(f"  vs Baseline: {improvement:+.1f}%")

        if improvement > 0:
            print(f"  ✅ IMPROVEMENT FOUND!")
        elif improvement > -5:
            print(f"  📊 Close to baseline (within 5%)")
        else:
            print(f"  ❌ Still underperforming")

    # Top 5 configurations overall
    print("\n" + "="*80)
    print("🎯 TOP 5 CONFIGURATIONS (ALL ARCHITECTURES)")
    print("="*80)

    all_sorted = sorted(all_results, key=lambda x: x['mean_best_jacard'], reverse=True)

    for rank, result in enumerate(all_sorted[:5], 1):
        print(f"\n{rank}. {result['config_name']}")
        print(f"   Performance: {result['mean_best_jacard']:.4f} ± {result['std_best_jacard']:.4f}")
        print(f"   Config: LR={result['config']['learning_rate']:.0e}, "
              f"Dropout={result['config']['dropout']}, "
              f"BatchSize={result['config']['batch_size']}")

    # Save summary
    summary = {
        'search_space': HYPERPARAMETER_GRID,
        'fixed_config': FIXED_CONFIG,
        'architectures': ARCHITECTURES,
        'total_configurations': len(all_results),
        'best_configs': {arch: best_configs[arch] for arch in ARCHITECTURES},
        'all_results': all_results,
        'baseline_reference': {
            'unet_performance': 0.6994,
            'unet_std': 0.0502,
            'unet_config': {'learning_rate': 5e-5, 'dropout': 0.3, 'batch_size': 4}
        }
    }

    with open(save_dir / 'hyperparameter_search_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*80)
    print("💾 Results saved to:")
    print(f"   {save_dir / 'hyperparameter_search_summary.json'}")
    print("="*80)

    return summary


def main():
    """Main hyperparameter search workflow."""

    print("\n🔍 STARTING HYPERPARAMETER SEARCH")
    print("   Target: Find optimal settings for ResUNet & Attention ResUNet")
    print("   Goal: Match or exceed U-Net's 69.94% Jaccard\n")

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"hyperparameter_search_{timestamp}")
    save_dir.mkdir(exist_ok=True)

    print(f"💾 Save directory: {save_dir}\n")

    # Load data
    script_dir = Path(__file__).parent
    images_dir = script_dir / "dataset_full_stack" / "images"
    masks_dir = script_dir / "dataset_full_stack" / "masks"

    X, y, filenames, densities = load_data(str(images_dir), str(masks_dir), IMG_SIZE)

    # Run search
    all_results = run_hyperparameter_search(X, y, densities, save_dir)

    # Analyze
    summary = analyze_results(all_results, save_dir)

    print("\n🏁 HYPERPARAMETER SEARCH COMPLETE!")

    return 0


if __name__ == "__main__":
    exit(main())
