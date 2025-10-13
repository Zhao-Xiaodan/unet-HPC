#!/usr/bin/env python3
"""
Architecture Comparison Study - ResUNet vs Attention ResUNet vs U-Net
======================================================================

PURPOSE:
Compare three U-Net architectures on microbead segmentation:
1. Standard U-Net (baseline: 60.97% ± 11.5% CV mean)
2. ResUNet (with residual connections)
3. Attention ResUNet (with residual connections + attention gates)

METHODOLOGY:
- 5-fold cross-validation for reliable comparison
- SAME hyperparameters across architectures:
  * filters=64 (baseline configuration)
  * dropout=0.3
  * combined loss (0.7×Dice + 0.3×Focal)
  * batch_size=4
- Metrics tracked:
  * Segmentation performance (Jaccard, Dice)
  * Training time per epoch
  * Model complexity (parameter count)
  * Convergence speed (best epoch)
  * Overfitting gap

EXPECTED OUTCOMES:
- ResUNet: Better gradient flow → faster convergence, potentially +2-5%
- Attention ResUNet: Boundary focus → better overlapping objects, potentially +3-7%
- Trade-off analysis: performance vs computational cost

BASELINE REFERENCE:
U-Net CV results from validation_cv_20251013_052113:
- Mean best val Jaccard: 60.97% ± 11.5%
- Overfitting gap: 1.93× ± 0.43×
- Best epoch: 9.6 ± 5.2
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import numpy as np
import pandas as pd
import cv2
import json
import time
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
from tensorflow.keras.callbacks import (
    Callback, ModelCheckpoint, CSVLogger,
    EarlyStopping, ReduceLROnPlateau
)
import tensorflow as tf

# Import models and loss functions
import sys
sys.path.append('/Users/xiaodan/unetCNN/unet-HPC')
from model_architectures import get_model, MODEL_ARCHITECTURES
from loss_functions_fixed import (
    combined_dice_focal_loss,
    jacard_coef,
    dice_coef
)

# Configuration - BASELINE from validated CV study
BASE_CONFIG = {
    'batch_size': 4,
    'dropout': 0.3,
    'loss_function': 'combined',
    'filters': 64,
    'n_folds': 5,  # Change to 3 if you want faster testing
}

# Architectures to test
ARCHITECTURES = ['unet', 'resunet', 'attention_resunet']

EPOCHS = 20
IMG_SIZE = 256
NUM_CLASSES = 1


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


class TimingCallback(Callback):
    """Track training time per epoch."""

    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_start is not None:
            self.epoch_times.append(time.time() - self.epoch_start)


class ArchitectureProgressCallback(Callback):
    """Track progress within each architecture-fold combination."""

    def __init__(self, architecture, fold_num):
        super().__init__()
        self.architecture = architecture
        self.fold_num = fold_num
        self.best_jacard = 0.0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_jacard = logs.get('val_jacard_coef', 0)
        train_jacard = logs.get('jacard_coef', 0)
        gap = train_jacard / max(val_jacard, 1e-6)

        if val_jacard > self.best_jacard:
            self.best_jacard = val_jacard
            self.best_epoch = epoch

        print(f'\n📊 {self.architecture.upper()} - FOLD {self.fold_num} - EPOCH {epoch+1}/{EPOCHS}:')
        print(f'   Train Jaccard:   {train_jacard:.4f}')
        print(f'   Val Jaccard:     {val_jacard:.4f}')
        print(f'   Overfitting Gap: {gap:.1f}×')
        print(f'   Best Val:        {self.best_jacard:.4f} (epoch {self.best_epoch+1})')


def load_data(images_dir, masks_dir, img_size=256):
    """Load and preprocess images and masks with metadata."""
    print("\n🔄 Loading dataset...")

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_files = sorted(images_dir.glob('*.tif'))

    images = []
    masks = []
    filenames = []
    densities = []

    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Load mask
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Resize
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        # Normalize
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # Calculate density
        density = mask.mean()

        images.append(img)
        masks.append(mask)
        filenames.append(img_path.name)
        densities.append(density)

    images = np.array(images)[..., np.newaxis]
    masks = np.array(masks)[..., np.newaxis]
    densities = np.array(densities)

    print(f"✅ Loaded {len(images)} image-mask pairs")
    print(f"   Image shape: {images.shape}")
    print(f"   Mask shape: {masks.shape}")

    return images, masks, filenames, densities


def create_stratified_folds(densities, n_folds=5):
    """Create stratified folds based on density."""
    density_bins = pd.qcut(densities, q=4, labels=False, duplicates='drop')

    print(f"\n📊 Creating {n_folds}-fold stratified split")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    return skf.split(np.zeros(len(densities)), density_bins)


def train_architecture_fold(architecture, fold_num, X_train, y_train, X_val, y_val, save_dir):
    """Train one architecture on one fold."""

    print(f"\n{'='*80}")
    print(f"{architecture.upper()} - FOLD {fold_num}/{BASE_CONFIG['n_folds']}: TRAINING")
    print(f"{'='*80}")

    # Build model
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    model = get_model(
        architecture,
        input_shape,
        NUM_CLASSES=1,
        dropout_rate=BASE_CONFIG['dropout'],
        batch_norm=True,
        filters=BASE_CONFIG['filters']
    )

    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

    print(f"\n   Model: {model.name}")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Training samples:     {len(X_train)}")
    print(f"   Validation samples:   {len(X_val)}")

    # Compile
    loss_fn = combined_dice_focal_loss

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=5e-5,
            clipnorm=1.0
        ),
        loss=loss_fn,
        metrics=['accuracy', jacard_coef, dice_coef]
    )

    # Callbacks
    fold_dir = save_dir / architecture / f'fold_{fold_num}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    nan_callback = TerminateOnNaN()
    timing_callback = TimingCallback()
    progress_callback = ArchitectureProgressCallback(architecture, fold_num)

    callbacks = [
        nan_callback,
        timing_callback,
        progress_callback,
        ModelCheckpoint(
            str(fold_dir / 'best_model.keras'),
            monitor='val_jacard_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(
            str(fold_dir / 'history.csv'),
            append=False
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_jacard_coef',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train
    print(f"\n🚀 Starting training...")

    train_start = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BASE_CONFIG['batch_size'],
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    train_duration = time.time() - train_start

    # Get results
    history_df = pd.read_csv(fold_dir / 'history.csv')

    results = {
        'architecture': architecture,
        'fold': fold_num,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'train_density': float(y_train.mean()),
        'val_density': float(y_val.mean()),
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'best_val_jacard': float(history_df['val_jacard_coef'].max()),
        'best_val_dice': float(history_df['val_dice_coef'].max()),
        'best_epoch': int(history_df['val_jacard_coef'].idxmax()),
        'final_val_jacard': float(history_df['val_jacard_coef'].iloc[-1]),
        'final_train_jacard': float(history_df['jacard_coef'].iloc[-1]),
        'min_val_jacard': float(history_df['val_jacard_coef'].min()),
        'overfitting_gap': float(history_df['jacard_coef'].iloc[-1] / max(history_df['val_jacard_coef'].iloc[-1], 1e-6)),
        'nan_detected': nan_callback.nan_detected,
        'epochs_trained': len(history_df),
        'total_training_time_sec': float(train_duration),
        'avg_epoch_time_sec': float(np.mean(timing_callback.epoch_times)) if timing_callback.epoch_times else 0,
        'epoch_times': [float(t) for t in timing_callback.epoch_times]
    }

    print(f"\n✅ {architecture.upper()} - FOLD {fold_num} COMPLETE:")
    print(f"   Best Val Jaccard:     {results['best_val_jacard']:.4f} (epoch {results['best_epoch']+1})")
    print(f"   Training time:        {train_duration:.1f}s ({results['avg_epoch_time_sec']:.1f}s/epoch)")
    print(f"   Overfitting Gap:      {results['overfitting_gap']:.1f}×")

    # Save fold results
    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def analyze_architecture_results(all_results, save_dir):
    """Analyze and compare results across all architectures."""

    print("\n" + "="*80)
    print("📊 ARCHITECTURE COMPARISON - FINAL ANALYSIS")
    print("="*80)

    # Organize by architecture
    by_arch = {}
    for result in all_results:
        arch = result['architecture']
        if arch not in by_arch:
            by_arch[arch] = []
        by_arch[arch].append(result)

    # Calculate statistics for each architecture
    comparison = {}

    for arch in ARCHITECTURES:
        if arch not in by_arch:
            continue

        df = pd.DataFrame(by_arch[arch])

        stats = {
            'architecture': arch,
            'n_folds': len(by_arch[arch]),
            'total_parameters': int(df['total_parameters'].iloc[0]),
            'best_val_jacard': {
                'mean': float(df['best_val_jacard'].mean()),
                'std': float(df['best_val_jacard'].std()),
                'min': float(df['best_val_jacard'].min()),
                'max': float(df['best_val_jacard'].max()),
                'values': df['best_val_jacard'].tolist()
            },
            'best_val_dice': {
                'mean': float(df['best_val_dice'].mean()),
                'std': float(df['best_val_dice'].std()),
            },
            'overfitting_gap': {
                'mean': float(df['overfitting_gap'].mean()),
                'std': float(df['overfitting_gap'].std()),
            },
            'best_epoch': {
                'mean': float(df['best_epoch'].mean()),
                'std': float(df['best_epoch'].std()),
                'values': df['best_epoch'].tolist()
            },
            'avg_epoch_time_sec': {
                'mean': float(df['avg_epoch_time_sec'].mean()),
                'std': float(df['avg_epoch_time_sec'].std()),
            },
            'total_training_time_sec': {
                'mean': float(df['total_training_time_sec'].mean()),
                'std': float(df['total_training_time_sec'].std()),
            }
        }

        comparison[arch] = stats

        print(f"\n🏗️  {arch.upper()} RESULTS:")
        print(f"   Parameters:           {stats['total_parameters']:,}")
        print(f"   Best Val Jaccard:     {stats['best_val_jacard']['mean']:.4f} ± {stats['best_val_jacard']['std']:.4f}")
        print(f"   Best Val Dice:        {stats['best_val_dice']['mean']:.4f} ± {stats['best_val_dice']['std']:.4f}")
        print(f"   Overfitting Gap:      {stats['overfitting_gap']['mean']:.1f}× ± {stats['overfitting_gap']['std']:.1f}×")
        print(f"   Best Epoch:           {stats['best_epoch']['mean']:.1f} ± {stats['best_epoch']['std']:.1f}")
        print(f"   Avg Epoch Time:       {stats['avg_epoch_time_sec']['mean']:.1f}s ± {stats['avg_epoch_time_sec']['std']:.1f}s")

    # Comparison table
    print("\n" + "="*80)
    print("📊 ARCHITECTURE COMPARISON TABLE")
    print("="*80)

    print(f"\n{'Metric':<25} | {'U-Net':<20} | {'ResUNet':<20} | {'Attention ResUNet':<20}")
    print("-" * 90)

    # Parameters
    if all(arch in comparison for arch in ARCHITECTURES):
        print(f"{'Parameters':<25} | {comparison['unet']['total_parameters']:>17,}   | {comparison['resunet']['total_parameters']:>17,}   | {comparison['attention_resunet']['total_parameters']:>17,}")

        # Jaccard
        unet_jac = comparison['unet']['best_val_jacard']['mean']
        resunet_jac = comparison['resunet']['best_val_jacard']['mean']
        attn_jac = comparison['attention_resunet']['best_val_jacard']['mean']

        print(f"{'Best Val Jaccard':<25} | {unet_jac:>17.4f}   | {resunet_jac:>17.4f}   | {attn_jac:>17.4f}")

        # Improvement
        resunet_imp = ((resunet_jac - unet_jac) / unet_jac) * 100
        attn_imp = ((attn_jac - unet_jac) / unet_jac) * 100

        print(f"{'Improvement vs U-Net':<25} | {'-':<20} | {resunet_imp:>17.1f}%  | {attn_imp:>17.1f}%")

        # Training time
        unet_time = comparison['unet']['avg_epoch_time_sec']['mean']
        resunet_time = comparison['resunet']['avg_epoch_time_sec']['mean']
        attn_time = comparison['attention_resunet']['avg_epoch_time_sec']['mean']

        print(f"{'Avg Epoch Time (s)':<25} | {unet_time:>17.1f}   | {resunet_time:>17.1f}   | {attn_time:>17.1f}")

        # Time overhead
        resunet_overhead = ((resunet_time - unet_time) / unet_time) * 100
        attn_overhead = ((attn_time - unet_time) / unet_time) * 100

        print(f"{'Time Overhead':<25} | {'-':<20} | {resunet_overhead:>17.1f}%  | {attn_overhead:>17.1f}%")

    # Ranking
    print("\n📊 RANKING:")

    if all(arch in comparison for arch in ARCHITECTURES):
        ranked = sorted(comparison.items(), key=lambda x: x[1]['best_val_jacard']['mean'], reverse=True)

        for rank, (arch, stats) in enumerate(ranked, 1):
            medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, '  ')
            print(f"   {medal} {rank}. {arch.upper()}: {stats['best_val_jacard']['mean']:.4f} ± {stats['best_val_jacard']['std']:.4f}")

    # Save summary
    summary = {
        'config': BASE_CONFIG,
        'architectures_tested': ARCHITECTURES,
        'comparison': comparison,
        'all_results': all_results,
        'baseline_reference': {
            'source': 'validation_cv_20251013_052113',
            'unet_cv_mean': 0.6097,
            'unet_cv_std': 0.1154
        }
    }

    with open(save_dir / 'architecture_comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {save_dir}")
    print(f"   - architecture_comparison_summary.json")
    print(f"   - {arch}/fold_*/history.csv")
    print(f"   - {arch}/fold_*/results.json")

    print("\n" + "="*80)
    print("🏁 ARCHITECTURE COMPARISON COMPLETE")
    print("="*80)

    return summary


def main():
    """Main architecture comparison workflow."""

    print("="*80)
    print("🏗️  ARCHITECTURE COMPARISON STUDY")
    print("="*80)
    print(f"\nTesting architectures: {', '.join([a.upper() for a in ARCHITECTURES])}")
    print(f"CV folds: {BASE_CONFIG['n_folds']}")
    print(f"Configuration: filters={BASE_CONFIG['filters']}, dropout={BASE_CONFIG['dropout']}")

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"validation_arch_comparison_{timestamp}")
    save_dir.mkdir(exist_ok=True)

    print(f"\n💾 Save directory: {save_dir}")

    # Load data
    script_dir = Path(__file__).parent
    images_dir = script_dir / "dataset_full_stack" / "images"
    masks_dir = script_dir / "dataset_full_stack" / "masks"

    X, y, filenames, densities = load_data(str(images_dir), str(masks_dir), IMG_SIZE)

    # Create stratified folds (same for all architectures)
    fold_splits = list(create_stratified_folds(densities, n_folds=BASE_CONFIG['n_folds']))

    # Train all architectures
    all_results = []

    for architecture in ARCHITECTURES:
        print(f"\n{'='*80}")
        print(f"🏗️  TESTING ARCHITECTURE: {architecture.upper()}")
        print(f"{'='*80}")

        for fold_num, (train_idx, val_idx) in enumerate(fold_splits, 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            results = train_architecture_fold(
                architecture, fold_num,
                X_train, y_train, X_val, y_val,
                save_dir
            )
            all_results.append(results)

    # Analyze and compare
    summary = analyze_architecture_results(all_results, save_dir)

    return 0


if __name__ == "__main__":
    exit(main())
