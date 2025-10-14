#!/usr/bin/env python3
"""
5-Fold Cross-Validation - PRIORITY 1 CRITICAL TEST

PURPOSE:
Get reliable performance estimates using cross-validation to overcome
the validation set bias problem identified in all previous tests.

PROBLEM IDENTIFIED:
All 3 tests (Phase 1, Focal Tversky, Small Model) peaked at epoch 1,
indicating validation set is not representative of training data.

SOLUTION:
5-fold stratified cross-validation provides:
- 5 different train/val splits
- Average performance across folds (more reliable)
- Confidence intervals
- Detection of split bias

EXPECTED OUTCOMES:
- If CV avg >> current val (13.8% best) → current split was biased
- If CV avg ~ current val → task is genuinely hard
- Fold variance reveals stability

CONFIGURATION:
Using BASELINE from Phase 1 for fair comparison:
- filters=64 (31M params) - proven capacity
- dropout=0.3 - original value
- combined loss - best from hyperparameter search
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import numpy as np
import pandas as pd
import cv2
import json
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

# Configuration - BASELINE from Phase 1
CONFIG = {
    'architecture': 'unet',
    'batch_size': 4,
    'dropout': 0.3,  # Original value (not 0.5)
    'loss_function': 'combined',
    'filters': 64,  # Original value (not 16)
    'n_folds': 5,
}

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


class FoldProgressCallback(Callback):
    """Track progress within each fold."""

    def __init__(self, fold_num):
        super().__init__()
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

        degradation = (self.best_jacard - val_jacard) / max(self.best_jacard, 1e-6)

        print(f'\n📊 FOLD {self.fold_num} - EPOCH {epoch+1}/{EPOCHS}:')
        print(f'   Train Jaccard:   {train_jacard:.4f}')
        print(f'   Val Jaccard:     {val_jacard:.4f}')
        print(f'   Overfitting Gap: {gap:.1f}×')
        print(f'   Best Val:        {self.best_jacard:.4f} (epoch {self.best_epoch+1})')
        print(f'   Degradation:     {degradation:.1%}')


def load_data(images_dir, masks_dir, img_size=256):
    """Load and preprocess images and masks with metadata."""
    print("\n🔄 Loading dataset...")

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_files = sorted(images_dir.glob('*.tif'))

    images = []
    masks = []
    filenames = []
    densities = []  # For stratified splitting

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

        # Calculate density for stratification
        density = mask.mean()  # Proportion of foreground pixels

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
    print(f"   Mean density: {densities.mean():.1%}")
    print(f"   Density range: {densities.min():.1%} - {densities.max():.1%}")

    return images, masks, filenames, densities


def create_stratified_folds(densities, n_folds=5):
    """Create stratified folds based on density."""
    # Bin densities into quartiles for stratification
    density_bins = pd.qcut(densities, q=4, labels=False, duplicates='drop')

    print(f"\n📊 Creating {n_folds}-fold stratified split:")
    print(f"   Density bins: {np.bincount(density_bins)}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    return skf.split(np.zeros(len(densities)), density_bins)


def train_fold(fold_num, X_train, y_train, X_val, y_val, save_dir):
    """Train model on one fold."""

    print(f"\n{'='*80}")
    print(f"FOLD {fold_num}/5: TRAINING")
    print(f"{'='*80}")
    print(f"   Training samples:   {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Train density:      {y_train.mean():.1%}")
    print(f"   Val density:        {y_val.mean():.1%}")

    # Build model
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    model = get_model(
        CONFIG['architecture'],
        input_shape,
        NUM_CLASSES=1,
        dropout_rate=CONFIG['dropout'],
        batch_norm=True,
        filters=CONFIG['filters']
    )

    total_params = model.count_params()
    print(f"\n   Model parameters:   {total_params:,}")

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
    fold_dir = save_dir / f'fold_{fold_num}'
    fold_dir.mkdir(exist_ok=True)

    nan_callback = TerminateOnNaN()
    progress_callback = FoldProgressCallback(fold_num)

    callbacks = [
        nan_callback,
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

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=CONFIG['batch_size'],
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Get results
    history_df = pd.read_csv(fold_dir / 'history.csv')

    results = {
        'fold': fold_num,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'train_density': float(y_train.mean()),
        'val_density': float(y_val.mean()),
        'best_val_jacard': float(history_df['val_jacard_coef'].max()),
        'best_epoch': int(history_df['val_jacard_coef'].idxmax()),
        'final_val_jacard': float(history_df['val_jacard_coef'].iloc[-1]),
        'final_train_jacard': float(history_df['jacard_coef'].iloc[-1]),
        'min_val_jacard': float(history_df['val_jacard_coef'].min()),
        'overfitting_gap': float(history_df['jacard_coef'].iloc[-1] / max(history_df['val_jacard_coef'].iloc[-1], 1e-6)),
        'nan_detected': nan_callback.nan_detected,
        'epochs_trained': len(history_df)
    }

    print(f"\n✅ FOLD {fold_num} COMPLETE:")
    print(f"   Best Val Jaccard:    {results['best_val_jacard']:.4f} (epoch {results['best_epoch']+1})")
    print(f"   Final Val Jaccard:   {results['final_val_jacard']:.4f}")
    print(f"   Overfitting Gap:     {results['overfitting_gap']:.1f}×")

    # Save fold results
    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def analyze_cv_results(fold_results, save_dir):
    """Analyze cross-validation results across all folds."""

    print("\n" + "="*80)
    print("📊 CROSS-VALIDATION RESULTS - FINAL ANALYSIS")
    print("="*80)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(fold_results)

    # Calculate statistics
    stats = {
        'best_val_jacard': {
            'mean': df['best_val_jacard'].mean(),
            'std': df['best_val_jacard'].std(),
            'min': df['best_val_jacard'].min(),
            'max': df['best_val_jacard'].max(),
            'values': df['best_val_jacard'].tolist()
        },
        'final_val_jacard': {
            'mean': df['final_val_jacard'].mean(),
            'std': df['final_val_jacard'].std(),
            'min': df['final_val_jacard'].min(),
            'max': df['final_val_jacard'].max(),
            'values': df['final_val_jacard'].tolist()
        },
        'overfitting_gap': {
            'mean': df['overfitting_gap'].mean(),
            'std': df['overfitting_gap'].std(),
            'min': df['overfitting_gap'].min(),
            'max': df['overfitting_gap'].max(),
            'values': df['overfitting_gap'].tolist()
        },
        'best_epoch': {
            'mean': df['best_epoch'].mean(),
            'std': df['best_epoch'].std(),
            'values': df['best_epoch'].tolist()
        },
        'epochs_trained': {
            'mean': df['epochs_trained'].mean(),
            'values': df['epochs_trained'].tolist()
        }
    }

    # Display results
    print(f"\n🎯 CROSS-VALIDATION PERFORMANCE:")
    print(f"\n   Best Val Jaccard:")
    print(f"     Mean:  {stats['best_val_jacard']['mean']:.4f} ± {stats['best_val_jacard']['std']:.4f}")
    print(f"     Range: {stats['best_val_jacard']['min']:.4f} - {stats['best_val_jacard']['max']:.4f}")
    print(f"     Folds: {stats['best_val_jacard']['values']}")

    print(f"\n   Final Val Jaccard:")
    print(f"     Mean:  {stats['final_val_jacard']['mean']:.4f} ± {stats['final_val_jacard']['std']:.4f}")
    print(f"     Range: {stats['final_val_jacard']['min']:.4f} - {stats['final_val_jacard']['max']:.4f}")

    print(f"\n   Overfitting Gap:")
    print(f"     Mean:  {stats['overfitting_gap']['mean']:.1f}× ± {stats['overfitting_gap']['std']:.1f}×")
    print(f"     Range: {stats['overfitting_gap']['min']:.1f}× - {stats['overfitting_gap']['max']:.1f}×")

    print(f"\n   Best Epoch:")
    print(f"     Mean:  {stats['best_epoch']['mean']:.1f}")
    print(f"     Folds: {stats['best_epoch']['values']}")

    # Comparison with previous tests
    print(f"\n📊 COMPARISON WITH PREVIOUS TESTS:")
    print(f"\n   Metric                | Phase 1 | CV Mean | Difference")
    print(f"   ----------------------|---------|---------|------------")
    print(f"   Best Val Jaccard      | 13.8%   | {stats['best_val_jacard']['mean']*100:.1f}%   | {(stats['best_val_jacard']['mean']-0.138)*100:+.1f}%")
    print(f"   Overfitting Gap       | 10.5×   | {stats['overfitting_gap']['mean']:.1f}×    | {stats['overfitting_gap']['mean']-10.5:+.1f}×")

    # Diagnosis
    print(f"\n🔍 DIAGNOSIS:")

    cv_mean = stats['best_val_jacard']['mean']
    phase1_best = 0.138

    if cv_mean > phase1_best * 1.2:
        print(f"\n   ✅ CV mean ({cv_mean:.1%}) >> Phase 1 ({phase1_best:.1%})")
        print(f"      → Phase 1 validation split was BIASED (unlucky split)")
        print(f"      → True performance is better than initially measured")
        print(f"      → Should use CV for future evaluations")
    elif cv_mean > phase1_best * 0.8:
        print(f"\n   📊 CV mean ({cv_mean:.1%}) ≈ Phase 1 ({phase1_best:.1%})")
        print(f"      → Phase 1 validation split was REPRESENTATIVE")
        print(f"      → Task performance is genuinely {cv_mean:.1%}")
        print(f"      → Problem is not split bias but task difficulty")
    else:
        print(f"\n   ⚠️  CV mean ({cv_mean:.1%}) < Phase 1 ({phase1_best:.1%})")
        print(f"      → Phase 1 validation split was OPTIMISTIC")
        print(f"      → True performance is worse than initially measured")

    # Check consistency of "best at epoch 1" pattern
    epoch_1_count = sum(1 for e in stats['best_epoch']['values'] if e <= 1)

    print(f"\n   Best Epoch Analysis:")
    print(f"     Folds peaking at epoch ≤1: {epoch_1_count}/5")

    if epoch_1_count >= 4:
        print(f"      → ❌ VALIDATION PROBLEM CONFIRMED across folds")
        print(f"      → Data distribution issue (not just split bias)")
        print(f"      → Need to investigate train/val differences")
    elif epoch_1_count >= 2:
        print(f"      → ⚠️  MIXED RESULTS - some folds learn, some don't")
        print(f"      → May indicate data heterogeneity")
    else:
        print(f"      → ✅ NORMAL PATTERN - validation improves with training")
        print(f"      → Phase 1's epoch 1 peak was due to unlucky split")

    # Check variance
    cv_std = stats['best_val_jacard']['std']
    cv_coef_var = cv_std / max(cv_mean, 1e-6)

    print(f"\n   Stability Analysis:")
    print(f"     Coefficient of variation: {cv_coef_var:.1%}")

    if cv_coef_var > 0.3:
        print(f"      → ⚠️  HIGH VARIANCE across folds")
        print(f"      → Performance depends heavily on train/val split")
        print(f"      → May indicate class imbalance or data heterogeneity")
    elif cv_coef_var > 0.15:
        print(f"      → 📊 MODERATE VARIANCE")
        print(f"      → Some split dependency but acceptable")
    else:
        print(f"      → ✅ LOW VARIANCE")
        print(f"      → Performance consistent across splits")
        print(f"      → Model is stable")

    # Save summary
    summary = {
        'config': CONFIG,
        'n_folds': len(fold_results),
        'statistics': stats,
        'fold_results': fold_results,
        'comparison': {
            'phase1_best_jacard': 0.138,
            'cv_mean_best_jacard': cv_mean,
            'improvement': float((cv_mean - 0.138) / 0.138),
            'phase1_gap': 10.5,
            'cv_mean_gap': stats['overfitting_gap']['mean'],
            'gap_change': float(stats['overfitting_gap']['mean'] - 10.5)
        },
        'diagnosis': {
            'split_bias': cv_mean > phase1_best * 1.2,
            'epoch_1_problem_persists': epoch_1_count >= 4,
            'high_variance': cv_coef_var > 0.3,
            'coefficient_of_variation': float(cv_coef_var)
        }
    }

    with open(save_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {save_dir}")
    print(f"   - cv_summary.json")
    print(f"   - fold_*/history.csv")
    print(f"   - fold_*/results.json")
    print(f"   - fold_*/best_model.keras")

    print("\n" + "="*80)
    print("🏁 CROSS-VALIDATION COMPLETE")
    print("="*80)

    return summary


def main():
    """Main cross-validation workflow."""

    print("="*80)
    print("🧪 5-FOLD CROSS-VALIDATION - PRIORITY 1")
    print("="*80)
    print("\nPURPOSE: Get reliable performance estimates")
    print("BASELINE CONFIG: filters=64, dropout=0.3 (same as Phase 1)")

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"validation_cv_{timestamp}")
    save_dir.mkdir(exist_ok=True)

    print(f"\n💾 Save directory: {save_dir}")

    # Load data
    script_dir = Path(__file__).parent
    images_dir = script_dir / "dataset_full_stack" / "images"
    masks_dir = script_dir / "dataset_full_stack" / "masks"

    X, y, filenames, densities = load_data(str(images_dir), str(masks_dir), IMG_SIZE)

    # Create stratified folds
    fold_splits = list(create_stratified_folds(densities, n_folds=CONFIG['n_folds']))

    # Train each fold
    fold_results = []

    for fold_num, (train_idx, val_idx) in enumerate(fold_splits, 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        results = train_fold(fold_num, X_train, y_train, X_val, y_val, save_dir)
        fold_results.append(results)

    # Analyze results
    summary = analyze_cv_results(fold_results, save_dir)

    return 0


if __name__ == "__main__":
    exit(main())
