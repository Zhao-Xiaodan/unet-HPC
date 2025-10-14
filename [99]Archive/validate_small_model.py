#!/usr/bin/env python3
"""
Small Model Validation - Priority 1 Test

PURPOSE:
Test if reducing model complexity improves validation performance.

HYPOTHESIS:
Current model (31M params) is too complex for dataset size (83 images).
Ratio: 378,117 params/image vs ResNet50's 21 params/image (18,000× worse).

SOLUTION:
Reduce model from filters=64 (31M params) to filters=16 (~2M params).
Increase dropout from 0.3 to 0.5 for stronger regularization.

EXPECTED OUTCOME:
- Validation Jaccard: 20-30% (vs current 13%)
- Overfitting gap: 3-5× (vs current 10-15×)
- Stable training without immediate collapse

COMPARISON WITH PREVIOUS TESTS:
Phase 1 (combined loss):     filters=64, dropout=0.3 → 13.8% val Jaccard, 10.5× gap
Focal Tversky:               filters=64, dropout=0.3 → 13.3% val Jaccard, 15.1× gap
This test (small model):     filters=16, dropout=0.5 → Expected 20-30% val Jaccard
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

from sklearn.model_selection import train_test_split
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

# Test configuration
TEST_CONFIG = {
    'architecture': 'unet',
    'batch_size': 4,
    'dropout': 0.5,  # INCREASED from 0.3 for stronger regularization
    'loss_function': 'combined',  # Same as Phase 1 baseline
    'filters': 16,  # KEY: REDUCED from 64 to reduce params from 31M to ~2M
}

EPOCHS = 20  # Quick validation
IMG_SIZE = 256
NUM_CLASSES = 1

# Success criteria
SUCCESS_CRITERIA = {
    'no_nan': True,
    'val_jacard_min': 0.15,  # Higher than Phase 1's 13.8%
    'overfitting_gap_max': 7.0,  # Lower than Phase 1's 10.5×
    'degradation_max': 0.5,  # Lower than Phase 1's 78%
    'min_improvement_epoch': 2  # Should improve beyond epoch 1
}


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
                print(f'\n❌ BATCH {batch}: INVALID LOSS DETECTED: {loss}')
                print(f'   Metrics: {logs}')
                self.nan_detected = True
                self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            if value is not None and (np.isnan(value) or np.isinf(value)):
                print(f'\n❌ EPOCH {epoch}: INVALID {key}: {value}')
                self.nan_detected = True
                self.model.stop_training = True


class ValidationProgressCallback(Callback):
    """Track validation progress and detect early collapse."""

    def __init__(self):
        super().__init__()
        self.best_jacard = 0.0
        self.best_epoch = 0
        self.history = {
            'epoch': [],
            'train_jacard': [],
            'val_jacard': [],
            'gap': [],
            'degradation': []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        val_jacard = logs.get('val_jacard_coef', 0)
        train_jacard = logs.get('jacard_coef', 0)

        # Calculate overfitting gap
        gap = train_jacard / max(val_jacard, 1e-6)

        # Track best validation performance
        if val_jacard > self.best_jacard:
            self.best_jacard = val_jacard
            self.best_epoch = epoch

        # Calculate degradation from best
        degradation = (self.best_jacard - val_jacard) / max(self.best_jacard, 1e-6)

        # Store history
        self.history['epoch'].append(epoch)
        self.history['train_jacard'].append(train_jacard)
        self.history['val_jacard'].append(val_jacard)
        self.history['gap'].append(gap)
        self.history['degradation'].append(degradation)

        # Progress report
        print(f'\n📊 EPOCH {epoch+1}/{EPOCHS} VALIDATION PROGRESS:')
        print(f'   Train Jaccard:  {train_jacard:.4f}')
        print(f'   Val Jaccard:    {val_jacard:.4f}')
        print(f'   Overfitting Gap: {gap:.1f}×')
        print(f'   Best Val:       {self.best_jacard:.4f} (epoch {self.best_epoch+1})')
        print(f'   Degradation:    {degradation:.1%}')

        # Warning for severe overfitting
        if gap > 10.0:
            print(f'   ⚠️  SEVERE OVERFITTING DETECTED (gap > 10×)')

        # Warning for collapse
        if degradation > 0.5 and epoch > 2:
            print(f'   ⚠️  VALIDATION COLLAPSE (>50% degradation from best)')


def load_data(images_dir, masks_dir, img_size=256):
    """Load and preprocess images and masks."""
    print("\n🔄 Loading dataset...")

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    # Get all image files
    image_files = sorted(images_dir.glob('*.tif'))

    images = []
    masks = []
    filenames = []

    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Load corresponding mask
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
        mask = (mask > 127).astype(np.float32)  # Binary threshold

        images.append(img)
        masks.append(mask)
        filenames.append(img_path.name)

    images = np.array(images)[..., np.newaxis]
    masks = np.array(masks)[..., np.newaxis]

    print(f"✅ Loaded {len(images)} image-mask pairs")
    print(f"   Image shape: {images.shape}")
    print(f"   Mask shape: {masks.shape}")
    print(f"   Foreground ratio: {masks.mean():.1%}")

    return images, masks, filenames


def train_model(config, X_train, y_train, X_val, y_val, save_dir):
    """Train model with specified configuration."""

    print(f"\n🏗️  Building model with configuration:")
    print(f"   Architecture: {config['architecture']}")
    print(f"   Filters: {config['filters']} (KEY: Reduced from 64)")
    print(f"   Dropout: {config['dropout']} (INCREASED from 0.3)")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Loss: {config['loss_function']}")

    # Build model with REDUCED filters
    arch = config['architecture']
    input_shape = (IMG_SIZE, IMG_SIZE, 1)

    model = get_model(
        arch,
        input_shape,
        NUM_CLASSES=1,
        dropout_rate=config['dropout'],
        batch_norm=True,
        filters=config['filters']  # KEY: 16 instead of default 64
    )

    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([np.prod(v.get_shape()) for v in model.trainable_weights])

    print(f"\n📊 Model complexity:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Parameters per training image: {total_params/len(X_train):,.0f}")
    print(f"   🎯 Expected: ~2M params (vs Phase 1's 31M params)")

    # Select loss function (same as Phase 1: Dice + Focal)
    loss_fn = combined_dice_focal_loss

    # Compile with GRADIENT CLIPPING
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=5e-5,
            clipnorm=1.0  # Gradient clipping for stability
        ),
        loss=loss_fn,
        metrics=['accuracy', jacard_coef, dice_coef]
    )

    # Callbacks
    callbacks = [
        TerminateOnNaN(),
        ValidationProgressCallback(),
        ModelCheckpoint(
            str(save_dir / 'best_model.keras'),
            monitor='val_jacard_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(
            str(save_dir / 'training_history.csv'),
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
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config['batch_size'],
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return model, history, callbacks[0].nan_detected, callbacks[1]


def analyze_results(history, progress_callback, nan_detected, save_dir):
    """Analyze training results and check success criteria."""

    print("\n" + "="*80)
    print("📊 SMALL MODEL TEST - FINAL RESULTS")
    print("="*80)

    # Get metrics
    history_df = pd.read_csv(save_dir / 'training_history.csv')

    final_loss = history_df['loss'].iloc[-1]
    final_val_jacard = history_df['val_jacard_coef'].iloc[-1]
    best_val_jacard = history_df['val_jacard_coef'].max()
    best_epoch = history_df['val_jacard_coef'].idxmax()
    min_val_jacard = history_df['val_jacard_coef'].min()

    final_train_jacard = history_df['jacard_coef'].iloc[-1]

    # Calculate overfitting metrics
    overfitting_gap = final_train_jacard / max(final_val_jacard, 1e-6)
    degradation = (best_val_jacard - final_val_jacard) / max(best_val_jacard, 1e-6)

    print(f"\n🎯 Performance Metrics:")
    print(f"   Final Loss:          {final_loss:.4f}")
    print(f"   Final Train Jaccard: {final_train_jacard:.4f}")
    print(f"   Final Val Jaccard:   {final_val_jacard:.4f}")
    print(f"   Best Val Jaccard:    {best_val_jacard:.4f} (epoch {best_epoch+1})")
    print(f"   Min Val Jaccard:     {min_val_jacard:.4f}")

    print(f"\n📈 Overfitting Analysis:")
    print(f"   Overfitting Gap:     {overfitting_gap:.1f}× (train/val)")
    print(f"   Degradation:         {degradation:.1%} (from best to final)")
    print(f"   Best Epoch:          {best_epoch+1}/{len(history_df)}")

    print(f"\n🔍 Stability Check:")
    print(f"   NaN Detected:        {'❌ YES' if nan_detected else '✅ NO'}")

    # Comparison with Phase 1
    print(f"\n📊 COMPARISON WITH PREVIOUS TESTS:")
    print(f"   Metric                  | Phase 1   | Focal Tversky | This Test | Change")
    print(f"   ------------------------|-----------|---------------|-----------|--------")
    print(f"   Best Val Jaccard        | 13.8%     | 13.3%         | {best_val_jacard*100:.1f}%     | {((best_val_jacard-0.138)/0.138)*100:+.0f}%")
    print(f"   Final Val Jaccard       | 3.0%      | 2.1%          | {final_val_jacard*100:.1f}%      | {((final_val_jacard-0.030)/0.030)*100:+.0f}%")
    print(f"   Overfitting Gap         | 10.5×     | 15.1×         | {overfitting_gap:.1f}×      | {overfitting_gap-10.5:+.1f}×")
    print(f"   Degradation             | 78%       | 95%           | {degradation*100:.0f}%       | {(degradation-0.78)*100:+.0f}%")
    print(f"   Best Epoch              | 1         | 1             | {best_epoch+1}         | {'✅ Improved' if best_epoch > 0 else '❌ Same'}")

    # Check success criteria
    print(f"\n✅ SUCCESS CRITERIA CHECK:")
    criteria_met = 0
    total_criteria = len(SUCCESS_CRITERIA)

    checks = {
        'no_nan': (not nan_detected, "No NaN/Inf detected"),
        'val_jacard_min': (best_val_jacard >= SUCCESS_CRITERIA['val_jacard_min'],
                          f"Best val Jaccard ≥ {SUCCESS_CRITERIA['val_jacard_min']:.1%}"),
        'overfitting_gap_max': (overfitting_gap <= SUCCESS_CRITERIA['overfitting_gap_max'],
                               f"Overfitting gap ≤ {SUCCESS_CRITERIA['overfitting_gap_max']:.1f}×"),
        'degradation_max': (degradation <= SUCCESS_CRITERIA['degradation_max'],
                           f"Degradation ≤ {SUCCESS_CRITERIA['degradation_max']:.0%}"),
        'min_improvement_epoch': (best_epoch >= SUCCESS_CRITERIA['min_improvement_epoch'],
                                 f"Best epoch ≥ {SUCCESS_CRITERIA['min_improvement_epoch']}")
    }

    for key, (passed, description) in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {description}")
        if passed:
            criteria_met += 1

    test_passed = criteria_met >= 4  # Need at least 4/5 criteria

    print(f"\n{'='*80}")
    print(f"🎯 OVERALL RESULT: {'✅ TEST PASSED' if test_passed else '❌ TEST FAILED'}")
    print(f"   Criteria met: {criteria_met}/{total_criteria}")
    print(f"{'='*80}")

    # Save summary
    summary = {
        'config': TEST_CONFIG,
        'final_loss': float(final_loss),
        'final_train_jacard': float(final_train_jacard),
        'final_val_jacard': float(final_val_jacard),
        'best_val_jacard': float(best_val_jacard),
        'min_val_jacard': float(min_val_jacard),
        'best_epoch': int(best_epoch),
        'overfitting_gap': float(overfitting_gap),
        'degradation': float(degradation),
        'nan_detected': bool(nan_detected),
        'criteria_met': int(criteria_met),
        'total_criteria': int(total_criteria),
        'test_passed': bool(test_passed),
        'comparison': {
            'phase1_best_jacard': 0.138,
            'focal_tversky_best_jacard': 0.133,
            'this_test_best_jacard': float(best_val_jacard),
            'improvement_vs_phase1': float((best_val_jacard - 0.138) / 0.138),
            'phase1_gap': 10.5,
            'focal_tversky_gap': 15.1,
            'this_test_gap': float(overfitting_gap),
            'gap_reduction': float(10.5 - overfitting_gap)
        }
    }

    with open(save_dir / 'test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Results saved to: {save_dir}")
    print(f"   - training_history.csv")
    print(f"   - test_summary.json")
    print(f"   - best_model.keras")

    return test_passed


def main():
    """Main validation workflow."""

    print("="*80)
    print("🧪 SMALL MODEL VALIDATION - PRIORITY 1 TEST")
    print("="*80)
    print("\nHYPOTHESIS: Reducing model complexity will improve validation performance")
    print("APPROACH: filters=64→16 (31M→2M params), dropout=0.3→0.5")
    print("EXPECTED: 20-30% val Jaccard, 3-5× overfitting gap, stable training")

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"validation_small_model_{timestamp}")
    save_dir.mkdir(exist_ok=True)

    print(f"\n💾 Save directory: {save_dir}")

    # Load data
    # Use relative path that works on both local and HPC
    script_dir = Path(__file__).parent
    images_dir = script_dir / "dataset_full_stack" / "images"
    masks_dir = script_dir / "dataset_full_stack" / "masks"

    X, y, filenames = load_data(str(images_dir), str(masks_dir), IMG_SIZE)

    # Split data (same split as previous tests for fair comparison)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(f"\n📊 Data split:")
    print(f"   Total images: {len(X)}")
    print(f"   Training: {len(X_train)}")
    print(f"   Validation: {len(X_val)}")

    # Train model
    model, history, nan_detected, progress_callback = train_model(
        TEST_CONFIG, X_train, y_train, X_val, y_val, save_dir
    )

    # Analyze results
    test_passed = analyze_results(history, progress_callback, nan_detected, save_dir)

    # Next steps recommendation
    print("\n" + "="*80)
    print("🔮 NEXT STEPS RECOMMENDATION:")
    print("="*80)

    if test_passed:
        print("\n✅ Small model shows improvement over Phase 1!")
        print("\n   Priority 2: Implement 5-fold cross-validation")
        print("   - More robust performance estimates")
        print("   - 20 validation images per fold (vs 15 currently)")
        print("   - Average metrics across folds")
        print("\n   Priority 3: Fine-tune small model hyperparameters")
        print("   - Test filters=24, 32 for optimal capacity")
        print("   - Test dropout=0.4, 0.6")
        print("   - Test different learning rates")
    else:
        print("\n❌ Small model did not meet success criteria")
        print("\n   If validation still collapses:")
        print("   - Validation set may not be representative")
        print("   - Need 5-fold cross-validation immediately")
        print("   - Analyze train/val distribution mismatch")
        print("\n   If overfitting persists (gap > 7×):")
        print("   - Increase dropout further (0.6-0.7)")
        print("   - Add more data augmentation")
        print("   - Consider even smaller model (filters=8)")

    print("\n" + "="*80)
    print("🏁 SMALL MODEL VALIDATION COMPLETE")
    print("="*80)

    return 0 if test_passed else 1


if __name__ == "__main__":
    exit(main())
