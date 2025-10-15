#!/usr/bin/env python3
"""
Phase 1B Test 1: Focal Tversky Loss
====================================
Tests if focal_tversky loss handles class imbalance better than combined loss.

Expected improvement:
- Validation Jaccard should stay > 10% throughout training
- Less overfitting (smaller train-val gap)
- Better handling of small objects (microbeads)

This is the HIGHEST IMPACT test - focal_tversky is specifically designed for:
- Severe class imbalance (92% background, 8% foreground)
- Small object detection (microbeads)
- Hard example mining (difficult-to-segment regions)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_architectures import get_model
from loss_functions_fixed import get_loss_function, dice_coef, jacard_coef

# Configuration - ONLY CHANGE: loss function
SIZE = 512
IMG_CHANNELS = 1
EPOCHS = 20
EARLY_STOP_PATIENCE = 10
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 5e-5

# Test configuration: focal_tversky loss
TEST_CONFIG = {
    'architecture': 'unet',
    'batch_size': 4,
    'dropout': 0.3,
    'loss_function': 'focal_tversky',  # ← CHANGED from 'combined'
}

TRAIN_PATH_IMAGES = './dataset_shrunk_masks/images/'
TRAIN_PATH_MASKS = './dataset_shrunk_masks/masks/'
OUTPUT_DIR = Path(f'./validation_focal_tversky_{datetime.now().strftime("%Y%m%d_%H%M%S")}')


class TerminateOnNaN(Callback):
    def __init__(self):
        super().__init__()
        self.nan_detected = False
        self.nan_epoch = None

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print(f'\n❌ BATCH {batch}: INVALID LOSS: {loss}')
                self.nan_detected = True
                self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if loss is not None and (np.isnan(loss) or np.isinf(loss)):
            print(f'\n❌ EPOCH {epoch}: Training loss invalid: {loss}')
            self.nan_detected = True
            self.nan_epoch = epoch
            self.model.stop_training = True

        if val_loss is not None and (np.isnan(val_loss) or np.isinf(val_loss)):
            print(f'\n❌ EPOCH {epoch}: Validation loss invalid: {val_loss}')
            self.nan_detected = True
            self.nan_epoch = epoch
            self.model.stop_training = True


class ValidationProgressCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_jacard = 0.0
        self.best_epoch = 0
        self.min_jacard = 1.0
        self.jacard_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_jacard = logs.get('val_jacard_coef', 0)
        self.jacard_history.append(val_jacard)

        if val_jacard > self.best_jacard:
            self.best_jacard = val_jacard
            self.best_epoch = epoch
            status = "✓ NEW BEST"
        else:
            status = ""

        if val_jacard < self.min_jacard:
            self.min_jacard = val_jacard

        train_jacard = logs.get('jacard_coef', 0)
        gap = train_jacard / max(val_jacard, 1e-6)

        print(f'\nEpoch {epoch+1:2d}/{EPOCHS}: '
              f'train_j={train_jacard:.4f}, val_j={val_jacard:.4f}, '
              f'gap={gap:.1f}× {status}')

        if (epoch + 1) % 5 == 0:
            degradation = (self.best_jacard - val_jacard) / max(self.best_jacard, 1e-6)
            print(f'\n--- Progress Summary (Epoch {epoch+1}) ---')
            print(f'Best Jaccard: {self.best_jacard:.4f} at epoch {self.best_epoch+1}')
            print(f'Current: {val_jacard:.4f} ({degradation*100:+.1f}% from best)')
            print(f'Min: {self.min_jacard:.4f}')
            print(f'Degradation: {degradation*100:.1f}% (lower is better)')

            if degradation < 0.2:
                print(f'✓ Status: STABLE (< 20% degradation)')
            elif degradation < 0.5:
                print(f'⚠ Status: Moderate degradation (20-50%)')
            else:
                print(f'❌ Status: Severe degradation (> 50%)')


def load_dataset():
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)

    if not os.path.exists(TRAIN_PATH_IMAGES):
        raise FileNotFoundError(f"Image directory not found: {TRAIN_PATH_IMAGES}")
    if not os.path.exists(TRAIN_PATH_MASKS):
        raise FileNotFoundError(f"Mask directory not found: {TRAIN_PATH_MASKS}")

    train_ids = sorted(next(os.walk(TRAIN_PATH_IMAGES))[2])
    X_train, y_train = [], []

    for n, id_ in enumerate(train_ids):
        image_path = os.path.join(TRAIN_PATH_IMAGES, id_)
        image = cv2.imread(image_path, 0)

        if image is not None:
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            X_train.append(np.array(image))

            mask_path = os.path.join(TRAIN_PATH_MASKS, id_)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                mask = Image.fromarray(mask)
                mask = mask.resize((SIZE, SIZE))
                y_train.append(np.array(mask))

    X_train = np.array(X_train) / 255.0
    y_train = np.array(y_train) / 255.0
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)

    print(f"✓ Dataset loaded: {len(X_train)} images")
    return X_train, y_train


def stratified_split_by_density(X, y, test_size=0.15, random_state=42):
    print("\n" + "="*80)
    print("SPLITTING DATASET")
    print("="*80)

    densities = [np.sum(mask > 0.5) / mask.size for mask in y]
    density_bins = pd.qcut(densities, q=min(5, len(densities)), labels=False, duplicates='drop')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=density_bins, random_state=random_state
    )

    print(f"✓ Split complete: {len(X_train)} train, {len(X_val)} val")
    return X_train, X_val, y_train, y_val


def train_model(X_train, X_val, y_train, y_val):
    print("\n" + "="*80)
    print("TESTING FOCAL TVERSKY LOSS")
    print("="*80)
    print(f"\nWhy focal_tversky for this dataset:")
    print(f"  1. Severe class imbalance: 92% background, 8% foreground")
    print(f"  2. Small objects: Microbeads are tiny relative to image size")
    print(f"  3. Hard examples: Overlapping beads, varied illumination")
    print(f"\nFocal Tversky parameters:")
    print(f"  α=0.7: Penalize false negatives 2.3× more than false positives")
    print(f"  β=0.3: Tolerate some over-segmentation to catch all beads")
    print(f"  γ=1.33: Focus on hard examples (paper-recommended value)")
    print("="*80)

    arch = TEST_CONFIG['architecture']
    bs = TEST_CONFIG['batch_size']
    dropout = TEST_CONFIG['dropout']
    loss_name = TEST_CONFIG['loss_function']

    input_shape = (SIZE, SIZE, IMG_CHANNELS)
    model = get_model(arch, input_shape, NUM_CLASSES=1, dropout_rate=dropout, batch_norm=True)

    print(f"\nModel: {model.name}")
    print(f"Parameters: {model.count_params():,}")

    loss_fn = get_loss_function(loss_name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss=loss_fn,
        metrics=['accuracy', jacard_coef, dice_coef]
    )

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=15,
        fill_mode='reflect',
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    val_datagen = ImageDataGenerator()

    checkpoint_path = OUTPUT_DIR / f"model_best.hdf5"

    nan_callback = TerminateOnNaN()
    progress_callback = ValidationProgressCallback()

    callbacks = [
        nan_callback,
        progress_callback,
        EarlyStopping(
            monitor='val_jacard_coef',
            patience=EARLY_STOP_PATIENCE,
            verbose=1,
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_jacard_coef',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    print(f"\nStarting training with focal_tversky loss...")
    print(f"Watch for: Validation Jaccard should stay > 10% (vs 3% with combined loss)\n")

    seed = 42
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=bs, seed=seed),
        validation_data=val_datagen.flow(X_val, y_val, batch_size=bs, seed=seed),
        steps_per_epoch=len(X_train) // bs,
        validation_steps=len(X_val) // bs,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return history, nan_callback, progress_callback


def analyze_results(history, nan_callback, progress_callback):
    print("\n" + "="*80)
    print("FOCAL TVERSKY TEST RESULTS")
    print("="*80)

    if nan_callback.nan_detected:
        print(f"\n❌ FAILED: NaN detected during training")
        return False

    hist_df = pd.DataFrame(history.history)

    if hist_df.isnull().any().any() or np.isinf(hist_df.values).any():
        print(f"\n❌ FAILED: NaN/inf in training history")
        return False

    best_val_jacard = progress_callback.best_jacard
    min_val_jacard = progress_callback.min_jacard
    final_val_jacard = hist_df['val_jacard_coef'].iloc[-1]
    best_epoch = progress_callback.best_epoch

    final_train_jacard = hist_df['jacard_coef'].iloc[-1]
    overfitting_gap = final_train_jacard / max(final_val_jacard, 1e-6)

    degradation = (best_val_jacard - min_val_jacard) / max(best_val_jacard, 1e-6)

    print(f"\n{'='*80}")
    print(f"FINAL METRICS")
    print(f"{'='*80}")
    print(f"Best validation Jaccard: {best_val_jacard:.4f} ({best_val_jacard*100:.1f}%) at epoch {best_epoch+1}")
    print(f"Final validation Jaccard: {final_val_jacard:.4f} ({final_val_jacard*100:.1f}%)")
    print(f"Minimum validation Jaccard: {min_val_jacard:.4f} ({min_val_jacard*100:.1f}%)")
    print(f"Degradation: {degradation*100:.1f}% (from best to worst)")
    print(f"\nOverfitting gap: {overfitting_gap:.1f}× (train/val ratio)")

    print(f"\n{'='*80}")
    print(f"COMPARISON WITH COMBINED LOSS")
    print(f"{'='*80}")
    print(f"Metric                      | Combined Loss | Focal Tversky | Improvement")
    print(f"----------------------------|---------------|---------------|-------------")
    print(f"Best Val Jaccard            | 13.8%         | {best_val_jacard*100:5.1f}%        | {(best_val_jacard/0.138-1)*100:+.0f}%")
    print(f"Final Val Jaccard           | 3.0%          | {final_val_jacard*100:5.1f}%        | {(final_val_jacard/0.030-1)*100:+.0f}%")
    print(f"Degradation                 | 78%           | {degradation*100:5.1f}%        | {(1-degradation/0.78)*100:+.0f}%")
    print(f"Overfitting gap             | 10.5×         | {overfitting_gap:5.1f}×        | {(1-overfitting_gap/10.5)*100:+.0f}%")

    print(f"\n{'='*80}")
    print(f"SUCCESS CRITERIA")
    print(f"{'='*80}")

    criteria_met = 0
    total_criteria = 5

    criterion_1 = not nan_callback.nan_detected
    print(f"✓ No NaN detected: {'PASS' if criterion_1 else 'FAIL'}")
    if criterion_1:
        criteria_met += 1

    criterion_2 = best_val_jacard > 0.15
    print(f"{'✓' if criterion_2 else '✗'} Best Jaccard > 15%: {'PASS' if criterion_2 else 'FAIL'} ({best_val_jacard*100:.1f}%)")
    if criterion_2:
        criteria_met += 1

    criterion_3 = final_val_jacard > 0.10
    print(f"{'✓' if criterion_3 else '✗'} Final Jaccard > 10%: {'PASS' if criterion_3 else 'FAIL'} ({final_val_jacard*100:.1f}%)")
    if criterion_3:
        criteria_met += 1

    criterion_4 = degradation < 0.5
    print(f"{'✓' if criterion_4 else '✗'} Degradation < 50%: {'PASS' if criterion_4 else 'FAIL'} ({degradation*100:.1f}%)")
    if criterion_4:
        criteria_met += 1

    criterion_5 = overfitting_gap < 5.0
    print(f"{'✓' if criterion_5 else '✗'} Overfitting < 5×: {'PASS' if criterion_5 else 'FAIL'} ({overfitting_gap:.1f}×)")
    if criterion_5:
        criteria_met += 1

    print(f"\n{'='*80}")
    if criteria_met >= 4:
        print(f"✓ TEST PASSED ({criteria_met}/{total_criteria} criteria met)")
        print(f"\nFocal Tversky loss is BETTER than combined loss!")
        print(f"Next step: Use focal_tversky in full hyperparameter search.")
    elif criteria_met >= 3:
        print(f"⚠ TEST PARTIAL ({criteria_met}/{total_criteria} criteria met)")
        print(f"\nFocal Tversky shows improvement but not enough.")
        print(f"Next step: Combine with stronger regularization or smaller model.")
    else:
        print(f"✗ TEST FAILED ({criteria_met}/{total_criteria} criteria met)")
        print(f"\nFocal Tversky did not improve over combined loss.")
    print(f"{'='*80}\n")

    hist_df.to_csv(OUTPUT_DIR / 'training_history.csv', index=False)

    results_summary = {
        'config': TEST_CONFIG,
        'best_val_jacard': float(best_val_jacard),
        'final_val_jacard': float(final_val_jacard),
        'min_val_jacard': float(min_val_jacard),
        'best_epoch': int(best_epoch),
        'degradation': float(degradation),
        'overfitting_gap': float(overfitting_gap),
        'nan_detected': nan_callback.nan_detected,
        'criteria_met': criteria_met,
        'total_criteria': total_criteria,
        'test_passed': criteria_met >= 4
    }

    with open(OUTPUT_DIR / 'test_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    return criteria_met >= 3


def main():
    print("="*80)
    print("PHASE 1B TEST 1: FOCAL TVERSKY LOSS")
    print("="*80)
    print(f"Testing if focal_tversky handles class imbalance better than combined")
    print(f"Expected: Val Jaccard stays > 10%, less degradation\n")

    OUTPUT_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()
    X_train, X_val, y_train, y_val = stratified_split_by_density(X, y)

    history, nan_callback, progress_callback = train_model(X_train, X_val, y_train, y_val)

    if history is None:
        print("\n❌ Training failed due to NaN")
        return

    success = analyze_results(history, nan_callback, progress_callback)

    if success:
        print("\n" + "="*80)
        print("✓ FOCAL TVERSKY TEST: SUCCESS")
        print("="*80)
        print("\nFocal Tversky loss works better for class imbalance!")
        print("Use this loss function for full hyperparameter search.")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
