#!/usr/bin/env python3
"""
Hyperparameter Search for Microbead Segmentation
=================================================
Systematic grid search to find optimal hyperparameters through experimental validation.

Search space based on domain knowledge:
- Learning rate: 1e-4 chosen empirically, test 5e-5, 1e-4, 2e-4
- Dropout: 0.3 may be too aggressive, test 0.0, 0.1, 0.2, 0.3
- Loss function: Dice only, or combined with Focal
- Batch size: 32 works, but test 16, 32, 48
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Import model definitions
import sys
if Path('models.py').exists():
    from models import UNet
elif Path('224_225_226_models.py').exists():
    sys.path.insert(0, '.')
    from models import UNet
else:
    raise FileNotFoundError("Model definition file not found!")

# Fixed parameters (not searched)
SIZE = 512  # Use original dataset resolution (512×512)
IMG_CHANNELS = 1
EPOCHS = 100
EARLY_STOP_PATIENCE = 20
VALIDATION_SPLIT = 0.15

# Hyperparameter search space
SEARCH_SPACE = {
    'learning_rate': [5e-5, 1e-4, 2e-4],           # Test around current 1e-4
    'batch_size': [4, 8, 16],                      # Reduced for 512×512 (4× memory vs 256×256)
    'dropout': [0.0, 0.1, 0.2, 0.3],               # Test from none to current 0.3
    'loss_type': ['dice', 'focal', 'combined'],     # Test different losses
}

# Dataset paths
TRAIN_PATH_IMAGES = './dataset_microscope/images/'
TRAIN_PATH_MASKS = './dataset_microscope/masks/'

def dice_coef(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss function"""
    return 1 - dice_coef(y_true, y_pred, smooth)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Binary focal loss"""
    y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
    pt = tf.where(keras.backend.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_weight = alpha * keras.backend.pow(1 - pt, gamma)
    focal_loss = -focal_weight * keras.backend.log(pt)
    return keras.backend.mean(focal_loss)

def combined_loss(y_true, y_pred):
    """Combined Dice + Focal loss (70% Dice, 30% Focal)"""
    return 0.7 * dice_loss(y_true, y_pred) + 0.3 * focal_loss(y_true, y_pred)

def jacard_coef(y_true, y_pred, smooth=1e-6):
    """Jaccard coefficient (IoU) metric"""
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    union = keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def load_dataset():
    """Load and preprocess dataset"""
    print("Loading dataset...")

    train_ids = sorted(next(os.walk(TRAIN_PATH_IMAGES))[2])

    X_train = []
    y_train = []

    for n, id_ in enumerate(train_ids):
        # Load image
        image_path = os.path.join(TRAIN_PATH_IMAGES, id_)
        image = cv2.imread(image_path, 0)

        if image is not None:
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            X_train.append(np.array(image))

            # Load mask
            mask_path = os.path.join(TRAIN_PATH_MASKS, id_)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                mask = Image.fromarray(mask)
                mask = mask.resize((SIZE, SIZE))
                y_train.append(np.array(mask))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Normalize
    X_train = X_train / 255.0
    y_train = y_train / 255.0

    # Expand dimensions
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)

    print(f"Dataset loaded: {len(X_train)} images")
    print(f"Image size: {SIZE}×{SIZE} (using original resolution)")

    return X_train, y_train

def stratified_split_by_density(X, y, test_size=0.15, random_state=42):
    """Split dataset stratified by object density"""
    # Calculate object density for each image
    densities = []
    for mask in y:
        positive_ratio = np.sum(mask > 0.5) / mask.size
        densities.append(positive_ratio)

    densities = np.array(densities)

    # Create density bins
    density_bins = pd.qcut(densities, q=min(5, len(densities)), labels=False, duplicates='drop')

    # Stratified split
    return train_test_split(X, y, test_size=test_size, stratify=density_bins, random_state=random_state)

def get_loss_function(loss_type):
    """Get loss function by type"""
    if loss_type == 'dice':
        return dice_loss
    elif loss_type == 'focal':
        return focal_loss
    elif loss_type == 'combined':
        return combined_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def train_with_hyperparameters(X_train, X_val, y_train, y_val, hyperparams, output_dir):
    """Train model with specific hyperparameters"""

    lr = hyperparams['learning_rate']
    bs = hyperparams['batch_size']
    dropout = hyperparams['dropout']
    loss_type = hyperparams['loss_type']

    print(f"\n{'='*80}")
    print(f"Training with: LR={lr}, BS={bs}, Dropout={dropout}, Loss={loss_type}")
    print(f"{'='*80}\n")

    # Build model
    input_shape = (SIZE, SIZE, IMG_CHANNELS)
    model = UNet(input_shape, NUM_CLASSES=1, dropout_rate=dropout, batch_norm=True)

    # Compile model
    loss_fn = get_loss_function(loss_type)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=['accuracy', jacard_coef, dice_coef]
    )

    # Data augmentation
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=15,
        fill_mode='reflect'
    )

    val_datagen = ImageDataGenerator()

    # Callbacks
    checkpoint_path = output_dir / f"model_lr{lr}_bs{bs}_dr{dropout}_{loss_type}.hdf5"
    callbacks = [
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
            verbose=0
        )
    ]

    # Train
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

    # Get best validation score
    best_val_jacard = max(history.history['val_jacard_coef'])
    best_epoch = history.history['val_jacard_coef'].index(best_val_jacard) + 1

    # Save history
    history_df = pd.DataFrame(history.history)
    history_path = output_dir / f"history_lr{lr}_bs{bs}_dr{dropout}_{loss_type}.csv"
    history_df.to_csv(history_path, index=False)

    print(f"\nBest Val Jaccard: {best_val_jacard:.4f} at epoch {best_epoch}")

    return {
        'learning_rate': lr,
        'batch_size': bs,
        'dropout': dropout,
        'loss_type': loss_type,
        'best_val_jacard': best_val_jacard,
        'best_epoch': best_epoch,
        'total_epochs': len(history.history['loss']),
        'final_val_jacard': history.history['val_jacard_coef'][-1],
        'checkpoint_path': str(checkpoint_path)
    }

def grid_search(search_type='grid', n_random=20):
    """
    Perform hyperparameter search

    Args:
        search_type: 'grid' for full grid search, 'random' for random search
        n_random: number of random combinations if search_type='random'
    """

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"hyperparam_search_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("HYPERPARAMETER SEARCH FOR MICROBEAD SEGMENTATION")
    print("="*80)
    print(f"\nSearch type: {search_type.upper()}")
    print(f"Output directory: {output_dir}")
    print(f"\nSearch space:")
    for param, values in SEARCH_SPACE.items():
        print(f"  {param}: {values}")

    # Load dataset
    X_data, y_data = load_dataset()

    # Stratified split
    print("\nPerforming stratified train/val split by object density...")
    X_train, X_val, y_train, y_val = stratified_split_by_density(
        X_data, y_data,
        test_size=VALIDATION_SPLIT
    )

    print(f"Train set: {len(X_train)} images")
    print(f"Val set: {len(X_val)} images")

    # Generate hyperparameter combinations
    if search_type == 'grid':
        # Full grid search
        import itertools
        param_combinations = list(itertools.product(
            SEARCH_SPACE['learning_rate'],
            SEARCH_SPACE['batch_size'],
            SEARCH_SPACE['dropout'],
            SEARCH_SPACE['loss_type']
        ))
        hyperparams_list = [
            {
                'learning_rate': lr,
                'batch_size': bs,
                'dropout': dr,
                'loss_type': loss
            }
            for lr, bs, dr, loss in param_combinations
        ]
    else:  # random search
        # Random sampling
        np.random.seed(42)
        hyperparams_list = []
        for _ in range(n_random):
            hyperparams_list.append({
                'learning_rate': np.random.choice(SEARCH_SPACE['learning_rate']),
                'batch_size': np.random.choice(SEARCH_SPACE['batch_size']),
                'dropout': np.random.choice(SEARCH_SPACE['dropout']),
                'loss_type': np.random.choice(SEARCH_SPACE['loss_type'])
            })

    total_combinations = len(hyperparams_list)
    print(f"\nTotal combinations to test: {total_combinations}")
    print(f"Image size: 512×512 (4× memory vs 256×256)")
    print(f"Batch sizes reduced: [4, 8, 16] (vs [16, 32, 48] for 256×256)")
    print(f"Estimated time: {total_combinations * 8} - {total_combinations * 15} minutes")
    print()

    # Train each combination
    results = []

    for i, hyperparams in enumerate(hyperparams_list, 1):
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT {i}/{total_combinations}")
        print(f"{'#'*80}")

        try:
            result = train_with_hyperparameters(
                X_train, X_val, y_train, y_val,
                hyperparams,
                output_dir
            )
            results.append(result)

            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('best_val_jacard', ascending=False)
            results_df.to_csv(output_dir / 'search_results.csv', index=False)

        except Exception as e:
            print(f"ERROR in experiment {i}: {e}")
            continue

        # Clear session to free memory
        keras.backend.clear_session()

    # Final results
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_val_jacard', ascending=False)

    print("\nTop 5 configurations:")
    print(results_df.head(5).to_string(index=False))

    # Save final results
    results_df.to_csv(output_dir / 'search_results_final.csv', index=False)

    # Save best hyperparameters as JSON
    best_config = results_df.iloc[0].to_dict()
    with open(output_dir / 'best_hyperparameters.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest configuration:")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  Loss Type: {best_config['loss_type']}")
    print(f"  Best Val Jaccard: {best_config['best_val_jacard']:.4f}")

    print(f"\nAll results saved to: {output_dir}/")

    return results_df, best_config

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter search for microbead segmentation')
    parser.add_argument('--search-type', choices=['grid', 'random'], default='random',
                       help='Type of search: grid (exhaustive) or random (sampling)')
    parser.add_argument('--n-random', type=int, default=20,
                       help='Number of random combinations (only for random search)')

    args = parser.parse_args()

    results_df, best_config = grid_search(
        search_type=args.search_type,
        n_random=args.n_random
    )
