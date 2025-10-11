#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Search for Microbead Segmentation
==============================================================
Tests combinations of:
- Architectures: U-Net, ResU-Net, Attention ResU-Net
- Batch Sizes: 8, 16, 32
- Loss Functions: Focal, Combined (Dice+Focal), Focal Tversky, Combined Tversky
- Dataset: dataset_shrunk_masks (512×512)

Based on insights from previous hyperparameter search analysis.
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import custom modules
from model_architectures import get_model
from loss_functions import get_loss_function, dice_coef, jacard_coef

# Fixed parameters (based on previous analysis)
SIZE = 512  # Use original dataset resolution
IMG_CHANNELS = 1
EPOCHS = 100
EARLY_STOP_PATIENCE = 30  # Increased from 20 based on analysis
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 5e-5  # Lower LR for stability with small batches (from analysis)

# Hyperparameter search space
SEARCH_SPACE = {
    'architecture': ['unet', 'resunet', 'attention_resunet'],
    'batch_size': [8, 16, 32],  # Larger batch sizes for gradient stability
    'loss_function': ['focal', 'combined', 'focal_tversky', 'combined_tversky'],
    'dropout': [0.3],  # Fixed at 0.3 (worked well in previous search)
}

# Dataset paths
TRAIN_PATH_IMAGES = './dataset_shrunk_masks/images/'
TRAIN_PATH_MASKS = './dataset_shrunk_masks/masks/'


def load_dataset():
    """
    Load and preprocess dataset from dataset_shrunk_masks

    Returns:
        X_train: Image array (N, 512, 512, 1)
        y_train: Mask array (N, 512, 512, 1)
    """
    print("Loading dataset from dataset_shrunk_masks...")

    # Check if directories exist
    if not os.path.exists(TRAIN_PATH_IMAGES):
        raise FileNotFoundError(f"Image directory not found: {TRAIN_PATH_IMAGES}")
    if not os.path.exists(TRAIN_PATH_MASKS):
        raise FileNotFoundError(f"Mask directory not found: {TRAIN_PATH_MASKS}")

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
            else:
                print(f"Warning: Mask not found for {id_}")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Normalize
    X_train = X_train / 255.0
    y_train = y_train / 255.0

    # Expand dimensions
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)

    print(f"Dataset loaded: {len(X_train)} images")
    print(f"Image size: {SIZE}×{SIZE} (original resolution)")
    print(f"Shape: X_train={X_train.shape}, y_train={y_train.shape}")

    return X_train, y_train


def stratified_split_by_density(X, y, test_size=0.15, random_state=42):
    """
    Split dataset stratified by object density

    Args:
        X: Image array
        y: Mask array
        test_size: Fraction for validation (default: 0.15)
        random_state: Random seed (default: 42)

    Returns:
        X_train, X_val, y_train, y_val: Split datasets
    """
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


def train_with_hyperparameters(X_train, X_val, y_train, y_val, hyperparams, output_dir):
    """
    Train model with specific hyperparameters

    Args:
        X_train: Training images
        X_val: Validation images
        y_train: Training masks
        y_val: Validation masks
        hyperparams: Dictionary of hyperparameters
        output_dir: Output directory path

    Returns:
        Dictionary with training results
    """
    arch = hyperparams['architecture']
    bs = hyperparams['batch_size']
    dropout = hyperparams['dropout']
    loss_name = hyperparams['loss_function']

    print(f"\n{'='*80}")
    print(f"Training: Arch={arch}, BS={bs}, Dropout={dropout}, Loss={loss_name}")
    print(f"{'='*80}\n")

    # Build model
    input_shape = (SIZE, SIZE, IMG_CHANNELS)
    model = get_model(arch, input_shape, NUM_CLASSES=1, dropout_rate=dropout, batch_norm=True)

    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_params():,}")

    # Get loss function
    loss_fn = get_loss_function(loss_name)

    # Compile model with fixed learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=['accuracy', jacard_coef, dice_coef]
    )

    # Data augmentation (strong augmentation based on analysis recommendations)
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=15,
        fill_mode='reflect',
        zoom_range=0.1,  # Additional augmentation
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    val_datagen = ImageDataGenerator()

    # Callbacks
    checkpoint_path = output_dir / f"model_{arch}_bs{bs}_dr{dropout}_{loss_name}.hdf5"
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
            patience=10,  # More patience for LR reduction
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
    history_path = output_dir / f"history_{arch}_bs{bs}_dr{dropout}_{loss_name}.csv"
    history_df.to_csv(history_path, index=False)

    print(f"\nBest Val Jaccard: {best_val_jacard:.4f} at epoch {best_epoch}")

    # Clear session to free memory
    keras.backend.clear_session()
    del model

    return {
        'architecture': arch,
        'batch_size': bs,
        'dropout': dropout,
        'loss_function': loss_name,
        'learning_rate': LEARNING_RATE,
        'best_val_jacard': best_val_jacard,
        'best_epoch': best_epoch,
        'total_epochs': len(history.history['loss']),
        'final_val_jacard': history.history['val_jacard_coef'][-1],
        'checkpoint_path': str(checkpoint_path)
    }


def grid_search(search_type='grid', n_random=30):
    """
    Perform comprehensive hyperparameter search

    Args:
        search_type: 'grid' for full grid search, 'random' for random sampling
        n_random: Number of random combinations if search_type='random'

    Returns:
        results_df: DataFrame with all results
        best_config: Dictionary with best configuration
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"hyperparam_comprehensive_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("COMPREHENSIVE HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"\nSearch type: {search_type.upper()}")
    print(f"Output directory: {output_dir}")
    print(f"Fixed learning rate: {LEARNING_RATE} (from analysis)")
    print(f"Early stopping patience: {EARLY_STOP_PATIENCE} epochs (increased)")
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
            SEARCH_SPACE['architecture'],
            SEARCH_SPACE['batch_size'],
            SEARCH_SPACE['dropout'],
            SEARCH_SPACE['loss_function']
        ))
        hyperparams_list = [
            {
                'architecture': arch,
                'batch_size': bs,
                'dropout': dr,
                'loss_function': loss
            }
            for arch, bs, dr, loss in param_combinations
        ]
    else:  # random search
        # Random sampling
        np.random.seed(42)
        hyperparams_list = []
        for _ in range(n_random):
            hyperparams_list.append({
                'architecture': np.random.choice(SEARCH_SPACE['architecture']),
                'batch_size': np.random.choice(SEARCH_SPACE['batch_size']),
                'dropout': np.random.choice(SEARCH_SPACE['dropout']),
                'loss_function': np.random.choice(SEARCH_SPACE['loss_function'])
            })

    total_combinations = len(hyperparams_list)
    print(f"\nTotal combinations to test: {total_combinations}")
    print(f"Estimated time: {total_combinations * 15} - {total_combinations * 30} minutes")
    print(f"(15-30 min per configuration at 512×512)")
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
            import traceback
            traceback.print_exc()
            continue

    # Final results
    print("\n" + "="*80)
    print("COMPREHENSIVE HYPERPARAMETER SEARCH COMPLETE")
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
    print(f"  Architecture: {best_config['architecture']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  Loss Function: {best_config['loss_function']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Best Val Jaccard: {best_config['best_val_jacard']:.4f}")

    print(f"\nAll results saved to: {output_dir}/")

    return results_df, best_config


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive hyperparameter search')
    parser.add_argument('--search-type', choices=['grid', 'random'], default='random',
                       help='Type of search: grid (exhaustive) or random (sampling)')
    parser.add_argument('--n-random', type=int, default=30,
                       help='Number of random combinations (only for random search)')

    args = parser.parse_args()

    results_df, best_config = grid_search(
        search_type=args.search_type,
        n_random=args.n_random
    )
