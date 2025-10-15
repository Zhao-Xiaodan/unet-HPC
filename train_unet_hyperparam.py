#!/usr/bin/env python3
"""
Standard UNet Training with Hyperparameter Search
==================================================

Trains standard UNet with hyperparameter tuning.
Saves BOTH best and final models with proper serialization.

Key Features:
1. NO Lambda layers - uses RepeatElements custom layer
2. Saves best model via ModelCheckpoint
3. Saves final model after training completes
4. Hyperparameter search for dropout, filters, batch_norm
5. Proper BinaryFocalLoss serialization
6. Cross-validation friendly

Author: Claude Code
Date: October 16, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Import custom modules
from models_fixed import build_unet, RepeatElements
from loss_functions_fixed import (
    combined_dice_focal_loss,
    jacard_coef,
    dice_coef,
    focal_loss,
    BinaryFocalLoss
)

# For data loading
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    'images_dir': './dataset_shrunk_masks/images/',
    'masks_dir': './dataset_shrunk_masks/masks/',
    'train_val_split': 0.8,  # 80% train, 20% validation

    # Output
    'output_dir': f'./unet_hyperparam_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Model
    'model_name': 'unet',

    # Image settings
    'img_size': 512,
    'img_channels': 3,  # RGB

    # Training hyperparameters to search
    'hyperparam_grid': {
        'n_filters': [16, 32, 64],  # Base number of filters
        'dropout': [0.1, 0.2, 0.3],  # Dropout rate
        'batch_norm': [True],  # Always use batch norm
        'learning_rate': [0.001, 0.003, 0.005],
    },

    # Training settings
    'epochs': 100,
    'batch_size': 4,
    'early_stopping_patience': 20,
    'reduce_lr_patience': 10,

    # Loss function
    'loss': 'binary_focal_loss',  # or 'combined_dice_focal'
    'focal_gamma': 2,
    'focal_alpha': 0.25,
}

# ============================================================================
# SETUP
# ============================================================================

def create_output_dir(config):
    """Create output directory structure."""
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)

    return output_dir

def print_header(config):
    """Print training header."""
    print("="*80)
    print("STANDARD UNET HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Model: Standard UNet")
    print(f"Image size: {config['img_size']}×{config['img_size']}×{config['img_channels']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print()
    print("Hyperparameter Grid:")
    for param, values in config['hyperparam_grid'].items():
        print(f"  {param}: {values}")
    print("="*80)
    print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config):
    """Load training and validation data with train/val split."""
    print("Loading data...")

    image_dataset = []
    mask_dataset = []

    # Load images
    images_dir = config['images_dir']
    masks_dir = config['masks_dir']

    image_files = sorted(os.listdir(images_dir))
    print(f"  Found {len(image_files)} files in image directory")

    for image_name in image_files:
        if image_name.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'tif', 'tiff']:
            # Load image
            image_path = os.path.join(images_dir, image_name)
            image = cv2.imread(image_path, 1)  # Read as RGB
            if image is None:
                print(f"  Warning: Could not read {image_name}")
                continue

            # Resize
            image = Image.fromarray(image)
            image = image.resize((config['img_size'], config['img_size']))
            image_dataset.append(np.array(image))

            # Load corresponding mask
            mask_path = os.path.join(masks_dir, image_name)
            if not os.path.exists(mask_path):
                print(f"  Warning: Mask not found for {image_name}")
                image_dataset.pop()  # Remove the image we just added
                continue

            mask = cv2.imread(mask_path, 0)  # Read as grayscale
            if mask is None:
                print(f"  Warning: Could not read mask for {image_name}")
                image_dataset.pop()
                continue

            mask = Image.fromarray(mask)
            mask = mask.resize((config['img_size'], config['img_size']))
            mask_dataset.append(np.array(mask))

    print(f"  Loaded {len(image_dataset)} image-mask pairs")

    # Convert to numpy arrays
    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)

    # Normalize images
    image_dataset = image_dataset / 255.0

    # Normalize masks (binary: 0 or 1)
    mask_dataset = mask_dataset / 255.0
    mask_dataset = np.expand_dims(mask_dataset, axis=-1)  # Add channel dimension

    # Split into train and validation with fixed random state
    X_train, X_val, y_train, y_val = train_test_split(
        image_dataset,
        mask_dataset,
        test_size=(1 - config['train_val_split']),
        random_state=42
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Image shape: {X_train.shape[1:]}")
    print(f"  Mask shape: {y_train.shape[1:]}")
    print()

    return X_train, X_val, y_train, y_val

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model(n_filters, dropout, batch_norm, config):
    """
    Build standard UNet model.

    Args:
        n_filters: Base number of filters
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization
        config: Global configuration

    Returns:
        Keras model
    """
    input_shape = (config['img_size'], config['img_size'], config['img_channels'])

    model = build_unet(
        input_shape=input_shape,
        n_filters=n_filters,
        dropout=dropout,
        batch_norm=batch_norm
    )

    return model

def compile_model(model, learning_rate, config):
    """Compile model with optimizer, loss, and metrics."""

    # Create loss function
    if config['loss'] == 'binary_focal_loss':
        loss_fn = BinaryFocalLoss(
            gamma=config['focal_gamma'],
            alpha=config['focal_alpha']
        )
    elif config['loss'] == 'combined_dice_focal':
        loss_fn = combined_dice_focal_loss
    else:
        raise ValueError(f"Unknown loss: {config['loss']}")

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[jacard_coef, dice_coef]
    )

    return model

# ============================================================================
# TRAINING
# ============================================================================

def create_callbacks(output_dir, experiment_name, config):
    """Create training callbacks."""
    checkpoint_dir = output_dir / 'checkpoints' / experiment_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=str(checkpoint_dir / 'best_model.keras'),
            monitor='val_jacard_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        ),

        # Early stopping
        EarlyStopping(
            monitor='val_jacard_coef',
            mode='max',
            patience=config['early_stopping_patience'],
            verbose=1,
            restore_best_weights=True
        ),

        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_jacard_coef',
            mode='max',
            factor=0.5,
            patience=config['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        ),

        # CSV logger
        CSVLogger(
            str(output_dir / 'logs' / f'{experiment_name}_history.csv'),
            append=False
        )
    ]

    return callbacks

def train_model(model, X_train, y_train, X_val, y_val, callbacks, config):
    """Train model and return history."""
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    return history

# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================

def generate_hyperparameter_combinations(hyperparam_grid):
    """Generate all combinations of hyperparameters."""
    import itertools

    keys = list(hyperparam_grid.keys())
    values = [hyperparam_grid[k] for k in keys]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations

def run_hyperparam_search(X_train, y_train, X_val, y_val, output_dir, config):
    """
    Run hyperparameter search for standard UNet.

    Returns:
        DataFrame with results for all hyperparameter combinations
    """
    print("="*80)
    print(f"HYPERPARAMETER SEARCH: STANDARD UNET")
    print("="*80)
    print()

    # Generate all hyperparameter combinations
    hyperparam_combos = generate_hyperparameter_combinations(config['hyperparam_grid'])

    print(f"Total combinations to try: {len(hyperparam_combos)}")
    print()

    results = []

    for idx, hyperparams in enumerate(hyperparam_combos):
        print(f"\n{'='*80}")
        print(f"Combination {idx+1}/{len(hyperparam_combos)}")
        print(f"{'='*80}")
        print("Hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
        print()

        # Create experiment name
        experiment_name = f"unet_" + "_".join([f"{k}{v}" for k, v in hyperparams.items()])
        experiment_name = experiment_name.replace('.', 'p')  # Replace dots in filenames

        try:
            # Build model
            print("Building model...")
            model = build_model(
                hyperparams['n_filters'],
                hyperparams['dropout'],
                hyperparams['batch_norm'],
                config
            )

            # Compile model
            model = compile_model(model, hyperparams['learning_rate'], config)

            print(f"Model parameters: {model.count_params():,}")
            print()

            # Create callbacks
            callbacks = create_callbacks(output_dir, experiment_name, config)

            # Train
            print("Starting training...")
            history = train_model(model, X_train, y_train, X_val, y_val, callbacks, config)

            # Get best metrics
            best_epoch = np.argmax(history.history['val_jacard_coef'])
            best_val_iou = history.history['val_jacard_coef'][best_epoch]
            best_val_dice = history.history['val_dice_coef'][best_epoch]
            final_val_iou = history.history['val_jacard_coef'][-1]
            final_val_dice = history.history['val_dice_coef'][-1]

            # Save final model (in addition to best)
            final_model_path = output_dir / 'models' / f'{experiment_name}_final.keras'
            model.save(final_model_path)
            print(f"\n✓ Saved final model: {final_model_path}")

            # Record results
            result = {
                'model': 'unet',
                'experiment_name': experiment_name,
                'best_epoch': best_epoch + 1,
                'best_val_iou': best_val_iou,
                'best_val_dice': best_val_dice,
                'final_val_iou': final_val_iou,
                'final_val_dice': final_val_dice,
                **hyperparams
            }
            results.append(result)

            print(f"\n{'='*80}")
            print("Results:")
            print(f"  Best epoch: {best_epoch + 1}")
            print(f"  Best Val IoU: {best_val_iou:.4f}")
            print(f"  Best Val Dice: {best_val_dice:.4f}")
            print(f"  Final Val IoU: {final_val_iou:.4f}")
            print(f"  Final Val Dice: {final_val_dice:.4f}")
            print(f"{'='*80}\n")

            # Clear session to free memory
            keras.backend.clear_session()

        except Exception as e:
            print(f"\n❌ Error training with hyperparameters {hyperparams}:")
            print(f"   {str(e)}")
            print()

            result = {
                'model': 'unet',
                'experiment_name': experiment_name,
                'error': str(e),
                **hyperparams
            }
            results.append(result)

            keras.backend.clear_session()
            continue

    return pd.DataFrame(results)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""
    # Print header
    print_header(CONFIG)

    # Create output directory
    output_dir = create_output_dir(CONFIG)

    # Save configuration
    config_path = output_dir / 'CONFIG.json'
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)
    print(f"✓ Saved configuration: {config_path}\n")

    # Load data
    X_train, X_val, y_train, y_val = load_data(CONFIG)

    # Run hyperparameter search
    results_df = run_hyperparam_search(X_train, y_train, X_val, y_val, output_dir, CONFIG)

    # Save results
    results_path = output_dir / 'unet_results.csv'
    results_df.to_csv(results_path, index=False)

    # Print summary
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total experiments: {len(results_df)}")
    print()

    # Print best results
    if 'best_val_iou' in results_df.columns:
        results_clean = results_df.dropna(subset=['best_val_iou'])
        if len(results_clean) > 0:
            best_idx = results_clean['best_val_iou'].idxmax()
            best_result = results_clean.loc[best_idx]

            print(f"STANDARD UNET - Best Configuration:")
            print(f"  Experiment: {best_result['experiment_name']}")
            print(f"  Best Val IoU: {best_result['best_val_iou']:.4f}")
            print(f"  Hyperparameters:")
            for param in CONFIG['hyperparam_grid'].keys():
                print(f"    {param}: {best_result[param]}")
            print()

    print(f"✓ Results saved: {results_path}")
    print("="*80)

if __name__ == '__main__':
    main()
