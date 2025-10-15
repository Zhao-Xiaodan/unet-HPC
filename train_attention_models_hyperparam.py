#!/usr/bin/env python3
"""
Attention Models Training with Hyperparameter Search
====================================================

Trains Attention UNet and Attention ResUNet with hyperparameter tuning.
Saves BOTH best and final models with proper serialization (no Lambda layers).

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
from models_fixed import build_attention_unet, build_attention_resunet, RepeatElements
from loss_functions_fixed import (
    combined_dice_focal_loss,
    jacard_coef,
    dice_coef,
    focal_loss,
    BinaryFocalLoss  # Make sure this exists in loss_functions_fixed.py
)
from data_generator import DataGenerator

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data paths
    'train_images': './dataset_new_shrunk/train/images/',
    'train_masks': './dataset_new_shrunk/train/masks/',
    'val_images': './dataset_new_shrunk/val/images/',
    'val_masks': './dataset_new_shrunk/val/masks/',

    # Output
    'output_dir': f'./attention_hyperparam_{datetime.now().strftime("%Y%m%d_%H%M%S")}',

    # Model selection
    'architectures': ['attention_unet', 'attention_resunet'],

    # Image settings
    'img_size': 512,
    'img_channels': 3,  # RGB

    # Training hyperparameters to search
    'hyperparam_grid': {
        'n_filters': [16, 32],  # Base number of filters
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
    print("ATTENTION MODELS HYPERPARAMETER SEARCH")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {config['output_dir']}")
    print(f"Architectures: {', '.join(config['architectures'])}")
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
    """Load training and validation data."""
    print("Loading data...")

    train_gen = DataGenerator(
        config['train_images'],
        config['train_masks'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        augment=True
    )

    val_gen = DataGenerator(
        config['val_images'],
        config['val_masks'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        augment=False
    )

    print(f"  Training samples: {len(train_gen) * config['batch_size']}")
    print(f"  Validation samples: {len(val_gen) * config['batch_size']}")
    print()

    return train_gen, val_gen

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model(architecture, n_filters, dropout, batch_norm, config):
    """
    Build model with specified architecture and hyperparameters.

    Args:
        architecture: 'attention_unet' or 'attention_resunet'
        n_filters: Base number of filters
        dropout: Dropout rate
        batch_norm: Whether to use batch normalization
        config: Global configuration

    Returns:
        Keras model
    """
    input_shape = (config['img_size'], config['img_size'], config['img_channels'])

    if architecture == 'attention_unet':
        model = build_attention_unet(
            input_shape=input_shape,
            n_filters=n_filters,
            dropout=dropout,
            batch_norm=batch_norm
        )
    elif architecture == 'attention_resunet':
        model = build_attention_resunet(
            input_shape=input_shape,
            n_filters=n_filters,
            dropout=dropout,
            batch_norm=batch_norm
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

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

def train_model(model, train_gen, val_gen, callbacks, config):
    """Train model and return history."""
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['epochs'],
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

def run_hyperparam_search(architecture, train_gen, val_gen, output_dir, config):
    """
    Run hyperparameter search for given architecture.

    Returns:
        DataFrame with results for all hyperparameter combinations
    """
    print("="*80)
    print(f"HYPERPARAMETER SEARCH: {architecture.upper()}")
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
        experiment_name = f"{architecture}_" + "_".join([f"{k}{v}" for k, v in hyperparams.items()])
        experiment_name = experiment_name.replace('.', 'p')  # Replace dots in filenames

        try:
            # Build model
            print("Building model...")
            model = build_model(
                architecture,
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
            history = train_model(model, train_gen, val_gen, callbacks, config)

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
                'architecture': architecture,
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
                'architecture': architecture,
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
    train_gen, val_gen = load_data(CONFIG)

    # Run hyperparameter search for each architecture
    all_results = []

    for architecture in CONFIG['architectures']:
        results_df = run_hyperparam_search(
            architecture,
            train_gen,
            val_gen,
            output_dir,
            CONFIG
        )
        all_results.append(results_df)

        # Save intermediate results
        results_path = output_dir / f'{architecture}_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Saved {architecture} results: {results_path}\n")

    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)
    final_results_path = output_dir / 'all_results.csv'
    final_results.to_csv(final_results_path, index=False)

    # Print summary
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total experiments: {len(final_results)}")
    print()

    # Print best results for each architecture
    for architecture in CONFIG['architectures']:
        arch_results = final_results[final_results['architecture'] == architecture]
        if 'best_val_iou' in arch_results.columns:
            arch_results = arch_results.dropna(subset=['best_val_iou'])
            if len(arch_results) > 0:
                best_idx = arch_results['best_val_iou'].idxmax()
                best_result = arch_results.loc[best_idx]

                print(f"{architecture.upper()} - Best Configuration:")
                print(f"  Experiment: {best_result['experiment_name']}")
                print(f"  Best Val IoU: {best_result['best_val_iou']:.4f}")
                print(f"  Hyperparameters:")
                for param in CONFIG['hyperparam_grid'].keys():
                    print(f"    {param}: {best_result[param]}")
                print()

    print(f"✓ All results saved: {final_results_path}")
    print("="*80)

if __name__ == '__main__':
    main()
