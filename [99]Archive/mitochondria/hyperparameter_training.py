#!/usr/bin/env python3
"""
Hyperparameter optimization training script for U-Net architectures.
Supports systematic grid search with configurable parameters.
"""

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from datetime import datetime
import cv2
from PIL import Image
from keras import backend as K
import argparse
import json

def setup_gpu():
    """Configure GPU memory growth to prevent allocation issues."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

def load_dataset():
    """Load and preprocess the mitochondria segmentation dataset."""
    print("Loading dataset...")

    # Dataset paths - Use full stack dataset (1,980 patches)
    image_directory = 'dataset_full_stack/images/'
    mask_directory = 'dataset_full_stack/masks/'

    if not os.path.exists(image_directory) or not os.path.exists(mask_directory):
        print(f"ERROR: Full stack dataset not found!")
        print(f"Required directories:")
        print(f"  {image_directory}")
        print(f"  {mask_directory}")
        print(f"")
        print(f"Please run: python3 create_full_dataset.py")
        print(f"Or copy dataset_full_stack/ from local machine")
        raise FileNotFoundError("Full stack dataset required for meaningful hyperparameter optimization")

    SIZE = 256
    image_dataset = []
    mask_dataset = []

    # Load images
    images = os.listdir(image_directory)
    for i, image_name in enumerate(images):
        if image_name.split('.')[-1] == 'tif':
            image = cv2.imread(image_directory + image_name, 1)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            image_dataset.append(np.array(image))

    # Load masks
    masks = os.listdir(mask_directory)
    for i, image_name in enumerate(masks):
        if image_name.split('.')[-1] == 'tif':
            image = cv2.imread(mask_directory + image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            mask_dataset.append(np.array(image))

    # Normalize
    image_dataset = np.array(image_dataset) / 255.0
    mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.0

    print(f"Loaded {len(image_dataset)} images and {len(mask_dataset)} masks")
    return image_dataset, mask_dataset

def create_data_splits(image_dataset, mask_dataset, random_state=42):
    """Create train/validation splits."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        image_dataset, mask_dataset,
        test_size=0.10,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def train_model(architecture, learning_rate, batch_size, epochs, X_train, X_test, y_train, y_test, output_dir):
    """Train a specific model configuration."""

    print(f"\n{'='*60}")
    print(f"TRAINING: {architecture}")
    print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {epochs}")
    print(f"{'='*60}")

    # Import models and metrics
    from models import UNet, Attention_UNet, Attention_ResUNet, jacard_coef

    # Install focal loss if needed
    try:
        from focal_loss import BinaryFocalLoss
        print("✓ focal_loss imported successfully")
    except ImportError:
        print("Installing focal_loss...")
        os.system("pip install focal-loss --user")
        from focal_loss import BinaryFocalLoss

    # Model configuration
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # Create model based on architecture
    if architecture == 'UNet':
        model = UNet(input_shape)
    elif architecture == 'Attention_UNet':
        model = Attention_UNet(input_shape)
    elif architecture == 'Attention_ResUNet':
        model = Attention_ResUNet(input_shape)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Compile model with hyperparameters
    model.compile(
        optimizer=Adam(lr=learning_rate, clipnorm=1.0),  # Added gradient clipping
        loss=BinaryFocalLoss(gamma=2),
        metrics=['accuracy', jacard_coef]
    )

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_jacard_coef',
            patience=10,  # Increased patience for stability
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_jacard_coef',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='max'
        )
    ]

    # Training
    start_time = datetime.now()

    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        training_successful = True

    except Exception as e:
        print(f"Training failed: {e}")
        training_successful = False
        history = None

    end_time = datetime.now()
    training_time = end_time - start_time

    # Save results
    results = {
        'architecture': architecture,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs_requested': epochs,
        'epochs_completed': len(history.history['loss']) if history else 0,
        'training_time_seconds': training_time.total_seconds(),
        'training_successful': training_successful
    }

    if training_successful and history:
        # Extract best metrics
        best_val_jaccard = max(history.history['val_jacard_coef'])
        best_epoch = history.history['val_jacard_coef'].index(best_val_jaccard) + 1
        final_val_loss = history.history['val_loss'][-1]
        final_train_loss = history.history['loss'][-1]

        # Calculate stability metrics
        last_10_epochs = min(10, len(history.history['val_loss']))
        val_loss_stability = np.std(history.history['val_loss'][-last_10_epochs:])

        results.update({
            'best_val_jaccard': best_val_jaccard,
            'best_epoch': best_epoch,
            'final_val_loss': final_val_loss,
            'final_train_loss': final_train_loss,
            'val_loss_stability': val_loss_stability,
            'overfitting_gap': final_val_loss - final_train_loss
        })

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_file = os.path.join(output_dir, f'{architecture}_lr{learning_rate}_bs{batch_size}_history.csv')
        history_df.to_csv(history_file)

        # Save model
        model_file = os.path.join(output_dir, f'{architecture}_lr{learning_rate}_bs{batch_size}_model.hdf5')
        model.save(model_file)

        print(f"\n✓ Training completed successfully!")
        print(f"  Best Val Jaccard: {best_val_jaccard:.4f} (epoch {best_epoch})")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Training Time: {training_time}")
        print(f"  Stability (std): {val_loss_stability:.4f}")

    else:
        print(f"\n✗ Training failed or incomplete")

    return results

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization training')
    parser.add_argument('--architecture', required=True, choices=['UNet', 'Attention_UNet', 'Attention_ResUNet'])
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--output_dir', required=True)

    args = parser.parse_args()

    # Setup
    setup_gpu()

    # Load data
    image_dataset, mask_dataset = load_dataset()
    X_train, X_test, y_train, y_test = create_data_splits(image_dataset, mask_dataset)

    print(f"Training set: {X_train.shape}, Validation set: {X_test.shape}")

    # Train model
    results = train_model(
        args.architecture, args.learning_rate, args.batch_size, args.epochs,
        X_train, X_test, y_train, y_test, args.output_dir
    )

    # Save individual results
    results_file = os.path.join(args.output_dir, f'{args.architecture}_lr{args.learning_rate}_bs{args.batch_size}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
