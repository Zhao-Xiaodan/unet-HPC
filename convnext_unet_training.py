#!/usr/bin/env python3
"""
ConvNeXt-UNet Dedicated Training Script

This script focuses specifically on training ConvNeXt-UNet with enhanced
cache clearing and dataset management to resolve persistent dataset conflicts.

Based on successful Swin-UNet training, optimized for ConvNeXt architecture.
"""

import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from datetime import datetime
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import json
import time
import gc
import shutil
import tempfile
import uuid

# =============================================================================
# Environment Setup with Enhanced Dataset Management
# =============================================================================

def setup_gpu():
    """Configure GPU memory growth to prevent allocation issues."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPUs")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPUs found. Running on CPU.")
        return False

def aggressive_cache_clearing():
    """Aggressive TensorFlow cache clearing specifically for ConvNeXt-UNet dataset issues"""
    print("🧹 Performing aggressive cache clearing for ConvNeXt-UNet...")

    try:
        # Clear TensorFlow session completely
        tf.keras.backend.clear_session()

        # Reset default graph
        tf.compat.v1.reset_default_graph() if hasattr(tf.compat.v1, 'reset_default_graph') else None

        # Clear all TensorFlow cache directories aggressively
        cache_patterns = [
            os.path.expanduser('~/.tensorflow*'),
            '/tmp/tf*',
            '/tmp/tensorflow*',
            '/tmp/*tf*',
            tempfile.gettempdir() + '/tf*',
            tempfile.gettempdir() + '/*tensorflow*',
            '/var/tmp/tf*',
            '/dev/shm/tf*'
        ]

        import glob
        cleared_count = 0
        for pattern in cache_patterns:
            for path in glob.glob(pattern):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                        cleared_count += 1
                    elif os.path.isfile(path):
                        os.remove(path)
                        cleared_count += 1
                except Exception as e:
                    pass  # Ignore individual failures

        # Clear Python cache
        gc.collect()

        # Reset TensorFlow completely
        try:
            tf.compat.v1.disable_eager_execution()
            tf.compat.v1.enable_eager_execution()
        except:
            pass

        # Generate unique session ID for this run
        session_id = str(uuid.uuid4())[:8]
        os.environ['TF_SESSION_ID'] = session_id

        print(f"✓ Aggressive cache clearing completed: {cleared_count} items cleared")
        print(f"✓ Unique session ID: {session_id}")

    except Exception as e:
        print(f"Warning: Aggressive cache clearing failed: {e}")

def create_unique_dataset(X, y, session_id):
    """Create dataset with unique naming to prevent conflicts"""
    try:
        # Use tf.data.Dataset.from_tensor_slices with unique naming
        dataset_name = f"convnext_dataset_{session_id}_{int(time.time())}"

        # Convert to tensors explicitly
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

        # Create dataset without caching
        dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))

        return dataset, dataset_name

    except Exception as e:
        print(f"Warning: Unique dataset creation failed: {e}")
        return None, None

# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(verbose=True):
    """Load and preprocess the mitochondria segmentation dataset."""
    if verbose:
        print("=== LOADING DATASET FOR CONVNEXT-UNET ===")

    # Dataset paths - prioritize full stack dataset
    dataset_dirs = [
        ('dataset_full_stack/images/', 'dataset_full_stack/masks/'),
        ('dataset/images/', 'dataset/masks/'),
        ('data/images/', 'data/masks/')
    ]

    image_directory = None
    mask_directory = None

    for img_dir, mask_dir in dataset_dirs:
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            image_directory = img_dir
            mask_directory = mask_dir
            break

    if not image_directory:
        raise FileNotFoundError("Dataset not found!")

    if verbose:
        print(f"Using dataset: {image_directory} and {mask_directory}")

    SIZE = 256
    image_dataset = []
    mask_dataset = []

    # Load images
    images = os.listdir(image_directory)
    for i, image_name in enumerate(images):
        if image_name.split('.')[-1] in ['tif', 'tiff', 'png', 'jpg']:
            image = cv2.imread(image_directory + image_name, 1)
            if image is not None:
                image = Image.fromarray(image)
                image = image.resize((SIZE, SIZE))
                image_dataset.append(np.array(image))

    # Load masks
    masks = os.listdir(mask_directory)
    for i, image_name in enumerate(masks):
        if image_name.split('.')[-1] in ['tif', 'tiff', 'png', 'jpg']:
            image = cv2.imread(mask_directory + image_name, 0)
            if image is not None:
                image = Image.fromarray(image)
                image = image.resize((SIZE, SIZE))
                mask_dataset.append(np.array(image))

    if len(image_dataset) == 0 or len(mask_dataset) == 0:
        raise ValueError("No valid images found in dataset directories!")

    # Normalize
    image_dataset = np.array(image_dataset) / 255.0
    mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.0

    if verbose:
        print(f"✓ Loaded {len(image_dataset)} images and {len(mask_dataset)} masks")

    return image_dataset, mask_dataset

def create_data_splits(image_dataset, mask_dataset, test_size=0.10, random_state=42):
    """Create train/validation splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        image_dataset, mask_dataset,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test

# =============================================================================
# ConvNeXt-UNet Training Function
# =============================================================================

def train_convnext_unet(X_train, X_test, y_train, y_test,
                       learning_rate=1e-4, batch_size=4, epochs=100, output_dir="./"):
    """
    Train ConvNeXt-UNet with enhanced dataset management.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: ConvNeXt-UNet (Dedicated)")
    print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Max Epochs: {epochs}")
    print(f"{'='*60}")

    # Aggressive cache clearing before starting
    aggressive_cache_clearing()

    # Import models and metrics
    from modern_unet_models import create_modern_unet

    # Import existing metrics
    try:
        from models import jacard_coef
    except ImportError:
        def jacard_coef(y_true, y_pred):
            import tensorflow as tf
            from tensorflow.keras import backend as K

            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            y_pred_binary = K.cast(K.greater(y_pred_f, 0.5), K.floatx())
            intersection = K.sum(y_true_f * y_pred_binary)
            union = K.sum(y_true_f) + K.sum(y_pred_binary) - intersection
            return (intersection + 1e-7) / (union + 1e-7)

    # Handle focal loss
    try:
        from focal_loss import BinaryFocalLoss
    except ImportError:
        class BinaryFocalLoss:
            def __init__(self, gamma=2.0, alpha=0.25):
                self.gamma = gamma
                self.alpha = alpha

            def __call__(self, y_true, y_pred):
                from tensorflow.keras import backend as K
                y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
                pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
                alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
                focal_loss = -alpha_t * K.pow(1 - pt, self.gamma) * K.log(pt)
                return K.mean(focal_loss)

    # Model configuration
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = X_train.shape[1], X_train.shape[2], X_train.shape[3]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    try:
        # Create ConvNeXt-UNet model
        print(f"Creating ConvNeXt-UNet model...")
        model = create_modern_unet('ConvNeXt_UNet', input_shape, num_classes=1)

        # Build model explicitly
        model.build(input_shape=(None,) + input_shape)
        print(f"Model parameters: {model.count_params():,}")

        # Use Adam optimizer for ConvNeXt
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=BinaryFocalLoss(gamma=2),
            metrics=['accuracy', jacard_coef]
        )

        # Callbacks
        model_filename = f"ConvNeXt_UNet_lr{learning_rate}_bs{batch_size}_model.hdf5"
        model_path = os.path.join(output_dir, model_filename)

        callbacks = [
            EarlyStopping(
                monitor='val_jacard_coef',
                patience=15,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_jacard_coef',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_jacard_coef',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                mode='max',
                verbose=1
            )
        ]

        print("Starting ConvNeXt-UNet training...")
        start_time = time.time()

        # ConvNeXt-specific optimizations
        print("Applying ConvNeXt-specific optimizations...")
        tf.config.experimental.enable_tensor_float_32_execution(False)

        # Train model with standard numpy arrays (avoid tf.data.Dataset for ConvNeXt)
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
            use_multiprocessing=False,  # Disable multiprocessing to avoid dataset conflicts
            workers=1
        )

        training_time = time.time() - start_time

        # Save training history
        history_df = pd.DataFrame(history.history)
        history_filename = f"ConvNeXt_UNet_lr{learning_rate}_bs{batch_size}_history.csv"
        history_path = os.path.join(output_dir, history_filename)
        history_df.to_csv(history_path)

        # Calculate metrics
        best_val_jaccard = float(max(history.history['val_jacard_coef']))
        best_epoch = int(np.argmax(history.history['val_jacard_coef']) + 1)
        final_val_loss = float(history.history['val_loss'][-1])
        final_train_loss = float(history.history['loss'][-1])

        # Calculate stability
        last_10_epochs = min(10, len(history.history['val_loss']))
        val_loss_stability = float(np.std(history.history['val_loss'][-last_10_epochs:]))

        # Training results
        results = {
            'model_name': 'ConvNeXt_UNet',
            'learning_rate': float(learning_rate),
            'batch_size': int(batch_size),
            'epochs_completed': int(len(history.history['loss'])),
            'training_time_seconds': float(training_time),
            'best_val_jaccard': best_val_jaccard,
            'best_epoch': best_epoch,
            'final_val_loss': final_val_loss,
            'final_train_loss': final_train_loss,
            'val_loss_stability': val_loss_stability,
            'overfitting_gap': float(final_val_loss - final_train_loss),
            'model_parameters': int(model.count_params())
        }

        # Save results
        results_filename = f"ConvNeXt_UNet_lr{learning_rate}_bs{batch_size}_results.json"
        results_path = os.path.join(output_dir, results_filename)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ ConvNeXt-UNet training completed successfully!")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Best Val Jaccard: {best_val_jaccard:.4f} (epoch {best_epoch})")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Stability (std): {val_loss_stability:.4f}")

        return model, history, results

    except Exception as e:
        print(f"\n✗ ConvNeXt-UNet training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

    finally:
        # Final cleanup
        aggressive_cache_clearing()

# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main training function for ConvNeXt-UNet."""
    print("=" * 70)
    print("CONVNEXT-UNET DEDICATED TRAINING FOR MITOCHONDRIA SEGMENTATION")
    print("=" * 70)
    print("Model: ConvNeXt-UNet (Modern CNN with improved efficiency)")
    print("Task: Mitochondria semantic segmentation")
    print("Framework: TensorFlow/Keras with enhanced dataset management")
    print()

    # Setup
    setup_gpu()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"convnext_unet_training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    try:
        # Load dataset
        image_dataset, mask_dataset = load_dataset()
        X_train, X_test, y_train, y_test = create_data_splits(image_dataset, mask_dataset)

        # Training configuration
        learning_rate = 1e-4
        batch_size = 4
        epochs = 100

        print(f"\nTraining Configuration:")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Max Epochs: {epochs}")
        print(f"  Input Shape: {X_train.shape[1:]}")

        # Train ConvNeXt-UNet
        model, history, results = train_convnext_unet(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            output_dir=output_dir
        )

        if model is not None and results is not None:
            print(f"\n{'='*70}")
            print("CONVNEXT-UNET TRAINING COMPLETED SUCCESSFULLY")
            print(f"{'='*70}")
            print(f"✓ Best Jaccard: {results['best_val_jaccard']:.4f} (epoch {results['best_epoch']})")
            print(f"✓ Training Time: {results['training_time_seconds']:.1f}s")
            print(f"✓ Parameters: {results['model_parameters']:,}")
            print(f"✓ Stability: {results['val_loss_stability']:.4f}")
            print(f"📁 All outputs saved in: {output_dir}")
        else:
            print(f"\n✗ ConvNeXt-UNet training failed!")

    except Exception as e:
        print(f"\n✗ Training script failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("CONVNEXT-UNET DEDICATED TRAINING COMPLETED")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()